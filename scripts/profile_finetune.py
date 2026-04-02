#!/usr/bin/env python3
"""Profile a realistic QLoRA finetune on Dolly-15K.

Usage:
    uv run python scripts/profile_finetune.py [--steps N] [--model MODEL] [--metal-trace]

Profiling outputs:
    profiles/cprofile_stats.prof     — cProfile dump (view with snakeviz)
    profiles/timeline.jsonl          — per-step timing + memory trace
    profiles/summary.txt             — human-readable summary
    profiles/mlx_trace.gputrace/     — Metal GPU trace (if --metal-trace)
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import logging
import os
import pstats
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing / memory context manager
# ---------------------------------------------------------------------------

@dataclass
class StepProfile:
    step: int = 0
    phase: str = ""
    wall_time_s: float = 0.0
    active_memory_gb: float = 0.0
    peak_memory_gb: float = 0.0
    tokens_processed: int = 0
    loss: float = 0.0
    extra: dict = field(default_factory=dict)


class Profiler:
    """Collects per-step timing and memory data."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps: list[StepProfile] = []
        self._phase_start: float = 0.0

    def begin(self, step: int, phase: str) -> StepProfile:
        sp = StepProfile(step=step, phase=phase)
        self._phase_start = time.perf_counter()
        return sp

    def end(self, sp: StepProfile, **extra) -> StepProfile:
        sp.wall_time_s = time.perf_counter() - self._phase_start
        sp.active_memory_gb = mx.get_active_memory() / 1e9
        sp.peak_memory_gb = mx.get_peak_memory() / 1e9
        sp.extra = extra
        self.steps.append(sp)
        return sp

    def save(self):
        # Timeline JSONL
        with open(self.output_dir / "timeline.jsonl", "w") as f:
            for sp in self.steps:
                f.write(json.dumps({
                    "step": sp.step,
                    "phase": sp.phase,
                    "wall_time_s": round(sp.wall_time_s, 6),
                    "active_memory_gb": round(sp.active_memory_gb, 3),
                    "peak_memory_gb": round(sp.peak_memory_gb, 3),
                    "tokens_processed": sp.tokens_processed,
                    "loss": round(sp.loss, 6),
                    **{k: round(v, 6) if isinstance(v, float) else v for k, v in sp.extra.items()},
                }) + "\n")

        # Summary
        phases = {}
        for sp in self.steps:
            phases.setdefault(sp.phase, []).append(sp)

        lines = ["=" * 70, "MLX-Tinker Profiling Summary", "=" * 70, ""]
        lines.append(f"Total steps recorded: {len(self.steps)}")
        lines.append(f"Peak memory: {max(s.peak_memory_gb for s in self.steps):.3f} GB")
        lines.append("")

        for phase, steps in phases.items():
            times = [s.wall_time_s for s in steps]
            avg = sum(times) / len(times)
            total = sum(times)
            tokens = sum(s.tokens_processed for s in steps)
            lines.append(f"--- {phase} ---")
            lines.append(f"  Count:    {len(steps)}")
            lines.append(f"  Total:    {total:.3f}s")
            lines.append(f"  Avg:      {avg:.4f}s")
            lines.append(f"  Min:      {min(times):.4f}s")
            lines.append(f"  Max:      {max(times):.4f}s")
            if tokens:
                lines.append(f"  Tokens:   {tokens}")
                lines.append(f"  Tok/s:    {tokens / total:.1f}")
            lines.append("")

        # Loss curve
        fb_steps = [s for s in self.steps if s.phase == "forward_backward"]
        if fb_steps:
            lines.append("--- Loss Curve ---")
            for s in fb_steps:
                bar = "#" * int(s.loss * 10)
                lines.append(f"  step {s.step:3d}: {s.loss:.6f}  {bar}")
            lines.append("")

        summary = "\n".join(lines)
        (self.output_dir / "summary.txt").write_text(summary)
        print(summary)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dolly_data(tokenizer, max_samples: int = 500, max_seq_len: int = 256):
    """Load Dolly-15K and format as SFT training data."""
    from datasets import load_dataset

    logger.info("Loading databricks/databricks-dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    data = []
    for example in ds:
        if len(data) >= max_samples:
            break

        instruction = example["instruction"]
        context = example.get("context", "")
        response = example["response"]

        if context:
            prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        full_text = prompt + response

        tokens = tokenizer.encode(full_text)
        if len(tokens) < 10 or len(tokens) > max_seq_len:
            continue

        prompt_tokens = tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)

        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        # Mask prompt tokens, train on response
        loss_mask = [0.0] * min(prompt_len - 1, len(target_ids))
        loss_mask += [1.0] * (len(target_ids) - len(loss_mask))

        data.append((input_ids, target_ids, loss_mask))

    logger.info("Prepared %d training samples (max_seq_len=%d)", len(data), max_seq_len)
    return data


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_finetune(
    model_name: str,
    num_steps: int,
    lora_rank: int,
    learning_rate: float,
    max_seq_len: int,
    output_dir: Path,
    metal_trace: bool,
):
    import mlx.optimizers as optim
    from mlx_lm import load as mlx_load

    from mlx_tinker.backend.lora_manager import LoRAManager
    from mlx_tinker.types import LoraConfig

    profiler = Profiler(output_dir)

    # ---------------------------------------------------------------
    # 1. Load model
    # ---------------------------------------------------------------
    sp = profiler.begin(0, "model_load")
    logger.info("Loading model: %s", model_name)
    model, tokenizer = mlx_load(model_name)
    mx.eval(model.parameters())
    sp = profiler.end(sp)
    logger.info("Model loaded in %.2fs, memory=%.2f GB", sp.wall_time_s, sp.active_memory_gb)

    # ---------------------------------------------------------------
    # 2. Apply QLoRA
    # ---------------------------------------------------------------
    sp = profiler.begin(0, "apply_qlora")
    lora_config = LoraConfig(rank=lora_rank, alpha=lora_rank * 2.0, train_attn=True, train_mlp=True)
    manager = LoRAManager()
    model = manager.apply_qlora(model, lora_config, quantize_bits=4, quantize_group_size=64)
    mx.eval(model.parameters())
    trainable, total = manager.get_trainable_param_count(model)
    sp = profiler.end(sp, trainable_params=trainable, total_params=total)
    logger.info(
        "QLoRA applied in %.2fs: %d trainable / %d total (%.2f%%)",
        sp.wall_time_s, trainable, total, 100 * trainable / total,
    )

    # ---------------------------------------------------------------
    # 3. Load data
    # ---------------------------------------------------------------
    sp = profiler.begin(0, "data_load")
    data = load_dolly_data(tokenizer, max_samples=num_steps * 2, max_seq_len=max_seq_len)
    sp = profiler.end(sp, num_samples=len(data))

    if len(data) < num_steps:
        logger.warning("Only %d samples, will cycle", len(data))

    # ---------------------------------------------------------------
    # 4. Setup optimizer
    # ---------------------------------------------------------------
    optimizer = optim.AdamW(learning_rate=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    # Compiled train step
    def train_step(input_ids, targets, loss_mask):
        def loss_fn(model):
            logits = model(input_ids)
            seq_len = min(logits.shape[1], targets.shape[1])
            log_probs = logits[:, :seq_len] - mx.logsumexp(logits[:, :seq_len], axis=-1, keepdims=True)
            target_lp = mx.take_along_axis(
                log_probs, targets[:, :seq_len, None].astype(mx.int32), axis=-1
            ).squeeze(-1)
            masked = -target_lp * loss_mask[:, :seq_len]
            return masked.sum() / mx.maximum(loss_mask[:, :seq_len].sum(), 1.0)

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss_val, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        return loss_val

    # ---------------------------------------------------------------
    # 5. Training loop with profiling
    # ---------------------------------------------------------------
    logger.info("Starting %d training steps...", num_steps)

    # Optional Metal GPU trace
    if metal_trace:
        trace_path = str(output_dir / "mlx_trace.gputrace")
        logger.info("Metal trace will be captured to %s", trace_path)
        os.environ["MTL_CAPTURE_ENABLED"] = "1"

    # Warmup step (JIT compilation)
    sp = profiler.begin(0, "warmup")
    input_ids, target_ids, loss_mask_list = data[0]
    inp = mx.array(input_ids)[None, :]
    tgt = mx.array(target_ids, dtype=mx.int32)[None, :]
    mask = mx.array(loss_mask_list, dtype=mx.float32)[None, :]
    loss = train_step(inp, tgt, mask)
    mx.eval(loss, model.parameters(), optimizer.state)
    profiler.end(sp, tokens=len(input_ids))
    logger.info("Warmup complete")

    # Start Metal trace after warmup (captures steady-state, not JIT)
    if metal_trace:
        try:
            mx.metal.start_capture(str(output_dir / "mlx_trace.gputrace"))
            logger.info("Metal GPU capture started")
        except Exception as e:
            logger.warning("Metal capture failed (need MLX_METAL_DEBUG=ON build): %s", e)
            metal_trace = False

    for step in range(num_steps):
        sample_idx = step % len(data)
        input_ids, target_ids, loss_mask_list = data[sample_idx]

        # -- forward_backward --
        sp = profiler.begin(step, "forward_backward")
        inp = mx.array(input_ids)[None, :]
        tgt = mx.array(target_ids, dtype=mx.int32)[None, :]
        mask = mx.array(loss_mask_list, dtype=mx.float32)[None, :]

        loss = train_step(inp, tgt, mask)
        mx.eval(loss, model.parameters(), optimizer.state)

        sp.loss = loss.item()
        sp.tokens_processed = len(input_ids)
        profiler.end(sp)

        if step % 10 == 0 or step == num_steps - 1:
            logger.info(
                "step %d/%d  loss=%.4f  time=%.3fs  mem=%.2fGB  tok=%d",
                step, num_steps, sp.loss, sp.wall_time_s, sp.active_memory_gb, sp.tokens_processed,
            )

    if metal_trace:
        try:
            mx.metal.stop_capture()
            logger.info("Metal GPU capture saved")
        except Exception:
            pass

    # ---------------------------------------------------------------
    # 6. Save profiles
    # ---------------------------------------------------------------
    profiler.save()
    logger.info("Profiling data saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile MLX-Tinker QLoRA finetune")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-MLX-4bit",
                        help="Model name (default: 4-bit Qwen3.5-0.8B for fast iteration)")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--output", default="profiles", help="Output directory")
    parser.add_argument("--metal-trace", action="store_true", help="Capture Metal GPU trace")
    parser.add_argument("--cprofile", action="store_true", help="Enable cProfile")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cprofile:
        pr = cProfile.Profile()
        pr.enable()

    run_finetune(
        model_name=args.model,
        num_steps=args.steps,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        output_dir=output_dir,
        metal_trace=args.metal_trace,
    )

    if args.cprofile:
        pr.disable()
        # Save binary profile (for snakeviz)
        pr.dump_stats(str(output_dir / "cprofile_stats.prof"))

        # Also save text summary
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(60)
        (output_dir / "cprofile_top60.txt").write_text(s.getvalue())

        logger.info("cProfile saved to %s/cprofile_stats.prof", output_dir)
        logger.info("  View with: uv run snakeviz %s/cprofile_stats.prof", output_dir)


if __name__ == "__main__":
    main()
