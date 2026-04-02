#!/usr/bin/env python3
"""Profile phase-by-phase memory for rollout generation and training.

This script exercises the real mlx-tinker backend stack:

1. load base model
2. apply QLoRA/create live training model
3. run rollout generation through ``MLXBackend.sample()``
4. optionally clear Python/MLX caches
5. run ``forward_backward()``
6. run ``optim_step()``

It is meant to validate whether rollout KV/state meaningfully overlaps with the
subsequent training phase in the current codebase.

Usage:
    uv run python scripts/profile_rl_memory.py
    uv run python scripts/profile_rl_memory.py --model Qwen/Qwen3.5-4B --rollouts 8
    uv run python scripts/profile_rl_memory.py --compare-cleanup
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.config import EngineConfig
from mlx_tinker.types import (
    AdamParams,
    CreateModelInput,
    Datum,
    ForwardBackwardInput,
    LoraConfig,
    LossFnInputs,
    ModelInput,
    OptimStepInput,
    SampleInput,
    SamplingParams,
    TensorData,
    UnloadModelInput,
)


@dataclass
class MemorySnapshot:
    mode: str
    phase: str
    wall_time_s: float
    active_memory_mb: float
    peak_memory_mb: float
    cache_memory_mb: float
    gc_gen0: int
    gc_gen1: int
    gc_gen2: int
    note: str = ""
    generated_tokens: int = 0
    generated_sequences: int = 0
    train_tokens: int = 0
    loss_sum: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output-dir", default="profiles/rl_memory")
    parser.add_argument("--model-id", default="profile_model")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--optimizer-type", default="adamw")
    parser.add_argument("--quantize-bits", type=int, default=4)
    parser.add_argument("--quantize-group-size", type=int, default=64)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--train-seq-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-kv-cache-size", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["baseline", "cleanup"],
        default="baseline",
        help="Whether to insert explicit gc/cache clears between sample and training.",
    )
    parser.add_argument(
        "--compare-cleanup",
        action="store_true",
        help="Run both baseline and cleanup modes in one invocation.",
    )
    return parser.parse_args()


def _clear_runtime_caches() -> None:
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def _snapshot(
    timeline: list[MemorySnapshot],
    *,
    mode: str,
    phase: str,
    started_at: float,
    note: str = "",
    generated_tokens: int = 0,
    generated_sequences: int = 0,
    train_tokens: int = 0,
    loss_sum: float = 0.0,
) -> None:
    mx.synchronize()
    gc_counts = gc.get_count()
    timeline.append(
        MemorySnapshot(
            mode=mode,
            phase=phase,
            wall_time_s=time.perf_counter() - started_at,
            active_memory_mb=mx.get_active_memory() / 1e6,
            peak_memory_mb=mx.get_peak_memory() / 1e6,
            cache_memory_mb=mx.get_cache_memory() / 1e6,
            gc_gen0=gc_counts[0],
            gc_gen1=gc_counts[1],
            gc_gen2=gc_counts[2],
            note=note,
            generated_tokens=generated_tokens,
            generated_sequences=generated_sequences,
            train_tokens=train_tokens,
            loss_sum=loss_sum,
        )
    )


def _repeat_to_length(tokens: list[int], target_len: int) -> list[int]:
    if not tokens:
        raise ValueError("Tokenizer produced no tokens for the synthetic prompt.")
    if len(tokens) >= target_len:
        return tokens[:target_len]
    repeats = (target_len + len(tokens) - 1) // len(tokens)
    return (tokens * repeats)[:target_len]


def _build_token_sequence(tokenizer: Any, target_len: int, *, prefix: str) -> list[int]:
    template = (
        f"{prefix}\n"
        "This is a synthetic profiling sample for mlx-tinker memory tracing. "
        "We repeat this text to build a stable token length.\n"
    )
    try:
        base = tokenizer.encode(template, add_special_tokens=False)
    except TypeError:
        base = tokenizer.encode(template)
    return _repeat_to_length(list(base), target_len)


def _make_training_batch(tokenizer: Any, seq_len: int, batch_size: int) -> list[Datum]:
    tokens = _build_token_sequence(tokenizer, seq_len + 1, prefix="Training batch")
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = [1.0] * len(target_tokens)
    datum = Datum(
        model_input=ModelInput.from_tokens(input_tokens),
        loss_fn_inputs=LossFnInputs(
            target_tokens=TensorData(data=target_tokens, dtype="int64"),
            weights=TensorData(data=weights, dtype="float32"),
        ),
    )
    return [datum.model_copy(deep=True) for _ in range(batch_size)]


def _run_mode(args: argparse.Namespace, mode: str) -> list[MemorySnapshot]:
    timeline: list[MemorySnapshot] = []
    config = EngineConfig(
        base_model=args.model,
        quantize_bits=args.quantize_bits,
        quantize_group_size=args.quantize_group_size,
        optimizer_type=args.optimizer_type,
        gradient_checkpointing=args.gradient_checkpointing,
        max_kv_cache_size=args.max_kv_cache_size,
    )
    backend = MLXBackend(config)
    model_id = f"{args.model_id}_{mode}"

    _clear_runtime_caches()
    mx.reset_peak_memory()
    started = time.perf_counter()
    _snapshot(timeline, mode=mode, phase="process_start", started_at=started)

    create_started = time.perf_counter()
    backend.create_model(
        model_id,
        CreateModelInput(lora_config=LoraConfig(rank=args.lora_rank, alpha=args.lora_rank * 2.0)),
    )
    tokenizer = backend._get_tokenizer(model_id)
    _snapshot(timeline, mode=mode, phase="create_model", started_at=create_started)

    prompt_tokens = _build_token_sequence(tokenizer, args.prompt_tokens, prefix="Sampling prompt")
    sample_request = SampleInput(
        model_id=model_id,
        num_samples=args.rollouts,
        prompt=ModelInput.from_tokens(prompt_tokens),
        sampling_params=SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        ),
    )

    mx.reset_peak_memory()
    sample_started = time.perf_counter()
    sample_output = backend.sample(model_id, sample_request)
    generated_tokens = sum(len(seq.tokens) for seq in sample_output.sequences)
    _snapshot(
        timeline,
        mode=mode,
        phase="sample",
        started_at=sample_started,
        generated_tokens=generated_tokens,
        generated_sequences=len(sample_output.sequences),
    )

    if mode == "cleanup":
        cleanup_started = time.perf_counter()
        del sample_output
        _clear_runtime_caches()
        _snapshot(
            timeline,
            mode=mode,
            phase="post_sample_cleanup",
            started_at=cleanup_started,
            note="Applied gc.collect() + mx.clear_cache()",
        )

    training_batch = _make_training_batch(tokenizer, args.train_seq_len, args.train_batch_size)
    fb_request = ForwardBackwardInput(data=training_batch, loss_fn="cross_entropy")

    mx.reset_peak_memory()
    fb_started = time.perf_counter()
    fb_output = backend.forward_backward(model_id, fb_request)
    _snapshot(
        timeline,
        mode=mode,
        phase="forward_backward",
        started_at=fb_started,
        train_tokens=args.train_seq_len * args.train_batch_size,
        loss_sum=float(fb_output.metrics.get("loss:sum", 0.0)),
    )

    mx.reset_peak_memory()
    opt_started = time.perf_counter()
    backend.optim_step(
        model_id,
        OptimStepInput(adam_params=AdamParams(learning_rate=args.learning_rate)),
    )
    _snapshot(timeline, mode=mode, phase="optim_step", started_at=opt_started)

    unload_started = time.perf_counter()
    backend.unload_model(model_id, UnloadModelInput())
    del backend
    _clear_runtime_caches()
    _snapshot(timeline, mode=mode, phase="after_unload", started_at=unload_started)
    return timeline


def _write_outputs(output_dir: Path, mode: str, timeline: list[MemorySnapshot]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{mode}.jsonl"
    csv_path = output_dir / f"{mode}.csv"
    summary_path = output_dir / f"{mode}_summary.txt"

    with jsonl_path.open("w") as f:
        for row in timeline:
            f.write(json.dumps(asdict(row)) + "\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(timeline[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in timeline)

    lines = [
        "=" * 72,
        f"RL Memory Profile: {mode}",
        "=" * 72,
        "",
    ]
    for row in timeline:
        lines.append(
            f"{row.phase:<20} "
            f"time={row.wall_time_s:>7.3f}s "
            f"active={row.active_memory_mb:>9.1f}MB "
            f"peak={row.peak_memory_mb:>9.1f}MB "
            f"cache={row.cache_memory_mb:>9.1f}MB"
        )
        if row.note:
            lines.append(f"  note: {row.note}")

    if len(timeline) >= 3:
        by_phase = {row.phase: row for row in timeline}
        sample_peak = by_phase.get("sample")
        cleanup_row = by_phase.get("post_sample_cleanup")
        fb_row = by_phase.get("forward_backward")
        if sample_peak and fb_row:
            lines.append("")
            lines.append(
                "Sample peak vs forward_backward peak: "
                f"{sample_peak.peak_memory_mb:.1f}MB vs {fb_row.peak_memory_mb:.1f}MB"
            )
        if cleanup_row and fb_row:
            lines.append(
                "Forward-backward active memory after explicit cleanup: "
                f"{fb_row.active_memory_mb:.1f}MB (post-cleanup active was {cleanup_row.active_memory_mb:.1f}MB)"
            )

    summary = "\n".join(lines) + "\n"
    summary_path.write_text(summary)
    print(summary)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    modes = ["baseline", "cleanup"] if args.compare_cleanup else [args.mode]

    for mode in modes:
        timeline = _run_mode(args, mode)
        _write_outputs(output_dir, mode, timeline)


if __name__ == "__main__":
    main()
