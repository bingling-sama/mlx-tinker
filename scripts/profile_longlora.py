#!/usr/bin/env python3
"""Compare baseline training against LongLoRA in isolated subprocesses.

Example:
    uv run python scripts/profile_longlora.py --model Qwen/Qwen3.5-0.8B --seq-len 2048
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LongLoRA against baseline training.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="Base model name or path")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length for the benchmark")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--group-size-ratio",
        type=float,
        default=0.25,
        help="LongLoRA group size ratio (default: 0.25)",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "longlora", "both"],
        default="both",
        help="Which mode to run",
    )
    return parser.parse_args()


def _single_run(args: argparse.Namespace, use_longlora: bool) -> dict:
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    from mlx_tinker.backend.gradient_checkpointing import enable_gradient_checkpointing
    from mlx_tinker.backend.longlora import enable_longlora_attention
    from mlx_tinker.backend.lora_manager import LoRAManager
    from mlx_tinker.backend.training import TrainingBackend
    from mlx_tinker.types import (
        Datum,
        EncodedTextChunk,
        ForwardBackwardInput,
        LossFnInputs,
        LoraConfig,
        ModelInput,
        TensorData,
    )

    random.seed(0)
    model, _tokenizer = mlx_load(args.model)
    manager = LoRAManager()
    model = manager.apply_qlora(
        model,
        LoraConfig(
            rank=args.rank,
            alpha=float(args.rank * 2),
            use_longlora=use_longlora,
            longlora_group_size_ratio=args.group_size_ratio,
        ),
        quantize_bits=4,
        quantize_group_size=64,
        train_embeddings=False,
        train_norms=False,
    )
    if use_longlora:
        enable_longlora_attention(model, group_size_ratio=args.group_size_ratio)
    enable_gradient_checkpointing(model)

    vocab_size = 151936
    request = ForwardBackwardInput(
        data=[
            Datum(
                model_input=ModelInput(
                    chunks=[
                        EncodedTextChunk(
                            tokens=[random.randrange(vocab_size) for _ in range(args.seq_len)]
                        )
                    ]
                ),
                loss_fn_inputs=LossFnInputs(
                    target_tokens=TensorData(
                        data=[random.randrange(vocab_size) for _ in range(args.seq_len)]
                    ),
                    weights=TensorData(data=[1.0] * args.seq_len),
                    advantages=TensorData(data=[0.0] * args.seq_len),
                    logprobs=TensorData(data=[0.0] * args.seq_len),
                ),
            )
        ],
        loss_fn="cross_entropy",
    )

    training = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=True)
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    output = training.forward_backward("bench", model, request)
    elapsed = time.perf_counter() - start
    return {
        "mode": "longlora" if use_longlora else "baseline",
        "seq_len": args.seq_len,
        "loss_sum": output.metrics["loss:sum"],
        "wall_time_s": elapsed,
        "active_memory_gb": mx.get_active_memory() / 1e9,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }


def _parse_embedded_json(stdout: str) -> dict:
    decoder = json.JSONDecoder()
    for start in range(len(stdout) - 1, -1, -1):
        if stdout[start] != "{":
            continue
        try:
            obj, end = decoder.raw_decode(stdout[start:])
        except json.JSONDecodeError:
            continue
        if stdout[start + end :].strip():
            continue
        return obj
    raise ValueError("No JSON object found in subprocess output")


def _spawn(args: argparse.Namespace, mode: str) -> dict:
    cmd = [
        sys.executable,
        __file__,
        "--model",
        args.model,
        "--seq-len",
        str(args.seq_len),
        "--rank",
        str(args.rank),
        "--group-size-ratio",
        str(args.group_size_ratio),
        "--mode",
        mode,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return _parse_embedded_json(proc.stdout)


def main() -> None:
    args = parse_args()
    if args.mode == "baseline":
        print(json.dumps(_single_run(args, use_longlora=False), indent=2))
        return
    if args.mode == "longlora":
        print(json.dumps(_single_run(args, use_longlora=True), indent=2))
        return

    baseline = _spawn(args, "baseline")
    longlora = _spawn(args, "longlora")
    comparison = {
        "baseline": baseline,
        "longlora": longlora,
        "delta_wall_time_s": longlora["wall_time_s"] - baseline["wall_time_s"],
        "delta_peak_memory_gb": longlora["peak_memory_gb"] - baseline["peak_memory_gb"],
        "speedup_pct": ((baseline["wall_time_s"] - longlora["wall_time_s"]) / baseline["wall_time_s"]) * 100.0,
        "peak_memory_reduction_pct": (
            (baseline["peak_memory_gb"] - longlora["peak_memory_gb"]) / baseline["peak_memory_gb"]
        )
        * 100.0,
    }
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
