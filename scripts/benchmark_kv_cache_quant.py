#!/usr/bin/env python3
"""Benchmark KV-cache quantization on the upstream mlx-lm generation path.

This script intentionally uses ``mlx_lm.generate.generate_step`` directly
because current mlx-tinker inference only forwards KV quantization options on
the single-sequence path if we wire them through. That makes this a good
measurement harness for the exact optimization under consideration before
changing the runtime.

Outputs:
    profiles/kv_cache_quant/kv_cache_quant.jsonl
    profiles/kv_cache_quant/kv_cache_quant.csv
    profiles/kv_cache_quant/summary.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.generate import generate_step
from mlx_lm.models import cache as cache_mod

from mlx_tinker.backend.lora_manager import LoRAManager
from mlx_tinker.types import LoraConfig


@dataclass
class BenchmarkRow:
    context_tokens: int
    kv_bits: str
    max_new_tokens: int
    quantized_kv_start: int
    kv_group_size: int
    time_to_first_token_s: float
    total_time_s: float
    tokens_per_s: float
    active_memory_mb: float
    peak_memory_mb: float
    cache_memory_mb: float
    prompt_cache_mb: float
    quantized_layers: int
    generated_tokens: int
    token_match_rate: float
    mean_abs_logprob_delta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output-dir", default="profiles/kv_cache_quant")
    parser.add_argument("--contexts", default="512,2048,8192")
    parser.add_argument("--kv-bits", default="none,8,4,2")
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qlora-rank", type=int, default=0)
    parser.add_argument("--weight-quantize-bits", type=int, default=4)
    parser.add_argument("--weight-quantize-group-size", type=int, default=64)
    return parser.parse_args()


def _repeat_to_length(tokens: list[int], target_len: int) -> list[int]:
    if not tokens:
        raise ValueError("Tokenizer produced no tokens for the synthetic prompt.")
    if len(tokens) >= target_len:
        return tokens[:target_len]
    repeats = (target_len + len(tokens) - 1) // len(tokens)
    return (tokens * repeats)[:target_len]


def _build_prompt(tokenizer: Any, target_len: int) -> list[int]:
    text = (
        "Long-context KV quantization benchmark prompt. "
        "This sentence is repeated to construct a stable prompt length. "
    )
    try:
        base = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        base = tokenizer.encode(text)
    return _repeat_to_length(list(base), target_len)


def _load_model(args: argparse.Namespace):
    model, tokenizer = mlx_load(args.model)
    if args.qlora_rank > 0:
        manager = LoRAManager()
        model = manager.apply_qlora(
            model,
            LoraConfig(rank=args.qlora_rank, alpha=args.qlora_rank * 2.0),
            quantize_bits=args.weight_quantize_bits,
            quantize_group_size=args.weight_quantize_group_size,
        )
    model.eval()
    mx.eval(model.parameters())
    return model, tokenizer


def _object_nbytes(obj: Any) -> int:
    if obj is None:
        return 0
    if isinstance(obj, mx.array):
        return int(obj.nbytes)
    if isinstance(obj, (list, tuple)):
        return sum(_object_nbytes(x) for x in obj)
    return 0


def _cache_nbytes(cache_obj: Any) -> int:
    try:
        return int(cache_obj.nbytes)
    except Exception:
        # Some upstream cache classes expose a broken nbytes property in this
        # environment. Fall back to summing their backing arrays directly.
        return _object_nbytes(
            (
                getattr(cache_obj, "keys", None),
                getattr(cache_obj, "values", None),
                getattr(cache_obj, "cache", None),
            )
        )


def _run_once(
    model,
    prompt_tokens: list[int],
    *,
    kv_bits: int | None,
    kv_group_size: int,
    quantized_kv_start: int,
    max_new_tokens: int,
    seed: int,
) -> tuple[list[int], list[float], dict[str, float]]:
    mx.random.seed(seed)
    prompt_cache = cache_mod.make_prompt_cache(model)
    prompt = mx.array(prompt_tokens)

    generated_tokens: list[int] = []
    token_logprobs: list[float] = []
    ttft: float | None = None

    mx.reset_peak_memory()
    start = time.perf_counter()
    for token, logprobs in generate_step(
        prompt=prompt,
        model=model,
        max_tokens=max_new_tokens,
        sampler=lambda x: mx.argmax(x, axis=-1),
        prompt_cache=prompt_cache,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    ):
        if ttft is None:
            ttft = time.perf_counter() - start
        token_id = int(token.item() if hasattr(token, "item") else token)
        log_probs_all = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
        token_logprob = float(log_probs_all.reshape(-1)[token_id].item())
        generated_tokens.append(token_id)
        token_logprobs.append(token_logprob)

    mx.synchronize()
    total_time = time.perf_counter() - start
    prompt_cache_mb = sum(_cache_nbytes(c) for c in prompt_cache) / 1e6
    quantized_layers = sum(1 for c in prompt_cache if getattr(c, "bits", None) is not None)
    metrics = {
        "time_to_first_token_s": ttft or total_time,
        "total_time_s": total_time,
        "tokens_per_s": len(generated_tokens) / total_time if total_time > 0 else 0.0,
        "active_memory_mb": mx.get_active_memory() / 1e6,
        "peak_memory_mb": mx.get_peak_memory() / 1e6,
        "cache_memory_mb": mx.get_cache_memory() / 1e6,
        "prompt_cache_mb": prompt_cache_mb,
        "quantized_layers": float(quantized_layers),
    }
    return generated_tokens, token_logprobs, metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    kv_bits_values: list[int | None] = []
    for raw in args.kv_bits.split(","):
        raw = raw.strip().lower()
        kv_bits_values.append(None if raw in {"none", "fp16", "off"} else int(raw))

    model, tokenizer = _load_model(args)
    rows: list[BenchmarkRow] = []

    for context_tokens in contexts:
        prompt_tokens = _build_prompt(tokenizer, context_tokens)

        baseline_tokens, baseline_logprobs, _baseline_metrics = _run_once(
            model,
            prompt_tokens,
            kv_bits=None,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )

        for kv_bits in kv_bits_values:
            generated_tokens, token_logprobs, metrics = _run_once(
                model,
                prompt_tokens,
                kv_bits=kv_bits,
                kv_group_size=args.kv_group_size,
                quantized_kv_start=args.quantized_kv_start,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
            )
            compare_len = min(len(baseline_tokens), len(generated_tokens))
            if compare_len == 0:
                token_match_rate = 1.0
                mean_abs_logprob_delta = 0.0
            else:
                matches = sum(
                    1
                    for idx in range(compare_len)
                    if baseline_tokens[idx] == generated_tokens[idx]
                )
                token_match_rate = matches / compare_len
                mean_abs_logprob_delta = sum(
                    abs(baseline_logprobs[idx] - token_logprobs[idx]) for idx in range(compare_len)
                ) / compare_len

            rows.append(
                BenchmarkRow(
                    context_tokens=context_tokens,
                    kv_bits="none" if kv_bits is None else str(kv_bits),
                    max_new_tokens=args.max_new_tokens,
                    quantized_kv_start=args.quantized_kv_start,
                    kv_group_size=args.kv_group_size,
                    time_to_first_token_s=metrics["time_to_first_token_s"],
                    total_time_s=metrics["total_time_s"],
                    tokens_per_s=metrics["tokens_per_s"],
                    active_memory_mb=metrics["active_memory_mb"],
                    peak_memory_mb=metrics["peak_memory_mb"],
                    cache_memory_mb=metrics["cache_memory_mb"],
                    prompt_cache_mb=metrics["prompt_cache_mb"],
                    quantized_layers=int(metrics["quantized_layers"]),
                    generated_tokens=len(generated_tokens),
                    token_match_rate=token_match_rate,
                    mean_abs_logprob_delta=mean_abs_logprob_delta,
                )
            )

    jsonl_path = output_dir / "kv_cache_quant.jsonl"
    csv_path = output_dir / "kv_cache_quant.csv"
    summary_path = output_dir / "summary.txt"

    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    lines = [
        "=" * 72,
        "KV Cache Quantization Benchmark",
        "=" * 72,
        "",
    ]
    for row in rows:
        lines.append(
            f"context={row.context_tokens:<6d} kv_bits={row.kv_bits:<4} "
            f"ttft={row.time_to_first_token_s:>6.3f}s total={row.total_time_s:>6.3f}s "
            f"tok/s={row.tokens_per_s:>7.2f} peak={row.peak_memory_mb:>9.1f}MB "
            f"cache={row.prompt_cache_mb:>8.1f}MB match={row.token_match_rate:>5.2f}"
        )

    summary = "\n".join(lines) + "\n"
    summary_path.write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
