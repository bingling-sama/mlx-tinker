#!/usr/bin/env python3
"""Compare memory usage and convergence of AdamW vs AdamW8Bit.

Usage:
    uv run python scripts/profile_optimizer_memory.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from mlx_tinker.backend.optimizers import AdamW8Bit


class SmallModel(nn.Module):
    """Model with enough parameters to show memory differences."""

    def __init__(self, vocab_size: int = 1024, dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = [nn.Linear(dim, dim) for _ in range(n_layers)]
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embed(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.head(h)


def measure_optimizer(opt_name: str, optimizer, model, num_steps: int = 20):
    """Run training steps and measure memory + time."""
    mx.eval(model.parameters())

    # Dummy data
    input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 4)  # [4, 8]
    targets = mx.array([[2, 3, 4, 5, 6, 7, 8, 9]] * 4, dtype=mx.int32)

    def loss_fn(model):
        logits = model(input_ids)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        target_lp = mx.take_along_axis(
            log_probs, targets[:, :, None], axis=-1
        ).squeeze(-1)
        return -target_lp.mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Warmup
    loss, grads = loss_and_grad(model)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters(), optimizer.state)

    mx.reset_peak_memory()

    losses = []
    start = time.perf_counter()
    for _ in range(num_steps):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)
        losses.append(loss.item())

    elapsed = time.perf_counter() - start
    peak_mem = mx.get_peak_memory() / 1e6

    # Measure optimizer state size
    state_size = 0
    for _, params_state in tree_flatten(optimizer.state):
        if isinstance(params_state, mx.array):
            state_size += params_state.nbytes

    # Count state arrays for more detail
    state_arrays = {}
    for name, arr in tree_flatten(optimizer.state):
        dtype = str(arr.dtype) if isinstance(arr, mx.array) else type(arr).__name__
        state_arrays.setdefault(dtype, 0)
        if isinstance(arr, mx.array):
            state_arrays[dtype] += arr.nbytes

    return {
        "name": opt_name,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "time_s": elapsed,
        "peak_mem_mb": peak_mem,
        "state_bytes": state_size,
        "state_breakdown": state_arrays,
        "steps": num_steps,
    }


def main():
    print("=" * 70)
    print("Optimizer Memory & Performance Comparison")
    print("=" * 70)

    # Count model params
    ref_model = SmallModel()
    mx.eval(ref_model.parameters())
    total_params = sum(p.size for _, p in tree_flatten(ref_model.parameters()))
    print(f"\nModel: {total_params:,} parameters")
    print(f"Expected fp32 state: ~{total_params * 4 * 2 / 1e6:.1f} MB (2x params for m,v)")
    print(f"Expected int8 state: ~{total_params * 1 * 2 / 1e6:.1f} MB (2x params as int8 + scales)")
    print()

    results = []

    # fp32 AdamW
    model_fp32 = SmallModel()
    mx.eval(model_fp32.parameters())
    opt_fp32 = optim.AdamW(learning_rate=1e-4)
    results.append(measure_optimizer("AdamW (fp32)", opt_fp32, model_fp32))

    # 8-bit AdamW
    model_8bit = SmallModel()
    mx.eval(model_8bit.parameters())
    opt_8bit = AdamW8Bit(learning_rate=1e-4)
    results.append(measure_optimizer("AdamW8Bit", opt_8bit, model_8bit))

    # Print results
    print(f"{'Optimizer':<20} {'First Loss':>12} {'Last Loss':>12} {'Time (s)':>10} {'Peak Mem (MB)':>14} {'State (MB)':>12}")
    print("-" * 82)
    for r in results:
        print(
            f"{r['name']:<20} {r['first_loss']:>12.4f} {r['last_loss']:>12.4f} "
            f"{r['time_s']:>10.3f} {r['peak_mem_mb']:>14.1f} {r['state_bytes']/1e6:>12.2f}"
        )

    print()
    if len(results) == 2:
        fp32 = results[0]
        q8 = results[1]
        savings = (1 - q8["state_bytes"] / max(fp32["state_bytes"], 1)) * 100
        print(f"State memory savings: {savings:.1f}%")
        print(f"Peak memory savings: {(1 - q8['peak_mem_mb'] / max(fp32['peak_mem_mb'], 1)) * 100:.1f}%")

    print()
    for r in results:
        print(f"{r['name']} state breakdown:")
        for dtype, nbytes in r["state_breakdown"].items():
            print(f"  {dtype}: {nbytes/1e6:.2f} MB")


if __name__ == "__main__":
    main()
