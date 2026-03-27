#!/usr/bin/env python3
"""Cross-framework numerical audit: MLX-tinker ops vs PyTorch ground truth.

Tests every computational operation in mlx-tinker against a PyTorch reference,
comparing forward values, autograd gradients, and finite-difference gradients.
Optionally profiles wall-clock time and memory for both paths.

Usage:
    python scripts/cross_framework_audit.py              # Run all tests
    python scripts/cross_framework_audit.py --loss-only  # Loss functions only
    python scripts/cross_framework_audit.py --profile    # Include profiling
    python scripts/cross_framework_audit.py --json       # Machine-readable output
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Framework imports
# ---------------------------------------------------------------------------

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

try:
    import torch
except ImportError:
    print("ERROR: torch is required. Install with: uv sync --extra stress", file=sys.stderr)
    sys.exit(2)

# MLX-tinker imports
from mlx_tinker.backend.loss_fns import (
    CE_CHUNK_SIZE,
    LOSS_FUNCTION_MAP,
    LossFnConfig,
    chunked_cross_entropy_loss,
    cispo_loss,
    cross_entropy_loss,
    dro_loss,
    importance_sampling_loss,
    ppo_loss,
)
from mlx_tinker.backend.optimizers import (
    AdamW8Bit,
    _dequantize_blockwise,
    _quantize_blockwise,
    create_dynamic_map,
)
from mlx_tinker.backend.training import _clip_grad_norm

# Optional bitsandbytes
try:
    import bitsandbytes as bnb

    HAS_BNB = True
except ImportError:
    HAS_BNB = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-framework numerical audit")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--loss-only", action="store_true", help="Run only loss function tests")
    g.add_argument("--chunked-ce", action="store_true", help="Run only chunked CE tests")
    g.add_argument("--optimizer", action="store_true", help="Run only optimizer tests")
    g.add_argument("--logprob", action="store_true", help="Run only logprob tests")
    g.add_argument("--grad-clip", action="store_true", help="Run only gradient clipping tests")
    g.add_argument("--full-step", action="store_true", help="Run only full training step tests")
    g.add_argument("--dynamic-map", action="store_true", help="Run only dynamic map tests")
    g.add_argument("--real-model", action="store_true", help="Run only real model inference tests")
    g.add_argument("--tinker-api", action="store_true", help="Run only Tinker API parity tests (requires TINKER_API_KEY)")
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B", help="HF model for real-model tests")
    p.add_argument("--profile", action="store_true", help="Enable profiling (warmup + N runs)")
    p.add_argument("--verbose", action="store_true", help="Show per-element diff details")
    p.add_argument("--json", action="store_true", help="JSON output for CI")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# TestResult & reporting
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    category: str
    name: str
    passed: bool
    max_diff: float = 0.0
    mlx_ms: float = 0.0
    pt_ms: float = 0.0
    detail: str = ""
    skipped: bool = False
    extra: dict = field(default_factory=dict)


def _status_str(r: TestResult) -> str:
    if r.skipped:
        return "\033[33mSKIP\033[0m"
    return "\033[32mPASS\033[0m" if r.passed else "\033[31mFAIL\033[0m"


def report_table(results: list[TestResult], verbose: bool = False) -> None:
    hdr = f"{'Category':<14} {'Test':<40} {'Status':<6} {'MaxDiff':>10} {'MLX(ms)':>9} {'PT(ms)':>9}"
    sep = "-" * 94
    print()
    print(sep)
    print("  MLX-TINKER CROSS-FRAMEWORK AUDIT")
    print(sep)
    print(hdr)
    print(sep)
    for r in results:
        status = "SKIP" if r.skipped else ("PASS" if r.passed else "FAIL")
        color = "\033[33m" if r.skipped else ("\033[32m" if r.passed else "\033[31m")
        reset = "\033[0m"
        diff_s = f"{r.max_diff:.2e}" if r.max_diff > 0 else "-"
        mlx_s = f"{r.mlx_ms:.2f}" if r.mlx_ms > 0 else "-"
        pt_s = f"{r.pt_ms:.2f}" if r.pt_ms > 0 else "-"
        print(f"{r.category:<14} {r.name:<40} {color}{status:<6}{reset} {diff_s:>10} {mlx_s:>9} {pt_s:>9}")
        if verbose and r.detail and not r.passed:
            for line in r.detail.strip().split("\n"):
                print(f"               {line}")
    print(sep)
    n_pass = sum(1 for r in results if r.passed and not r.skipped)
    n_fail = sum(1 for r in results if not r.passed and not r.skipped)
    n_skip = sum(1 for r in results if r.skipped)
    total = len(results)
    print(f"  Summary: {n_pass}/{total} PASS, {n_fail} FAIL, {n_skip} SKIP")
    print(sep)
    print()


def report_json(results: list[TestResult]) -> None:
    out = []
    for r in results:
        d = asdict(r)
        d["status"] = "SKIP" if r.skipped else ("PASS" if r.passed else "FAIL")
        out.append(d)
    n_fail = sum(1 for r in results if not r.passed and not r.skipped)
    print(json.dumps({"tests": out, "n_fail": n_fail}, indent=2, default=str))


# ---------------------------------------------------------------------------
# Tensor conversion utilities
# ---------------------------------------------------------------------------


def np_to_mx(arr: np.ndarray) -> mx.array:
    return mx.array(arr)


def np_to_pt(arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr)
    if requires_grad and t.is_floating_point():
        t = t.requires_grad_(True)
    return t


def mx_to_np(arr: mx.array) -> np.ndarray:
    if arr.dtype == mx.bfloat16:
        arr = arr.astype(mx.float32)
    mx.eval(arr)
    return np.array(arr)


def pt_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(
    a: np.ndarray,
    b: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> tuple[bool, float, str]:
    """Compare two numpy arrays. Returns (passed, max_abs_diff, detail_str)."""
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    if a.shape != b.shape:
        return False, float("inf"), f"Shape mismatch: {a.shape} vs {b.shape}"

    with np.errstate(invalid="ignore"):
        abs_diff = np.abs(a - b)
    max_abs = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0

    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom == 0, 1.0, denom)
    rel_diff = abs_diff / denom

    failures = (abs_diff > atol) & (rel_diff > rtol)
    n_fail = int(np.sum(failures))
    if n_fail > 0:
        worst = int(np.argmax(abs_diff))
        detail = (
            f"{n_fail}/{len(a)} elements failed (rtol={rtol}, atol={atol})\n"
            f"  worst idx={worst}: a={a[worst]:.8e} b={b[worst]:.8e} "
            f"abs={abs_diff[worst]:.4e} rel={rel_diff[worst]:.4e}"
        )
        return False, max_abs, detail
    return True, max_abs, ""


# ---------------------------------------------------------------------------
# Timing & memory
# ---------------------------------------------------------------------------


def time_fn(fn: Callable, n_warmup: int = 1, n_runs: int = 5) -> tuple[Any, float]:
    """Time a function, returning (result, median_ms)."""
    for _ in range(n_warmup):
        fn()
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return result, statistics.median(times)


def get_mlx_memory() -> dict:
    return {
        "active_mb": mx.get_active_memory() / (1024 * 1024),
        "peak_mb": mx.get_peak_memory() / (1024 * 1024),
    }


def get_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)  # bytes -> MB
    return ru / 1024  # KB -> MB


# ---------------------------------------------------------------------------
# PyTorch reference loss functions
# ---------------------------------------------------------------------------


def pt_cross_entropy_loss(
    target_lp: torch.Tensor,
    mask: torch.Tensor,
    _samp_lp: torch.Tensor,
    _adv: torch.Tensor,
    _cfg: dict,
) -> torch.Tensor:
    return (-target_lp * mask).sum()


def pt_importance_sampling_loss(
    target_lp: torch.Tensor,
    _mask: torch.Tensor,
    samp_lp: torch.Tensor,
    adv: torch.Tensor,
    _cfg: dict,
) -> torch.Tensor:
    ratio = torch.exp(target_lp - samp_lp)
    return -(ratio * adv).sum()


def pt_ppo_loss(
    target_lp: torch.Tensor,
    _mask: torch.Tensor,
    samp_lp: torch.Tensor,
    adv: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    ratio = torch.exp(target_lp - samp_lp)
    clipped = torch.clamp(ratio, 1.0 - cfg["clip_low"], 1.0 + cfg["clip_high"])
    return -torch.minimum(ratio * adv, clipped * adv).sum()


def pt_cispo_loss(
    target_lp: torch.Tensor,
    _mask: torch.Tensor,
    samp_lp: torch.Tensor,
    adv: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    ratio = torch.exp(target_lp - samp_lp)
    pos = adv > 0
    clipped = torch.where(
        pos,
        torch.clamp(ratio, 1.0 - cfg["clip_high"], 1.0 + cfg["clip_high"]),
        torch.clamp(ratio, 1.0 - cfg["clip_low"], 1.0 + cfg["clip_low"]),
    )
    return -(clipped.detach() * target_lp * adv).sum()


def pt_dro_loss(
    target_lp: torch.Tensor,
    _mask: torch.Tensor,
    samp_lp: torch.Tensor,
    adv: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    quad = (target_lp - samp_lp) ** 2
    return -(target_lp * adv - 0.5 * cfg["beta"] * quad).sum()


def pt_chunked_cross_entropy_loss(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """PyTorch reference of chunked CE with running logsumexp."""
    V = weight.shape[0]
    target_weight = weight[targets]  # [B, T, D]
    target_logits = (hidden * target_weight).sum(dim=-1)  # [B, T]

    running_max = torch.full(target_logits.shape, float("-inf"))
    running_sum_exp = torch.zeros_like(target_logits)

    for chunk_start in range(0, V, chunk_size):
        chunk_end = min(chunk_start + chunk_size, V)
        chunk_w = weight[chunk_start:chunk_end]
        chunk_logits = hidden @ chunk_w.T  # [B, T, chunk]

        chunk_max = chunk_logits.max(dim=-1).values
        new_max = torch.maximum(running_max, chunk_max)

        running_sum_exp = running_sum_exp * torch.exp(running_max - new_max)
        running_sum_exp = running_sum_exp + torch.exp(
            chunk_logits - new_max.unsqueeze(-1)
        ).sum(dim=-1)
        running_max = new_max

    logsumexp = running_max + torch.log(running_sum_exp)
    target_logprobs = target_logits - logsumexp
    return (-target_logprobs * mask).sum()


def pt_standard_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Standard (non-chunked) CE: full logits -> logsumexp -> gather."""
    logits = hidden @ weight.T  # [B, T, V]
    log_probs = torch.log_softmax(logits, dim=-1)
    target_lp = torch.gather(log_probs, 2, targets.unsqueeze(-1).long()).squeeze(-1)
    return (-target_lp * mask).sum()


# Mapping MLX loss names to (mlx_fn, pt_fn) pairs
MLX_LOSS_MAP = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
    "ppo": ppo_loss,
    "cispo": cispo_loss,
    "dro": dro_loss,
}

PT_LOSS_MAP = {
    "cross_entropy": pt_cross_entropy_loss,
    "importance_sampling": pt_importance_sampling_loss,
    "ppo": pt_ppo_loss,
    "cispo": pt_cispo_loss,
    "dro": pt_dro_loss,
}


# ---------------------------------------------------------------------------
# PyTorch reference optimizer (no bias correction, matching MLX)
# ---------------------------------------------------------------------------


class PtAdamWNoBiasCorrection:
    """Manual AdamW without bias correction — matches MLX optim.AdamW semantics."""

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state: dict[int, dict] = {}
        for p in params:
            self.state[id(p)] = {
                "m": torch.zeros_like(p),
                "v": torch.zeros_like(p),
            }

    def step(self, params_and_grads: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        b1, b2 = self.betas
        for p, g in params_and_grads:
            s = self.state[id(p)]
            s["m"] = b1 * s["m"] + (1 - b1) * g
            s["v"] = b2 * s["v"] + (1 - b2) * g**2
            update = s["m"] / (torch.sqrt(s["v"]) + self.eps)
            p.data = p.data * (1 - self.lr * self.weight_decay) - self.lr * update


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------


def gen_loss_inputs(
    rng: np.random.RandomState,
    regime: str,
    B: int = 2,
    T: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate (target_lp, mask, sampling_lp, advantages) as float32 numpy arrays."""
    if regime == "normal":
        target_lp = rng.uniform(-5.0, -0.1, (B, T)).astype(np.float32)
        mask = (rng.random((B, T)) > 0.3).astype(np.float32)
        sampling_lp = rng.uniform(-5.0, -0.1, (B, T)).astype(np.float32)
        advantages = rng.randn(B, T).astype(np.float32)
    elif regime == "extreme":
        target_lp = rng.uniform(-500.0, -100.0, (B, T)).astype(np.float32)
        mask = np.ones((B, T), dtype=np.float32)
        sampling_lp = rng.uniform(-500.0, -100.0, (B, T)).astype(np.float32)
        advantages = rng.randn(B, T).astype(np.float32)
    elif regime == "near_zero":
        target_lp = rng.uniform(-0.01, 0.0, (B, T)).astype(np.float32)
        mask = np.ones((B, T), dtype=np.float32)
        sampling_lp = rng.uniform(-0.01, 0.0, (B, T)).astype(np.float32)
        advantages = rng.randn(B, T).astype(np.float32) * 0.1
    elif regime == "large_ratio":
        target_lp = rng.uniform(-2.0, -0.1, (B, T)).astype(np.float32)
        mask = np.ones((B, T), dtype=np.float32)
        sampling_lp = target_lp - rng.uniform(2.0, 10.0, (B, T)).astype(np.float32)
        advantages = rng.randn(B, T).astype(np.float32)
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return target_lp, mask, sampling_lp, advantages


def gen_chunked_ce_inputs(
    rng: np.random.RandomState,
    B: int = 2,
    T: int = 8,
    V: int = 128,
    D: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate (hidden [B,T,D], weight [V,D], targets [B,T], mask [B,T])."""
    hidden = rng.randn(B, T, D).astype(np.float32) * 0.1
    weight = rng.randn(V, D).astype(np.float32) * 0.1
    targets = rng.randint(0, V, (B, T)).astype(np.int32)
    mask = (rng.random((B, T)) > 0.2).astype(np.float32)
    return hidden, weight, targets, mask


def gen_logits(
    rng: np.random.RandomState,
    B: int = 2,
    T: int = 8,
    V: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random logits [B,T,V] and target tokens [B,T]."""
    logits = rng.randn(B, T, V).astype(np.float32)
    targets = rng.randint(0, V, (B, T)).astype(np.int32)
    return logits, targets


def gen_optimizer_inputs(
    rng: np.random.RandomState,
    shape: tuple[int, ...] = (64, 32),
    n_steps: int = 5,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate initial params + list of gradient arrays."""
    params = rng.randn(*shape).astype(np.float32) * 0.1
    grads = [rng.randn(*shape).astype(np.float32) * 0.01 for _ in range(n_steps)]
    return params, grads


# ---------------------------------------------------------------------------
# Category A: Loss function parity
# ---------------------------------------------------------------------------


def _get_loss_cfg(loss_name: str) -> tuple[LossFnConfig, dict]:
    """Return (mlx_cfg, pt_cfg_dict) for a given loss function."""
    if loss_name in ("ppo", "cispo"):
        mlx_cfg = LossFnConfig(clip_low_threshold=0.2, clip_high_threshold=0.2)
        pt_cfg = {"clip_low": 0.2, "clip_high": 0.2, "beta": 0.05}
    elif loss_name == "dro":
        mlx_cfg = LossFnConfig(beta=0.05)
        pt_cfg = {"clip_low": 0.0, "clip_high": float("inf"), "beta": 0.05}
    else:
        mlx_cfg = LossFnConfig()
        pt_cfg = {"clip_low": 0.0, "clip_high": float("inf"), "beta": 0.05}
    return mlx_cfg, pt_cfg


def run_loss_tests(rng: np.random.RandomState, args: argparse.Namespace) -> list[TestResult]:
    results = []
    regimes = ["normal", "extreme", "near_zero", "large_ratio"]
    loss_names = ["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"]

    # A1: Forward parity across regimes
    for loss_name in loss_names:
        mlx_fn = MLX_LOSS_MAP[loss_name]
        pt_fn = PT_LOSS_MAP[loss_name]
        mlx_cfg, pt_cfg = _get_loss_cfg(loss_name)

        for regime in regimes:
            sub_rng = np.random.RandomState(rng.randint(0, 2**31))
            target_lp, mask, samp_lp, adv = gen_loss_inputs(sub_rng, regime)

            # MLX
            def run_mlx():
                val = mlx_fn(
                    np_to_mx(target_lp), np_to_mx(mask),
                    np_to_mx(samp_lp), np_to_mx(adv), mlx_cfg,
                )
                mx.eval(val)
                return val

            # PyTorch
            def run_pt():
                return pt_fn(
                    np_to_pt(target_lp), np_to_pt(mask),
                    np_to_pt(samp_lp), np_to_pt(adv), pt_cfg,
                )

            if args.profile:
                mlx_val, mlx_ms = time_fn(run_mlx, n_warmup=2, n_runs=5)
                pt_val, pt_ms = time_fn(run_pt, n_warmup=2, n_runs=5)
            else:
                mlx_val = run_mlx()
                mlx_ms = 0.0
                pt_val = run_pt()
                pt_ms = 0.0

            passed, max_diff, detail = compare(
                mx_to_np(mlx_val), pt_to_np(pt_val), rtol=1e-5, atol=1e-5
            )
            results.append(TestResult(
                category="Loss",
                name=f"{loss_name}/{regime}/fwd",
                passed=passed,
                max_diff=max_diff,
                mlx_ms=mlx_ms,
                pt_ms=pt_ms,
                detail=detail,
            ))

    # A2: Gradient parity (normal regime, grad w.r.t. target_logprobs)
    for loss_name in loss_names:
        mlx_fn = MLX_LOSS_MAP[loss_name]
        pt_fn = PT_LOSS_MAP[loss_name]
        mlx_cfg, pt_cfg = _get_loss_cfg(loss_name)

        sub_rng = np.random.RandomState(rng.randint(0, 2**31))
        target_lp, mask, samp_lp, adv = gen_loss_inputs(sub_rng, "normal")

        # MLX grad
        def mlx_loss_of_lp(lp):
            return mlx_fn(lp, np_to_mx(mask), np_to_mx(samp_lp), np_to_mx(adv), mlx_cfg)

        mlx_grad = mx.grad(mlx_loss_of_lp)(np_to_mx(target_lp))
        mx.eval(mlx_grad)
        mlx_grad_np = mx_to_np(mlx_grad)

        # PyTorch grad
        pt_lp = np_to_pt(target_lp, requires_grad=True)
        pt_loss = pt_fn(pt_lp, np_to_pt(mask), np_to_pt(samp_lp), np_to_pt(adv), pt_cfg)
        pt_loss.backward()
        pt_grad_np = pt_to_np(pt_lp.grad)

        passed, max_diff, detail = compare(mlx_grad_np, pt_grad_np, rtol=1e-3, atol=1e-5)
        results.append(TestResult(
            category="Loss",
            name=f"{loss_name}/grad_parity",
            passed=passed,
            max_diff=max_diff,
            detail=detail,
        ))

    # A3: Finite-difference checks (CE, IS, PPO, DRO — not CISPO)
    fd_losses = ["cross_entropy", "importance_sampling", "ppo", "dro"]
    for loss_name in fd_losses:
        mlx_fn = MLX_LOSS_MAP[loss_name]
        mlx_cfg, _ = _get_loss_cfg(loss_name)

        sub_rng = np.random.RandomState(rng.randint(0, 2**31))
        if loss_name in ("ppo", "importance_sampling"):
            # Keep ratios moderate to avoid steep gradients that break FD
            target_lp_np = sub_rng.uniform(-2.0, -0.5, (2, 6)).astype(np.float32)
            samp_lp_np = target_lp_np + sub_rng.uniform(-0.05, 0.05, (2, 6)).astype(np.float32)
            mask_np = np.ones((2, 6), dtype=np.float32)
            adv_np = sub_rng.randn(2, 6).astype(np.float32) * 0.5
            fd_eps = 5e-4
            fd_rtol, fd_atol = 5e-3, 2e-3
        elif loss_name == "dro":
            # DRO quadratic term amplifies FD error; use moderate inputs
            target_lp_np = sub_rng.uniform(-3.0, -0.5, (2, 6)).astype(np.float32)
            samp_lp_np = target_lp_np + sub_rng.uniform(-0.1, 0.1, (2, 6)).astype(np.float32)
            mask_np = np.ones((2, 6), dtype=np.float32)
            adv_np = sub_rng.randn(2, 6).astype(np.float32) * 0.5
            fd_eps = 1e-4
            fd_rtol, fd_atol = 5e-3, 2e-3
        else:
            target_lp_np, mask_np, samp_lp_np, adv_np = gen_loss_inputs(sub_rng, "normal", B=2, T=6)
            fd_eps = 1e-4
            fd_rtol, fd_atol = 2e-3, 1e-3

        # MLX autograd
        def mlx_loss_for_fd(lp):
            return mlx_fn(lp, np_to_mx(mask_np), np_to_mx(samp_lp_np), np_to_mx(adv_np), mlx_cfg)

        mlx_grad = mx.grad(mlx_loss_for_fd)(np_to_mx(target_lp_np))
        mx.eval(mlx_grad)
        analytical_np = mx_to_np(mlx_grad).flatten()

        # Central difference
        flat = target_lp_np.flatten()
        numerical = np.zeros_like(flat)
        for i in range(len(flat)):
            x_plus = flat.copy()
            x_plus[i] += fd_eps
            x_minus = flat.copy()
            x_minus[i] -= fd_eps
            f_plus = mlx_loss_for_fd(mx.array(x_plus.reshape(target_lp_np.shape)))
            f_minus = mlx_loss_for_fd(mx.array(x_minus.reshape(target_lp_np.shape)))
            mx.eval(f_plus, f_minus)
            numerical[i] = (f_plus.item() - f_minus.item()) / (2 * fd_eps)

        passed, max_diff, detail = compare(analytical_np, numerical, rtol=fd_rtol, atol=fd_atol)
        results.append(TestResult(
            category="Loss",
            name=f"{loss_name}/fd_check",
            passed=passed,
            max_diff=max_diff,
            detail=detail,
        ))

    # A4: CISPO analytical gradient check
    sub_rng = np.random.RandomState(rng.randint(0, 2**31))
    target_lp_np, _, samp_lp_np, adv_np = gen_loss_inputs(sub_rng, "normal", B=2, T=6)
    mask_np = np.ones((2, 6), dtype=np.float32)
    mlx_cfg, _ = _get_loss_cfg("cispo")

    def cispo_of_lp(lp):
        return cispo_loss(lp, np_to_mx(mask_np), np_to_mx(samp_lp_np), np_to_mx(adv_np), mlx_cfg)

    mlx_grad = mx.grad(cispo_of_lp)(np_to_mx(target_lp_np))
    mx.eval(mlx_grad)
    mlx_grad_np = mx_to_np(mlx_grad)

    # Analytical: d/d(target_lp) of -(sg(clipped_ratio) * target_lp * adv).sum()
    #           = -sg(clipped_ratio) * adv
    ratio_np = np.exp(target_lp_np - samp_lp_np)
    pos = adv_np > 0
    clip_lo, clip_hi = 0.2, 0.2
    clipped_np = np.where(
        pos,
        np.clip(ratio_np, 1.0 - clip_hi, 1.0 + clip_hi),
        np.clip(ratio_np, 1.0 - clip_lo, 1.0 + clip_lo),
    )
    expected_grad = -clipped_np * adv_np

    passed, max_diff, detail = compare(mlx_grad_np, expected_grad, rtol=1e-5, atol=1e-5)
    results.append(TestResult(
        category="Loss",
        name="cispo/analytical_grad",
        passed=passed,
        max_diff=max_diff,
        detail=detail,
    ))

    return results


# ---------------------------------------------------------------------------
# Category B: Chunked CE parity
# ---------------------------------------------------------------------------


def run_chunked_ce_tests(rng: np.random.RandomState, args: argparse.Namespace) -> list[TestResult]:
    import mlx_tinker.backend.loss_fns as lf_module

    results = []
    vocab_sizes = [128, 1024, 32000]
    chunk_sizes = [8, 64, 8192]

    # B1: Chunked vs standard CE (MLX only — algorithmic equivalence)
    for V in vocab_sizes:
        for cs in chunk_sizes:
            if cs >= V:
                continue  # chunking doesn't apply when chunk >= vocab
            sub_rng = np.random.RandomState(rng.randint(0, 2**31))
            hidden_np, weight_np, targets_np, mask_np = gen_chunked_ce_inputs(sub_rng, V=V, D=16)

            # Standard CE via full logits
            h_mx = np_to_mx(hidden_np)
            w_mx = np_to_mx(weight_np)
            t_mx = np_to_mx(targets_np)
            m_mx = np_to_mx(mask_np)

            logits_mx = h_mx @ mx.transpose(w_mx)  # [B, T, V]
            lp_mx = logits_mx - mx.logsumexp(logits_mx, axis=-1, keepdims=True)
            target_lp_mx = mx.take_along_axis(lp_mx, t_mx[:, :, None], axis=-1).squeeze(-1)
            standard_loss = (-target_lp_mx * m_mx).sum()
            mx.eval(standard_loss)

            # Chunked CE
            old_cs = lf_module.CE_CHUNK_SIZE
            try:
                lf_module.CE_CHUNK_SIZE = cs
                chunked_loss = chunked_cross_entropy_loss(h_mx, w_mx, t_mx, m_mx)
                mx.eval(chunked_loss)
            finally:
                lf_module.CE_CHUNK_SIZE = old_cs

            passed, max_diff, detail = compare(
                mx_to_np(standard_loss), mx_to_np(chunked_loss), rtol=1e-5, atol=1e-5
            )
            results.append(TestResult(
                category="ChunkedCE",
                name=f"mlx_std_vs_chunk/V={V}/cs={cs}",
                passed=passed,
                max_diff=max_diff,
                detail=detail,
            ))

    # B2: MLX chunked CE vs PyTorch chunked CE (cross-framework)
    for V in vocab_sizes:
        sub_rng = np.random.RandomState(rng.randint(0, 2**31))
        hidden_np, weight_np, targets_np, mask_np = gen_chunked_ce_inputs(sub_rng, V=V, D=16)
        cs = min(64, V)

        # MLX
        old_cs = lf_module.CE_CHUNK_SIZE
        try:
            lf_module.CE_CHUNK_SIZE = cs
            mlx_val = chunked_cross_entropy_loss(
                np_to_mx(hidden_np), np_to_mx(weight_np),
                np_to_mx(targets_np), np_to_mx(mask_np),
            )
            mx.eval(mlx_val)
        finally:
            lf_module.CE_CHUNK_SIZE = old_cs

        # PyTorch
        pt_val = pt_chunked_cross_entropy_loss(
            np_to_pt(hidden_np), np_to_pt(weight_np),
            np_to_pt(targets_np.astype(np.int64)), np_to_pt(mask_np),
            chunk_size=cs,
        )

        passed, max_diff, detail = compare(
            mx_to_np(mlx_val), pt_to_np(pt_val), rtol=1e-5, atol=1e-5
        )
        results.append(TestResult(
            category="ChunkedCE",
            name=f"mlx_vs_pt_chunked/V={V}",
            passed=passed,
            max_diff=max_diff,
            detail=detail,
        ))

    # B3: MLX chunked CE vs PyTorch standard CE (gold standard)
    for V in vocab_sizes:
        sub_rng = np.random.RandomState(rng.randint(0, 2**31))
        hidden_np, weight_np, targets_np, mask_np = gen_chunked_ce_inputs(sub_rng, V=V, D=16)

        # MLX chunked
        old_cs = lf_module.CE_CHUNK_SIZE
        try:
            lf_module.CE_CHUNK_SIZE = 64
            mlx_val = chunked_cross_entropy_loss(
                np_to_mx(hidden_np), np_to_mx(weight_np),
                np_to_mx(targets_np), np_to_mx(mask_np),
            )
            mx.eval(mlx_val)
        finally:
            lf_module.CE_CHUNK_SIZE = old_cs

        # PyTorch standard (non-chunked) — gold standard
        pt_val = pt_standard_cross_entropy(
            np_to_pt(hidden_np), np_to_pt(weight_np),
            np_to_pt(targets_np.astype(np.int64)), np_to_pt(mask_np),
        )

        passed, max_diff, detail = compare(
            mx_to_np(mlx_val), pt_to_np(pt_val), rtol=1e-4, atol=1e-4
        )
        results.append(TestResult(
            category="ChunkedCE",
            name=f"mlx_chunk_vs_pt_std/V={V}",
            passed=passed,
            max_diff=max_diff,
            detail=detail,
        ))

    # B4: Gradient check for chunked CE (V=128, cs=8)
    sub_rng = np.random.RandomState(rng.randint(0, 2**31))
    hidden_np, weight_np, targets_np, mask_np = gen_chunked_ce_inputs(sub_rng, B=1, T=4, V=128, D=16)

    # MLX gradient w.r.t. hidden_states
    def mlx_chunked_of_hidden(h):
        old = lf_module.CE_CHUNK_SIZE
        try:
            lf_module.CE_CHUNK_SIZE = 8
            return chunked_cross_entropy_loss(h, np_to_mx(weight_np), np_to_mx(targets_np), np_to_mx(mask_np))
        finally:
            lf_module.CE_CHUNK_SIZE = old

    mlx_grad = mx.grad(mlx_chunked_of_hidden)(np_to_mx(hidden_np))
    mx.eval(mlx_grad)
    mlx_grad_np = mx_to_np(mlx_grad)

    # PyTorch gradient w.r.t. hidden_states
    pt_h = np_to_pt(hidden_np, requires_grad=True)
    pt_loss = pt_standard_cross_entropy(
        pt_h, np_to_pt(weight_np),
        np_to_pt(targets_np.astype(np.int64)), np_to_pt(mask_np),
    )
    pt_loss.backward()
    pt_grad_np = pt_to_np(pt_h.grad)

    passed, max_diff, detail = compare(mlx_grad_np, pt_grad_np, rtol=1e-3, atol=1e-4)
    results.append(TestResult(
        category="ChunkedCE",
        name="grad_hidden/V=128",
        passed=passed,
        max_diff=max_diff,
        detail=detail,
    ))

    return results


# ---------------------------------------------------------------------------
# Category C: Optimizer parity
# ---------------------------------------------------------------------------


def run_optimizer_tests(rng: np.random.RandomState, args: argparse.Namespace) -> list[TestResult]:
    results = []

    # C1: Quantize/dequantize roundtrip
    sub_rng = np.random.RandomState(rng.randint(0, 2**31))
    tensor_np = sub_rng.randn(1024).astype(np.float32)
    q, absmax = _quantize_blockwise(mx.array(tensor_np))
    deq = _dequantize_blockwise(q, absmax)
    mx.eval(deq)
    deq_np = np.array(deq)[:1024]
    max_err = float(np.max(np.abs(deq_np - tensor_np)))
    tensor_range = float(tensor_np.max() - tensor_np.min())
    rel_err = max_err / tensor_range if tensor_range > 0 else max_err
    passed = rel_err < 0.01
    results.append(TestResult(
        category="Optimizer",
        name="quantize_roundtrip",
        passed=passed,
        max_diff=max_err,
        detail=f"rel_err={rel_err:.4e} (threshold: 1%)" if not passed else "",
    ))

    # C2-C4: AdamW8Bit vs fp32 AdamW at N steps
    for n_steps in [1, 5, 20]:
        sub_rng = np.random.RandomState(rng.randint(0, 2**31))
        param_np, grad_list = gen_optimizer_inputs(sub_rng, shape=(64, 32), n_steps=n_steps)
        lr = 1e-4
        betas = (0.9, 0.999)
        eps = 1e-8

        # MLX AdamW8Bit path
        mlx_param = mx.array(param_np.copy())

        class SingleParam(nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w

        model_8bit = SingleParam(mlx_param)
        opt_8bit = AdamW8Bit(learning_rate=lr, betas=list(betas), eps=eps)

        for step_i in range(n_steps):
            g = mx.array(grad_list[step_i])
            opt_8bit.update(model_8bit, {"w": g})
            mx.eval(model_8bit.parameters(), opt_8bit.state)

        mlx_result_np = mx_to_np(model_8bit.w)

        # PyTorch manual AdamW (no bias correction)
        pt_param = torch.tensor(param_np.copy())
        pt_opt = PtAdamWNoBiasCorrection([pt_param], lr=lr, betas=betas, eps=eps)
        for step_i in range(n_steps):
            g = torch.tensor(grad_list[step_i])
            pt_opt.step([(pt_param, g)])

        pt_result_np = pt_to_np(pt_param)

        # Compare via cosine similarity and magnitude ratio
        cos_sim = float(np.dot(mlx_result_np.flatten(), pt_result_np.flatten()) / (
            np.linalg.norm(mlx_result_np) * np.linalg.norm(pt_result_np) + 1e-12
        ))
        mag_ratio = float(np.linalg.norm(mlx_result_np) / (np.linalg.norm(pt_result_np) + 1e-12))
        max_abs_diff = float(np.max(np.abs(mlx_result_np - pt_result_np)))

        passed = cos_sim > 0.95 and 0.5 < mag_ratio < 2.0
        detail = f"cosine={cos_sim:.6f} mag_ratio={mag_ratio:.6f}" if not passed else ""
        results.append(TestResult(
            category="Optimizer",
            name=f"adamw8bit_vs_fp32/{n_steps}steps",
            passed=passed,
            max_diff=max_abs_diff,
            detail=detail,
            extra={"cosine_sim": cos_sim, "mag_ratio": mag_ratio},
        ))

    # C5: 8-bit vs fp32 drift (MLX only — isolates quantization error)
    sub_rng = np.random.RandomState(rng.randint(0, 2**31))
    param_np, grad_list = gen_optimizer_inputs(sub_rng, shape=(64, 32), n_steps=20)
    lr = 1e-4

    # MLX fp32 path
    model_fp32 = SingleParam(mx.array(param_np.copy()))
    opt_fp32 = optim.AdamW(learning_rate=lr)
    for step_i in range(20):
        g = mx.array(grad_list[step_i])
        opt_fp32.update(model_fp32, {"w": g})
        mx.eval(model_fp32.parameters(), opt_fp32.state)
    fp32_result = mx_to_np(model_fp32.w)

    # MLX 8-bit path
    model_8b = SingleParam(mx.array(param_np.copy()))
    opt_8b = AdamW8Bit(learning_rate=lr)
    for step_i in range(20):
        g = mx.array(grad_list[step_i])
        opt_8b.update(model_8b, {"w": g})
        mx.eval(model_8b.parameters(), opt_8b.state)
    q8_result = mx_to_np(model_8b.w)

    max_drift = float(np.max(np.abs(fp32_result - q8_result)))
    passed = max_drift < 0.05
    results.append(TestResult(
        category="Optimizer",
        name="adamw8bit_drift/20steps",
        passed=passed,
        max_diff=max_drift,
        detail=f"max_drift={max_drift:.6e} (threshold: 0.05)" if not passed else "",
    ))

    return results


# ---------------------------------------------------------------------------
# Category D: Dynamic map parity
# ---------------------------------------------------------------------------


def run_dynamic_map_tests(rng: np.random.RandomState, args: argparse.Namespace) -> list[TestResult]:
    results = []

    # D1: Properties
    dmap = create_dynamic_map(signed=True)
    checks = []
    checks.append(("len==256", len(dmap) == 256))
    checks.append(("sorted", dmap == sorted(dmap)))
    checks.append(("contains_0", 0.0 in dmap))

    # Symmetry: paired values should mirror (1.0 is an unpaired special value by design)
    positives = sorted(v for v in dmap if v > 0 and v != 1.0)
    negatives = sorted(-v for v in dmap if v < 0)
    if len(positives) == len(negatives):
        sym_ok = all(abs(p - n) < 1e-12 for p, n in zip(positives, negatives))
    else:
        sym_ok = False
    checks.append(("symmetric", sym_ok))

    all_ok = all(ok for _, ok in checks)
    detail = ", ".join(f"{name}={'OK' if ok else 'FAIL'}" for name, ok in checks)
    results.append(TestResult(
        category="DynMap",
        name="properties",
        passed=all_ok,
        detail="" if all_ok else detail,
    ))

    # D2: Unsigned map properties
    umap = create_dynamic_map(signed=False)
    u_checks = []
    u_checks.append(("len==256", len(umap) == 256))
    u_checks.append(("sorted", umap == sorted(umap)))
    u_checks.append(("all_nonneg", all(v >= 0 for v in umap)))
    all_ok = all(ok for _, ok in u_checks)
    detail = ", ".join(f"{name}={'OK' if ok else 'FAIL'}" for name, ok in u_checks)
    results.append(TestResult(
        category="DynMap",
        name="unsigned_properties",
        passed=all_ok,
        detail="" if all_ok else detail,
    ))

    # D3: bitsandbytes parity (optional)
    if HAS_BNB:
        try:
            bnb_map = bnb.functional.create_dynamic_map(signed=True)
            bnb_map_np = np.array(list(bnb_map), dtype=np.float32)
            mlx_map_np = np.array(dmap, dtype=np.float32)
            passed, max_diff, detail = compare(mlx_map_np, bnb_map_np, rtol=0, atol=1e-6)
            results.append(TestResult(
                category="DynMap",
                name="vs_bitsandbytes",
                passed=passed,
                max_diff=max_diff,
                detail=detail,
            ))
        except Exception as e:
            results.append(TestResult(
                category="DynMap", name="vs_bitsandbytes",
                passed=True, skipped=True, detail=f"bnb error: {e}",
            ))
    else:
        results.append(TestResult(
            category="DynMap", name="vs_bitsandbytes",
            passed=True, skipped=True, detail="bitsandbytes not installed",
        ))

    return results


# ---------------------------------------------------------------------------
# Category E: Log-probability computation
# ---------------------------------------------------------------------------


def run_logprob_tests(rng: np.random.RandomState, args: argparse.Namespace) -> list[TestResult]:
    results = []

    for V in [128, 1024, 32000]:
        sub_rng = np.random.RandomState(rng.randint(0, 2**31))
        logits_np, targets_np = gen_logits(sub_rng, B=2, T=8, V=V)

        # MLX path (matches training.py)
        logits_mx = np_to_mx(logits_np)
        lp_mx = logits_mx - mx.logsumexp(logits_mx, axis=-1, keepdims=True)
        target_lp_mx = mx.take_along_axis(
            lp_mx, np_to_mx(targets_np)[:, :, None].astype(mx.int32), axis=-1
        ).squeeze(-1)
        mx.eval(target_lp_mx)

        # PyTorch path
        logits_pt = np_to_pt(logits_np)
        lp_pt = torch.log_softmax(logits_pt, dim=-1)
        target_lp_pt = torch.gather(
            lp_pt, 2, torch.tensor(targets_np.astype(np.int64)).unsqueeze(-1)
        ).squeeze(-1)

        if args.profile:
            def run_mx():
                l = np_to_mx(logits_np)
                lp = l - mx.logsumexp(l, axis=-1, keepdims=True)
                out = mx.take_along_axis(lp, np_to_mx(targets_np)[:, :, None].astype(mx.int32), axis=-1)
                mx.eval(out)
                return out

            def run_pt():
                l = np_to_pt(logits_np)
                lp = torch.log_softmax(l, dim=-1)
                return torch.gather(lp, 2, torch.tensor(targets_np.astype(np.int64)).unsqueeze(-1))

            _, mlx_ms = time_fn(run_mx, n_warmup=2, n_runs=5)
            _, pt_ms = time_fn(run_pt, n_warmup=2, n_runs=5)
        else:
            mlx_ms = pt_ms = 0.0

        passed, max_diff, detail = compare(
            mx_to_np(target_lp_mx), pt_to_np(target_lp_pt), rtol=1e-5, atol=1e-5
        )
        results.append(TestResult(
            category="LogProb",
            name=f"target_logprobs/V={V}",
            passed=passed,
            max_diff=max_diff,
            mlx_ms=mlx_ms,
            pt_ms=pt_ms,
            detail=detail,
        ))

    return results


# ---------------------------------------------------------------------------
# Category F: Gradient clipping
# ---------------------------------------------------------------------------


def run_grad_clip_tests(rng: np.random.RandomState, args: argparse.Namespace) -> list[TestResult]:
    results = []

    sub_rng = np.random.RandomState(rng.randint(0, 2**31))
    grads_np = {f"layer{i}": sub_rng.randn(16, 16).astype(np.float32) for i in range(4)}

    # Compute true norm
    true_norm = float(math.sqrt(sum(np.sum(g**2) for g in grads_np.values())))

    for max_norm, label in [(1.0, "aggressive"), (1000.0, "no_op"), (true_norm * 1.01, "near_thresh")]:
        # MLX
        mlx_grads = {k: mx.array(v) for k, v in grads_np.items()}
        mlx_clipped, mlx_pre, mlx_post = _clip_grad_norm(mlx_grads, max_norm)
        mx.eval(*tree_flatten(mlx_clipped)[0:1])  # force eval
        mlx_clipped_np = {k: mx_to_np(v) for k, v in mlx_clipped.items()}

        # PyTorch reference
        pt_norm = math.sqrt(sum(np.sum(g**2) for g in grads_np.values()))
        if pt_norm > max_norm:
            scale = max_norm / (pt_norm + 1e-6)
            pt_clipped_np = {k: v * scale for k, v in grads_np.items()}
        else:
            pt_clipped_np = grads_np.copy()

        # Compare each grad tensor
        all_passed = True
        worst_diff = 0.0
        worst_detail = ""
        for key in grads_np:
            p, d, det = compare(mlx_clipped_np[key], pt_clipped_np[key], rtol=1e-5, atol=1e-5)
            if not p:
                all_passed = False
                worst_detail = det
            worst_diff = max(worst_diff, d)

        # Also compare reported norms
        norm_ok = abs(mlx_pre - pt_norm) < 1e-3
        all_passed = all_passed and norm_ok

        results.append(TestResult(
            category="GradClip",
            name=f"clip/{label}",
            passed=all_passed,
            max_diff=worst_diff,
            detail=worst_detail if not all_passed else "",
        ))

    return results


# ---------------------------------------------------------------------------
# Category G: Full training step
# ---------------------------------------------------------------------------


class PtTinyModel(torch.nn.Module):
    """PyTorch equivalent of tests/helpers.py TinyModel."""

    def __init__(self, vocab_size: int = 32, dim: int = 16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.head = torch.nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class MxTinyModel(nn.Module):
    """MLX TinyModel (same as tests/helpers.py)."""

    def __init__(self, vocab_size: int = 32, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.head(self.embed(x))


def _sync_weights_mx_to_pt(mx_model: MxTinyModel, pt_model: PtTinyModel) -> None:
    """Copy weights from MLX model to PyTorch model."""
    mx_embed = mx_to_np(mx_model.embed.weight)
    mx_head = mx_to_np(mx_model.head.weight)
    pt_model.embed.weight.data = torch.tensor(mx_embed)
    pt_model.head.weight.data = torch.tensor(mx_head)


def run_full_step_tests(rng: np.random.RandomState, args: argparse.Namespace) -> list[TestResult]:
    results = []
    vocab_size, dim = 32, 16

    # Create models with identical weights
    mx_model = MxTinyModel(vocab_size, dim)
    mx.eval(mx_model.parameters())
    pt_model = PtTinyModel(vocab_size, dim)
    _sync_weights_mx_to_pt(mx_model, pt_model)

    input_ids = [5, 10, 15, 20, 25]
    targets = [10, 15, 20, 25, 1]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    # G1: Logit parity
    mx_logits = mx_model(mx.array([input_ids]))
    mx.eval(mx_logits)
    pt_logits = pt_model(torch.tensor([input_ids]))

    passed, max_diff, detail = compare(mx_to_np(mx_logits), pt_to_np(pt_logits), rtol=1e-5, atol=1e-5)
    results.append(TestResult(
        category="FullStep", name="logit_parity", passed=passed, max_diff=max_diff, detail=detail,
    ))

    # G2: Loss parity
    mx_lp = mx_logits - mx.logsumexp(mx_logits, axis=-1, keepdims=True)
    mx_target_lp = mx.take_along_axis(mx_lp, mx.array([targets])[:, :, None], axis=-1).squeeze(-1)
    mx_loss = (-mx_target_lp * mx.array([weights], dtype=mx.float32)).sum()
    mx.eval(mx_loss)

    pt_lp = torch.log_softmax(pt_logits, dim=-1)
    pt_target_lp = torch.gather(pt_lp, 2, torch.tensor([targets]).unsqueeze(-1)).squeeze(-1)
    pt_loss = (-pt_target_lp * torch.tensor([weights])).sum()

    passed, max_diff, detail = compare(mx_to_np(mx_loss), pt_to_np(pt_loss), rtol=1e-5, atol=1e-5)
    results.append(TestResult(
        category="FullStep", name="loss_parity", passed=passed, max_diff=max_diff, detail=detail,
    ))

    # G3: Gradient parity (via autograd on both sides)
    def mx_loss_fn(model, input_ids_mx, targets_mx, weights_mx):
        logits = model(input_ids_mx)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        target_lp = mx.take_along_axis(lp, targets_mx[:, :, None], axis=-1).squeeze(-1)
        return (-target_lp * weights_mx).sum()

    loss_and_grad = nn.value_and_grad(mx_model, mx_loss_fn)
    mx_loss_val, mx_grads = loss_and_grad(
        mx_model,
        mx.array([input_ids]),
        mx.array([targets]),
        mx.array([weights], dtype=mx.float32),
    )
    mx.eval(mx_loss_val)

    # PyTorch backward
    pt_model2 = PtTinyModel(vocab_size, dim)
    _sync_weights_mx_to_pt(mx_model, pt_model2)  # use same pre-update weights
    # Need fresh forward pass for PyTorch grad
    pt_logits2 = pt_model2(torch.tensor([input_ids]))
    pt_lp2 = torch.log_softmax(pt_logits2, dim=-1)
    pt_target_lp2 = torch.gather(pt_lp2, 2, torch.tensor([targets]).unsqueeze(-1)).squeeze(-1)
    pt_loss2 = (-pt_target_lp2 * torch.tensor([weights])).sum()
    pt_loss2.backward()

    # Compare embed gradients
    mx_embed_grad = mx_to_np(mx_grads["embed"]["weight"])
    pt_embed_grad = pt_to_np(pt_model2.embed.weight.grad)
    passed_e, diff_e, det_e = compare(mx_embed_grad, pt_embed_grad, rtol=1e-3, atol=1e-5)

    # Compare head gradients
    mx_head_grad = mx_to_np(mx_grads["head"]["weight"])
    pt_head_grad = pt_to_np(pt_model2.head.weight.grad)
    passed_h, diff_h, det_h = compare(mx_head_grad, pt_head_grad, rtol=1e-3, atol=1e-5)

    passed = passed_e and passed_h
    max_diff = max(diff_e, diff_h)
    detail = det_e + "\n" + det_h if not passed else ""
    results.append(TestResult(
        category="FullStep", name="gradient_parity", passed=passed, max_diff=max_diff, detail=detail,
    ))

    # G4: Post-optimizer-step param parity (fp32 AdamW)
    lr = 1e-3

    # MLX: apply grads via fp32 AdamW
    mx_opt = optim.AdamW(learning_rate=lr)
    mx_opt.update(mx_model, mx_grads)
    mx.eval(mx_model.parameters(), mx_opt.state)

    # PyTorch: manual AdamW (no bias correction)
    pt_embed_w = pt_model2.embed.weight.data.clone()
    pt_head_w = pt_model2.head.weight.data.clone()
    pt_opt = PtAdamWNoBiasCorrection(
        [pt_embed_w, pt_head_w], lr=lr, betas=(0.9, 0.999), eps=1e-8,
    )
    pt_opt.step([
        (pt_embed_w, pt_model2.embed.weight.grad),
        (pt_head_w, pt_model2.head.weight.grad),
    ])

    passed_e, diff_e, det_e = compare(mx_to_np(mx_model.embed.weight), pt_to_np(pt_embed_w), rtol=1e-4, atol=1e-5)
    passed_h, diff_h, det_h = compare(mx_to_np(mx_model.head.weight), pt_to_np(pt_head_w), rtol=1e-4, atol=1e-5)

    passed = passed_e and passed_h
    max_diff = max(diff_e, diff_h)
    detail = det_e + "\n" + det_h if not passed else ""
    results.append(TestResult(
        category="FullStep", name="post_optim_parity", passed=passed, max_diff=max_diff, detail=detail,
    ))

    return results


# ---------------------------------------------------------------------------
# Category H: Real model inference (Qwen3.5-0.8B vs HF Transformers)
# ---------------------------------------------------------------------------

# Fixed prompts covering diverse token distributions
FIXED_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "In quantum mechanics, the uncertainty principle states that",
    "SELECT * FROM users WHERE",
    "Once upon a time, in a land far far away, there lived a",
]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))


def _kl_div(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(P||Q) from logits, averaged over positions."""
    p_logits = p_logits.astype(np.float64)
    q_logits = q_logits.astype(np.float64)
    p = np.exp(p_logits - np.max(p_logits, axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    q = np.exp(q_logits - np.max(q_logits, axis=-1, keepdims=True))
    q = q / q.sum(axis=-1, keepdims=True)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q)), axis=-1).mean())


def _top_k_agreement(a_logits: np.ndarray, b_logits: np.ndarray, k: int = 5) -> float:
    """Fraction of positions where top-k token sets agree."""
    n_pos = a_logits.shape[0]
    agree = 0
    for i in range(n_pos):
        a_top = set(np.argsort(a_logits[i])[-k:])
        b_top = set(np.argsort(b_logits[i])[-k:])
        if a_top == b_top:
            agree += 1
    return agree / n_pos


def run_real_model_tests(rng: np.random.RandomState, args: argparse.Namespace) -> list[TestResult]:
    results = []
    model_name = args.model

    # Try importing both frameworks
    try:
        from mlx_lm import load as mlx_load
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        results.append(TestResult(
            category="RealModel", name="import_check",
            passed=True, skipped=True, detail=f"Missing dependency: {e}",
        ))
        return results

    print(f"\n  Loading models ({model_name})...", flush=True)

    # Load tokenizer (shared)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load MLX model (fp16)
    mlx_model, _ = mlx_load(model_name)

    # Load HF model (bf16 on CPU to match MLX default)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
    )
    hf_model.eval()

    print("  Models loaded. Running tests...", flush=True)

    # H1: FP16 logit parity across fixed prompts
    all_cosines = []
    all_max_abs = []
    all_kl = []
    all_top5 = []
    all_logprob_rmse = []

    for prompt in FIXED_PROMPTS:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) < 2:
            continue

        # HF forward
        with torch.no_grad():
            hf_out = hf_model(torch.tensor([tokens]))
            hf_logits = hf_out.logits[0].float().numpy()  # [T, V]

        # MLX forward
        mlx_logits = mlx_model(mx.array([tokens]))
        mx.eval(mlx_logits)
        mlx_logits_np = mx_to_np(mlx_logits[0])  # [T, V]

        min_len = min(hf_logits.shape[0], mlx_logits_np.shape[0])
        hf_l = hf_logits[:min_len]
        mlx_l = mlx_logits_np[:min_len]

        # Per-position cosine similarity
        for i in range(min_len):
            all_cosines.append(_cosine_sim(hf_l[i], mlx_l[i]))

        all_max_abs.append(float(np.max(np.abs(hf_l.astype(np.float64) - mlx_l.astype(np.float64)))))
        all_kl.append(_kl_div(hf_l, mlx_l))
        all_top5.append(_top_k_agreement(hf_l, mlx_l, k=5))

        # Per-token logprob RMSE (next-token prediction)
        hf_lp = hf_l - np.max(hf_l, axis=-1, keepdims=True)
        hf_lp = hf_lp - np.log(np.sum(np.exp(hf_lp), axis=-1, keepdims=True))
        mlx_lp = mlx_l.astype(np.float64) - np.max(mlx_l.astype(np.float64), axis=-1, keepdims=True)
        mlx_lp = mlx_lp - np.log(np.sum(np.exp(mlx_lp), axis=-1, keepdims=True))

        target_ids = tokens[1:min_len + 1]
        if len(target_ids) >= min_len:
            target_ids = target_ids[:min_len]
            hf_target_lp = np.array([hf_lp[i, target_ids[i]] for i in range(len(target_ids))])
            mlx_target_lp = np.array([mlx_lp[i, target_ids[i]] for i in range(len(target_ids))])
            all_logprob_rmse.append(float(np.sqrt(np.mean((hf_target_lp - mlx_target_lp) ** 2))))

    mean_cosine = float(np.mean(all_cosines))
    mean_kl = float(np.mean(all_kl))
    mean_top5 = float(np.mean(all_top5))
    mean_rmse = float(np.mean(all_logprob_rmse)) if all_logprob_rmse else 0.0
    max_abs = float(np.max(all_max_abs))

    # H1a: Cosine similarity
    passed = mean_cosine > 0.999
    results.append(TestResult(
        category="RealModel", name=f"bf16_cosine_sim",
        passed=passed, max_diff=1.0 - mean_cosine,
        detail=f"mean={mean_cosine:.6f} min={min(all_cosines):.6f}" if not passed else "",
        extra={"mean_cosine": mean_cosine, "min_cosine": min(all_cosines)},
    ))

    # H1b: Max absolute diff (bf16 precision limits max_abs to ~2.0)
    passed = max_abs < 2.0
    results.append(TestResult(
        category="RealModel", name=f"bf16_max_abs_diff",
        passed=passed, max_diff=max_abs,
        detail=f"max_abs={max_abs:.4f}" if not passed else "",
    ))

    # H1c: KL divergence
    passed = mean_kl < 0.005
    results.append(TestResult(
        category="RealModel", name=f"bf16_kl_div",
        passed=passed, max_diff=mean_kl,
        detail=f"mean_kl={mean_kl:.6f}" if not passed else "",
    ))

    # H1d: Top-5 agreement (bf16 cross-framework: lower than fp32 due to precision)
    passed = mean_top5 > 0.70
    results.append(TestResult(
        category="RealModel", name=f"bf16_top5_agree",
        passed=passed, max_diff=1.0 - mean_top5,
        detail=f"mean_top5={mean_top5:.4f}" if not passed else "",
    ))

    # H1e: Logprob RMSE
    passed = mean_rmse < 0.05
    results.append(TestResult(
        category="RealModel", name=f"bf16_logprob_rmse",
        passed=passed, max_diff=mean_rmse,
        detail=f"mean_rmse={mean_rmse:.6f}" if not passed else "",
    ))

    # H2: Per-prompt determinism (same prompt, same logits every time)
    test_prompt = FIXED_PROMPTS[0]
    test_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    logits_runs = []
    for _ in range(3):
        out = mlx_model(mx.array([test_tokens]))
        mx.eval(out)
        logits_runs.append(mx_to_np(out[0]))

    # All runs should be bit-identical
    all_identical = all(np.array_equal(logits_runs[0], logits_runs[i]) for i in range(1, len(logits_runs)))
    results.append(TestResult(
        category="RealModel", name="determinism",
        passed=all_identical, max_diff=0.0,
        detail="" if all_identical else "Logits differ across runs with same input",
    ))

    # H3: Layer-level hidden state comparison (first + last layer)
    # Extract hidden states from both frameworks
    test_tokens = tokenizer.encode(FIXED_PROMPTS[0], add_special_tokens=False)
    if len(test_tokens) > 32:
        test_tokens = test_tokens[:32]

    # HF with output_hidden_states
    with torch.no_grad():
        hf_out = hf_model(torch.tensor([test_tokens]), output_hidden_states=True)
        hf_hidden = [h[0].float().numpy() for h in hf_out.hidden_states]  # list of [T, D]

    # MLX: manually extract intermediate hidden states
    # Access model.model (the backbone) for intermediate layers
    mlx_backbone = mlx_model.model if hasattr(mlx_model, "model") else mlx_model
    mlx_input = mx.array([test_tokens])

    # Embedding output — navigate nested model structure
    mlx_embed = None
    # Try common paths: model.embed_tokens, language_model.model.embed_tokens, etc.
    embed_paths = [
        lambda m: m.embed_tokens,
        lambda m: m.model.embed_tokens,
        lambda m: m.language_model.model.embed_tokens,
        lambda m: m.wte,
        lambda m: m.embed,
    ]
    for path_fn in embed_paths:
        try:
            embed_layer = path_fn(mlx_model)
            mlx_embed = embed_layer(mlx_input)
            break
        except (AttributeError, TypeError):
            continue

    if mlx_embed is not None:
        mx.eval(mlx_embed)
        mlx_embed_np = mx_to_np(mlx_embed[0])  # [T, D]
        hf_embed_np = hf_hidden[0]  # [T, D] — already numpy from list comprehension above

        min_len = min(mlx_embed_np.shape[0], hf_embed_np.shape[0])
        cos = _cosine_sim(mlx_embed_np[:min_len], hf_embed_np[:min_len])
        passed = cos > 0.999
        results.append(TestResult(
            category="RealModel", name="layer_embed/cosine",
            passed=passed, max_diff=1.0 - cos,
            detail=f"cosine={cos:.6f}" if not passed else "",
        ))
    else:
        results.append(TestResult(
            category="RealModel", name="layer_embed/cosine",
            passed=True, skipped=True, detail="Cannot access embedding layer",
        ))

    # Final logit comparison (already tested above, but good to have per-prompt)
    mlx_final = mlx_model(mx.array([test_tokens]))
    mx.eval(mlx_final)
    with torch.no_grad():
        hf_final = hf_model(torch.tensor([test_tokens])).logits[0].float().numpy()
    mlx_final_np = mx_to_np(mlx_final[0])
    min_len = min(mlx_final_np.shape[0], hf_final.shape[0])

    # Top-1 token agreement per position
    mlx_top1 = np.argmax(mlx_final_np[:min_len], axis=-1)
    hf_top1 = np.argmax(hf_final[:min_len], axis=-1)
    top1_agree = float(np.mean(mlx_top1 == hf_top1))
    passed = top1_agree > 0.90
    results.append(TestResult(
        category="RealModel", name="top1_token_agree",
        passed=passed, max_diff=1.0 - top1_agree,
        detail=f"top1_agree={top1_agree:.4f} ({int(top1_agree*min_len)}/{min_len})" if not passed else "",
        extra={"top1_agreement": top1_agree},
    ))

    # H4: 4-bit quantized — both sides quantized for fair comparison
    import mlx.nn as mlx_nn
    mlx_q_model, _ = mlx_load(model_name)
    mlx_nn.quantize(mlx_q_model, bits=4, group_size=64)

    # Try to load HF with BitsAndBytes 4-bit for fair comparison
    hf_4bit_model = None
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        hf_4bit_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="cpu", trust_remote_code=True,
        )
        hf_4bit_model.eval()
        hf_4bit_label = "4bit_bnb"
    except Exception:
        # Fall back to bf16 HF as reference (documents the quantization gap)
        hf_4bit_model = hf_model
        hf_4bit_label = "4bit_vs_bf16"

    q_cosines = []
    q_top5 = []
    for prompt in FIXED_PROMPTS[:3]:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) < 2:
            continue

        with torch.no_grad():
            hf_out = hf_4bit_model(torch.tensor([tokens]))
            hf_l = hf_out.logits[0].float().numpy()

        mlx_out = mlx_q_model(mx.array([tokens]))
        mx.eval(mlx_out)
        mlx_l = mx_to_np(mlx_out[0])

        min_len = min(hf_l.shape[0], mlx_l.shape[0])
        min_vocab = min(hf_l.shape[1], mlx_l.shape[1])
        for i in range(min_len):
            q_cosines.append(_cosine_sim(hf_l[i, :min_vocab], mlx_l[i, :min_vocab]))
        q_top5.append(_top_k_agreement(hf_l[:min_len, :min_vocab], mlx_l[:min_len, :min_vocab], k=5))

    mean_q_cos = float(np.mean(q_cosines))
    mean_q_top5 = float(np.mean(q_top5))

    # 4-bit comparison: fundamentally different quantization schemes
    # MLX uses groupwise quantization, BnB uses NF4 — expect significant divergence.
    # These thresholds catch catastrophic failures, not scheme equivalence.
    passed = mean_q_cos > 0.90
    results.append(TestResult(
        category="RealModel", name=f"{hf_4bit_label}_cosine",
        passed=passed, max_diff=1.0 - mean_q_cos,
        detail=f"mean={mean_q_cos:.6f} (MLX groupwise vs {'BnB NF4' if 'bnb' in hf_4bit_label else 'bf16'})",
        extra={"mean_cosine": mean_q_cos, "scheme": hf_4bit_label},
    ))

    # Top-5 diverges heavily across quantization schemes; this is a regression guard
    passed = mean_q_top5 > 0.05
    results.append(TestResult(
        category="RealModel", name=f"{hf_4bit_label}_top5",
        passed=passed, max_diff=1.0 - mean_q_top5,
        detail=f"mean={mean_q_top5:.4f}" if not passed else "",
        extra={"mean_top5": mean_q_top5},
    ))

    # Clean up to free memory
    del mlx_model, mlx_q_model, hf_model
    import gc
    gc.collect()

    return results


# ---------------------------------------------------------------------------
# Category I: Tinker API parity (real API vs local mlx-tinker backend)
# ---------------------------------------------------------------------------


def run_tinker_api_tests(rng: np.random.RandomState, args: argparse.Namespace) -> list[TestResult]:
    """Compare mlx-tinker local backend against real Tinker API.

    Requires TINKER_API_KEY in .env or environment. Runs SFT and RL
    training on WikiSQL data via both paths and compares loss curves.
    """
    import os
    import tempfile

    results = []

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        results.append(TestResult(
            category="TinkerAPI", name="api_key_check",
            passed=True, skipped=True, detail="TINKER_API_KEY not set",
        ))
        return results

    try:
        import tinker
        from transformers import AutoTokenizer
    except ImportError as e:
        results.append(TestResult(
            category="TinkerAPI", name="import_check",
            passed=True, skipped=True, detail=f"Missing dependency: {e}",
        ))
        return results

    from pathlib import Path

    from mlx_tinker.backend.mlx_backend import MLXBackend
    from mlx_tinker.config import EngineConfig
    from mlx_tinker.types import (
        AdamParams,
        CreateModelInput,
        Datum,
        EncodedTextChunk,
        ForwardBackwardInput,
        ForwardInput,
        LoraConfig,
        LossFnInputs,
        ModelInput,
        OptimStepInput,
        TensorData,
    )

    # Load WikiSQL fixture
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    wikisql_path = fixtures_dir / "wikisql_subset.json"
    if not wikisql_path.exists():
        results.append(TestResult(
            category="TinkerAPI", name="fixture_check",
            passed=False, detail=f"WikiSQL fixture not found: {wikisql_path}",
        ))
        return results

    import json as json_mod
    with open(wikisql_path) as f:
        wikisql_examples = json_mod.load(f)

    MODEL_NAME = "Qwen/Qwen3.5-4B"
    LORA_RANK = 8
    SFT_STEPS = 10
    RL_STEPS = 5
    LR = 1e-4

    def format_example(ex):
        cols = " | ".join(ex["columns"])
        return f"Table: {cols}\nQuestion: {ex['question']}\nSQL: ", ex["sql"]

    print(f"\n  Tinker API parity: {MODEL_NAME}, {SFT_STEPS} SFT + {RL_STEPS} RL steps", flush=True)

    # ---- Tinker API side ----
    import asyncio

    async def run_tinker_sft():
        service_client = tinker.ServiceClient()
        training_client = await service_client.create_lora_training_client_async(
            base_model=MODEL_NAME, rank=LORA_RANK,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        losses = []
        logprobs_out = []

        for step in range(SFT_STEPS):
            ex = wikisql_examples[step % len(wikisql_examples)]
            prompt, sql = format_example(ex)
            tokens = tokenizer.encode(prompt + sql)[:256]
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]

            datum = tinker.Datum(
                model_input=tinker.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "weights": [1.0] * len(target_tokens),
                },
            )
            fb = await training_client.forward_backward_async([datum], loss_fn="cross_entropy")
            if hasattr(fb, "result_async"):
                fb = await fb.result_async()
            else:
                fb = fb.result()

            loss_sum = fb.metrics.get("loss:sum", fb.metrics.get("mean_loss", 0.0))
            losses.append(loss_sum)

            # Capture logprobs if available
            if hasattr(fb, "loss_fn_outputs") and fb.loss_fn_outputs:
                lp = fb.loss_fn_outputs[0].get("logprobs", {})
                if hasattr(lp, "data"):
                    logprobs_out.append(lp.data[:5])
                elif isinstance(lp, dict) and "data" in lp:
                    logprobs_out.append(lp["data"][:5])

            adam = tinker.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.999, eps=1e-8)
            opt = await training_client.optim_step_async(adam)
            if hasattr(opt, "result_async"):
                await opt.result_async()
            else:
                opt.result()

        return losses, logprobs_out, tokenizer

    async def run_tinker_rl(tokenizer):
        service_client = tinker.ServiceClient()
        training_client = await service_client.create_lora_training_client_async(
            base_model=MODEL_NAME, rank=LORA_RANK,
        )
        losses = []
        for step in range(RL_STEPS):
            ex = wikisql_examples[step % len(wikisql_examples)]
            prompt, sql = format_example(ex)
            tokens = tokenizer.encode(prompt + sql)[:256]
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]

            prompt_tokens = tokenizer.encode(prompt)
            n_prompt = len(prompt_tokens) - 1
            n_tgt = len(target_tokens)

            advantages = [0.0] * min(n_prompt, n_tgt) + [1.0] * max(0, n_tgt - n_prompt)
            advantages = advantages[:n_tgt]
            logprobs = [0.0] * n_tgt

            datum = tinker.Datum(
                model_input=tinker.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "logprobs": logprobs,
                    "advantages": advantages,
                },
            )
            try:
                fb = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
                if hasattr(fb, "result_async"):
                    fb = await fb.result_async()
                else:
                    fb = fb.result()
                loss_sum = fb.metrics.get("loss:sum", fb.metrics.get("mean_loss", 0.0))
                losses.append(loss_sum)

                adam = tinker.AdamParams(learning_rate=5e-5, beta1=0.9, beta2=0.999, eps=1e-8)
                opt = await training_client.optim_step_async(adam)
                if hasattr(opt, "result_async"):
                    await opt.result_async()
                else:
                    opt.result()
            except Exception as e:
                print(f"    RL step {step} failed: {e}")
                break
        return losses

    print("  Running Tinker API SFT...", flush=True)
    try:
        tinker_sft_losses, tinker_logprobs, tokenizer = asyncio.run(run_tinker_sft())
    except Exception as e:
        results.append(TestResult(
            category="TinkerAPI", name="sft_api_call",
            passed=False, detail=f"Tinker API SFT failed: {e}",
        ))
        return results

    print(f"    Tinker SFT losses: {[f'{l:.2f}' for l in tinker_sft_losses[:5]]}...", flush=True)

    # ---- MLX local side (same model, same data) ----
    print("  Running local mlx-tinker SFT...", flush=True)
    config = EngineConfig(
        base_model=MODEL_NAME, quantize_bits=4, quantize_group_size=64,
        checkpoints_base=Path(tempfile.mkdtemp()),
    )
    backend = MLXBackend(config)
    lora_cfg = LoraConfig(rank=LORA_RANK, alpha=16.0, train_attn=True, train_mlp=True)
    backend.create_model("audit", CreateModelInput(lora_config=lora_cfg))
    local_tokenizer = backend.tokenizers["audit"]

    local_sft_losses = []
    local_logprobs = []
    for step in range(SFT_STEPS):
        ex = wikisql_examples[step % len(wikisql_examples)]
        prompt, sql = format_example(ex)
        tokens = local_tokenizer.encode(prompt + sql)[:256]
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        datum = Datum(
            model_input=ModelInput(chunks=[EncodedTextChunk(tokens=input_tokens)]),
            loss_fn_inputs=LossFnInputs(
                target_tokens=TensorData(data=target_tokens),
                weights=TensorData(data=[1.0] * len(target_tokens)),
                advantages=TensorData(data=[0.0] * len(target_tokens)),
                logprobs=TensorData(data=[0.0] * len(target_tokens)),
            ),
        )
        fb = backend.forward_backward("audit", ForwardBackwardInput(data=[datum], loss_fn="cross_entropy"))
        local_sft_losses.append(fb.metrics["loss:sum"])

        if fb.loss_fn_outputs and "logprobs" in fb.loss_fn_outputs[0]:
            lp_data = fb.loss_fn_outputs[0]["logprobs"]
            if hasattr(lp_data, "data"):
                local_logprobs.append(lp_data.data[:5])

        backend.optim_step("audit", OptimStepInput(adam_params=AdamParams(learning_rate=LR)))

    print(f"    Local SFT losses:  {[f'{l:.2f}' for l in local_sft_losses[:5]]}...", flush=True)

    # I1: SFT loss curve correlation
    tinker_arr = np.array(tinker_sft_losses[:SFT_STEPS])
    local_arr = np.array(local_sft_losses[:SFT_STEPS])
    min_len = min(len(tinker_arr), len(local_arr))
    if min_len >= 2:
        corr = float(np.corrcoef(tinker_arr[:min_len], local_arr[:min_len])[0, 1])
        passed = corr > 0.85 or np.isnan(corr) is False  # NaN if constant
        if np.isnan(corr):
            corr = 0.0
            passed = False
        passed = corr > 0.85
    else:
        corr = 0.0
        passed = False
    results.append(TestResult(
        category="TinkerAPI", name="sft_loss_correlation",
        passed=passed, max_diff=1.0 - corr,
        detail=f"correlation={corr:.4f}" if not passed else "",
        extra={"correlation": corr, "tinker_losses": tinker_sft_losses, "local_losses": local_sft_losses},
    ))

    # I2: Initial loss relative difference
    if tinker_sft_losses and local_sft_losses:
        rel_diff = abs(tinker_sft_losses[0] - local_sft_losses[0]) / (abs(tinker_sft_losses[0]) + 1e-8)
        passed = rel_diff < 0.25
        results.append(TestResult(
            category="TinkerAPI", name="sft_initial_loss",
            passed=passed, max_diff=rel_diff,
            detail=f"tinker={tinker_sft_losses[0]:.4f} local={local_sft_losses[0]:.4f} rel={rel_diff:.4f}" if not passed else "",
        ))

    # I3: Per-step relative difference
    if min_len >= 2:
        rel_diffs = np.abs(tinker_arr[:min_len] - local_arr[:min_len]) / (np.abs(tinker_arr[:min_len]) + 1e-8)
        max_rel = float(rel_diffs.max())
        mean_rel = float(rel_diffs.mean())
        passed = max_rel < 0.50  # 50% tolerance (different hardware, quantization)
        results.append(TestResult(
            category="TinkerAPI", name="sft_per_step_diff",
            passed=passed, max_diff=max_rel,
            detail=f"max_rel={max_rel:.4f} mean_rel={mean_rel:.4f}" if not passed else "",
        ))

    # I4: Both loss curves trend downward (SFT should converge)
    if len(tinker_sft_losses) >= 4 and len(local_sft_losses) >= 4:
        tinker_down = np.mean(tinker_sft_losses[-3:]) < np.mean(tinker_sft_losses[:3])
        local_down = np.mean(local_sft_losses[-3:]) < np.mean(local_sft_losses[:3])
        passed = tinker_down == local_down
        results.append(TestResult(
            category="TinkerAPI", name="sft_trend_agreement",
            passed=passed,
            detail=f"tinker_down={tinker_down} local_down={local_down}" if not passed else "",
        ))

    # I5: Logprob sign agreement (if captured)
    # Logprobs should be <= 0; allow 0.0 exactly (first token / padding)
    if tinker_logprobs and local_logprobs:
        min_lp = min(len(tinker_logprobs), len(local_logprobs))
        all_leq0_tinker = all(v <= 0.0 for lp in tinker_logprobs[:min_lp] for v in lp if lp)
        all_leq0_local = all(v <= 0.0 for lp in local_logprobs[:min_lp] for v in lp if lp)
        passed = all_leq0_tinker and all_leq0_local
        results.append(TestResult(
            category="TinkerAPI", name="logprobs_leq_zero",
            passed=passed,
            detail=f"tinker_ok={all_leq0_tinker} local_ok={all_leq0_local}" if not passed else "",
        ))

    # I-batch: Multi-datum batch linearity (same datum N times should give N * single loss)
    print("  Testing batch linearity...", flush=True)
    ex0 = wikisql_examples[0]
    prompt0, sql0 = format_example(ex0)
    tok0 = local_tokenizer.encode(prompt0 + sql0)[:256]
    inp0 = tok0[:-1]
    tgt0 = tok0[1:]
    n0 = len(tgt0)

    single_datum = Datum(
        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=inp0)]),
        loss_fn_inputs=LossFnInputs(
            target_tokens=TensorData(data=tgt0),
            weights=TensorData(data=[1.0] * n0),
            advantages=TensorData(data=[0.0] * n0),
            logprobs=TensorData(data=[0.0] * n0),
        ),
    )

    # Need a fresh model for batch tests (original already trained)
    backend_batch = MLXBackend(EngineConfig(
        base_model=MODEL_NAME, quantize_bits=4, quantize_group_size=64,
        checkpoints_base=Path(tempfile.mkdtemp()),
    ))
    backend_batch.create_model("batch", CreateModelInput(lora_config=lora_cfg))

    for batch_size in [1, 2, 4, 8]:
        fb = backend_batch.forward_backward(
            "batch",
            ForwardBackwardInput(data=[single_datum] * batch_size, loss_fn="cross_entropy"),
        )
        batch_loss = fb.metrics["loss:sum"]
        # Clear grads
        backend_batch.training.accumulated_grads["batch"] = None
        backend_batch.training.grad_accum_counts["batch"] = 0
        backend_batch.training.total_tokens["batch"] = 0.0

        if batch_size == 1:
            single_loss = batch_loss
        else:
            expected = single_loss * batch_size
            ratio = batch_loss / expected if expected != 0 else float("inf")
            passed = abs(ratio - 1.0) < 1e-5
            results.append(TestResult(
                category="TinkerAPI", name=f"batch_linearity/bs={batch_size}",
                passed=passed, max_diff=abs(ratio - 1.0),
                detail=f"got={batch_loss:.4f} expected={expected:.4f} ratio={ratio:.8f}" if not passed else "",
            ))

    # I-mixed: Mixed datum batch (different examples in one forward_backward)
    mixed_data = []
    for i in range(4):
        ex_i = wikisql_examples[i]
        pr_i, sql_i = format_example(ex_i)
        tok_i = local_tokenizer.encode(pr_i + sql_i)[:256]
        inp_i = tok_i[:-1]
        tgt_i = tok_i[1:]
        n_i = len(tgt_i)
        mixed_data.append(Datum(
            model_input=ModelInput(chunks=[EncodedTextChunk(tokens=inp_i)]),
            loss_fn_inputs=LossFnInputs(
                target_tokens=TensorData(data=tgt_i),
                weights=TensorData(data=[1.0] * n_i),
                advantages=TensorData(data=[0.0] * n_i),
                logprobs=TensorData(data=[0.0] * n_i),
            ),
        ))

    # Run all 4 individually
    individual_losses = []
    for d in mixed_data:
        fb = backend_batch.forward_backward(
            "batch", ForwardBackwardInput(data=[d], loss_fn="cross_entropy"),
        )
        individual_losses.append(fb.metrics["loss:sum"])
        backend_batch.training.accumulated_grads["batch"] = None
        backend_batch.training.grad_accum_counts["batch"] = 0
        backend_batch.training.total_tokens["batch"] = 0.0

    # Run all 4 batched
    fb_batched = backend_batch.forward_backward(
        "batch", ForwardBackwardInput(data=mixed_data, loss_fn="cross_entropy"),
    )
    batched_loss = fb_batched.metrics["loss:sum"]
    backend_batch.training.accumulated_grads["batch"] = None
    backend_batch.training.grad_accum_counts["batch"] = 0
    backend_batch.training.total_tokens["batch"] = 0.0

    expected_sum = sum(individual_losses)
    ratio = batched_loss / expected_sum if expected_sum != 0 else float("inf")
    passed = abs(ratio - 1.0) < 1e-5
    results.append(TestResult(
        category="TinkerAPI", name="batch_mixed_4_datums",
        passed=passed, max_diff=abs(ratio - 1.0),
        detail=(f"batched={batched_loss:.4f} sum_individual={expected_sum:.4f} "
                f"ratio={ratio:.8f}") if not passed else "",
    ))

    # I-accum: Gradient accumulation equivalence (same model, two paths)
    # Compare accumulated gradients from:
    #   Path A: 4 separate forward_backward calls
    #   Path B: 1 forward_backward call with 4 datums
    # Uses a single model to avoid quantization/init non-determinism.
    backend_accum = MLXBackend(EngineConfig(
        base_model=MODEL_NAME, quantize_bits=4, quantize_group_size=64,
        checkpoints_base=Path(tempfile.mkdtemp()),
    ))
    backend_accum.create_model("accum", CreateModelInput(lora_config=lora_cfg))

    from mlx.utils import tree_flatten as tf

    # Path A: Sequential (4 separate forward_backward calls)
    for d in mixed_data:
        backend_accum.forward_backward("accum", ForwardBackwardInput(data=[d], loss_fn="cross_entropy"))

    seq_grads = {n: mx_to_np(p) for n, p in tf(backend_accum.training.accumulated_grads["accum"])}
    seq_total_tokens = backend_accum.training.total_tokens["accum"]

    # Reset gradient state
    backend_accum.training.accumulated_grads["accum"] = None
    backend_accum.training.grad_accum_counts["accum"] = 0
    backend_accum.training.total_tokens["accum"] = 0.0

    # Path B: Batched (1 forward_backward with 4 datums)
    backend_accum.forward_backward("accum", ForwardBackwardInput(data=mixed_data, loss_fn="cross_entropy"))

    bat_grads = {n: mx_to_np(p) for n, p in tf(backend_accum.training.accumulated_grads["accum"])}
    bat_total_tokens = backend_accum.training.total_tokens["accum"]

    # Compare gradients
    max_grad_diff = 0.0
    max_grad_rel = 0.0
    for name in seq_grads:
        if name in bat_grads:
            s = seq_grads[name].flatten()
            b = bat_grads[name].flatten()
            abs_d = np.max(np.abs(s - b))
            denom = max(np.max(np.abs(s)), np.max(np.abs(b)), 1e-10)
            rel_d = abs_d / denom
            max_grad_diff = max(max_grad_diff, abs_d)
            max_grad_rel = max(max_grad_rel, rel_d)

    tokens_match = seq_total_tokens == bat_total_tokens
    # Float32 addition order may differ → allow small tolerance
    passed = max_grad_rel < 1e-5 and tokens_match
    results.append(TestResult(
        category="TinkerAPI", name="grad_accum_equivalence",
        passed=passed, max_diff=max_grad_diff,
        detail=(f"max_abs={max_grad_diff:.4e} max_rel={max_grad_rel:.4e} "
                f"tokens_match={tokens_match} "
                f"(seq={seq_total_tokens:.0f} bat={bat_total_tokens:.0f})"
                ) if not passed else "",
    ))

    # Reset
    backend_accum.training.accumulated_grads["accum"] = None
    backend_accum.training.grad_accum_counts["accum"] = 0
    backend_accum.training.total_tokens["accum"] = 0.0

    del backend_batch, backend_accum

    # ---- RL comparison ----
    print("  Running Tinker API RL...", flush=True)
    try:
        tinker_rl_losses = asyncio.run(run_tinker_rl(tokenizer))
    except Exception as e:
        results.append(TestResult(
            category="TinkerAPI", name="rl_api_call",
            passed=True, skipped=True, detail=f"Tinker RL failed: {e}",
        ))
        tinker_rl_losses = []

    if tinker_rl_losses:
        print(f"    Tinker RL losses: {[f'{l:.2f}' for l in tinker_rl_losses]}", flush=True)

        # Local RL
        print("  Running local mlx-tinker RL...", flush=True)
        backend2 = MLXBackend(EngineConfig(
            base_model=MODEL_NAME, quantize_bits=4, quantize_group_size=64,
            checkpoints_base=Path(tempfile.mkdtemp()),
        ))
        backend2.create_model("rl-audit", CreateModelInput(lora_config=lora_cfg))
        local_tokenizer2 = backend2.tokenizers["rl-audit"]

        local_rl_losses = []
        for step in range(RL_STEPS):
            ex = wikisql_examples[step % len(wikisql_examples)]
            prompt, sql = format_example(ex)
            tokens = local_tokenizer2.encode(prompt + sql)[:256]
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]

            prompt_tokens = local_tokenizer2.encode(prompt)
            n_prompt = len(prompt_tokens) - 1
            n_tgt = len(target_tokens)

            advantages = [0.0] * min(n_prompt, n_tgt) + [1.0] * max(0, n_tgt - n_prompt)
            advantages = advantages[:n_tgt]

            datum = Datum(
                model_input=ModelInput(chunks=[EncodedTextChunk(tokens=input_tokens)]),
                loss_fn_inputs=LossFnInputs(
                    target_tokens=TensorData(data=target_tokens),
                    weights=TensorData(data=[1.0] * n_tgt),
                    advantages=TensorData(data=advantages),
                    logprobs=TensorData(data=[0.0] * n_tgt),
                ),
            )
            fb = backend2.forward_backward("rl-audit", ForwardBackwardInput(data=[datum], loss_fn="importance_sampling"))
            local_rl_losses.append(fb.metrics["loss:sum"])
            backend2.optim_step("rl-audit", OptimStepInput(adam_params=AdamParams(learning_rate=5e-5)))

        print(f"    Local RL losses:  {[f'{l:.2f}' for l in local_rl_losses]}", flush=True)

        # I6: RL trend direction agreement
        if len(tinker_rl_losses) >= 2 and len(local_rl_losses) >= 2:
            t_trend = tinker_rl_losses[-1] - tinker_rl_losses[0]
            l_trend = local_rl_losses[-1] - local_rl_losses[0]
            if abs(t_trend) < 0.01 and abs(l_trend) < 0.01:
                passed = True  # both flat
            else:
                passed = (t_trend > 0) == (l_trend > 0)
            results.append(TestResult(
                category="TinkerAPI", name="rl_trend_agreement",
                passed=passed,
                detail=f"tinker={t_trend:+.4f} local={l_trend:+.4f}" if not passed else "",
            ))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    results: list[TestResult] = []

    # Determine which categories to run
    run_all = not any([
        args.loss_only, args.chunked_ce, args.optimizer,
        args.logprob, args.grad_clip, args.full_step, args.dynamic_map,
        args.real_model, args.tinker_api,
    ])

    if run_all or args.loss_only:
        results += run_loss_tests(np.random.RandomState(rng.randint(0, 2**31)), args)
    if run_all or args.chunked_ce:
        results += run_chunked_ce_tests(np.random.RandomState(rng.randint(0, 2**31)), args)
    if run_all or args.optimizer:
        results += run_optimizer_tests(np.random.RandomState(rng.randint(0, 2**31)), args)
    if run_all or args.dynamic_map:
        results += run_dynamic_map_tests(np.random.RandomState(rng.randint(0, 2**31)), args)
    if run_all or args.logprob:
        results += run_logprob_tests(np.random.RandomState(rng.randint(0, 2**31)), args)
    if run_all or args.grad_clip:
        results += run_grad_clip_tests(np.random.RandomState(rng.randint(0, 2**31)), args)
    if run_all or args.full_step:
        results += run_full_step_tests(np.random.RandomState(rng.randint(0, 2**31)), args)
    if run_all or args.real_model:
        results += run_real_model_tests(np.random.RandomState(rng.randint(0, 2**31)), args)
    if args.tinker_api:  # Opt-in only (not in run_all — requires API key + network)
        results += run_tinker_api_tests(np.random.RandomState(rng.randint(0, 2**31)), args)

    if args.json:
        report_json(results)
    else:
        report_table(results, verbose=args.verbose)

    n_fail = sum(1 for r in results if not r.passed and not r.skipped)
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
