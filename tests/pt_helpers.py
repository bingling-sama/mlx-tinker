"""PyTorch reference implementations for cross-framework parity tests.

Extracted from scripts/cross_framework_audit.py. This module assumes torch
is available — callers must use pytest.importorskip("torch") before importing.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch
import torch.nn

from mlx_tinker.backend.loss_fns import LossFnConfig

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
# Comparison utilities
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


def assert_close(
    a: np.ndarray,
    b: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    msg: str = "",
) -> None:
    """Assert two numpy arrays are close. Raises AssertionError with diagnostics."""
    passed, max_diff, detail = compare(a, b, rtol=rtol, atol=atol)
    if not passed:
        prefix = f"{msg}: " if msg else ""
        raise AssertionError(f"{prefix}max_diff={max_diff:.4e}\n{detail}")


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


def pt_standard_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Standard (non-chunked) CE: full logits -> logsumexp -> gather."""
    logits = hidden @ weight.T
    log_probs = torch.log_softmax(logits, dim=-1)
    target_lp = torch.gather(log_probs, 2, targets.unsqueeze(-1).long()).squeeze(-1)
    return (-target_lp * mask).sum()


def pt_chunked_cross_entropy_loss(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """PyTorch reference of chunked CE with running logsumexp."""
    V = weight.shape[0]
    target_weight = weight[targets]
    target_logits = (hidden * target_weight).sum(dim=-1)

    running_max = torch.full(target_logits.shape, float("-inf"))
    running_sum_exp = torch.zeros_like(target_logits)

    for chunk_start in range(0, V, chunk_size):
        chunk_end = min(chunk_start + chunk_size, V)
        chunk_w = weight[chunk_start:chunk_end]
        chunk_logits = hidden @ chunk_w.T

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


PT_LOSS_MAP = {
    "cross_entropy": pt_cross_entropy_loss,
    "importance_sampling": pt_importance_sampling_loss,
    "ppo": pt_ppo_loss,
    "cispo": pt_cispo_loss,
    "dro": pt_dro_loss,
}


# ---------------------------------------------------------------------------
# Config bridge
# ---------------------------------------------------------------------------


def get_loss_cfg(loss_name: str) -> tuple[LossFnConfig, dict]:
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
# PyTorch TinyModel (mirrors tests/helpers.py:TinyModel)
# ---------------------------------------------------------------------------


class PtTinyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, dim: int = 16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.head = torch.nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def sync_weights_mx_to_pt(mx_model, pt_model: PtTinyModel) -> None:
    """Copy weights from MLX TinyModel to PyTorch PtTinyModel."""
    mx_embed = mx_to_np(mx_model.embed.weight)
    mx_head = mx_to_np(mx_model.head.weight)
    pt_model.embed.weight.data = torch.tensor(mx_embed)
    pt_model.head.weight.data = torch.tensor(mx_head)
