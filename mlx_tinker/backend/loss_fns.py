"""Loss functions for training, implemented in MLX.

All loss functions share the same signature:
    (target_logprobs, loss_mask, sampling_logprobs, advantages, loss_fn_config) -> scalar loss

Matches Tinker API semantics: sum reduction, no mean normalization.
See https://tinker-docs.thinkingmachines.ai/losses for reference formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx


@dataclass
class LossFnConfig:
    clip_low_threshold: float = 0.0
    clip_high_threshold: float = float("inf")
    beta: float = 0.05


LossFn = Callable[[mx.array, mx.array, mx.array, mx.array, LossFnConfig], mx.array]


def cross_entropy_loss(
    target_logprobs: mx.array,
    loss_mask: mx.array,
    _sampling_logprobs: mx.array,
    _advantages: mx.array,
    _loss_fn_config: LossFnConfig,
) -> mx.array:
    """Standard cross-entropy loss for SFT (Tinker: sum reduction).

    loss = (-target_logprobs * weights).sum()
    """
    return (-target_logprobs * loss_mask).sum()


def importance_sampling_loss(
    target_logprobs: mx.array,
    _loss_mask: mx.array,
    sampling_logprobs: mx.array,
    advantages: mx.array,
    _loss_fn_config: LossFnConfig,
) -> mx.array:
    """Off-policy importance sampling loss (Tinker: sum reduction).

    loss = -(exp(logp_new - logp_old) * advantages).sum()

    Advantages=0 at prompt positions serves as implicit mask.
    """
    log_ratio = target_logprobs - sampling_logprobs
    ratio = mx.exp(log_ratio)
    return -(ratio * advantages).sum()


def ppo_loss(
    target_logprobs: mx.array,
    _loss_mask: mx.array,
    sampling_logprobs: mx.array,
    advantages: mx.array,
    loss_fn_config: LossFnConfig,
) -> mx.array:
    """PPO clipped surrogate loss (Tinker: sum reduction).

    loss = -min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv).sum()
    """
    log_ratio = target_logprobs - sampling_logprobs
    ratio = mx.exp(log_ratio)

    clip_low = loss_fn_config.clip_low_threshold
    clip_high = loss_fn_config.clip_high_threshold
    clipped_ratio = mx.clip(ratio, 1.0 - clip_low, 1.0 + clip_high)

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    return -mx.minimum(surr1, surr2).sum()


def cispo_loss(
    target_logprobs: mx.array,
    _loss_mask: mx.array,
    sampling_logprobs: mx.array,
    advantages: mx.array,
    loss_fn_config: LossFnConfig,
) -> mx.array:
    """CISPO loss (Tinker: sum reduction, stop-gradient on clipped ratio).

    loss = -(sg(clip(ratio, 1-eps_low/high, 1+eps_low/high)) * logprobs * advantages).sum()

    Uses asymmetric clipping: different thresholds for positive and negative advantages.
    """
    log_ratio = target_logprobs - sampling_logprobs
    ratio = mx.exp(log_ratio)

    clip_low = loss_fn_config.clip_low_threshold
    clip_high = loss_fn_config.clip_high_threshold

    positive_adv = advantages > 0
    clipped_ratio = mx.where(
        positive_adv,
        mx.clip(ratio, 1.0 - clip_high, 1.0 + clip_high),
        mx.clip(ratio, 1.0 - clip_low, 1.0 + clip_low),
    )

    return -(mx.stop_gradient(clipped_ratio) * target_logprobs * advantages).sum()


def dro_loss(
    target_logprobs: mx.array,
    _loss_mask: mx.array,
    sampling_logprobs: mx.array,
    advantages: mx.array,
    loss_fn_config: LossFnConfig,
) -> mx.array:
    """Direct Reward Optimization loss (Tinker: sum reduction).

    loss = -(logprobs * advantages - 0.5 * beta * (logprobs - sampling_logprobs)^2).sum()
    """
    beta = loss_fn_config.beta
    quadratic = (target_logprobs - sampling_logprobs) ** 2
    obj = target_logprobs * advantages - 0.5 * beta * quadratic
    return -obj.sum()


LOSS_FUNCTION_MAP: dict[str, LossFn] = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
    "ppo": ppo_loss,
    "cispo": cispo_loss,
    "dro": dro_loss,
}


# ---------------------------------------------------------------------------
# Chunked cross-entropy (memory-efficient)
# ---------------------------------------------------------------------------

CE_CHUNK_SIZE = 8192


def chunked_target_logprobs(
    hidden_states: mx.array,
    lm_head_weight: mx.array,
    targets: mx.array,
 ) -> mx.array:
    """Compute target-token logprobs without materializing full [B, T, V].

    Instead of computing all V logits at once, we:
    1. Compute the target token's logit directly (cheap: single gather)
    2. Compute logsumexp over the full vocab in chunks of CE_CHUNK_SIZE

    This reduces peak logit memory from [B, T, V] to [B, T, chunk_size].
    """
    hidden_states = hidden_states.astype(mx.float32)
    lm_head_weight = lm_head_weight.astype(mx.float32)
    V = lm_head_weight.shape[0]

    target_weight = lm_head_weight[targets]  # [B, T, D]
    target_logits = mx.sum(hidden_states * target_weight, axis=-1)  # [B, T]

    running_max = mx.full(target_logits.shape, float("-inf"))
    running_sum_exp = mx.zeros(target_logits.shape)

    for chunk_start in range(0, V, CE_CHUNK_SIZE):
        chunk_end = min(chunk_start + CE_CHUNK_SIZE, V)
        chunk_weight = lm_head_weight[chunk_start:chunk_end]  # [chunk, D]
        chunk_logits = hidden_states @ chunk_weight.T  # [B, T, chunk]

        chunk_max = mx.max(chunk_logits, axis=-1)  # [B, T]
        new_max = mx.maximum(running_max, chunk_max)

        running_sum_exp = running_sum_exp * mx.exp(running_max - new_max)
        running_sum_exp = running_sum_exp + mx.sum(
            mx.exp(chunk_logits - new_max[:, :, None]), axis=-1
        )
        running_max = new_max

    logsumexp = running_max + mx.log(running_sum_exp)
    return target_logits - logsumexp


def chunked_cross_entropy_loss(
    hidden_states: mx.array,
    lm_head_weight: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
) -> mx.array:
    """Memory-efficient cross-entropy that never materializes full [B, T, V]."""
    target_logprobs = chunked_target_logprobs(hidden_states, lm_head_weight, targets)

    return (-target_logprobs * loss_mask).sum()
