"""Loss functions for training, implemented in MLX.

All loss functions share the same signature:
    (target_logprobs, loss_mask, sampling_logprobs, advantages, loss_fn_config) -> scalar loss

This mirrors SkyRL-tx's convention so the engine can dispatch via LOSS_FUNCTION_MAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx


@dataclass
class LossFnConfig:
    clip_low_threshold: float = 0.0
    clip_high_threshold: float = float("inf")


LossFn = Callable[[mx.array, mx.array, mx.array, mx.array, LossFnConfig], mx.array]


def cross_entropy_loss(
    target_logprobs: mx.array,
    loss_mask: mx.array,
    _sampling_logprobs: mx.array,
    _advantages: mx.array,
    _loss_fn_config: LossFnConfig,
) -> mx.array:
    """Standard cross-entropy loss for SFT.

    target_logprobs: log P(target_token | context) for each position, shape [B, T]
    loss_mask: per-token weights (0 for prompt, 1 for completion), shape [B, T]
    """
    masked = -target_logprobs * loss_mask
    return masked.sum() / mx.maximum(loss_mask.sum(), 1.0)


def importance_sampling_loss(
    target_logprobs: mx.array,
    loss_mask: mx.array,
    sampling_logprobs: mx.array,
    advantages: mx.array,
    _loss_fn_config: LossFnConfig,
) -> mx.array:
    """Off-policy importance sampling loss for RL (REINFORCE-style).

    ratio = exp(logp_new - logp_old)
    loss = -mean(ratio * advantage * mask)
    """
    log_ratio = target_logprobs - sampling_logprobs
    ratio = mx.exp(log_ratio)
    per_token_loss = -ratio * advantages * loss_mask
    return per_token_loss.sum() / mx.maximum(loss_mask.sum(), 1.0)


def ppo_loss(
    target_logprobs: mx.array,
    loss_mask: mx.array,
    sampling_logprobs: mx.array,
    advantages: mx.array,
    loss_fn_config: LossFnConfig,
) -> mx.array:
    """PPO clipped surrogate loss.

    Clips the importance ratio to [1 - clip_high, 1 + clip_high] to prevent
    large policy updates.
    """
    log_ratio = target_logprobs - sampling_logprobs
    ratio = mx.exp(log_ratio)

    clip_eps = loss_fn_config.clip_high_threshold
    clipped_ratio = mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    per_token_loss = -mx.minimum(surr1, surr2) * loss_mask

    return per_token_loss.sum() / mx.maximum(loss_mask.sum(), 1.0)


def cispo_loss(
    target_logprobs: mx.array,
    loss_mask: mx.array,
    sampling_logprobs: mx.array,
    advantages: mx.array,
    loss_fn_config: LossFnConfig,
) -> mx.array:
    """CISPO (Clipped Importance Sampling Policy Optimization) loss.

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

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    per_token_loss = -mx.minimum(surr1, surr2) * loss_mask

    return per_token_loss.sum() / mx.maximum(loss_mask.sum(), 1.0)


LOSS_FUNCTION_MAP: dict[str, LossFn] = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
    "ppo": ppo_loss,
    "cispo": cispo_loss,
}
