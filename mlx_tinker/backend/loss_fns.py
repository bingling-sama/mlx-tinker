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


# ---------------------------------------------------------------------------
# Chunked cross-entropy (memory-efficient)
# ---------------------------------------------------------------------------

CE_CHUNK_SIZE = 8192


def chunked_cross_entropy_loss(
    hidden_states: mx.array,
    lm_head_weight: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
) -> mx.array:
    """Memory-efficient cross-entropy that never materializes full [B, T, V].

    Instead of computing all V logits at once, we:
    1. Compute the target token's logit directly (cheap: single gather)
    2. Compute logsumexp over the full vocab in chunks of CE_CHUNK_SIZE

    This reduces peak logit memory from [B, T, V] to [B, T, chunk_size].
    For Qwen3.5 (V=151,936), this is an 18x reduction.

    Args:
        hidden_states: Last hidden layer output, [B, T, D]
        lm_head_weight: Vocabulary projection weight, [V, D]
        targets: Target token IDs, [B, T]
        loss_mask: Per-token weights, [B, T]

    Returns:
        Scalar cross-entropy loss.
    """
    V = lm_head_weight.shape[0]

    # 1. Compute target token logits directly: gather the relevant rows
    #    target_weight: [B, T, D] — one weight vector per target token
    target_weight = lm_head_weight[targets]  # [B, T, D]
    # Dot product: sum over D dimension gives target logit per position
    target_logits = mx.sum(hidden_states * target_weight, axis=-1)  # [B, T]

    # 2. Compute logsumexp in chunks to avoid materializing [B, T, V]
    #    logsumexp(x) = c + log(sum(exp(x - c)))
    #    We compute this incrementally across vocab chunks.
    #    Use the log-sum-exp trick with running max for numerical stability.

    # Initialize with -inf so first chunk sets the values
    running_max = mx.full(target_logits.shape, float("-inf"))
    running_sum_exp = mx.zeros(target_logits.shape)

    for chunk_start in range(0, V, CE_CHUNK_SIZE):
        chunk_end = min(chunk_start + CE_CHUNK_SIZE, V)
        # Project hidden states onto this vocab chunk: [B, T, chunk_size]
        chunk_weight = lm_head_weight[chunk_start:chunk_end]  # [chunk, D]
        chunk_logits = hidden_states @ chunk_weight.T  # [B, T, chunk]

        # Update running logsumexp
        chunk_max = mx.max(chunk_logits, axis=-1)  # [B, T]
        new_max = mx.maximum(running_max, chunk_max)

        # Rescale previous sum to new max
        running_sum_exp = running_sum_exp * mx.exp(running_max - new_max)
        # Add this chunk's contribution
        running_sum_exp = running_sum_exp + mx.sum(
            mx.exp(chunk_logits - new_max[:, :, None]), axis=-1
        )
        running_max = new_max

    # Final logsumexp = running_max + log(running_sum_exp)
    logsumexp = running_max + mx.log(running_sum_exp)

    # 3. log_softmax(target) = target_logit - logsumexp
    target_logprobs = target_logits - logsumexp

    # 4. Standard masked CE loss
    masked = -target_logprobs * loss_mask
    return masked.sum() / mx.maximum(loss_mask.sum(), 1.0)
