"""Training-time LongLoRA attention patching for MLX models.

LongLoRA's core idea is shifted short attention (S2-Attn): replace full causal
attention during training with grouped causal attention after rolling half of
the heads by half a group. This reduces attention complexity while leaving
inference unchanged because the patch only activates when ``module.training`` is
true and there is no cache in use.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention

logger = logging.getLogger(__name__)

_LONGLORA_ENABLED_ATTR = "_mlx_tinker_longlora_enabled"
_LONGLORA_RATIO_ATTR = "_mlx_tinker_longlora_group_size_ratio"
_LONGLORA_CLASS_CACHE: dict[type, type] = {}


def _attention_module_candidates(model: nn.Module) -> Iterable[nn.Module]:
    """Yield full-attention modules that match mlx-lm's common projection layout."""
    for _name, module in model.named_modules():
        if not isinstance(module, nn.Module):
            continue
        if not all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "rope")):
            continue
        if hasattr(module, "n_heads") and hasattr(module, "n_kv_heads") and hasattr(module, "head_dim"):
            yield module
            continue
        if (
            hasattr(module, "num_attention_heads")
            and hasattr(module, "num_key_value_heads")
            and hasattr(module, "head_dim")
        ):
            yield module


def _attention_head_counts(module: nn.Module) -> tuple[int, int]:
    if hasattr(module, "n_heads") and hasattr(module, "n_kv_heads"):
        return int(module.n_heads), int(module.n_kv_heads)
    return int(module.num_attention_heads), int(module.num_key_value_heads)


def _shift_and_group(tensor: mx.array, group_size: int) -> mx.array:
    """Apply LongLoRA's head shift and pack sequence groups into the batch axis."""
    batch_size, num_heads, seq_len, head_dim = tensor.shape
    shifted = tensor
    shift = group_size // 2
    if shift > 0 and num_heads >= 2:
        first_half = tensor[:, : num_heads // 2]
        second_half = mx.roll(tensor[:, num_heads // 2 :], shift=-shift, axis=2)
        shifted = mx.concatenate([first_half, second_half], axis=1)
    num_groups = seq_len // group_size
    return shifted.transpose(0, 2, 1, 3).reshape(
        batch_size * num_groups, group_size, num_heads, head_dim
    ).transpose(0, 2, 1, 3)


def _ungroup_and_unshift(
    tensor: mx.array,
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    group_size: int,
) -> mx.array:
    """Undo grouping and head shift after grouped attention completes."""
    shift = group_size // 2
    output = tensor.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads, -1)
    if shift > 0 and num_heads >= 2:
        first_half = output[:, :, : num_heads // 2]
        second_half = mx.roll(output[:, :, num_heads // 2 :], shift=shift, axis=1)
        output = mx.concatenate([first_half, second_half], axis=2)
    return output


def _longlora_group_size(seq_len: int, group_size_ratio: float) -> int | None:
    if not (0.0 < group_size_ratio < 1.0):
        return None
    num_groups_float = 1.0 / group_size_ratio
    num_groups = int(round(num_groups_float))
    if abs(num_groups_float - num_groups) > 1e-6 or num_groups <= 1:
        return None
    if seq_len < num_groups or seq_len % num_groups != 0:
        return None
    group_size = seq_len // num_groups
    if group_size < 2:
        return None
    return group_size


def _project_qkv(module: nn.Module, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array | None]:
    batch_size, seq_len, _hidden = x.shape
    num_q_heads, num_kv_heads = _attention_head_counts(module)

    q_proj_output = module.q_proj(x)
    q_proj_output = q_proj_output.reshape(batch_size, seq_len, num_q_heads, -1)

    gate = None
    if q_proj_output.shape[-1] == 2 * int(module.head_dim):
        queries, gate = mx.split(q_proj_output, 2, axis=-1)
        gate = gate.reshape(batch_size, seq_len, -1)
    else:
        queries = q_proj_output

    if hasattr(module, "q_norm"):
        queries = module.q_norm(queries)
    queries = queries.transpose(0, 2, 1, 3)

    keys = module.k_proj(x).reshape(batch_size, seq_len, num_kv_heads, -1)
    if hasattr(module, "k_norm"):
        keys = module.k_norm(keys)
    keys = keys.transpose(0, 2, 1, 3)

    values = module.v_proj(x).reshape(batch_size, seq_len, num_kv_heads, -1).transpose(0, 2, 1, 3)
    return queries, keys, values, gate


def _longlora_attention_forward(module: nn.Module, x: mx.array, mask: Any = None) -> mx.array:
    batch_size, seq_len, _hidden = x.shape
    num_q_heads, _num_kv_heads = _attention_head_counts(module)
    group_size_ratio = float(getattr(module, _LONGLORA_RATIO_ATTR))
    group_size = _longlora_group_size(seq_len, group_size_ratio)
    if group_size is None:
        raise ValueError(
            f"LongLoRA requires seq_len divisible into equal groups for ratio {group_size_ratio}, got {seq_len}"
        )

    queries, keys, values, gate = _project_qkv(module, x)
    queries = module.rope(queries)
    keys = module.rope(keys)

    grouped_queries = _shift_and_group(queries, group_size)
    grouped_keys = _shift_and_group(keys, group_size)
    grouped_values = _shift_and_group(values, group_size)

    output = scaled_dot_product_attention(
        grouped_queries,
        grouped_keys,
        grouped_values,
        cache=None,
        scale=module.scale,
        mask="causal",
    )
    output = _ungroup_and_unshift(
        output,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_q_heads,
        group_size=group_size,
    ).reshape(batch_size, seq_len, -1)

    if gate is not None:
        output = output * mx.sigmoid(gate)
    return module.o_proj(output)


def _longlora_attention_subclass(base_cls: type) -> type:
    cached = _LONGLORA_CLASS_CACHE.get(base_cls)
    if cached is not None:
        return cached

    class LongLoRAAttention(base_cls):
        def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
            group_size_ratio = getattr(self, _LONGLORA_RATIO_ATTR, None)
            causal_mask = mask is None or (isinstance(mask, str) and mask == "causal")
            if (
                not self.training
                or cache is not None
                or group_size_ratio is None
                or not causal_mask
            ):
                return base_cls.__call__(self, x, mask, cache)

            try:
                return _longlora_attention_forward(self, x, mask=mask)
            except Exception:
                logger.debug(
                    "LongLoRA attention fell back to full attention for %s",
                    type(self).__name__,
                    exc_info=True,
                )
                return base_cls.__call__(self, x, mask, cache)

    LongLoRAAttention.__name__ = f"LongLoRA{base_cls.__name__}"
    LongLoRAAttention.__qualname__ = LongLoRAAttention.__name__
    LongLoRAAttention.__module__ = base_cls.__module__
    _LONGLORA_CLASS_CACHE[base_cls] = LongLoRAAttention
    return LongLoRAAttention


def enable_longlora_attention(model: nn.Module, group_size_ratio: float = 0.25) -> int:
    """Patch attention modules in-place with training-time LongLoRA attention."""
    enabled = 0
    for module in _attention_module_candidates(model):
        if getattr(module, _LONGLORA_ENABLED_ATTR, False):
            setattr(module, _LONGLORA_RATIO_ATTR, group_size_ratio)
            continue
        module.__class__ = _longlora_attention_subclass(module.__class__)
        setattr(module, _LONGLORA_RATIO_ATTR, group_size_ratio)
        setattr(module, _LONGLORA_ENABLED_ATTR, True)
        enabled += 1

    if enabled:
        logger.info(
            "Enabled LongLoRA attention on %d modules for %s (group_size_ratio=%.3f)",
            enabled,
            type(model).__name__,
            group_size_ratio,
        )
    return enabled
