"""Helpers to enable MLX gradient checkpointing on decoder block layers.

This patches model layer instances in place so parameter names and module
structure remain stable for LoRA save/load and checkpoint serialization.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import mlx.nn as nn
from mlx.nn.utils import checkpoint as checkpoint_module

logger = logging.getLogger(__name__)

_CHECKPOINT_ENABLED_ATTR = "_mlx_tinker_gradient_checkpointing_enabled"
_CHECKPOINT_CALL_ATTR = "_mlx_tinker_checkpointed_call"
_CHECKPOINT_CLASS_CACHE: dict[type, type] = {}


def _layer_list_candidates(model: nn.Module) -> Iterable[list[nn.Module] | None]:
    """Yield likely decoder-layer containers for mlx-lm style models."""
    try:
        yield getattr(model, "layers", None)
    except Exception:
        yield None

    nested = getattr(model, "model", None)
    if nested is not None:
        try:
            yield getattr(nested, "layers", None)
        except Exception:
            yield None

    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        try:
            yield getattr(language_model, "layers", None)
        except Exception:
            yield None
        nested = getattr(language_model, "model", None)
        if nested is not None:
            try:
                yield getattr(nested, "layers", None)
            except Exception:
                yield None


def _resolve_layers(model: nn.Module) -> list[nn.Module] | None:
    """Find the list object that contains decoder blocks for the model."""
    seen: set[int] = set()
    for layers in _layer_list_candidates(model):
        if isinstance(layers, list) and all(isinstance(layer, nn.Module) for layer in layers):
            if id(layers) in seen:
                continue
            seen.add(id(layers))
            return layers
    return None


def _checkpointed_layer_subclass(base_cls: type) -> type:
    """Build and cache a subclass that checkpoints the layer during training."""
    cached = _CHECKPOINT_CLASS_CACHE.get(base_cls)
    if cached is not None:
        return cached

    class CheckpointedLayer(base_cls):
        def __call__(self, *args, **kwargs):
            # Sampling paths set model.eval(); keep them on the original call path.
            if not self.training:
                return base_cls.__call__(self, *args, **kwargs)

            checkpointed = getattr(self, _CHECKPOINT_CALL_ATTR, None)
            if checkpointed is None:
                checkpointed = checkpoint_module(
                    self,
                    lambda *a, **k: base_cls.__call__(self, *a, **k),
                )
                setattr(self, _CHECKPOINT_CALL_ATTR, checkpointed)
            return checkpointed(*args, **kwargs)

    CheckpointedLayer.__name__ = f"Checkpointed{base_cls.__name__}"
    CheckpointedLayer.__qualname__ = CheckpointedLayer.__name__
    CheckpointedLayer.__module__ = base_cls.__module__
    _CHECKPOINT_CLASS_CACHE[base_cls] = CheckpointedLayer
    return CheckpointedLayer


def enable_gradient_checkpointing(model: nn.Module) -> int:
    """Patch model layers in place to recompute block activations in backward.

    Returns:
        Number of layers newly patched for checkpointed execution.
    """
    layers = _resolve_layers(model)
    if not layers:
        logger.warning(
            "Gradient checkpointing requested but no decoder layers were found on %s",
            type(model).__name__,
        )
        return 0

    enabled = 0
    for layer in layers:
        if getattr(layer, _CHECKPOINT_ENABLED_ATTR, False):
            continue
        layer.__class__ = _checkpointed_layer_subclass(layer.__class__)
        setattr(layer, _CHECKPOINT_ENABLED_ATTR, True)
        enabled += 1

    if enabled:
        logger.info(
            "Enabled gradient checkpointing on %d layers for %s",
            enabled,
            type(model).__name__,
        )
    return enabled

