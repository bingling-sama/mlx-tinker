"""Checkpoint save/load helpers for model weights, adapters, and optimizer state."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)


def save_training_checkpoint(
    model: nn.Module,
    optimizer_state: dict | None,
    checkpoint_dir: Path,
    metadata: dict | None = None,
) -> Path:
    """Save full training state: model weights + optimizer moments.

    Args:
        model: The MLX model (with LoRA params).
        optimizer_state: Adam optimizer state dict (moments, step counts).
        checkpoint_dir: Directory to save into.
        metadata: Optional metadata (step number, loss, etc.).

    Returns:
        The checkpoint directory path.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save all model weights (including LoRA)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(checkpoint_dir / "model.safetensors"), weights)

    # Save optimizer state
    if optimizer_state is not None:
        _save_optimizer_state(optimizer_state, checkpoint_dir / "optimizer")

    # Save metadata
    meta = metadata or {}
    (checkpoint_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    logger.info("Saved training checkpoint to %s", checkpoint_dir)
    return checkpoint_dir


def load_training_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
) -> dict | None:
    """Load model weights and return optimizer state if present.

    Args:
        model: The MLX model to load weights into.
        checkpoint_dir: Directory containing the checkpoint.

    Returns:
        Optimizer state dict, or None if not present.
    """
    weights_path = checkpoint_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"No model weights at {weights_path}")

    weights = mx.load(str(weights_path))
    model.load_weights(list(weights.items()), strict=False)
    logger.info("Loaded model weights from %s", weights_path)

    # Load optimizer state if present
    opt_dir = checkpoint_dir / "optimizer"
    if opt_dir.exists():
        opt_state = _load_optimizer_state(opt_dir)
        logger.info("Loaded optimizer state from %s", opt_dir)
        return opt_state

    return None


def save_sampler_weights(
    model: nn.Module,
    output_dir: Path,
    base_model: str,
    lora_config: dict | None = None,
    tokenizer_config: dict | None = None,
) -> Path:
    """Save a LoRA sampler checkpoint for inference.

    Sampler checkpoints are expected to be lightweight adapter-only exports plus
    enough metadata to reconstruct the base model at inference time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(output_dir / "adapters.safetensors"), adapter_weights)

    config_payload = {"base_model": base_model}
    if lora_config:
        config_payload["lora_config"] = lora_config
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    if tokenizer_config:
        (output_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2))

    logger.info("Saved sampler adapter weights to %s (%d tensors)", output_dir, len(adapter_weights))
    return output_dir


def _save_optimizer_state(state: dict, opt_dir: Path) -> None:
    """Serialize optimizer state (nested dicts of mx.arrays) to disk."""
    opt_dir.mkdir(parents=True, exist_ok=True)
    flat = _flatten_state(state)
    if flat:
        mx.savez(str(opt_dir / "state.npz"), **flat)


def _load_optimizer_state(opt_dir: Path) -> dict:
    """Deserialize optimizer state from disk."""
    state_path = opt_dir / "state.npz"
    if not state_path.exists():
        return {}
    flat = dict(mx.load(str(state_path)))
    return _unflatten_state(flat)


def _unflatten_state(flat: dict[str, mx.array]) -> dict:
    """Reconstruct nested state dict from flat key-value pairs."""
    nested: dict = {}
    for dotted_key, value in flat.items():
        parts = dotted_key.split(".")
        target = nested
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return nested


def _flatten_state(state: dict, prefix: str = "") -> dict[str, mx.array]:
    """Flatten nested state dict into flat key-value pairs."""
    flat = {}
    for key, value in state.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, mx.array):
            flat[full_key] = value
        elif isinstance(value, dict):
            flat.update(_flatten_state(value, full_key))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                if isinstance(v, mx.array):
                    flat[f"{full_key}.{i}"] = v
                elif isinstance(v, dict):
                    flat.update(_flatten_state(v, f"{full_key}.{i}"))
    return flat
