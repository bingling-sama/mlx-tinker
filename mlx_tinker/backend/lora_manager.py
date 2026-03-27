"""QLoRA adapter lifecycle management using mlx-lm tuner utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.utils import linear_to_lora_layers

from mlx_tinker.types import LoraConfig

logger = logging.getLogger(__name__)


class LoRAManager:
    """Manages QLoRA adapter creation, save, load, and removal."""

    def apply_qlora(
        self,
        model: nn.Module,
        lora_config: LoraConfig,
        quantize_bits: int = 4,
        quantize_group_size: int = 64,
    ) -> nn.Module:
        """Quantize base model to N-bit and apply LoRA adapters.

        Args:
            model: The loaded MLX model.
            lora_config: LoRA hyperparameters (rank, alpha, targets).
            quantize_bits: Quantization bit width (default 4 for QLoRA).
            quantize_group_size: Group size for quantization.

        Returns:
            The model with quantized base weights and trainable LoRA params.
        """
        # Quantize base weights
        nn.quantize(model, bits=quantize_bits, group_size=quantize_group_size)
        logger.info(
            "Quantized base model to %d-bit (group_size=%d)", quantize_bits, quantize_group_size
        )

        # Build LoRA config dict for mlx-lm's linear_to_lora_layers
        keys = []
        if lora_config.train_attn:
            keys.extend(["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"])
        if lora_config.train_mlp:
            keys.extend(["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])

        lora_cfg = {
            "rank": lora_config.rank,
            "alpha": lora_config.alpha,
            "scale": lora_config.alpha / lora_config.rank,
            "dropout": 0.0,
            "keys": keys,
        }

        # Count model layers
        num_layers = len(model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else -1

        # Apply LoRA layers via mlx-lm utility
        linear_to_lora_layers(model, num_layers=num_layers, config=lora_cfg)

        # Freeze base, keep LoRA trainable
        model.freeze()
        model.train()

        trainable = sum(p.size for _, p in model.trainable_parameters())
        total = sum(p.size for _, p in model.parameters())
        logger.info(
            "LoRA applied: %d trainable / %d total params (%.2f%%)",
            trainable,
            total,
            100.0 * trainable / total,
        )

        return model

    def save_adapter(self, model: nn.Module, path: str | Path, lora_config: LoraConfig) -> Path:
        """Save only the trainable LoRA weights to safetensors."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Collect trainable (LoRA) weights
        weights = dict(model.trainable_parameters())
        mx.save_safetensors(str(path / "adapters.safetensors"), weights)

        # Save LoRA config
        config = lora_config.model_dump()
        (path / "adapter_config.json").write_text(json.dumps(config, indent=2))

        logger.info("Saved LoRA adapter to %s (%d tensors)", path, len(weights))
        return path

    def load_adapter(self, model: nn.Module, path: str | Path) -> nn.Module:
        """Load LoRA weights from a checkpoint directory."""
        path = Path(path)
        weights_file = path / "adapters.safetensors"

        if not weights_file.exists():
            raise FileNotFoundError(f"No adapter weights at {weights_file}")

        weights = mx.load(str(weights_file))
        model.load_weights(list(weights.items()), strict=False)
        logger.info("Loaded LoRA adapter from %s (%d tensors)", path, len(weights))
        return model

    def get_trainable_param_count(self, model: nn.Module) -> tuple[int, int]:
        """Return (trainable_params, total_params)."""
        trainable = sum(p.size for _, p in model.trainable_parameters())
        total = sum(p.size for _, p in model.parameters())
        return trainable, total

    def save_full_model(self, model: nn.Module, path: str | Path, config: dict[str, Any]) -> Path:
        """Save all model weights (for sampler checkpoint)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        weights = dict(model.parameters())
        mx.save_safetensors(str(path / "model.safetensors"), weights)
        (path / "config.json").write_text(json.dumps(config, indent=2))

        logger.info("Saved full model to %s (%d tensors)", path, len(weights))
        return path
