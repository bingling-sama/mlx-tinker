"""QLoRA adapter lifecycle management using mlx-lm tuner utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
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
            keys.extend(
                [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                ]
            )
        if lora_config.train_mlp:
            keys.extend(["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])

        lora_cfg = {
            "rank": lora_config.rank,
            "alpha": lora_config.alpha,
            "scale": lora_config.alpha / lora_config.rank,
            "dropout": 0.0,
            "keys": keys,
        }

        # Count model layers — .layers may be a property on the top-level model
        # (as in Qwen3ForCausalLM) or on model.model
        if hasattr(model, "layers"):
            num_layers = len(model.layers)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            num_layers = len(model.model.layers)
        else:
            num_layers = -1

        # Apply LoRA layers via mlx-lm utility
        linear_to_lora_layers(model, num_layers=num_layers, config=lora_cfg)

        # Freeze all base weights, then unfreeze LoRA params
        model.freeze()
        # Selectively unfreeze LoRA parameters
        for name, p in tree_flatten(model.parameters()):
            if "lora_a" in name or "lora_b" in name:
                # Navigate to the parent module and unfreeze the param
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    try:
                        module = model
                        for attr in parts[0].split("."):
                            if attr.isdigit():
                                module = module[int(attr)]
                            else:
                                module = getattr(module, attr)
                        module.unfreeze(keys=[parts[1]])
                    except (AttributeError, IndexError, KeyError) as e:
                        logger.error("Failed to unfreeze LoRA param '%s': %s", name, e)
                        raise RuntimeError(
                            f"Failed to unfreeze LoRA parameter '{name}': {e}"
                        ) from e
        model.train()

        # Verify LoRA params are actually trainable
        trainable_names = [name for name, _ in tree_flatten(model.trainable_parameters())]
        lora_params = [n for n in trainable_names if "lora_a" in n or "lora_b" in n]
        if not lora_params:
            raise RuntimeError(
                "No LoRA parameters found after apply_qlora. "
                "Check that the model architecture matches the expected layer names."
            )

        trainable, total = self.get_trainable_param_count(model)
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
        weights = dict(tree_flatten(model.trainable_parameters()))
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
        trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
        total = sum(p.size for _, p in tree_flatten(model.parameters()))
        return trainable, total

    def save_full_model(self, model: nn.Module, path: str | Path, config: dict[str, Any]) -> Path:
        """Save all model weights (for sampler checkpoint)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        weights = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(str(path / "model.safetensors"), weights)
        (path / "config.json").write_text(json.dumps(config, indent=2))

        logger.info("Saved full model to %s (%d tensors)", path, len(weights))
        return path
