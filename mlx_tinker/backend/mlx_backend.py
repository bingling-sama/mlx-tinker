"""MLX Backend facade — coordinates model lifecycle, training, and inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.nn as nn
from mlx_lm import load as mlx_load

from mlx_tinker.backend.checkpointing import (
    load_training_checkpoint,
    save_sampler_weights,
    save_training_checkpoint,
)
from mlx_tinker.backend.inference import InferenceBackend
from mlx_tinker.backend.lora_manager import LoRAManager
from mlx_tinker.backend.training import TrainingBackend
from mlx_tinker.config import EngineConfig
from mlx_tinker.types import (
    CreateModelInput,
    CreateModelOutput,
    ForwardBackwardInput,
    ForwardBackwardOutput,
    ForwardInput,
    ForwardOutput,
    LoadWeightsInput,
    LoadWeightsOutput,
    LoraConfig,
    OptimStepInput,
    OptimStepOutput,
    SampleInput,
    SampleOutput,
    SaveWeightsForSamplerInput,
    SaveWeightsForSamplerOutput,
    SaveWeightsInput,
    SaveWeightsOutput,
    UnloadModelInput,
    UnloadModelOutput,
)

logger = logging.getLogger(__name__)

MAX_SAMPLES = 128


class MLXBackend:
    """Top-level backend managing models, training, and inference on MLX."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.training = TrainingBackend()
        self.inference = InferenceBackend()
        self.lora_manager = LoRAManager()

        # Model registry: model_id -> (model, tokenizer, lora_config)
        self.models: dict[str, nn.Module] = {}
        self.tokenizers: dict[str, Any] = {}
        self.lora_configs: dict[str, LoraConfig] = {}
        self.model_configs: dict[str, dict] = {}

        # Base model (shared, loaded once)
        self._base_model: nn.Module | None = None
        self._base_tokenizer: Any = None
        self._base_model_config: dict = {}

    def _ensure_base_model(self) -> None:
        """Load the base model if not already loaded."""
        if self._base_model is not None:
            return

        logger.info("Loading base model: %s", self.config.base_model)
        try:
            model, tokenizer = mlx_load(self.config.base_model)
        except Exception as e:
            logger.error("Failed to load base model %s: %s", self.config.base_model, e)
            raise ValueError(f"Failed to load base model '{self.config.base_model}': {e}") from e
        self._base_model = model
        self._base_tokenizer = tokenizer
        logger.info("Base model loaded")

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def create_model(self, model_id: str, request: CreateModelInput) -> CreateModelOutput:
        """Create a new QLoRA model from the base model."""
        self._ensure_base_model()

        # Load a fresh copy for this model_id
        logger.info("Creating model %s with LoRA rank=%d", model_id, request.lora_config.rank)
        try:
            model, tokenizer = mlx_load(self.config.base_model)
        except Exception as e:
            logger.error("Failed to load model for %s: %s", model_id, e)
            raise ValueError(f"Failed to load model for '{model_id}': {e}") from e

        # Apply QLoRA
        model = self.lora_manager.apply_qlora(
            model,
            request.lora_config,
            quantize_bits=self.config.quantize_bits,
            quantize_group_size=self.config.quantize_group_size,
        )

        self.models[model_id] = model
        self.tokenizers[model_id] = tokenizer
        self.lora_configs[model_id] = request.lora_config

        return CreateModelOutput(
            model_id=model_id,
            base_model=self.config.base_model,
            lora_config=request.lora_config,
        )

    def unload_model(self, model_id: str, _request: UnloadModelInput) -> UnloadModelOutput:
        """Unload a model from memory."""
        if model_id in self.models:
            del self.models[model_id]
            del self.tokenizers[model_id]
            del self.lora_configs[model_id]
            self.training.accumulated_grads.pop(model_id, None)
            self.training.grad_accum_counts.pop(model_id, None)
            self.training.optimizers.pop(model_id, None)
            logger.info("Unloaded model %s", model_id)

        return UnloadModelOutput(model_id=model_id, status="unloaded")

    def _get_model(self, model_id: str) -> nn.Module:
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found. Call create_model first.")
        return self.models[model_id]

    def _get_tokenizer(self, model_id: str) -> Any:
        if model_id not in self.tokenizers:
            raise ValueError(f"Tokenizer for {model_id} not found.")
        return self.tokenizers[model_id]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward_backward(
        self, model_id: str, request: ForwardBackwardInput
    ) -> ForwardBackwardOutput:
        model = self._get_model(model_id)
        model.train()
        return self.training.forward_backward(model_id, model, request)

    def forward(self, model_id: str, request: ForwardInput) -> ForwardOutput:
        model = self._get_model(model_id)
        model.eval()
        return self.training.forward(model_id, model, request)

    def optim_step(self, model_id: str, request: OptimStepInput) -> OptimStepOutput:
        model = self._get_model(model_id)
        return self.training.optim_step(model_id, model, request)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def sample(self, model_id: str | None, request: SampleInput) -> SampleOutput:
        """Generate samples. Uses model_id if provided, else base model."""
        if request.num_samples > MAX_SAMPLES:
            raise ValueError(f"num_samples={request.num_samples} exceeds maximum of {MAX_SAMPLES}")

        if model_id and model_id in self.models:
            model = self.models[model_id]
            tokenizer = self.tokenizers[model_id]
        else:
            self._ensure_base_model()
            model = self._base_model
            tokenizer = self._base_tokenizer

        model.eval()
        return self.inference.sample(model, tokenizer, request)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _validate_checkpoint_path(self, requested_path: str) -> Path:
        """Validate that a checkpoint path is within the allowed base directory."""
        resolved = Path(requested_path).resolve()
        base = self.config.checkpoints_base.resolve()
        if not str(resolved).startswith(str(base) + "/") and resolved != base:
            raise ValueError(
                f"Checkpoint path '{requested_path}' is outside the allowed "
                f"directory '{self.config.checkpoints_base}'"
            )
        return resolved

    def save_weights(self, model_id: str, request: SaveWeightsInput) -> SaveWeightsOutput:
        model = self._get_model(model_id)
        opt_state = self.training.get_optimizer_state(model_id)
        checkpoint_dir = self._validate_checkpoint_path(request.path)
        save_training_checkpoint(model, opt_state, checkpoint_dir)
        return SaveWeightsOutput(path=str(checkpoint_dir))

    def save_weights_for_sampler(
        self, model_id: str, request: SaveWeightsForSamplerInput
    ) -> SaveWeightsForSamplerOutput:
        model = self._get_model(model_id)
        path = request.path or str(self.config.checkpoints_base / model_id / "sampler" / "latest")
        safe_path = self._validate_checkpoint_path(path)
        save_sampler_weights(model, safe_path)
        return SaveWeightsForSamplerOutput(path=str(safe_path))

    def load_weights(self, model_id: str, request: LoadWeightsInput) -> LoadWeightsOutput:
        model = self._get_model(model_id)
        checkpoint_dir = (
            self.config.checkpoints_base / request.source_model_id / request.checkpoint_id
        )
        self._validate_checkpoint_path(str(checkpoint_dir))
        opt_state = load_training_checkpoint(model, checkpoint_dir)
        if opt_state is not None:
            self.training.load_optimizer_state(model_id, opt_state)
        return LoadWeightsOutput()
