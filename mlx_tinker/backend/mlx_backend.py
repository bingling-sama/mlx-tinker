"""MLX Backend facade — coordinates model lifecycle, training, and inference."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import uuid

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load

from mlx_tinker.backend.checkpointing import (
    load_training_checkpoint,
    save_sampler_weights,
    save_training_checkpoint,
)
from mlx_tinker.backend.uri import (
    format_tinker_path,
    is_tinker_path,
    tinker_path_to_relative_path,
)
from mlx_tinker.backend.gradient_checkpointing import enable_gradient_checkpointing
from mlx_tinker.backend.inference import InferenceBackend, prepare_sample_request
from mlx_tinker.backend.longlora import enable_longlora_attention
from mlx_tinker.backend.lora_manager import LoRAManager
from mlx_tinker.backend.transcript_cache import TranscriptPrefixCacheManager
from mlx_tinker.backend.training import TrainingBackend
from mlx_tinker.config import EngineConfig
from mlx_tinker.types import (
    CheckpointType,
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
        prefix_cache_bytes = int(max(0.0, config.prefix_cache_disk_limit_gb) * (1024**3))
        self.transcript_cache = TranscriptPrefixCacheManager(
            config.checkpoints_base / "prefix_cache",
            prefix_cache_bytes,
        )
        self.training = TrainingBackend(
            optimizer_type=config.optimizer_type,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        self.inference = InferenceBackend(
            max_kv_cache_size=config.max_kv_cache_size,
            kv_cache_bits=config.kv_cache_bits,
            kv_cache_group_size=config.kv_cache_group_size,
            quantized_kv_start=config.quantized_kv_start,
            transcript_cache=self.transcript_cache,
        )
        self.lora_manager = LoRAManager()

        # Model registry: model_id -> (model, tokenizer, lora_config)
        self.models: dict[str, nn.Module] = {}
        self.tokenizers: dict[str, Any] = {}
        self.lora_configs: dict[str, LoraConfig] = {}
        self.model_configs: dict[str, dict] = {}
        self.current_loaded_sampler_path: str | None = None

        # Single resident model slot. This may hold the live LoRA training model,
        # a plain base model, or an offline path-backed sampler model.
        self._base_model: nn.Module | None = None
        self._base_tokenizer: Any = None
        self._loaded_base_model_name: str | None = None
        self._base_model_config: dict = {}
        self._model_lock = threading.RLock()
        self._student_namespace_version = 0

    def _clear_sampling_state(self) -> None:
        self.current_loaded_sampler_path = None

    def close(self) -> None:
        self.transcript_cache.close()

    def _ensure_base_model(self, model_name: str | None = None) -> None:
        """Load a clean base model if not already resident."""
        requested_model = model_name or self.config.base_model
        if (
            self._base_model is not None
            and self._loaded_base_model_name == requested_model
            and self.current_loaded_sampler_path is None
        ):
            return

        logger.info("Loading base model: %s", requested_model)
        try:
            model, tokenizer = mlx_load(requested_model)
        except Exception as e:
            logger.error("Failed to load base model %s: %s", requested_model, e)
            raise ValueError(f"Failed to load base model '{requested_model}': {e}") from e
        self._base_model = model
        self._base_tokenizer = tokenizer
        self._loaded_base_model_name = requested_model
        mx.eval(model.parameters())
        self._clear_sampling_state()
        logger.info("Base model loaded")

    def _get_live_model_id(self) -> str | None:
        return next(iter(self.models), None) if self.models else None

    def _assert_teacher_sampling_supported(self, model_id: str) -> None:
        lora_config = self.lora_configs.get(model_id)
        if lora_config is None:
            return
        if lora_config.train_embeddings or lora_config.train_norms or lora_config.use_longlora:
            raise ValueError(
                "Teacher/base sampling requires LoRA-only tuning; "
                "train_embeddings, train_norms, and use_longlora must all be disabled."
            )

    def _cache_settings_fingerprint(self) -> str:
        return (
            f"max_kv={self.config.max_kv_cache_size}:"
            f"kv_bits={self.config.kv_cache_bits}:"
            f"kv_group_size={self.config.kv_cache_group_size}:"
            f"quantized_kv_start={self.config.quantized_kv_start}"
        )

    def _base_namespace(self, base_model: str) -> str:
        return f"base:{base_model}:{self._cache_settings_fingerprint()}"

    def _student_namespace(self, model_id: str) -> str:
        return (
            f"student:{model_id}:v{self._student_namespace_version}:"
            f"{self._cache_settings_fingerprint()}"
        )

    def _path_namespace(self, resolved_path: Path, base_model: str | None) -> str:
        stat_parts: list[str] = []
        for name in ("adapters.safetensors", "model.safetensors", "config.json"):
            candidate = resolved_path / name
            if candidate.exists():
                stat = candidate.stat()
                stat_parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
        digest = hashlib.sha256(
            "|".join(
                [
                    str(resolved_path),
                    base_model or self.config.base_model,
                    self._cache_settings_fingerprint(),
                    *stat_parts,
                ]
            ).encode("utf-8")
        ).hexdigest()
        return f"path:{digest}"

    def _invalidate_student_transcript_caches(self) -> None:
        self._student_namespace_version += 1
        self.transcript_cache.invalidate_namespace_prefix("student:")

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def create_model(self, model_id: str, request: CreateModelInput) -> CreateModelOutput:
        """Create a new QLoRA model from the base model."""
        with self._model_lock:
            logger.info("Creating model %s with LoRA rank=%d", model_id, request.lora_config.rank)
            if self.models:
                raise ValueError("Only one live training model is supported at a time")

            self._ensure_base_model(self.config.base_model)
            model = self._base_model
            tokenizer = self._base_tokenizer

            # Apply QLoRA in-place onto the single resident model.
            model = self.lora_manager.apply_qlora(
                model,
                request.lora_config,
                quantize_bits=self.config.quantize_bits,
                quantize_group_size=self.config.quantize_group_size,
                train_embeddings=request.lora_config.train_embeddings,
                train_norms=request.lora_config.train_norms,
            )
            if request.lora_config.use_longlora:
                enable_longlora_attention(
                    model,
                    group_size_ratio=request.lora_config.longlora_group_size_ratio,
                )
            if self.config.gradient_checkpointing:
                enable_gradient_checkpointing(model)

            self.models[model_id] = model
            self.tokenizers[model_id] = tokenizer
            self.lora_configs[model_id] = request.lora_config
            self._base_model = model
            self._base_tokenizer = tokenizer
            self._loaded_base_model_name = self.config.base_model
            self._clear_sampling_state()
            self._invalidate_student_transcript_caches()

            return CreateModelOutput(
                model_id=model_id,
                base_model=self.config.base_model,
                lora_config=request.lora_config,
            )

    def unload_model(self, model_id: str, _request: UnloadModelInput) -> UnloadModelOutput:
        """Unload a model from memory."""
        with self._model_lock:
            if model_id in self.models:
                del self.models[model_id]
                del self.tokenizers[model_id]
                del self.lora_configs[model_id]
                self.training.accumulated_grads.pop(model_id, None)
                self.training.grad_accum_counts.pop(model_id, None)
                self.training.optimizers.pop(model_id, None)
                self.training.clear_cache(model_id)
                self._base_model = None
                self._base_tokenizer = None
                self._loaded_base_model_name = None
                self._clear_sampling_state()
                self._invalidate_student_transcript_caches()
                logger.info("Unloaded model %s", model_id)

            return UnloadModelOutput(model_id=model_id, status="unloaded")

    def _load_sampling_model(
        self, model_path: str, base_model: str | None
    ) -> tuple[nn.Module, Any]:
        """Load a sampling model from checkpoint.

        Path-backed sampling is reserved for offline/manual evaluation. It is
        not compatible with a live training model under the single-model
        constraint, so we only allow it when no training model is resident.
        """
        if self.models:
            raise ValueError(
                "Path-backed sampling is unavailable while a live training model is resident"
            )

        resolved_path = self._validate_checkpoint_path(model_path)
        cache_key = str(resolved_path)
        if cache_key == self.current_loaded_sampler_path and self._base_model is not None:
            return self._base_model, self._base_tokenizer

        config_path = resolved_path / "config.json"
        config_payload: dict[str, Any] = {}
        if config_path.exists():
            config_payload = json.loads(config_path.read_text())

        resolved_base_model = (
            base_model
            or config_payload.get("base_model")
            or self.config.base_model
        )

        adapter_path = resolved_path / "adapters.safetensors"
        full_weights_path = resolved_path / "model.safetensors"

        if adapter_path.exists():
            lora_cfg = config_payload.get("lora_config")
            if not lora_cfg:
                raise ValueError(f"Missing lora_config in sampler config at {config_path}")

            self._ensure_base_model(resolved_base_model)
            model = self._base_model
            tokenizer = self._base_tokenizer
            model = self.lora_manager.apply_qlora(
                model,
                LoraConfig(**lora_cfg),
                quantize_bits=self.config.quantize_bits,
                quantize_group_size=self.config.quantize_group_size,
            )
            model = self.lora_manager.load_adapter(model, resolved_path)
            self._base_model = model
        elif full_weights_path.exists():
            self._ensure_base_model(resolved_base_model)
            model = self._base_model
            tokenizer = self._base_tokenizer
            weights = mx.load(str(full_weights_path))
            model.load_weights(list(weights.items()), strict=False)
        else:
            raise FileNotFoundError(
                "No sampler weights found in "
                f"{resolved_path}; expected adapters.safetensors or model.safetensors"
            )

        self._loaded_base_model_name = resolved_base_model
        self.current_loaded_sampler_path = cache_key
        return model, tokenizer

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
        with self._model_lock:
            model = self._get_model(model_id)
            model.train()
            return self.training.forward_backward(model_id, model, request)

    def forward_backward_batch(
        self, model_id: str, requests: list[ForwardBackwardInput]
    ) -> list[ForwardBackwardOutput]:
        with self._model_lock:
            model = self._get_model(model_id)
            model.train()
            return self.training.forward_backward_batch(model_id, model, requests)

    def forward(self, model_id: str, request: ForwardInput) -> ForwardOutput:
        with self._model_lock:
            model = self._get_model(model_id)
            model.eval()
            return self.training.forward(model_id, model, request)

    def forward_batch(
        self, model_id: str, requests: list[ForwardInput]
    ) -> list[ForwardOutput]:
        with self._model_lock:
            model = self._get_model(model_id)
            model.eval()
            return self.training.forward_batch(model_id, model, requests)

    def optim_step(self, model_id: str, request: OptimStepInput) -> OptimStepOutput:
        with self._model_lock:
            model = self._get_model(model_id)
            result = self.training.optim_step(model_id, model, request)
            # Any path-backed sampler must reload after weights change.
            self._clear_sampling_state()
            self._invalidate_student_transcript_caches()
            return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_eos_stop_token(request: SampleInput, tokenizer) -> SampleInput:
        """Ensure eos_token_id, common end tokens, and stop_strings are resolved into stop_tokens."""
        return prepare_sample_request(request, tokenizer)

    def _resolve_sampling_target(
        self, model_id: str | None, request: SampleInput
    ) -> tuple[nn.Module, Any, str, str | None]:
        resolved_model_id = model_id or request.model_id
        if request.model_path:
            resolved_path = self._validate_checkpoint_path(request.model_path)
            model, tokenizer = self._load_sampling_model(request.model_path, request.base_model)
            namespace = self._path_namespace(resolved_path, request.base_model)
            return model, tokenizer, "path", namespace

        if resolved_model_id is not None:
            if resolved_model_id in self.models:
                namespace = self._student_namespace(resolved_model_id)
                return (
                    self.models[resolved_model_id],
                    self.tokenizers[resolved_model_id],
                    "student",
                    namespace,
                )
            raise ValueError(f"Model {resolved_model_id} not found. Call create_model first.")

        live_model_id = self._get_live_model_id()
        if live_model_id is not None:
            requested_base = request.base_model or self.config.base_model
            if requested_base != self.config.base_model:
                raise ValueError(
                    "Teacher/base sampling against a different base model is unavailable while "
                    "a live training model is resident"
                )
            self._assert_teacher_sampling_supported(live_model_id)
            return (
                self.models[live_model_id],
                self.tokenizers[live_model_id],
                "teacher",
                self._base_namespace(requested_base),
            )

        self._ensure_base_model(request.base_model or self.config.base_model)
        requested_base = request.base_model or self.config.base_model
        return self._base_model, self._base_tokenizer, "base", self._base_namespace(requested_base)

    def _sample_resolved(
        self,
        model: nn.Module,
        tokenizer: Any,
        request: SampleInput,
        mode: str,
        namespace: str | None,
    ) -> SampleOutput:
        request = self._inject_eos_stop_token(request, tokenizer)
        model.eval()
        context = (
            self.lora_manager.temporarily_disable_lora(model)
            if mode == "teacher"
            else nullcontext()
        )
        with context:
            return self.inference.sample(model, tokenizer, request, namespace=namespace)

    def sample(self, model_id: str | None, request: SampleInput) -> SampleOutput:
        """Generate samples. Uses model_id if provided, else base model."""
        with self._model_lock:
            if request.num_samples > MAX_SAMPLES:
                raise ValueError(
                    f"num_samples={request.num_samples} exceeds maximum of {MAX_SAMPLES}"
                )

            model, tokenizer, mode, namespace = self._resolve_sampling_target(model_id, request)
            return self._sample_resolved(model, tokenizer, request, mode, namespace)

    def sample_batch(self, requests: list[tuple[str | None, SampleInput]]) -> list[SampleOutput]:
        """Generate samples for multiple requests, batching compatible ones."""
        with self._model_lock:
            if not requests:
                return []

            resolved: list[tuple[object, object, SampleInput, str, str | None]] = []
            for model_id, request in requests:
                if request.num_samples > MAX_SAMPLES:
                    raise ValueError(
                        f"num_samples={request.num_samples} exceeds maximum of {MAX_SAMPLES}"
                    )
                model, tokenizer, mode, namespace = self._resolve_sampling_target(model_id, request)
                request = self._inject_eos_stop_token(request, tokenizer)
                model.eval()
                resolved.append((model, tokenizer, request, mode, namespace))

            first_model, first_tokenizer, _first_request, first_mode, first_namespace = resolved[0]
            compatible = all(
                model is first_model
                and tokenizer is first_tokenizer
                and mode == first_mode
                and namespace == first_namespace
                for model, tokenizer, _request, mode, namespace in resolved
            )
            if compatible:
                context = (
                    self.lora_manager.temporarily_disable_lora(first_model)
                    if first_mode == "teacher"
                    else nullcontext()
                )
                with context:
                    return self.inference.sample_batch(
                        first_model,
                        first_tokenizer,
                        [request for _model, _tokenizer, request, _mode, _namespace in resolved],
                        namespace=first_namespace,
                    )

            return [
                self._sample_resolved(model, tokenizer, request, mode, namespace)
                for model, tokenizer, request, mode, namespace in resolved
            ]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _validate_checkpoint_path(self, requested_path: str) -> Path:
        """Resolve a checkpoint path and ensure it stays under checkpoints_base.

        Handles:
        1. `tinker://` virtual URIs (e.g. `tinker://<model_id>/weights/<checkpoint_id>`
           or `tinker://<model_id>/sampler_weights/<checkpoint_id>`)
        2. Bare checkpoint names such as ``step_0016``
        3. Local filesystem paths relative to checkpoints_base or cwd
        """
        if is_tinker_path(requested_path):
            rel_path, _ = tinker_path_to_relative_path(requested_path)
            candidate = Path(rel_path)
        else:
            candidate = Path(requested_path)

        base = self.config.checkpoints_base.resolve()

        candidates = [candidate] if candidate.is_absolute() else [
            (Path.cwd() / candidate),
            (self.config.checkpoints_base / candidate),
        ]

        for path_candidate in candidates:
            resolved = path_candidate.resolve()
            if str(resolved).startswith(str(base) + "/") or resolved == base:
                return resolved

        raise ValueError(
            f"Checkpoint path '{requested_path}' is outside the allowed "
            f"directory '{self.config.checkpoints_base}'"
        )

    def save_weights(self, model_id: str, request: SaveWeightsInput) -> SaveWeightsOutput:
        with self._model_lock:
            model = self._get_model(model_id)
            opt_state = self.training.get_optimizer_state(model_id)
            checkpoint_name = request.path or uuid.uuid4().hex[:8]
            if is_tinker_path(checkpoint_name):
                target_rel_path, _ = tinker_path_to_relative_path(checkpoint_name)
            elif "/" in checkpoint_name:
                target_rel_path = checkpoint_name
            else:
                target_rel_path = f"{model_id}/{checkpoint_name}"

            checkpoint_dir = self._validate_checkpoint_path(target_rel_path)
            lora_config = self.lora_configs.get(model_id)
            meta: dict[str, Any] = {
                "base_model": self.config.base_model,
            }
            if lora_config is not None:
                meta["lora_config"] = lora_config.model_dump()
            save_training_checkpoint(model, opt_state, checkpoint_dir, metadata=meta)

            # Return standard tinker:// URI
            # Extract checkpoint identifier relative to model directory
            base = self.config.checkpoints_base.resolve()
            rel_to_base = checkpoint_dir.relative_to(base).as_posix()
            parts = rel_to_base.split("/", 1)
            ckpt_id = parts[1] if len(parts) > 1 else parts[0]
            tinker_uri = format_tinker_path(model_id, ckpt_id, CheckpointType.TRAINING)
            return SaveWeightsOutput(path=tinker_uri)

    def save_weights_for_sampler(
        self, model_id: str, request: SaveWeightsForSamplerInput
    ) -> SaveWeightsForSamplerOutput:
        with self._model_lock:
            model = self._get_model(model_id)
            if request.path is None:
                self._clear_sampling_state()
                return SaveWeightsForSamplerOutput(
                    path=None,
                    sampling_session_id=request.sampling_session_id,
                )

            checkpoint_name = request.path
            if is_tinker_path(checkpoint_name):
                target_rel_path, _ = tinker_path_to_relative_path(checkpoint_name)
            elif "/" in checkpoint_name:
                target_rel_path = checkpoint_name
            else:
                target_rel_path = f"{model_id}/sampler/{checkpoint_name}"

            safe_path = self._validate_checkpoint_path(target_rel_path)
            lora_config = self.lora_configs.get(model_id)
            if lora_config is None:
                raise ValueError(f"LoRA config for model {model_id} not found")
            save_sampler_weights(
                model,
                safe_path,
                base_model=self.config.base_model,
                lora_config=lora_config.model_dump(),
            )
            self._clear_sampling_state()

            if request.ephemeral:
                out_path = None
            else:
                base = self.config.checkpoints_base.resolve()
                rel_to_base = safe_path.relative_to(base).as_posix()
                parts = rel_to_base.split("/")
                # If rel_to_base is model_id/sampler/export_name
                if len(parts) >= 3 and parts[1] == "sampler":
                    ckpt_id = "/".join(parts[2:])
                elif len(parts) > 1:
                    ckpt_id = "/".join(parts[1:])
                else:
                    ckpt_id = parts[0]
                out_path = format_tinker_path(model_id, ckpt_id, CheckpointType.SAMPLER)

            return SaveWeightsForSamplerOutput(
                path=out_path,
                sampling_session_id=request.sampling_session_id,
            )

    def load_weights(self, model_id: str, request: LoadWeightsInput) -> LoadWeightsOutput:
        with self._model_lock:
            model = self._get_model(model_id)
            if request.path is not None:
                checkpoint_dir = self._validate_checkpoint_path(request.path)
                return_path = request.path if is_tinker_path(request.path) else None
            else:
                if request.source_model_id is None or request.checkpoint_id is None:
                    raise ValueError(
                        "load_weights requires either path or both source_model_id and checkpoint_id"
                    )
                checkpoint_dir = (
                    self.config.checkpoints_base / request.source_model_id / request.checkpoint_id
                )
                checkpoint_dir = self._validate_checkpoint_path(str(checkpoint_dir))
                return_path = format_tinker_path(
                    request.source_model_id,
                    request.checkpoint_id,
                    CheckpointType.TRAINING,
                )
            opt_state = load_training_checkpoint(model, checkpoint_dir)
            if request.optimizer and opt_state is not None:
                self.training.load_optimizer_state(model_id, opt_state)
            self._clear_sampling_state()
            self._invalidate_student_transcript_caches()
            if return_path is None:
                base = self.config.checkpoints_base.resolve()
                rel = checkpoint_dir.relative_to(base).as_posix()
                parts = rel.split("/", 1)
                m_id = parts[0]
                c_id = parts[1] if len(parts) > 1 else "default"
                return_path = format_tinker_path(m_id, c_id, CheckpointType.TRAINING)
            return LoadWeightsOutput(path=return_path)

    def get_weights_info(self, requested_path: str) -> dict[str, Any]:
        """Inspect checkpoint files and metadata to return weights info."""
        checkpoint_dir = self._validate_checkpoint_path(requested_path)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        config_path = checkpoint_dir / "config.json"
        metadata_path = checkpoint_dir / "metadata.json"
        adapter_path = checkpoint_dir / "adapters.safetensors"
        model_path = checkpoint_dir / "model.safetensors"

        base_model = self.config.base_model
        is_lora = False
        lora_rank = None
        train_unembed = None
        train_mlp = None
        train_attn = None

        payload: dict[str, Any] = {}
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text())
            except Exception:
                pass
        elif metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text())
            except Exception:
                pass

        if "base_model" in payload and payload["base_model"]:
            base_model = payload["base_model"]

        lora_config_dict = payload.get("lora_config")
        if isinstance(lora_config_dict, dict):
            is_lora = True
            lora_rank = lora_config_dict.get("rank")
            train_unembed = lora_config_dict.get("train_unembed", False)
            train_mlp = lora_config_dict.get("train_mlp", True)
            train_attn = lora_config_dict.get("train_attn", True)
        elif adapter_path.exists():
            is_lora = True

        return {
            "base_model": base_model,
            "is_lora": is_lora,
            "lora_rank": lora_rank,
            "train_unembed": train_unembed,
            "train_mlp": train_mlp,
            "train_attn": train_attn,
        }
