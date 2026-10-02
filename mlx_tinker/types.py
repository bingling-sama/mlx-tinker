"""Tinker API-compatible types mirroring SkyRL-tx."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Tag, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RequestType(str, Enum):
    CREATE_MODEL = "create_model"
    FORWARD_BACKWARD = "forward_backward"
    FORWARD = "forward"
    OPTIM_STEP = "optim_step"
    SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
    SAVE_WEIGHTS = "save_weights"
    LOAD_WEIGHTS = "load_weights"
    SAMPLE = "sample"
    UNLOAD_MODEL = "unload_model"
    EXTERNAL = "external"


class CheckpointType(str, Enum):
    TRAINING = "training"
    SAMPLER = "sampler"


class RequestStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class CheckpointStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Loss type registry
# ---------------------------------------------------------------------------

LOSS_TYPES: dict[str, int] = {
    "cross_entropy": 0,
    "importance_sampling": 1,
    "ppo": 2,
    "cispo": 3,
    "dro": 4,
}


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class AdamParams(BaseModel):
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0  # 0 = disabled


class LoraConfig(BaseModel):
    rank: int
    alpha: float = 16.0
    seed: int | None = 42
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = False
    train_embeddings: bool = False
    train_norms: bool = False
    use_longlora: bool = False
    longlora_group_size_ratio: float = 0.25


class SamplingParams(BaseModel):
    temperature: float = 1.0
    max_tokens: int = 256
    seed: int = 0
    stop_tokens: list[int] | None = None
    stop_strings: list[str] | None = None
    stop: str | list[str | int] | int | None = None
    top_k: int = -1
    top_p: float = 1.0

    @model_validator(mode="before")
    @classmethod
    def _normalize_stop(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        stop = data.get("stop")
        if stop is not None:
            stop_strings = list(data.get("stop_strings") or [])
            stop_tokens = list(data.get("stop_tokens") or [])
            if isinstance(stop, str):
                if stop and stop not in stop_strings:
                    stop_strings.append(stop)
            elif isinstance(stop, (list, tuple)):
                for item in stop:
                    if isinstance(item, str):
                        if item and item not in stop_strings:
                            stop_strings.append(item)
                    elif isinstance(item, int):
                        if item not in stop_tokens:
                            stop_tokens.append(item)
            elif isinstance(stop, int):
                if stop not in stop_tokens:
                    stop_tokens.append(stop)
            if stop_strings:
                data["stop_strings"] = stop_strings
            if stop_tokens:
                data["stop_tokens"] = stop_tokens
        return data


class LossFnConfig(BaseModel):
    clip_low_threshold: float = 0.0
    clip_high_threshold: float = float("inf")


# ---------------------------------------------------------------------------
# Model input types
# ---------------------------------------------------------------------------


class EncodedTextChunk(BaseModel):
    type: Literal["encoded_text"] = "encoded_text"
    tokens: list[int]


class ImageChunk(BaseModel):
    type: Literal["image"] = "image"
    data: bytes
    format: Literal["png", "jpeg"]
    expected_tokens: int | None = None


class ImageAssetPointerChunk(BaseModel):
    type: Literal["image_asset_pointer"] = "image_asset_pointer"
    format: Literal["png", "jpeg"]
    location: str
    expected_tokens: int | None = None


def _chunk_discriminator(v):
    """Discriminate chunk type, defaulting to encoded_text when type is missing."""
    if isinstance(v, dict):
        return v.get("type", "encoded_text")
    return getattr(v, "type", "encoded_text")


ModelInputChunk = Annotated[
    Annotated[EncodedTextChunk, Tag("encoded_text")]
    | Annotated[ImageAssetPointerChunk, Tag("image_asset_pointer")]
    | Annotated[ImageChunk, Tag("image")],
    Discriminator(_chunk_discriminator),
]


class ModelInput(BaseModel):
    chunks: list[ModelInputChunk]

    @classmethod
    def from_tokens(cls, tokens: list[int]) -> ModelInput:
        return cls(chunks=[EncodedTextChunk(tokens=tokens)])

    def get_tokens(self) -> list[int]:
        tokens: list[int] = []
        for chunk in self.chunks:
            if isinstance(chunk, EncodedTextChunk):
                tokens.extend(chunk.tokens)
        return tokens


class TensorData(BaseModel):
    data: list[int] | list[float]
    dtype: Literal["float32", "int64"] | None = None
    shape: list[int] | None = None


class LossFnInputs(BaseModel):
    target_tokens: TensorData
    weights: TensorData | None = None
    advantages: TensorData | None = None
    logprobs: TensorData | None = None


class Datum(BaseModel):
    loss_fn_inputs: LossFnInputs
    model_input: ModelInput


# ---------------------------------------------------------------------------
# Training operation types
# ---------------------------------------------------------------------------


class CreateModelInput(BaseModel):
    lora_config: LoraConfig


class CreateModelOutput(BaseModel):
    model_id: str
    base_model: str
    lora_config: LoraConfig


class UnloadModelInput(BaseModel):
    pass


class UnloadModelOutput(BaseModel):
    model_id: str
    status: str
    type: str = "unload_model"


class ForwardBackwardInput(BaseModel):
    data: list[Datum]
    loss_fn: Literal["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"]
    loss_fn_config: dict[str, float] | None = None
    forward_only: bool = False


class ForwardBackwardOutput(BaseModel):
    loss_fn_output_type: str
    loss_fn_outputs: list[dict]
    metrics: dict


class ForwardInput(BaseModel):
    data: list[Datum]


class ForwardOutput(BaseModel):
    logprobs: list[list[float]]
    metrics: dict


class OptimStepInput(BaseModel):
    adam_params: AdamParams


class OptimStepOutput(BaseModel):
    metrics: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Weight management types
# ---------------------------------------------------------------------------


class SaveWeightsForSamplerInput(BaseModel):
    path: str | None = None
    sampling_session_seq_id: int | None = None
    seq_id: int | None = None
    sampling_session_id: str | None = None
    ephemeral: bool = False
    ttl_seconds: int | None = None


class SaveWeightsForSamplerOutput(BaseModel):
    path: str | None = None
    type: str = "save_weights_for_sampler"
    sampling_session_id: str | None = None


class SaveWeightsInput(BaseModel):
    path: str | None = None
    ttl_seconds: int | None = None


class SaveWeightsOutput(BaseModel):
    path: str
    type: str = "save_weights"


class LoadWeightsInput(BaseModel):
    source_model_id: str | None = None
    checkpoint_id: str | None = None
    path: str | None = None
    optimizer: bool = False


class LoadWeightsOutput(BaseModel):
    path: str | None = None
    type: str = "load_weights"


# ---------------------------------------------------------------------------
# Sampling / inference types
# ---------------------------------------------------------------------------


class GeneratedSequence(BaseModel):
    stop_reason: Literal["length", "stop"]
    tokens: list[int]
    logprobs: list[float]


class SampleInput(BaseModel):
    model_id: str | None = None
    base_model: str | None = None
    model_path: str | None = None
    sampling_session_id: str | None = None
    prompt: ModelInput
    sampling_params: SamplingParams
    num_samples: int = 1
    checkpoint_id: str = "latest"
    prompt_logprobs: bool = False


class SampleOutput(BaseModel):
    sequences: list[GeneratedSequence]
    prompt_logprobs: list[float] | None = None


# ---------------------------------------------------------------------------
# Internal batch types (engine -> backend)
# ---------------------------------------------------------------------------


class PreparedModelPassBatch(BaseModel):
    all_model_inputs: list[ModelInput]
    all_targets: list[list[int]]
    all_token_weights: list[list[float]]
    all_sampling_logprobs: list[list[float]]
    all_advantages: list[list[float]]
    all_model_ids: list[str]
    all_loss_fns: list[str]
    all_loss_fn_configs: list[dict[str, float] | None]
    request_batch_slices: list[tuple[str, str, int, int]]


class PreparedSampleBatch(BaseModel):
    all_model_inputs: list[ModelInput]
    all_sampling_params: list[SamplingParams]
    all_model_ids: list[str]
    all_checkpoint_ids: list[str]
    all_checkpoint_paths: list[str]
    needs_prompt_logprobs: bool = False
    request_batch_slices: list[tuple[str, str, int, int, bool]]


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class EngineMetrics(BaseModel):
    train_seq_len_jit_times: dict[int, float] = {}
    sample_seq_len_jit_times: dict[int, float] = {}


class ErrorResponse(BaseModel):
    error: str
    status: str = "error"


class TinkerPath(BaseModel):
    primary_id: str
    kind: str
    secondary_id: str
