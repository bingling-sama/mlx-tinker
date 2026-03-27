"""Tinker API-compatible types mirroring SkyRL-tx."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator

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


class LoraConfig(BaseModel):
    rank: int
    alpha: float
    seed: int = 42
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = False


class SamplingParams(BaseModel):
    temperature: float = 1.0
    max_tokens: int = 256
    seed: int = 0
    stop_tokens: list[int] | None = None
    stop_strings: list[str] | None = None
    top_k: int = -1
    top_p: float = 1.0


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


ModelInputChunk = Annotated[
    EncodedTextChunk | ImageAssetPointerChunk | ImageChunk,
    Discriminator("type"),
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


class LossFnInputs(BaseModel):
    target_tokens: TensorData
    weights: TensorData
    advantages: TensorData
    logprobs: TensorData


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
    loss_fn: Literal["cross_entropy", "importance_sampling", "ppo", "cispo"]
    loss_fn_config: dict[str, float] | None = None


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


class SaveWeightsForSamplerOutput(BaseModel):
    path: str | None = None
    type: str = "save_weights_for_sampler"
    sampling_session_id: str | None = None


class SaveWeightsInput(BaseModel):
    path: str


class SaveWeightsOutput(BaseModel):
    path: str
    type: str = "save_weights"


class LoadWeightsInput(BaseModel):
    source_model_id: str
    checkpoint_id: str


class LoadWeightsOutput(BaseModel):
    type: str = "load_weights"


# ---------------------------------------------------------------------------
# Sampling / inference types
# ---------------------------------------------------------------------------


class GeneratedSequence(BaseModel):
    stop_reason: Literal["length", "stop"]
    tokens: list[int]
    logprobs: list[float]


class SampleInput(BaseModel):
    base_model: str | None = None
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
