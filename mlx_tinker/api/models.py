"""Request/response models for the Tinker API endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from mlx_tinker.types import (
    AdamParams,
    Datum,
    LoraConfig,
    ModelInput,
    SamplingParams,
)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    tags: list[str] = []
    user_metadata: dict = {}
    sdk_version: str = "0.1.0"


class CreateSessionResponse(BaseModel):
    session_id: str
    info_message: str | None = None
    warning_message: str | None = None
    error_message: str | None = None


class SessionHeartbeatRequest(BaseModel):
    session_id: str


class SessionHeartbeatResponse(BaseModel):
    pass


class CreateSamplingSessionRequest(BaseModel):
    session_id: str
    base_model: str | None = None
    model_path: str | None = None


class CreateSamplingSessionResponse(BaseModel):
    sampling_session_id: str


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------


class CreateModelRequest(BaseModel):
    session_id: str
    base_model: str | None = None
    lora_config: LoraConfig


class CreateModelResponse(BaseModel):
    model_id: str
    base_model: str
    lora_config: LoraConfig | None = None
    status: str = "created"
    request_id: str = ""


class UnloadModelRequest(BaseModel):
    model_id: str


class UnloadModelResponse(BaseModel):
    request_id: str
    model_id: str


class GetInfoRequest(BaseModel):
    model_id: str


class ModelData(BaseModel):
    base_model: str
    lora_config: dict | None = None
    status: str


class ModelInfoResponse(BaseModel):
    model_id: str
    status: str
    model_data: ModelData


# ---------------------------------------------------------------------------
# Training operations
# ---------------------------------------------------------------------------


class ForwardBackwardRequest(BaseModel):
    model_id: str
    data: list[Datum]
    loss_fn: Literal["cross_entropy", "importance_sampling", "ppo", "cispo"]
    loss_fn_config: dict[str, float] | None = None


class ForwardRequest(BaseModel):
    model_id: str
    data: list[Datum]


class OptimStepRequest(BaseModel):
    model_id: str
    adam_params: AdamParams


# ---------------------------------------------------------------------------
# Weight management
# ---------------------------------------------------------------------------


class SaveWeightsRequest(BaseModel):
    model_id: str
    path: str


class SaveWeightsForSamplerRequest(BaseModel):
    model_id: str
    path: str | None = None
    sampling_session_seq_id: int | None = None
    seq_id: int | None = None
    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"


class LoadWeightsRequest(BaseModel):
    model_id: str
    source_model_id: str
    checkpoint_id: str


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class SampleRequest(BaseModel):
    num_samples: int = 1
    prompt: ModelInput
    sampling_params: SamplingParams
    base_model: str | None = None
    model_path: str | None = None
    sampling_session_id: str | None = None
    seq_id: int | None = None
    prompt_logprobs: bool | None = None
    topk_prompt_logprobs: int = 0


# ---------------------------------------------------------------------------
# Futures
# ---------------------------------------------------------------------------


class FutureResponse(BaseModel):
    future_id: str
    status: str = "pending"
    request_id: str = ""


class RetrieveFutureRequest(BaseModel):
    future_id: str


class RetrieveFutureResponse(BaseModel):
    status: str
    result: Any | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Health / capabilities
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"


class SupportedModel(BaseModel):
    name: str
    base_model: str


class GetServerCapabilitiesResponse(BaseModel):
    supported_models: list[SupportedModel]


class WeightsInfoRequest(BaseModel):
    model_path: str


class WeightsInfoResponse(BaseModel):
    base_model: str
    is_lora: bool
    lora_rank: int | None = None


class TelemetryRequest(BaseModel):
    event: str
    data: dict = {}


class TelemetryResponse(BaseModel):
    status: str = "accepted"


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


class Checkpoint(BaseModel):
    checkpoint_id: str
    checkpoint_type: str
    status: str
    created_at: str


class ListCheckpointsResponse(BaseModel):
    checkpoints: list[Checkpoint]


# ---------------------------------------------------------------------------
# Training runs
# ---------------------------------------------------------------------------


class TrainingRun(BaseModel):
    model_id: str
    base_model: str
    status: str
    created_at: str


class Cursor(BaseModel):
    offset: int
    limit: int


class TrainingRunsResponse(BaseModel):
    training_runs: list[TrainingRun]
    cursor: Cursor
