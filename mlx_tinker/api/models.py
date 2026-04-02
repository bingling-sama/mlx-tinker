"""Request/response models for the Tinker API endpoints.

Wire-compatible with the tinker SDK (v0.16+). Field names, nesting, and
type discriminators match the SDK's Pydantic models exactly so that
`tinker.ServiceClient(base_url=...)` works against this server.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from mlx_tinker.types import (
    AdamParams,
    ForwardBackwardInput,
    LoraConfig,
    ModelInput,
    SamplingParams,
)

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    tags: list[str] = []
    user_metadata: dict | None = None
    sdk_version: str = "0.1.0"
    project_id: str | None = None
    type: Literal["create_session"] = "create_session"


class CreateSessionResponse(BaseModel):
    session_id: str
    type: Literal["create_session"] = "create_session"
    info_message: str | None = None
    warning_message: str | None = None
    error_message: str | None = None


class SessionHeartbeatRequest(BaseModel):
    session_id: str


class SessionHeartbeatResponse(BaseModel):
    pass


class CreateSamplingSessionRequest(BaseModel):
    session_id: str
    sampling_session_seq_id: int = 0
    base_model: str | None = None
    model_path: str | None = None
    type: Literal["create_sampling_session"] = "create_sampling_session"


class CreateSamplingSessionResponse(BaseModel):
    sampling_session_id: str
    type: Literal["create_sampling_session"] = "create_sampling_session"


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------


class CreateModelRequest(BaseModel):
    session_id: str
    model_seq_id: int = 0
    base_model: str | None = None
    lora_config: LoraConfig | None = None
    user_metadata: dict | None = None
    type: Literal["create_model"] = "create_model"


class CreateModelResponse(BaseModel):
    model_id: str
    type: Literal["create_model"] = "create_model"


class UnloadModelRequest(BaseModel):
    model_id: str


class UnloadModelResponse(BaseModel):
    model_id: str
    type: Literal["unload_model"] = "unload_model"


class GetInfoRequest(BaseModel):
    model_id: str


class ModelData(BaseModel):
    arch: str | None = None
    model_name: str | None = None
    tokenizer_id: str | None = None


class GetInfoResponse(BaseModel):
    model_id: str
    model_data: ModelData
    is_lora: bool | None = None
    lora_rank: int | None = None
    model_name: str | None = None
    type: Literal["get_info"] | None = None


# ---------------------------------------------------------------------------
# Training operations — SDK nests data inside forward_backward_input /
# forward_input keys.
# ---------------------------------------------------------------------------


class ForwardBackwardRequest(BaseModel):
    forward_backward_input: ForwardBackwardInput
    model_id: str
    seq_id: int | None = None


class ForwardRequest(BaseModel):
    forward_input: ForwardBackwardInput
    model_id: str
    seq_id: int | None = None


class OptimStepRequest(BaseModel):
    adam_params: AdamParams
    model_id: str
    seq_id: int | None = None
    type: Literal["optim_step"] = "optim_step"


class OptimStepResponse(BaseModel):
    metrics: dict[str, float] | None = None


class ForwardBackwardOutputWire(BaseModel):
    """ForwardBackwardOutput as returned via retrieve_future."""
    loss_fn_output_type: str
    loss_fn_outputs: list[dict[str, Any]]
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Weight management
# ---------------------------------------------------------------------------


class SaveWeightsRequest(BaseModel):
    model_id: str
    path: str
    seq_id: int | None = None
    type: Literal["save_weights"] = "save_weights"


class SaveWeightsResponse(BaseModel):
    path: str | None = None
    type: Literal["save_weights"] | None = None


class SaveWeightsForSamplerRequest(BaseModel):
    model_id: str
    path: str | None = None
    sampling_session_seq_id: int | None = None
    seq_id: int | None = None
    ttl_seconds: int | None = None
    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"


class SaveWeightsForSamplerResponse(BaseModel):
    path: str | None = None
    sampling_session_id: str | None = None
    type: Literal["save_weights_for_sampler"] | None = None


class LoadWeightsRequest(BaseModel):
    model_id: str
    source_model_id: str | None = None
    checkpoint_id: str | None = None
    path: str | None = None
    optimizer: bool = False
    seq_id: int | None = None
    type: Literal["load_weights"] = "load_weights"


class LoadWeightsResponse(BaseModel):
    type: Literal["load_weights"] | None = None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class SampleRequest(BaseModel):
    model_id: str | None = None
    num_samples: int = 1
    prompt: ModelInput
    sampling_params: SamplingParams
    base_model: str | None = None
    model_path: str | None = None
    sampling_session_id: str | None = None
    seq_id: int | None = None
    prompt_logprobs: bool | None = None
    topk_prompt_logprobs: int = 0
    type: Literal["sample"] = "sample"


class SampledSequenceWire(BaseModel):
    stop_reason: str
    tokens: list[int]
    logprobs: list[float] | None = None


class SampleResponse(BaseModel):
    sequences: list[SampledSequenceWire]
    type: Literal["sample"] = "sample"
    prompt_logprobs: list[float | None] | None = None
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None = None


# ---------------------------------------------------------------------------
# Futures — SDK protocol
# ---------------------------------------------------------------------------


class UntypedAPIFuture(BaseModel):
    """Returned by submit endpoints. SDK polls retrieve_future with request_id."""
    request_id: str
    model_id: str | None = None


class RetrieveFutureRequest(BaseModel):
    request_id: str
    allow_metadata_only: bool = False


class TryAgainResponse(BaseModel):
    type: Literal["try_again"] = "try_again"
    request_id: str
    queue_state: Literal["active", "paused_capacity", "paused_rate_limit"] = "active"


class RequestFailedResponse(BaseModel):
    error: str
    category: str | None = None


# ---------------------------------------------------------------------------
# Health / capabilities
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"


class SupportedModel(BaseModel):
    model_name: str
    base_model: str | None = None


class GetServerCapabilitiesResponse(BaseModel):
    supported_models: list[SupportedModel]


class WeightsInfoRequest(BaseModel):
    model_path: str


class WeightsInfoResponse(BaseModel):
    base_model: str
    is_lora: bool
    lora_rank: int | None = None


class TelemetryRequest(BaseModel):
    event: str | None = None
    events: list[dict] | None = None
    data: dict = {}
    platform: str | None = None
    sdk_version: str | None = None
    session_id: str | None = None


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


# ---------------------------------------------------------------------------
# LoRA catalog / UI
# ---------------------------------------------------------------------------


class LoraStats(BaseModel):
    forward_backward_count: int = 0
    optim_step_count: int = 0
    sample_count: int = 0
    sampler_export_count: int = 0
    last_loss: float | None = None
    avg_loss: float | None = None
    total_tokens: float = 0.0
    last_grad_norm: float | None = None
    avg_grad_norm: float | None = None
    last_activity_at: str | None = None


class LoraCatalogItem(BaseModel):
    id: str
    relative_path: str | None = None
    display_name: str
    openai_model_id: str
    base_model: str
    created_at: str | None = None
    size_bytes: int = 0
    lora_config: dict[str, Any] = Field(default_factory=dict)
    is_live: bool = False
    is_exported: bool = False
    downloadable: bool = False
    status: str
    stats: LoraStats = Field(default_factory=LoraStats)


class LoraCatalogSummary(BaseModel):
    total_exported_loras: int = 0
    live_loras: int = 0
    unique_base_models: int = 0
    total_adapter_disk_usage_bytes: int = 0
    total_optim_steps: int = 0
    last_activity_at: str | None = None


class LoraCatalogResponse(BaseModel):
    summary: LoraCatalogSummary
    items: list[LoraCatalogItem]


class LoraArtifactFile(BaseModel):
    name: str
    size_bytes: int


class LoraSessionInfo(BaseModel):
    session_id: str
    status: str
    created_at: str
    last_heartbeat_at: str | None = None
    heartbeat_count: int = 0


class LoraSamplingSessionInfo(BaseModel):
    sampling_session_id: str
    created_at: str
    model_id: str | None = None
    base_model: str | None = None
    model_path: str | None = None


class LoraRecentFuture(BaseModel):
    request_id: int
    request_type: str
    status: str
    created_at: str
    completed_at: str | None = None


class LoraDetailResponse(BaseModel):
    item: LoraCatalogItem
    session: LoraSessionInfo | None = None
    sampling_sessions: list[LoraSamplingSessionInfo] = Field(default_factory=list)
    recent_futures: list[LoraRecentFuture] = Field(default_factory=list)
    files: list[LoraArtifactFile] = Field(default_factory=list)
