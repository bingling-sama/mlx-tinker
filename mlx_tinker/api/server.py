"""FastAPI server implementing the Tinker API endpoints.

Wire-compatible with the tinker SDK: submit endpoints return UntypedAPIFuture,
retrieve_future returns the raw typed result or TryAgainResponse.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from mlx_tinker.api.models import (
    CreateModelRequest,
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    ForwardBackwardRequest,
    ForwardRequest,
    GetInfoRequest,
    GetInfoResponse,
    GetServerCapabilitiesResponse,
    HealthResponse,
    LoadWeightsRequest,
    ModelData,
    OptimStepRequest,
    RetrieveFutureRequest,
    SampleRequest,
    SaveWeightsForSamplerRequest,
    SaveWeightsRequest,
    SessionHeartbeatRequest,
    SessionHeartbeatResponse,
    SupportedModel,
    TelemetryRequest,
    TelemetryResponse,
    TryAgainResponse,
    UnloadModelRequest,
    UntypedAPIFuture,
)
from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.config import EngineConfig
from mlx_tinker.db.database import close_db, get_session, init_db
from mlx_tinker.db.models import FutureDB, ModelDB, SamplingSessionDB, SessionDB
from mlx_tinker.engine.engine import TinkerEngine
from mlx_tinker.types import RequestStatus, RequestType

logger = logging.getLogger(__name__)

# Global references set during lifespan
_engine: TinkerEngine | None = None
_backend: MLXBackend | None = None
_config: EngineConfig | None = None


def create_app(config: EngineConfig | None = None) -> FastAPI:
    """Create the FastAPI application with the Tinker API routes."""
    if config is None:
        config = EngineConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _engine, _backend, _config
        _config = config

        # Init DB
        await init_db(config.database_path)
        logger.info("Database initialized at %s", config.database_path)

        # Init backend
        _backend = MLXBackend(config)

        # Warm up: eagerly load the base model
        _backend._ensure_base_model()

        # Start engine
        _engine = TinkerEngine(config, _backend)
        await _engine.start()

        # Register OpenAI-compatible routes now that backend is ready
        from mlx_tinker.api.openai_compat import register_openai_routes

        register_openai_routes(app, _backend)
        logger.info("OpenAI-compatible routes registered")

        yield

        # Shutdown
        if _engine is not None:
            await _engine.stop()
        await close_db()

    app = FastAPI(title="MLX-Tinker", version="0.1.0", lifespan=lifespan)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    """Register all Tinker API routes on the app."""

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/api/v1/healthz")
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/api/v1/get_server_capabilities")
    async def get_server_capabilities() -> GetServerCapabilitiesResponse:
        return GetServerCapabilitiesResponse(
            supported_models=[
                SupportedModel(
                    model_name=_config.base_model,
                    base_model=_config.base_model,
                )
            ]
        )

    @app.get("/")
    async def root():
        return {
            "service": "mlx-tinker",
            "version": "0.1.0",
            "backend": "mlx",
            "base_model": _config.base_model if _config else "unknown",
        }

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    @app.post("/api/v1/create_session")
    async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
        session_id = str(uuid.uuid4())
        async with get_session() as db:
            session = SessionDB(
                session_id=session_id,
                tags=request.tags,
                user_metadata=request.user_metadata or {},
                sdk_version=request.sdk_version,
            )
            db.add(session)
            await db.commit()
        return CreateSessionResponse(session_id=session_id)

    @app.post("/api/v1/session_heartbeat")
    async def session_heartbeat(request: SessionHeartbeatRequest) -> SessionHeartbeatResponse:
        async with get_session() as db:
            session = await db.get(SessionDB, request.session_id)
            if session:
                session.last_heartbeat_at = datetime.now(timezone.utc)
                session.heartbeat_count += 1
                db.add(session)
                await db.commit()
        return SessionHeartbeatResponse()

    @app.post("/api/v1/create_sampling_session")
    async def create_sampling_session(
        request: CreateSamplingSessionRequest,
    ) -> CreateSamplingSessionResponse:
        ss_id = str(uuid.uuid4())
        async with get_session() as db:
            ss = SamplingSessionDB(
                sampling_session_id=ss_id,
                session_id=request.session_id,
                base_model=request.base_model,
                model_path=request.model_path,
            )
            db.add(ss)
            await db.commit()
        return CreateSamplingSessionResponse(sampling_session_id=ss_id)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    @app.post("/api/v1/create_model")
    async def create_model(request: CreateModelRequest) -> UntypedAPIFuture:
        model_id = str(uuid.uuid4())
        lora_config_data = request.lora_config.model_dump() if request.lora_config else {}

        async with get_session() as db:
            future = FutureDB(
                request_type=RequestType.CREATE_MODEL,
                model_id=model_id,
                request_data={"lora_config": lora_config_data},
            )
            db.add(future)
            await db.commit()
            await db.refresh(future)

            model_db = ModelDB(
                model_id=model_id,
                base_model=request.base_model or _config.base_model,
                lora_config=lora_config_data,
                status="creating",
                request_id=future.request_id,
                session_id=request.session_id,
            )
            db.add(model_db)
            await db.commit()

        return UntypedAPIFuture(
            request_id=str(future.request_id),
            model_id=model_id,
        )

    @app.post("/api/v1/unload_model")
    async def unload_model(request: UnloadModelRequest) -> UntypedAPIFuture:
        async with get_session() as db:
            future = FutureDB(
                request_type=RequestType.UNLOAD_MODEL,
                model_id=request.model_id,
                request_data={},
            )
            db.add(future)
            await db.commit()
            await db.refresh(future)

        return UntypedAPIFuture(
            request_id=str(future.request_id),
            model_id=request.model_id,
        )

    @app.post("/api/v1/get_info")
    async def get_info(request: GetInfoRequest) -> GetInfoResponse:
        async with get_session() as db:
            model = await db.get(ModelDB, request.model_id)
            if model is None:
                raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

        lora_config = model.lora_config or {}
        return GetInfoResponse(
            model_id=model.model_id,
            model_data=ModelData(
                arch=None,
                model_name=model.base_model,
                tokenizer_id=model.base_model,
            ),
            is_lora=True,
            lora_rank=lora_config.get("rank"),
            model_name=model.base_model,
            type="get_info",
        )

    # ------------------------------------------------------------------
    # Training operations — return UntypedAPIFuture
    # ------------------------------------------------------------------

    @app.post("/api/v1/forward_backward")
    async def forward_backward(request: ForwardBackwardRequest) -> UntypedAPIFuture:
        fbi = request.forward_backward_input
        return await _create_future(
            RequestType.FORWARD_BACKWARD,
            request.model_id,
            {
                "data": [d.model_dump() for d in fbi.data],
                "loss_fn": fbi.loss_fn,
                "loss_fn_config": fbi.loss_fn_config,
            },
        )

    @app.post("/api/v1/forward")
    async def forward(request: ForwardRequest) -> UntypedAPIFuture:
        fi = request.forward_input
        return await _create_future(
            RequestType.FORWARD,
            request.model_id,
            {"data": [d.model_dump() for d in fi.data]},
        )

    @app.post("/api/v1/optim_step")
    async def optim_step(request: OptimStepRequest) -> UntypedAPIFuture:
        return await _create_future(
            RequestType.OPTIM_STEP,
            request.model_id,
            {"adam_params": request.adam_params.model_dump()},
        )

    # ------------------------------------------------------------------
    # Weight management — return UntypedAPIFuture
    # ------------------------------------------------------------------

    @app.post("/api/v1/save_weights")
    async def save_weights(request: SaveWeightsRequest) -> UntypedAPIFuture:
        return await _create_future(
            RequestType.SAVE_WEIGHTS,
            request.model_id,
            {"path": request.path},
        )

    @app.post("/api/v1/save_weights_for_sampler")
    async def save_weights_for_sampler(request: SaveWeightsForSamplerRequest) -> UntypedAPIFuture:
        request_payload = {
            "path": request.path,
            "sampling_session_seq_id": request.sampling_session_seq_id,
            "seq_id": request.seq_id,
            "ephemeral": False,
        }
        if request.path is None:
            sampling_session_id = str(uuid.uuid4())
            if _config is None:
                raise RuntimeError("Server config is not initialized")
            sampler_dir = _config.checkpoints_base / request.model_id / "sampler" / sampling_session_id
            async with get_session() as db:
                model = await db.get(ModelDB, request.model_id)
                if model is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model {request.model_id} not found",
                    )
                sampling_session = SamplingSessionDB(
                    sampling_session_id=sampling_session_id,
                    session_id=model.session_id,
                    base_model=model.base_model,
                    model_path=str(sampler_dir),
                )
                db.add(sampling_session)
                await db.commit()
            request_payload["path"] = str(sampler_dir)
            request_payload["sampling_session_id"] = sampling_session_id
            request_payload["ephemeral"] = True

        return await _create_future(
            RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
            request.model_id,
            request_payload,
        )

    @app.post("/api/v1/load_weights")
    async def load_weights(request: LoadWeightsRequest) -> UntypedAPIFuture:
        return await _create_future(
            RequestType.LOAD_WEIGHTS,
            request.model_id,
            {
                "source_model_id": request.source_model_id,
                "checkpoint_id": request.checkpoint_id,
            },
        )

    # ------------------------------------------------------------------
    # Sampling — returns UntypedAPIFuture
    # ------------------------------------------------------------------

    @app.post("/api/v1/asample")
    async def asample(request: SampleRequest) -> UntypedAPIFuture:
        resolved_model_id, resolved_request = await _resolve_sampling_request(request)
        return await _create_future(
            RequestType.SAMPLE,
            resolved_model_id,
            resolved_request,
        )

    # ------------------------------------------------------------------
    # Futures — SDK protocol: return raw typed result or TryAgainResponse
    # ------------------------------------------------------------------

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(request: RetrieveFutureRequest):
        deadline = asyncio.get_running_loop().time() + 1.0
        future = None
        while True:
            async with get_session() as db:
                future = await db.get(FutureDB, int(request.request_id))
                if future is None:
                    raise HTTPException(
                        status_code=404, detail=f"Future {request.request_id} not found"
                    )
            if future.status != RequestStatus.PENDING:
                break
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(0.1)

        if future.status == RequestStatus.PENDING:
            return TryAgainResponse(
                request_id=request.request_id,
                queue_state="active",
            )
        elif future.status == RequestStatus.FAILED:
            error = (
                future.result_data.get("error", "Unknown error")
                if future.result_data
                else "Unknown error"
            )
            return JSONResponse(
                status_code=200,
                content={"error": error, "category": "execution_error"},
            )
        else:
            return JSONResponse(
                status_code=200,
                content=future.result_data or {},
            )

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    @app.post("/api/v1/telemetry")
    async def telemetry(request: TelemetryRequest) -> TelemetryResponse:
        return TelemetryResponse(status="accepted")


async def _create_future(
    request_type: RequestType,
    model_id: str | None,
    request_data: dict,
) -> UntypedAPIFuture:
    """Helper to insert a future into the DB and return an UntypedAPIFuture."""
    async with get_session() as db:
        future = FutureDB(
            request_type=request_type,
            model_id=model_id,
            request_data=request_data,
        )
        db.add(future)
        await db.commit()
        await db.refresh(future)

    return UntypedAPIFuture(
        request_id=str(future.request_id),
        model_id=model_id,
    )


async def _resolve_sampling_request(
    request: SampleRequest,
) -> tuple[str | None, dict]:
    """Resolve a sample request into concrete backend request data."""
    resolved_model_id = request.model_id
    # Preserve SDK semantics for omitted optional fields. In particular,
    # prompt_logprobs should fall back to the backend default `False`, not
    # serialize as `null` and fail SampleInput validation later.
    resolved_request = request.model_dump(exclude_none=True)

    if request.sampling_session_id:
        async with get_session() as db:
            sampling_session = await db.get(SamplingSessionDB, request.sampling_session_id)
            if sampling_session is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Sampling session {request.sampling_session_id} not found",
                )

        resolved_request["model_path"] = sampling_session.model_path
        resolved_request["base_model"] = sampling_session.base_model or resolved_request.get(
            "base_model"
        )
        # Sampling sessions refer to exported weights, not an in-memory training model.
        resolved_model_id = None

    return resolved_model_id, resolved_request
