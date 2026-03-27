"""FastAPI server implementing the Tinker API endpoints."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from mlx_tinker.api.models import (
    CreateModelRequest,
    CreateModelResponse,
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    ForwardBackwardRequest,
    ForwardRequest,
    FutureResponse,
    GetInfoRequest,
    GetServerCapabilitiesResponse,
    HealthResponse,
    LoadWeightsRequest,
    ModelData,
    ModelInfoResponse,
    OptimStepRequest,
    RetrieveFutureRequest,
    RetrieveFutureResponse,
    SampleRequest,
    SaveWeightsForSamplerRequest,
    SaveWeightsRequest,
    SessionHeartbeatRequest,
    SessionHeartbeatResponse,
    SupportedModel,
    TelemetryRequest,
    TelemetryResponse,
    UnloadModelRequest,
    UnloadModelResponse,
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

        # Start engine
        _engine = TinkerEngine(config, _backend)
        await _engine.start()

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
                SupportedModel(name=_config.base_model, base_model=_config.base_model)
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
                user_metadata=request.user_metadata,
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
    async def create_model(request: CreateModelRequest) -> CreateModelResponse:
        model_id = str(uuid.uuid4())

        # Create future for engine processing
        async with get_session() as db:
            future = FutureDB(
                request_type=RequestType.CREATE_MODEL,
                model_id=model_id,
                request_data={"lora_config": request.lora_config.model_dump()},
            )
            db.add(future)
            await db.commit()
            await db.refresh(future)

            # Also register model in DB
            model_db = ModelDB(
                model_id=model_id,
                base_model=request.base_model or _config.base_model,
                lora_config=request.lora_config.model_dump(),
                status="creating",
                request_id=future.request_id,
                session_id=request.session_id,
            )
            db.add(model_db)
            await db.commit()

        return CreateModelResponse(
            model_id=model_id,
            base_model=request.base_model or _config.base_model,
            lora_config=request.lora_config,
            status="creating",
            request_id=str(future.request_id),
        )

    @app.post("/api/v1/unload_model")
    async def unload_model(request: UnloadModelRequest) -> UnloadModelResponse:
        async with get_session() as db:
            future = FutureDB(
                request_type=RequestType.UNLOAD_MODEL,
                model_id=request.model_id,
                request_data={},
            )
            db.add(future)
            await db.commit()
            await db.refresh(future)

        return UnloadModelResponse(
            request_id=str(future.request_id),
            model_id=request.model_id,
        )

    @app.post("/api/v1/get_info")
    async def get_info(request: GetInfoRequest) -> ModelInfoResponse:
        async with get_session() as db:
            model = await db.get(ModelDB, request.model_id)
            if model is None:
                raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

        return ModelInfoResponse(
            model_id=model.model_id,
            status=model.status,
            model_data=ModelData(
                base_model=model.base_model,
                lora_config=model.lora_config,
                status=model.status,
            ),
        )

    # ------------------------------------------------------------------
    # Training operations (return futures)
    # ------------------------------------------------------------------

    @app.post("/api/v1/forward_backward")
    async def forward_backward(request: ForwardBackwardRequest) -> FutureResponse:
        return await _create_future(
            RequestType.FORWARD_BACKWARD,
            request.model_id,
            {
                "data": [d.model_dump() for d in request.data],
                "loss_fn": request.loss_fn,
                "loss_fn_config": request.loss_fn_config,
            },
        )

    @app.post("/api/v1/forward")
    async def forward(request: ForwardRequest) -> FutureResponse:
        return await _create_future(
            RequestType.FORWARD,
            request.model_id,
            {"data": [d.model_dump() for d in request.data]},
        )

    @app.post("/api/v1/optim_step")
    async def optim_step(request: OptimStepRequest) -> FutureResponse:
        return await _create_future(
            RequestType.OPTIM_STEP,
            request.model_id,
            {"adam_params": request.adam_params.model_dump()},
        )

    # ------------------------------------------------------------------
    # Weight management (return futures)
    # ------------------------------------------------------------------

    @app.post("/api/v1/save_weights")
    async def save_weights(request: SaveWeightsRequest) -> FutureResponse:
        return await _create_future(
            RequestType.SAVE_WEIGHTS,
            request.model_id,
            {"path": request.path},
        )

    @app.post("/api/v1/save_weights_for_sampler")
    async def save_weights_for_sampler(request: SaveWeightsForSamplerRequest) -> FutureResponse:
        return await _create_future(
            RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
            request.model_id,
            {
                "path": request.path,
                "sampling_session_seq_id": request.sampling_session_seq_id,
                "seq_id": request.seq_id,
            },
        )

    @app.post("/api/v1/load_weights")
    async def load_weights(request: LoadWeightsRequest) -> FutureResponse:
        return await _create_future(
            RequestType.LOAD_WEIGHTS,
            request.model_id,
            {
                "source_model_id": request.source_model_id,
                "checkpoint_id": request.checkpoint_id,
            },
        )

    # ------------------------------------------------------------------
    # Sampling (returns future)
    # ------------------------------------------------------------------

    @app.post("/api/v1/asample")
    async def asample(request: SampleRequest) -> FutureResponse:
        return await _create_future(
            RequestType.SAMPLE,
            None,
            request.model_dump(),
        )

    # ------------------------------------------------------------------
    # Futures
    # ------------------------------------------------------------------

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(request: RetrieveFutureRequest) -> RetrieveFutureResponse:
        async with get_session() as db:
            future = await db.get(FutureDB, int(request.future_id))
            if future is None:
                raise HTTPException(
                    status_code=404, detail=f"Future {request.future_id} not found"
                )

        if future.status == RequestStatus.PENDING:
            return RetrieveFutureResponse(status="pending")
        elif future.status == RequestStatus.FAILED:
            error = future.result_data.get("error", "Unknown error") if future.result_data else "Unknown error"
            return RetrieveFutureResponse(status="failed", error=error)
        else:
            return RetrieveFutureResponse(status="completed", result=future.result_data)

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
) -> FutureResponse:
    """Helper to insert a future into the DB and return a FutureResponse."""
    async with get_session() as db:
        future = FutureDB(
            request_type=request_type,
            model_id=model_id,
            request_data=request_data,
        )
        db.add(future)
        await db.commit()
        await db.refresh(future)

    return FutureResponse(
        future_id=str(future.request_id),
        status="pending",
        request_id=str(future.request_id),
    )
