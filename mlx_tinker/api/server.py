"""FastAPI server implementing the Tinker API endpoints.

Wire-compatible with the tinker SDK: submit endpoints return UntypedAPIFuture,
retrieve_future returns the raw typed result or TryAgainResponse.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tarfile
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import zstandard as zstd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from sqlalchemy import func, select
from starlette.background import BackgroundTask

from mlx_tinker.api.models import (
    Checkpoint,
    CheckpointsListResponse,
    ClientConfigRequest,
    ClientConfigResponse,
    ClientDynamicConfigResponse,
    CreateModelRequest,
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    Cursor,
    ForwardBackwardRequest,
    ForwardRequest,
    GetInfoRequest,
    GetInfoResponse,
    GetSamplerResponse,
    GetServerCapabilitiesResponse,
    GetSessionResponse,
    HealthResponse,
    ListSessionsResponse,
    LoadWeightsRequest,
    ModelData,
    OptimStepRequest,
    RetrieveFutureRequest,
    SampleRequest,
    SaveWeightsForSamplerRequest,
    SaveWeightsRequest,
    SessionHeartbeatRequest,
    SessionHeartbeatResponse,
    SetTtlRequest,
    SupportedModel,
    TelemetryRequest,
    TelemetryResponse,
    TrainingRun,
    TrainingRunsResponse,
    TryAgainResponse,
    UnloadModelRequest,
    UntypedAPIFuture,
    WeightsInfoRequest,
    WeightsInfoResponse,
)
from mlx_tinker.api.proto_wire import (
    decode_forward_backward_proto,
    serialize_forward_backward_output_proto,
    serialize_sample_response_proto,
)
from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.backend.uri import format_tinker_path
from mlx_tinker.config import EngineConfig
from mlx_tinker.db.database import close_db, get_session, init_db
from mlx_tinker.db.models import CheckpointDB, FutureDB, ModelDB, SamplingSessionDB, SessionDB
from mlx_tinker.engine.engine import TinkerEngine
from mlx_tinker.types import CheckpointStatus, CheckpointType, RequestStatus, RequestType

logger = logging.getLogger(__name__)

# Global references set during lifespan
_engine: TinkerEngine | None = None
_backend: MLXBackend | None = None
_config: EngineConfig | None = None
_archive_cache: dict[str, dict] = {}


def create_app(config: EngineConfig | None = None) -> FastAPI:
    """Create the FastAPI application with the Tinker API routes."""
    global _config
    if config is None:
        config = EngineConfig()
    _config = config

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _engine, _backend, _config
        _config = config

        # Init DB
        await init_db(config.database_path)
        logger.info("Database initialized at %s", config.database_path)

        # Init backend (model loaded lazily on first request)
        _backend = MLXBackend(config)

        # Start engine
        _engine = TinkerEngine(config, _backend)
        await _engine.start()

        # Register OpenAI-compatible routes now that backend is ready
        from mlx_tinker.api.lora_ui import register_lora_routes
        from mlx_tinker.api.openai_compat import register_openai_routes

        register_openai_routes(app, _backend)
        register_lora_routes(app, _backend)
        logger.info("OpenAI-compatible routes registered")

        yield

        # Shutdown
        if _engine is not None:
            await _engine.stop()
        if _backend is not None:
            _backend.close()
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

    @app.post("/api/v1/client/config")
    async def client_config(request: ClientConfigRequest) -> ClientConfigResponse:
        return ClientConfigResponse()

    @app.post("/api/v1/client/dynamic_config")
    async def client_dynamic_config(request: ClientConfigRequest) -> ClientDynamicConfigResponse:
        return ClientDynamicConfigResponse()

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
                user_metadata=request.user_metadata or {},
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
    async def forward_backward(request: Request) -> UntypedAPIFuture:
        content_type = request.headers.get("content-type", "")
        if "application/x-protobuf" in content_type:
            raw_body = await request.body()
            if request.headers.get("content-encoding") == "zstd":
                raw_body = zstd.ZstdDecompressor().decompress(raw_body)
            model_id, request_data = decode_forward_backward_proto(raw_body)
            return await _create_future(
                RequestType.FORWARD_BACKWARD,
                model_id,
                request_data,
            )
        else:
            json_data = await request.json()
            try:
                parsed_req = ForwardBackwardRequest(**json_data)
            except ValidationError as e:
                raise RequestValidationError(e.errors())
            fbi = parsed_req.forward_backward_input
            return await _create_future(
                RequestType.FORWARD_BACKWARD,
                parsed_req.model_id,
                {
                    "data": [d.model_dump() for d in fbi.data],
                    "loss_fn": fbi.loss_fn,
                    "loss_fn_config": fbi.loss_fn_config,
                    "forward_only": False,
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
            {"path": request.path, "ttl_seconds": request.ttl_seconds},
        )

    @app.post("/api/v1/save_weights_for_sampler")
    async def save_weights_for_sampler(request: SaveWeightsForSamplerRequest) -> UntypedAPIFuture:
        request_payload = {
            "path": request.path,
            "sampling_session_seq_id": request.sampling_session_seq_id,
            "seq_id": request.seq_id,
            "sampling_session_id": None,
            "ephemeral": request.path is None,
            "ttl_seconds": request.ttl_seconds,
        }
        if request.path is None:
            sampling_session_id = str(uuid.uuid4())
            if _config is None:
                raise RuntimeError("Server config is not initialized")
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
                    model_id=request.model_id,
                    base_model=model.base_model,
                    model_path=None,
                )
                db.add(sampling_session)
                await db.commit()
            request_payload["sampling_session_id"] = sampling_session_id

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
                "path": request.path,
                "optimizer": request.optimizer,
            },
        )

    # ------------------------------------------------------------------
    # Sampling — returns UntypedAPIFuture
    # ------------------------------------------------------------------

    @app.post("/api/v1/asample")
    async def asample(request: SampleRequest) -> UntypedAPIFuture:
        resolved_model_id, resolved_request = await _resolve_sampling_request(request)
        num_samples = request.num_samples or 1
        sample_sequence_ids = [f"seq_{uuid.uuid4().hex}" for _ in range(num_samples)]
        return await _create_future(
            RequestType.SAMPLE,
            resolved_model_id,
            resolved_request,
            sample_sequence_ids=sample_sequence_ids,
        )

    # ------------------------------------------------------------------
    # Futures — SDK protocol: return raw typed result or TryAgainResponse
    # ------------------------------------------------------------------

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(request: Request):
        json_body = await request.json()
        req_obj = RetrieveFutureRequest(**json_body)
        deadline = asyncio.get_running_loop().time() + 1.0
        future = None
        while True:
            async with get_session() as db:
                future = await db.get(FutureDB, int(req_obj.request_id))
                if future is None:
                    raise HTTPException(
                        status_code=404, detail=f"Future {req_obj.request_id} not found"
                    )
            if future.status != RequestStatus.PENDING:
                break
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(0.1)

        if future.status == RequestStatus.PENDING:
            return TryAgainResponse(
                request_id=req_obj.request_id,
                queue_state="active",
            )
        elif future.status == RequestStatus.FAILED:
            error = (
                future.result_data.get("error", "Unknown error")
                if future.result_data
                else "Unknown error"
            )
            category = (
                future.result_data.get("category", "server")
                if future.result_data and future.result_data.get("category") in {"unknown", "server", "user"}
                else "server"
            )
            return JSONResponse(
                status_code=200,
                content={"error": error, "category": category},
            )
        else:
            accept_header = request.headers.get("accept", "")
            if "application/x-protobuf" in accept_header:
                if future.request_type == RequestType.FORWARD_BACKWARD:
                    proto_bytes = serialize_forward_backward_output_proto(future.result_data or {})
                    return Response(content=proto_bytes, media_type="application/x-protobuf")
                elif future.request_type == RequestType.SAMPLE:
                    proto_bytes = serialize_sample_response_proto(future.result_data or {})
                    return Response(content=proto_bytes, media_type="application/x-protobuf")

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

    # ------------------------------------------------------------------
    # Sampler & Weight Metadata (Phase 3)
    # ------------------------------------------------------------------

    @app.get("/api/v1/samplers/{sampler_id}")
    async def get_sampler(sampler_id: str) -> GetSamplerResponse:
        async with get_session() as db:
            sampling_session = await db.get(SamplingSessionDB, sampler_id)
            if sampling_session is None:
                raise HTTPException(status_code=404, detail=f"Sampler {sampler_id} not found")

        resolved_base_model = sampling_session.base_model
        if not resolved_base_model and _config is not None:
            resolved_base_model = _config.base_model
        elif not resolved_base_model:
            resolved_base_model = "unknown"

        return GetSamplerResponse(
            sampler_id=sampling_session.sampling_session_id,
            base_model=resolved_base_model,
            model_path=sampling_session.model_path,
        )

    @app.post("/api/v1/weights_info")
    async def weights_info(request: WeightsInfoRequest) -> WeightsInfoResponse:
        path = request.path
        if not path:
            raise HTTPException(status_code=400, detail="tinker_path or model_path must be provided")

        cfg = _config or EngineConfig()
        backend = _backend if (_backend is not None and _backend.config.checkpoints_base == cfg.checkpoints_base) else MLXBackend(cfg)

        try:
            info = backend.get_weights_info(path)
            return WeightsInfoResponse(**info)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to inspect weights: {e}")

    # ------------------------------------------------------------------
    # Training Run & Checkpoint Management (Phase 4)
    # ------------------------------------------------------------------

    @app.get("/api/v1/training_runs/{training_run_id}")
    async def get_training_run(
        training_run_id: str,
        access_scope: str = "owned",
    ) -> TrainingRun:
        clean_model_id = _normalize_model_id(training_run_id)
        async with get_session() as db:
            model = await db.get(ModelDB, clean_model_id)
            if model is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Training run {training_run_id} not found",
                )
        return await _build_training_run(model, _config)

    @app.get("/api/v1/training_runs")
    async def list_training_runs(
        limit: int = Query(20, ge=1),
        offset: int = Query(0, ge=0),
        access_scope: str = "owned",
    ) -> TrainingRunsResponse:
        async with get_session() as db:
            count_stmt = select(func.count()).select_from(ModelDB)
            total = (await db.execute(count_stmt)).scalar() or 0

            stmt = select(ModelDB).order_by(ModelDB.created_at.desc()).offset(offset).limit(limit)
            models = list((await db.execute(stmt)).scalars().all())

        runs = [await _build_training_run(m, _config) for m in models]
        return TrainingRunsResponse(
            training_runs=runs,
            cursor=Cursor(offset=offset, limit=limit, total_count=total),
        )

    @app.get("/api/v1/training_runs/{model_id}/checkpoints")
    async def list_checkpoints(model_id: str) -> CheckpointsListResponse:
        clean_model_id = _normalize_model_id(model_id)
        cfg = _config or EngineConfig()
        base = cfg.checkpoints_base.resolve()
        async with get_session() as db:
            model = await db.get(ModelDB, clean_model_id)
            if model is None and not (base / clean_model_id).is_dir():
                raise HTTPException(
                    status_code=404,
                    detail=f"Training run {model_id} not found",
                )

        checkpoints = await _get_checkpoints_for_model(clean_model_id, cfg)
        return CheckpointsListResponse(checkpoints=checkpoints, cursor=None)

    @app.get("/api/v1/checkpoints")
    async def list_user_checkpoints(
        limit: int = Query(100, ge=1),
        offset: int = Query(0, ge=0),
    ) -> CheckpointsListResponse:
        cfg = _config or EngineConfig()
        async with get_session() as db:
            models = list((await db.execute(select(ModelDB))).scalars().all())
            model_ids = {m.model_id for m in models}

        base = cfg.checkpoints_base.resolve()
        if base.is_dir():
            for child in base.iterdir():
                if child.is_dir() and child.name not in ("prefix_cache", "tmp"):
                    model_ids.add(child.name)

        all_checkpoints: list[Checkpoint] = []
        for mid in sorted(model_ids):
            ckpts = await _get_checkpoints_for_model(mid, cfg)
            all_checkpoints.extend(ckpts)

        all_checkpoints.sort(
            key=lambda c: c.time if isinstance(c.time, datetime) else datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        total = len(all_checkpoints)
        sliced = all_checkpoints[offset : offset + limit]
        return CheckpointsListResponse(
            checkpoints=sliced,
            cursor=Cursor(offset=offset, limit=limit, total_count=total),
        )

    @app.post("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id:path}/publish")
    async def publish_checkpoint(model_id: str, checkpoint_id: str) -> dict[str, str]:
        clean_model_id = _normalize_model_id(model_id)
        clean_id, pref_type = _normalize_checkpoint_id(checkpoint_id)
        cfg = _config or EngineConfig()
        base = cfg.checkpoints_base.resolve()

        async with get_session() as db:
            stmt = select(CheckpointDB).where(
                CheckpointDB.model_id == clean_model_id,
                (CheckpointDB.checkpoint_id == clean_id) | (CheckpointDB.checkpoint_id == checkpoint_id),
            )
            if pref_type:
                stmt = stmt.where(CheckpointDB.checkpoint_type == pref_type)
            row = (await db.execute(stmt)).scalars().first()

            if row is None:
                sampler_path = base / clean_model_id / "sampler" / clean_id
                training_path = base / clean_model_id / clean_id
                if pref_type == CheckpointType.SAMPLER and sampler_path.is_dir():
                    ckpt_type = CheckpointType.SAMPLER
                elif pref_type == CheckpointType.TRAINING and training_path.is_dir():
                    ckpt_type = CheckpointType.TRAINING
                elif sampler_path.is_dir():
                    ckpt_type = CheckpointType.SAMPLER
                elif training_path.is_dir():
                    ckpt_type = CheckpointType.TRAINING
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Checkpoint {checkpoint_id} not found for model {model_id}",
                    )
                row = CheckpointDB(
                    model_id=clean_model_id,
                    checkpoint_id=clean_id,
                    checkpoint_type=ckpt_type,
                    status=CheckpointStatus.COMPLETED,
                    public=True,
                )
                db.add(row)
                await db.commit()
                return {"status": "published"}

            if row.public:
                raise HTTPException(
                    status_code=409,
                    detail=f"Checkpoint {checkpoint_id} is already public",
                )

            row.public = True
            await db.commit()
            return {"status": "published"}

    @app.delete("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id:path}/publish")
    async def unpublish_checkpoint(model_id: str, checkpoint_id: str) -> dict[str, str]:
        clean_model_id = _normalize_model_id(model_id)
        clean_id, pref_type = _normalize_checkpoint_id(checkpoint_id)
        cfg = _config or EngineConfig()
        base = cfg.checkpoints_base.resolve()

        async with get_session() as db:
            stmt = select(CheckpointDB).where(
                CheckpointDB.model_id == clean_model_id,
                (CheckpointDB.checkpoint_id == clean_id) | (CheckpointDB.checkpoint_id == checkpoint_id),
            )
            if pref_type:
                stmt = stmt.where(CheckpointDB.checkpoint_type == pref_type)
            row = (await db.execute(stmt)).scalars().first()

            if row is None:
                sampler_path = base / clean_model_id / "sampler" / clean_id
                training_path = base / clean_model_id / clean_id
                if not (sampler_path.is_dir() or training_path.is_dir()):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Checkpoint {checkpoint_id} not found for model {model_id}",
                    )
                raise HTTPException(
                    status_code=409,
                    detail=f"Checkpoint {checkpoint_id} is already private",
                )

            if not row.public:
                raise HTTPException(
                    status_code=409,
                    detail=f"Checkpoint {checkpoint_id} is already private",
                )

            row.public = False
            await db.commit()
            return {"status": "unpublished"}

    @app.put("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id:path}/ttl")
    async def set_checkpoint_ttl(
        model_id: str,
        checkpoint_id: str,
        request: SetTtlRequest,
    ) -> dict[str, str]:
        if request.ttl_seconds is not None and request.ttl_seconds <= 0:
            raise HTTPException(status_code=400, detail="ttl_seconds must be positive")

        clean_model_id = _normalize_model_id(model_id)
        clean_id, pref_type = _normalize_checkpoint_id(checkpoint_id)
        cfg = _config or EngineConfig()
        base = cfg.checkpoints_base.resolve()

        async with get_session() as db:
            stmt = select(CheckpointDB).where(
                CheckpointDB.model_id == clean_model_id,
                (CheckpointDB.checkpoint_id == clean_id) | (CheckpointDB.checkpoint_id == checkpoint_id),
            )
            if pref_type:
                stmt = stmt.where(CheckpointDB.checkpoint_type == pref_type)
            row = (await db.execute(stmt)).scalars().first()

            new_expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=request.ttl_seconds)
                if request.ttl_seconds is not None
                else None
            )

            if row is None:
                sampler_path = base / clean_model_id / "sampler" / clean_id
                training_path = base / clean_model_id / clean_id
                if pref_type == CheckpointType.SAMPLER and sampler_path.is_dir():
                    ckpt_type = CheckpointType.SAMPLER
                elif pref_type == CheckpointType.TRAINING and training_path.is_dir():
                    ckpt_type = CheckpointType.TRAINING
                elif sampler_path.is_dir():
                    ckpt_type = CheckpointType.SAMPLER
                elif training_path.is_dir():
                    ckpt_type = CheckpointType.TRAINING
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Checkpoint {checkpoint_id} not found for model {model_id}",
                    )
                row = CheckpointDB(
                    model_id=clean_model_id,
                    checkpoint_id=clean_id,
                    checkpoint_type=ckpt_type,
                    status=CheckpointStatus.COMPLETED,
                    expires_at=new_expires_at,
                )
                db.add(row)
                await db.commit()
                return {"status": "updated"}

            row.expires_at = new_expires_at
            await db.commit()
            return {"status": "updated"}

    @app.get("/api/v1/sessions/{session_id}")
    async def get_session_by_id(session_id: str) -> GetSessionResponse:
        async with get_session() as db:
            sess = await db.get(SessionDB, session_id)
            if sess is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_id} not found",
                )

            stmt_models = select(ModelDB.model_id).where(ModelDB.session_id == session_id)
            model_ids = list((await db.execute(stmt_models)).scalars().all())

            stmt_samplers = select(SamplingSessionDB.sampling_session_id).where(
                SamplingSessionDB.session_id == session_id
            )
            sampler_ids = list((await db.execute(stmt_samplers)).scalars().all())

            return GetSessionResponse(
                training_run_ids=model_ids,
                sampler_ids=sampler_ids,
            )

    @app.get("/api/v1/sessions")
    async def list_sessions(
        limit: int = Query(100, ge=1),
        offset: int = Query(0, ge=0),
    ) -> ListSessionsResponse:
        async with get_session() as db:
            stmt = (
                select(SessionDB.session_id)
                .order_by(SessionDB.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            session_ids = list((await db.execute(stmt)).scalars().all())
            return ListSessionsResponse(sessions=session_ids)

    @app.get("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id:path}/archive")
    async def get_checkpoint_archive(
        model_id: str,
        checkpoint_id: str,
        request: Request,
    ) -> Response:
        clean_model_id = _normalize_model_id(model_id)
        clean_id, pref_type = _normalize_checkpoint_id(checkpoint_id)
        cfg = _config or EngineConfig()
        base = cfg.checkpoints_base.resolve()

        ckpt_dir: Path | None = None
        if pref_type == CheckpointType.SAMPLER:
            p = base / clean_model_id / "sampler" / clean_id
            if p.is_dir():
                ckpt_dir = p
        elif pref_type == CheckpointType.TRAINING:
            p = base / clean_model_id / clean_id
            if p.is_dir():
                ckpt_dir = p
        else:
            # Try sampler first, then training
            p_sampler = base / clean_model_id / "sampler" / clean_id
            p_training = base / clean_model_id / clean_id
            if p_sampler.is_dir():
                ckpt_dir = p_sampler
            elif p_training.is_dir():
                ckpt_dir = p_training

        if ckpt_dir is None or not ckpt_dir.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint {checkpoint_id} not found for model {model_id}",
            )

        # Purge any expired archive entries
        now = datetime.now(timezone.utc)
        expired_keys = [k for k, v in _archive_cache.items() if now > v.get("expires_at", now)]
        for k in expired_keys:
            _archive_cache.pop(k, None)

        download_id = str(uuid.uuid4())
        _archive_cache[download_id] = {
            "source_dir": ckpt_dir,
            "filename": f"{clean_model_id}_{clean_id}.tar.gz",
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=15),
        }

        # Build download URL
        base_url = str(request.base_url).rstrip("/")
        download_url = f"{base_url}/api/v1/archives/{download_id}/download"

        expires = datetime.now(timezone.utc) + timedelta(minutes=15)
        # RFC 7231 / RFC 1123 format: Sun, 06 Nov 1994 08:49:37 GMT
        expires_str = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")

        return Response(
            status_code=302,
            headers={
                "Location": download_url,
                "Expires": expires_str,
            },
        )

    @app.get("/api/v1/archives/{archive_id}/download")
    async def download_archive(archive_id: str) -> Response:
        entry = _archive_cache.get(archive_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Archive download link not found or expired")

        if datetime.now(timezone.utc) > entry["expires_at"]:
            _archive_cache.pop(archive_id, None)
            raise HTTPException(status_code=410, detail="Archive download link expired")

        source_dir: Path = entry["source_dir"]
        if not source_dir.is_dir():
            raise HTTPException(status_code=404, detail="Checkpoint directory no longer exists")

        # Create tar.gz in tempfile
        def _build_tar() -> Path:
            tmp = tempfile.NamedTemporaryFile(prefix="tinker-ckpt-", suffix=".tar.gz", delete=False)
            tmp_path = Path(tmp.name)
            tmp.close()
            with tarfile.open(tmp_path, "w:gz") as tar:
                tar.add(source_dir, arcname=source_dir.name)
            return tmp_path

        tar_path = await asyncio.to_thread(_build_tar)

        def _cleanup():
            try:
                tar_path.unlink(missing_ok=True)
            except OSError:
                pass

        return FileResponse(
            path=str(tar_path),
            media_type="application/gzip",
            filename=entry["filename"],
            background=BackgroundTask(_cleanup),
        )

    @app.delete("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id:path}")
    async def delete_checkpoint(model_id: str, checkpoint_id: str) -> dict[str, str]:
        clean_model_id = _normalize_model_id(model_id)
        clean_id, pref_type = _normalize_checkpoint_id(checkpoint_id)
        cfg = _config or EngineConfig()
        base = cfg.checkpoints_base.resolve()

        deleted_disk = False
        if pref_type in (None, CheckpointType.SAMPLER):
            sampler_path = base / clean_model_id / "sampler" / clean_id
            if sampler_path.is_dir():
                shutil.rmtree(sampler_path)
                deleted_disk = True

        if pref_type in (None, CheckpointType.TRAINING):
            training_path = base / clean_model_id / clean_id
            if training_path.is_dir():
                shutil.rmtree(training_path)
                deleted_disk = True

        async with get_session() as db:
            stmt = select(CheckpointDB).where(
                CheckpointDB.model_id == clean_model_id,
                (CheckpointDB.checkpoint_id == clean_id) | (CheckpointDB.checkpoint_id == checkpoint_id),
            )
            if pref_type:
                stmt = stmt.where(CheckpointDB.checkpoint_type == pref_type)
            rows = list((await db.execute(stmt)).scalars().all())

            for row in rows:
                await db.delete(row)

            if rows:
                await db.commit()

        if not deleted_disk and not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint {checkpoint_id} not found for model {model_id}",
            )

        return {"status": "deleted"}


def _normalize_checkpoint_id(checkpoint_id: str) -> tuple[str, CheckpointType | None]:
    """Normalize checkpoint_id, stripping any prefix like weights/ or sampler_weights/ or sampler/ or tinker://."""
    if checkpoint_id.startswith("tinker://"):
        parts = checkpoint_id[len("tinker://") :].split("/")
        if len(parts) >= 3:
            ctype = CheckpointType.TRAINING if parts[1] == "weights" else CheckpointType.SAMPLER
            return "/".join(parts[2:]), ctype
    if checkpoint_id.startswith("weights/"):
        return checkpoint_id[len("weights/") :], CheckpointType.TRAINING
    elif checkpoint_id.startswith("sampler_weights/"):
        return checkpoint_id[len("sampler_weights/") :], CheckpointType.SAMPLER
    elif checkpoint_id.startswith("sampler/"):
        return checkpoint_id[len("sampler/") :], CheckpointType.SAMPLER
    return checkpoint_id, None


def _normalize_model_id(model_id: str) -> str:
    """Extract clean model_id if passed as a tinker:// URI."""
    if model_id.startswith("tinker://"):
        return model_id[len("tinker://") :].split("/")[0]
    return model_id


async def _get_checkpoints_for_model(
    model_id: str, config: EngineConfig | None = None
) -> list[Checkpoint]:
    cfg = config or _config or EngineConfig()
    async with get_session() as db:
        stmt = select(CheckpointDB).where(CheckpointDB.model_id == model_id)
        db_rows = list((await db.execute(stmt)).scalars().all())

    db_by_key = {(r.checkpoint_id, r.checkpoint_type): r for r in db_rows}
    checkpoints: dict[tuple[str, str], Checkpoint] = {}

    base = cfg.checkpoints_base.resolve()
    model_dir = base / model_id

    # 1. Scan sampler checkpoints on disk: model_dir / "sampler" / <name>
    sampler_dir = model_dir / "sampler"
    if sampler_dir.is_dir():
        for child in sampler_dir.iterdir():
            if child.is_dir():
                ckpt_id = child.name
                size_bytes = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                mtime = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc)
                db_row = db_by_key.get((ckpt_id, CheckpointType.SAMPLER))
                public = db_row.public if db_row else False
                expires_at = db_row.expires_at if db_row else None
                created_at = (db_row.completed_at or db_row.created_at) if db_row else mtime
                if db_row and db_row.size_bytes is not None:
                    size_bytes = db_row.size_bytes
                tinker_path = format_tinker_path(model_id, ckpt_id, CheckpointType.SAMPLER)
                checkpoints[(ckpt_id, "sampler")] = Checkpoint(
                    checkpoint_id=ckpt_id,
                    checkpoint_type="sampler",
                    time=created_at,
                    tinker_path=tinker_path,
                    size_bytes=size_bytes,
                    public=public,
                    expires_at=expires_at,
                )

    # 2. Scan training checkpoints on disk: model_dir / <name> (excluding "sampler")
    if model_dir.is_dir():
        for child in model_dir.iterdir():
            if child.name == "sampler":
                continue
            if child.is_dir():
                ckpt_id = child.name
                size_bytes = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                mtime = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc)
                db_row = db_by_key.get((ckpt_id, CheckpointType.TRAINING))
                public = db_row.public if db_row else False
                expires_at = db_row.expires_at if db_row else None
                created_at = (db_row.completed_at or db_row.created_at) if db_row else mtime
                if db_row and db_row.size_bytes is not None:
                    size_bytes = db_row.size_bytes
                tinker_path = format_tinker_path(model_id, ckpt_id, CheckpointType.TRAINING)
                checkpoints[(ckpt_id, "training")] = Checkpoint(
                    checkpoint_id=ckpt_id,
                    checkpoint_type="training",
                    time=created_at,
                    tinker_path=tinker_path,
                    size_bytes=size_bytes,
                    public=public,
                    expires_at=expires_at,
                )

    # 3. Add any DB rows not already discovered from disk
    for r in db_rows:
        ckpt_type_str = "sampler" if r.checkpoint_type == CheckpointType.SAMPLER else "training"
        key = (r.checkpoint_id, ckpt_type_str)
        if key not in checkpoints:
            tinker_path = format_tinker_path(model_id, r.checkpoint_id, r.checkpoint_type)
            checkpoints[key] = Checkpoint(
                checkpoint_id=r.checkpoint_id,
                checkpoint_type=ckpt_type_str,
                time=r.completed_at or r.created_at,
                tinker_path=tinker_path,
                size_bytes=r.size_bytes,
                public=r.public,
                expires_at=r.expires_at,
            )

    result = list(checkpoints.values())
    result.sort(
        key=lambda c: c.time if isinstance(c.time, datetime) else datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return result


async def _build_training_run(
    model: ModelDB, config: EngineConfig | None = None
) -> TrainingRun:
    cfg = config or _config or EngineConfig()
    checkpoints = await _get_checkpoints_for_model(model.model_id, cfg)
    last_checkpoint = next((c for c in checkpoints if c.checkpoint_type == "training"), None)
    last_sampler_checkpoint = next((c for c in checkpoints if c.checkpoint_type == "sampler"), None)

    async with get_session() as db:
        stmt = select(func.max(FutureDB.created_at)).where(FutureDB.model_id == model.model_id)
        latest_future_time = (await db.execute(stmt)).scalar()

        session = await db.get(SessionDB, model.session_id)
        user_meta = None
        if model.user_metadata and isinstance(model.user_metadata, dict):
            user_meta = {str(k): str(v) for k, v in model.user_metadata.items()}
        elif session and session.user_metadata and isinstance(session.user_metadata, dict):
            user_meta = {str(k): str(v) for k, v in session.user_metadata.items()}

    last_request_time = latest_future_time or model.created_at
    lora_rank = model.lora_config.get("rank") if isinstance(model.lora_config, dict) else None
    is_lora = True
    if isinstance(model.lora_config, dict) and model.lora_config.get("is_lora") is False:
        is_lora = False

    return TrainingRun(
        training_run_id=model.model_id,
        base_model=model.base_model,
        model_owner="",
        is_lora=is_lora,
        corrupted=False,
        lora_rank=lora_rank,
        last_request_time=last_request_time,
        last_checkpoint=last_checkpoint,
        last_sampler_checkpoint=last_sampler_checkpoint,
        user_metadata=user_meta,
    )


async def _create_future(
    request_type: RequestType,
    model_id: str | None,
    request_data: dict,
    sample_sequence_ids: list[str] | None = None,
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
        sample_sequence_ids=sample_sequence_ids,
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

        resolved_request["base_model"] = sampling_session.base_model or resolved_request.get(
            "base_model"
        )
        if sampling_session.model_id is not None:
            resolved_model_id = sampling_session.model_id
            resolved_request.pop("model_path", None)
            resolved_request["model_id"] = sampling_session.model_id
        elif sampling_session.model_path is not None:
            resolved_request["model_path"] = sampling_session.model_path
            # Path-backed sessions refer to exported weights, not an in-memory training model.
            resolved_model_id = None
        else:
            resolved_request.pop("model_path", None)
            resolved_model_id = None

    return resolved_model_id, resolved_request
