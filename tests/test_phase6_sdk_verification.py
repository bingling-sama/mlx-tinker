"""End-to-end SDK verification tests for Phase 6.

Validates that Tinker SDK (v0.16+) clients (ServiceClient, TrainingClient, SamplingClient, RestClient)
interact seamlessly with mlx-tinker API server without 404s or Pydantic validation errors.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import io
import tarfile
import httpx
import pytest
import tinker
import tinker.types

from mlx_tinker.api.server import create_app
from mlx_tinker.config import EngineConfig
from mlx_tinker.db.database import get_session, init_db
from mlx_tinker.db.models import CheckpointDB, FutureDB, ModelDB, SamplingSessionDB, SessionDB
from mlx_tinker.types import CheckpointStatus, CheckpointType, RequestType


@pytest.fixture
def config(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = EngineConfig(
        base_model="Qwen/Qwen3.5-0.8B",
        database_path=tmp_path / "test.db",
        checkpoints_base=ckpt_dir,
    )
    return cfg


@pytest.fixture
def app(config):
    return create_app(config)


@pytest.fixture
def async_tinker_client(app, config):
    asyncio.run(init_db(config.database_path))
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    sdk_client = tinker._client.AsyncTinker(
        base_url="http://testserver",
        api_key="tml-local",
        http_client=client,
        _strict_response_validation=True,
    )
    yield sdk_client
    asyncio.run(client.aclose())


class TestPhase6SDKVerification:
    """End-to-end SDK parity verification across Service, Training, Sampling, and Rest interfaces."""

    @pytest.mark.anyio
    async def test_service_capabilities_and_session(self, async_tinker_client):
        # 1. capabilities
        caps = await async_tinker_client.service.get_server_capabilities()
        assert isinstance(caps, tinker.types.GetServerCapabilitiesResponse)
        assert len(caps.supported_models) > 0
        assert caps.supported_models[0].model_name == "Qwen/Qwen3.5-0.8B"

        # 2. create session
        sess_req = tinker.types.CreateSessionRequest(
            tags=["phase6"],
            user_metadata={"env": "test"},
            sdk_version="0.16.1",
        )
        sess_resp = await async_tinker_client.service.create_session(request=sess_req)
        assert isinstance(sess_resp, tinker.types.CreateSessionResponse)
        assert sess_resp.session_id

        # 3. session heartbeat
        hb_resp = await async_tinker_client.service.session_heartbeat(session_id=sess_resp.session_id)
        assert isinstance(hb_resp, tinker.types.SessionHeartbeatResponse)
        assert hb_resp.type == "session_heartbeat"

    @pytest.mark.anyio
    async def test_rest_client_training_runs_and_checkpoints(self, async_tinker_client, config):
        model_id = "run-phase6-test"
        session_id = "sess-p6-run"

        # Seed database
        async with get_session() as db:
            s = SessionDB(session_id=session_id, user_metadata={"project": "phase6"})
            db.add(s)
            m = ModelDB(
                model_id=model_id,
                base_model="Qwen/Qwen3.5-0.8B",
                lora_config={"rank": 16, "train_mlp": True, "train_attn": True, "train_unembed": True},
                status="ready",
                session_id=session_id,
            )
            db.add(m)
            await db.commit()

        # Create checkpoint on disk
        ckpt_dir = config.checkpoints_base / model_id / "step_0010"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "adapters.safetensors").write_bytes(b"lora-test-weights")
        (ckpt_dir / "config.json").write_text('{"rank": 16, "lora_config": {"rank": 16, "train_mlp": true}}')

        tinker_path = f"tinker://{model_id}/weights/step_0010"

        # 1. Get training run
        run = await async_tinker_client.get(f"/api/v1/training_runs/{model_id}", cast_to=tinker.types.TrainingRun)
        assert isinstance(run, tinker.types.TrainingRun)
        assert run.training_run_id == model_id
        assert run.lora_rank == 16
        assert run.is_lora is True
        assert run.last_checkpoint is not None
        assert run.last_checkpoint.checkpoint_id == "step_0010"

        # 2. List training runs
        runs_list = await async_tinker_client.get("/api/v1/training_runs", cast_to=tinker.types.TrainingRunsResponse)
        assert isinstance(runs_list, tinker.types.TrainingRunsResponse)
        assert any(r.training_run_id == model_id for r in runs_list.training_runs)
        assert runs_list.cursor.total_count >= 1

        # 3. List checkpoints for run
        ckpts = await async_tinker_client.weights.list(model_id)
        assert isinstance(ckpts, tinker.types.CheckpointsListResponse)
        assert len(ckpts.checkpoints) == 1
        assert ckpts.checkpoints[0].checkpoint_id == "step_0010"
        assert ckpts.checkpoints[0].public is False

        # 4. List user checkpoints
        all_ckpts = await async_tinker_client.get("/api/v1/checkpoints", cast_to=tinker.types.CheckpointsListResponse)
        assert isinstance(all_ckpts, tinker.types.CheckpointsListResponse)
        assert any(c.checkpoint_id == "step_0010" for c in all_ckpts.checkpoints)

        # 5. Weights info
        winfo = await async_tinker_client.post(
            "/api/v1/weights_info",
            body={"tinker_path": tinker_path},
            cast_to=tinker.types.WeightsInfoResponse,
        )
        assert isinstance(winfo, tinker.types.WeightsInfoResponse)
        assert winfo.base_model == "Qwen/Qwen3.5-0.8B"
        assert winfo.lora_rank == 16
        assert winfo.train_mlp is True

        # 6. Publish / Unpublish checkpoint
        await async_tinker_client.post(
            f"/api/v1/training_runs/{model_id}/checkpoints/step_0010/publish",
            cast_to=object,
        )
        ckpts_pub = await async_tinker_client.weights.list(model_id)
        assert ckpts_pub.checkpoints[0].public is True

        await async_tinker_client.delete(
            f"/api/v1/training_runs/{model_id}/checkpoints/step_0010/publish",
            cast_to=object,
        )
        ckpts_unpub = await async_tinker_client.weights.list(model_id)
        assert ckpts_unpub.checkpoints[0].public is False

        # 7. Set TTL
        await async_tinker_client.put(
            f"/api/v1/training_runs/{model_id}/checkpoints/step_0010/ttl",
            body={"ttl_seconds": 3600},
            cast_to=object,
        )
        ckpts_ttl = await async_tinker_client.weights.list(model_id)
        assert ckpts_ttl.checkpoints[0].expires_at is not None

        # 8. Archive URL redirect and download via official SDK method
        archive_resp = await async_tinker_client.weights.get_checkpoint_archive_url(
            model_id=model_id,
            checkpoint_id="step_0010",
        )
        assert isinstance(archive_resp, tinker.types.CheckpointArchiveUrlResponse)
        assert archive_resp.url
        assert archive_resp.expires

        # 9. Delete checkpoint via direct API delete
        await async_tinker_client.delete(f"/api/v1/training_runs/{model_id}/checkpoints/step_0010", cast_to=object)
        ckpts_after_del = await async_tinker_client.weights.list(model_id)
        assert len(ckpts_after_del.checkpoints) == 0

    @pytest.mark.anyio
    async def test_sessions_and_sampler_endpoints(self, async_tinker_client, config):
        session_id = "sess-p6-sampler"
        sampler_id = "sampler-p6-01"

        async with get_session() as db:
            s = SessionDB(session_id=session_id)
            db.add(s)
            m = ModelDB(model_id="run-p6-sampler", base_model="Qwen/Qwen3.5-0.8B", session_id=session_id, status="ready")
            db.add(m)
            ss = SamplingSessionDB(
                sampling_session_id=sampler_id,
                session_id=session_id,
                model_id="run-p6-sampler",
                base_model="Qwen/Qwen3.5-0.8B",
                model_path="tinker://run-p6-sampler/sampler_weights/export_01",
            )
            db.add(ss)
            await db.commit()

        # 1. Get sampler info
        sampler_info = await async_tinker_client.get(
            f"/api/v1/samplers/{sampler_id}",
            cast_to=tinker.types.GetSamplerResponse,
        )
        assert isinstance(sampler_info, tinker.types.GetSamplerResponse)
        assert sampler_info.sampler_id == sampler_id
        assert sampler_info.base_model == "Qwen/Qwen3.5-0.8B"
        assert sampler_info.model_path == "tinker://run-p6-sampler/sampler_weights/export_01"

        # 2. Get session
        sess_info = await async_tinker_client.get(
            f"/api/v1/sessions/{session_id}",
            cast_to=tinker.types.GetSessionResponse,
        )
        assert isinstance(sess_info, tinker.types.GetSessionResponse)
        assert sess_info.training_run_ids == ["run-p6-sampler"]
        assert sess_info.sampler_ids == [sampler_id]

        # 3. List sessions
        sess_list = await async_tinker_client.get(
            "/api/v1/sessions",
            cast_to=tinker.types.ListSessionsResponse,
        )
        assert isinstance(sess_list, tinker.types.ListSessionsResponse)
        assert any(s == session_id for s in sess_list.sessions)

    @pytest.mark.anyio
    async def test_training_model_lifecycle_sdk(self, async_tinker_client, config):
        # 1. Create model via SDK models resource
        create_model_req = tinker.types.CreateModelRequest(
            session_id="sess-lifecycle-sdk",
            model_seq_id=1,
            base_model="Qwen/Qwen3.5-0.8B",
            lora_config=tinker.types.LoraConfig(rank=8),
        )
        model_future = await async_tinker_client.models.create(request=create_model_req)
        assert isinstance(model_future, tinker.types.UntypedAPIFuture)
        model_id = model_future.model_id
        assert model_id

        # 2. Get info via SDK models resource
        get_info_req = tinker.types.GetInfoRequest(model_id=model_id)
        info_resp = await async_tinker_client.models.get_info(request=get_info_req)
        assert isinstance(info_resp, tinker.types.GetInfoResponse)
        assert info_resp.model_id == model_id
        assert info_resp.is_lora is True

        # 3. Unload model via API post
        unload_resp = await async_tinker_client.post(
            "/api/v1/unload_model",
            body={"model_id": model_id},
            cast_to=tinker.types.UntypedAPIFuture,
        )
        assert isinstance(unload_resp, tinker.types.UntypedAPIFuture)
