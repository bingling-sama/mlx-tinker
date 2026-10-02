"""Unit and integration tests for Phase 4: Training Run & Checkpoint Management REST API.

Tests:
- GET /api/v1/training_runs/{training_run_id} -> TrainingRun
- GET /api/v1/training_runs -> TrainingRunsResponse (with pagination and Cursor)
- GET /api/v1/training_runs/{model_id}/checkpoints -> CheckpointsListResponse
- GET /api/v1/checkpoints -> CheckpointsListResponse (global paginated list)
- POST /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/publish
- DELETE /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/publish
- PUT /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/ttl
- DELETE /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}
- Tinker SDK model compatibility validation
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
from starlette.testclient import TestClient
import tinker.types

from mlx_tinker.api.models import (
    Checkpoint,
    CheckpointsListResponse,
    Cursor,
    SetTtlRequest,
    TrainingRun,
    TrainingRunsResponse,
)
from mlx_tinker.api.server import create_app
from mlx_tinker.backend.uri import format_tinker_path
from mlx_tinker.config import EngineConfig
from mlx_tinker.db.database import get_session
from mlx_tinker.db.models import CheckpointDB, FutureDB, ModelDB, SessionDB
from mlx_tinker.types import CheckpointStatus, CheckpointType, RequestType


@pytest.fixture
def config(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return EngineConfig(
        base_model="Qwen/Qwen3.5-0.8B",
        database_path=tmp_path / "test.db",
        checkpoints_base=ckpt_dir,
    )


@pytest.fixture
def app(config):
    return create_app(config)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


class TestTrainingRunsEndpoints:
    """Tests for GET /api/v1/training_runs/{training_run_id} and GET /api/v1/training_runs."""

    @pytest.mark.anyio
    async def test_get_training_run_not_found(self, client):
        resp = client.get("/api/v1/training_runs/non-existent-run")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.anyio
    async def test_get_training_run_success(self, client, config):
        # Setup session and model in DB
        async with get_session() as db:
            sess = SessionDB(
                session_id="sess-run-1",
                user_metadata={"project": "phase4-test", "experiment": "lora-1"},
            )
            db.add(sess)
            model = ModelDB(
                model_id="run-001",
                base_model="Qwen/Qwen3.5-2B",
                lora_config={"rank": 16, "alpha": 32.0},
                status="ready",
                session_id="sess-run-1",
            )
            db.add(model)

            # Add a future to verify last_request_time
            fut = FutureDB(
                request_type=RequestType.OPTIM_STEP,
                model_id="run-001",
                request_data={},
            )
            db.add(fut)
            await db.commit()

        # Create a checkpoint on disk
        ckpt_dir = config.checkpoints_base / "run-001" / "step_0050"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "model.safetensors").write_bytes(b"data" * 100)

        # Create a sampler checkpoint on disk
        sampler_dir = config.checkpoints_base / "run-001" / "sampler" / "export_final"
        sampler_dir.mkdir(parents=True, exist_ok=True)
        (sampler_dir / "adapters.safetensors").write_bytes(b"lora" * 50)

        resp = client.get("/api/v1/training_runs/run-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["training_run_id"] == "run-001"
        assert data["base_model"] == "Qwen/Qwen3.5-2B"
        assert data["is_lora"] is True
        assert data["lora_rank"] == 16
        assert data["corrupted"] is False
        assert data["user_metadata"] == {"project": "phase4-test", "experiment": "lora-1"}
        assert data["last_checkpoint"] is not None
        assert data["last_checkpoint"]["checkpoint_id"] == "step_0050"
        assert data["last_checkpoint"]["checkpoint_type"] == "training"
        assert data["last_sampler_checkpoint"] is not None
        assert data["last_sampler_checkpoint"]["checkpoint_id"] == "export_final"
        assert data["last_sampler_checkpoint"]["checkpoint_type"] == "sampler"

        # Validate with official Tinker SDK model
        sdk_obj = tinker.types.TrainingRun(**data)
        assert sdk_obj.training_run_id == "run-001"
        assert sdk_obj.base_model == "Qwen/Qwen3.5-2B"
        assert sdk_obj.lora_rank == 16

        # Query using ParsedCheckpointTinkerPath (as RestClient.get_training_run_by_tinker_path does)
        parsed = tinker.types.ParsedCheckpointTinkerPath.from_tinker_path("tinker://run-001/weights/step_0050")
        resp_parsed = client.get(f"/api/v1/training_runs/{parsed.training_run_id}")
        assert resp_parsed.status_code == 200
        assert resp_parsed.json()["training_run_id"] == "run-001"

    @pytest.mark.anyio
    async def test_model_user_metadata_precedence(self, client):
        async with get_session() as db:
            sess = SessionDB(session_id="sess-meta", user_metadata={"scope": "session"})
            db.add(sess)
            m = ModelDB(
                model_id="run-meta-override",
                base_model="Qwen/Qwen3.5-0.8B",
                status="ready",
                session_id="sess-meta",
                user_metadata={"scope": "model", "custom": "yes"},
            )
            db.add(m)
            await db.commit()

        resp = client.get("/api/v1/training_runs/run-meta-override")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_metadata"] == {"scope": "model", "custom": "yes"}

    @pytest.mark.anyio
    async def test_list_training_runs_pagination(self, client):
        async with get_session() as db:
            sess = SessionDB(session_id="sess-list-1")
            db.add(sess)
            for i in range(5):
                m = ModelDB(
                    model_id=f"run-list-{i:02d}",
                    base_model="Qwen/Qwen3.5-0.8B",
                    lora_config={"rank": 8},
                    status="ready",
                    session_id="sess-list-1",
                )
                db.add(m)
            await db.commit()

        # Query first page: limit=2, offset=0
        resp1 = client.get("/api/v1/training_runs?limit=2&offset=0")
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert len(data1["training_runs"]) == 2
        assert data1["cursor"]["offset"] == 0
        assert data1["cursor"]["limit"] == 2
        assert data1["cursor"]["total_count"] >= 5

        # Query second page: limit=2, offset=2
        resp2 = client.get("/api/v1/training_runs?limit=2&offset=2")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert len(data2["training_runs"]) == 2
        assert data2["cursor"]["offset"] == 2

        # Validate with official Tinker SDK TrainingRunsResponse
        sdk_obj = tinker.types.TrainingRunsResponse(**data1)
        assert len(sdk_obj.training_runs) == 2
        assert sdk_obj.cursor.limit == 2


class TestCheckpointsEndpoints:
    """Tests for listing, publishing, updating TTL, and deleting checkpoints."""

    @pytest.mark.anyio
    async def test_list_model_checkpoints(self, client, config):
        async with get_session() as db:
            sess = SessionDB(session_id="sess-ckpt-1")
            db.add(sess)
            model = ModelDB(
                model_id="run-ckpt-1",
                base_model="Qwen/Qwen3.5-0.8B",
                lora_config={"rank": 8},
                status="ready",
                session_id="sess-ckpt-1",
            )
            db.add(model)
            await db.commit()

        # Create training checkpoint
        ckpt_dir = config.checkpoints_base / "run-ckpt-1" / "step_100"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "model.safetensors").write_bytes(b"x" * 2048)

        # Create sampler checkpoint
        sampler_dir = config.checkpoints_base / "run-ckpt-1" / "sampler" / "export_1"
        sampler_dir.mkdir(parents=True, exist_ok=True)
        (sampler_dir / "adapters.safetensors").write_bytes(b"y" * 1024)

        resp = client.get("/api/v1/training_runs/run-ckpt-1/checkpoints")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["checkpoints"]) == 2

        types_found = {c["checkpoint_type"] for c in data["checkpoints"]}
        assert types_found == {"training", "sampler"}

        # Validate with official Tinker SDK CheckpointsListResponse
        sdk_obj = tinker.types.CheckpointsListResponse(**data)
        assert len(sdk_obj.checkpoints) == 2

    def test_list_model_checkpoints_not_found(self, client):
        resp = client.get("/api/v1/training_runs/non-existent/checkpoints")
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_list_user_checkpoints_global(self, client, config):
        async with get_session() as db:
            sess = SessionDB(session_id="sess-user-ckpt")
            db.add(sess)
            m1 = ModelDB(
                model_id="run-u1",
                base_model="Qwen/Qwen3.5-0.8B",
                status="ready",
                session_id="sess-user-ckpt",
            )
            m2 = ModelDB(
                model_id="run-u2",
                base_model="Qwen/Qwen3.5-0.8B",
                status="ready",
                session_id="sess-user-ckpt",
            )
            db.add(m1)
            db.add(m2)
            await db.commit()

        # Checkpoint for run-u1
        (config.checkpoints_base / "run-u1" / "ckpt_a").mkdir(parents=True, exist_ok=True)
        (config.checkpoints_base / "run-u1" / "ckpt_a" / "model.safetensors").write_bytes(b"a")

        # Checkpoint for run-u2
        (config.checkpoints_base / "run-u2" / "ckpt_b").mkdir(parents=True, exist_ok=True)
        (config.checkpoints_base / "run-u2" / "ckpt_b" / "model.safetensors").write_bytes(b"b")

        resp = client.get("/api/v1/checkpoints?limit=10&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["checkpoints"]) >= 2
        assert data["cursor"]["total_count"] >= 2

        sdk_obj = tinker.types.CheckpointsListResponse(**data)
        assert len(sdk_obj.checkpoints) >= 2

    @pytest.mark.anyio
    async def test_publish_and_unpublish_checkpoint(self, client, config):
        run_id = "run-pub-1"
        ckpt_id = "step_001"
        ckpt_dir = config.checkpoints_base / run_id / ckpt_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "model.safetensors").write_bytes(b"data")

        # 1. Publish checkpoint
        resp_pub = client.post(f"/api/v1/training_runs/{run_id}/checkpoints/{ckpt_id}/publish")
        assert resp_pub.status_code == 200
        assert resp_pub.json()["status"] == "published"

        # Verify listed as public
        resp_list = client.get(f"/api/v1/training_runs/{run_id}/checkpoints")
        assert resp_list.status_code == 200
        ckpts = resp_list.json()["checkpoints"]
        assert any(c["checkpoint_id"] == ckpt_id and c["public"] is True for c in ckpts)

        # 2. Publish again -> 409 Conflict
        resp_pub_again = client.post(f"/api/v1/training_runs/{run_id}/checkpoints/{ckpt_id}/publish")
        assert resp_pub_again.status_code == 409
        assert "already public" in resp_pub_again.json()["detail"].lower()

        # 3. Also test publishing with weights/ prefix (as sent by ParsedCheckpointTinkerPath)
        resp_pub_prefix = client.post(f"/api/v1/training_runs/{run_id}/checkpoints/weights/{ckpt_id}/publish")
        assert resp_pub_prefix.status_code == 409

        # 4. Unpublish checkpoint
        resp_unpub = client.delete(f"/api/v1/training_runs/{run_id}/checkpoints/{ckpt_id}/publish")
        assert resp_unpub.status_code == 200
        assert resp_unpub.json()["status"] == "unpublished"

        # Verify listed as private
        resp_list2 = client.get(f"/api/v1/training_runs/{run_id}/checkpoints")
        assert resp_list2.status_code == 200
        ckpts2 = resp_list2.json()["checkpoints"]
        assert any(c["checkpoint_id"] == ckpt_id and c["public"] is False for c in ckpts2)

        # 5. Unpublish again -> 409 Conflict
        resp_unpub_again = client.delete(f"/api/v1/training_runs/{run_id}/checkpoints/{ckpt_id}/publish")
        assert resp_unpub_again.status_code == 409
        assert "already private" in resp_unpub_again.json()["detail"].lower()

    @pytest.mark.anyio
    async def test_set_checkpoint_ttl(self, client, config):
        run_id = "run-ttl-1"
        ckpt_id = "step_002"
        ckpt_dir = config.checkpoints_base / run_id / ckpt_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "model.safetensors").write_bytes(b"data")

        # 1. Invalid TTL <= 0 -> 400
        resp_bad = client.put(
            f"/api/v1/training_runs/{run_id}/checkpoints/{ckpt_id}/ttl",
            json={"ttl_seconds": 0},
        )
        assert resp_bad.status_code == 400

        # 2. Valid TTL -> 200
        resp_ok = client.put(
            f"/api/v1/training_runs/{run_id}/checkpoints/{ckpt_id}/ttl",
            json={"ttl_seconds": 3600},
        )
        assert resp_ok.status_code == 200
        assert resp_ok.json()["status"] == "updated"

        # Verify expires_at is set
        resp_list = client.get(f"/api/v1/training_runs/{run_id}/checkpoints")
        ckpts = resp_list.json()["checkpoints"]
        target = next(c for c in ckpts if c["checkpoint_id"] == ckpt_id)
        assert target["expires_at"] is not None

        # 3. Clear TTL (ttl_seconds=None) -> 200
        resp_clear = client.put(
            f"/api/v1/training_runs/{run_id}/checkpoints/{ckpt_id}/ttl",
            json={"ttl_seconds": None},
        )
        assert resp_clear.status_code == 200

        resp_list_cleared = client.get(f"/api/v1/training_runs/{run_id}/checkpoints")
        target_cleared = next(c for c in resp_list_cleared.json()["checkpoints"] if c["checkpoint_id"] == ckpt_id)
        assert target_cleared["expires_at"] is None

    @pytest.mark.anyio
    async def test_delete_checkpoint(self, client, config):
        run_id = "run-del-1"
        ckpt_id = "step_003"
        ckpt_dir = config.checkpoints_base / run_id / ckpt_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "model.safetensors").write_bytes(b"data")

        # Record in DB as well
        async with get_session() as db:
            sess = SessionDB(session_id="sess-del-1")
            db.add(sess)
            m = ModelDB(model_id=run_id, base_model="Qwen/Qwen3.5-0.8B", status="ready", session_id="sess-del-1")
            db.add(m)
            ckpt_db = CheckpointDB(
                model_id=run_id,
                checkpoint_id=ckpt_id,
                checkpoint_type=CheckpointType.TRAINING,
                status=CheckpointStatus.COMPLETED,
            )
            db.add(ckpt_db)
            await db.commit()

        # Delete checkpoint
        resp_del = client.delete(f"/api/v1/training_runs/{run_id}/checkpoints/{ckpt_id}")
        assert resp_del.status_code == 200
        assert resp_del.json()["status"] == "deleted"

        # Verify disk directory was removed
        assert not ckpt_dir.exists()

        # Verify DB row was removed
        resp_list = client.get(f"/api/v1/training_runs/{run_id}/checkpoints")
        assert resp_list.status_code == 200
        assert len(resp_list.json()["checkpoints"]) == 0

        # Deleting again -> 404
        resp_del_again = client.delete(f"/api/v1/training_runs/{run_id}/checkpoints/{ckpt_id}")
        assert resp_del_again.status_code == 404

    @pytest.mark.anyio
    async def test_sdk_direct_client_integration(self, client, app, config):
        """Test with Tinker's native AsyncTinker client directly."""
        import httpx
        from tinker._client import AsyncTinker

        async with get_session() as db:
            sess = SessionDB(session_id="sess-sdk-1")
            db.add(sess)
            m = ModelDB(model_id="run-sdk-1", base_model="Qwen/Qwen3.5-0.8B", status="ready", session_id="sess-sdk-1")
            db.add(m)
            await db.commit()

        (config.checkpoints_base / "run-sdk-1" / "ckpt_sdk").mkdir(parents=True, exist_ok=True)
        (config.checkpoints_base / "run-sdk-1" / "ckpt_sdk" / "model.safetensors").write_bytes(b"data")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
            sdk_client = AsyncTinker(base_url="http://test", api_key="tml-local", http_client=http_client)

            # 1. client.weights.list()
            res = await sdk_client.weights.list("run-sdk-1")
            assert isinstance(res, tinker.types.CheckpointsListResponse)
            assert len(res.checkpoints) == 1
            first_ckpt = res.checkpoints[0]
            first_id = first_ckpt.get("checkpoint_id") if isinstance(first_ckpt, dict) else first_ckpt.checkpoint_id
            assert first_id == "ckpt_sdk"

            # 2. client.get("/api/v1/training_runs")
            runs_res = await sdk_client.get("/api/v1/training_runs", cast_to=tinker.types.TrainingRunsResponse)
            assert isinstance(runs_res, tinker.types.TrainingRunsResponse)
            run_ids = [
                r.get("training_run_id") if isinstance(r, dict) else r.training_run_id
                for r in runs_res.training_runs
            ]
            assert "run-sdk-1" in run_ids

            # 3. client.get("/api/v1/training_runs/run-sdk-1")
            run_res = await sdk_client.get("/api/v1/training_runs/run-sdk-1", cast_to=tinker.types.TrainingRun)
            assert isinstance(run_res, tinker.types.TrainingRun)
            assert run_res.training_run_id == "run-sdk-1"

            # 4. client.weights.delete_checkpoint()
            await sdk_client.weights.delete_checkpoint(model_id="run-sdk-1", checkpoint_id="ckpt_sdk")

            # Verify deletion
            res_after = await sdk_client.weights.list("run-sdk-1")
            assert len(res_after.checkpoints) == 0
