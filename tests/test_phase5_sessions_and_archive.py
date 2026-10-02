"""Unit and integration tests for Phase 5: Sessions & Checkpoint Archive Downloads.

Tests:
- GET /api/v1/sessions/{session_id} -> GetSessionResponse(training_run_ids, sampler_ids)
- GET /api/v1/sessions -> ListSessionsResponse(sessions) with pagination
- GET /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive -> 302 Redirect with Location header
- GET /api/v1/archives/{archive_id}/download -> 200 OK downloading .tar.gz archive
- Tinker SDK weights.get_checkpoint_archive_url compatibility
- Tinker SDK model validation for GetSessionResponse and ListSessionsResponse
"""

from datetime import datetime, timezone
import io
import tarfile
import httpx
import pytest
from starlette.testclient import TestClient
import tinker
import tinker.types

from mlx_tinker.api.models import GetSessionResponse, ListSessionsResponse
from mlx_tinker.api.server import create_app
from mlx_tinker.config import EngineConfig
from mlx_tinker.db.database import get_session
from mlx_tinker.db.models import ModelDB, SamplingSessionDB, SessionDB


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


class TestSessionsEndpoints:
    """Tests for GET /api/v1/sessions/{session_id} and GET /api/v1/sessions."""

    @pytest.mark.anyio
    async def test_get_session_not_found(self, client):
        resp = client.get("/api/v1/sessions/non-existent-session")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.anyio
    async def test_get_session_success(self, client):
        async with get_session() as db:
            sess = SessionDB(
                session_id="sess-phase5-1",
                user_metadata={"proj": "test"},
            )
            db.add(sess)

            # Add two models
            m1 = ModelDB(
                model_id="run-p5-1",
                base_model="Qwen/Qwen3.5-0.8B",
                lora_config={},
                status="ready",
                session_id="sess-phase5-1",
            )
            m2 = ModelDB(
                model_id="run-p5-2",
                base_model="Qwen/Qwen3.5-2B",
                lora_config={},
                status="ready",
                session_id="sess-phase5-1",
            )
            db.add(m1)
            db.add(m2)

            # Add a sampler
            ss = SamplingSessionDB(
                sampling_session_id="sampler-p5-1",
                session_id="sess-phase5-1",
                model_id="run-p5-1",
            )
            db.add(ss)
            await db.commit()

        resp = client.get("/api/v1/sessions/sess-phase5-1")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data["training_run_ids"]) == {"run-p5-1", "run-p5-2"}
        assert data["sampler_ids"] == ["sampler-p5-1"]

        # Validate with official Tinker SDK model
        sdk_obj = tinker.types.GetSessionResponse(**data)
        assert set(sdk_obj.training_run_ids) == {"run-p5-1", "run-p5-2"}
        assert sdk_obj.sampler_ids == ["sampler-p5-1"]

    @pytest.mark.anyio
    async def test_list_sessions(self, client):
        async with get_session() as db:
            for i in range(5):
                db.add(SessionDB(session_id=f"sess-list-{i}"))
            await db.commit()

        # Test pagination limit/offset
        resp = client.get("/api/v1/sessions?limit=3&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert len(data["sessions"]) == 3

        # Validate with official Tinker SDK model
        sdk_obj = tinker.types.ListSessionsResponse(**data)
        assert len(sdk_obj.sessions) == 3


class TestCheckpointArchiveEndpoints:
    """Tests for GET /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive."""

    @pytest.mark.anyio
    async def test_archive_not_found(self, client):
        resp = client.get(
            "/api/v1/training_runs/non-existent-model/checkpoints/chk-1/archive",
            follow_redirects=False,
        )
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_archive_redirect_and_download_flow(self, client, config):
        # 1. Create a checkpoint directory with some test files
        model_id = "run-archive-test"
        ckpt_id = "step_0042"
        ckpt_dir = config.checkpoints_base / model_id / ckpt_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "adapters.safetensors").write_bytes(b"test-lora-weights-12345")
        (ckpt_dir / "config.json").write_text('{"rank": 8}')

        # 2. Call archive endpoint without following redirect
        resp = client.get(
            f"/api/v1/training_runs/{model_id}/checkpoints/{ckpt_id}/archive",
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "Location" in resp.headers
        assert "Expires" in resp.headers
        location = resp.headers["Location"]

        # Parse expires header as expected by Tinker SDK: "%a, %d %b %Y %H:%M:%S GMT"
        expires_str = resp.headers["Expires"]
        expires_dt = datetime.strptime(expires_str, "%a, %d %b %Y %H:%M:%S GMT")
        assert expires_dt is not None

        # Validate with official Tinker SDK CheckpointArchiveUrlResponse
        archive_resp = tinker.types.CheckpointArchiveUrlResponse(
            url=location,
            expires=expires_dt,
        )
        assert archive_resp.url == location

        # 3. Follow download URL
        dl_resp = client.get(location)
        assert dl_resp.status_code == 200
        assert dl_resp.headers["content-type"] == "application/gzip"

        # Check tar.gz archive content
        with tarfile.open(fileobj=io.BytesIO(dl_resp.content), mode="r:gz") as tar:
            names = tar.getnames()
            assert any("adapters.safetensors" in name for name in names)
            assert any("config.json" in name for name in names)

            # Verify file content inside archive
            adapter_member = next(m for m in tar.getmembers() if "adapters.safetensors" in m.name)
            extracted = tar.extractfile(adapter_member).read()
            assert extracted == b"test-lora-weights-12345"

    @pytest.mark.anyio
    async def test_archive_with_weights_prefix_and_sampler(self, client, config):
        model_id = "run-archive-sampler"
        ckpt_id = "export_1"
        sampler_dir = config.checkpoints_base / model_id / "sampler" / ckpt_id
        sampler_dir.mkdir(parents=True, exist_ok=True)
        (sampler_dir / "model.safetensors").write_bytes(b"sampler-data")

        # Test with sampler_weights/ prefix
        resp = client.get(
            f"/api/v1/training_runs/{model_id}/checkpoints/sampler_weights/{ckpt_id}/archive",
            follow_redirects=False,
        )
        assert resp.status_code == 302
        location = resp.headers["Location"]

        dl_resp = client.get(location)
        assert dl_resp.status_code == 200
        with tarfile.open(fileobj=io.BytesIO(dl_resp.content), mode="r:gz") as tar:
            assert any("model.safetensors" in name for name in tar.getnames())

    @pytest.mark.anyio
    async def test_sdk_weights_resource_archive_url_flow(self, client, config):
        """Verify official SDK AsyncWeightsResource.get_checkpoint_archive_url pattern."""
        model_id = "run-sdk-archive"
        ckpt_id = "step_0100"
        ckpt_dir = config.checkpoints_base / model_id / ckpt_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "adapters.safetensors").write_bytes(b"sdk-test-bytes")

        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
            resp = await async_client.get(
                f"/api/v1/training_runs/{model_id}/checkpoints/{ckpt_id}/archive",
                headers={"accept": "application/gzip"},
                follow_redirects=False,
            )
            assert resp.status_code == 302
            location = resp.headers.get("Location")
            assert location is not None
            expires_header = resp.headers.get("Expires")
            assert expires_header is not None
            expires = datetime.strptime(expires_header, "%a, %d %b %Y %H:%M:%S GMT")

            sdk_result = tinker.types.CheckpointArchiveUrlResponse(
                url=location,
                expires=expires,
            )
            assert sdk_result.url.startswith("http://testserver/api/v1/archives/")
