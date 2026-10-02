"""Unit tests for Phase 1: Schema & Data Model Alignment (Pydantic Models)."""

from datetime import datetime, timezone
import pytest
from starlette.testclient import TestClient

from mlx_tinker.api.models import (
    Checkpoint,
    CheckpointArchiveUrlResponse,
    CheckpointsListResponse,
    Cursor,
    FutureRetrieveRequest,
    GetInfoRequest,
    GetSamplerResponse,
    GetSessionResponse,
    ListCheckpointsResponse,
    ListSessionsResponse,
    LoadWeightsResponse,
    RequestFailedResponse,
    RetrieveFutureRequest,
    SaveWeightsRequest,
    SessionHeartbeatRequest,
    SessionHeartbeatResponse,
    TrainingRun,
    UnloadModelRequest,
    WeightsInfoRequest,
    WeightsInfoResponse,
)
from mlx_tinker.api.server import create_app
from mlx_tinker.config import EngineConfig
from mlx_tinker.types import RequestStatus


@pytest.fixture
def config(tmp_path):
    return EngineConfig(
        base_model="test-model",
        database_path=tmp_path / "test.db",
        checkpoints_base=tmp_path / "checkpoints",
    )


@pytest.fixture
def app(config):
    return create_app(config)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


class TestPhase1ModelsAlignment:
    """Verifies all Phase 1 schema changes match official Tinker specifications."""

    def test_save_weights_request_path_optional_and_ttl(self):
        # path is optional and defaults to None
        req = SaveWeightsRequest(model_id="test-model")
        assert req.model_id == "test-model"
        assert req.path is None
        assert req.ttl_seconds is None
        assert req.type == "save_weights"

        # path and ttl_seconds can be specified
        req2 = SaveWeightsRequest(
            model_id="test-model",
            path="weights/step_1",
            ttl_seconds=3600,
        )
        assert req2.path == "weights/step_1"
        assert req2.ttl_seconds == 3600

    def test_load_weights_response_has_path(self):
        resp = LoadWeightsResponse()
        assert resp.path is None

        resp_with_path = LoadWeightsResponse(path="tinker://test-model/weights/0")
        assert resp_with_path.path == "tinker://test-model/weights/0"
        assert resp_with_path.type == "load_weights" or resp_with_path.type is None

    def test_session_heartbeat_response_type(self):
        resp = SessionHeartbeatResponse()
        assert resp.type == "session_heartbeat"
        assert resp.model_dump() == {"type": "session_heartbeat"}

    def test_training_run_fields_alignment(self):
        now = datetime.now(timezone.utc)
        run = TrainingRun(
            training_run_id="run-123",
            base_model="Qwen/Qwen3.5-0.8B",
            model_owner="user1",
            is_lora=True,
            corrupted=False,
            lora_rank=8,
            last_request_time=now,
            user_metadata={"tag": "experiment-1"},
        )
        data = run.model_dump()
        assert data["training_run_id"] == "run-123"
        assert data["base_model"] == "Qwen/Qwen3.5-0.8B"
        assert data["model_owner"] == "user1"
        assert data["is_lora"] is True
        assert data["corrupted"] is False
        assert data["lora_rank"] == 8
        assert data["last_checkpoint"] is None
        assert data["last_sampler_checkpoint"] is None
        assert data["user_metadata"] == {"tag": "experiment-1"}

    def test_checkpoint_fields_alignment(self):
        now = datetime.now(timezone.utc)
        ckpt = Checkpoint(
            checkpoint_id="chk-001",
            checkpoint_type="training",
            time=now,
            tinker_path="tinker://run-123/weights/chk-001",
            size_bytes=1048576,
            public=True,
            expires_at=None,
        )
        data = ckpt.model_dump()
        assert data["checkpoint_id"] == "chk-001"
        assert data["checkpoint_type"] == "training"
        assert data["tinker_path"] == "tinker://run-123/weights/chk-001"
        assert data["size_bytes"] == 1048576
        assert data["public"] is True
        assert data["expires_at"] is None

    def test_cursor_total_count(self):
        cursor = Cursor(offset=0, limit=20, total_count=100)
        assert cursor.offset == 0
        assert cursor.limit == 20
        assert cursor.total_count == 100

        # default total_count
        cursor_default = Cursor(offset=10, limit=10)
        assert cursor_default.total_count == 0

    def test_weights_info_response_fields(self):
        resp = WeightsInfoResponse(
            base_model="Qwen/Qwen3.5-0.8B",
            is_lora=True,
            lora_rank=16,
            train_attn=True,
            train_mlp=False,
            train_unembed=False,
        )
        data = resp.model_dump()
        assert data["base_model"] == "Qwen/Qwen3.5-0.8B"
        assert data["is_lora"] is True
        assert data["lora_rank"] == 16
        assert data["train_attn"] is True
        assert data["train_mlp"] is False
        assert data["train_unembed"] is False

    def test_official_tinker_sdk_interop(self):
        """Verify that models serialized by mlx_tinker can be validated by official tinker SDK."""
        import tinker

        # 1. TrainingRun
        now = datetime.now(timezone.utc)
        tr = TrainingRun(
            training_run_id="run-sdk-1",
            base_model="Qwen/Qwen3.5-0.8B",
            model_owner="user1",
            is_lora=True,
            last_request_time=now,
        )
        tinker_tr = tinker.TrainingRun.model_validate(tr.model_dump())
        assert tinker_tr.training_run_id == "run-sdk-1"

        # 2. Checkpoint
        ckpt = Checkpoint(
            checkpoint_id="chk-1",
            checkpoint_type="training",
            time=now,
            tinker_path="tinker://run-sdk-1/weights/chk-1",
            size_bytes=1000,
        )
        tinker_ckpt = tinker.Checkpoint.model_validate(ckpt.model_dump())
        assert tinker_ckpt.checkpoint_id == "chk-1"

        # 3. Cursor
        cursor = Cursor(offset=10, limit=20, total_count=50)
        from tinker.types.cursor import Cursor as TinkerCursor
        tinker_cursor = TinkerCursor.model_validate(cursor.model_dump())
        assert tinker_cursor.total_count == 50

        # 4. WeightsInfoResponse
        wir = WeightsInfoResponse(
            base_model="Qwen/Qwen3.5-0.8B",
            is_lora=True,
            lora_rank=8,
            train_attn=True,
            train_mlp=False,
            train_unembed=False,
        )
        from tinker.types.weights_info_response import WeightsInfoResponse as TinkerWIR
        tinker_wir = TinkerWIR.model_validate(wir.model_dump())
        assert tinker_wir.train_attn is True

        # 5. SessionHeartbeatResponse
        shb = SessionHeartbeatResponse()
        from tinker.types.session_heartbeat_response import SessionHeartbeatResponse as TinkerSHB
        tinker_shb = TinkerSHB.model_validate(shb.model_dump())
        assert tinker_shb.type == "session_heartbeat"

        # 6. RequestFailedResponse
        rf = RequestFailedResponse(error="some error", category="server")
        from tinker.types.request_failed_response import RequestFailedResponse as TinkerRF
        tinker_rf = TinkerRF.model_validate(rf.model_dump())
        assert tinker_rf.category.value == "server"

        # 7. CheckpointsListResponse
        clr = CheckpointsListResponse(checkpoints=[ckpt], cursor=cursor)
        from tinker.types.checkpoints_list_response import CheckpointsListResponse as TinkerCLR
        tinker_clr = TinkerCLR.model_validate(clr.model_dump())
        assert len(tinker_clr.checkpoints) == 1
        assert tinker_clr.cursor.total_count == 50

        # 8. GetSamplerResponse
        gsr = GetSamplerResponse(sampler_id="s-1", base_model="Qwen/Qwen3.5-0.8B", model_path="path/1")
        from tinker.types.get_sampler_response import GetSamplerResponse as TinkerGSR
        tinker_gsr = TinkerGSR.model_validate(gsr.model_dump())
        assert tinker_gsr.sampler_id == "s-1"

        # 9. GetSessionResponse & ListSessionsResponse
        gsess = GetSessionResponse(training_run_ids=["r-1"], sampler_ids=["s-1"])
        from tinker.types.get_session_response import GetSessionResponse as TinkerGSess
        tinker_gsess = TinkerGSess.model_validate(gsess.model_dump())
        assert tinker_gsess.training_run_ids == ["r-1"]

        lsess = ListSessionsResponse(sessions=["sess-1", "sess-2"])
        from tinker.types.list_sessions_response import ListSessionsResponse as TinkerLSess
        tinker_lsess = TinkerLSess.model_validate(lsess.model_dump())
        assert tinker_lsess.sessions == ["sess-1", "sess-2"]

    def test_request_types_and_wire_alignment(self):
        # SessionHeartbeatRequest has type
        hb_req = SessionHeartbeatRequest(session_id="s1")
        assert hb_req.type == "session_heartbeat"

        # GetInfoRequest has type
        gi_req = GetInfoRequest(model_id="m1")
        assert gi_req.type == "get_info"

        # UnloadModelRequest has type
        ul_req = UnloadModelRequest(model_id="m1")
        assert ul_req.type == "unload_model"

        # WeightsInfoRequest accepts tinker_path and model_path
        wi_req = WeightsInfoRequest(tinker_path="tinker://m1/weights/001")
        assert wi_req.path == "tinker://m1/weights/001"
        assert wi_req.tinker_path == "tinker://m1/weights/001"

        wi_req2 = WeightsInfoRequest(model_path="checkpoints/m1/001")
        assert wi_req2.path == "checkpoints/m1/001"

        # FutureRetrieveRequest alias
        assert FutureRetrieveRequest is RetrieveFutureRequest
        assert CheckpointsListResponse is ListCheckpointsResponse


def test_api_session_heartbeat_wire_format(client: TestClient):
    """Test /api/v1/session_heartbeat endpoint returns type: session_heartbeat."""
    resp = client.post("/api/v1/create_session", json={"sdk_version": "0.1.0"})
    session_id = resp.json()["session_id"]

    hb_resp = client.post("/api/v1/session_heartbeat", json={"session_id": session_id})
    assert hb_resp.status_code == 200
    assert hb_resp.json() == {"type": "session_heartbeat"}


def test_api_retrieve_future_failure_category(client: TestClient):
    """Test /api/v1/retrieve_future returns category 'server' on failed futures instead of 'execution_error'."""
    from mlx_tinker.db.database import get_session
    from mlx_tinker.db.models import FutureDB

    async def _insert_failed_future():
        async with get_session() as db:
            future = FutureDB(
                model_id="test-model",
                request_type="forward_backward",
                status=RequestStatus.FAILED.value,
                error_message="Simulated backward failed",
                result_data={"error": "Simulated backward failed"},
            )
            db.add(future)
            await db.commit()
            await db.refresh(future)
            return future.request_id

    import asyncio
    req_id = asyncio.run(_insert_failed_future())

    resp = client.post("/api/v1/retrieve_future", json={"request_id": str(req_id)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == "Simulated backward failed"
    assert data["category"] == "server"
    assert data["category"] != "execution_error"
