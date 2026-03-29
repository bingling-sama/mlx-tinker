"""Integration tests for the FastAPI Tinker API endpoints.

Uses TestClient with a mock backend to test endpoint routing,
request validation, and future lifecycle without loading a real model.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mlx_tinker.api.server import create_app
from mlx_tinker.config import EngineConfig
from mlx_tinker.db.database import get_session
from mlx_tinker.db.models import FutureDB, SamplingSessionDB


@pytest.fixture
def config(tmp_path):
    return EngineConfig(
        base_model="test-model",
        database_path=tmp_path / "test.db",
        checkpoints_base=tmp_path / "checkpoints",
    )


@pytest.fixture
def app(config):
    """Create app but skip the engine startup (which would try to load the model)."""
    app = create_app(config)
    return app


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    def test_healthz(self, client):
        response = client.get("/api/v1/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "mlx-tinker"
        assert data["backend"] == "mlx"

    def test_server_capabilities(self, client):
        response = client.get("/api/v1/get_server_capabilities")
        assert response.status_code == 200
        data = response.json()
        assert len(data["supported_models"]) == 1
        model = data["supported_models"][0]
        assert model["model_name"] == "test-model"
        assert model["base_model"] == "test-model"


class TestSessionEndpoints:
    def test_create_session(self, client):
        response = client.post(
            "/api/v1/create_session",
            json={"tags": ["test"], "user_metadata": {"key": "val"}, "sdk_version": "0.1.0"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_session_heartbeat(self, client):
        # Create session first
        resp = client.post("/api/v1/create_session", json={"sdk_version": "0.1.0"})
        session_id = resp.json()["session_id"]

        # Heartbeat
        resp = client.post("/api/v1/session_heartbeat", json={"session_id": session_id})
        assert resp.status_code == 200

    def test_create_sampling_session(self, client):
        # Create session first
        resp = client.post("/api/v1/create_session", json={"sdk_version": "0.1.0"})
        session_id = resp.json()["session_id"]

        resp = client.post(
            "/api/v1/create_sampling_session",
            json={"session_id": session_id, "base_model": "test-model"},
        )
        assert resp.status_code == 200
        assert "sampling_session_id" in resp.json()


class TestModelLifecycle:
    def test_create_model(self, client):
        # Create session
        resp = client.post("/api/v1/create_session", json={"sdk_version": "0.1.0"})
        session_id = resp.json()["session_id"]

        resp = client.post(
            "/api/v1/create_model",
            json={
                "session_id": session_id,
                "base_model": "test-model",
                "lora_config": {"rank": 16, "alpha": 32.0, "seed": 42},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "model_id" in data
        assert "request_id" in data

    def test_get_info(self, client):
        # Create session + model
        resp = client.post("/api/v1/create_session", json={"sdk_version": "0.1.0"})
        session_id = resp.json()["session_id"]

        resp = client.post(
            "/api/v1/create_model",
            json={
                "session_id": session_id,
                "base_model": "test-model",
                "lora_config": {"rank": 8, "alpha": 16.0},
            },
        )
        model_id = resp.json()["model_id"]

        resp = client.post("/api/v1/get_info", json={"model_id": model_id})
        assert resp.status_code == 200
        assert resp.json()["model_id"] == model_id

    def test_get_info_not_found(self, client):
        resp = client.post("/api/v1/get_info", json={"model_id": "nonexistent"})
        assert resp.status_code == 404


class TestTrainingEndpoints:
    def _create_model(self, client) -> str:
        resp = client.post("/api/v1/create_session", json={"sdk_version": "0.1.0"})
        session_id = resp.json()["session_id"]
        resp = client.post(
            "/api/v1/create_model",
            json={
                "session_id": session_id,
                "base_model": "test-model",
                "lora_config": {"rank": 8, "alpha": 16.0},
            },
        )
        return resp.json()["model_id"]

    def test_forward_backward_creates_future(self, client):
        model_id = self._create_model(client)
        resp = client.post(
            "/api/v1/forward_backward",
            json={
                "model_id": model_id,
                "forward_backward_input": {
                    "data": [
                        {
                            "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                            "loss_fn_inputs": {
                                "target_tokens": {"data": [2, 3, 4]},
                                "weights": {"data": [1.0, 1.0, 1.0]},
                                "advantages": {"data": [0.0, 0.0, 0.0]},
                                "logprobs": {"data": [0.0, 0.0, 0.0]},
                            },
                        }
                    ],
                    "loss_fn": "cross_entropy",
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data

    def test_optim_step_creates_future(self, client):
        model_id = self._create_model(client)
        resp = client.post(
            "/api/v1/optim_step",
            json={
                "model_id": model_id,
                "adam_params": {"learning_rate": 0.001},
            },
        )
        assert resp.status_code == 200
        assert "request_id" in resp.json()


class TestSamplingEndpoints:
    def test_asample_omitted_prompt_logprobs_does_not_queue_null(self, client):
        resp = client.post("/api/v1/create_session", json={"sdk_version": "0.1.0"})
        session_id = resp.json()["session_id"]

        resp = client.post(
            "/api/v1/create_sampling_session",
            json={"session_id": session_id, "base_model": "test-model"},
        )
        sampling_session_id = resp.json()["sampling_session_id"]

        resp = client.post(
            "/api/v1/asample",
            json={
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                "sampling_params": {"temperature": 0.7, "max_tokens": 8, "seed": 1},
                "sampling_session_id": sampling_session_id,
                "num_samples": 1,
                "type": "sample",
            },
        )
        assert resp.status_code == 200
        request_id = int(resp.json()["request_id"])

        async def _load_future_request_data():
            async with get_session() as session:
                future = await session.get(FutureDB, request_id)
                return future.request_data

        request_data = asyncio.run(_load_future_request_data())
        assert "prompt_logprobs" not in request_data
        assert request_data["sampling_session_id"] == sampling_session_id
        assert request_data["model_path"] is None or isinstance(request_data["model_path"], str)


class TestFutureLifecycle:
    def test_retrieve_pending_future(self, client):
        # Create a future via forward_backward
        resp = client.post("/api/v1/create_session", json={"sdk_version": "0.1.0"})
        session_id = resp.json()["session_id"]
        resp = client.post(
            "/api/v1/create_model",
            json={
                "session_id": session_id,
                "base_model": "test-model",
                "lora_config": {"rank": 8, "alpha": 16.0},
            },
        )
        model_id = resp.json()["model_id"]

        resp = client.post(
            "/api/v1/forward_backward",
            json={
                "model_id": model_id,
                "forward_backward_input": {
                    "data": [
                        {
                            "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
                            "loss_fn_inputs": {
                                "target_tokens": {"data": [2]},
                                "weights": {"data": [1.0]},
                                "advantages": {"data": [0.0]},
                                "logprobs": {"data": [0.0]},
                            },
                        }
                    ],
                    "loss_fn": "cross_entropy",
                },
            },
        )
        request_id = resp.json()["request_id"]

        # Retrieve — should be pending (engine hasn't processed it)
        resp = client.post("/api/v1/retrieve_future", json={"request_id": request_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("type") == "try_again" or "error" in data

    def test_retrieve_nonexistent_future(self, client):
        resp = client.post("/api/v1/retrieve_future", json={"request_id": "99999"})
        assert resp.status_code == 404


class TestTelemetry:
    def test_telemetry(self, client):
        resp = client.post(
            "/api/v1/telemetry", json={"event": "test_event", "data": {"key": "val"}}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_telemetry_sdk_format(self, client):
        resp = client.post(
            "/api/v1/telemetry",
            json={
                "events": [{"event": "SESSION_START", "event_id": "abc", "severity": "INFO"}],
                "platform": "Darwin",
                "sdk_version": "0.16.1",
                "session_id": "test-session-id",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"


class TestRequestValidation:
    def test_create_model_missing_lora_config(self, client):
        resp = client.post("/api/v1/create_session", json={"sdk_version": "0.1.0"})
        session_id = resp.json()["session_id"]

        resp = client.post(
            "/api/v1/create_model",
            json={"session_id": session_id, "base_model": "test-model"},
        )
        assert resp.status_code == 200

    def test_forward_backward_missing_data(self, client):
        resp = client.post(
            "/api/v1/forward_backward",
            json={"model_id": "x", "forward_backward_input": {"loss_fn": "cross_entropy"}},
        )
        assert resp.status_code == 422

    def test_forward_backward_invalid_loss_fn(self, client):
        resp = client.post(
            "/api/v1/forward_backward",
            json={
                "model_id": "x",
                "forward_backward_input": {
                    "data": [
                        {
                            "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
                            "loss_fn_inputs": {
                                "target_tokens": {"data": [2]},
                                "weights": {"data": [1.0]},
                                "advantages": {"data": [0.0]},
                                "logprobs": {"data": [0.0]},
                            },
                        }
                    ],
                    "loss_fn": "invalid_loss",
                },
            },
        )
        assert resp.status_code == 422

    def test_create_model_defaults_base_model(self, client):
        resp = client.post("/api/v1/create_session", json={"sdk_version": "0.1.0"})
        session_id = resp.json()["session_id"]

        resp = client.post(
            "/api/v1/create_model",
            json={
                "session_id": session_id,
                "lora_config": {"rank": 8, "alpha": 16.0},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "model_id" in data
        assert "request_id" in data


class TestMoreEndpoints:
    def _create_model(self, client) -> str:
        resp = client.post("/api/v1/create_session", json={"sdk_version": "0.1.0"})
        session_id = resp.json()["session_id"]
        resp = client.post(
            "/api/v1/create_model",
            json={
                "session_id": session_id,
                "base_model": "test-model",
                "lora_config": {"rank": 8, "alpha": 16.0},
            },
        )
        return resp.json()["model_id"]

    def test_unload_model_creates_future(self, client):
        model_id = self._create_model(client)
        resp = client.post("/api/v1/unload_model", json={"model_id": model_id})
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert data["model_id"] == model_id

    def test_save_weights_creates_future(self, client):
        model_id = self._create_model(client)
        resp = client.post(
            "/api/v1/save_weights",
            json={"model_id": model_id, "path": "checkpoints/test"},
        )
        assert resp.status_code == 200
        assert "request_id" in resp.json()

    def test_sample_creates_future(self, client):
        resp = client.post(
            "/api/v1/asample",
            json={
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                "sampling_params": {"temperature": 1.0, "max_tokens": 5},
            },
        )
        assert resp.status_code == 200
        assert "request_id" in resp.json()

    def test_sample_preserves_model_id(self, client):
        model_id = self._create_model(client)
        resp = client.post(
            "/api/v1/asample",
            json={
                "model_id": model_id,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                "sampling_params": {"temperature": 1.0, "max_tokens": 5},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model_id"] == model_id

    def test_save_weights_for_sampler_creates_sampling_session(self, client, config):
        model_id = self._create_model(client)
        resp = client.post(
            "/api/v1/save_weights_for_sampler",
            json={"model_id": model_id, "sampling_session_seq_id": 0},
        )
        assert resp.status_code == 200
        request_id = int(resp.json()["request_id"])

        async def _load_rows():
            async with get_session() as session:
                future = await session.get(FutureDB, request_id)
                assert future is not None
                sampling_session_id = future.request_data["sampling_session_id"]
                sampling_session = await session.get(SamplingSessionDB, sampling_session_id)
                return future, sampling_session

        future, sampling_session = asyncio.run(_load_rows())
        assert future.request_data["path"].startswith(str(config.checkpoints_base / model_id / "sampler"))
        assert future.request_data["ephemeral"] is True
        assert sampling_session is not None
        assert sampling_session.model_path == future.request_data["path"]

    def test_sample_resolves_sampling_session_path(self, client):
        session_resp = client.post("/api/v1/create_session", json={"sdk_version": "0.1.0"})
        session_id = session_resp.json()["session_id"]
        sampling_resp = client.post(
            "/api/v1/create_sampling_session",
            json={
                "session_id": session_id,
                "base_model": "test-model",
                "model_path": "checkpoints/sampler-1",
            },
        )
        sampling_session_id = sampling_resp.json()["sampling_session_id"]

        resp = client.post(
            "/api/v1/asample",
            json={
                "sampling_session_id": sampling_session_id,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                "sampling_params": {"temperature": 1.0, "max_tokens": 5},
            },
        )
        assert resp.status_code == 200
        request_id = int(resp.json()["request_id"])
        assert resp.json()["model_id"] is None

        async def _load_future():
            async with get_session() as session:
                return await session.get(FutureDB, request_id)

        future = asyncio.run(_load_future())
        assert future is not None
        assert future.request_data["model_path"] == "checkpoints/sampler-1"
        assert future.request_data["base_model"] == "test-model"
