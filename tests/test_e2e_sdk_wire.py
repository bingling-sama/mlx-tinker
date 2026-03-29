"""End-to-end wire-format test: validates that request/response shapes match
what the tinker SDK sends and expects.

Tests the exact JSON shapes for the full SFT + RL training loop as the
tinker SDK would execute them, without requiring a running model or the
tinker SDK itself.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlx_tinker.api.server import create_app
from mlx_tinker.config import EngineConfig


@pytest.fixture
def config(tmp_path):
    return EngineConfig(
        base_model="test-model",
        database_path=tmp_path / "test.db",
        checkpoints_base=tmp_path / "checkpoints",
    )


@pytest.fixture
def client(config):
    app = create_app(config)
    with TestClient(app) as c:
        yield c


class TestSDKWireProtocol:
    """Validate the exact JSON shapes the tinker SDK sends/receives."""

    def _create_session(self, client) -> str:
        resp = client.post("/api/v1/create_session", json={
            "tags": ["test"],
            "user_metadata": {"project": "test"},
            "sdk_version": "0.16.1",
            "project_id": "test-project",
            "type": "create_session",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["type"] == "create_session"
        return data["session_id"]

    def _create_model(self, client, session_id: str) -> str:
        resp = client.post("/api/v1/create_model", json={
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": "test-model",
            "lora_config": {
                "rank": 8,
                "alpha": 16.0,
                "seed": 42,
                "train_attn": True,
                "train_mlp": True,
                "train_unembed": False,
            },
            "user_metadata": None,
            "type": "create_model",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert "model_id" in data
        return data["model_id"], data["request_id"]

    def test_create_session_wire_format(self, client):
        """SDK sends type discriminator and project_id."""
        resp = client.post("/api/v1/create_session", json={
            "tags": [],
            "sdk_version": "0.16.1",
            "project_id": "my-project",
            "type": "create_session",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "create_session"
        assert "session_id" in data
        assert "info_message" in data or data.get("info_message") is None

    def test_create_sampling_session_wire_format(self, client):
        """SDK sends sampling_session_seq_id and type."""
        session_id = self._create_session(client)
        resp = client.post("/api/v1/create_sampling_session", json={
            "session_id": session_id,
            "sampling_session_seq_id": 0,
            "base_model": "test-model",
            "type": "create_sampling_session",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "sampling_session_id" in data
        assert data["type"] == "create_sampling_session"

    def test_create_model_returns_untyped_api_future(self, client):
        """create_model should return UntypedAPIFuture with request_id."""
        session_id = self._create_session(client)
        _, request_id = self._create_model(client, session_id)
        assert request_id is not None
        assert len(request_id) > 0

    def test_forward_backward_nested_input(self, client):
        """SDK nests data inside forward_backward_input key."""
        session_id = self._create_session(client)
        model_id, _ = self._create_model(client, session_id)

        resp = client.post("/api/v1/forward_backward", json={
            "forward_backward_input": {
                "data": [{
                    "model_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                    },
                    "loss_fn_inputs": {
                        "target_tokens": {"data": [2, 3, 4], "dtype": "int64"},
                        "weights": {"data": [1.0, 1.0, 1.0], "dtype": "float32"},
                    },
                }],
                "loss_fn": "cross_entropy",
            },
            "model_id": model_id,
            "seq_id": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data

    def test_forward_backward_optional_fields(self, client):
        """SDK sends only target_tokens for CE (weights/advantages/logprobs optional)."""
        session_id = self._create_session(client)
        model_id, _ = self._create_model(client, session_id)

        resp = client.post("/api/v1/forward_backward", json={
            "forward_backward_input": {
                "data": [{
                    "model_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                    },
                    "loss_fn_inputs": {
                        "target_tokens": {"data": [2, 3, 4], "dtype": "int64"},
                    },
                }],
                "loss_fn": "cross_entropy",
            },
            "model_id": model_id,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data

    def test_importance_sampling_with_advantages(self, client):
        """SDK sends target_tokens + logprobs + advantages for IS loss."""
        session_id = self._create_session(client)
        model_id, _ = self._create_model(client, session_id)

        resp = client.post("/api/v1/forward_backward", json={
            "forward_backward_input": {
                "data": [{
                    "model_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                    },
                    "loss_fn_inputs": {
                        "target_tokens": {"data": [2, 3, 4], "dtype": "int64"},
                        "logprobs": {"data": [-1.0, -2.0, -1.5], "dtype": "float32"},
                        "advantages": {"data": [0.0, 1.0, 1.0], "dtype": "float32"},
                    },
                }],
                "loss_fn": "importance_sampling",
            },
            "model_id": model_id,
        })
        assert resp.status_code == 200

    def test_dro_loss_accepted(self, client):
        """DRO loss function should be accepted."""
        session_id = self._create_session(client)
        model_id, _ = self._create_model(client, session_id)

        resp = client.post("/api/v1/forward_backward", json={
            "forward_backward_input": {
                "data": [{
                    "model_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                    },
                    "loss_fn_inputs": {
                        "target_tokens": {"data": [2, 3, 4], "dtype": "int64"},
                        "logprobs": {"data": [-1.0, -2.0, -1.5], "dtype": "float32"},
                        "advantages": {"data": [0.0, 1.0, 1.0], "dtype": "float32"},
                    },
                }],
                "loss_fn": "dro",
                "loss_fn_config": {"beta": 0.1},
            },
            "model_id": model_id,
        })
        assert resp.status_code == 200

    def test_optim_step_with_seq_id(self, client):
        """SDK sends seq_id and type discriminator."""
        session_id = self._create_session(client)
        model_id, _ = self._create_model(client, session_id)

        resp = client.post("/api/v1/optim_step", json={
            "adam_params": {
                "learning_rate": 1e-4,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
            "model_id": model_id,
            "seq_id": 1,
            "type": "optim_step",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data

    def test_retrieve_future_pending_returns_try_again(self, client):
        """Pending future should return TryAgainResponse."""
        session_id = self._create_session(client)
        model_id, _ = self._create_model(client, session_id)

        resp = client.post("/api/v1/forward_backward", json={
            "forward_backward_input": {
                "data": [{
                    "model_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [1]}]
                    },
                    "loss_fn_inputs": {
                        "target_tokens": {"data": [2], "dtype": "int64"},
                    },
                }],
                "loss_fn": "cross_entropy",
            },
            "model_id": model_id,
        })
        request_id = resp.json()["request_id"]

        resp = client.post("/api/v1/retrieve_future", json={
            "request_id": request_id,
        })
        assert resp.status_code == 200
        data = resp.json()
        if data.get("type") == "try_again":
            assert data["request_id"] == request_id
            assert data["queue_state"] in ("active", "paused_capacity", "paused_rate_limit")
        else:
            assert "error" in data

    def test_get_info_response_shape(self, client):
        """get_info should return Tinker-compatible model info."""
        session_id = self._create_session(client)
        model_id, _ = self._create_model(client, session_id)

        resp = client.post("/api/v1/get_info", json={"model_id": model_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == model_id
        assert "model_data" in data
        assert "is_lora" in data
        assert "lora_rank" in data
        assert data["type"] == "get_info"

    def test_save_weights_for_sampler_with_ttl(self, client):
        """SDK sends ttl_seconds."""
        session_id = self._create_session(client)
        model_id, _ = self._create_model(client, session_id)

        resp = client.post("/api/v1/save_weights_for_sampler", json={
            "model_id": model_id,
            "ttl_seconds": 3600,
            "type": "save_weights_for_sampler",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data

    def test_sample_request_with_type(self, client):
        """SDK sends type discriminator on sample."""
        resp = client.post("/api/v1/asample", json={
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
            "sampling_params": {"temperature": 1.0, "max_tokens": 10},
            "num_samples": 1,
            "type": "sample",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data

    def test_server_capabilities_wire_format(self, client):
        """get_server_capabilities should use model_name field."""
        resp = client.get("/api/v1/get_server_capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["supported_models"]) >= 1
        model = data["supported_models"][0]
        assert "model_name" in model

    def test_full_sft_loop_wire_format(self, client):
        """Simulate a full SFT training loop as the SDK would execute it."""
        session_id = self._create_session(client)
        model_id, create_request_id = self._create_model(client, session_id)

        # Retrieve the create_model future (will be pending — engine not running)
        resp = client.post("/api/v1/retrieve_future", json={
            "request_id": create_request_id,
        })
        assert resp.status_code == 200

        # Submit forward_backward
        fb_resp = client.post("/api/v1/forward_backward", json={
            "forward_backward_input": {
                "data": [{
                    "model_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3, 4, 5]}]
                    },
                    "loss_fn_inputs": {
                        "target_tokens": {
                            "data": [2, 3, 4, 5, 6],
                            "dtype": "int64",
                        },
                        "weights": {
                            "data": [0.0, 1.0, 1.0, 1.0, 1.0],
                            "dtype": "float32",
                        },
                    },
                }],
                "loss_fn": "cross_entropy",
            },
            "model_id": model_id,
            "seq_id": 0,
        })
        assert fb_resp.status_code == 200
        fb_data = fb_resp.json()
        assert "request_id" in fb_data

        # Submit optim_step
        opt_resp = client.post("/api/v1/optim_step", json={
            "adam_params": {"learning_rate": 1e-4},
            "model_id": model_id,
            "seq_id": 0,
            "type": "optim_step",
        })
        assert opt_resp.status_code == 200
        assert "request_id" in opt_resp.json()

    def test_full_rl_loop_wire_format(self, client):
        """Simulate a full RL loop: sample → compute rewards → IS training."""
        session_id = self._create_session(client)
        model_id, _ = self._create_model(client, session_id)

        # Submit sample request
        sample_resp = client.post("/api/v1/asample", json={
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
            "sampling_params": {"temperature": 0.8, "max_tokens": 10},
            "num_samples": 2,
            "type": "sample",
        })
        assert sample_resp.status_code == 200
        assert "request_id" in sample_resp.json()

        # Submit IS forward_backward (simulating post-reward training)
        fb_resp = client.post("/api/v1/forward_backward", json={
            "forward_backward_input": {
                "data": [{
                    "model_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3, 4, 5]}]
                    },
                    "loss_fn_inputs": {
                        "target_tokens": {
                            "data": [2, 3, 4, 5, 6],
                            "dtype": "int64",
                        },
                        "logprobs": {
                            "data": [-1.0, -2.0, -1.5, -1.2, -0.8],
                            "dtype": "float32",
                        },
                        "advantages": {
                            "data": [0.0, 0.0, 1.0, 1.0, 1.0],
                            "dtype": "float32",
                        },
                    },
                }],
                "loss_fn": "importance_sampling",
            },
            "model_id": model_id,
            "seq_id": 1,
        })
        assert fb_resp.status_code == 200
        assert "request_id" in fb_resp.json()

        # Submit optim_step
        opt_resp = client.post("/api/v1/optim_step", json={
            "adam_params": {"learning_rate": 5e-5},
            "model_id": model_id,
            "seq_id": 1,
            "type": "optim_step",
        })
        assert opt_resp.status_code == 200
        assert "request_id" in opt_resp.json()

    def test_tensor_data_with_dtype_and_shape(self, client):
        """SDK sends dtype and shape fields on TensorData."""
        session_id = self._create_session(client)
        model_id, _ = self._create_model(client, session_id)

        resp = client.post("/api/v1/forward_backward", json={
            "forward_backward_input": {
                "data": [{
                    "model_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [1, 2]}]
                    },
                    "loss_fn_inputs": {
                        "target_tokens": {
                            "data": [2, 3],
                            "dtype": "int64",
                            "shape": [2],
                        },
                        "weights": {
                            "data": [1.0, 1.0],
                            "dtype": "float32",
                            "shape": [2],
                        },
                    },
                }],
                "loss_fn": "cross_entropy",
            },
            "model_id": model_id,
        })
        assert resp.status_code == 200
