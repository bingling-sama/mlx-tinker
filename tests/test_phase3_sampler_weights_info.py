"""Unit tests for Phase 3: Sampler & Weight Metadata Endpoints.

Tests:
- GET /api/v1/samplers/{sampler_id} -> GetSamplerResponse
- POST /api/v1/weights_info -> WeightsInfoResponse
- SDK RestClient / SamplingClient compatibility for these endpoints
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from starlette.testclient import TestClient
import tinker

from mlx_tinker.api.models import GetSamplerResponse, WeightsInfoResponse
from mlx_tinker.api.server import create_app
from mlx_tinker.backend.uri import format_tinker_path
from mlx_tinker.config import EngineConfig
from mlx_tinker.db.database import get_session
from mlx_tinker.db.models import SamplingSessionDB, SessionDB
from mlx_tinker.types import CheckpointType, LoraConfig, SaveWeightsForSamplerInput, SaveWeightsInput


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


class TestGetSamplerEndpoint:
    """Test GET /api/v1/samplers/{sampler_id}."""

    def test_get_sampler_not_found(self, client):
        resp = client.get("/api/v1/samplers/non-existent-id")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.anyio
    async def test_get_sampler_success(self, client):
        # Create a session and sampling session in the DB
        async with get_session() as db:
            session = SessionDB(session_id="sess-1")
            db.add(session)
            ss = SamplingSessionDB(
                sampling_session_id="sampler-xyz",
                session_id="sess-1",
                base_model="Qwen/Qwen3.5-2B",
                model_path="tinker://run-1/sampler_weights/export-1",
            )
            db.add(ss)
            await db.commit()

        resp = client.get("/api/v1/samplers/sampler-xyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sampler_id"] == "sampler-xyz"
        assert data["base_model"] == "Qwen/Qwen3.5-2B"
        assert data["model_path"] == "tinker://run-1/sampler_weights/export-1"

        # Validate with Tinker SDK type
        sdk_obj = tinker.types.GetSamplerResponse(**data)
        assert sdk_obj.sampler_id == "sampler-xyz"
        assert sdk_obj.base_model == "Qwen/Qwen3.5-2B"
        assert sdk_obj.model_path == "tinker://run-1/sampler_weights/export-1"

    @pytest.mark.anyio
    async def test_get_sampler_fallback_base_model(self, client, config):
        async with get_session() as db:
            session = SessionDB(session_id="sess-2")
            db.add(session)
            ss = SamplingSessionDB(
                sampling_session_id="sampler-fallback",
                session_id="sess-2",
                base_model=None,
                model_path=None,
            )
            db.add(ss)
            await db.commit()

        resp = client.get("/api/v1/samplers/sampler-fallback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sampler_id"] == "sampler-fallback"
        assert data["base_model"] == config.base_model
        assert data["model_path"] is None


class TestWeightsInfoEndpoint:
    """Test POST /api/v1/weights_info."""

    def test_weights_info_missing_path(self, client):
        resp = client.post("/api/v1/weights_info", json={})
        assert resp.status_code == 400

    def test_weights_info_non_existent_path(self, client):
        resp = client.post(
            "/api/v1/weights_info",
            json={"tinker_path": "tinker://run-non-existent/weights/step-1"},
        )
        assert resp.status_code == 404

    def test_weights_info_sampler_checkpoint(self, client, config):
        # Create a simulated sampler checkpoint
        ckpt_dir = config.checkpoints_base / "run-test" / "sampler" / "export-1"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "adapters.safetensors").write_bytes(b"dummy")
        config_data = {
            "base_model": "Qwen/Qwen3.5-4B",
            "lora_config": {
                "rank": 16,
                "alpha": 32.0,
                "train_attn": True,
                "train_mlp": False,
                "train_unembed": True,
            },
        }
        (ckpt_dir / "config.json").write_text(json.dumps(config_data))

        # Query using tinker:// URI
        tinker_path = format_tinker_path("run-test", "export-1", CheckpointType.SAMPLER)
        resp = client.post("/api/v1/weights_info", json={"tinker_path": tinker_path})
        assert resp.status_code == 200
        data = resp.json()
        assert data["base_model"] == "Qwen/Qwen3.5-4B"
        assert data["is_lora"] is True
        assert data["lora_rank"] == 16
        assert data["train_attn"] is True
        assert data["train_mlp"] is False
        assert data["train_unembed"] is True

        # Validate with Tinker SDK type
        sdk_obj = tinker.types.WeightsInfoResponse(**data)
        assert sdk_obj.base_model == "Qwen/Qwen3.5-4B"
        assert sdk_obj.is_lora is True
        assert sdk_obj.lora_rank == 16
        assert sdk_obj.train_attn is True
        assert sdk_obj.train_mlp is False
        assert sdk_obj.train_unembed is True

    def test_weights_info_training_checkpoint(self, client, config):
        # Create a training checkpoint with metadata.json
        ckpt_dir = config.checkpoints_base / "run-train-1" / "step_0010"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "model.safetensors").write_bytes(b"dummy")
        meta = {
            "base_model": "Qwen/Qwen3.5-0.8B",
            "lora_config": {
                "rank": 8,
                "train_attn": True,
                "train_mlp": True,
                "train_unembed": False,
            },
        }
        (ckpt_dir / "metadata.json").write_text(json.dumps(meta))

        tinker_path = format_tinker_path("run-train-1", "step_0010", CheckpointType.TRAINING)
        resp = client.post("/api/v1/weights_info", json={"tinker_path": tinker_path})
        assert resp.status_code == 200
        data = resp.json()
        assert data["base_model"] == "Qwen/Qwen3.5-0.8B"
        assert data["is_lora"] is True
        assert data["lora_rank"] == 8
        assert data["train_attn"] is True
        assert data["train_mlp"] is True
        assert data["train_unembed"] is False

    def test_weights_info_save_weights_integration(self, app, config):
        # Test backend.save_weights populates metadata and get_weights_info reads it
        from mlx_tinker.api import server as server_module
        backend = server_module._backend
        model_id = "test-model-save"
        backend.models[model_id] = MagicMock()
        backend.training.optimizers[model_id] = MagicMock()
        backend.lora_configs[model_id] = LoraConfig(
            rank=32,
            alpha=64.0,
            train_attn=True,
            train_mlp=True,
            train_unembed=True,
        )

        # Mock heavy safetensors save
        import mlx_tinker.backend.mlx_backend as backend_module
        orig_save = backend_module.save_training_checkpoint
        try:
            saved_meta = {}

            def mock_save(model, opt, ckpt_dir, metadata=None):
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                (ckpt_dir / "model.safetensors").write_bytes(b"dummy")
                (ckpt_dir / "metadata.json").write_text(json.dumps(metadata or {}))
                return ckpt_dir

            backend_module.save_training_checkpoint = mock_save

            out = backend.save_weights(model_id, SaveWeightsInput(path="step_0050"))
            info = backend.get_weights_info(out.path)
            assert info["base_model"] == config.base_model
            assert info["is_lora"] is True
            assert info["lora_rank"] == 32
            assert info["train_unembed"] is True
            assert info["train_mlp"] is True
            assert info["train_attn"] is True
        finally:
            backend_module.save_training_checkpoint = orig_save
