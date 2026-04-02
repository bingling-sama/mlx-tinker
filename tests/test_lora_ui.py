"""Tests for the built-in LoRA catalog and UI endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from mlx_tinker.api import lora_ui as lora_ui_module
from mlx_tinker.api.server import create_app
from mlx_tinker.config import EngineConfig
from mlx_tinker.db.database import get_session
from mlx_tinker.db.models import FutureDB, ModelDB, SamplingSessionDB, SessionDB
from mlx_tinker.types import LoraConfig, RequestStatus, RequestType, SaveWeightsForSamplerOutput


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
    with TestClient(app) as test_client:
        yield test_client


def _write_export(
    root: Path,
    relative_path: str,
    *,
    base_model: str = "test-model",
    lora_config: dict | None = None,
) -> Path:
    export_dir = root / relative_path
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "adapters.safetensors").write_bytes(b"adapter-weights")
    (export_dir / "config.json").write_text(
        json.dumps(
            {
                "base_model": base_model,
                "lora_config": lora_config or {"rank": 8, "alpha": 16.0, "train_attn": True},
            }
        ),
        encoding="utf-8",
    )
    (export_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return export_dir


async def _seed_history(model_id: str, *, status: str = "creating") -> None:
    async with get_session() as session:
        session_row = SessionDB(session_id="session-1", status="active", heartbeat_count=4)
        model_row = ModelDB(
            model_id=model_id,
            base_model="test-model",
            lora_config={"rank": 8, "alpha": 16.0},
            status=status,
            request_id=1,
            session_id=session_row.session_id,
        )
        sampling_session = SamplingSessionDB(
            sampling_session_id="sampling-1",
            session_id=session_row.session_id,
            model_id=model_id,
            base_model="test-model",
        )
        futures = [
            FutureDB(
                request_type=RequestType.CREATE_MODEL,
                model_id=model_id,
                status=RequestStatus.COMPLETED,
                request_data={},
                result_data={"model_id": model_id},
            ),
            FutureDB(
                request_type=RequestType.FORWARD_BACKWARD,
                model_id=model_id,
                status=RequestStatus.COMPLETED,
                request_data={},
                result_data={"metrics": {"loss:sum": -12.5, "num_sequences:sum": 1}},
            ),
            FutureDB(
                request_type=RequestType.OPTIM_STEP,
                model_id=model_id,
                status=RequestStatus.COMPLETED,
                request_data={},
                result_data={"metrics": {"total_tokens:sum": 64, "grad_norm:mean": 0.75}},
            ),
            FutureDB(
                request_type=RequestType.SAMPLE,
                model_id=model_id,
                status=RequestStatus.COMPLETED,
                request_data={},
                result_data={"sequences": []},
            ),
            FutureDB(
                request_type=RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
                model_id=model_id,
                status=RequestStatus.COMPLETED,
                request_data={"path": f"{model_id}/sampler/export-a"},
                result_data={"path": f"{model_id}/sampler/export-a"},
            ),
        ]
        session.add(session_row)
        session.add(model_row)
        session.add(sampling_session)
        for future in futures:
            session.add(future)
        await session.commit()


class TestLoraCatalogAPI:
    def test_ui_html_is_served(self, client):
        response = client.get("/ui/loras")
        assert response.status_code == 200
        assert "LoRA Control Room" in response.text
        assert "/ui/assets/loras.js" in response.text

    def test_catalog_discovers_exports_ignores_training_checkpoints_and_surfaces_live_stats(self, client, config):
        model_id = "model-live"
        _write_export(config.checkpoints_base, f"{model_id}/sampler/export-a")
        (config.checkpoints_base / "step_0001").mkdir(parents=True)
        (config.checkpoints_base / "step_0001" / "model.safetensors").write_bytes(b"full-model")
        (config.checkpoints_base / "prefix_cache").mkdir(parents=True, exist_ok=True)
        (config.checkpoints_base / "prefix_cache" / "manifest.sqlite3").write_bytes(b"cache")
        asyncio.run(_seed_history(model_id))

        backend = lora_ui_module._backend
        assert backend is not None
        backend.models[model_id] = MagicMock(name="live-model")
        backend.lora_configs[model_id] = LoraConfig(rank=8, alpha=16.0)

        response = client.get("/api/v1/loras")
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["total_exported_loras"] == 1
        assert data["summary"]["live_loras"] == 1
        assert data["summary"]["total_optim_steps"] == 1

        exported = [item for item in data["items"] if item["is_exported"]]
        live = [item for item in data["items"] if item["is_live"] and not item["is_exported"]]

        assert len(exported) == 1
        assert exported[0]["relative_path"] == f"{model_id}/sampler/export-a"
        assert exported[0]["openai_model_id"] == f"test-model:{model_id}/sampler/export-a"
        assert exported[0]["downloadable"] is True
        assert exported[0]["stats"]["optim_step_count"] == 1
        assert exported[0]["stats"]["last_loss"] == -12.5
        assert live[0]["status"] == "live"
        assert live[0]["stats"]["optim_step_count"] == 1

    def test_detail_endpoint_returns_files_sessions_and_recent_futures(self, client, config):
        model_id = "model-detail"
        _write_export(config.checkpoints_base, f"{model_id}/sampler/export-a")
        asyncio.run(_seed_history(model_id))

        listing = client.get("/api/v1/loras").json()
        exported = next(item for item in listing["items"] if item["is_exported"])

        response = client.get(f"/api/v1/loras/{exported['id']}")
        assert response.status_code == 200
        detail = response.json()
        assert detail["item"]["id"] == exported["id"]
        assert any(file["name"] == "adapters.safetensors" for file in detail["files"])
        assert detail["session"]["session_id"] == "session-1"
        assert any(future["request_type"] == "optim_step" for future in detail["recent_futures"])
        assert detail["sampling_sessions"][0]["sampling_session_id"] == "sampling-1"

    def test_download_endpoint_returns_zip_bundle(self, client, config):
        _write_export(config.checkpoints_base, "model-zip/sampler/export-a")

        listing = client.get("/api/v1/loras").json()
        exported = next(item for item in listing["items"] if item["is_exported"])

        response = client.get(f"/api/v1/loras/{exported['id']}/download")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

        archive = zipfile.ZipFile(io.BytesIO(response.content))
        assert sorted(archive.namelist()) == [
            "adapters.safetensors",
            "config.json",
            "tokenizer_config.json",
        ]

    def test_download_missing_lora_returns_404(self, client):
        response = client.get("/api/v1/loras/not-a-real-id/download")
        assert response.status_code == 404

    def test_live_export_endpoint_persists_export_records_future_and_returns_item(self, client, config, monkeypatch):
        backend = lora_ui_module._backend
        assert backend is not None
        backend.models["live-export"] = MagicMock(name="live-export")
        backend.lora_configs["live-export"] = LoraConfig(rank=4, alpha=8.0)

        def fake_save_weights_for_sampler(model_id, request):
            export_dir = backend.config.checkpoints_base / request.path
            _write_export(
                backend.config.checkpoints_base,
                request.path,
                base_model=backend.config.base_model,
                lora_config=backend.lora_configs[model_id].model_dump(),
            )
            return SaveWeightsForSamplerOutput(path=str(export_dir), sampling_session_id=None)

        monkeypatch.setattr(backend, "save_weights_for_sampler", fake_save_weights_for_sampler)

        response = client.post("/api/v1/loras/live/live-export/export")
        assert response.status_code == 200
        item = response.json()
        assert item["is_exported"] is True
        assert item["relative_path"].startswith("live-export/sampler/ui-export-")

        download = client.get(f"/api/v1/loras/{item['id']}/download")
        assert download.status_code == 200

        async def _load_futures():
            async with get_session() as session:
                rows = (
                    await session.execute(
                        select(FutureDB).where(
                            FutureDB.model_id == "live-export",
                            FutureDB.request_type == RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
                        )
                    )
                ).scalars().all()
                return list(rows)

        futures = asyncio.run(_load_futures())
        assert futures
        assert all(row.status == RequestStatus.COMPLETED for row in futures)
