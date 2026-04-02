"""Built-in LoRA web UI and supporting API routes."""

from __future__ import annotations

import asyncio
import re
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.background import BackgroundTask

from mlx_tinker.api.lora_catalog import LoraCatalogService, encode_lora_id
from mlx_tinker.api.lora_ui_assets import LORA_UI_CSS, LORA_UI_HTML, LORA_UI_JS
from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.db.database import get_session
from mlx_tinker.db.models import FutureDB
from mlx_tinker.engine.scheduler import complete_future, fail_future
from mlx_tinker.types import RequestType, SaveWeightsForSamplerInput

router = APIRouter()
_backend: MLXBackend | None = None


def register_lora_routes(app, backend: MLXBackend) -> None:
    """Register built-in LoRA UI routes on the FastAPI app."""
    global _backend
    _backend = backend
    app.include_router(router)


@router.get("/ui/loras", response_class=HTMLResponse)
async def lora_ui_index() -> HTMLResponse:
    _require_backend()
    return HTMLResponse(LORA_UI_HTML)


@router.get("/ui/assets/loras.css")
async def lora_ui_css() -> Response:
    return Response(content=LORA_UI_CSS, media_type="text/css")


@router.get("/ui/assets/loras.js")
async def lora_ui_js() -> Response:
    return Response(content=LORA_UI_JS, media_type="application/javascript")


@router.get("/api/v1/loras")
async def list_loras():
    service = _service()
    return await service.list_catalog()


@router.get("/api/v1/loras/{lora_id}")
async def get_lora_detail(lora_id: str):
    service = _service()
    detail = await service.get_detail(lora_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="LoRA not found")
    return detail


@router.get("/api/v1/loras/{lora_id}/download")
async def download_lora(lora_id: str):
    service = _service()
    artifact = await service.resolve_download_artifact(lora_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Exported LoRA not found")

    archive_path = _write_lora_zip(artifact.resolved_path)
    filename = f"{_slugify(artifact.base_model)}__{artifact.resolved_path.name}.zip"
    return FileResponse(
        path=archive_path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(_cleanup_temp_file, archive_path),
    )


@router.post("/api/v1/loras/live/{model_id}/export")
async def export_live_lora(model_id: str):
    backend = _require_backend()
    if model_id not in backend.models:
        raise HTTPException(status_code=404, detail=f"Live model {model_id} not found")

    export_name = _live_export_name()
    relative_path = str(Path(model_id) / "sampler" / export_name)
    future = await _create_export_future(model_id=model_id, relative_path=relative_path)

    try:
        result = await asyncio.to_thread(
            backend.save_weights_for_sampler,
            model_id,
            SaveWeightsForSamplerInput(path=relative_path, ephemeral=False),
        )
    except Exception as exc:
        async with get_session() as session:
            await fail_future(session, int(future.request_id or 0), str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to export live LoRA: {exc}") from exc

    async with get_session() as session:
        await complete_future(session, int(future.request_id or 0), result.model_dump())

    item_id = encode_lora_id("artifact", Path(relative_path).as_posix())
    item = await _service().get_item(item_id)
    if item is None:
        raise HTTPException(status_code=500, detail="Export succeeded but catalog entry was not found")
    return item


def _service() -> LoraCatalogService:
    return LoraCatalogService(_require_backend())


def _require_backend() -> MLXBackend:
    if _backend is None:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    return _backend


async def _create_export_future(*, model_id: str, relative_path: str) -> FutureDB:
    async with get_session() as session:
        future = FutureDB(
            request_type=RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
            model_id=model_id,
            request_data={
                "path": relative_path,
                "sampling_session_seq_id": None,
                "seq_id": None,
                "sampling_session_id": None,
                "ephemeral": False,
            },
        )
        session.add(future)
        await session.commit()
        await session.refresh(future)
        return future


def _write_lora_zip(artifact_dir: Path) -> str:
    with tempfile.NamedTemporaryFile(prefix="mlx-tinker-lora-", suffix=".zip", delete=False) as handle:
        archive_path = Path(handle.name)

    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("adapters.safetensors", "config.json", "tokenizer_config.json"):
            candidate = artifact_dir / name
            if candidate.exists() and candidate.is_file():
                archive.write(candidate, arcname=name)
    return str(archive_path)


def _cleanup_temp_file(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _live_export_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"ui-export-{timestamp}-{uuid.uuid4().hex[:8]}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "model"
