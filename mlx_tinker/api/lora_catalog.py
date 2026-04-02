"""Shared LoRA catalog helpers for UI and OpenAI-compatible model listing."""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from mlx_tinker.api.models import (
    LoraArtifactFile,
    LoraCatalogItem,
    LoraCatalogResponse,
    LoraCatalogSummary,
    LoraDetailResponse,
    LoraRecentFuture,
    LoraSamplingSessionInfo,
    LoraSessionInfo,
    LoraStats,
)
from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.db.database import get_session
from mlx_tinker.db.models import FutureDB, ModelDB, SamplingSessionDB, SessionDB
from mlx_tinker.types import RequestStatus, RequestType

_RELEVANT_FUTURE_TYPES = (
    RequestType.CREATE_MODEL,
    RequestType.FORWARD_BACKWARD,
    RequestType.OPTIM_STEP,
    RequestType.SAMPLE,
    RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
    RequestType.UNLOAD_MODEL,
)


@dataclass
class _AggregatedStats:
    forward_backward_count: int = 0
    optim_step_count: int = 0
    sample_count: int = 0
    sampler_export_count: int = 0
    loss_sum: float = 0.0
    loss_count: int = 0
    last_loss: float | None = None
    total_tokens: float = 0.0
    grad_norm_sum: float = 0.0
    grad_norm_count: int = 0
    last_grad_norm: float | None = None
    last_activity_at: datetime | None = None

    def to_api(self) -> LoraStats:
        return LoraStats(
            forward_backward_count=self.forward_backward_count,
            optim_step_count=self.optim_step_count,
            sample_count=self.sample_count,
            sampler_export_count=self.sampler_export_count,
            last_loss=self.last_loss,
            avg_loss=(self.loss_sum / self.loss_count) if self.loss_count else None,
            total_tokens=self.total_tokens,
            last_grad_norm=self.last_grad_norm,
            avg_grad_norm=(self.grad_norm_sum / self.grad_norm_count) if self.grad_norm_count else None,
            last_activity_at=_isoformat(self.last_activity_at),
        )


@dataclass(frozen=True)
class _ArtifactRecord:
    id: str
    relative_path: str
    resolved_path: Path
    display_name: str
    openai_model_id: str
    base_model: str
    created_at: datetime | None
    size_bytes: int
    lora_config: dict[str, Any]
    owner_model_id: str | None


@dataclass(frozen=True)
class _LiveRecord:
    id: str
    model_id: str
    display_name: str
    openai_model_id: str
    base_model: str
    created_at: datetime | None
    lora_config: dict[str, Any]


@dataclass
class _CatalogSnapshot:
    items: list[LoraCatalogItem]
    summary: LoraCatalogSummary
    artifact_records_by_id: dict[str, _ArtifactRecord] = field(default_factory=dict)
    live_records_by_id: dict[str, _LiveRecord] = field(default_factory=dict)
    model_rows: dict[str, ModelDB] = field(default_factory=dict)
    session_rows: dict[str, SessionDB] = field(default_factory=dict)
    sampling_sessions_by_model: dict[str, list[SamplingSessionDB]] = field(default_factory=dict)
    recent_futures_by_model: dict[str, list[FutureDB]] = field(default_factory=dict)


def encode_lora_id(kind: str, value: str) -> str:
    raw = f"{kind}:{value}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_lora_id(encoded: str) -> tuple[str, str] | None:
    padding = "=" * (-len(encoded) % 4)
    try:
        decoded = base64.urlsafe_b64decode((encoded + padding).encode("ascii")).decode("utf-8")
    except Exception:
        return None
    kind, sep, value = decoded.partition(":")
    if not sep or not kind or not value:
        return None
    return kind, value


class LoraCatalogService:
    """Build a merged view of persisted LoRA exports and live in-memory LoRAs."""

    def __init__(self, backend: MLXBackend) -> None:
        self.backend = backend

    async def list_catalog(self) -> LoraCatalogResponse:
        snapshot = await self._build_snapshot()
        return LoraCatalogResponse(summary=snapshot.summary, items=snapshot.items)

    async def get_detail(self, lora_id: str) -> LoraDetailResponse | None:
        snapshot = await self._build_snapshot()
        item = next((entry for entry in snapshot.items if entry.id == lora_id), None)
        if item is None:
            return None

        owner_model_id = None
        files: list[LoraArtifactFile] = []
        if lora_id in snapshot.artifact_records_by_id:
            artifact = snapshot.artifact_records_by_id[lora_id]
            owner_model_id = artifact.owner_model_id
            files = [
                LoraArtifactFile(name=path.name, size_bytes=path.stat().st_size)
                for path in sorted(artifact.resolved_path.iterdir())
                if path.is_file()
            ]
        elif lora_id in snapshot.live_records_by_id:
            owner_model_id = snapshot.live_records_by_id[lora_id].model_id

        session_info: LoraSessionInfo | None = None
        if owner_model_id is not None:
            model_row = snapshot.model_rows.get(owner_model_id)
            if model_row is not None:
                session_row = snapshot.session_rows.get(model_row.session_id)
                if session_row is not None:
                    session_info = LoraSessionInfo(
                        session_id=session_row.session_id,
                        status=session_row.status,
                        created_at=_isoformat(session_row.created_at) or "",
                        last_heartbeat_at=_isoformat(session_row.last_heartbeat_at),
                        heartbeat_count=session_row.heartbeat_count,
                    )

        sampling_sessions = [
            LoraSamplingSessionInfo(
                sampling_session_id=row.sampling_session_id,
                created_at=_isoformat(row.created_at) or "",
                model_id=row.model_id,
                base_model=row.base_model,
                model_path=row.model_path,
            )
            for row in snapshot.sampling_sessions_by_model.get(owner_model_id or "", [])[:10]
        ]

        recent_futures = [
            LoraRecentFuture(
                request_id=int(future.request_id or 0),
                request_type=future.request_type.value,
                status=future.status.value,
                created_at=_isoformat(future.created_at) or "",
                completed_at=_isoformat(future.completed_at),
            )
            for future in snapshot.recent_futures_by_model.get(owner_model_id or "", [])[:10]
        ]

        return LoraDetailResponse(
            item=item,
            session=session_info,
            sampling_sessions=sampling_sessions,
            recent_futures=recent_futures,
            files=files,
        )

    async def get_item(self, lora_id: str) -> LoraCatalogItem | None:
        snapshot = await self._build_snapshot()
        return next((entry for entry in snapshot.items if entry.id == lora_id), None)

    async def resolve_download_artifact(self, lora_id: str) -> _ArtifactRecord | None:
        snapshot = await self._build_snapshot()
        return snapshot.artifact_records_by_id.get(lora_id)

    async def list_openai_export_models(self) -> list[LoraCatalogItem]:
        snapshot = await self._build_snapshot()
        return [item for item in snapshot.items if item.is_exported]

    async def _build_snapshot(self) -> _CatalogSnapshot:
        model_rows, session_rows, sampling_sessions, futures = await self._load_db_rows()
        model_rows_by_id = {row.model_id: row for row in model_rows}
        session_rows_by_id = {row.session_id: row for row in session_rows}

        sampling_sessions_by_model: dict[str, list[SamplingSessionDB]] = defaultdict(list)
        for row in sorted(sampling_sessions, key=lambda row: row.created_at, reverse=True):
            if row.model_id:
                sampling_sessions_by_model[row.model_id].append(row)

        stats_by_model, recent_futures_by_model, latest_future_by_model = self._aggregate_futures(
            futures=futures,
            model_rows_by_id=model_rows_by_id,
            session_rows_by_id=session_rows_by_id,
            sampling_sessions_by_model=sampling_sessions_by_model,
        )

        artifact_records = self._scan_artifacts(model_rows_by_id)
        live_records = self._live_records(model_rows_by_id)

        items: list[LoraCatalogItem] = []
        artifact_records_by_id = {record.id: record for record in artifact_records}
        live_records_by_id = {record.id: record for record in live_records}

        for record in artifact_records:
            stats = stats_by_model.get(record.owner_model_id or "", _AggregatedStats()).to_api()
            items.append(
                LoraCatalogItem(
                    id=record.id,
                    relative_path=record.relative_path,
                    display_name=record.display_name,
                    openai_model_id=record.openai_model_id,
                    base_model=record.base_model,
                    created_at=_isoformat(record.created_at),
                    size_bytes=record.size_bytes,
                    lora_config=record.lora_config,
                    is_live=False,
                    is_exported=True,
                    downloadable=True,
                    status=self._resolve_status(
                        model_id=record.owner_model_id,
                        is_live=False,
                        is_exported=True,
                        model_rows_by_id=model_rows_by_id,
                        latest_future_by_model=latest_future_by_model,
                    ),
                    stats=stats,
                )
            )

        for record in live_records:
            stats = stats_by_model.get(record.model_id, _AggregatedStats()).to_api()
            items.append(
                LoraCatalogItem(
                    id=record.id,
                    relative_path=None,
                    display_name=record.display_name,
                    openai_model_id=record.openai_model_id,
                    base_model=record.base_model,
                    created_at=_isoformat(record.created_at),
                    size_bytes=0,
                    lora_config=record.lora_config,
                    is_live=True,
                    is_exported=False,
                    downloadable=False,
                    status="live",
                    stats=stats,
                )
            )

        items.sort(key=_item_sort_key, reverse=True)

        unique_base_models = {item.base_model for item in items}
        last_activity_candidates: list[datetime] = []
        for item in items:
            dt = _parse_datetime(item.stats.last_activity_at) or _parse_datetime(item.created_at)
            if dt is not None:
                last_activity_candidates.append(dt)

        summary = LoraCatalogSummary(
            total_exported_loras=sum(1 for item in items if item.is_exported),
            live_loras=sum(1 for item in items if item.is_live and not item.is_exported),
            unique_base_models=len(unique_base_models),
            total_adapter_disk_usage_bytes=sum(item.size_bytes for item in items if item.is_exported),
            total_optim_steps=sum(stats.optim_step_count for stats in stats_by_model.values()),
            last_activity_at=_isoformat(max(last_activity_candidates) if last_activity_candidates else None),
        )

        return _CatalogSnapshot(
            items=items,
            summary=summary,
            artifact_records_by_id=artifact_records_by_id,
            live_records_by_id=live_records_by_id,
            model_rows=model_rows_by_id,
            session_rows=session_rows_by_id,
            sampling_sessions_by_model=dict(sampling_sessions_by_model),
            recent_futures_by_model=recent_futures_by_model,
        )

    async def _load_db_rows(
        self,
    ) -> tuple[list[ModelDB], list[SessionDB], list[SamplingSessionDB], list[FutureDB]]:
        try:
            async with get_session() as session:
                models = list((await session.execute(select(ModelDB))).scalars().all())
                sessions = list((await session.execute(select(SessionDB))).scalars().all())
                sampling_sessions = list((await session.execute(select(SamplingSessionDB))).scalars().all())
                futures = list(
                    (
                        await session.execute(
                            select(FutureDB)
                            .where(FutureDB.request_type.in_(_RELEVANT_FUTURE_TYPES))
                            .order_by(FutureDB.created_at.desc())
                        )
                    )
                    .scalars()
                    .all()
                )
        except RuntimeError:
            return [], [], [], []
        return models, sessions, sampling_sessions, futures

    def _aggregate_futures(
        self,
        *,
        futures: list[FutureDB],
        model_rows_by_id: dict[str, ModelDB],
        session_rows_by_id: dict[str, SessionDB],
        sampling_sessions_by_model: dict[str, list[SamplingSessionDB]],
    ) -> tuple[dict[str, _AggregatedStats], dict[str, list[FutureDB]], dict[str, FutureDB]]:
        stats_by_model: dict[str, _AggregatedStats] = defaultdict(_AggregatedStats)
        recent_futures_by_model: dict[str, list[FutureDB]] = defaultdict(list)
        latest_future_by_model: dict[str, FutureDB] = {}

        for future in futures:
            if future.model_id is None:
                continue

            if future.model_id not in latest_future_by_model:
                latest_future_by_model[future.model_id] = future
            if len(recent_futures_by_model[future.model_id]) < 10:
                recent_futures_by_model[future.model_id].append(future)

            stats = stats_by_model[future.model_id]
            activity_at = future.completed_at or future.created_at
            stats.last_activity_at = _max_datetime(stats.last_activity_at, activity_at)

            if future.status != RequestStatus.COMPLETED:
                continue

            if future.request_type == RequestType.FORWARD_BACKWARD:
                stats.forward_backward_count += 1
                loss = _extract_metric(future.result_data, "loss:sum")
                if loss is not None:
                    stats.loss_sum += loss
                    stats.loss_count += 1
                    if stats.last_loss is None:
                        stats.last_loss = loss
            elif future.request_type == RequestType.OPTIM_STEP:
                stats.optim_step_count += 1
                total_tokens = _extract_metric(future.result_data, "total_tokens:sum")
                if total_tokens is not None:
                    stats.total_tokens += total_tokens
                grad_norm = _extract_metric(future.result_data, "grad_norm:mean")
                if grad_norm is not None:
                    stats.grad_norm_sum += grad_norm
                    stats.grad_norm_count += 1
                    if stats.last_grad_norm is None:
                        stats.last_grad_norm = grad_norm
            elif future.request_type == RequestType.SAMPLE:
                stats.sample_count += 1
            elif future.request_type == RequestType.SAVE_WEIGHTS_FOR_SAMPLER:
                stats.sampler_export_count += 1

        for model_id, model_row in model_rows_by_id.items():
            stats = stats_by_model[model_id]
            stats.last_activity_at = _max_datetime(stats.last_activity_at, model_row.created_at)
            session_row = session_rows_by_id.get(model_row.session_id)
            if session_row is not None:
                stats.last_activity_at = _max_datetime(
                    stats.last_activity_at,
                    session_row.last_heartbeat_at or session_row.created_at,
                )
            for sampling_session in sampling_sessions_by_model.get(model_id, []):
                stats.last_activity_at = _max_datetime(stats.last_activity_at, sampling_session.created_at)

        return dict(stats_by_model), dict(recent_futures_by_model), latest_future_by_model

    def _scan_artifacts(self, model_rows_by_id: dict[str, ModelDB]) -> list[_ArtifactRecord]:
        base = self.backend.config.checkpoints_base.resolve()
        records: list[_ArtifactRecord] = []
        for adapter_path in sorted(base.rglob("adapters.safetensors")):
            if "prefix_cache" in adapter_path.parts:
                continue

            artifact_dir = adapter_path.parent.resolve()
            config_path = artifact_dir / "config.json"
            if not config_path.exists():
                continue

            try:
                relative_path = artifact_dir.relative_to(base).as_posix()
            except ValueError:
                continue

            payload = _load_json(config_path)
            base_model = str(payload.get("base_model") or self.backend.config.base_model)
            lora_config = payload.get("lora_config") if isinstance(payload.get("lora_config"), dict) else {}
            owner_model_id = self._extract_owner_model_id(relative_path, model_rows_by_id)
            size_bytes = sum(path.stat().st_size for path in artifact_dir.iterdir() if path.is_file())
            created_at = _artifact_timestamp(artifact_dir)
            records.append(
                _ArtifactRecord(
                    id=encode_lora_id("artifact", relative_path),
                    relative_path=relative_path,
                    resolved_path=artifact_dir,
                    display_name=artifact_dir.name,
                    openai_model_id=f"{base_model}:{relative_path}",
                    base_model=base_model,
                    created_at=created_at,
                    size_bytes=size_bytes,
                    lora_config=lora_config,
                    owner_model_id=owner_model_id,
                )
            )
        return records

    def _live_records(self, model_rows_by_id: dict[str, ModelDB]) -> list[_LiveRecord]:
        records: list[_LiveRecord] = []
        for model_id in sorted(self.backend.models):
            model_row = model_rows_by_id.get(model_id)
            lora_config = {}
            if model_id in self.backend.lora_configs:
                lora_config = self.backend.lora_configs[model_id].model_dump()
            elif model_row is not None and isinstance(model_row.lora_config, dict):
                lora_config = dict(model_row.lora_config)

            base_model = (
                model_row.base_model
                if model_row is not None
                else self.backend.config.base_model
            )
            records.append(
                _LiveRecord(
                    id=encode_lora_id("live", model_id),
                    model_id=model_id,
                    display_name=f"Live {model_id[:8]}",
                    openai_model_id=model_id,
                    base_model=base_model,
                    created_at=model_row.created_at if model_row is not None else None,
                    lora_config=lora_config,
                )
            )
        return records

    @staticmethod
    def _extract_owner_model_id(
        relative_path: str,
        model_rows_by_id: dict[str, ModelDB],
    ) -> str | None:
        parts = Path(relative_path).parts
        if len(parts) >= 3 and parts[1] == "sampler":
            return parts[0]
        if len(parts) == 1 and parts[0] in model_rows_by_id:
            return parts[0]
        return None

    @staticmethod
    def _resolve_status(
        *,
        model_id: str | None,
        is_live: bool,
        is_exported: bool,
        model_rows_by_id: dict[str, ModelDB],
        latest_future_by_model: dict[str, FutureDB],
    ) -> str:
        if is_live:
            return "live"
        if model_id is None:
            return "exported" if is_exported else "unknown"

        latest_future = latest_future_by_model.get(model_id)
        if latest_future is not None:
            if latest_future.request_type == RequestType.CREATE_MODEL:
                if latest_future.status == RequestStatus.PENDING:
                    return "creating"
                if latest_future.status == RequestStatus.FAILED:
                    return "failed"
            if latest_future.request_type == RequestType.UNLOAD_MODEL and latest_future.status == RequestStatus.COMPLETED:
                return "unloaded" if not is_exported else "exported"

        model_row = model_rows_by_id.get(model_id)
        if model_row is not None and model_row.status == "failed":
            return "failed"
        if is_exported:
            return "exported"
        return model_row.status if model_row is not None else "unknown"


def _extract_metric(result_data: dict[str, Any] | None, metric_name: str) -> float | None:
    if not isinstance(result_data, dict):
        return None
    metrics = result_data.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(metric_name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _isoformat(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _max_datetime(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _artifact_timestamp(artifact_dir: Path) -> datetime | None:
    mtimes = [path.stat().st_mtime for path in artifact_dir.iterdir() if path.is_file()]
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc)


def _item_sort_key(item: LoraCatalogItem) -> tuple[int, float]:
    dt = _parse_datetime(item.stats.last_activity_at) or _parse_datetime(item.created_at)
    return (1 if item.is_live else 0, dt.timestamp() if dt is not None else 0.0)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
