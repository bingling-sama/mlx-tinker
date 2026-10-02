"""SQLModel ORM definitions for the Tinker futures database."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime
from sqlmodel import Field, SQLModel

from mlx_tinker.types import CheckpointStatus, CheckpointType, RequestStatus, RequestType


class SessionDB(SQLModel, table=True):
    __tablename__ = "sessions"

    session_id: str = Field(primary_key=True)
    tags: list[str] = Field(default_factory=list, sa_type=JSON)
    user_metadata: dict = Field(default_factory=dict, sa_type=JSON)
    sdk_version: str = "0.1.0"
    status: str = Field(default="active", index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=DateTime(timezone=True),
    )
    last_heartbeat_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),
        index=True,
    )
    heartbeat_count: int = 0


class ModelDB(SQLModel, table=True):
    __tablename__ = "models"

    model_id: str = Field(primary_key=True)
    base_model: str
    lora_config: dict = Field(default_factory=dict, sa_type=JSON)
    status: str = Field(index=True)
    request_id: int = 0
    session_id: str = Field(foreign_key="sessions.session_id", index=True)
    user_metadata: dict = Field(default_factory=dict, sa_type=JSON)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=DateTime(timezone=True),
    )


class FutureDB(SQLModel, table=True):
    __tablename__ = "futures"

    request_id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"autoincrement": True},
    )
    request_type: RequestType
    model_id: str | None = Field(default=None, index=True)
    request_data: dict = Field(default_factory=dict, sa_type=JSON)
    result_data: dict | None = Field(default=None, sa_type=JSON)
    status: RequestStatus = Field(default=RequestStatus.PENDING, index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=DateTime(timezone=True),
    )
    completed_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),
    )


class CheckpointDB(SQLModel, table=True):
    __tablename__ = "checkpoints"

    model_id: str = Field(foreign_key="models.model_id", primary_key=True)
    checkpoint_id: str = Field(primary_key=True)
    checkpoint_type: CheckpointType = Field(primary_key=True)
    status: CheckpointStatus
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=DateTime(timezone=True),
    )
    completed_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),
    )
    error_message: str | None = None
    public: bool = Field(default=False)
    expires_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),
    )
    size_bytes: int | None = None


class SamplingSessionDB(SQLModel, table=True):
    __tablename__ = "sampling_sessions"

    sampling_session_id: str = Field(primary_key=True)
    session_id: str = Field(foreign_key="sessions.session_id", index=True)
    sampling_session_seq_id: int = 0
    model_id: str | None = Field(default=None, foreign_key="models.model_id", index=True)
    base_model: str | None = None
    model_path: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=DateTime(timezone=True),
    )
