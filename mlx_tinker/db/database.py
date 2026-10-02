"""Async SQLite database engine with WAL mode."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

_engine = None
_session_factory = None


async def _ensure_schema_compatibility(conn) -> None:
    """Apply lightweight additive migrations for local SQLite databases."""
    columns = await conn.exec_driver_sql("PRAGMA table_info(sampling_sessions)")
    existing = {row[1] for row in columns.fetchall()}
    if "model_id" not in existing:
        await conn.exec_driver_sql("ALTER TABLE sampling_sessions ADD COLUMN model_id VARCHAR")

    ckpt_columns = await conn.exec_driver_sql("PRAGMA table_info(checkpoints)")
    ckpt_existing = {row[1] for row in ckpt_columns.fetchall()}
    if ckpt_existing:
        if "public" not in ckpt_existing:
            await conn.exec_driver_sql("ALTER TABLE checkpoints ADD COLUMN public BOOLEAN DEFAULT 0")
        if "expires_at" not in ckpt_existing:
            await conn.exec_driver_sql("ALTER TABLE checkpoints ADD COLUMN expires_at DATETIME")
        if "size_bytes" not in ckpt_existing:
            await conn.exec_driver_sql("ALTER TABLE checkpoints ADD COLUMN size_bytes INTEGER")

    model_columns = await conn.exec_driver_sql("PRAGMA table_info(models)")
    model_existing = {row[1] for row in model_columns.fetchall()}
    if model_existing and "user_metadata" not in model_existing:
        await conn.exec_driver_sql("ALTER TABLE models ADD COLUMN user_metadata JSON")


async def init_db(db_path: str | Path = "tinker.db") -> None:
    """Initialize the async SQLite engine with WAL mode and create all tables."""
    global _engine, _session_factory

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"sqlite+aiosqlite:///{db_path}"
    _engine = create_async_engine(url, echo=False)

    # Enable WAL mode for concurrent reads
    async with _engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
        await conn.run_sync(SQLModel.metadata.create_all)
        await _ensure_schema_compatibility(conn)

    _session_factory = sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


def get_session() -> AsyncSession:
    """Get a new async database session."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _session_factory()


async def close_db() -> None:
    """Close the database engine."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
