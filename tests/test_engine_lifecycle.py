"""Tests for TinkerEngine lifecycle — shutdown drains pending futures."""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import MagicMock

from mlx_tinker.config import EngineConfig
from mlx_tinker.db.database import close_db, get_session, init_db
from mlx_tinker.db.models import FutureDB
from mlx_tinker.engine.engine import TinkerEngine
from mlx_tinker.types import RequestStatus, RequestType


@pytest_asyncio.fixture
async def db(tmp_path):
    await init_db(tmp_path / "test.db")
    yield
    await close_db()


@pytest.fixture
def engine(db):
    config = EngineConfig(base_model="test", engine_cycle_ms=10)
    backend = MagicMock()
    return TinkerEngine(config, backend)


class TestShutdownDrain:
    @pytest.mark.asyncio
    async def test_pending_futures_marked_failed(self, engine):
        # Insert 3 PENDING futures
        async with get_session() as session:
            for i in range(3):
                future = FutureDB(
                    request_type=RequestType.FORWARD_BACKWARD,
                    model_id=f"model-{i}",
                    request_data={},
                    status=RequestStatus.PENDING,
                )
                session.add(future)
            await session.commit()

        # Start and immediately stop the engine
        await engine.start()
        await engine.stop()

        # Verify all futures are FAILED with "Engine shutdown"
        async with get_session() as session:
            from sqlmodel import select

            stmt = select(FutureDB)
            result = await session.execute(stmt)
            futures = result.scalars().all()

            assert len(futures) == 3
            for future in futures:
                assert future.status == RequestStatus.FAILED
                assert future.result_data is not None
                assert "Engine shutdown" in future.result_data["error"]
