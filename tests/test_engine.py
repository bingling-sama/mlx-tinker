"""Tests for the engine scheduler — barrier-aware batching logic."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from mlx_tinker.db.database import close_db, get_session, init_db
from mlx_tinker.db.models import FutureDB
from mlx_tinker.engine.scheduler import (
    complete_future,
    fail_future,
    find_batchable_requests,
    find_next_single_request,
    find_sample_requests,
)
from mlx_tinker.types import RequestStatus, RequestType


@pytest_asyncio.fixture
async def db(tmp_path):
    """Initialize a fresh in-memory DB for each test."""
    await init_db(tmp_path / "test_engine.db")
    yield
    await close_db()


async def _insert_future(
    request_type: RequestType,
    model_id: str = "model-1",
    request_data: dict | None = None,
) -> int:
    async with get_session() as session:
        f = FutureDB(
            request_type=request_type,
            model_id=model_id,
            request_data=request_data or {},
        )
        session.add(f)
        await session.commit()
        await session.refresh(f)
        return f.request_id


class TestFindBatchableRequests:
    @pytest.mark.asyncio
    async def test_empty_db(self, db):
        async with get_session() as session:
            batch = await find_batchable_requests(session, RequestType.FORWARD_BACKWARD)
            assert batch == []

    @pytest.mark.asyncio
    async def test_batches_forward_backward(self, db):
        await _insert_future(RequestType.FORWARD_BACKWARD)
        await _insert_future(RequestType.FORWARD_BACKWARD)
        await _insert_future(RequestType.FORWARD_BACKWARD)

        async with get_session() as session:
            batch = await find_batchable_requests(session, RequestType.FORWARD_BACKWARD)
            assert len(batch) == 3

    @pytest.mark.asyncio
    async def test_stops_at_barrier(self, db):
        await _insert_future(RequestType.FORWARD_BACKWARD)
        await _insert_future(RequestType.FORWARD_BACKWARD)
        await _insert_future(RequestType.OPTIM_STEP)
        await _insert_future(RequestType.FORWARD_BACKWARD)

        async with get_session() as session:
            batch = await find_batchable_requests(session, RequestType.FORWARD_BACKWARD)
            # Should only get the 2 before the barrier
            assert len(batch) == 2

    @pytest.mark.asyncio
    async def test_respects_max_batch_size(self, db):
        for _ in range(10):
            await _insert_future(RequestType.FORWARD_BACKWARD)

        async with get_session() as session:
            batch = await find_batchable_requests(
                session, RequestType.FORWARD_BACKWARD, max_batch_size=3
            )
            assert len(batch) == 3

    @pytest.mark.asyncio
    async def test_load_weights_is_barrier(self, db):
        await _insert_future(RequestType.FORWARD_BACKWARD)
        await _insert_future(RequestType.LOAD_WEIGHTS)
        await _insert_future(RequestType.FORWARD_BACKWARD)

        async with get_session() as session:
            batch = await find_batchable_requests(session, RequestType.FORWARD_BACKWARD)
            assert len(batch) == 1


class TestFindNextSingleRequest:
    @pytest.mark.asyncio
    async def test_returns_optim_step(self, db):
        rid = await _insert_future(RequestType.OPTIM_STEP)

        async with get_session() as session:
            req = await find_next_single_request(session)
            assert req is not None
            assert req.request_id == rid

    @pytest.mark.asyncio
    async def test_blocked_by_forward_backward(self, db):
        await _insert_future(RequestType.FORWARD_BACKWARD)
        await _insert_future(RequestType.OPTIM_STEP)

        async with get_session() as session:
            req = await find_next_single_request(session)
            # FB before OPTIM_STEP means OPTIM must wait
            assert req is None

    @pytest.mark.asyncio
    async def test_empty_db(self, db):
        async with get_session() as session:
            req = await find_next_single_request(session)
            assert req is None


class TestFindSampleRequests:
    @pytest.mark.asyncio
    async def test_batches_samples(self, db):
        await _insert_future(RequestType.SAMPLE)
        await _insert_future(RequestType.SAMPLE)

        async with get_session() as session:
            batch = await find_sample_requests(session)
            assert len(batch) == 2


class TestCompleteFuture:
    @pytest.mark.asyncio
    async def test_marks_completed(self, db):
        rid = await _insert_future(RequestType.FORWARD_BACKWARD)

        async with get_session() as session:
            await complete_future(session, rid, {"loss": 0.5})

        async with get_session() as session:
            f = await session.get(FutureDB, rid)
            assert f.status == RequestStatus.COMPLETED
            assert f.result_data == {"loss": 0.5}
            assert f.completed_at is not None

    @pytest.mark.asyncio
    async def test_fail_future(self, db):
        rid = await _insert_future(RequestType.FORWARD_BACKWARD)

        async with get_session() as session:
            await fail_future(session, rid, "test error")

        async with get_session() as session:
            f = await session.get(FutureDB, rid)
            assert f.status == RequestStatus.FAILED
            assert f.result_data["error"] == "test error"
