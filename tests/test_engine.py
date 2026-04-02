"""Tests for the engine scheduler — barrier-aware batching logic."""

import importlib
from types import SimpleNamespace

import pytest
import pytest_asyncio

from mlx_tinker.config import EngineConfig
from mlx_tinker.db.database import close_db, get_session, init_db
from mlx_tinker.db.models import FutureDB, ModelDB, SessionDB
from mlx_tinker.engine.engine import TinkerEngine
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


class TestGroupSampleFutures:
    def test_groups_by_sampling_tuple_key_without_constructing_sample_input(self, monkeypatch):
        engine_module = importlib.import_module("mlx_tinker.engine.engine")
        engine = TinkerEngine(EngineConfig(), backend=SimpleNamespace())

        def fail_if_constructed(**_kwargs):
            raise AssertionError("SampleInput should not be constructed during grouping")

        monkeypatch.setattr(engine_module, "SampleInput", fail_if_constructed)
        futures = [
            SimpleNamespace(
                model_id="model-1",
                request_data={
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "sampling_params": {"temperature": 1.0, "max_tokens": 4, "seed": 7},
                    "prompt_logprobs": False,
                },
            ),
            SimpleNamespace(
                model_id="model-1",
                request_data={
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "sampling_params": {"max_tokens": 4, "seed": 7, "temperature": 1.0},
                    "prompt_logprobs": False,
                },
            ),
            SimpleNamespace(
                model_id="model-1",
                request_data={
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "sampling_params": {"temperature": 1.0, "max_tokens": 4, "seed": 9},
                    "prompt_logprobs": False,
                },
            ),
        ]

        groups = engine._group_sample_futures(futures)

        assert [len(group) for group in groups] == [2, 1]

    def test_malformed_sampling_params_do_not_crash_grouping(self):
        engine = TinkerEngine(EngineConfig(), backend=SimpleNamespace())
        futures = [
            SimpleNamespace(
                model_id="model-1",
                request_data={
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "sampling_params": "not-a-dict",
                    "prompt_logprobs": False,
                },
            ),
            SimpleNamespace(
                model_id="model-1",
                request_data={
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "sampling_params": {"temperature": 1.0, "max_tokens": 4, "seed": 7},
                    "prompt_logprobs": False,
                },
            ),
            SimpleNamespace(
                model_id="model-1",
                request_data={
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "sampling_params": "still-not-a-dict",
                    "prompt_logprobs": False,
                },
            ),
        ]

        groups = engine._group_sample_futures(futures)

        assert [len(group) for group in groups] == [1, 1, 1]

    def test_groups_teacher_and_student_requests_separately(self):
        engine = TinkerEngine(EngineConfig(), backend=SimpleNamespace())
        futures = [
            SimpleNamespace(
                model_id="model-1",
                request_data={
                    "sampling_params": {"temperature": 1.0, "max_tokens": 4, "seed": 7},
                    "prompt_logprobs": False,
                },
            ),
            SimpleNamespace(
                model_id=None,
                request_data={
                    "base_model": "test-model",
                    "sampling_params": {"temperature": 1.0, "max_tokens": 4, "seed": 7},
                    "prompt_logprobs": False,
                },
            ),
            SimpleNamespace(
                model_id=None,
                request_data={
                    "model_path": "checkpoints/model-1/sampler/manual",
                    "sampling_params": {"temperature": 1.0, "max_tokens": 4, "seed": 7},
                    "prompt_logprobs": False,
                },
            ),
        ]

        groups = engine._group_sample_futures(futures)

        assert [len(group) for group in groups] == [1, 1, 1]


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


async def _insert_session_and_model(model_id: str, *, status: str = "creating") -> None:
    async with get_session() as session:
        session_row = SessionDB(session_id="session-1")
        model_row = ModelDB(
            model_id=model_id,
            base_model="test-model",
            lora_config={"rank": 8, "alpha": 16.0},
            status=status,
            request_id=1,
            session_id=session_row.session_id,
        )
        session.add(session_row)
        session.add(model_row)
        await session.commit()


class TestModelStatusUpdates:
    @pytest.mark.asyncio
    async def test_create_model_marks_row_ready(self, db):
        model_id = "model-ready"
        await _insert_session_and_model(model_id, status="creating")
        rid = await _insert_future(
            RequestType.CREATE_MODEL,
            model_id=model_id,
            request_data={"lora_config": {"rank": 8, "alpha": 16.0}},
        )

        backend = SimpleNamespace(
            create_model=lambda model_id, request: SimpleNamespace(model_dump=lambda: {"model_id": model_id})
        )
        engine = TinkerEngine(EngineConfig(), backend=backend)

        async with get_session() as session:
            future = await session.get(FutureDB, rid)

        await engine._handle_create_model(future)

        async with get_session() as session:
            model = await session.get(ModelDB, model_id)
            future = await session.get(FutureDB, rid)
            assert model.status == "ready"
            assert future.status == RequestStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_create_model_failure_marks_row_failed(self, db):
        model_id = "model-failed"
        await _insert_session_and_model(model_id, status="creating")
        rid = await _insert_future(
            RequestType.CREATE_MODEL,
            model_id=model_id,
            request_data={"lora_config": {"rank": 8, "alpha": 16.0}},
        )

        backend = SimpleNamespace(create_model=lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")))
        engine = TinkerEngine(EngineConfig(), backend=backend)

        async with get_session() as session:
            future = await session.get(FutureDB, rid)

        await engine._dispatch_single(future)

        async with get_session() as session:
            model = await session.get(ModelDB, model_id)
            future = await session.get(FutureDB, rid)
            assert model.status == "failed"
            assert future.status == RequestStatus.FAILED

    @pytest.mark.asyncio
    async def test_unload_model_marks_row_unloaded(self, db):
        model_id = "model-unloaded"
        await _insert_session_and_model(model_id, status="ready")
        rid = await _insert_future(RequestType.UNLOAD_MODEL, model_id=model_id, request_data={})

        backend = SimpleNamespace(
            unload_model=lambda model_id, request: SimpleNamespace(
                model_dump=lambda: {"model_id": model_id, "status": "unloaded"}
            )
        )
        engine = TinkerEngine(EngineConfig(), backend=backend)

        async with get_session() as session:
            future = await session.get(FutureDB, rid)

        await engine._handle_unload_model(future)

        async with get_session() as session:
            model = await session.get(ModelDB, model_id)
            future = await session.get(FutureDB, rid)
            assert model.status == "unloaded"
            assert future.status == RequestStatus.COMPLETED
