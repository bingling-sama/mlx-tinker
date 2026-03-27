"""TinkerEngine: async polling loop that dispatches requests to the MLX backend."""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime, timezone

from sqlalchemy import update

from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.config import EngineConfig
from mlx_tinker.db.database import get_session
from mlx_tinker.db.models import FutureDB
from mlx_tinker.engine.scheduler import (
    complete_future,
    fail_future,
    find_batchable_requests,
    find_next_single_request,
    find_sample_requests,
)
from mlx_tinker.types import (
    CreateModelInput,
    ForwardBackwardInput,
    ForwardInput,
    LoadWeightsInput,
    OptimStepInput,
    RequestStatus,
    RequestType,
    SampleInput,
    SaveWeightsForSamplerInput,
    SaveWeightsInput,
    UnloadModelInput,
)

logger = logging.getLogger(__name__)


class TinkerEngine:
    """Background engine that polls the DB and dispatches work to MLXBackend.

    Runs as an asyncio task in the same process as the API server.
    Uses asyncio.to_thread for synchronous MLX operations to avoid
    blocking the event loop.
    """

    def __init__(self, config: EngineConfig, backend: MLXBackend) -> None:
        self.config = config
        self.backend = backend
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the engine polling loop as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("TinkerEngine started (cycle=%dms)", self.config.engine_cycle_ms)

    async def stop(self) -> None:
        """Signal the engine to stop and wait for it to drain."""
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None

        # Mark remaining pending futures as failed
        try:
            async with get_session() as session:
                stmt = (
                    update(FutureDB)
                    .where(FutureDB.status == RequestStatus.PENDING)
                    .values(
                        status=RequestStatus.FAILED,
                        result_data={"error": "Engine shutdown"},
                        completed_at=datetime.now(timezone.utc),
                    )
                )
                await session.execute(stmt)
                await session.commit()
        except Exception:
            logger.error("Failed to drain pending futures on shutdown:\n%s", traceback.format_exc())

        logger.info("TinkerEngine stopped")

    async def _run_loop(self) -> None:
        """Main polling loop — runs every engine_cycle_ms."""
        cycle_sec = self.config.engine_cycle_ms / 1000.0

        while self._running:
            try:
                await self._process_cycle()
            except Exception:
                logger.error("Engine cycle error:\n%s", traceback.format_exc())

            await asyncio.sleep(cycle_sec)

    async def _process_cycle(self) -> None:
        """Single engine cycle: find and dispatch pending requests."""
        async with get_session() as session:
            # 1. Process batchable forward_backward requests
            fb_batch = await find_batchable_requests(
                session,
                RequestType.FORWARD_BACKWARD,
                max_batch_size=self.config.max_batch_size,
            )
            for future in fb_batch:
                await self._dispatch_forward_backward(future)

            # 2. Process batchable forward requests
            fwd_batch = await find_batchable_requests(
                session,
                RequestType.FORWARD,
                max_batch_size=self.config.max_batch_size,
            )
            for future in fwd_batch:
                await self._dispatch_forward(future)

            # 3. Process sample requests
            sample_batch = await find_sample_requests(
                session,
                max_batch_size=self.config.max_batch_size,
            )
            for future in sample_batch:
                await self._dispatch_sample(future)

            # 4. Process single (barrier) requests
            single = await find_next_single_request(session)
            if single is not None:
                await self._dispatch_single(single)

    async def _dispatch_forward_backward(self, future: FutureDB) -> None:
        """Dispatch a forward_backward request to the backend."""
        try:
            request = ForwardBackwardInput(**future.request_data)
            result = await asyncio.to_thread(
                self.backend.forward_backward, future.model_id, request
            )
            async with get_session() as session:
                await complete_future(session, future.request_id, result.model_dump())
        except Exception as e:
            logger.error("forward_backward failed for request %d: %s", future.request_id, e)
            async with get_session() as session:
                await fail_future(session, future.request_id, str(e))

    async def _dispatch_forward(self, future: FutureDB) -> None:
        """Dispatch a forward request to the backend."""
        try:
            request = ForwardInput(**future.request_data)
            result = await asyncio.to_thread(
                self.backend.forward, future.model_id, request
            )
            async with get_session() as session:
                await complete_future(session, future.request_id, result.model_dump())
        except Exception as e:
            logger.error("forward failed for request %d: %s", future.request_id, e)
            async with get_session() as session:
                await fail_future(session, future.request_id, str(e))

    async def _dispatch_sample(self, future: FutureDB) -> None:
        """Dispatch a sample request to the backend."""
        try:
            request = SampleInput(**future.request_data)
            result = await asyncio.to_thread(
                self.backend.sample, future.model_id, request
            )
            async with get_session() as session:
                await complete_future(session, future.request_id, result.model_dump())
        except Exception as e:
            logger.error("sample failed for request %d: %s", future.request_id, e)
            async with get_session() as session:
                await fail_future(session, future.request_id, str(e))

    async def _dispatch_single(self, future: FutureDB) -> None:
        """Dispatch a single (non-batchable) request to the backend."""
        handler = {
            RequestType.CREATE_MODEL: self._handle_create_model,
            RequestType.OPTIM_STEP: self._handle_optim_step,
            RequestType.SAVE_WEIGHTS: self._handle_save_weights,
            RequestType.SAVE_WEIGHTS_FOR_SAMPLER: self._handle_save_weights_for_sampler,
            RequestType.LOAD_WEIGHTS: self._handle_load_weights,
            RequestType.UNLOAD_MODEL: self._handle_unload_model,
        }.get(future.request_type)

        if handler is None:
            logger.warning("Unknown single request type: %s", future.request_type)
            async with get_session() as session:
                await fail_future(
                    session, future.request_id, f"Unknown request type: {future.request_type}"
                )
            return

        try:
            await handler(future)
        except Exception as e:
            logger.error(
                "%s failed for request %d: %s", future.request_type, future.request_id, e
            )
            async with get_session() as session:
                await fail_future(session, future.request_id, str(e))

    async def _handle_create_model(self, future: FutureDB) -> None:
        request = CreateModelInput(**future.request_data)
        result = await asyncio.to_thread(
            self.backend.create_model, future.model_id, request
        )
        async with get_session() as session:
            await complete_future(session, future.request_id, result.model_dump())

    async def _handle_optim_step(self, future: FutureDB) -> None:
        request = OptimStepInput(**future.request_data)
        result = await asyncio.to_thread(
            self.backend.optim_step, future.model_id, request
        )
        async with get_session() as session:
            await complete_future(session, future.request_id, result.model_dump())

    async def _handle_save_weights(self, future: FutureDB) -> None:
        request = SaveWeightsInput(**future.request_data)
        result = await asyncio.to_thread(
            self.backend.save_weights, future.model_id, request
        )
        async with get_session() as session:
            await complete_future(session, future.request_id, result.model_dump())

    async def _handle_save_weights_for_sampler(self, future: FutureDB) -> None:
        request = SaveWeightsForSamplerInput(**future.request_data)
        result = await asyncio.to_thread(
            self.backend.save_weights_for_sampler, future.model_id, request
        )
        async with get_session() as session:
            await complete_future(session, future.request_id, result.model_dump())

    async def _handle_load_weights(self, future: FutureDB) -> None:
        request = LoadWeightsInput(**future.request_data)
        result = await asyncio.to_thread(
            self.backend.load_weights, future.model_id, request
        )
        async with get_session() as session:
            await complete_future(session, future.request_id, result.model_dump())

    async def _handle_unload_model(self, future: FutureDB) -> None:
        request = UnloadModelInput(**future.request_data)
        result = await asyncio.to_thread(
            self.backend.unload_model, future.model_id, request
        )
        async with get_session() as session:
            await complete_future(session, future.request_id, result.model_dump())
