"""TinkerEngine: async polling loop that dispatches requests to the MLX backend."""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime, timezone

from sqlalchemy import update

from mlx_tinker.backend.inference import _sampling_params_key_from_mapping
from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.config import EngineConfig
from mlx_tinker.db.database import get_session
from mlx_tinker.db.models import FutureDB, ModelDB
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


def _sampling_mode_key(model_id: str | None, request_data: dict | None) -> tuple[str, str | None]:
    """Return a grouping key that prevents mixing incompatible sampling modes."""
    request_data = request_data or {}
    model_path = request_data.get("model_path")
    if isinstance(model_path, str) and model_path:
        return ("path", model_path)
    if model_id is not None:
        return ("student", model_id)
    base_model = request_data.get("base_model")
    return ("teacher", base_model if isinstance(base_model, str) else None)


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
                did_work = await self._process_cycle()
            except Exception:
                logger.error("Engine cycle error:\n%s", traceback.format_exc())
                did_work = False

            if did_work:
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(cycle_sec)

    async def _process_cycle(self) -> bool:
        """Single engine cycle: find and dispatch pending requests."""
        did_work = False
        async with get_session() as session:
            # 1. Process batchable forward_backward requests
            fb_batch = await find_batchable_requests(
                session,
                RequestType.FORWARD_BACKWARD,
                max_batch_size=self.config.max_batch_size,
            )
            if fb_batch:
                await self._dispatch_forward_backward_batch(fb_batch)
                did_work = True

            # 2. Process batchable forward requests
            fwd_batch = await find_batchable_requests(
                session,
                RequestType.FORWARD,
                max_batch_size=self.config.max_batch_size,
            )
            if fwd_batch:
                await self._dispatch_forward_batch(fwd_batch)
                did_work = True

            # 3. Process sample requests
            sample_batch = await find_sample_requests(
                session,
                max_batch_size=self.config.max_batch_size,
            )
            for futures in self._group_sample_futures(sample_batch):
                if len(futures) == 1:
                    await self._dispatch_sample(futures[0])
                else:
                    await self._dispatch_sample_batch(futures)
                did_work = True

            # 4. Process single (barrier) requests
            single = await find_next_single_request(session)
            if single is not None:
                await self._dispatch_single(single)
                did_work = True

        return did_work

    def _group_sample_futures(self, futures: list[FutureDB]) -> list[list[FutureDB]]:
        """Group sample requests that can share a single backend sampling call."""
        groups: list[list[FutureDB]] = []
        current: list[FutureDB] = []
        current_key: tuple | None = None

        for future in futures:
            request_data = future.request_data or {}
            sampling_params_key = _sampling_params_key_from_mapping(
                request_data.get("sampling_params")
            )
            if sampling_params_key is None:
                sampling_params_key = ("__invalid__", id(future))
            key = (
                _sampling_mode_key(future.model_id, request_data),
                sampling_params_key,
                bool(request_data.get("prompt_logprobs")),
            )
            if current and key != current_key:
                groups.append(current)
                current = []
            current.append(future)
            current_key = key

        if current:
            groups.append(current)
        return groups

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

    async def _dispatch_forward_backward_batch(self, futures: list[FutureDB]) -> None:
        """Dispatch a batch of forward_backward requests, coalescing compatible ones."""
        if not futures:
            return
        if len(futures) == 1:
            await self._dispatch_forward_backward(futures[0])
            return

        groups: dict[tuple, list[FutureDB]] = {}
        for future in futures:
            req_data = future.request_data or {}
            cfg_items = tuple(sorted((req_data.get("loss_fn_config") or {}).items()))
            key = (future.model_id, req_data.get("loss_fn", "cross_entropy"), cfg_items)
            groups.setdefault(key, []).append(future)

        for (model_id, _loss_fn, _cfg), group_futures in groups.items():
            if len(group_futures) == 1:
                await self._dispatch_forward_backward(group_futures[0])
                continue

            try:
                requests = [ForwardBackwardInput(**f.request_data) for f in group_futures]
                results = await asyncio.to_thread(
                    self.backend.forward_backward_batch,
                    model_id,
                    requests,
                )
                async with get_session() as session:
                    for future, result in zip(group_futures, results, strict=True):
                        await complete_future(session, future.request_id, result.model_dump())
            except Exception as e:
                logger.error(
                    "forward_backward batch failed for requests %s: %s",
                    [f.request_id for f in group_futures],
                    e,
                )
                async with get_session() as session:
                    for future in group_futures:
                        await fail_future(session, future.request_id, str(e))

    async def _dispatch_forward(self, future: FutureDB) -> None:
        """Dispatch a forward request to the backend."""
        try:
            request = ForwardInput(**future.request_data)
            result = await asyncio.to_thread(self.backend.forward, future.model_id, request)
            async with get_session() as session:
                await complete_future(session, future.request_id, result.model_dump())
        except Exception as e:
            logger.error("forward failed for request %d: %s", future.request_id, e)
            async with get_session() as session:
                await fail_future(session, future.request_id, str(e))

    async def _dispatch_forward_batch(self, futures: list[FutureDB]) -> None:
        """Dispatch a batch of forward requests, coalescing compatible ones."""
        if not futures:
            return
        if len(futures) == 1:
            await self._dispatch_forward(futures[0])
            return

        groups: dict[str | None, list[FutureDB]] = {}
        for future in futures:
            groups.setdefault(future.model_id, []).append(future)

        for model_id, group_futures in groups.items():
            if len(group_futures) == 1:
                await self._dispatch_forward(group_futures[0])
                continue

            try:
                requests = [ForwardInput(**f.request_data) for f in group_futures]
                results = await asyncio.to_thread(
                    self.backend.forward_batch,
                    model_id,
                    requests,
                )
                async with get_session() as session:
                    for future, result in zip(group_futures, results, strict=True):
                        await complete_future(session, future.request_id, result.model_dump())
            except Exception as e:
                logger.error(
                    "forward batch failed for requests %s: %s",
                    [f.request_id for f in group_futures],
                    e,
                )
                async with get_session() as session:
                    for future in group_futures:
                        await fail_future(session, future.request_id, str(e))

    async def _dispatch_sample(self, future: FutureDB) -> None:
        """Dispatch a sample request to the backend."""
        try:
            request = SampleInput(**future.request_data)
            result = await asyncio.to_thread(self.backend.sample, future.model_id, request)
            async with get_session() as session:
                await complete_future(session, future.request_id, result.model_dump())
        except Exception as e:
            logger.error("sample failed for request %d: %s", future.request_id, e)
            async with get_session() as session:
                await fail_future(session, future.request_id, str(e))

    async def _dispatch_sample_batch(self, futures: list[FutureDB]) -> None:
        """Dispatch a compatible sample batch to the backend in one call."""
        try:
            requests = [SampleInput(**future.request_data) for future in futures]
            request_batch = [
                (future.model_id, request)
                for future, request in zip(futures, requests, strict=True)
            ]
            results = await asyncio.to_thread(
                self.backend.sample_batch,
                request_batch,
            )
            async with get_session() as session:
                for future, result in zip(futures, results, strict=True):
                    await complete_future(session, future.request_id, result.model_dump())
        except Exception as e:
            logger.error(
                "sample batch failed for requests %s: %s",
                [f.request_id for f in futures],
                e,
            )
            async with get_session() as session:
                for future in futures:
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
            logger.error("%s failed for request %d: %s", future.request_type, future.request_id, e)
            async with get_session() as session:
                await fail_future(session, future.request_id, str(e))
            if future.request_type == RequestType.CREATE_MODEL and future.model_id is not None:
                await self._set_model_status(future.model_id, "failed")

    async def _handle_create_model(self, future: FutureDB) -> None:
        request = CreateModelInput(**future.request_data)
        result = await asyncio.to_thread(self.backend.create_model, future.model_id, request)
        async with get_session() as session:
            await complete_future(session, future.request_id, result.model_dump())
        if future.model_id is not None:
            await self._set_model_status(future.model_id, "ready")

    async def _handle_optim_step(self, future: FutureDB) -> None:
        request = OptimStepInput(**future.request_data)
        result = await asyncio.to_thread(self.backend.optim_step, future.model_id, request)
        async with get_session() as session:
            await complete_future(session, future.request_id, result.model_dump())

    async def _handle_save_weights(self, future: FutureDB) -> None:
        request = SaveWeightsInput(**future.request_data)
        result = await asyncio.to_thread(self.backend.save_weights, future.model_id, request)
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
        result = await asyncio.to_thread(self.backend.load_weights, future.model_id, request)
        async with get_session() as session:
            await complete_future(session, future.request_id, result.model_dump())

    async def _handle_unload_model(self, future: FutureDB) -> None:
        request = UnloadModelInput(**future.request_data)
        result = await asyncio.to_thread(self.backend.unload_model, future.model_id, request)
        async with get_session() as session:
            await complete_future(session, future.request_id, result.model_dump())
        if future.model_id is not None:
            await self._set_model_status(future.model_id, "unloaded")

    async def _set_model_status(self, model_id: str, status: str) -> None:
        async with get_session() as session:
            model = await session.get(ModelDB, model_id)
            if model is None:
                return
            model.status = status
            session.add(model)
            await session.commit()
