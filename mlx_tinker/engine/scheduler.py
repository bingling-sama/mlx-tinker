"""Barrier-aware request batching for the Tinker engine.

The scheduler groups pending requests into batches while respecting
ordering constraints: OPTIM_STEP and LOAD_WEIGHTS act as barriers
that must not be reordered past FORWARD_BACKWARD requests.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mlx_tinker.db.models import FutureDB
from mlx_tinker.types import RequestStatus, RequestType

logger = logging.getLogger(__name__)

# Request types that act as barriers — must wait for all preceding
# forward/backward requests to complete before executing
BARRIER_TYPES = {RequestType.OPTIM_STEP, RequestType.LOAD_WEIGHTS}

# Request types that can be batched together
BATCHABLE_TRAIN_TYPES = {RequestType.FORWARD_BACKWARD, RequestType.FORWARD}
BATCHABLE_SAMPLE_TYPES = {RequestType.SAMPLE}

# Single-execution types (not batched)
SINGLE_TYPES = {
    RequestType.CREATE_MODEL,
    RequestType.OPTIM_STEP,
    RequestType.LOAD_WEIGHTS,
    RequestType.SAVE_WEIGHTS,
    RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
    RequestType.UNLOAD_MODEL,
}


async def find_batchable_requests(
    session: AsyncSession,
    request_type: RequestType,
    model_id: str | None = None,
    max_batch_size: int = 8,
) -> list[FutureDB]:
    """Find pending requests of the given type that can be batched.

    For training requests, stops at the first barrier (OPTIM_STEP or LOAD_WEIGHTS).
    """
    query = (
        select(FutureDB)
        .where(FutureDB.status == RequestStatus.PENDING)
        .order_by(FutureDB.request_id)
    )

    if model_id:
        query = query.where(FutureDB.model_id == model_id)

    result = await session.execute(query)
    all_pending = result.scalars().all()

    batch: list[FutureDB] = []
    for future in all_pending:
        # If we hit a barrier, stop batching
        if future.request_type in BARRIER_TYPES:
            break

        if future.request_type == request_type:
            batch.append(future)
            if len(batch) >= max_batch_size:
                break

    return batch


async def find_next_single_request(
    session: AsyncSession,
    model_id: str | None = None,
) -> FutureDB | None:
    """Find the next pending single-execution request.

    Only returns a single request if all preceding batchable requests
    for the same model have been completed.
    """
    query = (
        select(FutureDB)
        .where(FutureDB.status == RequestStatus.PENDING)
        .order_by(FutureDB.request_id)
    )

    if model_id:
        query = query.where(FutureDB.model_id == model_id)

    result = await session.execute(query)
    all_pending = result.scalars().all()

    for future in all_pending:
        if future.request_type in SINGLE_TYPES:
            return future
        # If there are batchable requests before a single request,
        # the single request must wait
        if future.request_type in BATCHABLE_TRAIN_TYPES | BATCHABLE_SAMPLE_TYPES:
            return None

    return None


async def find_sample_requests(
    session: AsyncSession,
    max_batch_size: int = 8,
) -> list[FutureDB]:
    """Find pending sample requests that can be batched."""
    query = (
        select(FutureDB)
        .where(
            FutureDB.status == RequestStatus.PENDING,
            FutureDB.request_type == RequestType.SAMPLE,
        )
        .order_by(FutureDB.request_id)
        .limit(max_batch_size)
    )

    result = await session.execute(query)
    return list(result.scalars().all())


async def complete_future(
    session: AsyncSession,
    request_id: int,
    result_data: dict,
    status: RequestStatus = RequestStatus.COMPLETED,
) -> None:
    """Mark a future as completed with its result data."""
    future = await session.get(FutureDB, request_id)
    if future is None:
        logger.warning("Tried to complete non-existent future %d", request_id)
        return

    future.status = status
    future.result_data = result_data
    future.completed_at = datetime.now(timezone.utc)
    session.add(future)
    await session.commit()


async def fail_future(
    session: AsyncSession,
    request_id: int,
    error: str,
) -> None:
    """Mark a future as failed with an error message."""
    await complete_future(
        session,
        request_id,
        result_data={"error": error},
        status=RequestStatus.FAILED,
    )
