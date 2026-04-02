"""Async rollout worker for the Hermes live-RL bridge."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Optional

from .api_server import HermesCombineServer, HermesOPDServer, HermesRLServer
from .config import HermesRLConfig
from .data_formatter import TrainingSample


class RolloutWorker:
    """Owns the FastAPI proxy and exposes a queue of ready training groups."""

    def __init__(
        self,
        config: HermesRLConfig,
        *,
        sampling_client=None,
        scorer=None,
        tokenizer=None,
    ) -> None:
        self.config = config
        self.running = True
        self.output_queue: queue.Queue = queue.Queue(maxsize=100_000)
        self.worker_thread: Optional[threading.Thread] = None
        self._submission_enabled = threading.Event()

        method = config.method.lower()
        if method == "rl":
            self._server = HermesRLServer(
                config=config,
                output_queue=self.output_queue,
                submission_enabled=self._submission_enabled,
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                prm_scorer=scorer,
            )
        elif method == "opd":
            self._server = HermesOPDServer(
                config=config,
                output_queue=self.output_queue,
                submission_enabled=self._submission_enabled,
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                opd_scorer=scorer,
            )
        elif method == "combine":
            self._server = HermesCombineServer(
                config=config,
                output_queue=self.output_queue,
                submission_enabled=self._submission_enabled,
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                scorer=scorer,
            )
        else:
            raise ValueError(f"Unknown Hermes RL method: {method!r}")

    def start(self) -> None:
        self._server.start()
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=lambda: asyncio.run(self._keepalive()),
                daemon=True,
            )
            self.worker_thread.start()

    async def _keepalive(self) -> None:
        while self.running:
            await asyncio.sleep(1.0)

    def stop(self) -> None:
        self.running = False
        self._submission_enabled.clear()
        self._server.stop()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def pause_submission(self) -> None:
        if self._submission_enabled.is_set():
            self._submission_enabled.clear()
            self._server.purge_record_files()

    def resume_submission(self) -> None:
        if not self._submission_enabled.is_set():
            self._submission_enabled.set()

    def update_sampling_client(self, client) -> None:
        self._server.update_sampling_client(client)

    def drain_eval_scores(self) -> list[float]:
        return self._server.drain_eval_scores()

    def reset_eval_scores(self) -> None:
        self._server.reset_eval_scores()


async def drain_output_queue(batch_size: int, worker: RolloutWorker) -> list[list[TrainingSample]]:
    """Block until ``batch_size`` training groups are available."""

    data: list[list[TrainingSample]] = []
    pending: dict[int, list[TrainingSample]] = {}
    start = time.time()
    last_log = start

    while len(data) < batch_size:
        while True:
            try:
                group_id, group = worker.output_queue.get_nowait()
                pending[group_id] = group
            except queue.Empty:
                break

        for group_id in sorted(list(pending.keys())):
            if len(data) >= batch_size:
                break
            data.append(pending.pop(group_id))

        if time.time() - last_log > 30:
            print(
                f"[HermesRollout] waiting for samples: {len(data)}/{batch_size} "
                f"queue={worker.output_queue.qsize()}",
                flush=True,
            )
            last_log = time.time()

        if len(data) < batch_size:
            await asyncio.sleep(0.05)

    train_epochs = worker.config.train_epochs
    if train_epochs > 1:
        original = list(data)
        for _ in range(train_epochs - 1):
            data.extend(original)
        print(
            f"[HermesRollout] duplicated {len(original)} groups x{train_epochs} "
            f"= {len(data)} groups for training",
            flush=True,
        )

    print(f"[HermesRollout] drained {len(data)} groups in {time.time() - start:.2f}s", flush=True)
    return data
