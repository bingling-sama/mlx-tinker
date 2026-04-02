"""End-to-end local RL training loop for OpenClaw + mlx-tinker.

.. deprecated::
    This module duplicates the learning layer from upstream OpenClaw-RL
    (openclaw-tinker/trainer.py). The canonical path is to run upstream
    ``openclaw-tinker/run.py`` directly against mlx-tinker via::

        TINKER_BASE_URL=http://localhost:8010 TINKER_API_KEY=tml-local \\
            python .external/openclaw-rl/openclaw-tinker/run.py --method combine

    Or use the wrapper: ``bash scripts/run_openclaw_rl.sh --method combine``

    This local trainer is retained for quick experiments but will be removed
    once end-to-end validation passes against the upstream path.

Supports all three methods:
  - rl: Binary RL with GRPO advantages (scalar reward broadcast)
  - opd: On-Policy Distillation with teacher logprobs + scalar advantage
  - combine: Combined OPD + RL with weighted per-token advantages
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tinker
from transformers import AutoTokenizer

from .adapter import LocalTinkerClientFactory
from .api_server import OpenClawLocalProxy
from .config import LocalRLConfig
from .curriculum import TaskSpec, load_curriculum, split_curriculum
from .data_formatter import (
    TrainingSample,
    batch_to_datums,
    batch_to_datums_combined,
)
from .gateway_client import OpenClawGatewayClient
from .scorers import LocalTeacherLogprobExtractor
from .user_simulator import OpenClawUserSimulator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepReport:
    step: int
    task_id: str
    mean_reward: float
    passed_rollouts: int
    losses: tuple[float, ...]
    method: str = ""
    sample_types: dict[str, int] | None = None


class LocalRLTrainer:
    def __init__(self, config: LocalRLConfig) -> None:
        self.config = config
        self.client_factory = LocalTinkerClientFactory(
            base_url=config.runtime_urls.mlx_tinker_base_url,
            api_key=config.mlx_tinker_api_key,
            model_name=config.model_name,
        )
        self.training_client = None
        self.proxy = None
        self.gateway_client = None
        self.simulator = None
        self.tokenizer = None
        self.teacher_extractor: LocalTeacherLogprobExtractor | None = None
        self.train_tasks: list[TaskSpec] = []
        self.eval_tasks: list[TaskSpec] = []

    async def setup(self) -> None:
        logger.info("Loading tokenizer for %s", self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        self.train_tasks, self.eval_tasks = split_curriculum(load_curriculum(self.config.curriculum_path))

        self.training_client = await self.client_factory.create_training_client(rank=self.config.lora_rank)
        sampling_client = await self.training_client.save_weights_and_get_sampling_client_async(
            name=f"{self.config.save_name_prefix}_bootstrap"
        )
        self.proxy = OpenClawLocalProxy(
            sampling_client=sampling_client,
            tokenizer=self.tokenizer,
            served_model_name=self.config.served_model_name,
            api_key=self.config.mlx_tinker_api_key,
            host=self.config.proxy_host,
            port=self.config.proxy_port,
            max_completion_tokens=self.config.proxy_max_tokens,
        )
        self.proxy.start()

        self.gateway_client = OpenClawGatewayClient(
            ws_url=self.config.gateway_ws_url,
            token=self.config.gateway_token,
            request_timeout_ms=self.config.request_timeout_ms,
        )
        await self.gateway_client.connect()
        self.simulator = OpenClawUserSimulator(self.gateway_client)

        # Teacher model for OPD / combine methods
        if self.config.method in ("opd", "combine"):
            logger.info(
                "Creating teacher sampling client (base model: %s)",
                self.config.resolved_teacher_model(),
            )
            teacher_client = await self.client_factory.create_teacher_client(
                teacher_model_name=(
                    self.config.teacher_model_name
                    if self.config.teacher_model_name
                    else None
                ),
            )
            self.teacher_extractor = LocalTeacherLogprobExtractor(
                sampling_client=teacher_client,
                tokenizer=self.tokenizer,
            )
            logger.info("Teacher logprob extractor ready")

    async def close(self) -> None:
        if self.gateway_client is not None:
            await self.gateway_client.close()
        if self.proxy is not None:
            self.proxy.stop()
        self.client_factory.close()

    async def evaluate(self, tasks: list[TaskSpec]) -> dict[str, Any]:
        assert self.simulator is not None
        results = []
        for task in tasks:
            try:
                result = await self.simulator.run_task(task)
                results.append(
                    {
                        "task_id": task.task_id,
                        "passed": result.evaluation.passed,
                        "reward": result.evaluation.reward,
                        "failed_checks": list(result.evaluation.failed_checks),
                        "final_assistant_text": result.final_assistant_text,
                    }
                )
            except Exception as e:
                logger.warning("Eval task %s failed: %s", task.task_id, e)
                results.append(
                    {
                        "task_id": task.task_id,
                        "passed": False,
                        "reward": -1.0,
                        "failed_checks": [f"eval_error: {e}"],
                        "final_assistant_text": "",
                    }
                )
        mean_reward = sum(item["reward"] for item in results) / len(results) if results else 0.0
        pass_rate = sum(1 for item in results if item["passed"]) / len(results) if results else 0.0
        return {"mean_reward": mean_reward, "pass_rate": pass_rate, "results": results}

    async def run(self) -> dict[str, Any]:
        await self.setup()
        assert self.simulator is not None
        assert self.training_client is not None
        assert self.proxy is not None

        baseline_eval = await self.evaluate(self.eval_tasks)
        step_reports: list[dict[str, Any]] = []

        for step in range(1, self.config.max_steps + 1):
            task = self.train_tasks[(step - 1) % len(self.train_tasks)]
            rollout_payloads = []
            losses = []

            for rollout_idx in range(self.config.rollout_count):
                try:
                    cursor = self.proxy.record_cursor()
                    task_result = await self.simulator.run_task(task)
                    records = self.proxy.records_since(cursor)
                    rollout_payloads.append((task_result, records))
                except Exception as e:
                    logger.warning(
                        "Step %d rollout %d failed for %s: %s",
                        step, rollout_idx, task.task_id, e,
                    )

            training_samples = self._build_training_samples(task.task_id, rollout_payloads)

            # Extract teacher logprobs for OPD / combine methods
            if self.config.method in ("opd", "combine") and self.teacher_extractor:
                training_samples = await self._enrich_with_teacher_logprobs(
                    task, training_samples
                )

            # Duplicate samples for multi-epoch training (combine typically uses 2)
            if self.config.train_epochs > 1 and training_samples:
                original = list(training_samples)
                for _ in range(self.config.train_epochs - 1):
                    training_samples.extend(original)
                logger.info(
                    "Duplicated %d samples x%d = %d for training",
                    len(original),
                    self.config.train_epochs,
                    len(training_samples),
                )

            if training_samples:
                datums = self._build_datums(training_samples)
                if datums:
                    # Process datums in micro-batches to avoid GPU OOM.
                    # Gradients accumulate across forward_backward calls;
                    # a single optim_step applies the combined update.
                    bs = max(1, self.config.batch_size)
                    step_loss = 0.0
                    for i in range(0, len(datums), bs):
                        micro = datums[i : i + bs]
                        fb_future = await self.training_client.forward_backward_async(
                            micro, loss_fn=self.config.loss_fn
                        )
                        fb_result = await fb_future.result_async(timeout=600)
                        step_loss += float(fb_result.metrics.get("loss:sum", 0.0))
                    losses.append(step_loss)
                    opt_future = await self.training_client.optim_step_async(
                        tinker.AdamParams(learning_rate=self.config.learning_rate)
                    )
                    await opt_future.result_async(timeout=600)
                    sampling_client = await self.training_client.save_weights_and_get_sampling_client_async(
                        name=f"{self.config.save_name_prefix}_step_{step:04d}"
                    )
                    self.proxy.update_sampling_client(sampling_client)

            rewards = [payload[0].evaluation.reward for payload in rollout_payloads]
            sample_types = self._count_sample_types(training_samples)
            step_reports.append(
                StepReport(
                    step=step,
                    task_id=task.task_id,
                    mean_reward=sum(rewards) / len(rewards),
                    passed_rollouts=sum(1 for reward in rewards if reward > 0),
                    losses=tuple(losses),
                    method=self.config.method,
                    sample_types=sample_types,
                ).__dict__
            )
            self._log_step(step, task.task_id, rewards, losses, training_samples)

        final_eval = await self.evaluate(self.eval_tasks)
        report = {
            "model_name": self.config.model_name,
            "method": self.config.method,
            "loss_fn": self.config.loss_fn,
            "curriculum_path": str(self.config.curriculum_path),
            "runtime": self.config.container_runtime.value,
            "provider_base_url": self.config.runtime_urls.provider_base_url,
            "w_opd": self.config.w_opd,
            "w_rl": self.config.w_rl,
            "train_epochs": self.config.train_epochs,
            "baseline_eval": baseline_eval,
            "final_eval": final_eval,
            "steps": step_reports,
        }
        self._write_report(report)
        await self.close()
        return report

    def _build_training_samples(
        self,
        task_id: str,
        rollout_payloads: list[tuple[Any, list[Any]]],
    ) -> list[TrainingSample]:
        samples: list[TrainingSample] = []
        for task_result, records in rollout_payloads:
            for record in records:
                if not record.response_tokens:
                    continue
                samples.append(
                    TrainingSample(
                        task_id=task_id,
                        prompt_tokens=list(record.prompt_tokens),
                        response_tokens=list(record.response_tokens),
                        response_logprobs=list(record.response_logprobs),
                        reward=task_result.evaluation.reward,
                        prompt_text=record.prompt_text,
                        response_text=record.response_text,
                        loss_mask=[1] * len(record.response_tokens),
                    )
                )
        return samples

    async def _enrich_with_teacher_logprobs(
        self,
        task: TaskSpec,
        samples: list[TrainingSample],
    ) -> list[TrainingSample]:
        """Extract teacher logprobs and set sample_type for each sample."""
        assert self.teacher_extractor is not None
        enriched: list[TrainingSample] = []
        for sample in samples:
            try:
                teacher_lps = await self.teacher_extractor.extract_teacher_logprobs(
                    task=task,
                    prompt_text=sample.prompt_text,
                    response_tokens=list(sample.response_tokens),
                    response_text=sample.response_text,
                )
                has_teacher = any(lp != 0.0 for lp in teacher_lps)
                has_reward = sample.reward != 0.0

                if self.config.method == "combine":
                    if has_teacher and has_reward:
                        sample_type = "opd+rl"
                    elif has_teacher:
                        sample_type = "opd"
                    elif has_reward:
                        sample_type = "rl"
                    else:
                        sample_type = ""
                else:
                    sample_type = "opd" if has_teacher else ""

                enriched.append(
                    TrainingSample(
                        task_id=sample.task_id,
                        prompt_tokens=sample.prompt_tokens,
                        response_tokens=sample.response_tokens,
                        response_logprobs=sample.response_logprobs,
                        reward=sample.reward,
                        prompt_text=sample.prompt_text,
                        response_text=sample.response_text,
                        teacher_logprobs=teacher_lps,
                        loss_mask=sample.loss_mask,
                        sample_type=sample_type,
                    )
                )
            except Exception as e:
                logger.warning(
                    "Teacher logprob extraction failed for task=%s: %s",
                    sample.task_id,
                    e,
                )
                # Fall back to the original sample without teacher logprobs
                enriched.append(sample)
        return enriched

    def _build_datums(self, training_samples: list[TrainingSample]) -> list:
        """Convert training samples to Tinker Datums based on method."""
        if self.config.method == "combine":
            return batch_to_datums_combined(
                training_samples,
                w_opd=self.config.w_opd,
                w_rl=self.config.w_rl,
                max_prompt_tokens=self.config.train_max_prompt_tokens,
                max_response_tokens=self.config.train_max_response_tokens,
            )
        else:
            return batch_to_datums(
                training_samples,
                max_prompt_tokens=self.config.train_max_prompt_tokens,
                max_response_tokens=self.config.train_max_response_tokens,
            )

    def _count_sample_types(self, samples: list[TrainingSample]) -> dict[str, int]:
        """Count sample types for logging."""
        counts: dict[str, int] = {"opd+rl": 0, "opd": 0, "rl": 0, "": 0}
        for sample in samples:
            key = sample.sample_type if sample.sample_type in counts else ""
            counts[key] += 1
        return counts

    def _log_step(
        self,
        step: int,
        task_id: str,
        rewards: list[float],
        losses: list[float],
        samples: list[TrainingSample],
    ) -> None:
        """Log step summary based on method."""
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        method = self.config.method

        if method == "rl":
            success = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0
            logger.info(
                "step %d task=%s | mean_reward=%.3f success=%.2f losses=%s",
                step, task_id, mean_r, success, losses,
            )
        elif method == "opd":
            has_teacher = sum(1 for s in samples if s.teacher_logprobs is not None)
            logger.info(
                "step %d task=%s | mean_reward=%.3f teacher_samples=%d/%d losses=%s",
                step, task_id, mean_r, has_teacher, len(samples), losses,
            )
        elif method == "combine":
            types = self._count_sample_types(samples)
            logger.info(
                "step %d task=%s | mean_reward=%.3f "
                "opd+rl=%d opd=%d rl=%d none=%d losses=%s",
                step, task_id, mean_r,
                types["opd+rl"], types["opd"], types["rl"], types[""],
                losses,
            )

    def _write_report(self, report: dict[str, Any]) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.output_dir / "local_rl_report.json"
        path.write_text(json.dumps(report, indent=2))
        logger.info("Wrote local RL report to %s", path)
        return path
