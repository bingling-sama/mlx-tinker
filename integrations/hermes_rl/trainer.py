"""Trainer for the Hermes live-RL bridge on top of local mlx-tinker."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os

from transformers import AutoTokenizer

from integrations.openclaw_rl_local.adapter import LocalTinkerClientFactory

from .config import HermesRLConfig
from .data_formatter import (
    TrainingSample,
    batch_to_datums,
    batch_to_datums_combined,
    compute_grpo_advantages,
)
from .rollout import RolloutWorker, drain_output_queue

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_RESET = "\033[0m"


class Trainer:
    """End-to-end Hermes live-RL trainer using mlx-tinker-backed Tinker clients."""

    def __init__(self, config: HermesRLConfig):
        self.config = config
        self.client_factory: LocalTinkerClientFactory | None = None
        self.training_client = None
        self.sampling_client = None
        self.rollout_worker: RolloutWorker | None = None
        self._wandb = None

    async def setup(self) -> None:
        if os.environ.get("WANDB_DISABLED", "").strip().lower() not in {"1", "true", "yes"}:
            try:
                wandb = importlib.import_module("wandb")
                self._wandb = wandb.init(
                    project=self.config.wandb_project,
                    name=os.environ.get("WANDB_RUN_NAME", ""),
                )
            except Exception as exc:
                logger.warning("[Trainer] wandb init failed: %s", exc)

        logger.info("[Trainer] loading tokenizer for %s", self.config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)

        self.client_factory = LocalTinkerClientFactory(
            base_url=self.config.tinker_base_url,
            api_key=self.config.tinker_api_key,
            model_name=self.config.model_name,
        )
        self.training_client = await self.client_factory.create_training_client(
            rank=self.config.lora_rank
        )

        if self.config.resume_from_ckpt:
            logger.info("[Trainer] resuming from checkpoint: %s", self.config.resume_from_ckpt)
            await self.training_client.load_state_async(self.config.resume_from_ckpt)

        self.sampling_client = (
            await self.training_client.save_weights_and_get_sampling_client_async(
                name="hermes_rl_bootstrap"
            )
        )
        logger.info("[Trainer] initial sampling client ready")

        teacher_client = await self.client_factory.create_teacher_client(
            teacher_model_name=self.config.resolved_teacher_model()
        )
        logger.info("[Trainer] teacher sampling client ready")

        method = self.config.method.lower()
        if method == "rl":
            from .scorers import PRMScorer

            scorer = PRMScorer(
                teacher_sampling_client=teacher_client,
                tokenizer=tokenizer,
                prm_m=self.config.prm_m,
                temperature=self.config.prm_temperature,
                max_tokens=self.config.prm_max_tokens,
            )
        elif method == "opd":
            from .scorers import OPDScorer

            scorer = OPDScorer(
                teacher_sampling_client=teacher_client,
                tokenizer=tokenizer,
                prm_m=self.config.prm_m,
                temperature=self.config.prm_temperature,
                max_tokens=self.config.prm_max_tokens,
                eval_mode=self.config.eval_mode,
            )
        elif method == "combine":
            from .scorers import CombinedScorer

            scorer = CombinedScorer(
                teacher_sampling_client=teacher_client,
                tokenizer=tokenizer,
                prm_m=self.config.prm_m,
                temperature=self.config.prm_temperature,
                max_tokens=self.config.prm_max_tokens,
            )
        else:
            raise ValueError(f"Unknown Hermes RL method: {method!r}")

        self.rollout_worker = RolloutWorker(
            config=self.config,
            sampling_client=self.sampling_client,
            scorer=scorer,
            tokenizer=tokenizer,
        )

    async def _train_on_batch(self, batch: list[TrainingSample], step: int) -> None:
        import tinker

        method = self.config.method.lower()
        max_tok = self.config.max_context_tokens
        if method == "combine":
            datums = batch_to_datums_combined(
                batch,
                w_opd=self.config.w_opd,
                w_rl=self.config.w_rl,
                max_tokens=max_tok,
            )
        else:
            advantages = compute_grpo_advantages(batch)
            datums = batch_to_datums(batch, advantages, max_tokens=max_tok)

        if not datums:
            logger.error("[Trainer] empty batch at step %d; skipping", step)
            return

        if len(datums) < len(batch):
            logger.warning(
                "[Trainer] only %d/%d samples converted to datums at step %d",
                len(datums),
                len(batch),
                step,
            )

        logger.info("[Trainer] step %d: forward_backward (%d datums)", step, len(datums))
        fb_future = await self.training_client.forward_backward_async(
            datums,
            loss_fn=self.config.loss_fn,
        )
        fb_output = await fb_future.result_async()
        logger.info(
            "[Trainer] step %d: forward_backward done metrics=%s",
            step,
            getattr(fb_output, "metrics", None),
        )

        logger.info("[Trainer] step %d: optim_step", step)
        optim_future = await self.training_client.optim_step_async(
            tinker.AdamParams(learning_rate=self.config.learning_rate)
        )
        optim_output = await optim_future.result_async()
        logger.info(
            "[Trainer] step %d: optim_step done metrics=%s",
            step,
            getattr(optim_output, "metrics", None),
        )

        logger.info("[Trainer] step %d: pausing inference for weight swap", step)
        assert self.rollout_worker is not None
        self.rollout_worker.pause_submission()
        try:
            self.sampling_client = await asyncio.wait_for(
                self.training_client.save_weights_and_get_sampling_client_async(
                    name=f"hermes_{method}_lora"
                ),
                timeout=self.config.save_weights_timeout,
            )
        finally:
            self.rollout_worker.update_sampling_client(self.sampling_client)
            self.rollout_worker.resume_submission()
        logger.info("[Trainer] step %d: inference resumed", step)

        if step % self.config.save_interval == 0 or step == self.config.max_steps:
            try:
                resolved = await self.training_client.save_state_async(name=f"step_{step:04d}")
                logger.info("[Trainer] checkpoint saved: %s", getattr(resolved, "path", ""))
            except Exception as exc:
                logger.error("[Trainer] save_state failed at step %d: %s", step, exc, exc_info=True)

        self._log_step(batch, step, method)

    def _log_step(self, batch: list[TrainingSample], step: int, method: str) -> None:
        rewards = [sample.reward for sample in batch]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        log_dict = {
            "train/step": step,
            "train/mean_reward": mean_reward,
            "train/batch_size": len(batch),
        }

        if method == "rl":
            success = sum(1 for reward in rewards if reward > 0) / len(rewards) if rewards else 0.0
            logger.info(
                "%s[Trainer] step %d done | batch=%d mean_reward=%.3f success=%.2f%s",
                _GREEN,
                step,
                len(batch),
                mean_reward,
                success,
                _RESET,
            )
            log_dict["train/success_rate"] = success
        elif method == "opd":
            teacher_samples = sum(1 for sample in batch if sample.teacher_logprobs is not None)
            logger.info(
                "%s[Trainer] step %d done | batch=%d mean_reward=%.3f teacher_samples=%d%s",
                _GREEN,
                step,
                len(batch),
                mean_reward,
                teacher_samples,
                _RESET,
            )
            log_dict["train/teacher_samples"] = teacher_samples
        elif method == "combine":
            sample_types = {"opd+rl": 0, "opd": 0, "rl": 0}
            for sample in batch:
                if sample.sample_type in sample_types:
                    sample_types[sample.sample_type] += 1
            logger.info(
                "%s[Trainer] step %d done | batch=%d mean_reward=%.3f opd+rl=%d opd=%d rl=%d%s",
                _GREEN,
                step,
                len(batch),
                mean_reward,
                sample_types["opd+rl"],
                sample_types["opd"],
                sample_types["rl"],
                _RESET,
            )
            log_dict.update(
                {
                    "train/opd_rl_samples": sample_types["opd+rl"],
                    "train/opd_only_samples": sample_types["opd"],
                    "train/rl_only_samples": sample_types["rl"],
                }
            )

        if self._wandb:
            self._wandb.log(log_dict, step=step)

    async def run(self) -> None:
        await self.setup()
        assert self.rollout_worker is not None

        self.rollout_worker.start()
        self.rollout_worker.resume_submission()
        logger.info(
            "[Trainer] Hermes proxy starting at %s:%d (method=%s)",
            self.config.proxy_host,
            self.config.proxy_port,
            self.config.method,
        )

        for step in range(1, self.config.max_steps + 1):
            logger.info(
                "[Trainer] step %d/%d - collecting batch (size=%d)",
                step,
                self.config.max_steps,
                self.config.batch_size,
            )
            self.rollout_worker.reset_eval_scores()
            groups = await drain_output_queue(self.config.batch_size, self.rollout_worker)
            batch = [sample for group in groups for sample in group]

            eval_scores = self.rollout_worker.drain_eval_scores()
            if eval_scores:
                average = sum(eval_scores) / len(eval_scores)
                logger.info("[Trainer] prm_eval_score=%.4f (n=%d)", average, len(eval_scores))
                if self._wandb:
                    self._wandb.log({"rollout/prm_eval_score": average}, step=step)

            await self._train_on_batch(batch, step)

        logger.info("[Trainer] training complete (%d steps)", self.config.max_steps)
        self.cleanup()

    def cleanup(self) -> None:
        if self._wandb:
            self._wandb.finish()
        if self.rollout_worker:
            self.rollout_worker.stop()
        if self.client_factory:
            self.client_factory.close()
