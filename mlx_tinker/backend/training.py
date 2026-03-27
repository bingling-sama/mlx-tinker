"""Training operations: forward_backward, forward, optim_step using MLX autodiff."""

from __future__ import annotations

import logging
from operator import add

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from mlx_tinker.backend.loss_fns import LOSS_FUNCTION_MAP, LossFnConfig
from mlx_tinker.types import (
    ForwardBackwardInput,
    ForwardBackwardOutput,
    ForwardInput,
    ForwardOutput,
    OptimStepInput,
    OptimStepOutput,
)

logger = logging.getLogger(__name__)


class TrainingBackend:
    """Handles gradient computation, accumulation, and optimizer steps on MLX."""

    def __init__(self) -> None:
        self.accumulated_grads: dict[str, dict | None] = {}
        self.grad_accum_counts: dict[str, int] = {}
        self.optimizers: dict[str, optim.OptimizerBase] = {}

    def ensure_optimizer(self, model_id: str, model: nn.Module) -> None:
        """Lazily create an AdamW optimizer for the model."""
        if model_id not in self.optimizers:
            self.optimizers[model_id] = optim.AdamW(learning_rate=1e-5)
            self.accumulated_grads[model_id] = None
            self.grad_accum_counts[model_id] = 0

    def forward_backward(
        self,
        model_id: str,
        model: nn.Module,
        request: ForwardBackwardInput,
    ) -> ForwardBackwardOutput:
        """Compute loss and accumulate gradients without applying them.

        Gradients are accumulated across multiple forward_backward calls
        until optim_step is invoked.
        """
        self.ensure_optimizer(model_id, model)
        loss_fn_impl = LOSS_FUNCTION_MAP[request.loss_fn]

        # Parse loss_fn_config
        cfg = LossFnConfig()
        if request.loss_fn_config:
            for key in request.loss_fn_config:
                if key not in ("clip_low_threshold", "clip_high_threshold"):
                    logger.warning("Unknown loss_fn_config key ignored: %s", key)
            if "clip_low_threshold" in request.loss_fn_config:
                cfg.clip_low_threshold = float(request.loss_fn_config["clip_low_threshold"])
            if "clip_high_threshold" in request.loss_fn_config:
                cfg.clip_high_threshold = float(request.loss_fn_config["clip_high_threshold"])
            if cfg.clip_low_threshold < 0:
                raise ValueError(f"clip_low_threshold must be >= 0, got {cfg.clip_low_threshold}")
            if cfg.clip_high_threshold < 0:
                raise ValueError(f"clip_high_threshold must be >= 0, got {cfg.clip_high_threshold}")

        # Prepare batched tensors from request data
        all_losses = []
        all_grads = []

        for datum in request.data:
            input_tokens = mx.array(datum.model_input.get_tokens())[None, :]  # [1, T]
            target_tokens = mx.array(datum.loss_fn_inputs.target_tokens.data, dtype=mx.int32)[
                None, :
            ]
            token_weights = mx.array(datum.loss_fn_inputs.weights.data, dtype=mx.float32)[None, :]
            advantages = mx.array(datum.loss_fn_inputs.advantages.data, dtype=mx.float32)[None, :]
            sampling_logprobs = mx.array(datum.loss_fn_inputs.logprobs.data, dtype=mx.float32)[
                None, :
            ]

            # Trim to same length
            input_len = input_tokens.shape[1]
            target_len = target_tokens.shape[1]
            weights_len = token_weights.shape[1]
            seq_len = min(input_len, target_len, weights_len)
            if not (input_len == target_len == weights_len):
                logger.warning(
                    "Sequence length mismatch: input=%d, target=%d, weights=%d (truncating to %d)",
                    input_len,
                    target_len,
                    weights_len,
                    seq_len,
                )
            input_tokens = input_tokens[:, :seq_len]
            target_tokens = target_tokens[:, :seq_len]
            token_weights = token_weights[:, :seq_len]
            if advantages.shape[1] > 0:
                advantages = advantages[:, :seq_len]
            if sampling_logprobs.shape[1] > 0:
                sampling_logprobs = sampling_logprobs[:, :seq_len]

            def compute_loss(
                model: nn.Module,
                input_ids: mx.array,
                targets: mx.array,
                weights: mx.array,
                adv: mx.array,
                samp_lp: mx.array,
            ) -> mx.array:
                logits = model(input_ids)
                # Compute log probs of target tokens
                log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                # Gather target token log probs
                target_lp = mx.take_along_axis(
                    log_probs, targets[:, :, None].astype(mx.int32), axis=-1
                ).squeeze(-1)
                return loss_fn_impl(target_lp, weights, samp_lp, adv, cfg)

            loss_and_grad = nn.value_and_grad(model, compute_loss)
            loss_val, grads = loss_and_grad(
                model,
                input_tokens,
                target_tokens,
                token_weights,
                advantages,
                sampling_logprobs,
            )
            mx.eval(loss_val)
            all_losses.append(loss_val.item())
            all_grads.append(grads)

        # Accumulate gradients
        combined_grads = all_grads[0]
        for g in all_grads[1:]:
            combined_grads = tree_map(add, combined_grads, g)

        if self.accumulated_grads[model_id] is None:
            self.accumulated_grads[model_id] = combined_grads
        else:
            self.accumulated_grads[model_id] = tree_map(
                add, self.accumulated_grads[model_id], combined_grads
            )
        self.grad_accum_counts[model_id] += len(request.data)

        logger.info(
            "forward_backward model=%s loss=%.6f n_datum=%d accum_count=%d",
            model_id,
            sum(all_losses) / len(all_losses),
            len(request.data),
            self.grad_accum_counts[model_id],
        )

        return ForwardBackwardOutput(
            loss_fn_output_type=request.loss_fn,
            loss_fn_outputs=[{"loss": v} for v in all_losses],
            metrics={
                "mean_loss": sum(all_losses) / len(all_losses),
                "num_sequences": len(request.data),
            },
        )

    def forward(
        self,
        model_id: str,
        model: nn.Module,
        request: ForwardInput,
    ) -> ForwardOutput:
        """Forward pass only — return per-token log probabilities without gradients."""
        all_logprobs = []

        for datum in request.data:
            input_tokens = mx.array(datum.model_input.get_tokens())[None, :]
            logits = model(input_tokens)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

            target_tokens = mx.array(datum.loss_fn_inputs.target_tokens.data, dtype=mx.int32)[
                None, :
            ]
            seq_len = min(log_probs.shape[1], target_tokens.shape[1])
            target_lp = mx.take_along_axis(
                log_probs[:, :seq_len], target_tokens[:, :seq_len, None].astype(mx.int32), axis=-1
            ).squeeze(-1)
            mx.eval(target_lp)
            all_logprobs.append(target_lp[0].tolist())

        return ForwardOutput(
            logprobs=all_logprobs,
            metrics={"num_sequences": len(request.data)},
        )

    def optim_step(
        self,
        model_id: str,
        model: nn.Module,
        request: OptimStepInput,
    ) -> OptimStepOutput:
        """Apply accumulated gradients to model parameters via AdamW."""
        self.ensure_optimizer(model_id, model)
        optimizer = self.optimizers[model_id]
        grads = self.accumulated_grads[model_id]

        if grads is None:
            logger.warning("optim_step called with no accumulated gradients for model=%s", model_id)
            return OptimStepOutput(metrics={"warning": "no_gradients"})

        # Update optimizer hyperparams
        ap = request.adam_params
        optimizer.learning_rate = ap.learning_rate
        # Note: MLX AdamW constructor sets betas; we recreate if they changed
        if hasattr(optimizer, "betas"):
            if optimizer.betas != (ap.beta1, ap.beta2):
                optimizer = optim.AdamW(
                    learning_rate=ap.learning_rate,
                    betas=(ap.beta1, ap.beta2),
                    eps=ap.eps,
                    weight_decay=ap.weight_decay,
                )
                self.optimizers[model_id] = optimizer

        # Average gradients if multiple forward_backward calls accumulated
        n = self.grad_accum_counts[model_id]
        if n > 1:
            grads = tree_map(lambda g: g / n, grads)

        # Apply
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Clear accumulation
        self.accumulated_grads[model_id] = None
        self.grad_accum_counts[model_id] = 0

        logger.info("optim_step model=%s lr=%.2e grad_accum=%d", model_id, ap.learning_rate, n)

        return OptimStepOutput(metrics={"learning_rate": ap.learning_rate, "grad_accum_steps": n})

    def get_optimizer_state(self, model_id: str) -> dict | None:
        """Return serializable optimizer state for checkpointing."""
        if model_id not in self.optimizers:
            return None
        return self.optimizers[model_id].state

    def load_optimizer_state(self, model_id: str, state: dict) -> None:
        """Restore optimizer state from a checkpoint."""
        if model_id in self.optimizers:
            self.optimizers[model_id].state = state
