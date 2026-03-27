"""Training operations: forward_backward, forward, optim_step using MLX autodiff."""

from __future__ import annotations

import logging
from operator import add
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from mlx_tinker.backend.loss_fns import (
    LOSS_FUNCTION_MAP,
    LossFnConfig,
    chunked_cross_entropy_loss,
)
from mlx_tinker.backend.optimizers import AdamW8Bit
from mlx_tinker.types import (
    ForwardBackwardInput,
    ForwardBackwardOutput,
    ForwardInput,
    ForwardOutput,
    OptimStepInput,
    OptimStepOutput,
)

logger = logging.getLogger(__name__)


def _has_split_lm_head(model: nn.Module) -> bool:
    """Check if model has a separate backbone (.model) and lm_head."""
    return (
        hasattr(model, "model") and hasattr(model, "lm_head") and hasattr(model.lm_head, "weight")
    )


def _clip_grad_norm(grads: dict, max_norm: float) -> dict:
    """Clip gradient global norm to max_norm."""
    from mlx.utils import tree_flatten

    flat = tree_flatten(grads)
    total_norm_sq = sum(mx.sum(mx.square(g)).item() for _, g in flat)
    total_norm = total_norm_sq**0.5

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        grads = tree_map(lambda g: g * scale, grads)

    return grads


class TrainingBackend:
    """Handles gradient computation, accumulation, and optimizer steps on MLX."""

    def __init__(
        self,
        optimizer_type: Literal["adamw_8bit", "adamw", "adafactor", "lion"] = "adamw_8bit",
        gradient_checkpointing: bool = True,
    ) -> None:
        self.optimizer_type = optimizer_type
        self.gradient_checkpointing = gradient_checkpointing

        self.accumulated_grads: dict[str, dict | None] = {}
        self.grad_accum_counts: dict[str, int] = {}
        self.total_tokens: dict[str, float] = {}
        self.optimizers: dict[str, optim.Optimizer] = {}

    def _create_optimizer(self, learning_rate: float = 1e-5) -> optim.Optimizer:
        """Create an optimizer based on the configured type."""
        if self.optimizer_type == "adamw_8bit":
            return AdamW8Bit(learning_rate=learning_rate)
        elif self.optimizer_type == "adamw":
            return optim.AdamW(learning_rate=learning_rate)
        elif self.optimizer_type == "adafactor":
            return optim.Adafactor(learning_rate=learning_rate)
        elif self.optimizer_type == "lion":
            return optim.Lion(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    def ensure_optimizer(self, model_id: str, model: nn.Module) -> None:
        """Lazily create an optimizer for the model."""
        if model_id not in self.optimizers:
            self.optimizers[model_id] = self._create_optimizer()
            self.accumulated_grads[model_id] = None
            self.grad_accum_counts[model_id] = 0
            self.total_tokens[model_id] = 0.0

    def forward_backward(
        self,
        model_id: str,
        model: nn.Module,
        request: ForwardBackwardInput,
    ) -> ForwardBackwardOutput:
        """Compute loss and accumulate gradients without applying them.

        Gradients are accumulated across multiple forward_backward calls
        until optim_step is invoked. Uses sum-reduction for correct
        gradient accumulation across variable-length sequences.
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
        batch_token_count = 0.0

        for datum in request.data:
            input_tokens = mx.array(datum.model_input.get_tokens())[None, :]
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

            # Track total unmasked tokens for correct gradient averaging
            mask_count = mx.sum(token_weights > 0).item()
            batch_token_count += max(mask_count, 1.0)

            # Use chunked CE for SFT when model supports split forward
            use_chunked = request.loss_fn == "cross_entropy" and _has_split_lm_head(model)

            def compute_loss(
                model: nn.Module,
                input_ids: mx.array,
                targets: mx.array,
                weights: mx.array,
                adv: mx.array,
                samp_lp: mx.array,
            ) -> mx.array:
                if use_chunked:
                    # Memory-efficient: never materialize [B, T, V]
                    hidden = model.model(input_ids)
                    return chunked_cross_entropy_loss(
                        hidden,
                        model.lm_head.weight,
                        targets,
                        weights,
                    )
                else:
                    logits = model(input_ids)
                    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                    target_lp = mx.take_along_axis(
                        log_probs,
                        targets[:, :, None].astype(mx.int32),
                        axis=-1,
                    ).squeeze(-1)
                    return loss_fn_impl(target_lp, weights, samp_lp, adv, cfg)

            loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

            loss_val, grads = loss_and_grad_fn(
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
        self.total_tokens[model_id] += batch_token_count

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
                log_probs[:, :seq_len],
                target_tokens[:, :seq_len, None].astype(mx.int32),
                axis=-1,
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
        """Apply accumulated gradients to model parameters."""
        self.ensure_optimizer(model_id, model)
        optimizer = self.optimizers[model_id]
        grads = self.accumulated_grads[model_id]

        if grads is None:
            logger.warning(
                "optim_step called with no accumulated gradients for model=%s",
                model_id,
            )
            return OptimStepOutput(metrics={"grad_accum_steps": 0})

        ap = request.adam_params

        # Update optimizer hyperparams
        optimizer.learning_rate = ap.learning_rate

        # Recreate optimizer if betas changed (for AdamW/AdamW8Bit)
        if hasattr(optimizer, "betas"):
            if optimizer.betas != (ap.beta1, ap.beta2):
                optimizer = self._create_optimizer(learning_rate=ap.learning_rate)
                if hasattr(optimizer, "betas"):
                    optimizer.betas = (ap.beta1, ap.beta2)
                if hasattr(optimizer, "eps"):
                    optimizer.eps = ap.eps
                if hasattr(optimizer, "weight_decay"):
                    optimizer.weight_decay = ap.weight_decay
                self.optimizers[model_id] = optimizer

        # Average gradients by total token count (Unsloth-style fix)
        total_tok = self.total_tokens[model_id]
        if total_tok > 1:
            grads = tree_map(lambda g: g / total_tok, grads)

        # Gradient clipping
        if ap.grad_clip_norm > 0:
            grads = _clip_grad_norm(grads, ap.grad_clip_norm)

        # Apply optimizer step
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Clear accumulation
        n = self.grad_accum_counts[model_id]
        self.accumulated_grads[model_id] = None
        self.grad_accum_counts[model_id] = 0
        self.total_tokens[model_id] = 0.0

        logger.info(
            "optim_step model=%s lr=%.2e grad_accum=%d total_tokens=%.0f",
            model_id,
            ap.learning_rate,
            n,
            total_tok,
        )

        return OptimStepOutput(
            metrics={
                "learning_rate": ap.learning_rate,
                "grad_accum_steps": n,
                "total_tokens": total_tok,
            }
        )

    def get_optimizer_state(self, model_id: str) -> dict | None:
        """Return serializable optimizer state for checkpointing."""
        if model_id not in self.optimizers:
            return None
        return self.optimizers[model_id].state

    def load_optimizer_state(self, model_id: str, state: dict) -> None:
        """Restore optimizer state from a checkpoint."""
        if model_id in self.optimizers:
            self.optimizers[model_id].state = state
