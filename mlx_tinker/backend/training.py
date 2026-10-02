"""Training operations: forward_backward, forward, optim_step using MLX autodiff."""

from __future__ import annotations

import logging
import math
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_reduce

from mlx_tinker.backend.loss_fns import (
    LOSS_FUNCTION_MAP,
    LossFnConfig,
    chunked_target_logprobs,
)
from mlx_tinker.backend.optimizers import AdamW8Bit
from mlx_tinker.types import (
    ForwardBackwardInput,
    ForwardBackwardOutput,
    ForwardInput,
    ForwardOutput,
    OptimStepInput,
    OptimStepOutput,
    TensorData,
)

logger = logging.getLogger(__name__)


def _has_split_lm_head(model: nn.Module) -> bool:
    """Check if model exposes a chunked-logprob compatible split head."""
    if not (
        hasattr(model, "model") and hasattr(model, "lm_head") and hasattr(model.lm_head, "weight")
    ):
        return False

    lm_head_weight = model.lm_head.weight
    if getattr(lm_head_weight, "ndim", 0) != 2:
        return False
    if getattr(model.lm_head, "bits", None) is not None:
        return False

    hidden_dim = None
    for embed_name in ("embed_tokens", "wte", "tok_embeddings", "embed"):
        embed = getattr(model.model, embed_name, None)
        if embed is None:
            embed = getattr(model, embed_name, None)
        weight = getattr(embed, "weight", None)
        if weight is not None and getattr(weight, "ndim", 0) >= 2:
            hidden_dim = weight.shape[-1]
            break

    if hidden_dim is not None and lm_head_weight.shape[-1] != hidden_dim:
        return False
    return True


def _materialize_tree(tree: dict, async_eval: bool = False) -> None:
    """Force evaluation of every leaf in a tree."""
    leaves = [leaf for _name, leaf in tree_flatten(tree)]
    if leaves:
        if async_eval:
            mx.async_eval(*leaves)
        else:
            mx.eval(*leaves)


def _grad_norm_array(grads: dict) -> mx.array:
    """Compute global gradient norm as a single MLX scalar."""
    total_norm_sq = tree_reduce(
        lambda acc, g: acc + mx.sum(mx.square(g.astype(mx.float32))),
        grads,
        mx.array(0.0, dtype=mx.float32),
    )
    return mx.sqrt(total_norm_sq)


def _clip_grad_norm(grads: dict, max_norm: float) -> tuple[dict, float, float]:
    """Clip gradient global norm to max_norm with a single sync boundary.

    Returns:
        (clipped_grads, pre_clip_norm, post_clip_norm)
    """
    grad_norm = _grad_norm_array(grads)
    grad_norm_clipped = grad_norm

    if max_norm > 0:
        max_norm_array = mx.array(max_norm, dtype=mx.float32)
        grad_norm_clipped = mx.minimum(grad_norm, max_norm_array)
        scale = mx.minimum(mx.array(1.0, dtype=mx.float32), max_norm_array / (grad_norm + 1e-6))
        grads = tree_map(lambda g: g * scale, grads)

    mx.eval(grad_norm, grad_norm_clipped)
    return grads, float(grad_norm.item()), float(grad_norm_clipped.item())


def _compute_target_logprobs(logits: mx.array, targets: mx.array) -> mx.array:
    """Gather target-token logprobs in float32 for stable loss computation."""
    logits_f32 = logits.astype(mx.float32)
    log_probs = logits_f32 - mx.logsumexp(logits_f32, axis=-1, keepdims=True)
    return mx.take_along_axis(log_probs, targets[:, :, None].astype(mx.int32), axis=-1).squeeze(-1)


def _pad_1d_rows(rows: list[list[float]] | list[list[int]], dtype) -> mx.array:
    """Pad variable-length 1D rows with zeros to form a dense 2D tensor."""
    if not rows:
        return mx.zeros((0, 0), dtype=dtype)
    max_len = max(len(row) for row in rows)
    padded = [row + [0] * (max_len - len(row)) for row in rows]
    return mx.array(padded, dtype=dtype)


def _prepare_batch_tensors(request: ForwardBackwardInput | ForwardInput) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    list[int],
]:
    """Convert a request into padded batch tensors with per-datum lengths."""
    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    weight_rows: list[list[float]] = []
    advantage_rows: list[list[float]] = []
    sampling_lp_rows: list[list[float]] = []
    seq_lens: list[int] = []

    for datum in request.data:
        lfi = datum.loss_fn_inputs
        input_tokens = list(datum.model_input.get_tokens())
        target_tokens = list(lfi.target_tokens.data)

        input_len = len(input_tokens)
        target_len = len(target_tokens)
        weights_len = len(lfi.weights.data) if lfi.weights is not None else min(input_len, target_len)
        advantages_len = (
            len(lfi.advantages.data) if lfi.advantages is not None else min(input_len, target_len)
        )
        sampling_lp_len = (
            len(lfi.logprobs.data) if lfi.logprobs is not None else min(input_len, target_len)
        )

        seq_len = min(input_len, target_len, weights_len, advantages_len, sampling_lp_len)
        if not (input_len == target_len == weights_len == advantages_len == sampling_lp_len):
            logger.warning(
                "Sequence length mismatch: input=%d target=%d weights=%d advantages=%d logprobs=%d "
                "(truncating to %d)",
                input_len,
                target_len,
                weights_len,
                advantages_len,
                sampling_lp_len,
                seq_len,
            )

        input_rows.append(input_tokens[:seq_len])
        target_rows.append(target_tokens[:seq_len])
        weight_rows.append(
            list(lfi.weights.data[:seq_len]) if lfi.weights is not None else [1.0] * seq_len
        )
        advantage_rows.append(
            list(lfi.advantages.data[:seq_len]) if lfi.advantages is not None else [0.0] * seq_len
        )
        sampling_lp_rows.append(
            list(lfi.logprobs.data[:seq_len]) if lfi.logprobs is not None else [0.0] * seq_len
        )
        seq_lens.append(seq_len)

    return (
        _pad_1d_rows(input_rows, mx.int32),
        _pad_1d_rows(target_rows, mx.int32),
        _pad_1d_rows(weight_rows, mx.float32),
        _pad_1d_rows(advantage_rows, mx.float32),
        _pad_1d_rows(sampling_lp_rows, mx.float32),
        seq_lens,
    )


class TrainingBackend:
    """Handles gradient computation, accumulation, and optimizer steps on MLX."""

    def __init__(
        self,
        optimizer_type: Literal["adamw_8bit", "adamw", "adafactor", "lion"] = "adamw",
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

    def _forward_backward_internal(
        self,
        model_id: str,
        model: nn.Module,
        request: ForwardBackwardInput,
    ) -> tuple[ForwardBackwardOutput, list[float | None]]:
        self.ensure_optimizer(model_id, model)
        loss_fn_impl = LOSS_FUNCTION_MAP[request.loss_fn]

        # Parse loss_fn_config
        cfg = LossFnConfig()
        known_keys = {"clip_low_threshold", "clip_high_threshold", "beta"}
        if request.loss_fn_config:
            for key in request.loss_fn_config:
                if key not in known_keys:
                    logger.warning("Unknown loss_fn_config key ignored: %s", key)
            if "clip_low_threshold" in request.loss_fn_config:
                cfg.clip_low_threshold = float(request.loss_fn_config["clip_low_threshold"])
            if "clip_high_threshold" in request.loss_fn_config:
                cfg.clip_high_threshold = float(request.loss_fn_config["clip_high_threshold"])
            if "beta" in request.loss_fn_config:
                cfg.beta = float(request.loss_fn_config["beta"])
            if cfg.clip_low_threshold < 0:
                raise ValueError(f"clip_low_threshold must be >= 0, got {cfg.clip_low_threshold}")
            if cfg.clip_high_threshold < 0:
                raise ValueError(f"clip_high_threshold must be >= 0, got {cfg.clip_high_threshold}")

        input_tokens, target_tokens, token_weights, advantages, sampling_logprobs, seq_lens = (
            _prepare_batch_tensors(request)
        )

        batch_token_count = mx.maximum(
            mx.sum((token_weights > 0).astype(mx.float32)),
            mx.array(1.0, dtype=mx.float32),
        )

        # Prefer the split-backbone path whenever we can compute target logprobs
        # directly from hidden states without materializing full-vocab logits.
        use_chunked = _has_split_lm_head(model)
        captured_logprobs = [None]

        def compute_loss(
            model: nn.Module,
            input_ids: mx.array,
            targets: mx.array,
            weights: mx.array,
            adv: mx.array,
            samp_lp: mx.array,
        ) -> mx.array:
            if use_chunked:
                hidden = model.model(input_ids)
                target_lp = chunked_target_logprobs(
                    hidden,
                    model.lm_head.weight,
                    targets,
                )
            else:
                logits = model(input_ids)
                target_lp = _compute_target_logprobs(logits, targets)
            captured_logprobs[0] = target_lp
            return loss_fn_impl(target_lp, weights, samp_lp, adv, cfg)

        loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

        loss_val, combined_grads = loss_and_grad_fn(
            model,
            input_tokens,
            target_tokens,
            token_weights,
            advantages,
            sampling_logprobs,
        )
        eval_targets = [loss_val, batch_token_count]
        if captured_logprobs[0] is not None:
            eval_targets.append(captured_logprobs[0])
        mx.eval(*eval_targets)

        if self.accumulated_grads[model_id] is None:
            self.accumulated_grads[model_id] = combined_grads
            _materialize_tree(self.accumulated_grads[model_id], async_eval=True)
        else:
            self.accumulated_grads[model_id] = tree_map(
                lambda acc, cur: acc + cur,
                self.accumulated_grads[model_id],
                combined_grads,
            )
            _materialize_tree(self.accumulated_grads[model_id], async_eval=True)
        self.grad_accum_counts[model_id] += len(request.data)
        self.total_tokens[model_id] += float(batch_token_count.item())

        all_logprobs_out = []
        per_seq_losses: list[float | None] = []
        if captured_logprobs[0] is not None:
            target_lp = captured_logprobs[0]
            for row_idx, seq_len in enumerate(seq_lens):
                lp_list = target_lp[row_idx, :seq_len].tolist()
                all_logprobs_out.append({"logprobs": TensorData(data=lp_list, dtype="float32")})
                row_lp = target_lp[row_idx, :seq_len]
                row_w = token_weights[row_idx, :seq_len]
                if request.loss_fn == "cross_entropy":
                    row_loss = float((-row_lp * row_w).sum().item())
                elif request.loss_fn == "importance_sampling":
                    row_adv = advantages[row_idx, :seq_len]
                    row_samp = sampling_logprobs[row_idx, :seq_len]
                    row_loss = float((-(mx.exp(row_lp - row_samp) * row_adv)).sum().item())
                else:
                    row_loss = None
                per_seq_losses.append(row_loss)
        else:
            all_logprobs_out = [
                {"logprobs": TensorData(data=[], dtype="float32")} for _ in request.data
            ]
            per_seq_losses = [None] * len(request.data)

        loss_sum = loss_val.item()
        logger.info(
            "forward_backward model=%s loss:sum=%.6f n_datum=%d accum_count=%d",
            model_id,
            loss_sum,
            len(request.data),
            self.grad_accum_counts[model_id],
        )

        output = ForwardBackwardOutput(
            loss_fn_output_type=request.loss_fn,
            loss_fn_outputs=[lp.copy() for lp in all_logprobs_out],
            metrics={
                "loss:sum": loss_sum,
                "num_sequences:sum": float(len(request.data)),
            },
        )
        return output, per_seq_losses

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
        output, _ = self._forward_backward_internal(model_id, model, request)
        return output

    def forward_backward_batch(
        self,
        model_id: str,
        model: nn.Module,
        requests: list[ForwardBackwardInput],
    ) -> list[ForwardBackwardOutput]:
        """Process a batch of ForwardBackwardInput requests with hardware coalescing."""
        if not requests:
            return []
        if len(requests) == 1:
            return [self.forward_backward(model_id, model, requests[0])]

        first = requests[0]
        can_coalesce = all(
            r.loss_fn == first.loss_fn and r.loss_fn_config == first.loss_fn_config
            for r in requests[1:]
        )
        if not can_coalesce:
            return [self.forward_backward(model_id, model, r) for r in requests]

        slice_lens = [len(r.data) for r in requests]
        combined_data = [d for r in requests for d in r.data]
        combined_request = ForwardBackwardInput(
            data=combined_data,
            loss_fn=first.loss_fn,
            loss_fn_config=first.loss_fn_config,
        )

        combined_output, per_seq_losses = self._forward_backward_internal(
            model_id, model, combined_request
        )

        outputs: list[ForwardBackwardOutput] = []
        offset = 0
        total_loss = combined_output.metrics.get("loss:sum", 0.0)
        total_sequences = sum(slice_lens)

        for count in slice_lens:
            req_outputs = combined_output.loss_fn_outputs[offset : offset + count]
            req_seq_losses = per_seq_losses[offset : offset + count]
            if all(sl is not None for sl in req_seq_losses):
                req_loss_sum = sum(req_seq_losses)
            else:
                req_loss_sum = (
                    total_loss * (count / total_sequences) if total_sequences > 0 else 0.0
                )

            outputs.append(
                ForwardBackwardOutput(
                    loss_fn_output_type=first.loss_fn,
                    loss_fn_outputs=req_outputs,
                    metrics={
                        "loss:sum": req_loss_sum,
                        "num_sequences:sum": float(count),
                    },
                )
            )
            offset += count

        return outputs

    def forward(
        self,
        model_id: str,
        model: nn.Module,
        request: ForwardInput,
    ) -> ForwardOutput:
        """Forward pass only — return per-token log probabilities without gradients."""
        input_tokens, target_tokens, _weights, _advantages, _sampling_logprobs, seq_lens = (
            _prepare_batch_tensors(request)
        )

        if _has_split_lm_head(model):
            hidden = model.model(input_tokens)
            target_lp = chunked_target_logprobs(hidden, model.lm_head.weight, target_tokens)
        else:
            logits = model(input_tokens)
            target_lp = _compute_target_logprobs(logits, target_tokens)
        mx.eval(target_lp)

        all_logprobs = [
            target_lp[row_idx, :seq_len].tolist() for row_idx, seq_len in enumerate(seq_lens)
        ]

        return ForwardOutput(
            logprobs=all_logprobs,
            metrics={"num_sequences:sum": len(request.data)},
        )

    def forward_batch(
        self,
        model_id: str,
        model: nn.Module,
        requests: list[ForwardInput],
    ) -> list[ForwardOutput]:
        """Process a batch of ForwardInput requests coalesced into a single forward pass."""
        if not requests:
            return []
        if len(requests) == 1:
            return [self.forward(model_id, model, requests[0])]

        slice_lens = [len(r.data) for r in requests]
        combined_data = [d for r in requests for d in r.data]
        combined_request = ForwardInput(data=combined_data)

        combined_output = self.forward(model_id, model, combined_request)
        outputs: list[ForwardOutput] = []
        offset = 0
        for count in slice_lens:
            outputs.append(
                ForwardOutput(
                    logprobs=combined_output.logprobs[offset : offset + count],
                    metrics={"num_sequences:sum": count},
                )
            )
            offset += count

        return outputs

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
            return OptimStepOutput(metrics={"grad_accum_steps:sum": 0})

        ap = request.adam_params

        # Update optimizer hyperparams
        optimizer.learning_rate = mx.array(ap.learning_rate, dtype=mx.float32)
        if hasattr(optimizer, "betas"):
            optimizer.betas = (ap.beta1, ap.beta2)
        if hasattr(optimizer, "eps"):
            optimizer.eps = ap.eps
        if hasattr(optimizer, "weight_decay"):
            optimizer.weight_decay = ap.weight_decay

        # Average gradients by total token count (Unsloth-style fix)
        total_tok = self.total_tokens[model_id]
        if total_tok > 1:
            grads = tree_map(lambda g: g / total_tok, grads)

        grads, grad_norm, grad_norm_clipped = _clip_grad_norm(grads, ap.grad_clip_norm)

        # Guard: skip update if gradient norm is non-finite
        if not math.isfinite(grad_norm):
            logger.warning(
                "optim_step skipped: non-finite grad_norm=%.4e for model=%s",
                grad_norm,
                model_id,
            )
            n = self.grad_accum_counts[model_id]
            self.accumulated_grads[model_id] = None
            self.grad_accum_counts[model_id] = 0
            self.total_tokens[model_id] = 0.0
            return OptimStepOutput(
                metrics={
                    "learning_rate:unique": ap.learning_rate,
                    "grad_accum_steps:sum": n,
                    "total_tokens:sum": total_tok,
                    "grad_norm:mean": grad_norm,
                    "grad_norm_clipped:mean": grad_norm,
                    "skipped:sum": 1.0,
                }
            )

        # Apply optimizer step
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Clear accumulation
        n = self.grad_accum_counts[model_id]
        self.accumulated_grads[model_id] = None
        self.grad_accum_counts[model_id] = 0
        self.total_tokens[model_id] = 0.0

        logger.info(
            "optim_step model=%s lr=%.2e grad_accum=%d total_tokens=%.0f grad_norm=%.4f",
            model_id,
            ap.learning_rate,
            n,
            total_tok,
            grad_norm,
        )

        return OptimStepOutput(
            metrics={
                "learning_rate:unique": ap.learning_rate,
                "grad_accum_steps:sum": n,
                "total_tokens:sum": total_tok,
                "grad_norm:mean": grad_norm,
                "grad_norm_clipped:mean": grad_norm_clipped,
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
