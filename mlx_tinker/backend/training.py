"""Training operations: forward_backward, forward, optim_step using MLX autodiff."""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Callable, Literal

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
    
    if max_norm > 0:
        max_norm_array = mx.array(max_norm, dtype=mx.float32)
        grad_norm_clipped = mx.minimum(grad_norm, max_norm_array)
        scale = mx.minimum(mx.array(1.0, dtype=mx.float32), max_norm_array / (grad_norm + 1e-6))
        grads = tree_map(lambda g: g * scale.astype(g.dtype), grads)
    else:
        grad_norm_clipped = grad_norm

    mx.eval(grad_norm, grad_norm_clipped)
    return grads, float(grad_norm.item()), float(grad_norm_clipped.item())


def _compute_target_logprobs(logits: mx.array, targets: mx.array) -> mx.array:
    """Gather target-token logprobs in float32 without materializing full [B, T, V] tensor."""
    logits_f32 = logits.astype(mx.float32)
    target_logits = mx.take_along_axis(
        logits_f32, targets[:, :, None].astype(mx.int32), axis=-1
    ).squeeze(-1)
    lse = mx.logsumexp(logits_f32, axis=-1)
    return target_logits - lse


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
        self._step_cache: dict[tuple, Callable] = {}

    def clear_cache(self, model_id: str | None = None) -> None:
        """Clear cached compiled steps for a specific model or all models."""
        if model_id is None:
            self._step_cache.clear()
        else:
            self._step_cache = {
                k: v for k, v in self._step_cache.items() if k[0] != model_id
            }

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
        cfg_key = (
            cfg.clip_low_threshold,
            cfg.clip_high_threshold,
            cfg.beta,
        )
        cache_key = (
            model_id,
            id(model),
            request.loss_fn,
            use_chunked,
            cfg_key,
            request.forward_only,
        )

        step_fn = self._step_cache.get(cache_key)
        if step_fn is None:
            def compute_loss(
                m: nn.Module,
                input_ids: mx.array,
                targets: mx.array,
                weights: mx.array,
                adv: mx.array,
                samp_lp: mx.array,
            ) -> tuple[mx.array, mx.array]:
                if use_chunked:
                    hidden = m.model(input_ids)
                    target_lp = chunked_target_logprobs(
                        hidden,
                        m.lm_head.weight,
                        targets,
                    )
                else:
                    logits = m(input_ids)
                    target_lp = _compute_target_logprobs(logits, targets)
                loss = loss_fn_impl(target_lp, weights, samp_lp, adv, cfg)
                return loss, target_lp

            state = [model.state]
            if request.forward_only:
                @partial(mx.compile, inputs=state, outputs=state)
                def compiled_fwd(input_ids, targets, weights, adv, samp_lp):
                    return compute_loss(model, input_ids, targets, weights, adv, samp_lp)

                step_fn = compiled_fwd
            else:
                loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

                @partial(mx.compile, inputs=state, outputs=state)
                def compiled_bwd(input_ids, targets, weights, adv, samp_lp):
                    return loss_and_grad_fn(model, input_ids, targets, weights, adv, samp_lp)

                step_fn = compiled_bwd

            self._step_cache[cache_key] = step_fn

        if request.forward_only:
            loss_val, target_lp = step_fn(
                input_tokens,
                target_tokens,
                token_weights,
                advantages,
                sampling_logprobs,
            )
            combined_grads = None
            
            # Vectorized per-sequence loss tensor
            if request.loss_fn == "cross_entropy":
                per_seq_tensor = (-target_lp * token_weights).sum(axis=-1)
            elif request.loss_fn == "importance_sampling":
                per_seq_tensor = (-(mx.exp(target_lp - sampling_logprobs) * advantages)).sum(axis=-1)
            elif request.loss_fn == "ppo":
                ratio = mx.exp(target_lp - sampling_logprobs)
                clip_low = cfg.clip_low_threshold
                clip_high = cfg.clip_high_threshold
                clipped_ratio = mx.clip(ratio, 1.0 - clip_low, 1.0 + clip_high)
                surr1 = ratio * advantages
                surr2 = clipped_ratio * advantages
                per_seq_tensor = -mx.minimum(surr1, surr2).sum(axis=-1)
            elif request.loss_fn == "cispo":
                ratio = mx.exp(target_lp - sampling_logprobs)
                clip_low = cfg.clip_low_threshold
                clip_high = cfg.clip_high_threshold
                positive_adv = advantages > 0
                clipped_ratio = mx.where(
                    positive_adv,
                    mx.clip(ratio, 1.0 - clip_high, 1.0 + clip_high),
                    mx.clip(ratio, 1.0 - clip_low, 1.0 + clip_low),
                )
                per_seq_tensor = -(mx.stop_gradient(clipped_ratio) * target_lp * advantages).sum(axis=-1)
            elif request.loss_fn == "dro":
                beta = cfg.beta
                quadratic = (target_lp - sampling_logprobs) ** 2
                obj = target_lp * advantages - 0.5 * beta * quadratic
                per_seq_tensor = -obj.sum(axis=-1)
            else:
                per_seq_tensor = None

            eval_targets = [loss_val, batch_token_count, target_lp]
            if per_seq_tensor is not None:
                eval_targets.append(per_seq_tensor)
            mx.eval(*eval_targets)
        else:
            (loss_val, target_lp), combined_grads = step_fn(
                input_tokens,
                target_tokens,
                token_weights,
                advantages,
                sampling_logprobs,
            )

            # Vectorized per-sequence loss tensor
            if request.loss_fn == "cross_entropy":
                per_seq_tensor = (-target_lp * token_weights).sum(axis=-1)
            elif request.loss_fn == "importance_sampling":
                per_seq_tensor = (-(mx.exp(target_lp - sampling_logprobs) * advantages)).sum(axis=-1)
            elif request.loss_fn == "ppo":
                ratio = mx.exp(target_lp - sampling_logprobs)
                clip_low = cfg.clip_low_threshold
                clip_high = cfg.clip_high_threshold
                clipped_ratio = mx.clip(ratio, 1.0 - clip_low, 1.0 + clip_high)
                surr1 = ratio * advantages
                surr2 = clipped_ratio * advantages
                per_seq_tensor = -mx.minimum(surr1, surr2).sum(axis=-1)
            elif request.loss_fn == "cispo":
                ratio = mx.exp(target_lp - sampling_logprobs)
                clip_low = cfg.clip_low_threshold
                clip_high = cfg.clip_high_threshold
                positive_adv = advantages > 0
                clipped_ratio = mx.where(
                    positive_adv,
                    mx.clip(ratio, 1.0 - clip_high, 1.0 + clip_high),
                    mx.clip(ratio, 1.0 - clip_low, 1.0 + clip_low),
                )
                per_seq_tensor = -(mx.stop_gradient(clipped_ratio) * target_lp * advantages).sum(axis=-1)
            elif request.loss_fn == "dro":
                beta = cfg.beta
                quadratic = (target_lp - sampling_logprobs) ** 2
                obj = target_lp * advantages - 0.5 * beta * quadratic
                per_seq_tensor = -obj.sum(axis=-1)
            else:
                per_seq_tensor = None

            eval_targets = [loss_val, batch_token_count, target_lp]
            if per_seq_tensor is not None:
                eval_targets.append(per_seq_tensor)
            mx.eval(*eval_targets)

        if not request.forward_only:
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
        
        # Batch collect python lists to reduce overhead
        target_lp_lists = target_lp.tolist()
        
        for row_idx, seq_len in enumerate(seq_lens):
            lp_list = target_lp_lists[row_idx][:seq_len]
            all_logprobs_out.append({"logprobs": TensorData(data=lp_list, dtype="float32")})

        if per_seq_tensor is not None:
            per_seq_losses = [float(v) for v in per_seq_tensor.tolist()]
        else:
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

        use_chunked = _has_split_lm_head(model)
        cache_key = (
            model_id,
            id(model),
            "forward_only",
            use_chunked,
        )

        step_fn = self._step_cache.get(cache_key)
        if step_fn is None:
            def _compute_lp(m: nn.Module, input_ids: mx.array, targets: mx.array) -> mx.array:
                if use_chunked:
                    hidden = m.model(input_ids)
                    return chunked_target_logprobs(hidden, m.lm_head.weight, targets)
                else:
                    logits = m(input_ids)
                    return _compute_target_logprobs(logits, targets)

            state = [model.state]
            @partial(mx.compile, inputs=state, outputs=state)
            def compiled_fwd(input_ids, targets):
                return _compute_lp(model, input_ids, targets)
            
            step_fn = compiled_fwd
            self._step_cache[cache_key] = step_fn

        target_lp = step_fn(input_tokens, target_tokens)
        mx.eval(target_lp)

        # Batch convert to python lists
        target_lp_lists = target_lp.tolist()
        all_logprobs = [
            target_lp_lists[row_idx][:seq_len] for row_idx, seq_len in enumerate(seq_lens)
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
