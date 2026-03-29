"""Unit tests for the training backend (forward_backward, optim_step)."""

import math

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_tinker.backend.training import TrainingBackend, _clip_grad_norm
from mlx_tinker.types import (
    AdamParams,
    ForwardBackwardInput,
    ForwardInput,
    OptimStepInput,
)
from tests.helpers import TinyModel, make_datum


class BFloat16LogitsModel(nn.Module):
    """Model that emits bf16 logits to exercise numerically sensitive paths."""

    def __init__(self, logits):
        super().__init__()
        self.base_logits = mx.array(logits, dtype=mx.float32)
        self.offset = mx.array(0.0, dtype=mx.float32)

    def __call__(self, _x: mx.array) -> mx.array:
        return (self.base_logits + self.offset).astype(mx.bfloat16)


@pytest.fixture
def model():
    m = TinyModel(vocab_size=32, dim=16)
    mx.eval(m.parameters())
    return m


@pytest.fixture
def training():
    return TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)


class TestForwardBackward:
    def test_basic(self, model, training):
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")

        result = training.forward_backward("test", model, request)

        assert result.loss_fn_output_type == "cross_entropy"
        assert len(result.loss_fn_outputs) == 1
        assert "logprobs" in result.loss_fn_outputs[0]
        assert "loss:sum" in result.metrics
        assert result.metrics["loss:sum"] > 0
        assert result.metrics["num_sequences:sum"] == 1.0

    def test_gradient_accumulation(self, model, training):
        datum = make_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])
        request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")

        training.forward_backward("test", model, request)
        assert training.grad_accum_counts["test"] == 1

        training.forward_backward("test", model, request)
        assert training.grad_accum_counts["test"] == 2

        # Grads should be accumulated
        assert training.accumulated_grads["test"] is not None

    def test_multiple_data_in_batch(self, model, training):
        d1 = make_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])
        d2 = make_datum([5, 6, 7], [6, 7, 8], [1.0, 1.0, 1.0])
        request = ForwardBackwardInput(data=[d1, d2], loss_fn="cross_entropy")

        result = training.forward_backward("test", model, request)
        assert len(result.loss_fn_outputs) == 2
        assert training.grad_accum_counts["test"] == 2

    def test_cross_entropy_uses_float32_logprob_math_for_bfloat16_logits(self, training):
        model = BFloat16LogitsModel(
            [[[0.33, -0.71, 1.42, -0.18], [1.11, -0.22, 0.07, -1.35]]]
        )
        datum = make_datum([0, 1], [2, 0], [1.0, 0.75])
        request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")

        result = training.forward_backward("bf16-ce", model, request)

        logits_bf16 = model(mx.array([[0, 1]], dtype=mx.int32))
        targets = mx.array([[2, 0]], dtype=mx.int32)
        weights = mx.array([[1.0, 0.75]], dtype=mx.float32)

        lowp_lp = mx.take_along_axis(
            logits_bf16 - mx.logsumexp(logits_bf16, axis=-1, keepdims=True),
            targets[:, :, None],
            axis=-1,
        ).squeeze(-1)
        lowp_loss = (-lowp_lp * weights).sum()

        logits_f32 = logits_bf16.astype(mx.float32)
        highp_lp = mx.take_along_axis(
            logits_f32 - mx.logsumexp(logits_f32, axis=-1, keepdims=True),
            targets[:, :, None],
            axis=-1,
        ).squeeze(-1)
        highp_loss = (-highp_lp * weights).sum()
        mx.eval(lowp_loss, highp_loss)

        assert abs(result.metrics["loss:sum"] - highp_loss.item()) < 1e-6
        assert abs(result.metrics["loss:sum"] - lowp_loss.item()) > 1e-3

    def test_importance_sampling_uses_float32_logprob_math_for_bfloat16_logits(self, training):
        model = BFloat16LogitsModel(
            [[[0.41, -0.57, 1.08, -1.19], [-0.22, 0.93, -0.44, 0.15]]]
        )
        datum = make_datum(
            [0, 1],
            [1, 3],
            [1.0, 1.0],
            advantages=[1.25, -0.5],
            logprobs=[-1.7, -0.35],
        )
        request = ForwardBackwardInput(data=[datum], loss_fn="importance_sampling")

        result = training.forward_backward("bf16-is", model, request)

        logits_bf16 = model(mx.array([[0, 1]], dtype=mx.int32))
        targets = mx.array([[1, 3]], dtype=mx.int32)
        old_lp = mx.array([[-1.7, -0.35]], dtype=mx.float32)
        adv = mx.array([[1.25, -0.5]], dtype=mx.float32)

        lowp_target_lp = mx.take_along_axis(
            logits_bf16 - mx.logsumexp(logits_bf16, axis=-1, keepdims=True),
            targets[:, :, None],
            axis=-1,
        ).squeeze(-1)
        lowp_loss = -(mx.exp(lowp_target_lp - old_lp) * adv).sum()

        logits_f32 = logits_bf16.astype(mx.float32)
        highp_target_lp = mx.take_along_axis(
            logits_f32 - mx.logsumexp(logits_f32, axis=-1, keepdims=True),
            targets[:, :, None],
            axis=-1,
        ).squeeze(-1)
        highp_loss = -(mx.exp(highp_target_lp - old_lp) * adv).sum()
        mx.eval(lowp_loss, highp_loss)

        assert abs(result.metrics["loss:sum"] - highp_loss.item()) < 1e-6
        assert abs(result.metrics["loss:sum"] - lowp_loss.item()) > 1e-3


class TestOptimStep:
    def test_basic(self, model, training):
        from mlx.utils import tree_flatten

        # Record initial weights
        initial_weights = {k: v.tolist() for k, v in tree_flatten(model.parameters())}

        # Forward backward
        datum = make_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])
        fb_request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        training.forward_backward("test", model, fb_request)

        # Optim step
        opt_request = OptimStepInput(
            adam_params=AdamParams(learning_rate=0.1, weight_decay=0.0)
        )
        result = training.optim_step("test", model, opt_request)

        assert result.metrics is not None
        assert result.metrics["grad_accum_steps:sum"] == 1

        # Weights should have changed
        new_weights = {k: v.tolist() for k, v in tree_flatten(model.parameters())}
        changed = False
        for key in initial_weights:
            if initial_weights[key] != new_weights.get(key):
                changed = True
                break
        assert changed, "Weights should change after optim_step"

    def test_clears_grads_after_step(self, model, training):
        datum = make_datum([1, 2], [2, 3], [1.0, 1.0])
        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        )

        assert training.accumulated_grads["test"] is not None

        training.optim_step(
            "test",
            model,
            OptimStepInput(adam_params=AdamParams(learning_rate=0.01)),
        )

        assert training.accumulated_grads["test"] is None
        assert training.grad_accum_counts["test"] == 0

    def test_loss_decreases(self, model, training):
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_request = OptimStepInput(
            adam_params=AdamParams(learning_rate=0.01, weight_decay=0.0)
        )

        losses = []
        for _ in range(20):
            result = training.forward_backward("test", model, fb_request)
            losses.append(result.metrics["loss:sum"])
            training.optim_step("test", model, opt_request)

        # Average of last 5 losses should be lower than average of first 5
        avg_first = sum(losses[:5]) / 5
        avg_last = sum(losses[-5:]) / 5
        assert avg_last < avg_first, (
            f"Loss should trend downward: first-5 avg={avg_first:.4f}, "
            f"last-5 avg={avg_last:.4f}, all={losses}"
        )


    def test_optim_step_no_gradients_returns_zero_steps(self, model, training):
        opt_request = OptimStepInput(
            adam_params=AdamParams(learning_rate=0.01)
        )
        result = training.optim_step("fresh_model", model, opt_request)
        assert result.metrics is not None
        assert result.metrics["grad_accum_steps:sum"] == 0

    def test_gradient_clipping(self, model, training):
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        training.forward_backward("test", model, fb_request)

        # Apply with gradient clipping
        opt_request = OptimStepInput(
            adam_params=AdamParams(learning_rate=0.01, grad_clip_norm=0.1)
        )
        result = training.optim_step("test", model, opt_request)
        assert result.metrics is not None
        assert result.metrics["grad_accum_steps:sum"] == 1


class TestForward:
    def test_returns_logprobs(self, model, training):
        datum = make_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])
        request = ForwardInput(data=[datum])

        result = training.forward("test", model, request)
        assert len(result.logprobs) == 1
        assert len(result.logprobs[0]) == 3
        # Log probs should be negative
        assert all(lp <= 0 for lp in result.logprobs[0])

    def test_forward_uses_float32_logprob_math_for_bfloat16_logits(self, training):
        model = BFloat16LogitsModel(
            [[[0.19, -0.88, 1.21, -0.27], [0.73, -0.31, -0.14, 0.51]]]
        )
        datum = make_datum([0, 1], [2, 3], [1.0, 1.0])
        result = training.forward("bf16-forward", model, ForwardInput(data=[datum]))

        logits_bf16 = model(mx.array([[0, 1]], dtype=mx.int32))
        targets = mx.array([[2, 3]], dtype=mx.int32)
        logits_f32 = logits_bf16.astype(mx.float32)
        expected = mx.take_along_axis(
            logits_f32 - mx.logsumexp(logits_f32, axis=-1, keepdims=True),
            targets[:, :, None],
            axis=-1,
        ).squeeze(-1)
        mx.eval(expected)

        assert result.logprobs[0] == pytest.approx(expected[0].tolist(), abs=1e-6)


# ---------------------------------------------------------------------------
# Gradient correctness (Tier 2)
# ---------------------------------------------------------------------------


class TestGradientCorrectness:
    def test_gradient_norm_reasonable(self, model, training):
        """Global gradient norm should be finite and within a reasonable range."""
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        )
        grads = training.accumulated_grads["test"]
        flat = tree_flatten(grads)
        norm_sq = sum(mx.sum(mx.square(g)).item() for _, g in flat)
        norm = norm_sq**0.5
        assert math.isfinite(norm), f"Gradient norm not finite: {norm}"
        assert 0.001 < norm < 100.0, f"Gradient norm {norm} outside [0.001, 100]"

    def test_gradient_accumulation_is_additive(self, model, training):
        """2x forward_backward ≈ 2x single-step gradients."""
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_req = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")

        # Single step
        training.forward_backward("single", model, fb_req)
        single_grads = {
            k: v.tolist() for k, v in tree_flatten(training.accumulated_grads["single"])
        }

        # Double step (accumulated)
        training.forward_backward("double", model, fb_req)
        training.forward_backward("double", model, fb_req)
        double_grads = {
            k: v.tolist() for k, v in tree_flatten(training.accumulated_grads["double"])
        }

        for key in single_grads:
            s = single_grads[key]
            d = double_grads[key]
            # Flatten for comparison
            for sv, dv in zip(
                [x for row in s for x in (row if isinstance(row, list) else [row])],
                [x for row in d for x in (row if isinstance(row, list) else [row])],
            ):
                if abs(sv) > 1e-8:
                    ratio = dv / sv
                    assert 1.8 < ratio < 2.2, (
                        f"Accumulated grad should be ~2x single: {key} ratio={ratio}"
                    )

    def test_grad_clip_norm_limits_norm(self):
        """Post-clip gradient norm should be <= max_norm."""
        from mlx.utils import tree_map

        grads = {"a": mx.array([10.0, 20.0, 30.0]), "b": mx.array([5.0, 15.0])}
        clipped, pre_norm, post_norm = _clip_grad_norm(grads, max_norm=1.0)

        assert pre_norm > 1.0, "Pre-clip norm should exceed max_norm"
        assert post_norm <= 1.0 + 1e-6, f"Post-clip norm {post_norm} > max_norm"


# ---------------------------------------------------------------------------
# NaN/Inf guards (Tier 2)
# ---------------------------------------------------------------------------


class TestNaNInfGuards:
    def test_loss_is_finite_every_step(self, model, training):
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_req = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_req = OptimStepInput(adam_params=AdamParams(learning_rate=0.01, weight_decay=0.0))

        for step in range(20):
            result = training.forward_backward("test", model, fb_req)
            loss = result.metrics["loss:sum"]
            assert math.isfinite(loss), f"Non-finite loss at step {step}: {loss}"
            training.optim_step("test", model, opt_req)

    def test_gradients_have_no_nan(self, model, training):
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        )
        grads = training.accumulated_grads["test"]
        for name, g in tree_flatten(grads):
            mx.eval(g)
            assert not mx.any(mx.isnan(g)).item(), f"NaN in gradient {name}"

    def test_gradients_have_no_inf(self, model, training):
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        )
        grads = training.accumulated_grads["test"]
        for name, g in tree_flatten(grads):
            mx.eval(g)
            assert not mx.any(mx.isinf(g)).item(), f"Inf in gradient {name}"

    def test_extreme_input_tokens_no_nan(self, model, training):
        """Token IDs at boundaries (0 and vocab_size-1) should not cause NaN."""
        datum = make_datum([0, 31, 0, 31], [31, 0, 31, 0], [1.0, 1.0, 1.0, 1.0])
        result = training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        )
        loss = result.metrics["loss:sum"]
        assert math.isfinite(loss), f"NaN/Inf with boundary tokens: {loss}"

    def test_optim_step_skips_on_nan_gradients(self, model, training):
        """optim_step should skip update and reset state when gradients contain NaN."""
        from mlx.utils import tree_flatten, tree_map

        datum = make_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])
        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        )
        # Capture weights before
        weights_before = [(k, v.tolist()) for k, v in tree_flatten(model.parameters())]

        # Inject NaN into accumulated gradients
        training.accumulated_grads["test"] = tree_map(
            lambda g: g * mx.array(float("nan")), training.accumulated_grads["test"]
        )

        result = training.optim_step(
            "test", model, OptimStepInput(adam_params=AdamParams(learning_rate=0.01))
        )
        assert result.metrics.get("skipped:sum") == 1.0
        assert training.accumulated_grads["test"] is None
        assert training.grad_accum_counts["test"] == 0

        # Verify weights unchanged
        weights_after = [(k, v.tolist()) for k, v in tree_flatten(model.parameters())]
        assert weights_before == weights_after


# ---------------------------------------------------------------------------
# Gradient norm tracking (Phase 8)
# ---------------------------------------------------------------------------


class TestGradientNormTracking:
    def test_grad_norm_in_optim_metrics(self, model, training):
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        )
        result = training.optim_step(
            "test", model, OptimStepInput(adam_params=AdamParams(learning_rate=0.01))
        )
        assert "grad_norm:mean" in result.metrics
        assert isinstance(result.metrics["grad_norm:mean"], float)
        assert result.metrics["grad_norm:mean"] > 0
        assert math.isfinite(result.metrics["grad_norm:mean"])

    def test_grad_norm_decreases_with_convergence(self, model, training):
        """Gradient norms should decrease as loss converges."""
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_req = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_req = OptimStepInput(adam_params=AdamParams(learning_rate=0.01, weight_decay=0.0))

        norms = []
        for _ in range(20):
            training.forward_backward("test", model, fb_req)
            result = training.optim_step("test", model, opt_req)
            norms.append(result.metrics["grad_norm:mean"])

        avg_first = sum(norms[:5]) / 5
        avg_last = sum(norms[-5:]) / 5
        assert avg_last < avg_first, (
            f"Grad norm should decrease: first-5={avg_first:.4f} last-5={avg_last:.4f}"
        )

    def test_clipped_grad_norm_bounded(self, model, training):
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        )
        result = training.optim_step(
            "test", model,
            OptimStepInput(adam_params=AdamParams(learning_rate=0.01, grad_clip_norm=0.1)),
        )
        assert result.metrics["grad_norm_clipped:mean"] <= 0.1 + 1e-6, (
            f"Clipped norm {result.metrics['grad_norm_clipped:mean']} > 0.1"
        )
