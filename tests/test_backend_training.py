"""Unit tests for the training backend (forward_backward, optim_step)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_tinker.backend.training import TrainingBackend
from mlx_tinker.types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardInput,
    ForwardInput,
    LossFnInputs,
    ModelInput,
    OptimStepInput,
    TensorData,
)


class TinyModel(nn.Module):
    """Minimal model for testing: embedding + linear head."""

    def __init__(self, vocab_size: int = 32, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embed(x)
        return self.head(h)


@pytest.fixture
def model():
    m = TinyModel(vocab_size=32, dim=16)
    mx.eval(m.parameters())
    return m


@pytest.fixture
def training():
    return TrainingBackend()


def _make_datum(tokens: list[int], targets: list[int], weights: list[float]) -> Datum:
    """Helper to create a Datum for testing."""
    return Datum(
        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=tokens)]),
        loss_fn_inputs=LossFnInputs(
            target_tokens=TensorData(data=targets),
            weights=TensorData(data=weights),
            advantages=TensorData(data=[0.0] * len(targets)),
            logprobs=TensorData(data=[0.0] * len(targets)),
        ),
    )


class TestForwardBackward:
    def test_basic(self, model, training):
        datum = _make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")

        result = training.forward_backward("test", model, request)

        assert result.loss_fn_output_type == "cross_entropy"
        assert len(result.loss_fn_outputs) == 1
        assert "loss" in result.loss_fn_outputs[0]
        assert result.loss_fn_outputs[0]["loss"] > 0
        assert result.metrics["num_sequences"] == 1

    def test_gradient_accumulation(self, model, training):
        datum = _make_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])
        request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")

        training.forward_backward("test", model, request)
        assert training.grad_accum_counts["test"] == 1

        training.forward_backward("test", model, request)
        assert training.grad_accum_counts["test"] == 2

        # Grads should be accumulated
        assert training.accumulated_grads["test"] is not None

    def test_multiple_data_in_batch(self, model, training):
        d1 = _make_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])
        d2 = _make_datum([5, 6, 7], [6, 7, 8], [1.0, 1.0, 1.0])
        request = ForwardBackwardInput(data=[d1, d2], loss_fn="cross_entropy")

        result = training.forward_backward("test", model, request)
        assert len(result.loss_fn_outputs) == 2
        assert training.grad_accum_counts["test"] == 2


class TestOptimStep:
    def test_basic(self, model, training):
        from mlx.utils import tree_flatten

        # Record initial weights
        initial_weights = {k: v.tolist() for k, v in tree_flatten(model.parameters())}

        # Forward backward
        datum = _make_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])
        fb_request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        training.forward_backward("test", model, fb_request)

        # Optim step
        opt_request = OptimStepInput(
            adam_params=AdamParams(learning_rate=0.1, weight_decay=0.0)
        )
        result = training.optim_step("test", model, opt_request)

        assert result.metrics is not None
        assert result.metrics["grad_accum_steps"] == 1

        # Weights should have changed
        new_weights = {k: v.tolist() for k, v in tree_flatten(model.parameters())}
        changed = False
        for key in initial_weights:
            if initial_weights[key] != new_weights.get(key):
                changed = True
                break
        assert changed, "Weights should change after optim_step"

    def test_clears_grads_after_step(self, model, training):
        datum = _make_datum([1, 2], [2, 3], [1.0, 1.0])
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
        datum = _make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_request = OptimStepInput(
            adam_params=AdamParams(learning_rate=0.01, weight_decay=0.0)
        )

        losses = []
        for _ in range(5):
            result = training.forward_backward("test", model, fb_request)
            losses.append(result.loss_fn_outputs[0]["loss"])
            training.optim_step("test", model, opt_request)

        # Loss should generally decrease over steps
        assert losses[-1] < losses[0], f"Loss should decrease: {losses}"


class TestForward:
    def test_returns_logprobs(self, model, training):
        datum = _make_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])
        request = ForwardInput(data=[datum])

        result = training.forward("test", model, request)
        assert len(result.logprobs) == 1
        assert len(result.logprobs[0]) == 3
        # Log probs should be negative
        assert all(lp <= 0 for lp in result.logprobs[0])
