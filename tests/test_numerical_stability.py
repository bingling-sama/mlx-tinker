"""Numerical stability edge-case tests.

Tests extreme inputs that could cause NaN/Inf in production.
Cross-cuts loss functions, training loop, and inference.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_tinker.backend.loss_fns import (
    LossFnConfig,
    chunked_cross_entropy_loss,
    cispo_loss,
    cross_entropy_loss,
    importance_sampling_loss,
    ppo_loss,
)
from mlx_tinker.backend.training import TrainingBackend
from mlx_tinker.backend.inference import InferenceBackend
from mlx_tinker.types import (
    AdamParams,
    ForwardBackwardInput,
    ModelInput,
    EncodedTextChunk,
    OptimStepInput,
    SampleInput,
    SamplingParams,
)
from tests.helpers import TinyModel, TinyModelWithCache, FakeTokenizer, make_datum


class TestLossFunctionStability:
    """Extreme inputs to loss functions should not produce NaN/Inf."""

    def test_extreme_negative_logprobs(self):
        """Very negative log-probs should give large but finite loss."""
        cfg = LossFnConfig()
        target_lp = mx.array([[-1000.0, -500.0]])
        mask = mx.ones((1, 2))
        dummy = mx.zeros((1, 2))

        loss = cross_entropy_loss(target_lp, mask, dummy, dummy, cfg)
        mx.eval(loss)
        assert math.isfinite(loss.item()), f"Loss is not finite: {loss.item()}"
        assert loss.item() > 0

    def test_zero_logprobs(self):
        """Log-prob of 0.0 means probability=1. CE loss should be 0."""
        cfg = LossFnConfig()
        target_lp = mx.array([[0.0, 0.0]])
        mask = mx.ones((1, 2))
        dummy = mx.zeros((1, 2))

        loss = cross_entropy_loss(target_lp, mask, dummy, dummy, cfg)
        mx.eval(loss)
        assert abs(loss.item()) < 1e-6

    def test_large_importance_ratio(self):
        """exp(10) ≈ 22026 — IS loss should still be finite."""
        cfg = LossFnConfig()
        new_lp = mx.array([[0.0]])
        old_lp = mx.array([[-10.0]])
        mask = mx.ones((1, 1))
        advantages = mx.array([[1.0]])

        loss = importance_sampling_loss(new_lp, mask, old_lp, advantages, cfg)
        mx.eval(loss)
        assert math.isfinite(loss.item()), f"IS loss not finite with ratio exp(10): {loss.item()}"

    def test_very_small_importance_ratio(self):
        """exp(-20) ≈ 2e-9 — should not underflow to exactly 0."""
        cfg = LossFnConfig()
        new_lp = mx.array([[-20.0]])
        old_lp = mx.array([[0.0]])
        mask = mx.ones((1, 1))
        advantages = mx.array([[1.0]])

        loss = importance_sampling_loss(new_lp, mask, old_lp, advantages, cfg)
        mx.eval(loss)
        assert math.isfinite(loss.item())
        # exp(-20) * 1.0 ≈ 2e-9, loss = -2e-9 ≈ 0 but not exactly 0
        assert loss.item() != 0.0

    def test_ppo_with_extreme_ratio(self):
        """PPO with exp(5) ≈ 148 ratio — should clip and stay finite."""
        cfg = LossFnConfig(clip_high_threshold=0.2)
        new_lp = mx.array([[0.0]])
        old_lp = mx.array([[-5.0]])
        mask = mx.ones((1, 1))
        advantages = mx.array([[1.0]])

        loss = ppo_loss(new_lp, mask, old_lp, advantages, cfg)
        mx.eval(loss)
        assert math.isfinite(loss.item())
        # Should be clipped to 1.2
        assert abs(loss.item() - (-1.2)) < 1e-4

    def test_cispo_with_extreme_ratio_both_signs(self):
        cfg = LossFnConfig(clip_low_threshold=0.1, clip_high_threshold=0.2)
        new_lp = mx.array([[0.0, 0.0]])
        old_lp = mx.array([[-5.0, -5.0]])
        mask = mx.ones((1, 2))
        advantages = mx.array([[1.0, -1.0]])

        loss = cispo_loss(new_lp, mask, old_lp, advantages, cfg)
        mx.eval(loss)
        assert math.isfinite(loss.item())


class TestChunkedCEStability:
    """Chunked cross-entropy with edge-case hidden states."""

    def test_large_logit_range(self):
        """Hidden states producing logits in [-100, +100] should be stable."""
        mx.random.seed(42)
        vocab_size, dim = 32, 8
        lm_head = nn.Linear(dim, vocab_size, bias=False)
        mx.eval(lm_head.parameters())

        # Scale hidden states to produce large logits
        hidden = mx.random.normal((1, 4, dim)) * 10.0
        targets = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        mask = mx.ones((1, 4))
        mx.eval(hidden)

        loss = chunked_cross_entropy_loss(hidden, lm_head.weight, targets, mask)
        mx.eval(loss)
        assert math.isfinite(loss.item()), f"Loss not finite with large logits: {loss.item()}"

    def test_uniform_hidden_states(self):
        """All-same hidden states → loss should be approximately log(V)."""
        vocab_size, dim = 32, 8
        weight = mx.random.normal((vocab_size, dim))
        mx.eval(weight)

        # All hidden states identical → all positions have same logit distribution
        hidden = mx.ones((1, 4, dim)) * 0.1
        targets = mx.array([[0, 0, 0, 0]], dtype=mx.int32)
        mask = mx.ones((1, 4))

        loss = chunked_cross_entropy_loss(hidden, weight, targets, mask)
        mx.eval(loss)
        assert math.isfinite(loss.item())
        # For uniform logits, CE ≈ log(V), but logits aren't uniform (weight varies)
        # Just check it's positive and reasonable
        assert loss.item() > 0

    def test_tiny_chunk_size_stability(self):
        """CE_CHUNK_SIZE=2 (very small) should still match standard CE."""
        import mlx_tinker.backend.loss_fns as lf

        old_chunk = lf.CE_CHUNK_SIZE
        lf.CE_CHUNK_SIZE = 2

        try:
            mx.random.seed(0)
            vocab_size, dim = 16, 8
            weight = mx.random.normal((vocab_size, dim))
            hidden = mx.random.normal((1, 3, dim))
            targets = mx.array([[1, 2, 3]], dtype=mx.int32)
            mask = mx.ones((1, 3))
            mx.eval(weight, hidden)

            # Standard CE
            logits = hidden @ weight.T
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            target_lp = mx.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)
            standard = (-target_lp * mask).sum()
            mx.eval(standard)

            chunked = chunked_cross_entropy_loss(hidden, weight, targets, mask)
            mx.eval(chunked)

            assert abs(standard.item() - chunked.item()) < 1e-4, (
                f"Chunk=2 differs: standard={standard.item():.6f} chunked={chunked.item():.6f}"
            )
        finally:
            lf.CE_CHUNK_SIZE = old_chunk


class TestTrainingLoopStability:
    """Training loop should remain numerically stable under various conditions."""

    def test_normal_training_all_finite(self):
        """50 steps of normal training should produce all-finite losses."""
        mx.random.seed(42)
        model = TinyModel(vocab_size=32, dim=16)
        mx.eval(model.parameters())
        training = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)

        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_req = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_req = OptimStepInput(adam_params=AdamParams(learning_rate=0.01, weight_decay=0.0))

        for step in range(50):
            result = training.forward_backward("test", model, fb_req)
            loss = result.metrics["loss:sum"]
            assert math.isfinite(loss), f"Non-finite loss at step {step}: {loss}"
            training.optim_step("test", model, opt_req)

    def test_aggressive_lr_stays_finite(self):
        """LR=0.1 may diverge but should not produce NaN/Inf."""
        mx.random.seed(42)
        model = TinyModel(vocab_size=32, dim=16)
        mx.eval(model.parameters())
        training = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)

        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_req = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_req = OptimStepInput(adam_params=AdamParams(learning_rate=0.1, weight_decay=0.0))

        for step in range(20):
            result = training.forward_backward("test", model, fb_req)
            loss = result.metrics["loss:sum"]
            assert math.isfinite(loss), f"Non-finite loss at step {step} with LR=0.1: {loss}"
            training.optim_step("test", model, opt_req)

    def test_zero_lr_no_parameter_change(self):
        """LR=0 should produce no parameter changes."""
        from mlx.utils import tree_flatten

        mx.random.seed(42)
        model = TinyModel(vocab_size=32, dim=16)
        mx.eval(model.parameters())
        initial_params = {k: v.tolist() for k, v in tree_flatten(model.parameters())}

        training = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])

        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        )
        training.optim_step(
            "test", model,
            OptimStepInput(adam_params=AdamParams(learning_rate=0.0, weight_decay=0.0)),
        )

        new_params = {k: v.tolist() for k, v in tree_flatten(model.parameters())}
        for key in initial_params:
            assert initial_params[key] == new_params[key], f"Param {key} changed with LR=0"


class TestInferenceStability:
    """Inference sampling should remain stable under extreme parameters."""

    def _make_model_and_tokenizer(self):
        model = TinyModelWithCache(vocab_size=32, dim=16)
        mx.eval(model.parameters())
        return model, FakeTokenizer()

    def test_temperature_very_high(self):
        """Temperature=100.0 should produce valid (finite) logprobs."""
        model, tokenizer = self._make_model_and_tokenizer()
        inference = InferenceBackend()

        result = inference.sample(
            model, tokenizer,
            SampleInput(
                prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
                sampling_params=SamplingParams(temperature=100.0, max_tokens=5),
                num_samples=1,
            ),
        )
        assert len(result.sequences) == 1
        for lp in result.sequences[0].logprobs:
            assert math.isfinite(lp), f"Non-finite logprob with temp=100: {lp}"

    def test_temperature_very_low(self):
        """Temperature=0.001 should produce finite logprobs and valid tokens."""
        model, tokenizer = self._make_model_and_tokenizer()
        inference = InferenceBackend()

        result = inference.sample(
            model, tokenizer,
            SampleInput(
                prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
                sampling_params=SamplingParams(temperature=0.001, max_tokens=5),
                num_samples=1,
            ),
        )
        assert len(result.sequences) == 1
        for lp in result.sequences[0].logprobs:
            assert math.isfinite(lp), f"Non-finite logprob with temp=0.001: {lp}"
