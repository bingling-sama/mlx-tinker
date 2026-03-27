"""Unit tests for MLX loss functions."""

import mlx.core as mx
import pytest

from mlx_tinker.backend.loss_fns import (
    LOSS_FUNCTION_MAP,
    LossFnConfig,
    chunked_cross_entropy_loss,
    cispo_loss,
    cross_entropy_loss,
    importance_sampling_loss,
    ppo_loss,
)


@pytest.fixture
def cfg():
    return LossFnConfig()


@pytest.fixture
def ppo_cfg():
    return LossFnConfig(clip_high_threshold=0.2)


@pytest.fixture
def cispo_cfg():
    return LossFnConfig(clip_low_threshold=0.1, clip_high_threshold=0.2)


class TestCrossEntropyLoss:
    def test_basic(self, cfg):
        # log probs of -1.0 with full mask => loss = 1.0
        target_lp = mx.full((2, 4), -1.0)
        mask = mx.ones((2, 4))
        dummy = mx.zeros((2, 4))

        loss = cross_entropy_loss(target_lp, mask, dummy, dummy, cfg)
        mx.eval(loss)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_masked_positions_ignored(self, cfg):
        # Only positions with mask=1 contribute
        target_lp = mx.array([[-2.0, -1.0, -3.0, -0.5]])
        mask = mx.array([[0.0, 1.0, 0.0, 1.0]])
        dummy = mx.zeros_like(target_lp)

        loss = cross_entropy_loss(target_lp, mask, dummy, dummy, cfg)
        mx.eval(loss)
        expected = (1.0 + 0.5) / 2.0  # mean of unmasked
        assert abs(loss.item() - expected) < 1e-5

    def test_all_zero_mask(self, cfg):
        target_lp = mx.full((1, 4), -1.0)
        mask = mx.zeros((1, 4))
        dummy = mx.zeros((1, 4))

        loss = cross_entropy_loss(target_lp, mask, dummy, dummy, cfg)
        mx.eval(loss)
        assert loss.item() == 0.0

    def test_gradient_flows(self, cfg):
        target_lp = mx.array([[-1.0, -2.0]], dtype=mx.float32)
        mask = mx.ones((1, 2))
        dummy = mx.zeros((1, 2))

        grad_fn = mx.grad(lambda lp: cross_entropy_loss(lp, mask, dummy, dummy, cfg))
        grads = grad_fn(target_lp)
        mx.eval(grads)
        # Gradient of -mean(lp * mask) w.r.t. lp is -mask/sum(mask)
        assert grads.shape == (1, 2)


class TestImportanceSamplingLoss:
    def test_on_policy(self, cfg):
        # When new_lp == old_lp, ratio=1, loss = -mean(advantage * mask)
        lp = mx.full((1, 3), -1.0)
        mask = mx.ones((1, 3))
        advantages = mx.array([[1.0, 2.0, 3.0]])

        loss = importance_sampling_loss(lp, mask, lp, advantages, cfg)
        mx.eval(loss)
        expected = -(1.0 + 2.0 + 3.0) / 3.0
        assert abs(loss.item() - expected) < 1e-5

    def test_off_policy(self, cfg):
        new_lp = mx.array([[-0.5, -0.5]])
        old_lp = mx.array([[-1.0, -1.0]])
        mask = mx.ones((1, 2))
        advantages = mx.array([[1.0, 1.0]])

        loss = importance_sampling_loss(new_lp, mask, old_lp, advantages, cfg)
        mx.eval(loss)
        # ratio = exp(-0.5 - (-1.0)) = exp(0.5) ≈ 1.6487
        import math

        expected = -math.exp(0.5)
        assert abs(loss.item() - expected) < 1e-4


class TestPPOLoss:
    def test_no_clipping_on_policy(self, ppo_cfg):
        lp = mx.full((1, 2), -1.0)
        mask = mx.ones((1, 2))
        advantages = mx.array([[1.0, 1.0]])

        loss = ppo_loss(lp, mask, lp, advantages, ppo_cfg)
        mx.eval(loss)
        # ratio=1, clipped_ratio=1, min(1*1, 1*1) = 1, loss = -1.0
        assert abs(loss.item() - (-1.0)) < 1e-5

    def test_clipping_active(self, ppo_cfg):
        # Large ratio should be clipped
        new_lp = mx.array([[0.0]])
        old_lp = mx.array([[-2.0]])
        mask = mx.ones((1, 1))
        advantages = mx.array([[1.0]])

        loss = ppo_loss(new_lp, mask, old_lp, advantages, ppo_cfg)
        mx.eval(loss)
        # ratio = exp(2.0) ≈ 7.39, clipped to 1.2
        # loss = -min(7.39, 1.2) = -1.2
        assert abs(loss.item() - (-1.2)) < 1e-4


class TestCISPOLoss:
    def test_positive_advantage_clipping(self, cispo_cfg):
        new_lp = mx.array([[0.0]])
        old_lp = mx.array([[-2.0]])
        mask = mx.ones((1, 1))
        advantages = mx.array([[1.0]])  # positive

        loss = cispo_loss(new_lp, mask, old_lp, advantages, cispo_cfg)
        mx.eval(loss)
        # ratio ≈ 7.39, positive adv → clip to [0.8, 1.2], clipped_ratio=1.2
        # surr1 = 7.39*1 = 7.39, surr2 = 1.2*1 = 1.2
        # loss = -min(7.39, 1.2) = -1.2
        assert abs(loss.item() - (-1.2)) < 1e-4


class TestLossFunctionMap:
    def test_all_registered(self):
        assert "cross_entropy" in LOSS_FUNCTION_MAP
        assert "importance_sampling" in LOSS_FUNCTION_MAP
        assert "ppo" in LOSS_FUNCTION_MAP
        assert "cispo" in LOSS_FUNCTION_MAP

    def test_all_return_scalar_with_correct_sign(self, cfg):
        """Verify all loss functions return a scalar and produce expected values."""
        for name, fn in LOSS_FUNCTION_MAP.items():
            lp = mx.full((1, 2), -1.0)
            mask = mx.ones((1, 2))
            adv = mx.ones((1, 2))
            result = fn(lp, mask, lp, adv, cfg)
            mx.eval(result)
            assert result.ndim == 0, f"{name} should return scalar"
            assert result.dtype == mx.float32, f"{name} should return float32"

        # Cross-entropy with log-prob=-1 and full mask should give loss=1.0
        ce = cross_entropy_loss(mx.full((1, 2), -1.0), mx.ones((1, 2)), mx.zeros((1, 2)), mx.zeros((1, 2)), cfg)
        mx.eval(ce)
        assert abs(ce.item() - 1.0) < 1e-5, f"CE loss should be 1.0, got {ce.item()}"

        # On-policy IS loss (ratio=1) should equal -mean(advantages)
        lp = mx.full((1, 2), -1.0)
        adv = mx.array([[2.0, 4.0]])
        is_loss = importance_sampling_loss(lp, mx.ones((1, 2)), lp, adv, cfg)
        mx.eval(is_loss)
        assert abs(is_loss.item() - (-3.0)) < 1e-5, f"IS loss should be -3.0, got {is_loss.item()}"

    def test_all_zero_mask_returns_zero(self, cfg):
        """All loss functions should return 0 when mask is all zeros."""
        for name, fn in LOSS_FUNCTION_MAP.items():
            lp = mx.full((1, 3), -1.0)
            zero_mask = mx.zeros((1, 3))
            adv = mx.ones((1, 3))
            result = fn(lp, zero_mask, lp, adv, cfg)
            mx.eval(result)
            assert result.item() == 0.0, f"{name} should return 0 with zero mask"


class TestPPONegativeAdvantages:
    def test_negative_advantage_clips_correctly(self):
        """With negative advantages, PPO should clip the upper bound."""
        ppo_cfg = LossFnConfig(clip_high_threshold=0.2)
        # ratio ≈ 7.39 (large), negative advantage
        new_lp = mx.array([[0.0]])
        old_lp = mx.array([[-2.0]])
        mask = mx.ones((1, 1))
        advantages = mx.array([[-1.0]])

        loss = ppo_loss(new_lp, mask, old_lp, advantages, ppo_cfg)
        mx.eval(loss)
        # ratio=7.39, adv=-1: surr1=7.39*(-1)=-7.39, clipped=1.2*(-1)=-1.2
        # loss = -min(-7.39, -1.2) = -(-7.39) = 7.39
        import math

        expected = math.exp(2.0)
        assert abs(loss.item() - expected) < 1e-3


class TestCISPONegativeAdvantages:
    def test_negative_advantage_uses_clip_low(self):
        """With negative advantages, CISPO should use clip_low_threshold."""
        cispo_cfg = LossFnConfig(clip_low_threshold=0.1, clip_high_threshold=0.2)
        new_lp = mx.array([[0.0]])
        old_lp = mx.array([[-2.0]])
        mask = mx.ones((1, 1))
        advantages = mx.array([[-1.0]])  # negative

        loss = cispo_loss(new_lp, mask, old_lp, advantages, cispo_cfg)
        mx.eval(loss)
        # ratio ≈ 7.39, negative adv → clip to [0.9, 1.1]
        # clipped_ratio = 1.1, surr1 = 7.39*(-1) = -7.39, surr2 = 1.1*(-1) = -1.1
        # loss = -min(-7.39, -1.1) = 7.39
        import math

        expected = math.exp(2.0)
        assert abs(loss.item() - expected) < 1e-3


class TestChunkedCrossEntropy:
    def test_matches_standard_ce(self):
        """Chunked CE should produce the same result as standard CE."""
        import mlx.nn as nn

        mx.random.seed(42)
        vocab_size = 64
        dim = 16
        seq_len = 8

        lm_head = nn.Linear(dim, vocab_size, bias=False)
        mx.eval(lm_head.parameters())

        hidden = mx.random.normal((1, seq_len, dim))
        targets = mx.array([[2, 5, 10, 3, 7, 1, 4, 9]], dtype=mx.int32)
        mask = mx.array([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        mx.eval(hidden)

        # Standard CE
        logits = hidden @ lm_head.weight.T
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        target_lp = mx.take_along_axis(
            log_probs, targets[:, :, None], axis=-1
        ).squeeze(-1)
        cfg = LossFnConfig()
        standard_loss = cross_entropy_loss(
            target_lp, mask, mx.zeros_like(mask), mx.zeros_like(mask), cfg
        )
        mx.eval(standard_loss)

        # Chunked CE
        chunked_loss = chunked_cross_entropy_loss(
            hidden, lm_head.weight, targets, mask
        )
        mx.eval(chunked_loss)

        assert abs(standard_loss.item() - chunked_loss.item()) < 1e-5, (
            f"Standard CE {standard_loss.item():.6f} != "
            f"Chunked CE {chunked_loss.item():.6f}"
        )

    def test_small_chunk_size(self):
        """Chunked CE should work correctly with very small chunk sizes."""
        import mlx.nn as nn

        import mlx_tinker.backend.loss_fns as lf

        old_chunk = lf.CE_CHUNK_SIZE
        lf.CE_CHUNK_SIZE = 8  # Very small chunks

        try:
            mx.random.seed(0)
            vocab_size = 32
            dim = 8
            lm_head = nn.Linear(dim, vocab_size, bias=False)
            mx.eval(lm_head.parameters())

            hidden = mx.random.normal((1, 4, dim))
            targets = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
            mask = mx.ones((1, 4))
            mx.eval(hidden)

            # Standard CE
            logits = hidden @ lm_head.weight.T
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            target_lp = mx.take_along_axis(
                log_probs, targets[:, :, None], axis=-1
            ).squeeze(-1)
            standard = cross_entropy_loss(
                target_lp, mask, mx.zeros_like(mask), mx.zeros_like(mask),
                LossFnConfig(),
            )
            mx.eval(standard)

            chunked = chunked_cross_entropy_loss(
                hidden, lm_head.weight, targets, mask
            )
            mx.eval(chunked)

            assert abs(standard.item() - chunked.item()) < 1e-4
        finally:
            lf.CE_CHUNK_SIZE = old_chunk

    def test_gradient_flows(self):
        """Gradients should flow through chunked CE."""
        import mlx.nn as nn

        mx.random.seed(42)
        vocab_size = 32
        dim = 8
        lm_head = nn.Linear(dim, vocab_size, bias=False)
        mx.eval(lm_head.parameters())

        hidden = mx.random.normal((1, 4, dim))
        targets = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        mask = mx.ones((1, 4))
        mx.eval(hidden)

        def loss_fn(h):
            return chunked_cross_entropy_loss(
                h, lm_head.weight, targets, mask
            )

        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(hidden)
        mx.eval(grads)

        assert grads.shape == hidden.shape
        assert mx.any(grads != 0).item(), "Gradients should be non-zero"
