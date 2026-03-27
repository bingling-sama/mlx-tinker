"""Unit tests for MLX loss functions."""

import mlx.core as mx
import pytest

from mlx_tinker.backend.loss_fns import (
    LOSS_FUNCTION_MAP,
    LossFnConfig,
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

    def test_callable(self, cfg):
        for name, fn in LOSS_FUNCTION_MAP.items():
            lp = mx.full((1, 2), -1.0)
            mask = mx.ones((1, 2))
            adv = mx.ones((1, 2))
            result = fn(lp, mask, lp, adv, cfg)
            mx.eval(result)
            assert result.ndim == 0, f"{name} should return scalar"
