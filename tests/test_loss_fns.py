"""Unit tests for MLX loss functions."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_tinker.backend.loss_fns import (
    LOSS_FUNCTION_MAP,
    LossFnConfig,
    chunked_cross_entropy_loss,
    cispo_loss,
    cross_entropy_loss,
    dro_loss,
    importance_sampling_loss,
    ppo_loss,
)
from tests.helpers import finite_difference_check, finite_difference_check_argnum


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
        # log probs of -1.0 with full mask => loss = (-(-1.0)*1.0) * 8 = 8.0 (sum reduction)
        target_lp = mx.full((2, 4), -1.0)
        mask = mx.ones((2, 4))
        dummy = mx.zeros((2, 4))

        loss = cross_entropy_loss(target_lp, mask, dummy, dummy, cfg)
        mx.eval(loss)
        assert abs(loss.item() - 8.0) < 1e-5

    def test_masked_positions_ignored(self, cfg):
        # Only positions with mask=1 contribute
        target_lp = mx.array([[-2.0, -1.0, -3.0, -0.5]])
        mask = mx.array([[0.0, 1.0, 0.0, 1.0]])
        dummy = mx.zeros_like(target_lp)

        loss = cross_entropy_loss(target_lp, mask, dummy, dummy, cfg)
        mx.eval(loss)
        expected = 1.0 + 0.5  # sum of unmasked: -(-1.0)*1 + -(-0.5)*1
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
        # Gradient of -sum(lp * mask) w.r.t. lp is -mask
        assert grads.shape == (1, 2)


class TestChunkedCrossEntropyPrecision:
    def test_bfloat16_inputs_accumulate_in_float32(self):
        rng = np.random.RandomState(0)
        hidden_np = rng.normal(size=(1, 3, 8)).astype(np.float32)
        weight_np = rng.normal(size=(16, 8)).astype(np.float32)
        targets_np = np.array([[1, 7, 3]], dtype=np.int32)
        mask_np = np.array([[1.0, 0.5, 1.0]], dtype=np.float32)

        hidden_bf16 = mx.array(hidden_np, dtype=mx.bfloat16)
        weight_bf16 = mx.array(weight_np, dtype=mx.bfloat16)
        targets = mx.array(targets_np, dtype=mx.int32)
        mask = mx.array(mask_np, dtype=mx.float32)

        chunked = chunked_cross_entropy_loss(hidden_bf16, weight_bf16, targets, mask)
        mx.eval(chunked)

        logits = hidden_bf16.astype(mx.float32) @ weight_bf16.astype(mx.float32).T
        target_lp = mx.take_along_axis(
            logits - mx.logsumexp(logits, axis=-1, keepdims=True),
            targets[:, :, None],
            axis=-1,
        ).squeeze(-1)
        expected = (-target_lp * mask).sum()
        mx.eval(expected)

        assert abs(chunked.item() - expected.item()) < 1e-4


class TestImportanceSamplingLoss:
    def test_on_policy(self, cfg):
        # When new_lp == old_lp, ratio=1, loss = -sum(advantages)
        lp = mx.full((1, 3), -1.0)
        mask = mx.ones((1, 3))
        advantages = mx.array([[1.0, 2.0, 3.0]])

        loss = importance_sampling_loss(lp, mask, lp, advantages, cfg)
        mx.eval(loss)
        expected = -(1.0 + 2.0 + 3.0)
        assert abs(loss.item() - expected) < 1e-5

    def test_off_policy(self, cfg):
        new_lp = mx.array([[-0.5, -0.5]])
        old_lp = mx.array([[-1.0, -1.0]])
        mask = mx.ones((1, 2))
        advantages = mx.array([[1.0, 1.0]])

        loss = importance_sampling_loss(new_lp, mask, old_lp, advantages, cfg)
        mx.eval(loss)
        # ratio = exp(-0.5 - (-1.0)) = exp(0.5) ≈ 1.6487, 2 tokens
        # sum: -(exp(0.5)*1 + exp(0.5)*1) = -2*exp(0.5)
        import math

        expected = -2 * math.exp(0.5)
        assert abs(loss.item() - expected) < 1e-4


class TestPPOLoss:
    def test_no_clipping_on_policy(self, ppo_cfg):
        lp = mx.full((1, 2), -1.0)
        mask = mx.ones((1, 2))
        advantages = mx.array([[1.0, 1.0]])

        loss = ppo_loss(lp, mask, lp, advantages, ppo_cfg)
        mx.eval(loss)
        # ratio=1, clipped_ratio=1, min(1*1, 1*1) = 1, sum of 2 tokens: -2.0
        assert abs(loss.item() - (-2.0)) < 1e-5

    def test_clipping_active(self, ppo_cfg):
        # Large ratio should be clipped
        new_lp = mx.array([[0.0]])
        old_lp = mx.array([[-2.0]])
        mask = mx.ones((1, 1))
        advantages = mx.array([[1.0]])

        loss = ppo_loss(new_lp, mask, old_lp, advantages, ppo_cfg)
        mx.eval(loss)
        # ratio = exp(2.0) ≈ 7.39, clip(ratio, 1.0-0, 1.0+0.2) = 1.2
        # loss = -min(7.39, 1.2).sum() = -1.2
        assert abs(loss.item() - (-1.2)) < 1e-4


class TestCISPOLoss:
    def test_positive_advantage_clipping(self, cispo_cfg):
        new_lp = mx.array([[-0.5]])
        old_lp = mx.array([[-2.5]])
        mask = mx.ones((1, 1))
        advantages = mx.array([[1.0]])  # positive

        loss = cispo_loss(new_lp, mask, old_lp, advantages, cispo_cfg)
        mx.eval(loss)
        # ratio = exp(-0.5 - (-2.5)) = exp(2.0) ≈ 7.39
        # positive adv → clip to [1-0.2, 1+0.2] = [0.8, 1.2], clipped_ratio=1.2
        # loss = -(sg(1.2) * (-0.5) * 1.0).sum() = -(1.2 * -0.5) = 0.6
        assert abs(loss.item() - 0.6) < 1e-4


class TestLossFunctionMap:
    def test_all_registered(self):
        assert "cross_entropy" in LOSS_FUNCTION_MAP
        assert "importance_sampling" in LOSS_FUNCTION_MAP
        assert "ppo" in LOSS_FUNCTION_MAP
        assert "cispo" in LOSS_FUNCTION_MAP
        assert "dro" in LOSS_FUNCTION_MAP

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

        # Cross-entropy with log-prob=-1 and full mask: sum of 2 tokens = 2.0
        ce = cross_entropy_loss(mx.full((1, 2), -1.0), mx.ones((1, 2)), mx.zeros((1, 2)), mx.zeros((1, 2)), cfg)
        mx.eval(ce)
        assert abs(ce.item() - 2.0) < 1e-5, f"CE loss should be 2.0, got {ce.item()}"

        # On-policy IS loss (ratio=1) should equal -sum(advantages)
        lp = mx.full((1, 2), -1.0)
        adv = mx.array([[2.0, 4.0]])
        is_loss = importance_sampling_loss(lp, mx.ones((1, 2)), lp, adv, cfg)
        mx.eval(is_loss)
        assert abs(is_loss.item() - (-6.0)) < 1e-5, f"IS loss should be -6.0, got {is_loss.item()}"

    def test_all_zero_mask_returns_zero(self, cfg):
        """CE returns 0 with zero mask; IS/PPO/CISPO/DRO return 0 with zero advantages."""
        # CE uses mask directly
        lp = mx.full((1, 3), -1.0)
        zero_mask = mx.zeros((1, 3))
        dummy = mx.zeros((1, 3))
        result = cross_entropy_loss(lp, zero_mask, dummy, dummy, cfg)
        mx.eval(result)
        assert result.item() == 0.0, "CE should return 0 with zero mask"

        # IS/PPO/CISPO/DRO use advantages as implicit mask (advantages=0 → no contribution)
        zero_adv = mx.zeros((1, 3))
        mask = mx.ones((1, 3))
        for name in ("importance_sampling", "ppo", "cispo", "dro"):
            fn = LOSS_FUNCTION_MAP[name]
            result = fn(lp, mask, lp, zero_adv, cfg)
            mx.eval(result)
            assert abs(result.item()) < 1e-6, f"{name} should return 0 with zero advantages"


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
        new_lp = mx.array([[-0.5]])
        old_lp = mx.array([[-2.5]])
        mask = mx.ones((1, 1))
        advantages = mx.array([[-1.0]])  # negative

        loss = cispo_loss(new_lp, mask, old_lp, advantages, cispo_cfg)
        mx.eval(loss)
        # ratio = exp(-0.5 - (-2.5)) = exp(2.0) ≈ 7.39
        # negative adv → clip to [1-0.1, 1+0.1] = [0.9, 1.1], clipped=1.1
        # loss = -(sg(1.1) * (-0.5) * (-1.0)).sum() = -(0.55) = -0.55
        expected = -0.55
        assert abs(loss.item() - expected) < 1e-4


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


# ---------------------------------------------------------------------------
# Finite-difference gradient checks (Tier 1 — Critical)
# ---------------------------------------------------------------------------


class TestFiniteDifferenceGradients:
    """Verify mx.grad matches (f(x+ε)-f(x-ε))/2ε for every loss function."""

    def test_cross_entropy_fd(self):
        cfg = LossFnConfig()
        target_lp = mx.array([[-1.0, -2.0, -0.5, -1.5]], dtype=mx.float32)
        mask = mx.array([[1.0, 1.0, 0.0, 1.0]])
        dummy = mx.zeros_like(target_lp)

        finite_difference_check_argnum(
            cross_entropy_loss,
            (target_lp, mask, dummy, dummy, cfg),
            argnum=0,
            rtol=2e-3,
            atol=1e-3,
        )

    def test_importance_sampling_fd(self):
        cfg = LossFnConfig()
        new_lp = mx.array([[-0.8, -1.2, -0.5]], dtype=mx.float32)
        old_lp = mx.array([[-1.0, -1.0, -1.0]], dtype=mx.float32)
        mask = mx.ones((1, 3))
        advantages = mx.array([[1.0, -0.5, 2.0]], dtype=mx.float32)

        finite_difference_check_argnum(
            importance_sampling_loss,
            (new_lp, mask, old_lp, advantages, cfg),
            argnum=0,
            rtol=2e-3,
            atol=5e-3,
        )

    def test_ppo_loss_fd(self):
        """PPO has clip kinks — use wider epsilon and tolerances."""
        cfg = LossFnConfig(clip_high_threshold=0.2)
        # Keep ratios clearly in the interior of clipped/unclipped regions
        new_lp = mx.array([[-0.95, -1.5]], dtype=mx.float32)  # ratio ≈ 1.05, 0.61
        old_lp = mx.array([[-1.0, -1.0]], dtype=mx.float32)
        mask = mx.ones((1, 2))
        advantages = mx.array([[1.0, 1.0]], dtype=mx.float32)

        finite_difference_check_argnum(
            ppo_loss,
            (new_lp, mask, old_lp, advantages, cfg),
            argnum=0,
            epsilon=5e-4,
            rtol=5e-3,
            atol=1e-4,
        )

    def test_cispo_loss_gradient_matches_expected(self):
        """CISPO uses stop_gradient on clipped_ratio, so autograd gradient is
        -(sg(clipped_ratio) * advantages), NOT the full numerical derivative.
        Verify autograd matches this expected formula.
        """
        cfg = LossFnConfig(clip_low_threshold=0.1, clip_high_threshold=0.2)
        new_lp = mx.array([[-0.95, -1.5]], dtype=mx.float32)
        old_lp = mx.array([[-1.0, -1.0]], dtype=mx.float32)
        mask = mx.ones((1, 2))
        advantages = mx.array([[1.0, -0.5]], dtype=mx.float32)

        grad_fn = mx.grad(lambda lp: cispo_loss(lp, mask, old_lp, advantages, cfg))
        grads = grad_fn(new_lp)
        mx.eval(grads)

        log_ratio = new_lp - old_lp
        ratio = mx.exp(log_ratio)
        positive_adv = advantages > 0
        clipped_ratio = mx.where(
            positive_adv,
            mx.clip(ratio, 1.0 - cfg.clip_high_threshold, 1.0 + cfg.clip_high_threshold),
            mx.clip(ratio, 1.0 - cfg.clip_low_threshold, 1.0 + cfg.clip_low_threshold),
        )
        expected_grad = -clipped_ratio * advantages
        mx.eval(expected_grad)

        abs_diff = mx.max(mx.abs(grads - expected_grad)).item()
        assert abs_diff < 1e-5, (
            f"CISPO gradient mismatch: max abs_diff={abs_diff:.6e}"
        )

    def test_chunked_ce_fd_hidden(self):
        """Finite-difference check for chunked CE w.r.t. hidden_states."""
        mx.random.seed(42)
        vocab_size, dim, seq_len = 32, 8, 4
        lm_head = nn.Linear(dim, vocab_size, bias=False)
        mx.eval(lm_head.parameters())

        hidden = mx.random.normal((1, seq_len, dim))
        targets = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        mask = mx.ones((1, seq_len))
        mx.eval(hidden)

        def loss_fn(h):
            return chunked_cross_entropy_loss(h, lm_head.weight, targets, mask)

        finite_difference_check(loss_fn, hidden, rtol=5e-2, atol=5e-3)

    def test_chunked_ce_fd_weight(self):
        """Finite-difference check for chunked CE w.r.t. lm_head weight."""
        mx.random.seed(42)
        vocab_size, dim, seq_len = 16, 8, 3
        weight = mx.random.normal((vocab_size, dim))
        hidden = mx.random.normal((1, seq_len, dim))
        targets = mx.array([[1, 2, 3]], dtype=mx.int32)
        mask = mx.ones((1, seq_len))
        mx.eval(weight, hidden)

        def loss_fn(w):
            return chunked_cross_entropy_loss(hidden, w, targets, mask)

        finite_difference_check(loss_fn, weight, rtol=1e-1, atol=2e-2)


# ---------------------------------------------------------------------------
# Multi-token mixed-batch tests (Tier 1)
# ---------------------------------------------------------------------------


class TestPPOMixedBatch:
    """PPO with mixed positive/negative advantages and partial clipping."""

    def test_mixed_positive_negative_advantages(self):
        """[1,6] batch: first 3 tokens positive adv, last 3 negative."""
        cfg = LossFnConfig(clip_high_threshold=0.2)
        # On-policy (ratio=1) to isolate advantage sign effect
        lp = mx.full((1, 6), -1.0)
        mask = mx.ones((1, 6))
        advantages = mx.array([[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]])

        loss = ppo_loss(lp, mask, lp, advantages, cfg)
        mx.eval(loss)
        # ratio=1, clipped=1, surr1=surr2=adv, min=adv
        # loss = -sum(advantages) = -(1+2+3-1-2-3) = 0
        assert abs(loss.item() - 0.0) < 1e-5

    def test_some_clipped_some_not(self):
        """Tokens where ratio=1 (no clip) mixed with ratio>>1 (clipped)."""
        cfg = LossFnConfig(clip_high_threshold=0.2)
        # Token 0: on-policy, token 1: off-policy (large ratio)
        new_lp = mx.array([[-1.0, 0.0]])
        old_lp = mx.array([[-1.0, -2.0]])
        mask = mx.ones((1, 2))
        advantages = mx.array([[1.0, 1.0]])

        loss = ppo_loss(new_lp, mask, old_lp, advantages, cfg)
        mx.eval(loss)

        # Token 0: ratio=1, surr1=surr2=1, loss_tok=-1
        # Token 1: ratio=exp(2)≈7.39, clipped=1.2, surr1=7.39, surr2=1.2, min=1.2, loss_tok=-1.2
        # sum reduction: -(1.0 + 1.2) = -2.2
        expected = -(1.0 + 1.2)
        assert abs(loss.item() - expected) < 1e-4

    def test_verify_clipping_branch_selection(self):
        """Explicitly verify which surrogate is selected by min(surr1, surr2)."""
        cfg = LossFnConfig(clip_high_threshold=0.2)

        # Positive advantage + large ratio → surr2 < surr1 (clipped wins)
        new_lp = mx.array([[0.0]])
        old_lp = mx.array([[-2.0]])
        mask = mx.ones((1, 1))
        adv_pos = mx.array([[1.0]])

        loss_pos = ppo_loss(new_lp, mask, old_lp, adv_pos, cfg)
        mx.eval(loss_pos)
        ratio = math.exp(2.0)
        surr1_pos = ratio * 1.0
        surr2_pos = 1.2 * 1.0  # clipped
        assert surr2_pos < surr1_pos  # clipped is smaller
        assert abs(loss_pos.item() - (-surr2_pos)) < 1e-4

        # Negative advantage + large ratio → surr1 < surr2 (unclipped wins)
        adv_neg = mx.array([[-1.0]])
        loss_neg = ppo_loss(new_lp, mask, old_lp, adv_neg, cfg)
        mx.eval(loss_neg)
        surr1_neg = ratio * (-1.0)
        surr2_neg = 1.2 * (-1.0)
        assert surr1_neg < surr2_neg  # unclipped is smaller (more negative)
        assert abs(loss_neg.item() - (-surr1_neg)) < 1e-3


class TestCISPOMixedBatch:
    """CISPO with asymmetric clipping and mixed advantages."""

    def test_asymmetric_clipping_mixed_advantages(self):
        """Positive advantages use clip_high, negative use clip_low."""
        cfg = LossFnConfig(clip_low_threshold=0.1, clip_high_threshold=0.2)
        # Large ratio, mixed advantages, non-zero logprobs
        new_lp = mx.array([[-0.5, -0.5]])
        old_lp = mx.array([[-2.5, -2.5]])
        mask = mx.ones((1, 2))
        advantages = mx.array([[1.0, -1.0]])

        loss = cispo_loss(new_lp, mask, old_lp, advantages, cfg)
        mx.eval(loss)

        # ratio = exp(-0.5 - (-2.5)) = exp(2.0) ≈ 7.39
        # CISPO: -(sg(clipped_ratio) * target_logprobs * advantages).sum()
        # tok0 (positive adv): clip to [0.8, 1.2], clipped=1.2
        #   contribution: sg(1.2) * (-0.5) * 1.0 = -0.6
        # tok1 (negative adv): clip to [0.9, 1.1], clipped=1.1
        #   contribution: sg(1.1) * (-0.5) * (-1.0) = 0.55
        # loss = -(-0.6 + 0.55) = -(−0.05) = 0.05
        expected = 0.05
        assert abs(loss.item() - expected) < 1e-4

    def test_clip_low_vs_clip_high_differs(self):
        """Swapping advantage signs changes the loss because thresholds differ."""
        cfg = LossFnConfig(clip_low_threshold=0.5, clip_high_threshold=0.2)
        new_lp = mx.array([[-0.5]])
        old_lp = mx.array([[-2.5]])
        mask = mx.ones((1, 1))

        loss_pos = cispo_loss(new_lp, mask, old_lp, mx.array([[1.0]]), cfg)
        loss_neg = cispo_loss(new_lp, mask, old_lp, mx.array([[-1.0]]), cfg)
        mx.eval(loss_pos, loss_neg)

        # ratio=exp(2)≈7.39
        # positive adv: clip to [0.8, 1.2], clipped=1.2, loss=-(1.2*-0.5*1)=0.6
        # negative adv: clip to [0.5, 1.5], clipped=1.5, loss=-(1.5*-0.5*-1)=-0.75
        assert loss_pos.item() != loss_neg.item()

    def test_edge_case_zero_advantage(self):
        """Zero advantage → loss should be zero (ratio * 0 = 0)."""
        cfg = LossFnConfig(clip_low_threshold=0.1, clip_high_threshold=0.2)
        new_lp = mx.array([[0.0, -1.0]])
        old_lp = mx.array([[-2.0, -0.5]])
        mask = mx.ones((1, 2))
        advantages = mx.zeros((1, 2))

        loss = cispo_loss(new_lp, mask, old_lp, advantages, cfg)
        mx.eval(loss)
        assert abs(loss.item()) < 1e-6
