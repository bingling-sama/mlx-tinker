"""Cross-framework parity tests: MLX operations vs PyTorch ground truth.

Every core numerical operation in mlx-tinker is tested against a PyTorch
reference implementation using identical inputs (same numpy seed). This
provides mechanistic proof of correctness beyond self-referencing MLX tests.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch required for parity tests")

from mlx_tinker.backend.loss_fns import (  # noqa: E402
    LOSS_FUNCTION_MAP,
    chunked_cross_entropy_loss,
)
from mlx_tinker.backend.optimizers import AdamW8Bit  # noqa: E402
from mlx_tinker.backend.training import _clip_grad_norm  # noqa: E402
from tests.helpers import TinyModel  # noqa: E402
from tests.pt_helpers import (  # noqa: E402
    PT_LOSS_MAP,
    PtAdamWNoBiasCorrection,
    PtTinyModel,
    assert_close,
    gen_chunked_ce_inputs,
    gen_logits,
    gen_loss_inputs,
    gen_optimizer_inputs,
    get_loss_cfg,
    mx_to_np,
    np_to_mx,
    np_to_pt,
    pt_standard_cross_entropy,
    pt_to_np,
    sync_weights_mx_to_pt,
)

SEED = 42
ALL_LOSSES = ["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"]
ALL_REGIMES = ["normal", "extreme", "near_zero", "large_ratio"]


# ---------------------------------------------------------------------------
# Loss forward parity: MLX vs PyTorch on identical inputs
# ---------------------------------------------------------------------------


class TestLossForwardParity:
    @pytest.mark.parametrize("loss_name", ALL_LOSSES)
    @pytest.mark.parametrize("regime", ALL_REGIMES)
    def test_forward_parity(self, loss_name, regime):
        rng = np.random.RandomState(SEED)
        target_lp, mask, samp_lp, adv = gen_loss_inputs(rng, regime)

        mlx_fn = LOSS_FUNCTION_MAP[loss_name]
        pt_fn = PT_LOSS_MAP[loss_name]
        mlx_cfg, pt_cfg = get_loss_cfg(loss_name)

        mlx_val = mlx_fn(
            np_to_mx(target_lp), np_to_mx(mask),
            np_to_mx(samp_lp), np_to_mx(adv), mlx_cfg,
        )
        mx.eval(mlx_val)

        pt_val = pt_fn(
            np_to_pt(target_lp), np_to_pt(mask),
            np_to_pt(samp_lp), np_to_pt(adv), pt_cfg,
        )

        assert_close(
            mx_to_np(mlx_val), pt_to_np(pt_val),
            rtol=1e-5, atol=1e-5,
            msg=f"{loss_name}/{regime} forward",
        )


# ---------------------------------------------------------------------------
# Loss gradient parity: MLX autograd vs PyTorch autograd
# ---------------------------------------------------------------------------


class TestLossGradientParity:
    @pytest.mark.parametrize("loss_name", ALL_LOSSES)
    def test_gradient_parity(self, loss_name):
        rng = np.random.RandomState(SEED)
        target_lp, mask, samp_lp, adv = gen_loss_inputs(rng, "normal")

        mlx_fn = LOSS_FUNCTION_MAP[loss_name]
        pt_fn = PT_LOSS_MAP[loss_name]
        mlx_cfg, pt_cfg = get_loss_cfg(loss_name)

        # MLX gradient w.r.t. target_logprobs
        def mlx_loss_of_lp(lp):
            return mlx_fn(lp, np_to_mx(mask), np_to_mx(samp_lp), np_to_mx(adv), mlx_cfg)

        mlx_grad = mx.grad(mlx_loss_of_lp)(np_to_mx(target_lp))
        mx.eval(mlx_grad)

        # PyTorch gradient w.r.t. target_logprobs
        pt_lp = np_to_pt(target_lp, requires_grad=True)
        pt_loss = pt_fn(pt_lp, np_to_pt(mask), np_to_pt(samp_lp), np_to_pt(adv), pt_cfg)
        pt_loss.backward()

        assert_close(
            mx_to_np(mlx_grad), pt_to_np(pt_lp.grad),
            rtol=1e-3, atol=1e-5,
            msg=f"{loss_name} gradient",
        )


# ---------------------------------------------------------------------------
# Chunked CE parity: MLX chunked vs PyTorch standard CE
# ---------------------------------------------------------------------------


class TestChunkedCEParity:
    @pytest.mark.parametrize("V", [128, 1024])
    def test_mlx_chunked_vs_pt_standard(self, V):
        """Gold standard: MLX chunked CE vs PyTorch log_softmax+gather."""
        import mlx_tinker.backend.loss_fns as lf_module

        rng = np.random.RandomState(SEED)
        hidden_np, weight_np, targets_np, mask_np = gen_chunked_ce_inputs(rng, V=V, D=16)

        # MLX chunked CE
        old_cs = lf_module.CE_CHUNK_SIZE
        try:
            lf_module.CE_CHUNK_SIZE = 64
            mlx_val = chunked_cross_entropy_loss(
                np_to_mx(hidden_np), np_to_mx(weight_np),
                np_to_mx(targets_np), np_to_mx(mask_np),
            )
            mx.eval(mlx_val)
        finally:
            lf_module.CE_CHUNK_SIZE = old_cs

        # PyTorch standard (non-chunked) — gold standard
        pt_val = pt_standard_cross_entropy(
            np_to_pt(hidden_np), np_to_pt(weight_np),
            np_to_pt(targets_np.astype(np.int64)), np_to_pt(mask_np),
        )

        assert_close(
            mx_to_np(mlx_val), pt_to_np(pt_val),
            rtol=1e-4, atol=1e-4,
            msg=f"chunked_ce_vs_pt_std V={V}",
        )

    @pytest.mark.parametrize("V", [128, 1024])
    def test_mlx_chunked_vs_pt_chunked(self, V):
        """Both chunked implementations should match exactly."""
        import mlx_tinker.backend.loss_fns as lf_module
        from tests.pt_helpers import pt_chunked_cross_entropy_loss

        rng = np.random.RandomState(SEED)
        hidden_np, weight_np, targets_np, mask_np = gen_chunked_ce_inputs(rng, V=V, D=16)
        cs = 64

        # MLX chunked
        old_cs = lf_module.CE_CHUNK_SIZE
        try:
            lf_module.CE_CHUNK_SIZE = cs
            mlx_val = chunked_cross_entropy_loss(
                np_to_mx(hidden_np), np_to_mx(weight_np),
                np_to_mx(targets_np), np_to_mx(mask_np),
            )
            mx.eval(mlx_val)
        finally:
            lf_module.CE_CHUNK_SIZE = old_cs

        # PyTorch chunked
        pt_val = pt_chunked_cross_entropy_loss(
            np_to_pt(hidden_np), np_to_pt(weight_np),
            np_to_pt(targets_np.astype(np.int64)), np_to_pt(mask_np),
            chunk_size=cs,
        )

        assert_close(
            mx_to_np(mlx_val), pt_to_np(pt_val),
            rtol=1e-5, atol=1e-5,
            msg=f"chunked_ce_vs_pt_chunked V={V}",
        )

    def test_gradient_vs_pt(self):
        """Gradient of chunked CE (hidden_states) vs PyTorch standard CE."""
        import mlx_tinker.backend.loss_fns as lf_module

        rng = np.random.RandomState(SEED)
        hidden_np, weight_np, targets_np, mask_np = gen_chunked_ce_inputs(
            rng, B=1, T=4, V=128, D=16,
        )

        # MLX gradient w.r.t. hidden_states
        def mlx_chunked_of_hidden(h):
            old = lf_module.CE_CHUNK_SIZE
            try:
                lf_module.CE_CHUNK_SIZE = 8
                return chunked_cross_entropy_loss(
                    h, np_to_mx(weight_np), np_to_mx(targets_np), np_to_mx(mask_np),
                )
            finally:
                lf_module.CE_CHUNK_SIZE = old

        mlx_grad = mx.grad(mlx_chunked_of_hidden)(np_to_mx(hidden_np))
        mx.eval(mlx_grad)

        # PyTorch gradient w.r.t. hidden_states
        pt_h = np_to_pt(hidden_np, requires_grad=True)
        pt_loss = pt_standard_cross_entropy(
            pt_h, np_to_pt(weight_np),
            np_to_pt(targets_np.astype(np.int64)), np_to_pt(mask_np),
        )
        pt_loss.backward()

        assert_close(
            mx_to_np(mlx_grad), pt_to_np(pt_h.grad),
            rtol=1e-3, atol=1e-4,
            msg="chunked_ce gradient vs pt",
        )


# ---------------------------------------------------------------------------
# Log-probability parity
# ---------------------------------------------------------------------------


class TestLogProbParity:
    @pytest.mark.parametrize("V", [128, 1024])
    def test_target_logprobs(self, V):
        """MLX logsumexp+take_along_axis vs PyTorch log_softmax+gather."""
        rng = np.random.RandomState(SEED)
        logits_np, targets_np = gen_logits(rng, B=2, T=8, V=V)

        # MLX path (matches training.py)
        logits_mx = np_to_mx(logits_np)
        lp_mx = logits_mx - mx.logsumexp(logits_mx, axis=-1, keepdims=True)
        target_lp_mx = mx.take_along_axis(
            lp_mx, np_to_mx(targets_np)[:, :, None].astype(mx.int32), axis=-1,
        ).squeeze(-1)
        mx.eval(target_lp_mx)

        # PyTorch path
        logits_pt = np_to_pt(logits_np)
        lp_pt = torch.log_softmax(logits_pt, dim=-1)
        target_lp_pt = torch.gather(
            lp_pt, 2, torch.tensor(targets_np.astype(np.int64)).unsqueeze(-1),
        ).squeeze(-1)

        assert_close(
            mx_to_np(target_lp_mx), pt_to_np(target_lp_pt),
            rtol=1e-5, atol=1e-5,
            msg=f"logprob V={V}",
        )


# ---------------------------------------------------------------------------
# Gradient clipping parity
# ---------------------------------------------------------------------------


class TestGradClipParity:
    @pytest.mark.parametrize(
        "max_norm,label",
        [(1.0, "aggressive"), (1000.0, "no_op")],
        ids=["aggressive", "no_op"],
    )
    def test_clip_parity(self, max_norm, label):
        rng = np.random.RandomState(SEED)
        grads_np = {f"layer{i}": rng.randn(16, 16).astype(np.float32) for i in range(4)}

        # MLX
        mlx_grads = {k: mx.array(v) for k, v in grads_np.items()}
        mlx_clipped, mlx_pre, mlx_post = _clip_grad_norm(mlx_grads, max_norm)
        mlx_clipped_np = {k: mx_to_np(v) for k, v in mlx_clipped.items()}

        # PyTorch reference (manual)
        pt_norm = math.sqrt(sum(np.sum(g**2) for g in grads_np.values()))
        if pt_norm > max_norm:
            scale = max_norm / (pt_norm + 1e-6)
            pt_clipped_np = {k: v * scale for k, v in grads_np.items()}
        else:
            pt_clipped_np = {k: v.copy() for k, v in grads_np.items()}

        # Compare each gradient tensor
        for key in grads_np:
            assert_close(
                mlx_clipped_np[key], pt_clipped_np[key],
                rtol=1e-5, atol=1e-5,
                msg=f"grad_clip/{label}/{key}",
            )

        # Compare reported pre-clip norm
        assert abs(mlx_pre - pt_norm) < 1e-3, (
            f"Pre-clip norm mismatch: MLX={mlx_pre:.6f} PT={pt_norm:.6f}"
        )

    def test_clip_near_threshold(self):
        rng = np.random.RandomState(SEED)
        grads_np = {f"layer{i}": rng.randn(16, 16).astype(np.float32) for i in range(4)}
        true_norm = math.sqrt(sum(np.sum(g**2) for g in grads_np.values()))
        max_norm = true_norm * 1.01  # just above threshold → no clip

        mlx_grads = {k: mx.array(v) for k, v in grads_np.items()}
        mlx_clipped, mlx_pre, mlx_post = _clip_grad_norm(mlx_grads, max_norm)
        mlx_clipped_np = {k: mx_to_np(v) for k, v in mlx_clipped.items()}

        # Should be unclipped (norm < max_norm)
        for key in grads_np:
            assert_close(
                mlx_clipped_np[key], grads_np[key],
                rtol=1e-5, atol=1e-5,
                msg=f"grad_clip/near_threshold/{key}",
            )


# ---------------------------------------------------------------------------
# Full training step parity (logits → loss → gradients → optimizer update)
# ---------------------------------------------------------------------------


class TestFullTrainingStepParity:
    def test_full_step(self):
        """End-to-end: logit → loss → gradient → post-optimizer param parity."""
        vocab_size, dim = 32, 16
        input_ids = [5, 10, 15, 20, 25]
        targets = [10, 15, 20, 25, 1]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        # Create MLX model and sync weights to PyTorch
        mx.random.seed(SEED)
        mx_model = TinyModel(vocab_size, dim)
        mx.eval(mx_model.parameters())

        pt_model = PtTinyModel(vocab_size, dim)
        sync_weights_mx_to_pt(mx_model, pt_model)

        # --- G1: Logit parity ---
        mx_logits = mx_model(mx.array([input_ids]))
        mx.eval(mx_logits)
        pt_logits = pt_model(torch.tensor([input_ids]))

        assert_close(
            mx_to_np(mx_logits), pt_to_np(pt_logits),
            rtol=1e-5, atol=1e-5,
            msg="logit parity",
        )

        # --- G2: Loss parity ---
        mx_lp = mx_logits - mx.logsumexp(mx_logits, axis=-1, keepdims=True)
        mx_target_lp = mx.take_along_axis(
            mx_lp, mx.array([targets])[:, :, None], axis=-1,
        ).squeeze(-1)
        mx_loss = (-mx_target_lp * mx.array([weights], dtype=mx.float32)).sum()
        mx.eval(mx_loss)

        pt_lp = torch.log_softmax(pt_logits, dim=-1)
        pt_target_lp = torch.gather(
            pt_lp, 2, torch.tensor([targets]).unsqueeze(-1),
        ).squeeze(-1)
        pt_loss = (-pt_target_lp * torch.tensor([weights])).sum()

        assert_close(
            mx_to_np(mx_loss), pt_to_np(pt_loss),
            rtol=1e-5, atol=1e-5,
            msg="loss parity",
        )

        # --- G3: Gradient parity ---
        def mx_loss_fn(model, input_ids_mx, targets_mx, weights_mx):
            logits = model(input_ids_mx)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            target_lp = mx.take_along_axis(lp, targets_mx[:, :, None], axis=-1).squeeze(-1)
            return (-target_lp * weights_mx).sum()

        loss_and_grad = nn.value_and_grad(mx_model, mx_loss_fn)
        mx_loss_val, mx_grads = loss_and_grad(
            mx_model,
            mx.array([input_ids]),
            mx.array([targets]),
            mx.array([weights], dtype=mx.float32),
        )
        mx.eval(mx_loss_val)

        # Fresh PyTorch forward for gradients (reuse synced weights)
        pt_model2 = PtTinyModel(vocab_size, dim)
        sync_weights_mx_to_pt(mx_model, pt_model2)
        pt_logits2 = pt_model2(torch.tensor([input_ids]))
        pt_lp2 = torch.log_softmax(pt_logits2, dim=-1)
        pt_target_lp2 = torch.gather(
            pt_lp2, 2, torch.tensor([targets]).unsqueeze(-1),
        ).squeeze(-1)
        pt_loss2 = (-pt_target_lp2 * torch.tensor([weights])).sum()
        pt_loss2.backward()

        assert_close(
            mx_to_np(mx_grads["embed"]["weight"]),
            pt_to_np(pt_model2.embed.weight.grad),
            rtol=1e-3, atol=1e-5,
            msg="embed gradient parity",
        )
        assert_close(
            mx_to_np(mx_grads["head"]["weight"]),
            pt_to_np(pt_model2.head.weight.grad),
            rtol=1e-3, atol=1e-5,
            msg="head gradient parity",
        )

        # --- G4: Post-optimizer param parity (fp32 AdamW) ---
        lr = 1e-3
        mx_opt = optim.AdamW(learning_rate=lr)
        mx_opt.update(mx_model, mx_grads)
        mx.eval(mx_model.parameters(), mx_opt.state)

        pt_embed_w = pt_model2.embed.weight.data.clone()
        pt_head_w = pt_model2.head.weight.data.clone()
        pt_opt = PtAdamWNoBiasCorrection(
            [pt_embed_w, pt_head_w], lr=lr, betas=(0.9, 0.999), eps=1e-8,
        )
        pt_opt.step([
            (pt_embed_w, pt_model2.embed.weight.grad),
            (pt_head_w, pt_model2.head.weight.grad),
        ])

        assert_close(
            mx_to_np(mx_model.embed.weight), pt_to_np(pt_embed_w),
            rtol=1e-4, atol=1e-5,
            msg="post-optim embed parity",
        )
        assert_close(
            mx_to_np(mx_model.head.weight), pt_to_np(pt_head_w),
            rtol=1e-4, atol=1e-5,
            msg="post-optim head parity",
        )


# ---------------------------------------------------------------------------
# Optimizer parity: AdamW8Bit vs fp32 reference
# ---------------------------------------------------------------------------


class TestOptimizerParity:
    @pytest.mark.parametrize("n_steps", [1, 5, 20])
    def test_adamw8bit_vs_fp32(self, n_steps):
        """MLX AdamW8Bit should track fp32 AdamW direction and magnitude."""
        rng = np.random.RandomState(SEED)
        param_np, grad_list = gen_optimizer_inputs(rng, shape=(64, 32), n_steps=n_steps)
        lr = 1e-4
        betas = (0.9, 0.999)
        eps = 1e-8

        # MLX AdamW8Bit path
        class SingleParam(nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w

        model_8bit = SingleParam(mx.array(param_np.copy()))
        opt_8bit = AdamW8Bit(learning_rate=lr, betas=list(betas), eps=eps)

        for step_i in range(n_steps):
            g = mx.array(grad_list[step_i])
            opt_8bit.update(model_8bit, {"w": g})
            mx.eval(model_8bit.parameters(), opt_8bit.state)

        mlx_result = mx_to_np(model_8bit.w)

        # PyTorch fp32 AdamW (no bias correction)
        pt_param = torch.tensor(param_np.copy())
        pt_opt = PtAdamWNoBiasCorrection([pt_param], lr=lr, betas=betas, eps=eps)
        for step_i in range(n_steps):
            g = torch.tensor(grad_list[step_i])
            pt_opt.step([(pt_param, g)])

        pt_result = pt_to_np(pt_param)

        # Cosine similarity and magnitude ratio
        cos_sim = float(np.dot(mlx_result.flatten(), pt_result.flatten()) / (
            np.linalg.norm(mlx_result) * np.linalg.norm(pt_result) + 1e-12
        ))
        mag_ratio = float(
            np.linalg.norm(mlx_result) / (np.linalg.norm(pt_result) + 1e-12)
        )

        assert cos_sim > 0.95, (
            f"Cosine similarity too low after {n_steps} steps: {cos_sim:.6f}"
        )
        assert 0.5 < mag_ratio < 2.0, (
            f"Magnitude ratio out of range after {n_steps} steps: {mag_ratio:.6f}"
        )
