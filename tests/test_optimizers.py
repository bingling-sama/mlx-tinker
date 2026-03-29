"""Tests for 8-bit AdamW optimizer and quantization utilities."""

import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_tinker.backend.lora_manager import LoRAManager
from mlx_tinker.backend.optimizers import (
    AdamW8Bit,
    _dequantize_blockwise,
    _quantize_blockwise,
    create_dynamic_map,
)
from mlx_tinker.backend.training import TrainingBackend
from mlx_tinker.types import (
    AdamParams,
    ForwardBackwardInput,
    LoraConfig,
    OptimStepInput,
)
from tests.helpers import TinyLM, TinyModel, make_datum


class TestQuantizeBlockwise:
    def test_roundtrip_accuracy(self):
        mx.random.seed(42)
        tensor = mx.random.normal((1024,))
        mx.eval(tensor)

        quantized, absmax = _quantize_blockwise(tensor)
        recovered = _dequantize_blockwise(quantized, absmax)
        mx.eval(recovered)

        # Error should be small relative to the range of the tensor
        tensor_range = mx.max(tensor).item() - mx.min(tensor).item()
        max_error = mx.max(mx.abs(tensor - recovered)).item()
        assert max_error < 0.01 * tensor_range, (
            f"Roundtrip error {max_error:.6f} exceeds 1% of range {tensor_range:.4f}"
        )

    def test_zeros(self):
        tensor = mx.zeros((512,))
        quantized, absmax = _quantize_blockwise(tensor)
        recovered = _dequantize_blockwise(quantized, absmax)
        mx.eval(recovered)

        assert mx.allclose(recovered, mx.zeros((512,)), atol=1e-10).item()

    def test_output_dtypes(self):
        tensor = mx.random.normal((256,))
        mx.eval(tensor)

        quantized, absmax = _quantize_blockwise(tensor)
        mx.eval(quantized, absmax)

        assert quantized.dtype == mx.uint8
        assert absmax.dtype == mx.float32

    def test_non_multiple_of_block_size(self):
        # 300 is not a multiple of 256
        tensor = mx.random.normal((300,))
        mx.eval(tensor)

        quantized, absmax = _quantize_blockwise(tensor, block_size=256)
        recovered = _dequantize_blockwise(quantized, absmax, block_size=256)
        mx.eval(recovered)

        assert recovered.shape == (300,)
        tensor_range = mx.max(tensor).item() - mx.min(tensor).item()
        max_error = mx.max(mx.abs(tensor - recovered)).item()
        assert max_error < 0.01 * tensor_range, (
            f"Roundtrip error {max_error:.6f} exceeds 1% of range {tensor_range:.4f}"
        )


class TestAdamW8Bit:
    def test_convergence(self):
        mx.random.seed(0)
        model = TinyModel(vocab_size=32, dim=16)
        mx.eval(model.parameters())

        training = TrainingBackend(optimizer_type="adamw_8bit", gradient_checkpointing=False)

        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_request = OptimStepInput(
            adam_params=AdamParams(learning_rate=1e-3, weight_decay=0.0)
        )

        losses = []
        for _ in range(30):
            result = training.forward_backward("test", model, fb_request)
            losses.append(result.metrics["loss:sum"])
            training.optim_step("test", model, opt_request)

        avg_first = sum(losses[:5]) / 5
        avg_last = sum(losses[-5:]) / 5
        assert avg_last < avg_first, (
            f"Loss should decrease: first-5 avg={avg_first:.4f}, "
            f"last-5 avg={avg_last:.4f}"
        )

    def test_state_dtypes(self):
        mx.random.seed(0)
        model = TinyModel(vocab_size=32, dim=16)
        mx.eval(model.parameters())

        optimizer = AdamW8Bit(learning_rate=1e-3)
        optimizer.init(model.trainable_parameters())

        # Do one forward/backward + step to populate state
        training = TrainingBackend(optimizer_type="adamw_8bit", gradient_checkpointing=False)
        datum = make_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])
        fb_request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_request = OptimStepInput(
            adam_params=AdamParams(learning_rate=1e-3, weight_decay=0.0)
        )

        training.forward_backward("test", model, fb_request)
        training.optim_step("test", model, opt_request)

        # Check internal optimizer state has int8 m and v
        opt = training.optimizers["test"]
        state = opt.state

        def _check_state(s):
            """Recursively check state dicts for int8 m/v."""
            if isinstance(s, dict):
                if "m" in s and "v" in s:
                    mx.eval(s["m"], s["v"])
                    assert s["m"].dtype == mx.uint8, f"Expected m to be uint8, got {s['m'].dtype}"
                    assert s["v"].dtype == mx.uint8, f"Expected v to be uint8, got {s['v'].dtype}"
                    return True
                return any(_check_state(v) for v in s.values())
            elif isinstance(s, (list, tuple)):
                return any(_check_state(v) for v in s)
            return False

        found = _check_state(state)
        assert found, "Should find uint8 m and v in optimizer state"

    def test_matches_fp32_direction(self):
        """Both adamw_8bit and adamw should reduce loss over 10 steps."""
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_request = OptimStepInput(
            adam_params=AdamParams(learning_rate=0.01, weight_decay=0.0)
        )

        results = {}
        for opt_type in ["adamw_8bit", "adamw"]:
            mx.random.seed(42)
            model = TinyModel(vocab_size=32, dim=16)
            mx.eval(model.parameters())

            training = TrainingBackend(optimizer_type=opt_type, gradient_checkpointing=False)

            losses = []
            for _ in range(10):
                result = training.forward_backward("test", model, fb_request)
                losses.append(result.metrics["loss:sum"])
                training.optim_step("test", model, opt_request)

            results[opt_type] = losses

        # Both should reduce loss
        for opt_type, losses in results.items():
            assert losses[-1] < losses[0], (
                f"{opt_type} should reduce loss: first={losses[0]:.4f}, "
                f"last={losses[-1]:.4f}"
            )

    def test_stable_at_lr_1e3(self):
        """8-bit AdamW with tree quantization should be stable at LR=1e-3."""
        mx.random.seed(42)
        model = TinyModel(vocab_size=32, dim=16)
        mx.eval(model.parameters())

        training = TrainingBackend(
            optimizer_type="adamw_8bit", gradient_checkpointing=False
        )
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_request = OptimStepInput(
            adam_params=AdamParams(learning_rate=1e-3, weight_decay=0.0)
        )

        losses = []
        for _ in range(20):
            result = training.forward_backward("test", model, fb_request)
            loss = result.metrics["loss:sum"]
            losses.append(loss)
            training.optim_step("test", model, opt_request)

        # Should not diverge (no inf/nan)

        assert all(math.isfinite(v) for v in losses), (
            f"Loss should be finite at LR=1e-3: {losses}"
        )
        # Should trend down
        avg_first = sum(losses[:5]) / 5
        avg_last = sum(losses[-5:]) / 5
        assert avg_last < avg_first

    def test_optimizer_hyperparam_updates_preserve_state(self):
        mx.random.seed(42)
        model = TinyModel(vocab_size=32, dim=16)
        mx.eval(model.parameters())

        training = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [1.0, 1.0, 1.0, 1.0])
        fb_request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")

        first_opt = OptimStepInput(
            adam_params=AdamParams(
                learning_rate=1e-3,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                weight_decay=0.0,
            )
        )
        second_opt = OptimStepInput(
            adam_params=AdamParams(
                learning_rate=2e-3,
                beta1=0.8,
                beta2=0.95,
                eps=1e-6,
                weight_decay=0.1,
            )
        )

        training.forward_backward("test", model, fb_request)
        training.optim_step("test", model, first_opt)

        optimizer = training.optimizers["test"]
        first_state = optimizer.state
        mx.eval(first_state["step"])
        assert first_state["step"].item() == 1

        training.forward_backward("test", model, fb_request)
        training.optim_step("test", model, second_opt)

        assert training.optimizers["test"] is optimizer
        mx.eval(optimizer.state["step"], optimizer.learning_rate)
        assert optimizer.state["step"].item() == 2
        assert optimizer.learning_rate.item() == pytest.approx(2e-3)
        assert optimizer.betas == (0.8, 0.95)
        assert optimizer.eps == pytest.approx(1e-6)
        assert optimizer.weight_decay == pytest.approx(0.1)


class TestDynamicMap:
    def test_map_size(self):
        dmap = create_dynamic_map()
        assert len(dmap) == 256

    def test_map_is_sorted(self):
        dmap = create_dynamic_map()
        for i in range(len(dmap) - 1):
            assert dmap[i] <= dmap[i + 1], f"Map not sorted at index {i}"

    def test_map_contains_zero(self):
        dmap = create_dynamic_map()
        assert 0.0 in dmap

    def test_map_has_positive_and_negative(self):
        """Signed map should have both positive and negative values."""
        dmap = create_dynamic_map(signed=True)
        positives = [v for v in dmap if v > 0]
        negatives = [v for v in dmap if v < 0]
        assert len(positives) > 50, "Should have many positive values"
        assert len(negatives) > 50, "Should have many negative values"
        # Counts may differ by 1 due to special 1.0 value
        assert abs(len(positives) - len(negatives)) <= 1


# ---------------------------------------------------------------------------
# QLoRA gradient flow (Tier 1)
# ---------------------------------------------------------------------------


class TestQLoRAGradientFlow:
    """Verify gradients flow through quantized base + LoRA adapters."""

    def _setup_qlora_model(self):
        mx.random.seed(42)
        model = TinyLM(vocab_size=128, dim=64, num_layers=2)
        mx.eval(model.parameters())
        manager = LoRAManager()
        lora_config = LoraConfig(rank=4, alpha=8.0, seed=42, train_attn=True, train_mlp=True)
        model = manager.apply_qlora(model, lora_config, quantize_bits=4, quantize_group_size=32)
        return model

    def test_qlora_gradients_nonzero(self):
        """LoRA params should get non-zero gradients after lora_b becomes non-zero.

        At init, lora_b=0 so lora_a gets zero grad (expected). After one step,
        lora_b becomes non-zero and ALL LoRA params should get gradients.
        """
        model = self._setup_qlora_model()
        training = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)

        datum = make_datum(
            [1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0],
            advantages=[0.0, 1.0, 1.0, 1.0],
        )
        # Step 0: lora_b is zero, so lora_a grad is zero — take one optim step
        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="importance_sampling")
        )
        training.optim_step("test", model, OptimStepInput(
            adam_params=AdamParams(learning_rate=1e-3, weight_decay=0.0)
        ))

        # Step 1: now lora_b is non-zero, ALL LoRA params should get gradients
        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="importance_sampling")
        )

        grads = training.accumulated_grads["test"]
        assert grads is not None

        flat_grads = tree_flatten(grads)
        assert len(flat_grads) > 0, "Should have gradients"
        for name, g in flat_grads:
            mx.eval(g)
            assert mx.any(g != 0).item(), f"Gradient for {name} is all zeros after step 1"

    def test_qlora_gradient_magnitude_reasonable(self):
        """LoRA gradient norms should not vanish or explode (after warmup step)."""
        model = self._setup_qlora_model()
        training = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)

        datum = make_datum(
            [1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0],
            advantages=[0.0, 1.0, 1.0, 1.0],
        )
        # Take one step to activate lora_b
        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="importance_sampling")
        )
        training.optim_step("test", model, OptimStepInput(
            adam_params=AdamParams(learning_rate=1e-3, weight_decay=0.0)
        ))

        # Now check gradient magnitudes
        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="importance_sampling")
        )

        grads = training.accumulated_grads["test"]
        for name, g in tree_flatten(grads):
            mx.eval(g)
            norm = mx.sqrt(mx.sum(mx.square(g))).item()
            assert norm < 1e2, (
                f"Gradient norm for {name} is {norm:.6e} — exploding (> 1e2)"
            )

    def test_qlora_base_weights_frozen(self):
        """Gradient tree should only contain LoRA keys (base weights frozen)."""
        model = self._setup_qlora_model()
        training = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)

        datum = make_datum(
            [1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0],
            advantages=[0.0, 1.0, 1.0, 1.0],
        )
        training.forward_backward(
            "test", model, ForwardBackwardInput(data=[datum], loss_fn="importance_sampling")
        )

        grads = training.accumulated_grads["test"]
        for name, _ in tree_flatten(grads):
            assert "lora" in name.lower(), (
                f"Non-LoRA param has gradient: {name}"
            )


# ---------------------------------------------------------------------------
# Per-parameter optimizer accuracy (P2)
# ---------------------------------------------------------------------------


class TestOptimizerPerParameterAccuracy:
    """Verify per-parameter update error stays bounded across optimizer types."""

    def test_multi_step_parameter_error_bounded(self):
        """adamw vs adamw_8bit per-parameter absolute diff should stay bounded over 10 steps.

        8-bit quantization introduces noise in optimizer moments. Near-zero
        parameters cause high relative differences, so we use absolute tolerance.
        """
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_request = OptimStepInput(
            adam_params=AdamParams(learning_rate=1e-3, weight_decay=0.0)
        )
        n_steps = 10

        param_history = {}
        for opt_type in ["adamw", "adamw_8bit"]:
            mx.random.seed(42)
            model = TinyModel(vocab_size=32, dim=16)
            mx.eval(model.parameters())

            training = TrainingBackend(
                optimizer_type=opt_type, gradient_checkpointing=False
            )
            step_params = []

            for _ in range(n_steps):
                training.forward_backward("test", model, fb_request)
                training.optim_step("test", model, opt_request)

                snapshot = {}
                for k, v in tree_flatten(model.parameters()):
                    mx.eval(v)
                    snapshot[k] = np.array(v)
                step_params.append(snapshot)

            param_history[opt_type] = step_params

        for step in range(n_steps):
            fp32_params = param_history["adamw"][step]
            int8_params = param_history["adamw_8bit"][step]

            for key in fp32_params:
                fp32_val = fp32_params[key]
                int8_val = int8_params[key]
                max_abs_diff = np.max(np.abs(fp32_val - int8_val))

                assert max_abs_diff < 0.05, (
                    f"Step {step}, param {key}: "
                    f"max abs diff {max_abs_diff:.6f} > 0.05"
                )
