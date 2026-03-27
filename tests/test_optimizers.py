"""Tests for 8-bit AdamW optimizer and quantization utilities."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_tinker.backend.optimizers import AdamW8Bit, _quantize_blockwise, _dequantize_blockwise
from mlx_tinker.backend.training import TrainingBackend
from mlx_tinker.types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardInput,
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

        assert quantized.dtype == mx.int8
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

        datum = _make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_request = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_request = OptimStepInput(
            adam_params=AdamParams(learning_rate=1e-3, weight_decay=0.0)
        )

        losses = []
        for _ in range(30):
            result = training.forward_backward("test", model, fb_request)
            losses.append(result.loss_fn_outputs[0]["loss"])
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
        datum = _make_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])
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
                    assert s["m"].dtype == mx.int8, f"Expected m to be int8, got {s['m'].dtype}"
                    assert s["v"].dtype == mx.int8, f"Expected v to be int8, got {s['v'].dtype}"
                    return True
                return any(_check_state(v) for v in s.values())
            elif isinstance(s, (list, tuple)):
                return any(_check_state(v) for v in s)
            return False

        found = _check_state(state)
        assert found, "Should find int8 m and v in optimizer state"

    def test_matches_fp32_direction(self):
        """Both adamw_8bit and adamw should reduce loss over 10 steps."""
        datum = _make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
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
                losses.append(result.loss_fn_outputs[0]["loss"])
                training.optim_step("test", model, opt_request)

            results[opt_type] = losses

        # Both should reduce loss
        for opt_type, losses in results.items():
            assert losses[-1] < losses[0], (
                f"{opt_type} should reduce loss: first={losses[0]:.4f}, "
                f"last={losses[-1]:.4f}"
            )
