"""End-to-end integration test using the TinyModel (no real model download required).

Tests the full pipeline: create model -> forward_backward -> optim_step -> sample.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_tinker.backend.training import TrainingBackend
from mlx_tinker.backend.inference import InferenceBackend
from mlx_tinker.types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardInput,
    LossFnInputs,
    ModelInput,
    OptimStepInput,
    SampleInput,
    SamplingParams,
    TensorData,
)


class TinyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, cache=None):
        return self.linear(x), cache


class TinyModel(nn.Module):
    """Minimal model compatible with mlx-lm generate_step."""

    def __init__(self, vocab_size: int = 32, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = [TinyLayer(dim)]
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        h = self.embed(x)
        if cache is not None:
            for i, layer in enumerate(self.layers):
                h, cache[i] = layer(h, cache[i])
        else:
            for layer in self.layers:
                h, _ = layer(h)
        return self.head(h)


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text):
        return [ord(c) % 32 for c in text]

    def decode(self, tokens):
        return "".join(chr((t % 26) + 65) for t in tokens)


def _make_datum(tokens, targets, weights):
    return Datum(
        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=tokens)]),
        loss_fn_inputs=LossFnInputs(
            target_tokens=TensorData(data=targets),
            weights=TensorData(data=weights),
            advantages=TensorData(data=[0.0] * len(targets)),
            logprobs=TensorData(data=[0.0] * len(targets)),
        ),
    )


class TestEndToEnd:
    def test_train_then_sample(self):
        """Full loop: train for 20 steps, verify loss trends down, then sample."""
        model = TinyModel(vocab_size=32, dim=16)
        mx.eval(model.parameters())
        tokenizer = FakeTokenizer()

        training = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)
        inference = InferenceBackend()

        # Train
        losses = []
        for step in range(20):
            datum = _make_datum([1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [0.0, 1.0, 1.0, 1.0, 1.0])
            fb_result = training.forward_backward(
                "test", model, ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
            )
            losses.append(fb_result.loss_fn_outputs[0]["loss"])

            training.optim_step(
                "test",
                model,
                OptimStepInput(adam_params=AdamParams(learning_rate=0.01)),
            )

        # Average of last 5 losses should be lower than average of first 5
        avg_first = sum(losses[:5]) / 5
        avg_last = sum(losses[-5:]) / 5
        assert avg_last < avg_first, (
            f"Loss should trend downward: first-5 avg={avg_first:.4f}, "
            f"last-5 avg={avg_last:.4f}"
        )

        # Sample
        model.eval()
        result = inference.sample(
            model,
            tokenizer,
            SampleInput(
                prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
                sampling_params=SamplingParams(temperature=0.5, max_tokens=10),
                num_samples=2,
            ),
        )

        assert len(result.sequences) == 2
        for seq in result.sequences:
            assert len(seq.tokens) <= 10
            assert len(seq.logprobs) == len(seq.tokens)

    def test_gradient_accumulation_workflow(self):
        """Test accumulating gradients over multiple forward_backward calls."""
        model = TinyModel(vocab_size=32, dim=16)
        mx.eval(model.parameters())

        training = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)

        # Accumulate 3 forward_backward calls
        for _ in range(3):
            datum = _make_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0])
            training.forward_backward(
                "test", model, ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
            )

        assert training.grad_accum_counts["test"] == 3

        # Single optim step uses averaged gradients
        from mlx.utils import tree_flatten
        initial_params = {k: v.tolist() for k, v in tree_flatten(model.parameters())}
        training.optim_step(
            "test",
            model,
            OptimStepInput(adam_params=AdamParams(learning_rate=0.01)),
        )

        # Params should have changed
        new_params = {k: v.tolist() for k, v in tree_flatten(model.parameters())}
        assert initial_params != new_params

        # Accumulators should be cleared
        assert training.accumulated_grads["test"] is None
        assert training.grad_accum_counts["test"] == 0

    def test_rl_workflow_with_tiny_model(self):
        """Test the full RL workflow with a tiny model."""
        model = TinyModel(vocab_size=32, dim=16)
        mx.eval(model.parameters())
        tokenizer = FakeTokenizer()

        training = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)
        inference = InferenceBackend()

        # Sample
        model.eval()
        sample_result = inference.sample(
            model,
            tokenizer,
            SampleInput(
                prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
                sampling_params=SamplingParams(temperature=0.8, max_tokens=5),
                num_samples=2,
            ),
        )

        # Build RL training data
        for seq in sample_result.sequences:
            prompt = [1, 2, 3]
            full = prompt + seq.tokens
            input_tokens = full[:-1]
            target_tokens = full[1:]
            weights = [0.0] * (len(prompt) - 1) + [1.0] * len(seq.tokens)
            weights = weights[: len(target_tokens)]
            advantages = [0.0] * (len(prompt) - 1) + [1.0] * len(seq.tokens)
            advantages = advantages[: len(target_tokens)]
            old_lp = [0.0] * (len(prompt) - 1) + seq.logprobs
            old_lp = old_lp[: len(target_tokens)]

            datum = Datum(
                model_input=ModelInput(chunks=[EncodedTextChunk(tokens=input_tokens)]),
                loss_fn_inputs=LossFnInputs(
                    target_tokens=TensorData(data=target_tokens),
                    weights=TensorData(data=weights),
                    advantages=TensorData(data=advantages),
                    logprobs=TensorData(data=old_lp),
                ),
            )

            model.train()
            training.forward_backward(
                "test",
                model,
                ForwardBackwardInput(data=[datum], loss_fn="importance_sampling"),
            )

        # Optim step
        result = training.optim_step(
            "test",
            model,
            OptimStepInput(adam_params=AdamParams(learning_rate=0.001)),
        )

        assert result.metrics is not None
        assert result.metrics["grad_accum_steps"] == 2
