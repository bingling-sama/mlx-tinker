"""Cookbook verification: Supervised Fine-Tuning (SFT).

Replicates the tinker-cookbook/recipes/sl_basic.py pattern:
  1. Create model with QLoRA
  2. forward_backward with cross_entropy on chat data
  3. optim_step
  4. Verify loss decreases over 5 steps
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.config import EngineConfig
from mlx_tinker.types import (
    AdamParams,
    CreateModelInput,
    Datum,
    EncodedTextChunk,
    ForwardBackwardInput,
    LossFnInputs,
    LoraConfig,
    ModelInput,
    OptimStepInput,
    TensorData,
)

pytestmark = pytest.mark.cookbook

NUM_STEPS = 5


@pytest.fixture
def backend(model_name, tmp_path):
    config = EngineConfig(
        base_model=model_name,
        quantize_bits=4,
        quantize_group_size=64,
        checkpoints_base=tmp_path / "checkpoints",
    )
    return MLXBackend(config)


def _make_sft_datum(tokenizer, prompt: str, completion: str) -> Datum:
    """Create an SFT datum: mask prompt tokens (weight=0), train on completion (weight=1)."""
    prompt_tokens = tokenizer.encode(prompt)
    completion_tokens = tokenizer.encode(completion)
    all_tokens = prompt_tokens + completion_tokens

    # Target is shifted by 1
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)

    # Pad weights to match
    weights = weights[: len(target_tokens)]

    return Datum(
        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=input_tokens)]),
        loss_fn_inputs=LossFnInputs(
            target_tokens=TensorData(data=target_tokens),
            weights=TensorData(data=weights),
            advantages=TensorData(data=[0.0] * len(target_tokens)),
            logprobs=TensorData(data=[0.0] * len(target_tokens)),
        ),
    )


SFT_EXAMPLES = [
    ("What is the capital of France?", " The capital of France is Paris."),
    ("Explain photosynthesis.", " Photosynthesis is the process by which plants convert sunlight."),
    ("What is 2+2?", " 2+2 equals 4."),
    ("Name a programming language.", " Python is a popular programming language."),
    ("What is gravity?", " Gravity is a fundamental force that attracts objects with mass."),
]


class TestSFTWorkflow:
    def test_loss_decreases_over_steps(self, backend, model_name):
        """Full SFT workflow: create model, train 5 steps, verify loss decreases."""
        # Create model
        lora_config = LoraConfig(rank=8, alpha=16.0, train_attn=True, train_mlp=True)
        result = backend.create_model("sft-test", CreateModelInput(lora_config=lora_config))
        assert result.model_id == "sft-test"

        tokenizer = backend.tokenizers["sft-test"]

        # Train
        losses = []
        for step in range(NUM_STEPS):
            prompt, completion = SFT_EXAMPLES[step % len(SFT_EXAMPLES)]
            datum = _make_sft_datum(tokenizer, prompt, completion)

            # forward_backward
            fb_result = backend.forward_backward(
                "sft-test",
                ForwardBackwardInput(data=[datum], loss_fn="cross_entropy"),
            )
            step_loss = fb_result.loss_fn_outputs[0]["loss"]
            losses.append(step_loss)

            # optim_step
            backend.optim_step(
                "sft-test",
                OptimStepInput(adam_params=AdamParams(learning_rate=1e-4)),
            )

        print(f"\n=== SFT Cookbook Test ===")
        print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
        print(f"  First: {losses[0]:.4f}, Last: {losses[-1]:.4f}")

        assert losses[-1] < losses[0], f"Loss should decrease: {losses}"

    def test_multiple_data_per_step(self, backend, model_name):
        """Test batching multiple SFT examples in a single forward_backward call."""
        lora_config = LoraConfig(rank=8, alpha=16.0)
        backend.create_model("sft-batch", CreateModelInput(lora_config=lora_config))
        tokenizer = backend.tokenizers["sft-batch"]

        data = [
            _make_sft_datum(tokenizer, p, c) for p, c in SFT_EXAMPLES[:3]
        ]

        result = backend.forward_backward(
            "sft-batch",
            ForwardBackwardInput(data=data, loss_fn="cross_entropy"),
        )

        assert len(result.loss_fn_outputs) == 3
        assert all("loss" in o for o in result.loss_fn_outputs)
        assert result.metrics["num_sequences"] == 3
