"""Cookbook verification: Reinforcement Learning (RL).

Replicates the tinker-cookbook/recipes/rl_basic.py pattern:
  1. Create model with QLoRA
  2. Sample rollouts from the model
  3. Compute rewards and advantages
  4. forward_backward with importance_sampling loss
  5. optim_step
  6. Verify the RL loop completes without errors and loss changes
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
    SampleInput,
    SamplingParams,
    TensorData,
)

pytestmark = pytest.mark.cookbook

NUM_RL_STEPS = 3


@pytest.fixture
def backend(model_name, tmp_path):
    config = EngineConfig(
        base_model=model_name,
        quantize_bits=4,
        quantize_group_size=64,
        checkpoints_base=tmp_path / "checkpoints",
    )
    return MLXBackend(config)


def _simple_reward(tokens: list[int], tokenizer) -> float:
    """Simple reward function: longer responses get higher reward."""
    text = tokenizer.decode(tokens)
    # Reward based on length (normalized)
    return min(len(text) / 100.0, 1.0)


def _compute_advantages(rewards: list[float]) -> list[float]:
    """Simple advantage computation: reward - mean(rewards)."""
    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    return [r - mean_r for r in rewards]


class TestRLWorkflow:
    def test_rl_loop(self, backend, model_name):
        """Full RL workflow: sample -> reward -> forward_backward(IS) -> optim_step."""
        lora_config = LoraConfig(rank=8, alpha=16.0, train_attn=True, train_mlp=True)
        backend.create_model("rl-test", CreateModelInput(lora_config=lora_config))
        tokenizer = backend.tokenizers["rl-test"]

        prompts = [
            "Solve: What is 15 + 27?",
            "Explain why the sky is blue.",
            "Write a haiku about coding.",
        ]

        rl_losses = []

        for step in range(NUM_RL_STEPS):
            prompt_text = prompts[step % len(prompts)]
            prompt_tokens = tokenizer.encode(prompt_text)

            # 1. Sample rollouts
            sample_result = backend.sample(
                "rl-test",
                SampleInput(
                    prompt=ModelInput(chunks=[EncodedTextChunk(tokens=prompt_tokens)]),
                    sampling_params=SamplingParams(temperature=0.8, max_tokens=32),
                    num_samples=2,
                    prompt_logprobs=False,
                ),
            )

            assert len(sample_result.sequences) == 2

            # 2. Compute rewards
            rewards = []
            for seq in sample_result.sequences:
                r = _simple_reward(seq.tokens, tokenizer)
                rewards.append(r)

            advantages = _compute_advantages(rewards)

            # 3. Build training data from rollouts
            data = []
            for i, seq in enumerate(sample_result.sequences):
                full_tokens = prompt_tokens + seq.tokens
                input_tokens = full_tokens[:-1]
                target_tokens = full_tokens[1:]

                # Mask prompt tokens (weight=0), train on generated tokens (weight=1)
                n_prompt = len(prompt_tokens) - 1
                n_gen = len(seq.tokens)
                weights = [0.0] * n_prompt + [1.0] * n_gen
                weights = weights[: len(target_tokens)]

                # Per-token advantage (broadcast sequence advantage)
                adv = [0.0] * n_prompt + [advantages[i]] * n_gen
                adv = adv[: len(target_tokens)]

                # Old log probs from sampling
                old_lp = [0.0] * n_prompt + seq.logprobs[: n_gen]
                old_lp = old_lp[: len(target_tokens)]

                data.append(
                    Datum(
                        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=input_tokens)]),
                        loss_fn_inputs=LossFnInputs(
                            target_tokens=TensorData(data=target_tokens),
                            weights=TensorData(data=weights),
                            advantages=TensorData(data=adv),
                            logprobs=TensorData(data=old_lp),
                        ),
                    )
                )

            # 4. forward_backward with importance sampling
            fb_result = backend.forward_backward(
                "rl-test",
                ForwardBackwardInput(data=data, loss_fn="importance_sampling"),
            )
            rl_losses.append(fb_result.metrics["mean_loss"])

            # 5. optim_step
            backend.optim_step(
                "rl-test",
                OptimStepInput(adam_params=AdamParams(learning_rate=5e-5)),
            )

        print(f"\n=== RL Cookbook Test ===")
        print(f"  RL losses: {[f'{l:.4f}' for l in rl_losses]}")

        # RL loss should change (not necessarily monotonically decrease)
        assert len(rl_losses) == NUM_RL_STEPS
        # At minimum, the RL loop should complete without errors
        # and produce non-trivial loss values
        assert all(isinstance(l, float) for l in rl_losses), "All losses should be floats"
        assert any(l != rl_losses[0] for l in rl_losses), "Loss should change across RL steps"

    def test_ppo_loss(self, backend, model_name):
        """Test PPO loss variant for RL."""
        lora_config = LoraConfig(rank=8, alpha=16.0)
        backend.create_model("ppo-test", CreateModelInput(lora_config=lora_config))
        tokenizer = backend.tokenizers["ppo-test"]

        prompt_tokens = tokenizer.encode("What is 1+1?")
        sample_result = backend.sample(
            "ppo-test",
            SampleInput(
                prompt=ModelInput(chunks=[EncodedTextChunk(tokens=prompt_tokens)]),
                sampling_params=SamplingParams(temperature=0.8, max_tokens=16),
                num_samples=1,
            ),
        )

        seq = sample_result.sequences[0]
        full_tokens = prompt_tokens + seq.tokens
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]
        n_prompt = len(prompt_tokens) - 1
        n_gen = len(seq.tokens)
        weights = [0.0] * n_prompt + [1.0] * n_gen
        weights = weights[: len(target_tokens)]
        adv = [0.0] * n_prompt + [1.0] * n_gen
        adv = adv[: len(target_tokens)]
        old_lp = [0.0] * n_prompt + seq.logprobs[: n_gen]
        old_lp = old_lp[: len(target_tokens)]

        datum = Datum(
            model_input=ModelInput(chunks=[EncodedTextChunk(tokens=input_tokens)]),
            loss_fn_inputs=LossFnInputs(
                target_tokens=TensorData(data=target_tokens),
                weights=TensorData(data=weights),
                advantages=TensorData(data=adv),
                logprobs=TensorData(data=old_lp),
            ),
        )

        result = backend.forward_backward(
            "ppo-test",
            ForwardBackwardInput(
                data=[datum],
                loss_fn="ppo",
                loss_fn_config={"clip_high_threshold": 0.2},
            ),
        )

        assert result.loss_fn_output_type == "ppo"
        assert len(result.loss_fn_outputs) == 1
