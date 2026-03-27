"""Cookbook verification: Tool Use with RL.

Replicates the tinker-cookbook/search_tool/ pattern:
  1. Model generates a response that may include tool calls
  2. Environment executes tool calls and returns results
  3. Model continues generation with tool results
  4. Compute rewards based on final answer quality
  5. Train with importance_sampling loss on full trajectory

This tests the multi-turn RL loop with tool-augmented trajectories.
"""

from __future__ import annotations

import json
import re

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


# Simulated tool environment
TOOL_REGISTRY = {
    "calculator": lambda expr: str(eval(expr)),  # Simple calculator
    "lookup": lambda query: f"Result for '{query}': 42",  # Mock lookup
}

TOOL_CALL_MARKER = "<tool_call>"
TOOL_RESULT_MARKER = "<tool_result>"
END_MARKER = "<end>"


def _detect_tool_call(text: str) -> tuple[str, str] | None:
    """Detect if text contains a tool call pattern."""
    # Pattern: <tool_call>tool_name(arg)</tool_call>
    match = re.search(r"<tool_call>(\w+)\((.*?)\)</tool_call>", text)
    if match:
        return match.group(1), match.group(2)
    return None


def _execute_tool(tool_name: str, tool_arg: str) -> str:
    """Execute a tool and return result string."""
    if tool_name in TOOL_REGISTRY:
        try:
            return TOOL_REGISTRY[tool_name](tool_arg)
        except Exception as e:
            return f"Error: {e}"
    return f"Unknown tool: {tool_name}"


def _reward_fn(final_text: str, expected_contains: str) -> float:
    """Reward based on whether the final answer contains expected content."""
    if expected_contains.lower() in final_text.lower():
        return 1.0
    return 0.0


@pytest.fixture
def backend(model_name, tmp_path):
    config = EngineConfig(
        base_model=model_name,
        quantize_bits=4,
        quantize_group_size=64,
        checkpoints_base=tmp_path / "checkpoints",
    )
    return MLXBackend(config)


TOOL_USE_EXAMPLES = [
    {
        "system": "You have access to tools: calculator(expr), lookup(query). Use <tool_call>tool(arg)</tool_call> to call them.",
        "user": "What is 15 * 23?",
        "expected_contains": "345",
    },
    {
        "system": "You have access to tools: calculator(expr), lookup(query).",
        "user": "Look up the answer to life.",
        "expected_contains": "42",
    },
]


class TestToolUseWorkflow:
    def test_tool_use_rl_loop(self, backend, model_name):
        """Full tool-use RL loop: generate -> tool call -> continue -> reward -> train."""
        lora_config = LoraConfig(rank=8, alpha=16.0, train_attn=True, train_mlp=True)
        backend.create_model("tool-test", CreateModelInput(lora_config=lora_config))
        tokenizer = backend.tokenizers["tool-test"]

        example = TOOL_USE_EXAMPLES[0]

        # Build initial prompt
        prompt_text = f"System: {example['system']}\nUser: {example['user']}\nAssistant:"
        prompt_tokens = tokenizer.encode(prompt_text)

        # === Turn 1: Generate initial response (may contain tool call) ===
        sample_result = backend.sample(
            "tool-test",
            SampleInput(
                prompt=ModelInput(chunks=[EncodedTextChunk(tokens=prompt_tokens)]),
                sampling_params=SamplingParams(temperature=0.7, max_tokens=50),
                num_samples=1,
            ),
        )

        seq = sample_result.sequences[0]
        turn1_text = tokenizer.decode(seq.tokens)
        turn1_tokens = seq.tokens
        turn1_logprobs = seq.logprobs

        # === Check for tool call (simulated) ===
        tool_call = _detect_tool_call(turn1_text)
        all_tokens = prompt_tokens + turn1_tokens
        all_logprobs = [0.0] * len(prompt_tokens) + turn1_logprobs

        if tool_call:
            tool_name, tool_arg = tool_call
            tool_result = _execute_tool(tool_name, tool_arg)

            # Append tool result to context
            tool_result_text = f"\n{TOOL_RESULT_MARKER}{tool_result}{END_MARKER}\nAssistant:"
            tool_result_tokens = tokenizer.encode(tool_result_text)
            all_tokens = all_tokens + tool_result_tokens

            # === Turn 2: Generate continuation after tool result ===
            sample_result2 = backend.sample(
                "tool-test",
                SampleInput(
                    prompt=ModelInput(chunks=[EncodedTextChunk(tokens=all_tokens)]),
                    sampling_params=SamplingParams(temperature=0.7, max_tokens=30),
                    num_samples=1,
                ),
            )

            seq2 = sample_result2.sequences[0]
            all_tokens = all_tokens + seq2.tokens
            # Pad logprobs for tool result tokens (not from model)
            all_logprobs = all_logprobs + [0.0] * len(tool_result_tokens) + seq2.logprobs

        # === Compute reward ===
        final_text = tokenizer.decode(all_tokens)
        reward = _reward_fn(final_text, example["expected_contains"])

        # === Build training datum from full trajectory ===
        input_tokens = all_tokens[:-1]
        target_tokens = all_tokens[1:]

        # Weight: 0 for prompt, 1 for model-generated tokens
        n_prompt = len(prompt_tokens) - 1
        weights = [0.0] * n_prompt + [1.0] * (len(target_tokens) - n_prompt)
        weights = weights[: len(target_tokens)]

        # Advantage = reward (simple, no baseline)
        advantages = [0.0] * n_prompt + [reward] * (len(target_tokens) - n_prompt)
        advantages = advantages[: len(target_tokens)]

        # Old log probs
        old_lp = all_logprobs[1:]  # Shift to align with targets
        old_lp = old_lp[: len(target_tokens)]
        # Pad if needed
        while len(old_lp) < len(target_tokens):
            old_lp.append(0.0)

        datum = Datum(
            model_input=ModelInput(chunks=[EncodedTextChunk(tokens=input_tokens)]),
            loss_fn_inputs=LossFnInputs(
                target_tokens=TensorData(data=target_tokens),
                weights=TensorData(data=weights),
                advantages=TensorData(data=advantages),
                logprobs=TensorData(data=old_lp),
            ),
        )

        # === Train on trajectory ===
        fb_result = backend.forward_backward(
            "tool-test",
            ForwardBackwardInput(data=[datum], loss_fn="importance_sampling"),
        )

        backend.optim_step(
            "tool-test",
            OptimStepInput(adam_params=AdamParams(learning_rate=5e-5)),
        )

        print(f"\n=== Tool Use Cookbook Test ===")
        print(f"  Tool call detected: {tool_call is not None}")
        print(f"  Reward: {reward}")
        print(f"  Loss: {fb_result.metrics['mean_loss']:.4f}")
        print(f"  Trajectory length: {len(all_tokens)} tokens")

        # The loop should complete without errors
        assert fb_result.loss_fn_output_type == "importance_sampling"

    def test_multi_tool_multi_step(self, backend, model_name):
        """Test training on multiple tool-use trajectories over multiple steps."""
        lora_config = LoraConfig(rank=8, alpha=16.0)
        backend.create_model("tool-multi", CreateModelInput(lora_config=lora_config))
        tokenizer = backend.tokenizers["tool-multi"]

        losses = []
        for example in TOOL_USE_EXAMPLES:
            prompt_text = f"System: {example['system']}\nUser: {example['user']}\nAssistant:"
            prompt_tokens = tokenizer.encode(prompt_text)

            sample_result = backend.sample(
                "tool-multi",
                SampleInput(
                    prompt=ModelInput(chunks=[EncodedTextChunk(tokens=prompt_tokens)]),
                    sampling_params=SamplingParams(temperature=0.7, max_tokens=30),
                    num_samples=1,
                ),
            )

            seq = sample_result.sequences[0]
            all_tokens = prompt_tokens + seq.tokens

            reward = _reward_fn(tokenizer.decode(all_tokens), example["expected_contains"])

            input_tokens = all_tokens[:-1]
            target_tokens = all_tokens[1:]
            n_prompt = len(prompt_tokens) - 1
            weights = [0.0] * n_prompt + [1.0] * (len(target_tokens) - n_prompt)
            weights = weights[: len(target_tokens)]
            advantages = [0.0] * n_prompt + [reward] * (len(target_tokens) - n_prompt)
            advantages = advantages[: len(target_tokens)]
            old_lp = [0.0] * n_prompt + seq.logprobs
            old_lp = old_lp[: len(target_tokens)]
            while len(old_lp) < len(target_tokens):
                old_lp.append(0.0)

            datum = Datum(
                model_input=ModelInput(chunks=[EncodedTextChunk(tokens=input_tokens)]),
                loss_fn_inputs=LossFnInputs(
                    target_tokens=TensorData(data=target_tokens),
                    weights=TensorData(data=weights),
                    advantages=TensorData(data=advantages),
                    logprobs=TensorData(data=old_lp),
                ),
            )

            fb_result = backend.forward_backward(
                "tool-multi",
                ForwardBackwardInput(data=[datum], loss_fn="importance_sampling"),
            )
            losses.append(fb_result.metrics["mean_loss"])

            backend.optim_step(
                "tool-multi",
                OptimStepInput(adam_params=AdamParams(learning_rate=5e-5)),
            )

        print(f"\n=== Multi-Tool Cookbook Test ===")
        print(f"  Losses: {[f'{l:.4f}' for l in losses]}")

        assert len(losses) == len(TOOL_USE_EXAMPLES)
