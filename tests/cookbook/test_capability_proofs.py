"""Capability proofs for SFT and GRPO-style RL on a tiny exact-match task.

These tests are intentionally closer to the official tinker-cookbook loops than
the current WikiSQL cookbook checks:

- SFT is judged by capability improvement, not just loss decrease.
- RL uses grouped rollouts with reward centering and proves post-train reward /
  accuracy improvement on the same task family.

The task is a small arbitrary codeword->label mapping. That keeps the action
space tiny enough for fast local verification while still requiring actual
learning, since the mapping is not in the base model's pretraining data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.config import EngineConfig
from mlx_tinker.types import (
    AdamParams,
    CreateModelInput,
    Datum,
    EncodedTextChunk,
    ForwardBackwardInput,
    LoraConfig,
    LossFnInputs,
    ModelInput,
    OptimStepInput,
    SampleInput,
    SamplingParams,
    TensorData,
)

pytestmark = pytest.mark.cookbook

SFT_PROOF_STEPS = 48
PIPELINE_SFT_STEPS = 12
RL_WARMSTART_STEPS = 12
RL_PROOF_STEPS = 32
GROUP_SIZE = 8
LABELS = ("YES", "NO")


@dataclass(frozen=True)
class LabelExample:
    codeword: str
    label: str


MAPPING = (
    LabelExample("zib", "YES"),
    LabelExample("wug", "NO"),
    LabelExample("dax", "YES"),
    LabelExample("blicket", "NO"),
)

TRAIN_TEMPLATES = (
    "Reply with YES or NO only.\nQ: {codeword}\nA:",
    "Return YES or NO.\nCodeword={codeword}\nLabel:",
)
EVAL_TEMPLATES = (
    "Reply with YES or NO only.\nQ: {codeword}\nA:",
    "Return YES or NO.\nCodeword={codeword}\nLabel:",
)


@pytest.fixture
def backend(model_name, tmp_path):
    config = EngineConfig(
        base_model=model_name,
        quantize_bits=4,
        quantize_group_size=64,
        checkpoints_base=tmp_path / "checkpoints",
    )
    return MLXBackend(config)


def _train_examples() -> list[tuple[str, str]]:
    return [
        (template.format(codeword=ex.codeword), ex.label)
        for ex in MAPPING
        for template in TRAIN_TEMPLATES
    ]


def _eval_examples() -> list[tuple[str, str]]:
    return [
        (template.format(codeword=ex.codeword), ex.label)
        for ex in MAPPING
        for template in EVAL_TEMPLATES
    ]


def _extract_label(text: str) -> str | None:
    match = re.search(r"\b(YES|NO)\b", text.upper())
    if match:
        return match.group(1)
    compact = re.sub(r"\s+", "", text.upper())
    if compact.startswith("YES"):
        return "YES"
    if compact.startswith("NO"):
        return "NO"
    return None


def _make_sft_datum(tokenizer, prompt: str, label: str) -> Datum:
    prompt_tokens = tokenizer.encode(prompt)
    completion_tokens = tokenizer.encode(" " + label)
    all_tokens = prompt_tokens + completion_tokens
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    n_prompt = len(prompt_tokens) - 1
    weights = [0.0] * n_prompt + [1.0] * len(completion_tokens)
    weights = weights[: len(target_tokens)]
    zeros = [0.0] * len(target_tokens)
    return Datum(
        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=input_tokens)]),
        loss_fn_inputs=LossFnInputs(
            target_tokens=TensorData(data=target_tokens),
            weights=TensorData(data=weights),
            advantages=TensorData(data=zeros),
            logprobs=TensorData(data=zeros),
        ),
    )


def _make_rl_datum(tokenizer, prompt: str, sequence, advantage: float) -> Datum:
    prompt_tokens = tokenizer.encode(prompt)
    full_tokens = prompt_tokens + sequence.tokens
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    n_prompt = len(prompt_tokens) - 1
    n_gen = len(sequence.tokens)

    weights = [0.0] * n_prompt + [1.0] * n_gen
    weights = weights[: len(target_tokens)]
    advantages = [0.0] * n_prompt + [advantage] * n_gen
    advantages = advantages[: len(target_tokens)]
    old_logprobs = [0.0] * n_prompt + list((sequence.logprobs or [])[:n_gen])
    old_logprobs = old_logprobs[: len(target_tokens)]
    if len(old_logprobs) < len(target_tokens):
        old_logprobs.extend([0.0] * (len(target_tokens) - len(old_logprobs)))

    return Datum(
        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=input_tokens)]),
        loss_fn_inputs=LossFnInputs(
            target_tokens=TensorData(data=target_tokens),
            weights=TensorData(data=weights),
            advantages=TensorData(data=advantages),
            logprobs=TensorData(data=old_logprobs),
        ),
    )


def _sample_label(backend: MLXBackend, model_id: str, tokenizer, prompt: str, *, seed: int) -> str | None:
    prompt_tokens = tokenizer.encode(prompt)
    result = backend.sample(
        model_id,
        SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=prompt_tokens)]),
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=4,
                stop=["\n"],
                seed=seed,
            ),
            num_samples=1,
            prompt_logprobs=False,
        ),
    )
    text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
    return _extract_label(text)


def _evaluate_accuracy(
    backend: MLXBackend,
    model_id: str,
    tokenizer,
    examples: list[tuple[str, str]],
) -> float:
    correct = 0
    for idx, (prompt, label) in enumerate(examples):
        if _sample_label(backend, model_id, tokenizer, prompt, seed=idx) == label:
            correct += 1
    return correct / len(examples)


def _evaluate_mean_reward(
    backend: MLXBackend,
    model_id: str,
    tokenizer,
    examples: list[tuple[str, str]],
    *,
    group_size: int,
) -> float:
    total_reward = 0.0
    total_samples = 0
    for idx, (prompt, label) in enumerate(examples):
        prompt_tokens = tokenizer.encode(prompt)
        sample_result = backend.sample(
            model_id,
            SampleInput(
                prompt=ModelInput(chunks=[EncodedTextChunk(tokens=prompt_tokens)]),
                sampling_params=SamplingParams(
                    temperature=1.0,
                    max_tokens=4,
                    stop=["\n"],
                    seed=10_000 + idx,
                ),
                num_samples=group_size,
                prompt_logprobs=False,
            ),
        )
        for sequence in sample_result.sequences:
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            total_reward += _reward_for_label(text, label)
            total_samples += 1
    return total_reward / total_samples


def _run_sft_steps(
    backend: MLXBackend,
    model_id: str,
    tokenizer,
    examples: list[tuple[str, str]],
    *,
    num_steps: int,
    learning_rate: float,
) -> list[float]:
    losses: list[float] = []
    for step in range(num_steps):
        prompt, label = examples[step % len(examples)]
        datum = _make_sft_datum(tokenizer, prompt, label)
        fb_result = backend.forward_backward(
            model_id,
            ForwardBackwardInput(data=[datum], loss_fn="cross_entropy"),
        )
        losses.append(float(fb_result.metrics["loss:sum"]))
        backend.optim_step(
            model_id,
            OptimStepInput(adam_params=AdamParams(learning_rate=learning_rate)),
        )
    return losses


def _reward_for_label(text: str, label: str) -> float:
    return 1.0 if _extract_label(text) == label else 0.0


def _run_rl_steps(
    backend: MLXBackend,
    model_id: str,
    tokenizer,
    examples: list[tuple[str, str]],
    *,
    num_steps: int,
    learning_rate: float,
    group_size: int,
) -> tuple[list[float], int]:
    mean_rewards: list[float] = []
    non_zero_advantage_steps = 0

    for step in range(num_steps):
        prompt, gold_label = examples[step % len(examples)]
        prompt_tokens = tokenizer.encode(prompt)
        sample_result = backend.sample(
            model_id,
            SampleInput(
                prompt=ModelInput(chunks=[EncodedTextChunk(tokens=prompt_tokens)]),
                sampling_params=SamplingParams(
                    temperature=1.0,
                    max_tokens=4,
                    stop=["\n"],
                    seed=step,
                ),
                num_samples=group_size,
                prompt_logprobs=False,
            ),
        )

        rewards = []
        datums = []
        for sequence in sample_result.sequences:
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            rewards.append(_reward_for_label(text, gold_label))

        mean_reward = sum(rewards) / len(rewards)
        advantages = [reward - mean_reward for reward in rewards]
        mean_rewards.append(mean_reward)
        if any(abs(advantage) > 1e-6 for advantage in advantages):
            non_zero_advantage_steps += 1
        else:
            continue

        for sequence, advantage in zip(sample_result.sequences, advantages, strict=True):
            datums.append(_make_rl_datum(tokenizer, prompt, sequence, advantage))

        fb_result = backend.forward_backward(
            model_id,
            ForwardBackwardInput(data=datums, loss_fn="importance_sampling"),
        )
        assert isinstance(fb_result.metrics["loss:sum"], float)
        backend.optim_step(
            model_id,
            OptimStepInput(adam_params=AdamParams(learning_rate=learning_rate)),
        )

    return mean_rewards, non_zero_advantage_steps


class TestCapabilityProofs:
    def test_base_then_sft_then_rl_progression(self, backend, model_name):
        train_examples = _train_examples()
        eval_examples = _eval_examples()

        backend.create_model(
            "capability-pipeline",
            CreateModelInput(
                lora_config=LoraConfig(rank=8, alpha=16.0, seed=2, train_attn=True, train_mlp=True)
            ),
        )
        tokenizer = backend.tokenizers["capability-pipeline"]

        base_accuracy = _evaluate_accuracy(backend, "capability-pipeline", tokenizer, eval_examples)
        base_reward = _evaluate_mean_reward(
            backend,
            "capability-pipeline",
            tokenizer,
            eval_examples,
            group_size=GROUP_SIZE,
        )

        sft_losses = _run_sft_steps(
            backend,
            "capability-pipeline",
            tokenizer,
            train_examples,
            num_steps=PIPELINE_SFT_STEPS,
            learning_rate=2e-4,
        )
        sft_accuracy = _evaluate_accuracy(backend, "capability-pipeline", tokenizer, eval_examples)
        sft_reward = _evaluate_mean_reward(
            backend,
            "capability-pipeline",
            tokenizer,
            eval_examples,
            group_size=GROUP_SIZE,
        )

        rl_mean_rewards, non_zero_advantage_steps = _run_rl_steps(
            backend,
            "capability-pipeline",
            tokenizer,
            train_examples,
            num_steps=RL_PROOF_STEPS,
            learning_rate=1e-4,
            group_size=GROUP_SIZE,
        )
        rl_accuracy = _evaluate_accuracy(backend, "capability-pipeline", tokenizer, eval_examples)
        rl_reward = _evaluate_mean_reward(
            backend,
            "capability-pipeline",
            tokenizer,
            eval_examples,
            group_size=GROUP_SIZE,
        )

        print("\n=== Base -> SFT -> RL Progression ===")
        print(f"  Base accuracy: {base_accuracy:.3f}")
        print(f"  SFT accuracy:  {sft_accuracy:.3f}")
        print(f"  RL accuracy:   {rl_accuracy:.3f}")
        print(f"  Base reward: {base_reward:.3f}")
        print(f"  SFT reward:  {sft_reward:.3f}")
        print(f"  RL reward:   {rl_reward:.3f}")
        print(f"  SFT loss: {sft_losses[0]:.4f} -> {sft_losses[-1]:.4f}")
        print(f"  RL mean reward during train: {sum(rl_mean_rewards) / len(rl_mean_rewards):.3f}")
        print(f"  RL non-zero advantage steps: {non_zero_advantage_steps}/{RL_PROOF_STEPS}")

        assert sft_accuracy >= base_accuracy + 0.30
        assert sft_reward >= base_reward + 0.30
        assert sft_losses[-1] < sft_losses[0]
        assert non_zero_advantage_steps >= RL_PROOF_STEPS // 4
        assert rl_reward >= sft_reward + 0.01
        assert rl_accuracy >= sft_accuracy - 1e-6

    def test_sft_improves_exact_match_accuracy(self, backend, model_name):
        train_examples = _train_examples()
        eval_examples = _eval_examples()

        backend.create_model(
            "capability-sft",
            CreateModelInput(
                lora_config=LoraConfig(rank=8, alpha=16.0, seed=0, train_attn=True, train_mlp=True)
            ),
        )
        tokenizer = backend.tokenizers["capability-sft"]

        base_accuracy = _evaluate_accuracy(backend, "capability-sft", tokenizer, eval_examples)
        losses = _run_sft_steps(
            backend,
            "capability-sft",
            tokenizer,
            train_examples,
            num_steps=SFT_PROOF_STEPS,
            learning_rate=2e-4,
        )
        tuned_accuracy = _evaluate_accuracy(backend, "capability-sft", tokenizer, eval_examples)

        print("\n=== SFT Capability Proof ===")
        print(f"  Base accuracy:  {base_accuracy:.3f}")
        print(f"  Tuned accuracy: {tuned_accuracy:.3f}")
        print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

        assert tuned_accuracy >= base_accuracy + 0.30, (
            f"SFT should improve exact-match accuracy: {base_accuracy:.3f} -> {tuned_accuracy:.3f}"
        )
        assert tuned_accuracy >= 0.75, f"SFT final accuracy too low: {tuned_accuracy:.3f}"
        assert losses[-1] < losses[0], "SFT loss should decrease during training"

    def test_grpo_style_rl_improves_exact_match_accuracy(self, backend, model_name):
        train_examples = _train_examples()
        eval_examples = _eval_examples()

        backend.create_model(
            "capability-rl",
            CreateModelInput(
                lora_config=LoraConfig(rank=8, alpha=16.0, seed=1, train_attn=True, train_mlp=True)
            ),
        )
        tokenizer = backend.tokenizers["capability-rl"]

        # Small warm-start so the model occasionally emits the correct label and
        # the RL reward signal is not completely sparse.
        _run_sft_steps(
            backend,
            "capability-rl",
            tokenizer,
            train_examples,
            num_steps=RL_WARMSTART_STEPS,
            learning_rate=2e-4,
        )
        pre_rl_accuracy = _evaluate_accuracy(backend, "capability-rl", tokenizer, eval_examples)
        pre_rl_reward = _evaluate_mean_reward(
            backend,
            "capability-rl",
            tokenizer,
            eval_examples,
            group_size=GROUP_SIZE,
        )

        mean_rewards, non_zero_advantage_steps = _run_rl_steps(
            backend,
            "capability-rl",
            tokenizer,
            train_examples,
            num_steps=RL_PROOF_STEPS,
            learning_rate=1e-4,
            group_size=GROUP_SIZE,
        )

        post_rl_accuracy = _evaluate_accuracy(backend, "capability-rl", tokenizer, eval_examples)
        post_rl_reward = _evaluate_mean_reward(
            backend,
            "capability-rl",
            tokenizer,
            eval_examples,
            group_size=GROUP_SIZE,
        )

        print("\n=== RL Capability Proof ===")
        print(f"  Pre-RL accuracy:  {pre_rl_accuracy:.3f}")
        print(f"  Post-RL accuracy: {post_rl_accuracy:.3f}")
        print(f"  Pre-RL sampled reward:  {pre_rl_reward:.3f}")
        print(f"  Post-RL sampled reward: {post_rl_reward:.3f}")
        print(f"  Train-time mean reward: {sum(mean_rewards) / len(mean_rewards):.3f}")
        print(f"  Non-zero advantage steps: {non_zero_advantage_steps}/{RL_PROOF_STEPS}")

        assert non_zero_advantage_steps >= RL_PROOF_STEPS // 4, (
            f"Expected many non-zero advantage steps, got {non_zero_advantage_steps}"
        )
        assert post_rl_reward >= pre_rl_reward + 0.10, (
            f"RL should improve sampled reward: {pre_rl_reward:.3f} -> {post_rl_reward:.3f}"
        )
        assert post_rl_accuracy >= pre_rl_accuracy - 1e-6, (
            f"RL should not regress greedy accuracy: {pre_rl_accuracy:.3f} -> {post_rl_accuracy:.3f}"
        )
