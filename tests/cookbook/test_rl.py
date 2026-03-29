"""Cookbook verification: Reinforcement Learning (RL) on WikiSQL.

Uses the atropos sql_query_env pattern: sample SQL -> execute -> reward -> train.
The RL loop matches the tinker-atropos training pipeline.
"""

from __future__ import annotations

import re
import sqlite3

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

NUM_RL_STEPS = 10


@pytest.fixture
def backend(model_name, tmp_path):
    config = EngineConfig(
        base_model=model_name,
        quantize_bits=4,
        quantize_group_size=64,
        checkpoints_base=tmp_path / "checkpoints",
    )
    return MLXBackend(config)


def _format_prompt(example: dict) -> str:
    """Format WikiSQL example as a prompt (atropos sql_query_env pattern)."""
    header = example.get("columns", [])
    rows = example.get("rows", [])

    cols_str = " | ".join(header) if header else ""
    sample_rows = ""
    for row_data in rows[:3] if rows else []:
        sample_rows += " | ".join(str(v) for v in row_data) + "\n"

    return (
        f"Table columns: {cols_str}\n"
        f"Sample data:\n{sample_rows}\n"
        f"Question: {example.get('question', '')}\n"
        f"SQL: "
    )


def _extract_sql(text: str) -> str | None:
    """Extract a SQL statement from generated text."""
    match = re.search(r"(SELECT\b[^;]*)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _execute_sql_reward(sql: str, example: dict) -> float:
    """Execute SQL against the WikiSQL table in sqlite3 and return reward."""
    header = example.get("columns", [])
    rows = example.get("rows", [])
    if not header or not rows:
        return -1.0

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    try:
        cols_def = ", ".join(f'"{h}" TEXT' for h in header)
        cur.execute(f"CREATE TABLE data ({cols_def})")
        placeholders = ", ".join("?" * len(header))
        cur.executemany(
            f"INSERT INTO data VALUES ({placeholders})",
            [tuple(str(v) for v in r) for r in rows],
        )
        cur.execute(sql)
        result = cur.fetchall()
        return 1.0 if result else -1.0
    except Exception:
        return -1.0
    finally:
        conn.close()


def _compute_reward(generated_text: str, example: dict) -> float:
    """Compute reward by executing extracted SQL against the table."""
    sql = _extract_sql(generated_text)
    if sql is None:
        return -1.0
    return _execute_sql_reward(sql, example)


def _compute_advantages(rewards: list[float]) -> list[float]:
    """Normalize rewards to advantages (tinker-atropos pattern)."""
    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    return [r - mean_r for r in rewards]


class TestRLWorkflow:
    def test_rl_loop(self, backend, model_name, wikisql_data):
        """Full RL workflow: sample -> reward -> IS loss -> optim_step."""
        lora_config = LoraConfig(rank=8, alpha=16.0, train_attn=True, train_mlp=True)
        backend.create_model("rl-test", CreateModelInput(lora_config=lora_config))
        tokenizer = backend.tokenizers["rl-test"]

        rl_losses = []

        for step in range(NUM_RL_STEPS):
            example = wikisql_data[step % len(wikisql_data)]
            prompt_text = _format_prompt(example)
            prompt_tokens = tokenizer.encode(prompt_text)

            sample_result = backend.sample(
                "rl-test",
                SampleInput(
                    prompt=ModelInput(chunks=[EncodedTextChunk(tokens=prompt_tokens)]),
                    sampling_params=SamplingParams(temperature=0.8, max_tokens=64),
                    num_samples=2,
                    prompt_logprobs=False,
                ),
            )
            assert len(sample_result.sequences) == 2

            rewards = []
            for seq in sample_result.sequences:
                gen_text = tokenizer.decode(seq.tokens)
                r = _compute_reward(gen_text, example)
                rewards.append(r)

            advantages = _compute_advantages(rewards)

            # 3. Build training data from rollouts (tinker-atropos Datum pattern)
            data = []
            for i, seq in enumerate(sample_result.sequences):
                full_tokens = prompt_tokens + seq.tokens
                input_tokens = full_tokens[:-1]
                target_tokens = full_tokens[1:]

                n_prompt = len(prompt_tokens) - 1
                n_gen = len(seq.tokens)
                weights = [0.0] * n_prompt + [1.0] * n_gen
                weights = weights[: len(target_tokens)]

                adv = [0.0] * n_prompt + [advantages[i]] * n_gen
                adv = adv[: len(target_tokens)]

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

            # 4. Forward-backward with importance sampling
            fb_result = backend.forward_backward(
                "rl-test",
                ForwardBackwardInput(data=data, loss_fn="importance_sampling"),
            )
            rl_losses.append(fb_result.metrics["loss:sum"])

            # 5. Optimizer step
            backend.optim_step(
                "rl-test",
                OptimStepInput(adam_params=AdamParams(learning_rate=5e-5)),
            )

        print("\n=== RL Cookbook Test (WikiSQL) ===")
        print(f"  RL losses: {[f'{v:.4f}' for v in rl_losses]}")

        assert len(rl_losses) == NUM_RL_STEPS
        assert all(isinstance(v, float) for v in rl_losses)
        # IS loss is 0 when all samples get equal reward (advantages normalize to 0)
        if not all(v == 0.0 for v in rl_losses):
            assert any(v != rl_losses[0] for v in rl_losses), (
                "Loss should change across RL steps"
            )

    def test_ppo_loss(self, backend, model_name, wikisql_data):
        """Test PPO loss variant for RL on WikiSQL."""
        lora_config = LoraConfig(rank=8, alpha=16.0)
        backend.create_model("ppo-test", CreateModelInput(lora_config=lora_config))
        tokenizer = backend.tokenizers["ppo-test"]

        example = wikisql_data[0]
        prompt_text = _format_prompt(example)
        prompt_tokens = tokenizer.encode(prompt_text)

        sample_result = backend.sample(
            "ppo-test",
            SampleInput(
                prompt=ModelInput(chunks=[EncodedTextChunk(tokens=prompt_tokens)]),
                sampling_params=SamplingParams(temperature=0.8, max_tokens=32),
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
