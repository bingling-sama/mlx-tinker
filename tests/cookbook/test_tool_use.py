"""Cookbook verification: Tool Use with RL (SQL executor as tool).

Uses the atropos sql_query_env pattern where the SQL executor is the tool:
  1. Model generates SQL query
  2. SQL executor tool runs the query against the table
  3. Model receives execution result
  4. Reward based on correctness
  5. Train with importance_sampling loss on full trajectory
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


@pytest.fixture
def backend(model_name, tmp_path):
    config = EngineConfig(
        base_model=model_name,
        quantize_bits=4,
        quantize_group_size=64,
        checkpoints_base=tmp_path / "checkpoints",
    )
    return MLXBackend(config)


def _format_tool_prompt(example: dict) -> str:
    """Format WikiSQL example as a tool-use prompt."""
    header = example.get("columns", [])
    rows = example.get("rows", [])

    cols_str = " | ".join(header) if header else ""
    sample_rows = ""
    for row_data in rows[:3] if rows else []:
        sample_rows += " | ".join(str(v) for v in row_data) + "\n"

    return (
        f"You have access to a SQL executor tool. "
        f"Write SQL to answer the question.\n\n"
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


class TestToolUseWorkflow:
    def test_sql_tool_rl_loop(self, backend, model_name, wikisql_data):
        """RL loop with SQL executor as tool: generate -> execute -> reward -> train."""
        lora_config = LoraConfig(rank=8, alpha=16.0, train_attn=True, train_mlp=True)
        backend.create_model("tool-test", CreateModelInput(lora_config=lora_config))
        tokenizer = backend.tokenizers["tool-test"]

        example = wikisql_data[0]
        prompt_text = _format_tool_prompt(example)
        prompt_tokens = tokenizer.encode(prompt_text)

        sample_result = backend.sample(
            "tool-test",
            SampleInput(
                prompt=ModelInput(chunks=[EncodedTextChunk(tokens=prompt_tokens)]),
                sampling_params=SamplingParams(temperature=0.7, max_tokens=64),
                num_samples=1,
            ),
        )

        seq = sample_result.sequences[0]
        gen_text = tokenizer.decode(seq.tokens)
        all_tokens = prompt_tokens + seq.tokens
        all_logprobs = [0.0] * len(prompt_tokens) + seq.logprobs

        reward = _compute_reward(gen_text, example)

        input_tokens = all_tokens[:-1]
        target_tokens = all_tokens[1:]
        n_prompt = len(prompt_tokens) - 1
        weights = [0.0] * n_prompt + [1.0] * (len(target_tokens) - n_prompt)
        weights = weights[: len(target_tokens)]
        advantages = [0.0] * n_prompt + [reward] * (len(target_tokens) - n_prompt)
        advantages = advantages[: len(target_tokens)]
        old_lp = all_logprobs[1:]
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
            "tool-test",
            ForwardBackwardInput(data=[datum], loss_fn="importance_sampling"),
        )
        backend.optim_step(
            "tool-test",
            OptimStepInput(adam_params=AdamParams(learning_rate=5e-5)),
        )

        print("\n=== SQL Tool-Use Test ===")
        print(f"  Generated SQL: {gen_text[:100]}...")
        print(f"  Reward: {reward}")
        print(f"  Loss: {fb_result.metrics['loss:sum']:.4f}")

        assert fb_result.loss_fn_output_type == "importance_sampling"

    def test_multi_example_tool_use(self, backend, model_name, wikisql_data):
        """Train on multiple WikiSQL examples with tool-use pattern."""
        lora_config = LoraConfig(rank=8, alpha=16.0)
        backend.create_model("tool-multi", CreateModelInput(lora_config=lora_config))
        tokenizer = backend.tokenizers["tool-multi"]

        losses = []
        for example in wikisql_data[:3]:
            prompt_text = _format_tool_prompt(example)
            prompt_tokens = tokenizer.encode(prompt_text)

            sample_result = backend.sample(
                "tool-multi",
                SampleInput(
                    prompt=ModelInput(chunks=[EncodedTextChunk(tokens=prompt_tokens)]),
                    sampling_params=SamplingParams(temperature=0.7, max_tokens=32),
                    num_samples=1,
                ),
            )

            seq = sample_result.sequences[0]
            all_tokens = prompt_tokens + seq.tokens
            gen_text = tokenizer.decode(seq.tokens)
            reward = _compute_reward(gen_text, example)

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
            losses.append(fb_result.metrics["loss:sum"])
            backend.optim_step(
                "tool-multi",
                OptimStepInput(adam_params=AdamParams(learning_rate=5e-5)),
            )

        print("\n=== Multi-Example Tool-Use Test ===")
        print(f"  Losses: {[f'{v:.4f}' for v in losses]}")

        assert len(losses) == 3
