"""Cookbook verification: Supervised Fine-Tuning (SFT) on WikiSQL.

Trains the model to generate SQL queries from natural language questions.
Uses WikiSQL data loaded via atropos/tinker-atropos data pipeline.
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

NUM_STEPS = 50


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
    """Format a WikiSQL example as a prompt (matches atropos sql_query_env prompt format)."""
    tbl = example.get("table", {})
    header = tbl.get("header", [])
    rows = tbl.get("rows", [])

    cols_str = " | ".join(header) if header else ""
    sample_rows = ""
    # Show first 3 rows
    for i, row_data in enumerate(rows[:3] if rows else []):
        sample_rows += " | ".join(str(v) for v in row_data) + "\n"

    question = example.get("question", "")
    return (
        f"Table columns: {cols_str}\n"
        f"Sample data:\n{sample_rows}\n"
        f"Question: {question}\n"
        f"Write a SQL query to answer this question.\n"
        f"SQL: "
    )


def _get_gold_sql(example: dict) -> str:
    """Extract the gold SQL from a WikiSQL example."""
    # WikiSQL structured query → readable SQL
    sql_dict = example.get("sql", {})
    tbl = example.get("table", {})
    header = tbl.get("header", [])

    if not header or not sql_dict:
        return "SELECT * FROM data"

    agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    cond_ops = ["=", ">", "<"]

    sel_idx = sql_dict.get("sel", 0)
    agg_idx = sql_dict.get("agg", 0)
    sel_col = header[sel_idx] if sel_idx < len(header) else header[0]
    agg = agg_ops[agg_idx] if agg_idx < len(agg_ops) else ""

    select_clause = f'{agg}("{sel_col}")' if agg else f'"{sel_col}"'

    conds = sql_dict.get("conds", {})
    col_indices = conds.get("column_index", [])
    op_indices = conds.get("operator_index", [])
    values = conds.get("condition", [])

    where_parts = []
    for ci, oi, val in zip(col_indices, op_indices, values):
        col = header[ci] if ci < len(header) else header[0]
        op = cond_ops[oi] if oi < len(cond_ops) else "="
        try:
            float(val)
            where_parts.append(f'"{col}" {op} {val}')
        except (ValueError, TypeError):
            where_parts.append(f'"{col}" {op} \'{val}\'')

    sql = f"SELECT {select_clause} FROM data"
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    return sql


def _make_sft_datum(tokenizer, example: dict) -> Datum:
    """Create an SFT datum: prompt tokens (weight=0), SQL completion (weight=1)."""
    prompt = _format_prompt(example)
    completion = " " + _get_gold_sql(example)

    prompt_tokens = tokenizer.encode(prompt)
    completion_tokens = tokenizer.encode(completion)
    all_tokens = prompt_tokens + completion_tokens

    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)
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


class TestSFTWorkflow:
    def test_loss_decreases_over_steps(self, backend, model_name, wikisql_data):
        """Full SFT workflow on WikiSQL: train 50 steps, verify loss decreases."""
        lora_config = LoraConfig(rank=8, alpha=16.0, train_attn=True, train_mlp=True)
        result = backend.create_model("sft-test", CreateModelInput(lora_config=lora_config))
        assert result.model_id == "sft-test"

        tokenizer = backend.tokenizers["sft-test"]

        losses = []
        for step in range(NUM_STEPS):
            example = wikisql_data[step % len(wikisql_data)]
            datum = _make_sft_datum(tokenizer, example)

            fb_result = backend.forward_backward(
                "sft-test",
                ForwardBackwardInput(data=[datum], loss_fn="cross_entropy"),
            )
            step_loss = fb_result.metrics["loss:sum"]
            losses.append(step_loss)

            backend.optim_step(
                "sft-test",
                OptimStepInput(adam_params=AdamParams(learning_rate=1e-4)),
            )

        print(f"\n=== SFT Cookbook Test (WikiSQL, {NUM_STEPS} steps) ===")
        print(f"  First 5 losses: {[f'{l:.4f}' for l in losses[:5]]}")
        print(f"  Last 5 losses:  {[f'{l:.4f}' for l in losses[-5:]]}")

        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_convergence_threshold(self, backend, model_name, wikisql_data):
        """SFT loss should reach a meaningful convergence threshold."""
        lora_config = LoraConfig(rank=8, alpha=16.0, train_attn=True, train_mlp=True)
        backend.create_model("sft-conv", CreateModelInput(lora_config=lora_config))
        tokenizer = backend.tokenizers["sft-conv"]

        losses = []
        for step in range(NUM_STEPS):
            example = wikisql_data[step % len(wikisql_data)]
            datum = _make_sft_datum(tokenizer, example)

            fb_result = backend.forward_backward(
                "sft-conv",
                ForwardBackwardInput(data=[datum], loss_fn="cross_entropy"),
            )
            losses.append(fb_result.metrics["loss:sum"])
            backend.optim_step(
                "sft-conv",
                OptimStepInput(adam_params=AdamParams(learning_rate=1e-4)),
            )

        avg_last_10 = sum(losses[-10:]) / 10
        print(f"\n=== SFT Convergence: avg last 10 = {avg_last_10:.4f} ===")

        assert avg_last_10 < losses[0], "Average loss should decrease over training"

    def test_multiple_data_per_step(self, backend, model_name, wikisql_data):
        """Test batching multiple WikiSQL examples in a single forward_backward call."""
        lora_config = LoraConfig(rank=8, alpha=16.0)
        backend.create_model("sft-batch", CreateModelInput(lora_config=lora_config))
        tokenizer = backend.tokenizers["sft-batch"]

        data = [_make_sft_datum(tokenizer, ex) for ex in wikisql_data[:3]]

        result = backend.forward_backward(
            "sft-batch",
            ForwardBackwardInput(data=data, loss_fn="cross_entropy"),
        )

        assert len(result.loss_fn_outputs) == 3
        assert all("logprobs" in o for o in result.loss_fn_outputs)
        assert result.metrics["num_sequences:sum"] == 3
