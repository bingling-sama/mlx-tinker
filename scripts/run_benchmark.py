#!/usr/bin/env python3
"""Tinker vs mlx-tinker Benchmark: Inference, SFT, and RL on WikiSQL.

Runs 6 experiments comparing Tinker (cloud GPU) and mlx-tinker (local Apple
Silicon) using the tinker-atropos SQL evaluation pattern.

Experiments:
  1. Inference (base model) on Tinker cloud
  2. Inference (base model) on mlx-tinker local
  3. SFT training (50 steps, batch=2) on Tinker cloud
  4. SFT training (50 steps, batch=2) on mlx-tinker local
  5. RL training (50 steps, IS loss) on Tinker cloud
  6. RL training (50 steps, IS loss) on mlx-tinker local

Prerequisites:
  - TINKER_API_KEY in .env or environment (for cloud experiments)
  - mlx-tinker server running: uv run python -m mlx_tinker --model Qwen/Qwen3.5-0.8B --port 8010
  - Install extras: uv sync --extra benchmark

Usage:
  uv run python scripts/run_benchmark.py                     # Both backends
  uv run python scripts/run_benchmark.py --tinker-only        # Cloud only
  uv run python scripts/run_benchmark.py --mlx-only           # Local only
  uv run python scripts/run_benchmark.py --sft-steps 5 --rl-steps 5  # Quick test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import tinker

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("BENCHMARK_MODEL", "Qwen/Qwen3.5-0.8B")
LORA_RANK = 8
LORA_ALPHA = 16.0
SFT_STEPS = 150
RL_STEPS = 50
SFT_BATCH_SIZE = 2
RL_NUM_ROLLOUTS = 8
SFT_LR = 1e-4
RL_LR = 5e-5
MAX_SEQ_LEN = 256
MAX_GEN_TOKENS = 128
EVAL_TEMPERATURE = 0.0  # greedy
RL_TEMPERATURE = 0.8
TRAIN_SIZE = 40
EVAL_SIZE = 100

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
OUTPUT_DIR = Path(__file__).parent.parent / "workspace_reports" / "benchmark"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_wikisql() -> list[dict]:
    path = FIXTURES_DIR / "wikisql_subset.json"
    if not path.exists():
        print(f"ERROR: WikiSQL fixture not found at {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def format_prompt(example: dict) -> str:
    """Format WikiSQL example as a text-to-SQL prompt (atropos sql_query_env pattern)."""
    cols_str = " | ".join(example.get("columns", []))
    rows = example.get("rows", [])
    sample_rows = ""
    for row_data in rows[:3]:
        sample_rows += " | ".join(str(v) for v in row_data) + "\n"
    question = example.get("question", "")
    return (
        f"Table columns: {cols_str}\n"
        f"Sample data:\n{sample_rows}\n"
        f"Question: {question}\n"
        f"SQL: "
    )


def extract_sql(text: str) -> str | None:
    """Extract a SQL statement from generated text (first line only)."""
    match = re.search(r"(SELECT\b[^;\n]*)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def execute_sql(sql: str, example: dict) -> list | None:
    """Execute SQL against an in-memory SQLite table built from the example."""
    header = example.get("columns", [])
    rows = example.get("rows", [])
    if not header or not rows:
        return None
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
        return cur.fetchall()
    except Exception:
        return None
    finally:
        conn.close()


def compute_sql_reward(generated_text: str, example: dict) -> float:
    """Compute reward: +1 if predicted SQL matches gold results, -1 otherwise."""
    sql = extract_sql(generated_text)
    if sql is None:
        return -1.0
    pred_result = execute_sql(normalize_table_name(sql), example)
    gold_result = gold_sql_result(example)
    if pred_result is None or gold_result is None:
        return -1.0
    return 1.0 if set(map(tuple, pred_result)) == set(map(tuple, gold_result)) else -1.0


def gold_sql_result(example: dict) -> list | None:
    """Execute the gold SQL (adjusting table name) and return results."""
    gold = example.get("sql", "")
    # Fixture gold SQL uses "FROM table" but our SQLite table is named "data"
    adjusted = re.sub(r"\bFROM\s+table\b", "FROM data", gold, flags=re.IGNORECASE)
    # Quote column names that may have spaces
    return execute_sql(adjusted, example)


def normalize_table_name(sql: str) -> str:
    """Normalize 'FROM table' → 'FROM data' to match our SQLite eval schema."""
    return re.sub(r"\bFROM\s+table\b", "FROM data", sql, flags=re.IGNORECASE)


def check_exec_match(pred_sql: str | None, example: dict) -> bool:
    """Check if predicted SQL produces the same result set as gold SQL."""
    if pred_sql is None:
        return False
    pred_result = execute_sql(normalize_table_name(pred_sql), example)
    gold_result = gold_sql_result(example)
    if pred_result is None or gold_result is None:
        return False
    return set(map(tuple, pred_result)) == set(map(tuple, gold_result))


# ---------------------------------------------------------------------------
# Datum construction
# ---------------------------------------------------------------------------


def make_sft_datum(tokenizer, example: dict) -> tinker.Datum:
    """Build an SFT datum: prompt tokens (weight=0), SQL completion (weight=1)."""
    prompt = format_prompt(example)
    gold = example.get("sql", "SELECT 1")
    completion = " " + gold
    prompt_tokens = tokenizer.encode(prompt)
    completion_tokens = tokenizer.encode(completion)
    all_tokens = (prompt_tokens + completion_tokens)[:MAX_SEQ_LEN]

    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    n_prompt = min(len(prompt_tokens) - 1, len(target_tokens))
    weights = [0.0] * n_prompt + [1.0] * (len(target_tokens) - n_prompt)

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "weights": weights,
        },
    )


def make_rl_datum(
    tokenizer,
    prompt_tokens: list[int],
    seq_tokens: list[int],
    seq_logprobs: list[float],
    advantage: float,
) -> tinker.Datum:
    """Build an RL datum from a sampled sequence with IS loss fields."""
    full_tokens = (prompt_tokens + seq_tokens)[:MAX_SEQ_LEN]
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    n_prompt = min(len(prompt_tokens) - 1, len(target_tokens))
    n_gen = len(target_tokens) - n_prompt

    # IS loss uses advantages=0 at prompt positions as implicit mask (no weights field)
    adv = [0.0] * n_prompt + [advantage] * n_gen
    old_lp = [0.0] * n_prompt + seq_logprobs[:n_gen]
    # Pad if needed
    while len(old_lp) < len(target_tokens):
        old_lp.append(0.0)

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "advantages": adv,
            "logprobs": old_lp,
        },
    )


# ---------------------------------------------------------------------------
# Future resolution helper
# ---------------------------------------------------------------------------


async def resolve(result):
    """Resolve an APIFuture or return the result directly."""
    if hasattr(result, "result_async"):
        return await result.result_async()
    if hasattr(result, "result"):
        return result.result()
    return result


# ---------------------------------------------------------------------------
# Memory tracking (mlx-tinker / Apple Silicon only)
# ---------------------------------------------------------------------------

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def reset_memory():
    if HAS_MLX:
        mx.reset_peak_memory()


def get_peak_memory_gb() -> float | None:
    if HAS_MLX:
        mx.eval(mx.zeros(1))  # force sync
        return mx.get_peak_memory() / 1e9
    return None


# ---------------------------------------------------------------------------
# Experiment 1: Inference Evaluation
# ---------------------------------------------------------------------------


async def run_inference_eval(
    service_client: tinker.ServiceClient,
    eval_examples: list[dict],
    tokenizer,
    backend_name: str,
) -> dict:
    """Evaluate base model SQL generation accuracy on eval split."""
    print(f"\n{'='*60}")
    print(f"  [{backend_name}] Experiment: Base Model Inference")
    print(f"{'='*60}")

    if backend_name == "mlx-tinker":
        reset_memory()

    t_total_start = time.time()

    if backend_name == "mlx-tinker":
        # mlx-tinker requires creating a model first, then getting a sampling client
        training_client = await service_client.create_lora_training_client_async(
            base_model=MODEL_NAME, rank=LORA_RANK
        )
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    else:
        sampling_client = await service_client.create_sampling_client_async(
            base_model=MODEL_NAME
        )

    per_example = []
    correct = 0

    for i, example in enumerate(eval_examples):
        prompt = format_prompt(example)
        prompt_tokens = tokenizer.encode(prompt)

        t0 = time.time()
        response = await sampling_client.sample_async(
            prompt=tinker.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=EVAL_TEMPERATURE, max_tokens=MAX_GEN_TOKENS
            ),
        )
        dt = time.time() - t0

        seq = response.sequences[0]
        gen_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        pred_sql = extract_sql(gen_text)
        is_correct = check_exec_match(pred_sql, example)
        if is_correct:
            correct += 1

        per_example.append(
            {
                "idx": i,
                "question": example.get("question", ""),
                "gold_sql": example.get("sql", ""),
                "pred_sql": pred_sql,
                "correct": is_correct,
                "time_s": round(dt, 3),
                "gen_text_preview": gen_text[:200],
            }
        )
        status = "OK" if is_correct else "MISS"
        print(f"  [{i+1}/{len(eval_examples)}] {status}  {dt:.2f}s  SQL: {(pred_sql or '(none)')[:60]}")

    total_time = time.time() - t_total_start
    accuracy = correct / len(eval_examples) if eval_examples else 0.0
    memory_gb = get_peak_memory_gb() if backend_name == "mlx-tinker" else None

    result = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(eval_examples),
        "total_time_s": round(total_time, 2),
        "avg_time_per_sample_s": round(total_time / len(eval_examples), 3) if eval_examples else 0,
        "peak_memory_gb": round(memory_gb, 2) if memory_gb else None,
        "per_example": per_example,
    }
    print(f"  Accuracy: {correct}/{len(eval_examples)} = {accuracy:.1%}")
    print(f"  Total time: {total_time:.1f}s")
    return result


# ---------------------------------------------------------------------------
# Experiment 2: SFT Training
# ---------------------------------------------------------------------------


async def run_sft_training(
    service_client: tinker.ServiceClient,
    train_examples: list[dict],
    eval_examples: list[dict],
    tokenizer,
    backend_name: str,
    num_steps: int,
) -> dict:
    """Train with SFT (cross_entropy, batch_size=2) and evaluate."""
    print(f"\n{'='*60}")
    print(f"  [{backend_name}] Experiment: SFT Training ({num_steps} steps, batch={SFT_BATCH_SIZE})")
    print(f"{'='*60}")

    if backend_name == "mlx-tinker":
        reset_memory()

    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=LORA_RANK
    )

    training_logs = []
    t_total_start = time.time()

    for step in range(num_steps):
        # Select batch_size examples
        datums = []
        for b in range(SFT_BATCH_SIZE):
            idx = (step * SFT_BATCH_SIZE + b) % len(train_examples)
            datums.append(make_sft_datum(tokenizer, train_examples[idx]))

        # Forward-backward
        t_fb = time.time()
        fb_future = await training_client.forward_backward_async(
            datums, loss_fn="cross_entropy"
        )
        fb_result = await resolve(fb_future)
        fb_time = time.time() - t_fb

        loss_sum = fb_result.metrics.get("loss:sum", fb_result.metrics.get("mean_loss", 0.0))

        # Optim step
        t_opt = time.time()
        optim_future = await training_client.optim_step_async(
            tinker.AdamParams(learning_rate=SFT_LR, beta1=0.9, beta2=0.999, eps=1e-8)
        )
        await resolve(optim_future)
        opt_time = time.time() - t_opt

        step_log = {
            "step": step,
            "loss_sum": round(loss_sum, 4),
            "fb_time_s": round(fb_time, 3),
            "optim_time_s": round(opt_time, 3),
            "step_time_s": round(fb_time + opt_time, 3),
        }
        training_logs.append(step_log)

        if step % 10 == 0 or step == num_steps - 1:
            print(f"  Step {step:3d}: loss={loss_sum:8.4f}  fb={fb_time:.2f}s  opt={opt_time:.2f}s")

    total_train_time = time.time() - t_total_start
    print(f"  Training complete: {total_train_time:.1f}s total")

    # Post-training evaluation
    print(f"  Running post-training evaluation...")
    t_eval_start = time.time()

    try:
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        eval_results = []
        correct = 0
        for i, example in enumerate(eval_examples):
            prompt = format_prompt(example)
            prompt_tokens = tokenizer.encode(prompt)
            response = await sampling_client.sample_async(
                prompt=tinker.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    temperature=EVAL_TEMPERATURE, max_tokens=MAX_GEN_TOKENS
                ),
            )
            gen_text = tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True)
            pred_sql = extract_sql(gen_text)
            is_correct = check_exec_match(pred_sql, example)
            if is_correct:
                correct += 1
            eval_results.append(
                {
                    "idx": i,
                    "question": example.get("question", ""),
                    "pred_sql": pred_sql,
                    "correct": is_correct,
                }
            )

        eval_accuracy = correct / len(eval_examples) if eval_examples else 0.0
        print(f"  Post-train accuracy: {correct}/{len(eval_examples)} = {eval_accuracy:.1%}")
    except Exception as e:
        print(f"  Post-train eval failed: {e}")
        eval_results = []
        eval_accuracy = None

    total_eval_time = time.time() - t_eval_start
    memory_gb = get_peak_memory_gb() if backend_name == "mlx-tinker" else None

    return {
        "num_steps": num_steps,
        "batch_size": SFT_BATCH_SIZE,
        "learning_rate": SFT_LR,
        "total_train_time_s": round(total_train_time, 2),
        "total_eval_time_s": round(total_eval_time, 2),
        "avg_step_time_s": round(total_train_time / num_steps, 3) if num_steps else 0,
        "initial_loss": training_logs[0]["loss_sum"] if training_logs else None,
        "final_loss": training_logs[-1]["loss_sum"] if training_logs else None,
        "eval_accuracy": round(eval_accuracy, 4) if eval_accuracy is not None else None,
        "peak_memory_gb": round(memory_gb, 2) if memory_gb else None,
        "training_logs": training_logs,
        "eval_details": eval_results,
    }


# ---------------------------------------------------------------------------
# Experiment 3: RL Training
# ---------------------------------------------------------------------------


async def run_rl_training(
    service_client: tinker.ServiceClient,
    train_examples: list[dict],
    eval_examples: list[dict],
    tokenizer,
    backend_name: str,
    num_steps: int,
) -> dict:
    """Train with RL (importance_sampling, sample every step) and evaluate."""
    print(f"\n{'='*60}")
    print(f"  [{backend_name}] Experiment: RL Training ({num_steps} steps)")
    print(f"{'='*60}")

    if backend_name == "mlx-tinker":
        reset_memory()

    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=LORA_RANK
    )

    training_logs = []
    t_total_start = time.time()

    for step in range(num_steps):
        example = train_examples[step % len(train_examples)]
        prompt_text = format_prompt(example)
        prompt_tokens = tokenizer.encode(prompt_text)

        # Save weights and get sampling client (every step per user request)
        t_save = time.time()
        try:
            sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        except Exception as e:
            print(f"  Step {step}: save_weights failed: {e}, using base model sampling")
            sampling_client = await service_client.create_sampling_client_async(
                base_model=MODEL_NAME
            )
        save_time = time.time() - t_save

        # Sample rollouts
        t_sample = time.time()
        try:
            response = await sampling_client.sample_async(
                prompt=tinker.ModelInput.from_ints(prompt_tokens),
                num_samples=RL_NUM_ROLLOUTS,
                sampling_params=tinker.SamplingParams(
                    temperature=RL_TEMPERATURE, max_tokens=MAX_GEN_TOKENS
                ),
            )
        except Exception as e:
            print(f"  Step {step}: sample failed: {e}")
            training_logs.append({"step": step, "error": str(e)})
            continue
        sample_time = time.time() - t_sample

        # Compute rewards
        rewards = []
        generated_sqls = []
        for seq in response.sequences:
            gen_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            sql = extract_sql(gen_text)
            generated_sqls.append(sql)
            rewards.append(compute_sql_reward(gen_text, example))

        # Compute advantages (tinker-atropos pattern: reward - mean)
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        advantages = [r - mean_reward for r in rewards]

        # Build RL datums
        datums = []
        for i, seq in enumerate(response.sequences):
            logprobs = seq.logprobs if seq.logprobs else [0.0] * len(seq.tokens)
            datums.append(
                make_rl_datum(tokenizer, prompt_tokens, seq.tokens, logprobs, advantages[i])
            )

        # Forward-backward with importance sampling
        t_fb = time.time()
        try:
            fb_future = await training_client.forward_backward_async(
                datums, loss_fn="importance_sampling"
            )
            fb_result = await resolve(fb_future)
            loss_sum = fb_result.metrics.get("loss:sum", fb_result.metrics.get("mean_loss", 0.0))
        except Exception as e:
            print(f"  Step {step}: forward_backward failed: {e}")
            training_logs.append({"step": step, "error": str(e)})
            continue
        fb_time = time.time() - t_fb

        # Optim step
        t_opt = time.time()
        optim_future = await training_client.optim_step_async(
            tinker.AdamParams(learning_rate=RL_LR, beta1=0.9, beta2=0.999, eps=1e-8)
        )
        await resolve(optim_future)
        opt_time = time.time() - t_opt

        step_log = {
            "step": step,
            "loss_sum": round(loss_sum, 4),
            "rewards": rewards,
            "advantages": [round(a, 4) for a in advantages],
            "mean_reward": round(mean_reward, 4),
            "save_time_s": round(save_time, 3),
            "sample_time_s": round(sample_time, 3),
            "fb_time_s": round(fb_time, 3),
            "optim_time_s": round(opt_time, 3),
            "step_time_s": round(save_time + sample_time + fb_time + opt_time, 3),
            "generated_sqls": generated_sqls,
        }
        training_logs.append(step_log)

        if step % 10 == 0 or step == num_steps - 1:
            print(
                f"  Step {step:3d}: loss={loss_sum:8.4f}  rewards={rewards}  "
                f"sample={sample_time:.2f}s  fb={fb_time:.2f}s"
            )

    total_train_time = time.time() - t_total_start
    print(f"  RL Training complete: {total_train_time:.1f}s total")

    # Post-training evaluation
    print(f"  Running post-training evaluation...")
    t_eval_start = time.time()

    try:
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        eval_results = []
        correct = 0
        for i, example in enumerate(eval_examples):
            prompt = format_prompt(example)
            prompt_tokens = tokenizer.encode(prompt)
            response = await sampling_client.sample_async(
                prompt=tinker.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    temperature=EVAL_TEMPERATURE, max_tokens=MAX_GEN_TOKENS
                ),
            )
            gen_text = tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True)
            pred_sql = extract_sql(gen_text)
            is_correct = check_exec_match(pred_sql, example)
            if is_correct:
                correct += 1
            eval_results.append(
                {
                    "idx": i,
                    "question": example.get("question", ""),
                    "pred_sql": pred_sql,
                    "correct": is_correct,
                }
            )

        eval_accuracy = correct / len(eval_examples) if eval_examples else 0.0
        print(f"  Post-train accuracy: {correct}/{len(eval_examples)} = {eval_accuracy:.1%}")
    except Exception as e:
        print(f"  Post-train eval failed: {e}")
        eval_results = []
        eval_accuracy = None

    total_eval_time = time.time() - t_eval_start
    memory_gb = get_peak_memory_gb() if backend_name == "mlx-tinker" else None

    # Compute reward trajectory
    valid_logs = [l for l in training_logs if "error" not in l]
    mean_rewards = [l["mean_reward"] for l in valid_logs]

    return {
        "num_steps": num_steps,
        "num_rollouts": RL_NUM_ROLLOUTS,
        "learning_rate": RL_LR,
        "total_train_time_s": round(total_train_time, 2),
        "total_eval_time_s": round(total_eval_time, 2),
        "avg_step_time_s": round(total_train_time / num_steps, 3) if num_steps else 0,
        "initial_loss": valid_logs[0]["loss_sum"] if valid_logs else None,
        "final_loss": valid_logs[-1]["loss_sum"] if valid_logs else None,
        "mean_reward_first_10": round(sum(mean_rewards[:10]) / min(10, len(mean_rewards)), 4)
        if mean_rewards
        else None,
        "mean_reward_last_10": round(sum(mean_rewards[-10:]) / min(10, len(mean_rewards)), 4)
        if mean_rewards
        else None,
        "eval_accuracy": round(eval_accuracy, 4) if eval_accuracy is not None else None,
        "peak_memory_gb": round(memory_gb, 2) if memory_gb else None,
        "training_logs": training_logs,
        "eval_details": eval_results,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _fmt(val, fmt=".4f"):
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def _pct(val):
    if val is None:
        return "N/A"
    return f"{val:.1%}"


def _delta(a, b, fmt=".4f"):
    if a is None or b is None:
        return "N/A"
    diff = b - a
    return f"{diff:+{fmt}}"


def _speedup(cloud_s, local_s):
    if cloud_s is None or local_s is None or local_s == 0:
        return "N/A"
    ratio = cloud_s / local_s
    return f"{ratio:.2f}x"


def generate_report(tinker_r: dict | None, mlx_r: dict | None) -> str:
    """Generate a comprehensive markdown comparison report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append(f"# Tinker vs mlx-tinker Benchmark Report\n")
    lines.append(f"**Generated:** {now}  ")
    lines.append(
        f"**Model:** {MODEL_NAME} | LoRA rank {LORA_RANK} | 4-bit quantization  "
    )
    lines.append(f"**Dataset:** WikiSQL ({TRAIN_SIZE} train, {EVAL_SIZE} eval)\n")

    # Executive Summary
    lines.append("## Executive Summary\n")
    if tinker_r and mlx_r:
        lines.append(
            "This report compares training and inference performance between "
            "Tinker (cloud GPU) and mlx-tinker (local Apple Silicon / Metal). "
            "Both backends use the same tinker SDK API, ensuring identical "
            "training semantics. Differences arise from hardware characteristics "
            "(GPU vs Metal), quantization implementations, and network overhead.\n"
        )
    elif mlx_r:
        lines.append(
            "This report captures mlx-tinker (local Apple Silicon) performance "
            "on WikiSQL inference, SFT, and RL training.\n"
        )
    else:
        lines.append("This report captures Tinker (cloud GPU) performance.\n")

    # Helper to get nested values
    def get(results, *keys):
        if results is None:
            return None
        v = results
        for k in keys:
            if isinstance(v, dict):
                v = v.get(k)
            else:
                return None
        return v

    # Summary Table
    lines.append("## Summary Table\n")
    lines.append("| Metric | Tinker (Cloud) | mlx-tinker (Local) | Delta |")
    lines.append("|--------|:--------------:|:------------------:|:-----:|")

    rows = [
        (
            "Base inference accuracy",
            _pct(get(tinker_r, "inference", "accuracy")),
            _pct(get(mlx_r, "inference", "accuracy")),
            _delta(
                get(tinker_r, "inference", "accuracy"),
                get(mlx_r, "inference", "accuracy"),
            ),
        ),
        (
            "SFT post-train accuracy",
            _pct(get(tinker_r, "sft", "eval_accuracy")),
            _pct(get(mlx_r, "sft", "eval_accuracy")),
            _delta(
                get(tinker_r, "sft", "eval_accuracy"),
                get(mlx_r, "sft", "eval_accuracy"),
            ),
        ),
        (
            "RL post-train accuracy",
            _pct(get(tinker_r, "rl", "eval_accuracy")),
            _pct(get(mlx_r, "rl", "eval_accuracy")),
            _delta(
                get(tinker_r, "rl", "eval_accuracy"),
                get(mlx_r, "rl", "eval_accuracy"),
            ),
        ),
        (
            "SFT initial loss",
            _fmt(get(tinker_r, "sft", "initial_loss")),
            _fmt(get(mlx_r, "sft", "initial_loss")),
            _delta(
                get(tinker_r, "sft", "initial_loss"),
                get(mlx_r, "sft", "initial_loss"),
            ),
        ),
        (
            "SFT final loss",
            _fmt(get(tinker_r, "sft", "final_loss")),
            _fmt(get(mlx_r, "sft", "final_loss")),
            _delta(
                get(tinker_r, "sft", "final_loss"),
                get(mlx_r, "sft", "final_loss"),
            ),
        ),
        (
            "RL initial loss",
            _fmt(get(tinker_r, "rl", "initial_loss")),
            _fmt(get(mlx_r, "rl", "initial_loss")),
            _delta(
                get(tinker_r, "rl", "initial_loss"),
                get(mlx_r, "rl", "initial_loss"),
            ),
        ),
        (
            "RL final loss",
            _fmt(get(tinker_r, "rl", "final_loss")),
            _fmt(get(mlx_r, "rl", "final_loss")),
            _delta(
                get(tinker_r, "rl", "final_loss"),
                get(mlx_r, "rl", "final_loss"),
            ),
        ),
        (
            "SFT avg step time",
            _fmt(get(tinker_r, "sft", "avg_step_time_s"), ".3f") + "s"
            if get(tinker_r, "sft", "avg_step_time_s")
            else "N/A",
            _fmt(get(mlx_r, "sft", "avg_step_time_s"), ".3f") + "s"
            if get(mlx_r, "sft", "avg_step_time_s")
            else "N/A",
            _speedup(
                get(tinker_r, "sft", "avg_step_time_s"),
                get(mlx_r, "sft", "avg_step_time_s"),
            ),
        ),
        (
            "RL avg step time",
            _fmt(get(tinker_r, "rl", "avg_step_time_s"), ".3f") + "s"
            if get(tinker_r, "rl", "avg_step_time_s")
            else "N/A",
            _fmt(get(mlx_r, "rl", "avg_step_time_s"), ".3f") + "s"
            if get(mlx_r, "rl", "avg_step_time_s")
            else "N/A",
            _speedup(
                get(tinker_r, "rl", "avg_step_time_s"),
                get(mlx_r, "rl", "avg_step_time_s"),
            ),
        ),
        (
            "SFT total wall time",
            _fmt(get(tinker_r, "sft", "total_train_time_s"), ".1f") + "s"
            if get(tinker_r, "sft", "total_train_time_s")
            else "N/A",
            _fmt(get(mlx_r, "sft", "total_train_time_s"), ".1f") + "s"
            if get(mlx_r, "sft", "total_train_time_s")
            else "N/A",
            _speedup(
                get(tinker_r, "sft", "total_train_time_s"),
                get(mlx_r, "sft", "total_train_time_s"),
            ),
        ),
        (
            "RL total wall time",
            _fmt(get(tinker_r, "rl", "total_train_time_s"), ".1f") + "s"
            if get(tinker_r, "rl", "total_train_time_s")
            else "N/A",
            _fmt(get(mlx_r, "rl", "total_train_time_s"), ".1f") + "s"
            if get(mlx_r, "rl", "total_train_time_s")
            else "N/A",
            _speedup(
                get(tinker_r, "rl", "total_train_time_s"),
                get(mlx_r, "rl", "total_train_time_s"),
            ),
        ),
        (
            "Peak memory (SFT)",
            "N/A",
            _fmt(get(mlx_r, "sft", "peak_memory_gb"), ".2f") + " GB"
            if get(mlx_r, "sft", "peak_memory_gb")
            else "N/A",
            "--",
        ),
        (
            "Peak memory (RL)",
            "N/A",
            _fmt(get(mlx_r, "rl", "peak_memory_gb"), ".2f") + " GB"
            if get(mlx_r, "rl", "peak_memory_gb")
            else "N/A",
            "--",
        ),
    ]
    for label, cloud, local, delta in rows:
        lines.append(f"| {label} | {cloud} | {local} | {delta} |")

    # Experiment details
    for exp_name, exp_key in [
        ("Base Model Inference", "inference"),
        ("SFT Training", "sft"),
        ("RL Training", "rl"),
    ]:
        lines.append(f"\n## Experiment: {exp_name}\n")

        for backend_name, results in [("Tinker (Cloud)", tinker_r), ("mlx-tinker (Local)", mlx_r)]:
            exp = get(results, exp_key)
            if exp is None:
                continue

            lines.append(f"### {backend_name}\n")

            if exp_key == "inference":
                lines.append(f"- **Accuracy:** {_pct(exp.get('accuracy'))}")
                lines.append(f"- **Total time:** {exp.get('total_time_s', 'N/A')}s")
                lines.append(
                    f"- **Avg time/sample:** {exp.get('avg_time_per_sample_s', 'N/A')}s"
                )
                if exp.get("peak_memory_gb"):
                    lines.append(f"- **Peak memory:** {exp['peak_memory_gb']} GB")
                lines.append("")
                lines.append("| # | Question | Pred SQL | Correct |")
                lines.append("|---|----------|----------|:-------:|")
                for ex in exp.get("per_example", []):
                    q = ex["question"][:40] + "..." if len(ex.get("question", "")) > 40 else ex.get("question", "")
                    sql = (ex.get("pred_sql") or "(none)")[:40]
                    c = "Y" if ex.get("correct") else "N"
                    lines.append(f"| {ex['idx']+1} | {q} | `{sql}` | {c} |")

            elif exp_key == "sft":
                lines.append(f"- **Steps:** {exp.get('num_steps')}, batch_size={exp.get('batch_size')}")
                lines.append(f"- **Learning rate:** {exp.get('learning_rate')}")
                lines.append(
                    f"- **Loss:** {_fmt(exp.get('initial_loss'))} -> {_fmt(exp.get('final_loss'))}"
                )
                lines.append(f"- **Total train time:** {exp.get('total_train_time_s')}s")
                lines.append(f"- **Avg step time:** {exp.get('avg_step_time_s')}s")
                lines.append(f"- **Post-train accuracy:** {_pct(exp.get('eval_accuracy'))}")
                if exp.get("peak_memory_gb"):
                    lines.append(f"- **Peak memory:** {exp['peak_memory_gb']} GB")
                lines.append("")
                lines.append("**Loss curve (every 5 steps):**\n")
                lines.append("| Step | Loss |")
                lines.append("|-----:|-----:|")
                for log in exp.get("training_logs", []):
                    if log["step"] % 5 == 0 or log["step"] == exp.get("num_steps", 50) - 1:
                        lines.append(f"| {log['step']} | {log['loss_sum']:.4f} |")

            elif exp_key == "rl":
                lines.append(f"- **Steps:** {exp.get('num_steps')}")
                lines.append(f"- **Rollouts/step:** {exp.get('num_rollouts')}")
                lines.append(f"- **Learning rate:** {exp.get('learning_rate')}")
                lines.append(
                    f"- **Loss:** {_fmt(exp.get('initial_loss'))} -> {_fmt(exp.get('final_loss'))}"
                )
                lines.append(
                    f"- **Mean reward (first 10):** {_fmt(exp.get('mean_reward_first_10'))}"
                )
                lines.append(
                    f"- **Mean reward (last 10):** {_fmt(exp.get('mean_reward_last_10'))}"
                )
                lines.append(f"- **Total train time:** {exp.get('total_train_time_s')}s")
                lines.append(f"- **Avg step time:** {exp.get('avg_step_time_s')}s")
                lines.append(f"- **Post-train accuracy:** {_pct(exp.get('eval_accuracy'))}")
                if exp.get("peak_memory_gb"):
                    lines.append(f"- **Peak memory:** {exp['peak_memory_gb']} GB")
                lines.append("")
                lines.append("**RL training curve (every 5 steps):**\n")
                lines.append("| Step | Loss | Mean Reward | Advantages |")
                lines.append("|-----:|-----:|:-----------:|:----------:|")
                for log in exp.get("training_logs", []):
                    if "error" in log:
                        continue
                    if log["step"] % 5 == 0 or log["step"] == exp.get("num_steps", 50) - 1:
                        lines.append(
                            f"| {log['step']} | {log['loss_sum']:.4f} | "
                            f"{log.get('mean_reward', 'N/A')} | "
                            f"{log.get('advantages', 'N/A')} |"
                        )

    # Loss curve correlation (if both available)
    if tinker_r and mlx_r:
        lines.append("\n## Loss Curve Correlation\n")
        for exp_key, label in [("sft", "SFT"), ("rl", "RL")]:
            t_logs = get(tinker_r, exp_key, "training_logs") or []
            m_logs = get(mlx_r, exp_key, "training_logs") or []
            t_losses = [l["loss_sum"] for l in t_logs if "loss_sum" in l]
            m_losses = [l["loss_sum"] for l in m_logs if "loss_sum" in l]
            n = min(len(t_losses), len(m_losses))
            if n > 2:
                # Pearson r
                t = t_losses[:n]
                m = m_losses[:n]
                mean_t = sum(t) / n
                mean_m = sum(m) / n
                cov = sum((a - mean_t) * (b - mean_m) for a, b in zip(t, m)) / n
                std_t = (sum((a - mean_t) ** 2 for a in t) / n) ** 0.5
                std_m = (sum((b - mean_m) ** 2 for b in m) / n) ** 0.5
                if std_t > 0 and std_m > 0:
                    r = cov / (std_t * std_m)
                    lines.append(f"- **{label} Pearson r:** {r:.4f}")
                else:
                    lines.append(f"- **{label}:** insufficient variance for correlation")

    # Conclusion
    lines.append("\n## Methodology\n")
    lines.append(
        "- **Evaluation pattern:** tinker-atropos SQL query environment "
        "(generate SQL -> execute against SQLite -> compare result set with gold)\n"
        "- **SFT:** cross_entropy loss, weight masking (prompt=0, completion=1)\n"
        "- **RL:** importance_sampling loss, on-policy sampling (refresh every step), "
        "advantage = reward - mean(rewards)\n"
        "- **Wall-clock times** include network latency for cloud, "
        "Metal compute for local\n"
        "- **Memory** measured via `mx.metal.get_peak_memory()` (local only)\n"
    )

    lines.append("\n---\n")
    lines.append("*Report generated by `scripts/run_benchmark.py`*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run_all_experiments(
    service_client: tinker.ServiceClient,
    backend_name: str,
    train_data: list[dict],
    eval_data: list[dict],
    tokenizer,
    sft_steps: int,
    rl_steps: int,
    skip_inference: bool = False,
    skip_sft: bool = False,
    skip_rl: bool = False,
) -> dict:
    results = {
        "backend": backend_name,
        "model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not skip_inference:
        results["inference"] = await run_inference_eval(
            service_client, eval_data, tokenizer, backend_name
        )

    if not skip_sft:
        results["sft"] = await run_sft_training(
            service_client, train_data, eval_data, tokenizer, backend_name, sft_steps
        )

    if not skip_rl:
        results["rl"] = await run_rl_training(
            service_client, train_data, eval_data, tokenizer, backend_name, rl_steps
        )

    return results


# ---------------------------------------------------------------------------
# MLX-Tinker direct backend experiments (bypasses SDK compatibility issues)
# ---------------------------------------------------------------------------


async def run_mlx_experiments(
    train_data: list[dict],
    eval_data: list[dict],
    tokenizer,
    sft_steps: int,
    rl_steps: int,
    skip_inference: bool = False,
    skip_sft: bool = False,
    skip_rl: bool = False,
) -> dict:
    """Run all experiments using mlx-tinker backend directly (like cookbook tests)."""
    from mlx_tinker.backend.mlx_backend import MLXBackend
    from mlx_tinker.config import EngineConfig
    from mlx_tinker.types import (
        AdamParams as MLXAdamParams,
        CreateModelInput,
        Datum as MLXDatum,
        EncodedTextChunk as MLXEncodedTextChunk,
        ForwardBackwardInput,
        LoraConfig as MLXLoraConfig,
        LossFnInputs,
        ModelInput as MLXModelInput,
        OptimStepInput,
        SampleInput,
        SamplingParams as MLXSamplingParams,
        TensorData,
    )
    import tempfile

    results = {
        "backend": "mlx-tinker",
        "model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    tmp_dir = tempfile.mkdtemp(prefix="mlx_benchmark_")
    config = EngineConfig(
        base_model=MODEL_NAME,
        quantize_bits=4,
        quantize_group_size=64,
        checkpoints_base=Path(tmp_dir) / "checkpoints",
    )

    def _make_mlx_sft_datum(example: dict) -> MLXDatum:
        prompt = format_prompt(example)
        gold = example.get("sql", "SELECT 1")
        completion = " " + gold
        prompt_tokens = tokenizer.encode(prompt)
        completion_tokens = tokenizer.encode(completion)
        all_tokens = (prompt_tokens + completion_tokens)[:MAX_SEQ_LEN]
        input_tokens = all_tokens[:-1]
        target_tokens = all_tokens[1:]
        n_prompt = min(len(prompt_tokens) - 1, len(target_tokens))
        weights = [0.0] * n_prompt + [1.0] * (len(target_tokens) - n_prompt)
        return MLXDatum(
            model_input=MLXModelInput(chunks=[MLXEncodedTextChunk(tokens=input_tokens)]),
            loss_fn_inputs=LossFnInputs(
                target_tokens=TensorData(data=target_tokens),
                weights=TensorData(data=weights),
                advantages=TensorData(data=[0.0] * len(target_tokens)),
                logprobs=TensorData(data=[0.0] * len(target_tokens)),
            ),
        )

    def _make_mlx_rl_datum(
        prompt_tokens: list[int],
        seq_tokens: list[int],
        seq_logprobs: list[float],
        advantage: float,
    ) -> MLXDatum:
        full_tokens = (prompt_tokens + seq_tokens)[:MAX_SEQ_LEN]
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]
        n_prompt = min(len(prompt_tokens) - 1, len(target_tokens))
        n_gen = len(target_tokens) - n_prompt
        weights = [0.0] * n_prompt + [1.0] * n_gen
        adv = [0.0] * n_prompt + [advantage] * n_gen
        old_lp = [0.0] * n_prompt + seq_logprobs[:n_gen]
        while len(old_lp) < len(target_tokens):
            old_lp.append(0.0)
        return MLXDatum(
            model_input=MLXModelInput(chunks=[MLXEncodedTextChunk(tokens=input_tokens)]),
            loss_fn_inputs=LossFnInputs(
                target_tokens=TensorData(data=target_tokens),
                weights=TensorData(data=weights),
                advantages=TensorData(data=adv),
                logprobs=TensorData(data=old_lp),
            ),
        )

    # --- Inference ---
    if not skip_inference:
        print(f"\n{'='*60}")
        print(f"  [mlx-tinker] Experiment: Base Model Inference")
        print(f"{'='*60}")
        reset_memory()
        backend = MLXBackend(config)
        lora_cfg = MLXLoraConfig(rank=LORA_RANK, alpha=LORA_ALPHA)
        backend.create_model("infer", CreateModelInput(lora_config=lora_cfg))

        per_example = []
        correct = 0
        t_total = time.time()
        for i, example in enumerate(eval_data):
            prompt = format_prompt(example)
            prompt_tokens = tokenizer.encode(prompt)
            t0 = time.time()
            sample_result = backend.sample(
                "infer",
                SampleInput(
                    prompt=MLXModelInput(chunks=[MLXEncodedTextChunk(tokens=prompt_tokens)]),
                    sampling_params=MLXSamplingParams(temperature=EVAL_TEMPERATURE, max_tokens=MAX_GEN_TOKENS),
                    num_samples=1,
                ),
            )
            dt = time.time() - t0
            gen_text = tokenizer.decode(sample_result.sequences[0].tokens, skip_special_tokens=True)
            pred_sql = extract_sql(gen_text)
            is_correct = check_exec_match(pred_sql, example)
            if is_correct:
                correct += 1
            per_example.append({
                "idx": i, "question": example.get("question", ""),
                "gold_sql": example.get("sql", ""), "pred_sql": pred_sql,
                "correct": is_correct, "time_s": round(dt, 3),
                "gen_text_preview": gen_text[:200],
            })
            status = "OK" if is_correct else "MISS"
            print(f"  [{i+1}/{len(eval_data)}] {status}  {dt:.2f}s  SQL: {(pred_sql or '(none)')[:60]}")

        total_time = time.time() - t_total
        accuracy = correct / len(eval_data)
        mem = get_peak_memory_gb()
        results["inference"] = {
            "accuracy": round(accuracy, 4), "correct": correct, "total": len(eval_data),
            "total_time_s": round(total_time, 2),
            "avg_time_per_sample_s": round(total_time / len(eval_data), 3),
            "peak_memory_gb": round(mem, 2) if mem else None,
            "per_example": per_example,
        }
        print(f"  Accuracy: {correct}/{len(eval_data)} = {accuracy:.1%}")
        del backend

    # --- SFT ---
    if not skip_sft:
        print(f"\n{'='*60}")
        print(f"  [mlx-tinker] Experiment: SFT Training ({sft_steps} steps, batch={SFT_BATCH_SIZE})")
        print(f"{'='*60}")
        reset_memory()
        backend = MLXBackend(config)
        lora_cfg = MLXLoraConfig(rank=LORA_RANK, alpha=LORA_ALPHA, train_attn=True, train_mlp=True)
        backend.create_model("sft", CreateModelInput(lora_config=lora_cfg))

        training_logs = []
        t_total = time.time()
        for step in range(sft_steps):
            datums = []
            for b in range(SFT_BATCH_SIZE):
                idx = (step * SFT_BATCH_SIZE + b) % len(train_data)
                datums.append(_make_mlx_sft_datum(train_data[idx]))
            t_fb = time.time()
            fb_result = backend.forward_backward("sft", ForwardBackwardInput(data=datums, loss_fn="cross_entropy"))
            fb_time = time.time() - t_fb
            loss_sum = fb_result.metrics["loss:sum"]
            t_opt = time.time()
            backend.optim_step("sft", OptimStepInput(adam_params=MLXAdamParams(learning_rate=SFT_LR)))
            opt_time = time.time() - t_opt
            training_logs.append({
                "step": step, "loss_sum": round(loss_sum, 4),
                "fb_time_s": round(fb_time, 3), "optim_time_s": round(opt_time, 3),
                "step_time_s": round(fb_time + opt_time, 3),
            })
            if step % 10 == 0 or step == sft_steps - 1:
                print(f"  Step {step:3d}: loss={loss_sum:8.4f}  fb={fb_time:.2f}s  opt={opt_time:.2f}s")

        total_train_time = time.time() - t_total
        print(f"  Training complete: {total_train_time:.1f}s total")

        # Post-train eval
        print(f"  Running post-training evaluation...")
        eval_results = []
        correct = 0
        t_eval = time.time()
        for i, example in enumerate(eval_data):
            prompt = format_prompt(example)
            prompt_tokens = tokenizer.encode(prompt)
            sample_result = backend.sample(
                "sft",
                SampleInput(
                    prompt=MLXModelInput(chunks=[MLXEncodedTextChunk(tokens=prompt_tokens)]),
                    sampling_params=MLXSamplingParams(temperature=EVAL_TEMPERATURE, max_tokens=MAX_GEN_TOKENS),
                    num_samples=1,
                ),
            )
            gen_text = tokenizer.decode(sample_result.sequences[0].tokens, skip_special_tokens=True)
            pred_sql = extract_sql(gen_text)
            is_correct = check_exec_match(pred_sql, example)
            if is_correct:
                correct += 1
            eval_results.append({"idx": i, "question": example.get("question", ""), "pred_sql": pred_sql, "correct": is_correct})

        eval_accuracy = correct / len(eval_data)
        total_eval_time = time.time() - t_eval
        mem = get_peak_memory_gb()
        print(f"  Post-train accuracy: {correct}/{len(eval_data)} = {eval_accuracy:.1%}")
        results["sft"] = {
            "num_steps": sft_steps, "batch_size": SFT_BATCH_SIZE, "learning_rate": SFT_LR,
            "total_train_time_s": round(total_train_time, 2),
            "total_eval_time_s": round(total_eval_time, 2),
            "avg_step_time_s": round(total_train_time / sft_steps, 3),
            "initial_loss": training_logs[0]["loss_sum"],
            "final_loss": training_logs[-1]["loss_sum"],
            "eval_accuracy": round(eval_accuracy, 4),
            "peak_memory_gb": round(mem, 2) if mem else None,
            "training_logs": training_logs,
            "eval_details": eval_results,
        }
        del backend

    # --- RL ---
    if not skip_rl:
        print(f"\n{'='*60}")
        print(f"  [mlx-tinker] Experiment: RL Training ({rl_steps} steps)")
        print(f"{'='*60}")
        reset_memory()
        backend = MLXBackend(config)
        lora_cfg = MLXLoraConfig(rank=LORA_RANK, alpha=LORA_ALPHA, train_attn=True, train_mlp=True)
        backend.create_model("rl", CreateModelInput(lora_config=lora_cfg))

        training_logs = []
        t_total = time.time()
        for step in range(rl_steps):
            example = train_data[step % len(train_data)]
            prompt_text = format_prompt(example)
            prompt_tokens = tokenizer.encode(prompt_text)

            t_sample = time.time()
            sample_result = backend.sample(
                "rl",
                SampleInput(
                    prompt=MLXModelInput(chunks=[MLXEncodedTextChunk(tokens=prompt_tokens)]),
                    sampling_params=MLXSamplingParams(temperature=RL_TEMPERATURE, max_tokens=MAX_GEN_TOKENS),
                    num_samples=RL_NUM_ROLLOUTS,
                    prompt_logprobs=False,
                ),
            )
            sample_time = time.time() - t_sample

            rewards = []
            generated_sqls = []
            for seq in sample_result.sequences:
                gen_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                sql = extract_sql(gen_text)
                generated_sqls.append(sql)
                rewards.append(compute_sql_reward(gen_text, example))

            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            advantages = [r - mean_reward for r in rewards]

            datums = []
            for i, seq in enumerate(sample_result.sequences):
                logprobs = seq.logprobs if seq.logprobs else [0.0] * len(seq.tokens)
                datums.append(_make_mlx_rl_datum(prompt_tokens, seq.tokens, logprobs, advantages[i]))

            t_fb = time.time()
            fb_result = backend.forward_backward("rl", ForwardBackwardInput(data=datums, loss_fn="importance_sampling"))
            fb_time = time.time() - t_fb
            loss_sum = fb_result.metrics["loss:sum"]

            t_opt = time.time()
            backend.optim_step("rl", OptimStepInput(adam_params=MLXAdamParams(learning_rate=RL_LR)))
            opt_time = time.time() - t_opt

            training_logs.append({
                "step": step, "loss_sum": round(loss_sum, 4),
                "rewards": rewards, "advantages": [round(a, 4) for a in advantages],
                "mean_reward": round(mean_reward, 4),
                "sample_time_s": round(sample_time, 3),
                "fb_time_s": round(fb_time, 3), "optim_time_s": round(opt_time, 3),
                "step_time_s": round(sample_time + fb_time + opt_time, 3),
                "generated_sqls": generated_sqls,
            })
            if step % 10 == 0 or step == rl_steps - 1:
                print(f"  Step {step:3d}: loss={loss_sum:8.4f}  rewards={rewards}  sample={sample_time:.2f}s  fb={fb_time:.2f}s")

        total_train_time = time.time() - t_total
        print(f"  RL Training complete: {total_train_time:.1f}s total")

        # Post-train eval
        print(f"  Running post-training evaluation...")
        eval_results = []
        correct = 0
        t_eval = time.time()
        for i, example in enumerate(eval_data):
            prompt = format_prompt(example)
            prompt_tokens = tokenizer.encode(prompt)
            sample_result = backend.sample(
                "rl",
                SampleInput(
                    prompt=MLXModelInput(chunks=[MLXEncodedTextChunk(tokens=prompt_tokens)]),
                    sampling_params=MLXSamplingParams(temperature=EVAL_TEMPERATURE, max_tokens=MAX_GEN_TOKENS),
                    num_samples=1,
                ),
            )
            gen_text = tokenizer.decode(sample_result.sequences[0].tokens, skip_special_tokens=True)
            pred_sql = extract_sql(gen_text)
            is_correct = check_exec_match(pred_sql, example)
            if is_correct:
                correct += 1
            eval_results.append({"idx": i, "question": example.get("question", ""), "pred_sql": pred_sql, "correct": is_correct})

        eval_accuracy = correct / len(eval_data)
        total_eval_time = time.time() - t_eval
        mem = get_peak_memory_gb()
        print(f"  Post-train accuracy: {correct}/{len(eval_data)} = {eval_accuracy:.1%}")

        valid_logs = [l for l in training_logs if "error" not in l]
        mean_rewards = [l["mean_reward"] for l in valid_logs]
        results["rl"] = {
            "num_steps": rl_steps, "num_rollouts": RL_NUM_ROLLOUTS, "learning_rate": RL_LR,
            "total_train_time_s": round(total_train_time, 2),
            "total_eval_time_s": round(total_eval_time, 2),
            "avg_step_time_s": round(total_train_time / rl_steps, 3),
            "initial_loss": valid_logs[0]["loss_sum"] if valid_logs else None,
            "final_loss": valid_logs[-1]["loss_sum"] if valid_logs else None,
            "mean_reward_first_10": round(sum(mean_rewards[:10]) / min(10, len(mean_rewards)), 4) if mean_rewards else None,
            "mean_reward_last_10": round(sum(mean_rewards[-10:]) / min(10, len(mean_rewards)), 4) if mean_rewards else None,
            "eval_accuracy": round(eval_accuracy, 4),
            "peak_memory_gb": round(mem, 2) if mem else None,
            "training_logs": training_logs,
            "eval_details": eval_results,
        }
        del backend

    return results


async def main():
    global MODEL_NAME  # noqa: PLW0603
    parser = argparse.ArgumentParser(description="Tinker vs mlx-tinker Benchmark")
    parser.add_argument("--tinker-only", action="store_true", help="Only run Tinker cloud")
    parser.add_argument("--mlx-only", action="store_true", help="Only run mlx-tinker local")
    parser.add_argument(
        "--mlx-url",
        default="http://localhost:8000",
        help="mlx-tinker server URL (default: http://localhost:8000)",
    )
    parser.add_argument("--sft-steps", type=int, default=SFT_STEPS)
    parser.add_argument("--rl-steps", type=int, default=RL_STEPS)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Model to benchmark (default: {MODEL_NAME})",
    )
    args = parser.parse_args()
    MODEL_NAME = args.model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting benchmark...")
    print(
        f"  backends: tinker={'no' if args.mlx_only else 'yes'} mlx={'no' if args.tinker_only else 'yes'}"
    )
    print(
        f"  config: sft_steps={args.sft_steps} rl_steps={args.rl_steps} "
        f"skip_inference={args.skip_inference} skip_sft={args.skip_sft} skip_rl={args.skip_rl}"
    )
    print(f"  output_dir: {output_dir}")

    # Load data
    print("Loading WikiSQL data...")
    all_data = load_wikisql()
    train_data = all_data[:TRAIN_SIZE]
    eval_data = all_data[TRAIN_SIZE : TRAIN_SIZE + EVAL_SIZE]
    print(f"  Train: {len(train_data)} examples, Eval: {len(eval_data)} examples")

    # Load tokenizer
    print(f"Loading tokenizer for {MODEL_NAME}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    run_tinker = not args.mlx_only
    run_mlx = not args.tinker_only

    tinker_results = None
    mlx_results = None

    # --- Tinker Cloud ---
    if run_tinker:
        api_key = os.environ.get("TINKER_API_KEY")
        if not api_key:
            print("\nWARNING: TINKER_API_KEY not set. Skipping Tinker cloud experiments.")
            print("  Set it in .env or environment to enable cloud benchmarks.")
            run_tinker = False
        else:
            print(f"\n{'#'*60}")
            print(f"  TINKER CLOUD EXPERIMENTS")
            print(f"{'#'*60}")
            try:
                client = tinker.ServiceClient()
                tinker_results = await run_all_experiments(
                    client,
                    "tinker",
                    train_data,
                    eval_data,
                    tokenizer,
                    args.sft_steps,
                    args.rl_steps,
                    args.skip_inference,
                    args.skip_sft,
                    args.skip_rl,
                )
                out_path = output_dir / "tinker_results.json"
                with open(out_path, "w") as f:
                    json.dump(tinker_results, f, indent=2, default=str)
                print(f"\nTinker results saved to {out_path}")
            except Exception as e:
                print(f"\nTinker cloud experiments failed: {e}")
                import traceback

                traceback.print_exc()

    # --- mlx-tinker Local (direct backend, no server needed) ---
    if run_mlx:
        print(f"\n{'#'*60}")
        print(f"  MLX-TINKER LOCAL EXPERIMENTS (direct backend)")
        print(f"{'#'*60}")

        try:
            mlx_results = await run_mlx_experiments(
                train_data,
                eval_data,
                tokenizer,
                args.sft_steps,
                args.rl_steps,
                args.skip_inference,
                args.skip_sft,
                args.skip_rl,
            )
            out_path = output_dir / "mlx_results.json"
            with open(out_path, "w") as f:
                json.dump(mlx_results, f, indent=2, default=str)
            print(f"\nmlx-tinker results saved to {out_path}")
        except Exception as e:
            print(f"\nmlx-tinker experiments failed: {e}")
            import traceback

            traceback.print_exc()

    # --- Generate Report ---
    if tinker_results or mlx_results:
        report = generate_report(tinker_results, mlx_results)
        report_path = output_dir / "benchmark_comparison_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")
    else:
        print("\nNo results to report. Check backend connectivity.")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    asyncio.run(main())
