"""Helpers for local-only, sequential UAT smoke runs."""

from __future__ import annotations

import asyncio
import difflib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import tinker

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "uat"
REPORT_DIR = Path(__file__).resolve().parents[2] / "workspace_reports" / "uat"
MODEL_NAME = os.environ.get("MLX_TINKER_UAT_MODEL", "Qwen/Qwen3.5-0.8B")
LORA_RANK = int(os.environ.get("MLX_TINKER_UAT_LORA_RANK", "8"))
LEARNING_RATE = float(os.environ.get("MLX_TINKER_UAT_LR", "1e-4"))
RL_LEARNING_RATE = float(os.environ.get("MLX_TINKER_UAT_RL_LR", "5e-5"))
BATCH_SIZE = int(os.environ.get("MLX_TINKER_UAT_BATCH_SIZE", "4"))
MAX_SEQ_LEN = int(os.environ.get("MLX_TINKER_UAT_MAX_SEQ_LEN", "768"))
LOCAL_BASE_URL = os.environ.get("MLX_TINKER_UAT_LOCAL_BASE_URL", "http://127.0.0.1:8010")
LOCAL_API_KEY = os.environ.get("MLX_TINKER_UAT_LOCAL_API_KEY", "tml-local")
TRAIN_EXAMPLE_LIMIT = int(os.environ.get("MLX_TINKER_UAT_TRAIN_EXAMPLES", "40"))
EVAL_EXAMPLE_LIMIT = int(os.environ.get("MLX_TINKER_UAT_EVAL_EXAMPLES", "10"))
SFT_STEPS = int(os.environ.get("MLX_TINKER_UAT_SFT_STEPS", "50"))
RL_STEPS = int(os.environ.get("MLX_TINKER_UAT_RL_STEPS", "50"))
RL_NUM_ROLLOUTS = int(os.environ.get("MLX_TINKER_UAT_RL_NUM_ROLLOUTS", "8"))
REQUEST_TIMEOUT_S = float(os.environ.get("MLX_TINKER_UAT_REQUEST_TIMEOUT_S", "90"))
HEAVY_REQUEST_TIMEOUT_S = float(
    os.environ.get(
        "MLX_TINKER_UAT_HEAVY_REQUEST_TIMEOUT_S",
        str(max(REQUEST_TIMEOUT_S, 180.0)),
    )
)
TOTAL_TIMEOUT_S = float(os.environ.get("MLX_TINKER_UAT_TOTAL_TIMEOUT_S", "5400"))
SFT_REGRESSION_TOLERANCE = float(os.environ.get("MLX_TINKER_UAT_SFT_REGRESSION_TOL", "-0.05"))
RL_REGRESSION_TOLERANCE = float(os.environ.get("MLX_TINKER_UAT_RL_REGRESSION_TOL", "-0.20"))


async def resolve_client_result(result, timeout_s: float = REQUEST_TIMEOUT_S):
    if hasattr(result, "result_async"):
        return await asyncio.wait_for(
            result.result_async(timeout=timeout_s),
            timeout=timeout_s + 1.0,
        )
    if hasattr(result, "result"):
        return result.result()
    return result


async def call_with_timeout(awaitable, timeout_s: float, label: str):
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_s)
    except TimeoutError as exc:
        raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s") from exc


def load_examples(dataset_name: str, split: str, limit: int | None = None) -> list[dict]:
    path = FIXTURE_DIR / f"{dataset_name}_{split}.json"
    with open(path) as f:
        data = json.load(f)
    return data[:limit] if limit is not None else data


def normalize_text(text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_sql(text: str) -> str | None:
    match = re.search(r"(SELECT\b[^;]*)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def normalize_table_name(sql: str) -> str:
    return re.sub(r"\bFROM\s+table\b", "FROM data", sql, flags=re.IGNORECASE)


def normalize_sql_text(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().lower()


def execute_sql(sql: str, example: dict) -> list | None:
    import sqlite3

    header = example.get("columns", [])
    rows = example.get("rows", [])
    if not header or not rows:
        return None
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    try:
        cols_def = ", ".join(f'"{col}" TEXT' for col in header)
        cur.execute(f"CREATE TABLE data ({cols_def})")
        placeholders = ", ".join("?" * len(header))
        cur.executemany(
            f"INSERT INTO data VALUES ({placeholders})",
            [tuple(str(v) for v in row) for row in rows],
        )
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return None
    finally:
        conn.close()


def wikisql_score_prediction(text: str, example: dict) -> bool:
    sql = extract_sql(text)
    if sql is None:
        return False
    pred = execute_sql(normalize_table_name(sql), example)
    gold = execute_sql(normalize_table_name(example["sql"]), example)
    return pred is not None and gold is not None and set(map(tuple, pred)) == set(map(tuple, gold))


def wikisql_reward_prediction(text: str, example: dict) -> float:
    """Provide denser RL reward while keeping exact execution-match for eval."""
    sql = extract_sql(text)
    if sql is None:
        return -1.0

    normalized_sql = normalize_table_name(sql)
    normalized_gold = normalize_table_name(example["sql"])
    similarity = difflib.SequenceMatcher(
        None,
        normalize_sql_text(normalized_sql),
        normalize_sql_text(normalized_gold),
    ).ratio()

    pred = execute_sql(normalized_sql, example)
    gold = execute_sql(normalized_gold, example)
    if pred is None or gold is None:
        return -0.5 + 0.5 * similarity

    pred_rows = set(map(tuple, pred))
    gold_rows = set(map(tuple, gold))
    if pred_rows == gold_rows:
        return 1.0
    overlap_bonus = 0.2 if pred_rows & gold_rows else 0.0
    non_empty_bonus = 0.1 if pred_rows else 0.0
    return min(0.9, 0.1 + 0.5 * similarity + overlap_bonus + non_empty_bonus)


def cuad_score_prediction(text: str, example: dict) -> bool:
    return normalize_text(text) == normalize_text(example["answer"])


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    max_tokens: int

    def make_prompt(self, example: dict) -> str:
        if self.name == "wikisql":
            sample_rows = "\n".join(" | ".join(str(v) for v in row) for row in example["rows"][:3])
            return (
                f'Table columns: {" | ".join(example["columns"])}\n'
                f"Sample data:\n{sample_rows}\n"
                f'Question: {example["question"]}\n'
                "SQL: "
            )
        return (
            f'Clause type: {example["clause_type"]}\n'
            f'Question: {example["question"]}\n'
            f'Contract excerpt:\n{example["context"]}\n\n'
            "Answer:"
        )

    def make_completion(self, example: dict) -> str:
        if self.name == "wikisql":
            return " " + example["sql"]
        return " " + example["answer"]

    def is_correct(self, text: str, example: dict) -> bool:
        if self.name == "wikisql":
            return wikisql_score_prediction(text, example)
        return cuad_score_prediction(text, example)

    def reward(self, text: str, example: dict) -> float:
        if self.name == "wikisql":
            return wikisql_reward_prediction(text, example)
        return 1.0 if cuad_score_prediction(text, example) else -1.0


WIKISQL_SPEC = DatasetSpec(name="wikisql", max_tokens=96)
CUAD_SPEC = DatasetSpec(name="cuad", max_tokens=48)


def get_local_service_client() -> tinker.ServiceClient:
    return tinker.ServiceClient(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)


def build_sft_datum(tokenizer, spec: DatasetSpec, example: dict) -> tinker.Datum:
    prompt = spec.make_prompt(example)
    completion = spec.make_completion(example)
    prompt_tokens = tokenizer.encode(prompt)
    completion_tokens = tokenizer.encode(completion)
    all_tokens = (prompt_tokens + completion_tokens)[:MAX_SEQ_LEN]
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    n_prompt = min(len(prompt_tokens) - 1, len(target_tokens))
    weights = [0.0] * n_prompt + [1.0] * (len(target_tokens) - n_prompt)
    weights = weights[: len(target_tokens)]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def build_rl_datum(
    tokenizer,
    spec: DatasetSpec,
    example: dict,
    generated_tokens: list[int],
    generated_logprobs: list[float],
    advantage: float,
) -> tinker.Datum:
    prompt_tokens = tokenizer.encode(spec.make_prompt(example))
    full_tokens = (prompt_tokens + list(generated_tokens))[:MAX_SEQ_LEN]
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    n_prompt = min(len(prompt_tokens) - 1, len(target_tokens))
    n_gen = max(len(target_tokens) - n_prompt, 0)

    advantages = [0.0] * n_prompt + [advantage] * n_gen
    old_logprobs = [0.0] * n_prompt + list(generated_logprobs[:n_gen])
    while len(old_logprobs) < len(target_tokens):
        old_logprobs.append(0.0)

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "advantages": advantages[: len(target_tokens)],
            "logprobs": old_logprobs[: len(target_tokens)],
        },
    )


def compute_advantages(rewards: list[float]) -> list[float]:
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return [reward - mean_reward for reward in rewards]


async def evaluate_sampling_client(
    sampling_client,
    tokenizer,
    spec: DatasetSpec,
    eval_examples: list[dict],
) -> float:
    correct = 0
    for example in eval_examples:
        prompt_tokens = tokenizer.encode(spec.make_prompt(example))
        response = await call_with_timeout(
            sampling_client.sample_async(
                prompt=tinker.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=tinker.SamplingParams(temperature=0.0, max_tokens=spec.max_tokens),
            ),
            REQUEST_TIMEOUT_S,
            "sample_async(eval)",
        )
        generated = tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True)
        if spec.is_correct(generated, example):
            correct += 1
    return correct / len(eval_examples)


async def create_training_client(service_client: tinker.ServiceClient):
    return await call_with_timeout(
        service_client.create_lora_training_client_async(
            base_model=MODEL_NAME,
            rank=LORA_RANK,
        ),
        HEAVY_REQUEST_TIMEOUT_S,
        "create_lora_training_client_async",
    )


async def evaluate_training_client(
    training_client,
    tokenizer,
    spec: DatasetSpec,
    eval_examples: list[dict],
) -> float:
    sampling_client = await call_with_timeout(
        training_client.save_weights_and_get_sampling_client_async(),
        HEAVY_REQUEST_TIMEOUT_S,
        "save_weights_and_get_sampling_client_async(eval)",
    )
    return await evaluate_sampling_client(sampling_client, tokenizer, spec, eval_examples)


async def run_sft_phase(
    training_client,
    tokenizer,
    spec: DatasetSpec,
    train_examples: list[dict],
    *,
    num_steps: int,
) -> list[float]:
    losses: list[float] = []
    for step in range(num_steps):
        batch = []
        for b in range(BATCH_SIZE):
            idx = (step * BATCH_SIZE + b) % len(train_examples)
            batch.append(build_sft_datum(tokenizer, spec, train_examples[idx]))

        fb_future = await call_with_timeout(
            training_client.forward_backward_async(batch, loss_fn="cross_entropy"),
            HEAVY_REQUEST_TIMEOUT_S,
            "forward_backward_async(sft)",
        )
        fb_result = await resolve_client_result(fb_future, timeout_s=HEAVY_REQUEST_TIMEOUT_S)
        loss_sum = float(fb_result.metrics.get("loss:sum", 0.0))
        if not math.isfinite(loss_sum):
            raise AssertionError(f"Non-finite SFT loss at step {step}: {loss_sum}")
        losses.append(loss_sum)

        opt_future = await call_with_timeout(
            training_client.optim_step_async(tinker.AdamParams(learning_rate=LEARNING_RATE)),
            HEAVY_REQUEST_TIMEOUT_S,
            "optim_step_async(sft)",
        )
        await resolve_client_result(opt_future, timeout_s=HEAVY_REQUEST_TIMEOUT_S)
    return losses


async def run_rl_phase(
    service_client: tinker.ServiceClient,
    training_client,
    tokenizer,
    spec: DatasetSpec,
    train_examples: list[dict],
    *,
    num_steps: int,
    num_rollouts: int,
) -> dict:
    losses: list[float] = []
    mean_rewards: list[float] = []
    non_zero_advantage_steps = 0

    for step in range(num_steps):
        example = train_examples[step % len(train_examples)]
        prompt_tokens = tokenizer.encode(spec.make_prompt(example))

        sampling_client = await call_with_timeout(
            training_client.save_weights_and_get_sampling_client_async(),
            HEAVY_REQUEST_TIMEOUT_S,
            f"save_weights_and_get_sampling_client_async(step={step})",
        )
        response = await call_with_timeout(
            sampling_client.sample_async(
                prompt=tinker.ModelInput.from_ints(prompt_tokens),
                num_samples=num_rollouts,
                sampling_params=tinker.SamplingParams(temperature=0.8, max_tokens=spec.max_tokens),
            ),
            HEAVY_REQUEST_TIMEOUT_S,
            f"sample_async(rl, step={step})",
        )
        if len(response.sequences) != num_rollouts:
            raise AssertionError(
                f"Expected {num_rollouts} rollouts at step {step}, got {len(response.sequences)}"
            )

        rewards = []
        datums = []
        for seq in response.sequences:
            gen_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            reward = spec.reward(gen_text, example)
            rewards.append(reward)

        advantages = compute_advantages(rewards)
        if any(abs(adv) > 1e-6 for adv in advantages):
            non_zero_advantage_steps += 1

        for seq, advantage in zip(response.sequences, advantages, strict=True):
            datums.append(
                build_rl_datum(
                    tokenizer,
                    spec,
                    example,
                    seq.tokens,
                    seq.logprobs or [0.0] * len(seq.tokens),
                    advantage,
                )
            )

        loss_sum = 0.0
        for start in range(0, len(datums), BATCH_SIZE):
            rl_batch = datums[start : start + BATCH_SIZE]
            fb_future = await call_with_timeout(
                training_client.forward_backward_async(rl_batch, loss_fn="importance_sampling"),
                HEAVY_REQUEST_TIMEOUT_S,
                f"forward_backward_async(rl, step={step}, batch_start={start})",
            )
            fb_result = await resolve_client_result(fb_future, timeout_s=HEAVY_REQUEST_TIMEOUT_S)
            batch_loss = float(fb_result.metrics.get("loss:sum", 0.0))
            if not math.isfinite(batch_loss):
                raise AssertionError(
                    f"Non-finite RL loss at step {step}, batch_start={start}: {batch_loss}"
                )
            loss_sum += batch_loss

        losses.append(loss_sum)
        mean_rewards.append(sum(rewards) / len(rewards))

        opt_future = await call_with_timeout(
            training_client.optim_step_async(tinker.AdamParams(learning_rate=RL_LEARNING_RATE)),
            HEAVY_REQUEST_TIMEOUT_S,
            f"optim_step_async(rl, step={step})",
        )
        await resolve_client_result(opt_future, timeout_s=HEAVY_REQUEST_TIMEOUT_S)

    return {
        "losses": losses,
        "mean_rewards": mean_rewards,
        "non_zero_advantage_steps": non_zero_advantage_steps,
        "num_steps": num_steps,
        "num_rollouts": num_rollouts,
        "group_size": num_rollouts,
    }


async def run_sft_backend_uat(
    service_client: tinker.ServiceClient,
    tokenizer,
    spec: DatasetSpec,
    train_examples: list[dict],
    eval_examples: list[dict],
) -> dict:
    t_start = time.time()
    base_sampling_client = await call_with_timeout(
        service_client.create_sampling_client_async(base_model=MODEL_NAME),
        REQUEST_TIMEOUT_S,
        "create_sampling_client_async(base)",
    )
    base_accuracy = await evaluate_sampling_client(base_sampling_client, tokenizer, spec, eval_examples)

    training_client = await create_training_client(service_client)
    losses = await run_sft_phase(training_client, tokenizer, spec, train_examples, num_steps=SFT_STEPS)
    tuned_accuracy = await evaluate_training_client(training_client, tokenizer, spec, eval_examples)

    return {
        "base_accuracy": base_accuracy,
        "tuned_accuracy": tuned_accuracy,
        "improvement": tuned_accuracy - base_accuracy,
        "num_steps": SFT_STEPS,
        "batch_size": BATCH_SIZE,
        "losses": losses,
        "total_time_s": round(time.time() - t_start, 3),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
    }


async def run_sft_then_rl_backend_uat(
    service_client: tinker.ServiceClient,
    tokenizer,
    spec: DatasetSpec,
    train_examples: list[dict],
    eval_examples: list[dict],
    *,
    sft_steps: int = SFT_STEPS,
    rl_steps: int = RL_STEPS,
    num_rollouts: int = RL_NUM_ROLLOUTS,
) -> dict:
    """Run realistic local UAT: SFT warm-start followed by RL continuation."""
    if spec.name != "wikisql":
        raise ValueError("RL UAT currently supports WikiSQL only")

    t_start = time.time()
    base_sampling_client = await call_with_timeout(
        service_client.create_sampling_client_async(base_model=MODEL_NAME),
        REQUEST_TIMEOUT_S,
        "create_sampling_client_async(base)",
    )
    base_accuracy = await evaluate_sampling_client(base_sampling_client, tokenizer, spec, eval_examples)

    training_client = await create_training_client(service_client)

    sft_losses = await run_sft_phase(
        training_client,
        tokenizer,
        spec,
        train_examples,
        num_steps=sft_steps,
    )
    sft_accuracy = await evaluate_training_client(training_client, tokenizer, spec, eval_examples)

    rl_phase = await run_rl_phase(
        service_client,
        training_client,
        tokenizer,
        spec,
        train_examples,
        num_steps=rl_steps,
        num_rollouts=num_rollouts,
    )
    rl_accuracy = await evaluate_training_client(training_client, tokenizer, spec, eval_examples)

    return {
        "base_accuracy": base_accuracy,
        "sft_accuracy": sft_accuracy,
        "rl_accuracy": rl_accuracy,
        "sft_improvement": sft_accuracy - base_accuracy,
        "rl_improvement_over_sft": rl_accuracy - sft_accuracy,
        "rl_improvement_over_base": rl_accuracy - base_accuracy,
        "sft": {
            "num_steps": sft_steps,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "losses": sft_losses,
        },
        "rl": {
            **rl_phase,
            "learning_rate": RL_LEARNING_RATE,
        },
        "total_time_s": round(time.time() - t_start, 3),
    }


def write_report(dataset_name: str, report: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORT_DIR / f"{dataset_name}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
