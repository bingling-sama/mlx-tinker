#!/usr/bin/env python3
"""One-time script: Generate reference training logs via the Tinker API.

Runs SFT and RL training on WikiSQL data against the real Tinker API,
records per-step metrics, and saves to tests/fixtures/tinker_reference_logs.json.

This is run ONCE to create golden reference data. The output is committed
to the repo and used by tests/stress/test_tinker_equivalence.py to verify
that mlx-tinker produces comparable results.

Usage:
    # Requires TINKER_API_KEY in .env
    uv run python scripts/generate_reference_logs.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TINKER_API_KEY = os.environ.get("TINKER_API_KEY")
if not TINKER_API_KEY:
    print("ERROR: TINKER_API_KEY not set. Add it to .env or environment.")
    sys.exit(1)

MODEL_NAME = "Qwen/Qwen3.5-4B"
LORA_RANK = 8
SFT_STEPS = 50
RL_STEPS = 10
OUTPUT_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "tinker_reference_logs.json"


async def main():
    import tinker
    from transformers import AutoTokenizer

    print("Connecting to Tinker API...")
    service_client = tinker.ServiceClient()

    print(f"Creating LoRA training client for {MODEL_NAME} (rank={LORA_RANK})...")
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME,
        rank=LORA_RANK,
    )

    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load WikiSQL data from local fixture (same data used by stress tests)
    fixtures_path = Path(__file__).parent.parent / "tests" / "fixtures" / "wikisql_subset.json"
    if not fixtures_path.exists():
        print(f"ERROR: WikiSQL fixture not found at {fixtures_path}")
        sys.exit(1)
    with open(fixtures_path) as f:
        all_examples = json.load(f)
    wikisql_examples = all_examples[:SFT_STEPS]

    print(f"Loaded {len(wikisql_examples)} WikiSQL examples")

    def format_example(example):
        cols_str = " | ".join(example["columns"])
        sql = example["sql"]
        prompt = f"Table: {cols_str}\nQuestion: {example['question']}\nSQL: "
        return prompt, sql

    # --- SFT Training ---
    print(f"\n=== Running {SFT_STEPS} SFT steps ===")
    sft_logs = []

    for step in range(SFT_STEPS):
        example = wikisql_examples[step % len(wikisql_examples)]
        prompt, sql = format_example(example)
        full_text = prompt + sql

        tokens = tokenizer.encode(full_text)[:256]
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "weights": [1.0] * len(target_tokens),
            },
        )

        fb_result = await training_client.forward_backward_async(
            [datum], loss_fn="cross_entropy"
        )
        if hasattr(fb_result, "result_async"):
            fb_result = await fb_result.result_async()
        else:
            fb_result = fb_result.result()

        loss_sum = fb_result.metrics.get(
            "loss:sum", fb_result.metrics.get("mean_loss", 0.0)
        )
        n_tgt = len(target_tokens)
        mean_loss = loss_sum / n_tgt if n_tgt > 0 else loss_sum

        # Optim step
        adam_params = tinker.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.999, eps=1e-8)
        optim_result = await training_client.optim_step_async(adam_params)
        if hasattr(optim_result, "result_async"):
            await optim_result.result_async()
        else:
            optim_result.result()

        sft_logs.append({
            "step": step,
            "loss": mean_loss,
            "loss_mean": mean_loss,
            "loss_sum": loss_sum,
            "n_tokens": n_tgt,
            "question": example["question"],
        })

        if step % 10 == 0:
            print(f"  Step {step}: mean_loss={mean_loss:.4f} (sum={loss_sum:.4f}, n={n_tgt})")

    # --- RL Training (cookbook pattern: target_tokens + logprobs + advantages) ---
    print(f"\n=== Running {RL_STEPS} RL steps ===")
    rl_logs = []

    for step in range(RL_STEPS):
        example = wikisql_examples[step % len(wikisql_examples)]
        prompt, gold_sql = format_example(example)

        full_text = prompt + gold_sql
        tokens = tokenizer.encode(full_text)[:256]
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        prompt_tokens = tokenizer.encode(prompt)
        n_prompt = len(prompt_tokens) - 1
        n_tgt = len(target_tokens)

        # advantages: 0 for prompt observation, +1.0 for action tokens
        advantages = [0.0] * min(n_prompt, n_tgt)
        advantages += [1.0] * max(0, n_tgt - n_prompt)
        advantages = advantages[:n_tgt]
        # logprobs: 0 placeholder (no reference rollout logprobs)
        logprobs = [0.0] * n_tgt

        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "logprobs": logprobs,
                "advantages": advantages,
            },
        )

        try:
            fb_result = await training_client.forward_backward_async(
                [datum], loss_fn="importance_sampling"
            )
            if hasattr(fb_result, "result_async"):
                fb_result = await fb_result.result_async()
            else:
                fb_result = fb_result.result()

            loss_sum = fb_result.metrics.get(
                "loss:sum", fb_result.metrics.get("mean_loss", 0.0)
            )
            mean_loss = loss_sum / n_tgt if n_tgt > 0 else loss_sum

            adam_params = tinker.AdamParams(
                learning_rate=5e-5, beta1=0.9, beta2=0.999, eps=1e-8
            )
            optim_result = await training_client.optim_step_async(adam_params)
            if hasattr(optim_result, "result_async"):
                await optim_result.result_async()
            else:
                optim_result.result()

            rl_logs.append({
                "step": step,
                "loss": mean_loss,
                "loss_mean": mean_loss,
                "loss_sum": loss_sum,
                "n_tokens": n_tgt,
                "question": example["question"],
            })
            print(f"  RL Step {step}: mean_loss={mean_loss:.4f} (sum={loss_sum:.4f})")
        except tinker.BadRequestError as e:
            print(f"  RL failed at step {step}: {e}")
            break

    # --- Save ---
    output = {
        "model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "tokenization": "model_tokenizer",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sft_steps": sft_logs,
        "rl_steps": rl_logs,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Reference logs saved to {OUTPUT_PATH} ===")
    first_sft = sft_logs[0]["loss"]
    last_sft = sft_logs[-1]["loss"]
    print(f"  SFT: {len(sft_logs)} steps, mean_loss {first_sft:.4f} -> {last_sft:.4f}")
    if rl_logs:
        first_rl = rl_logs[0]["loss"]
        last_rl = rl_logs[-1]["loss"]
        print(f"  RL:  {len(rl_logs)} steps, mean_loss {first_rl:.4f} -> {last_rl:.4f}")
    else:
        print("  RL:  0 steps (API did not support IS loss)")


if __name__ == "__main__":
    asyncio.run(main())
