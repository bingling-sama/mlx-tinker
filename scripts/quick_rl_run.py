"""Quick RL training run for reward curve plot.

Bypasses the OpenClaw gateway (slow Docker path) and runs rollouts
directly against the mlx-tinker proxy using simple text evaluation.

Usage:
    uv run python scripts/quick_rl_run.py [--steps 16] [--rollouts 4]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import tinker
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrations.openclaw_rl_local.curriculum import (
    TaskSpec,
    evaluate_task_output,
    load_curriculum,
    split_curriculum,
)
from integrations.openclaw_rl_local.data_formatter import (
    TrainingSample,
    batch_to_datums,
    compute_grpo_advantages,
)

logger = logging.getLogger(__name__)


async def run_rollout(
    sampling_client,
    tokenizer,
    task: TaskSpec,
) -> tuple[float, TrainingSample | None]:
    """Run a single rollout: prompt → sample → evaluate."""
    # Build prompt from task
    prompt = task.system_setup + "\n\nUser request:\n" + task.user_turns[0].content
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Apply chat template
    messages = [
        {"role": "system", "content": task.system_setup},
        {"role": "user", "content": task.user_turns[0].content},
    ]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Sample from the model
    chunk = tinker.EncodedTextChunk(tokens=list(prompt_tokens), type="encoded_text")
    model_input = tinker.ModelInput(chunks=[chunk])
    params = tinker.SamplingParams(temperature=0.7, max_tokens=512, top_p=0.95)

    result = await sampling_client.sample_async(
        prompt=model_input, num_samples=1, sampling_params=params,
    )

    seq = result.sequences[0]
    response_tokens = list(seq.tokens)
    response_logprobs = [float(lp) for lp in seq.logprobs]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

    # Evaluate using curriculum success checks
    evaluation = evaluate_task_output(task, response_text)
    reward = evaluation.reward

    sample = TrainingSample(
        task_id=task.task_id,
        prompt_tokens=list(prompt_tokens),
        response_tokens=response_tokens,
        response_logprobs=response_logprobs,
        reward=reward,
        prompt_text=prompt_text,
        response_text=response_text,
    )
    return reward, sample


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--output", default="workspace_reports/openclaw_local/local_rl_report.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    logger.info("Loading tokenizer for %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    logger.info("Creating Tinker clients")
    sc = tinker.ServiceClient()
    training_client = await sc.create_lora_training_client_async(
        base_model=args.model, rank=args.lora_rank,
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        name="quick_rl_bootstrap",
    )

    # Load curriculum
    curriculum = load_curriculum(Path("integrations/openclaw_rl_local/curriculum/wildclaw_text_v1.yaml"))
    train_tasks, eval_tasks = split_curriculum(curriculum)
    logger.info("Loaded %d train tasks, %d eval tasks", len(train_tasks), len(eval_tasks))

    # Quick eval
    async def evaluate(tasks):
        results = []
        for task in tasks:
            reward, _ = await run_rollout(sampling_client, tokenizer, task)
            results.append({"task_id": task.task_id, "passed": reward > 0, "reward": reward})
        mean_r = sum(r["reward"] for r in results) / len(results) if results else 0
        pr = sum(1 for r in results if r["passed"]) / len(results) if results else 0
        return {"mean_reward": mean_r, "pass_rate": pr, "results": results}

    logger.info("Running baseline eval")
    baseline_eval = await evaluate(eval_tasks)
    logger.info("Baseline: mean_reward=%.2f pass_rate=%.0f%%", baseline_eval["mean_reward"], baseline_eval["pass_rate"] * 100)

    step_reports = []
    for step in range(1, args.steps + 1):
        task = train_tasks[(step - 1) % len(train_tasks)]
        rewards = []
        samples = []

        for r_idx in range(args.rollouts):
            reward, sample = await run_rollout(sampling_client, tokenizer, task)
            rewards.append(reward)
            if sample:
                samples.append(sample)

        mean_reward = sum(rewards) / len(rewards)
        passed = sum(1 for r in rewards if r > 0)

        # Train on collected samples
        losses = []
        if samples:
            advantages = compute_grpo_advantages(samples)
            datums = batch_to_datums(
                samples, advantages,
                max_prompt_tokens=2048, max_response_tokens=2048,
            )
            # Process one datum at a time to avoid OOM
            step_loss = 0.0
            for datum in datums:
                fb_future = await training_client.forward_backward_async([datum], loss_fn="ppo")
                fb_result = await fb_future.result_async(timeout=300)
                step_loss += float(fb_result.metrics.get("loss:sum", 0.0))
            losses.append(step_loss)

            opt_future = await training_client.optim_step_async(
                tinker.AdamParams(learning_rate=args.lr)
            )
            await opt_future.result_async(timeout=300)

            sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                name=f"quick_rl_step_{step:04d}",
            )

        step_reports.append({
            "step": step,
            "task_id": task.task_id,
            "mean_reward": mean_reward,
            "passed_rollouts": passed,
            "losses": losses,
        })
        logger.info(
            "Step %d/%d: %s reward=%.1f passed=%d/%d loss=%s",
            step, args.steps, task.task_id, mean_reward, passed, args.rollouts,
            [f"{l:.1f}" for l in losses],
        )

    logger.info("Running final eval")
    final_eval = await evaluate(eval_tasks)
    logger.info("Final: mean_reward=%.2f pass_rate=%.0f%%", final_eval["mean_reward"], final_eval["pass_rate"] * 100)

    report = {
        "model_name": args.model,
        "method": "rl",
        "curriculum_path": "integrations/openclaw_rl_local/curriculum/wildclaw_text_v1.yaml",
        "runtime": "direct",
        "provider_base_url": "direct_tinker_sdk",
        "baseline_eval": baseline_eval,
        "final_eval": final_eval,
        "steps": step_reports,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", out)

    sc.holder.close()


if __name__ == "__main__":
    asyncio.run(main())
