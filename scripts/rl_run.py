"""RL and Combined method training + full curriculum evaluation.

Supports both pure RL (PPO with GRPO advantages) and Combined (OPD + RL)
with hint-enhanced teacher distillation. Evaluates on ALL curriculum tasks
(trainable + eval-only) for both baseline and final checkpoints.

For combined method on single-model-in-memory: the teacher uses the same
model weights but with a hint-enhanced prompt (success criteria appended).
This creates a meaningful OPD distillation signal even without a separate
teacher model.

Usage:
    # RL training (16 steps, 4 rollouts)
    uv run python scripts/rl_run.py --method rl --steps 16 --rollouts 4

    # Combined training
    uv run python scripts/rl_run.py --method combine --steps 16 --rollouts 4

    # Custom output
    uv run python scripts/rl_run.py --method rl --output reports/my_run.json
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
    batch_to_datums_combined,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_hint(task: TaskSpec) -> str:
    """Build textual hint from task success checks for teacher prompt."""
    hints: list[str] = []
    for check in task.success_checks:
        if check.kind == "contains_all":
            hints.append(f"Response must contain: {', '.join(check.values)}")
        elif check.kind == "contains_any":
            hints.append(f"Response should mention: {', '.join(check.values)}")
    return "; ".join(hints)


def build_prompt(tokenizer, task: TaskSpec) -> tuple[str, list[int]]:
    """Build prompt text and token IDs from task spec."""
    messages = [
        {"role": "system", "content": task.system_setup},
        {"role": "user", "content": task.user_turns[0].content},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return text, list(tokens)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


async def run_rollout(
    sampling_client,
    tokenizer,
    task: TaskSpec,
    max_tokens: int = 512,
) -> tuple[float, TrainingSample]:
    """Single rollout: prompt -> sample -> evaluate."""
    prompt_text, prompt_tokens = build_prompt(tokenizer, task)

    chunk = tinker.EncodedTextChunk(tokens=prompt_tokens, type="encoded_text")
    model_input = tinker.ModelInput(chunks=[chunk])
    params = tinker.SamplingParams(temperature=0.7, max_tokens=max_tokens, top_p=0.95)

    result = await sampling_client.sample_async(
        prompt=model_input, num_samples=1, sampling_params=params
    )

    seq = result.sequences[0]
    response_tokens = list(seq.tokens)
    response_logprobs = [float(lp) for lp in seq.logprobs]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

    evaluation = evaluate_task_output(task, response_text)

    sample = TrainingSample(
        task_id=task.task_id,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        response_logprobs=response_logprobs,
        reward=evaluation.reward,
        prompt_text=prompt_text,
        response_text=response_text,
        loss_mask=[1] * len(response_tokens),
    )
    return evaluation.reward, sample


# ---------------------------------------------------------------------------
# Teacher logprob extraction (combined method)
# ---------------------------------------------------------------------------


async def extract_teacher_logprobs(
    sampling_client,
    tokenizer,
    task: TaskSpec,
    sample: TrainingSample,
) -> list[float]:
    """Extract teacher logprobs using hint-enhanced prompt.

    Even with single-model-in-memory, the hint creates a meaningful
    distillation signal by giving the teacher privileged information
    about the desired output.
    """
    resp_len = len(sample.response_tokens)
    if resp_len == 0:
        return []

    hint = build_hint(task)
    enhanced_prompt = (
        f"{sample.prompt_text}\n\nHint: {hint}" if hint else sample.prompt_text
    )

    try:
        enhanced_tokens = list(
            tokenizer.encode(enhanced_prompt, add_special_tokens=False)
        )
    except TypeError:
        enhanced_tokens = list(tokenizer.encode(enhanced_prompt))

    full_text = enhanced_prompt + sample.response_text
    try:
        full_tokens = list(tokenizer.encode(full_text, add_special_tokens=False))
    except TypeError:
        full_tokens = list(tokenizer.encode(full_text))

    prompt_len = len(enhanced_tokens)

    model_input = tinker.ModelInput.from_ints(full_tokens)
    params = tinker.SamplingParams(temperature=0.0, max_tokens=1)

    try:
        result = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=params,
            include_prompt_logprobs=True,
            topk_prompt_logprobs=0,
        )

        raw_lps = result.prompt_logprobs or []
        teacher_lps = [
            float(lp) if lp is not None else 0.0 for lp in raw_lps[prompt_len:]
        ]

        if len(teacher_lps) > resp_len:
            teacher_lps = teacher_lps[:resp_len]
        elif len(teacher_lps) < resp_len:
            teacher_lps += [0.0] * (resp_len - len(teacher_lps))

        return teacher_lps

    except Exception as e:
        logger.warning(
            "Teacher logprob extraction failed for %s: %s", sample.task_id, e
        )
        return [0.0] * resp_len


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def evaluate_all_tasks(
    sampling_client,
    tokenizer,
    tasks: list[TaskSpec],
    max_tokens: int = 512,
) -> dict:
    """Evaluate model on all tasks, return summary + per-task results."""
    results = []
    for task in tasks:
        try:
            reward, sample = await run_rollout(
                sampling_client, tokenizer, task, max_tokens=max_tokens
            )
            results.append(
                {
                    "task_id": task.task_id,
                    "passed": reward > 0,
                    "reward": reward,
                    "response_text": sample.response_text[:500],
                }
            )
            logger.info(
                "  eval %s: reward=%.1f %s",
                task.task_id,
                reward,
                "PASS" if reward > 0 else "FAIL",
            )
        except Exception as e:
            logger.warning("Eval task %s failed: %s", task.task_id, e)
            results.append(
                {
                    "task_id": task.task_id,
                    "passed": False,
                    "reward": -1.0,
                    "response_text": f"error: {e}",
                }
            )

    mean_reward = (
        sum(r["reward"] for r in results) / len(results) if results else 0.0
    )
    pass_rate = (
        sum(1 for r in results if r["passed"]) / len(results) if results else 0.0
    )
    return {"mean_reward": mean_reward, "pass_rate": pass_rate, "results": results}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(description="RL / Combined training + full eval")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--method", choices=["rl", "combine"], default="rl")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--loss-fn", default="ppo")
    parser.add_argument("--w-opd", type=float, default=1.0)
    parser.add_argument("--w-rl", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-response-tokens", type=int, default=2048)
    parser.add_argument("--max-gen-tokens", type=int, default=512)
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            f"workspace_reports/openclaw_local/{args.method}_report.json"
        )

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("=== %s training: %d steps, %d rollouts ===", args.method.upper(), args.steps, args.rollouts)
    logger.info("Model: %s  LoRA rank: %d  LR: %s", args.model, args.lora_rank, args.lr)
    if args.method == "combine":
        logger.info("Combined weights: w_opd=%.2f  w_rl=%.2f", args.w_opd, args.w_rl)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    sc = tinker.ServiceClient()
    training_client = await sc.create_lora_training_client_async(
        base_model=args.model, rank=args.lora_rank
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        name=f"{args.method}_bootstrap"
    )

    # Load curriculum
    curriculum = load_curriculum(
        Path("integrations/openclaw_rl_local/curriculum/wildclaw_text_v1.yaml")
    )
    train_tasks, eval_tasks = split_curriculum(curriculum)
    all_tasks = train_tasks + eval_tasks
    logger.info(
        "Curriculum: %d train + %d eval = %d total tasks",
        len(train_tasks), len(eval_tasks), len(all_tasks),
    )

    # ---- Baseline evaluation (all tasks) ----
    logger.info("Running baseline evaluation on all %d tasks...", len(all_tasks))
    baseline_eval = await evaluate_all_tasks(
        sampling_client, tokenizer, all_tasks, max_tokens=args.max_gen_tokens
    )
    logger.info(
        "Baseline: mean_reward=%.2f  pass_rate=%.0f%%",
        baseline_eval["mean_reward"],
        baseline_eval["pass_rate"] * 100,
    )

    # ---- Training loop ----
    step_reports: list[dict] = []

    for step in range(1, args.steps + 1):
        task = train_tasks[(step - 1) % len(train_tasks)]
        rewards: list[float] = []
        samples: list[TrainingSample] = []

        # Rollouts
        for r_idx in range(args.rollouts):
            try:
                reward, sample = await run_rollout(
                    sampling_client, tokenizer, task, max_tokens=args.max_gen_tokens
                )
                rewards.append(reward)
                samples.append(sample)
            except Exception as e:
                logger.warning("Step %d rollout %d failed: %s", step, r_idx, e)

        # Combined: extract teacher logprobs for each sample
        if args.method == "combine" and samples:
            enriched: list[TrainingSample] = []
            for sample in samples:
                teacher_lps = await extract_teacher_logprobs(
                    sampling_client, tokenizer, task, sample
                )
                has_teacher = any(lp != 0.0 for lp in teacher_lps)
                has_reward = sample.reward != 0.0

                if has_teacher and has_reward:
                    sample_type = "opd+rl"
                elif has_teacher:
                    sample_type = "opd"
                elif has_reward:
                    sample_type = "rl"
                else:
                    sample_type = ""

                enriched.append(
                    TrainingSample(
                        task_id=sample.task_id,
                        prompt_tokens=sample.prompt_tokens,
                        response_tokens=sample.response_tokens,
                        response_logprobs=sample.response_logprobs,
                        reward=sample.reward,
                        prompt_text=sample.prompt_text,
                        response_text=sample.response_text,
                        teacher_logprobs=teacher_lps,
                        loss_mask=sample.loss_mask,
                        sample_type=sample_type,
                    )
                )
            samples = enriched

        # Build datums and train
        losses: list[float] = []
        if samples:
            if args.method == "combine":
                datums = batch_to_datums_combined(
                    samples,
                    w_opd=args.w_opd,
                    w_rl=args.w_rl,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_response_tokens=args.max_response_tokens,
                )
            else:
                datums = batch_to_datums(
                    samples,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_response_tokens=args.max_response_tokens,
                )

            if datums:
                step_loss = 0.0
                for i in range(0, len(datums), args.batch_size):
                    micro = datums[i : i + args.batch_size]
                    fb_future = await training_client.forward_backward_async(
                        micro, loss_fn=args.loss_fn
                    )
                    fb_result = await fb_future.result_async(timeout=600)
                    step_loss += float(fb_result.metrics.get("loss:sum", 0.0))
                losses.append(step_loss)

                opt_future = await training_client.optim_step_async(
                    tinker.AdamParams(learning_rate=args.lr)
                )
                await opt_future.result_async(timeout=600)

                sampling_client = (
                    await training_client.save_weights_and_get_sampling_client_async(
                        name=f"{args.method}_step_{step:04d}"
                    )
                )

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        passed = sum(1 for r in rewards if r > 0)

        sample_types = {"opd+rl": 0, "opd": 0, "rl": 0, "": 0}
        for s in samples:
            key = s.sample_type if s.sample_type in sample_types else ""
            sample_types[key] += 1

        step_reports.append(
            {
                "step": step,
                "task_id": task.task_id,
                "mean_reward": mean_reward,
                "passed_rollouts": passed,
                "losses": losses,
                "method": args.method,
                "sample_types": sample_types,
            }
        )

        logger.info(
            "Step %d/%d: %s  reward=%.1f  passed=%d/%d  loss=%s",
            step,
            args.steps,
            task.task_id,
            mean_reward,
            passed,
            args.rollouts,
            [f"{l:.1f}" for l in losses],
        )

    # ---- Final evaluation (all tasks) ----
    logger.info("Running final evaluation on all %d tasks...", len(all_tasks))
    final_eval = await evaluate_all_tasks(
        sampling_client, tokenizer, all_tasks, max_tokens=args.max_gen_tokens
    )
    logger.info(
        "Final: mean_reward=%.2f  pass_rate=%.0f%%",
        final_eval["mean_reward"],
        final_eval["pass_rate"] * 100,
    )

    # ---- Write report ----
    report = {
        "model_name": args.model,
        "method": args.method,
        "loss_fn": args.loss_fn,
        "curriculum_path": "integrations/openclaw_rl_local/curriculum/wildclaw_text_v1.yaml",
        "runtime": "direct",
        "provider_base_url": "direct_tinker_sdk",
        "w_opd": args.w_opd,
        "w_rl": args.w_rl,
        "train_epochs": 1,
        "rollouts_per_step": args.rollouts,
        "lora_rank": args.lora_rank,
        "learning_rate": args.lr,
        "baseline_eval": baseline_eval,
        "final_eval": final_eval,
        "steps": step_reports,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", out)

    # Summary
    baseline_pr = baseline_eval["pass_rate"] * 100
    final_pr = final_eval["pass_rate"] * 100
    positive_steps = sum(1 for s in step_reports if s["mean_reward"] > 0)
    logger.info(
        "=== SUMMARY: %s | Baseline %.0f%% -> Final %.0f%% | %d/%d steps positive ===",
        args.method.upper(), baseline_pr, final_pr, positive_steps, args.steps,
    )

    sc.holder.close()


if __name__ == "__main__":
    asyncio.run(main())
