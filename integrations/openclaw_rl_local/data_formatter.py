"""Convert proxy-recorded assistant responses into Tinker training datums.

.. deprecated::
    This module duplicates data formatting logic from upstream OpenClaw-RL
    (openclaw-tinker/data_formatter.py). The canonical path is upstream.
    See ``scripts/run_openclaw_rl.sh`` for the recommended training path.

Supports all three methods:

  RL (sample_to_datum):
    advantage = scalar GRPO reward, masked by loss_mask
    Used via: batch_to_datums(samples, max_prompt_tokens, max_response_tokens)

  OPD (sample_to_datum with teacher_logprobs):
    advantage = scalar reward + per-token (teacher_lp - student_lp), masked
    Used via: batch_to_datums(samples, ...)

  Combined (sample_to_datum_combined):
    advantage = w_opd * (teacher_lp - student_lp) + w_rl * reward, per-token
    Used via: batch_to_datums_combined(samples, w_opd, w_rl, ...)

Tinker Datum convention:
  model_input   - input tokens (all but the last token of the full sequence)
  loss_fn_inputs:
    target_tokens - full sequence left-shifted by 1
    logprobs      - prompt positions = 0.0, response positions = sampled logprob
    advantages    - prompt = 0.0, response = advantage * loss_mask
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import tinker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingSample:
    task_id: str
    prompt_tokens: list[int]
    response_tokens: list[int]
    response_logprobs: list[float]
    reward: float
    prompt_text: str
    response_text: str
    teacher_logprobs: list[float] | None = None
    loss_mask: list[int] | None = None
    sample_type: str = ""  # "opd+rl", "opd", "rl", ""


def _fit(values: list[float], length: int) -> list[float]:
    if len(values) >= length:
        return list(values[:length])
    return list(values) + [0.0] * (length - len(values))


def _sanitize(values: list[float], *, label: str, task_id: str) -> list[float]:
    cleaned = list(values)
    for index, value in enumerate(cleaned):
        if not math.isfinite(value):
            logger.warning("non-finite %s at index=%d task=%s", label, index, task_id)
            cleaned[index] = 0.0
    return cleaned


def _effective_mask(sample: TrainingSample) -> list[int]:
    """Return per-response-token loss mask, defaulting to all-ones."""
    if sample.loss_mask is not None:
        return list(sample.loss_mask[: len(sample.response_tokens)])
    return [1] * len(sample.response_tokens)


def compute_grpo_advantages(samples: list[TrainingSample]) -> list[float]:
    return [sample.reward for sample in samples]


def truncate_sample(
    sample: TrainingSample,
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> TrainingSample:
    prompt_tokens = list(sample.prompt_tokens[-max_prompt_tokens:]) if max_prompt_tokens > 0 else []
    response_tokens = (
        list(sample.response_tokens[:max_response_tokens]) if max_response_tokens > 0 else []
    )
    response_logprobs = (
        list(sample.response_logprobs[: len(response_tokens)]) if response_tokens else []
    )
    teacher_logprobs = None
    if sample.teacher_logprobs is not None:
        teacher_logprobs = list(sample.teacher_logprobs[: len(response_tokens)])
    loss_mask = None
    if sample.loss_mask is not None:
        loss_mask = list(sample.loss_mask[: len(response_tokens)])
    return TrainingSample(
        task_id=sample.task_id,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        response_logprobs=response_logprobs,
        reward=sample.reward,
        prompt_text=sample.prompt_text,
        response_text=sample.response_text,
        teacher_logprobs=teacher_logprobs,
        loss_mask=loss_mask,
        sample_type=sample.sample_type,
    )


# ---------------------------------------------------------------------------
# RL / OPD datum conversion
# ---------------------------------------------------------------------------


def sample_to_datum(sample: TrainingSample, advantage: float):
    """Convert one sample + scalar advantage into a Tinker Datum (RL / OPD).

    For OPD samples with teacher_logprobs, the advantage is augmented with
    per-token distillation signal: (teacher_lp - student_lp).
    """
    all_tokens = list(sample.prompt_tokens) + list(sample.response_tokens)
    if len(all_tokens) < 2:
        raise ValueError(f"Cannot build datum from empty sequence for task={sample.task_id}")

    prompt_len = len(sample.prompt_tokens)
    target_tokens = all_tokens[1:]
    sequence_len = len(target_tokens)
    response_len = max(sequence_len - max(prompt_len - 1, 0), 0)

    mask = _effective_mask(sample)
    resp_advantages = [advantage * float(mask[i]) for i in range(response_len)]

    # OPD: add per-token distillation advantage (teacher_lp - student_lp)
    if sample.teacher_logprobs is not None:
        for i in range(min(len(resp_advantages), len(sample.teacher_logprobs))):
            student_lp = sample.response_logprobs[i] if i < len(sample.response_logprobs) else 0.0
            teacher_lp = sample.teacher_logprobs[i]
            resp_advantages[i] += (teacher_lp - student_lp) * float(mask[i])

    logprobs = [0.0] * max(prompt_len - 1, 0) + list(sample.response_logprobs[:response_len])
    advantages = [0.0] * max(prompt_len - 1, 0) + resp_advantages
    logprobs = _sanitize(_fit(logprobs, sequence_len), label="logprobs", task_id=sample.task_id)
    advantages = _sanitize(
        _fit(advantages, sequence_len),
        label="advantages",
        task_id=sample.task_id,
    )

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": logprobs,
            "advantages": advantages,
        },
    )


def batch_to_datums(
    samples: list[TrainingSample],
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> list:
    advantages = compute_grpo_advantages(samples)
    datums = []
    for sample, advantage in zip(samples, advantages, strict=True):
        truncated = truncate_sample(
            sample,
            max_prompt_tokens=max_prompt_tokens,
            max_response_tokens=max_response_tokens,
        )
        if not truncated.response_tokens:
            logger.warning("skipping empty truncated response for task=%s", sample.task_id)
            continue
        try:
            datums.append(sample_to_datum(truncated, advantage))
        except Exception as e:
            logger.error("failed to convert sample task=%s: %s", sample.task_id, e, exc_info=True)
    return datums


# ---------------------------------------------------------------------------
# Combined datum conversion
# ---------------------------------------------------------------------------


def sample_to_datum_combined(
    sample: TrainingSample,
    w_opd: float = 1.0,
    w_rl: float = 1.0,
):
    """Convert one sample into a Tinker Datum with combined OPD+RL advantages.

    combined_adv_i = w_opd * (teacher_lp_i - student_lp_i) + w_rl * reward

    Matches upstream combine_loss convention:
        combined_advantages = w_opd * teacher_advantages + w_rl * grpo_advantages
    where teacher_advantages = teacher_logp - old_logp (per-token, raw)
    and   grpo_advantages   = reward broadcast (scalar).
    """
    all_tokens = list(sample.prompt_tokens) + list(sample.response_tokens)
    if len(all_tokens) < 2:
        raise ValueError(f"Cannot build datum from empty sequence for task={sample.task_id}")

    prompt_len = len(sample.prompt_tokens)
    target_tokens = all_tokens[1:]
    sequence_len = len(target_tokens)
    response_len = max(sequence_len - max(prompt_len - 1, 0), 0)

    mask = _effective_mask(sample)

    resp_advantages = []
    for i in range(response_len):
        m = float(mask[i]) if i < len(mask) else 0.0

        # RL component: broadcast scalar reward
        rl_adv = w_rl * sample.reward * m

        # OPD component: per-token (teacher_lp - student_lp)
        opd_adv = 0.0
        if sample.teacher_logprobs is not None and i < len(sample.teacher_logprobs):
            student_lp = sample.response_logprobs[i] if i < len(sample.response_logprobs) else 0.0
            teacher_lp = sample.teacher_logprobs[i]
            opd_adv = w_opd * (teacher_lp - student_lp) * m

        resp_advantages.append(rl_adv + opd_adv)

    logprobs = [0.0] * max(prompt_len - 1, 0) + list(sample.response_logprobs[:response_len])
    advantages = [0.0] * max(prompt_len - 1, 0) + resp_advantages
    logprobs = _sanitize(_fit(logprobs, sequence_len), label="logprobs", task_id=sample.task_id)
    advantages = _sanitize(
        _fit(advantages, sequence_len),
        label="advantages",
        task_id=sample.task_id,
    )

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": logprobs,
            "advantages": advantages,
        },
    )


def batch_to_datums_combined(
    samples: list[TrainingSample],
    *,
    w_opd: float = 1.0,
    w_rl: float = 1.0,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> list:
    """Convert a batch of samples to Tinker Datums with combined OPD+RL advantages."""
    datums = []
    for sample in samples:
        truncated = truncate_sample(
            sample,
            max_prompt_tokens=max_prompt_tokens,
            max_response_tokens=max_response_tokens,
        )
        if not truncated.response_tokens:
            logger.warning("skipping empty truncated response for task=%s", sample.task_id)
            continue
        try:
            datums.append(sample_to_datum_combined(truncated, w_opd=w_opd, w_rl=w_rl))
        except Exception as e:
            logger.error(
                "failed to convert combined sample task=%s: %s",
                sample.task_id,
                e,
                exc_info=True,
            )
    return datums
