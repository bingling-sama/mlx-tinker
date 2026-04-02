"""Convert Hermes live-RL samples into Tinker training datums."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """One trainable Hermes sample reconstructed by the bridge."""

    session_id: str
    turn_num: int
    prompt_tokens: list[int]
    response_tokens: list[int]
    response_logprobs: list[float]
    loss_mask: list[int]
    reward: float
    prompt_text: str = ""
    response_text: str = ""
    teacher_logprobs: Optional[list[float]] = None
    sample_type: str = ""


def _fit(values: list[float], length: int) -> list[float]:
    if len(values) >= length:
        return list(values[:length])
    return list(values) + [0.0] * (length - len(values))


def _sanitize(values: list[float], *, label: str, session_id: str, turn_num: int) -> list[float]:
    cleaned = list(values)
    for index, value in enumerate(cleaned):
        if not math.isfinite(value):
            logger.warning(
                "[DataFormatter] non-finite %s at idx=%d session=%s turn=%d",
                label,
                index,
                session_id,
                turn_num,
            )
            cleaned[index] = 0.0
    return cleaned


def _build_datum(
    all_tokens: list[int],
    logprobs: list[float],
    advantages: list[float],
    *,
    session_id: str,
    turn_num: int,
):
    import tinker

    if len(all_tokens) < 2:
        raise ValueError(
            f"Cannot build datum from empty sequence: session={session_id} turn={turn_num}"
        )

    target_tokens = all_tokens[1:]
    sequence_len = len(target_tokens)
    logprobs = _sanitize(
        _fit(logprobs, sequence_len),
        label="logprobs",
        session_id=session_id,
        turn_num=turn_num,
    )
    advantages = _sanitize(
        _fit(advantages, sequence_len),
        label="advantages",
        session_id=session_id,
        turn_num=turn_num,
    )

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": logprobs,
            "advantages": advantages,
        },
    )


def sample_to_datum(sample: TrainingSample, advantage: float, max_tokens: int = 0):
    prompt_tokens = list(sample.prompt_tokens)
    response_tokens = list(sample.response_tokens)

    if max_tokens > 0:
        total = len(prompt_tokens) + len(response_tokens)
        if total > max_tokens:
            keep = max(1, max_tokens - len(response_tokens))
            prompt_tokens = prompt_tokens[-keep:]
            logger.info(
                "[DataFormatter] truncated prompt to %d tokens (session=%s turn=%d)",
                len(prompt_tokens),
                sample.session_id,
                sample.turn_num,
            )

    all_tokens = prompt_tokens + response_tokens
    prompt_len = len(prompt_tokens)
    response_len = len(response_tokens)
    mask = list(sample.loss_mask[:response_len]) if sample.loss_mask else [1] * response_len

    resp_advantages = [advantage * float(mask[i]) for i in range(response_len)]
    if sample.teacher_logprobs is not None:
        for i in range(min(len(resp_advantages), len(sample.teacher_logprobs))):
            student_lp = sample.response_logprobs[i] if i < len(sample.response_logprobs) else 0.0
            teacher_lp = sample.teacher_logprobs[i]
            resp_advantages[i] += (teacher_lp - student_lp) * float(mask[i])

    logprobs = [0.0] * max(prompt_len - 1, 0) + list(sample.response_logprobs[:response_len])
    advantages = [0.0] * max(prompt_len - 1, 0) + resp_advantages
    return _build_datum(
        all_tokens,
        logprobs,
        advantages,
        session_id=sample.session_id,
        turn_num=sample.turn_num,
    )


def batch_to_datums(
    batch: list[TrainingSample],
    advantages: list[float],
    max_tokens: int = 0,
) -> list:
    datums = []
    for sample, advantage in zip(batch, advantages, strict=True):
        if not sample.response_tokens:
            logger.warning(
                "[DataFormatter] skipping empty response session=%s turn=%d",
                sample.session_id,
                sample.turn_num,
            )
            continue
        try:
            datums.append(sample_to_datum(sample, advantage, max_tokens=max_tokens))
        except Exception as exc:
            logger.error(
                "[DataFormatter] failed to convert session=%s turn=%d: %s",
                sample.session_id,
                sample.turn_num,
                exc,
                exc_info=True,
            )
    return datums


def sample_to_datum_combined(
    sample: TrainingSample,
    w_opd: float = 1.0,
    w_rl: float = 1.0,
    max_tokens: int = 0,
):
    prompt_tokens = list(sample.prompt_tokens)
    response_tokens = list(sample.response_tokens)
    if max_tokens > 0:
        total = len(prompt_tokens) + len(response_tokens)
        if total > max_tokens:
            keep = max(1, max_tokens - len(response_tokens))
            prompt_tokens = prompt_tokens[-keep:]

    all_tokens = prompt_tokens + response_tokens
    prompt_len = len(prompt_tokens)
    response_len = len(response_tokens)
    mask = list(sample.loss_mask[:response_len]) if sample.loss_mask else [1] * response_len

    resp_advantages: list[float] = []
    for i in range(response_len):
        weight = float(mask[i]) if i < len(mask) else 0.0
        rl_adv = w_rl * sample.reward * weight
        opd_adv = 0.0
        if sample.teacher_logprobs is not None and i < len(sample.teacher_logprobs):
            student_lp = sample.response_logprobs[i] if i < len(sample.response_logprobs) else 0.0
            teacher_lp = sample.teacher_logprobs[i]
            opd_adv = w_opd * (teacher_lp - student_lp) * weight
        resp_advantages.append(rl_adv + opd_adv)

    logprobs = [0.0] * max(prompt_len - 1, 0) + list(sample.response_logprobs[:response_len])
    advantages = [0.0] * max(prompt_len - 1, 0) + resp_advantages
    return _build_datum(
        all_tokens,
        logprobs,
        advantages,
        session_id=sample.session_id,
        turn_num=sample.turn_num,
    )


def batch_to_datums_combined(
    batch: list[TrainingSample],
    w_opd: float = 1.0,
    w_rl: float = 1.0,
    max_tokens: int = 0,
) -> list:
    datums = []
    for sample in batch:
        if not sample.response_tokens:
            logger.warning(
                "[DataFormatter] skipping empty combined response session=%s turn=%d",
                sample.session_id,
                sample.turn_num,
            )
            continue
        try:
            datums.append(
                sample_to_datum_combined(
                    sample,
                    w_opd=w_opd,
                    w_rl=w_rl,
                    max_tokens=max_tokens,
                )
            )
        except Exception as exc:
            logger.error(
                "[DataFormatter] failed to convert combined session=%s turn=%d: %s",
                sample.session_id,
                sample.turn_num,
                exc,
                exc_info=True,
            )
    return datums


def compute_grpo_advantages(batch: list[TrainingSample]) -> list[float]:
    return [sample.reward for sample in batch]
