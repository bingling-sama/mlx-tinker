"""Local scorer modules for teacher logprob extraction and hint generation.

.. deprecated::
    This module uses a simplified hint source (task success_checks) instead
    of the LLM hint-judge flow used by upstream OpenClaw-RL
    (openclaw-tinker/scorers.py). The canonical scoring path is upstream.
    See ``scripts/run_openclaw_rl.sh`` for the recommended training path.

For the local mlx-tinker integration, both teacher and student share the same
server. The teacher is the base model (no LoRA); the student is the LoRA-adapted
model. Teacher logprobs are extracted via the Tinker sampling API with
include_prompt_logprobs=True.

Modules:
  - LocalTeacherLogprobExtractor: extract per-token teacher logprobs by scoring
    the student's response tokens through the base model with an enhanced prompt
    (hint appended from task success criteria).
  - build_hint_from_task: construct a textual hint from task success checks.
"""

from __future__ import annotations

import logging
from typing import Any

import tinker

from .curriculum import TaskSpec

logger = logging.getLogger(__name__)


def build_hint_from_task(task: TaskSpec) -> str:
    """Build a textual hint from the task's success checks.

    The hint is appended to the teacher's prompt so it can generate logprobs
    that reflect the desired outcome -- the teacher "knows" what the correct
    response should contain.
    """
    hints: list[str] = []
    for check in task.success_checks:
        if check.kind == "contains_all":
            hints.append(f"Response must contain: {', '.join(check.values)}")
        elif check.kind == "contains_any":
            hints.append(f"Response should mention: {', '.join(check.values)}")
        elif check.kind == "regex":
            hints.append(f"Response should match patterns: {', '.join(check.values)}")
    return "; ".join(hints) if hints else ""


class LocalTeacherLogprobExtractor:
    """Extract per-token teacher logprobs using the base model as teacher.

    For local training the "hint" is derived from the task's success criteria,
    appended to the prompt so the teacher model has privileged information about
    the desired output.

    The teacher scores the student's response tokens by running them through
    the base model (no LoRA) and collecting prompt logprobs for the response
    portion of the concatenated [enhanced_prompt + response] sequence.
    """

    def __init__(
        self,
        sampling_client: Any,
        tokenizer: Any,
    ) -> None:
        self.client = sampling_client
        self.tokenizer = tokenizer

    async def extract_teacher_logprobs(
        self,
        task: TaskSpec,
        prompt_text: str,
        response_tokens: list[int],
        response_text: str,
    ) -> list[float]:
        """Extract per-token teacher logprobs for the student's response.

        Args:
            task: The task spec (used to build the hint).
            prompt_text: The original prompt text seen by the student.
            response_tokens: The student's response token IDs.
            response_text: The student's decoded response text.

        Returns:
            List of teacher log-probabilities, one per response token.
            On failure, returns all-zeros matching response_tokens length.
        """
        response_len = len(response_tokens)
        if response_len == 0:
            return []

        hint = build_hint_from_task(task)
        if hint:
            enhanced_prompt = f"{prompt_text}\n\nHint: {hint}"
        else:
            enhanced_prompt = prompt_text

        try:
            # Tokenize the enhanced prompt and the full sequence
            enhanced_tokens = self._encode(enhanced_prompt)
            full_text = enhanced_prompt + response_text
            full_tokens = self._encode(full_text)

            prompt_token_count = len(enhanced_tokens)

            # Query teacher model: feed full sequence, request prompt logprobs
            model_input = tinker.ModelInput.from_ints(full_tokens)
            sampling_params = tinker.SamplingParams(temperature=0.0, max_tokens=1)

            response = await self.client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
                include_prompt_logprobs=True,
                topk_prompt_logprobs=0,
            )

            prompt_logprobs = response.prompt_logprobs or []

            # Extract logprobs for the response portion only
            teacher_lps = [
                float(lp) if lp is not None else 0.0
                for lp in prompt_logprobs[prompt_token_count:]
            ]

            # Align to response_tokens length
            if len(teacher_lps) > response_len:
                teacher_lps = teacher_lps[:response_len]
            elif len(teacher_lps) < response_len:
                logger.warning(
                    "teacher logprobs length mismatch: got %d, expected %d for task=%s",
                    len(teacher_lps),
                    response_len,
                    task.task_id,
                )
                teacher_lps += [0.0] * (response_len - len(teacher_lps))

            return teacher_lps

        except Exception as e:
            logger.error(
                "teacher logprob extraction failed for task=%s: %s",
                task.task_id,
                e,
                exc_info=True,
            )
            return [0.0] * response_len

    def _encode(self, text: str) -> list[int]:
        """Tokenize text, handling different tokenizer APIs."""
        try:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self.tokenizer.encode(text))
