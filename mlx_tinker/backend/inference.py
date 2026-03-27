"""Inference / sampling backend using mlx-lm generation."""

from __future__ import annotations

import logging
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from mlx_tinker.types import GeneratedSequence, SampleInput, SampleOutput, SamplingParams

logger = logging.getLogger(__name__)


def _build_sampler(sp: SamplingParams) -> Callable[[mx.array], mx.array]:
    """Build an mlx-lm sampler from Tinker SamplingParams."""
    return make_sampler(temp=sp.temperature, top_p=sp.top_p)


class InferenceBackend:
    """Token generation using mlx-lm, with per-request sampling parameters."""

    def sample(
        self,
        model: nn.Module,
        tokenizer: object,
        request: SampleInput,
    ) -> SampleOutput:
        """Generate token sequences from the model.

        For each of `num_samples`, generate a completion given the prompt.
        Respects per-request temperature, top_k, top_p, max_tokens, and stop conditions.
        """
        prompt_tokens = request.prompt.get_tokens()
        sp = request.sampling_params
        sampler = _build_sampler(sp)
        sequences: list[GeneratedSequence] = []

        for _ in range(request.num_samples):
            generated_tokens: list[int] = []
            generated_logprobs: list[float] = []

            prompt_array = mx.array(prompt_tokens)

            for token, logprobs in generate_step(
                prompt=prompt_array,
                model=model,
                max_tokens=sp.max_tokens,
                sampler=sampler,
            ):
                token_id = token.item()
                # logprobs is the full logit distribution; extract log prob of chosen token
                log_probs_all = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
                token_logprob = log_probs_all.reshape(-1)[token_id].item()

                generated_tokens.append(token_id)
                generated_logprobs.append(token_logprob)

                # Check stop conditions
                if sp.stop_tokens and token_id in sp.stop_tokens:
                    sequences.append(
                        GeneratedSequence(
                            stop_reason="stop",
                            tokens=generated_tokens,
                            logprobs=generated_logprobs,
                        )
                    )
                    break

                if len(generated_tokens) >= sp.max_tokens:
                    sequences.append(
                        GeneratedSequence(
                            stop_reason="length",
                            tokens=generated_tokens,
                            logprobs=generated_logprobs,
                        )
                    )
                    break
            else:
                sequences.append(
                    GeneratedSequence(
                        stop_reason="length",
                        tokens=generated_tokens,
                        logprobs=generated_logprobs,
                    )
                )

        # Optionally compute prompt log probs
        prompt_lp = None
        if request.prompt_logprobs and len(prompt_tokens) > 1:
            prompt_lp = self._compute_prompt_logprobs(model, prompt_tokens)

        logger.info(
            "sample num_samples=%d max_tokens=%d generated=%s",
            request.num_samples,
            sp.max_tokens,
            [len(s.tokens) for s in sequences],
        )

        return SampleOutput(sequences=sequences, prompt_logprobs=prompt_lp)

    def _compute_prompt_logprobs(
        self,
        model: nn.Module,
        prompt_tokens: list[int],
    ) -> list[float]:
        """Compute per-token log probabilities for the prompt."""
        input_ids = mx.array(prompt_tokens)[None, :]
        logits = model(input_ids)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        targets = mx.array(prompt_tokens[1:])[None, :, None]
        target_lp = mx.take_along_axis(
            log_probs[:, :-1], targets.astype(mx.int32), axis=-1
        ).squeeze(-1)
        mx.eval(target_lp)

        return [0.0] + target_lp[0].tolist()
