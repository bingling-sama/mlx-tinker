"""Inference / sampling backend using mlx-lm generation."""

from __future__ import annotations

import logging
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_tinker.types import GeneratedSequence, SampleInput, SampleOutput, SamplingParams

logger = logging.getLogger(__name__)


def _has_kv_cache_support(model: nn.Module) -> bool:
    """Check if a model supports KV cache (i.e., is a real mlx-lm model)."""
    if not hasattr(model, "layers"):
        return False
    if len(model.layers) == 0:
        return False
    # Check if layers have attention with KV cache support
    layer = model.layers[0]
    return hasattr(layer, "self_attn") or hasattr(layer, "attention")


def _sample_token(logits: mx.array, temperature: float, top_p: float) -> int:
    """Sample a token from logits with temperature and top-p."""
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1).item()

    logits = logits / temperature

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_indices = mx.argsort(-logits)
        sorted_logits = logits[sorted_indices]
        probs = mx.softmax(sorted_logits, axis=-1)
        cumsum = mx.cumsum(probs, axis=-1)
        # Mask tokens beyond top-p
        mask = cumsum - probs <= top_p
        sorted_logits = mx.where(mask, sorted_logits, mx.array(float("-inf")))
        probs = mx.softmax(sorted_logits, axis=-1)
        idx = mx.random.categorical(mx.log(probs + 1e-10))
        mx.eval(idx)
        return sorted_indices[idx.item()].item()

    probs = mx.softmax(logits, axis=-1)
    token = mx.random.categorical(mx.log(probs + 1e-10))
    mx.eval(token)
    return token.item()


class InferenceBackend:
    """Token generation using mlx-lm, with per-request sampling parameters."""

    def sample(
        self,
        model: nn.Module,
        tokenizer: object,
        request: SampleInput,
    ) -> SampleOutput:
        """Generate token sequences from the model.

        Uses mlx-lm's generate_step for real models with KV cache support,
        falls back to a simple autoregressive loop for simpler models.
        """
        prompt_tokens = request.prompt.get_tokens()
        sp = request.sampling_params

        if _has_kv_cache_support(model):
            sequences = self._sample_with_generate_step(model, prompt_tokens, sp, request.num_samples)
        else:
            sequences = self._sample_simple(model, prompt_tokens, sp, request.num_samples)

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

    def _sample_with_generate_step(
        self,
        model: nn.Module,
        prompt_tokens: list[int],
        sp: SamplingParams,
        num_samples: int,
    ) -> list[GeneratedSequence]:
        """Use mlx-lm's generate_step for models with KV cache."""
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=sp.temperature, top_p=sp.top_p)
        sequences: list[GeneratedSequence] = []

        for _ in range(num_samples):
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
                log_probs_all = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
                token_logprob = log_probs_all.reshape(-1)[token_id].item()

                generated_tokens.append(token_id)
                generated_logprobs.append(token_logprob)

                if sp.stop_tokens and token_id in sp.stop_tokens:
                    sequences.append(GeneratedSequence(
                        stop_reason="stop", tokens=generated_tokens, logprobs=generated_logprobs,
                    ))
                    break

                if len(generated_tokens) >= sp.max_tokens:
                    sequences.append(GeneratedSequence(
                        stop_reason="length", tokens=generated_tokens, logprobs=generated_logprobs,
                    ))
                    break
            else:
                sequences.append(GeneratedSequence(
                    stop_reason="length", tokens=generated_tokens, logprobs=generated_logprobs,
                ))

        return sequences

    def _sample_simple(
        self,
        model: nn.Module,
        prompt_tokens: list[int],
        sp: SamplingParams,
        num_samples: int,
    ) -> list[GeneratedSequence]:
        """Simple autoregressive generation without KV cache."""
        sequences: list[GeneratedSequence] = []

        for _ in range(num_samples):
            generated_tokens: list[int] = []
            generated_logprobs: list[float] = []
            current_tokens = list(prompt_tokens)

            for _ in range(sp.max_tokens):
                input_ids = mx.array(current_tokens)[None, :]
                logits = model(input_ids)
                mx.eval(logits)

                # Get logits for last position
                last_logits = logits[0, -1]
                log_probs = last_logits - mx.logsumexp(last_logits, keepdims=True)

                # Sample
                token_id = _sample_token(last_logits, sp.temperature, sp.top_p)
                token_logprob = log_probs[token_id].item()

                generated_tokens.append(token_id)
                generated_logprobs.append(token_logprob)
                current_tokens.append(token_id)

                if sp.stop_tokens and token_id in sp.stop_tokens:
                    sequences.append(GeneratedSequence(
                        stop_reason="stop", tokens=generated_tokens, logprobs=generated_logprobs,
                    ))
                    break
            else:
                sequences.append(GeneratedSequence(
                    stop_reason="length", tokens=generated_tokens, logprobs=generated_logprobs,
                ))

        return sequences

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
