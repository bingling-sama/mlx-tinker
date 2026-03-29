"""Inference / sampling backend using mlx-lm generation."""

from __future__ import annotations

from collections.abc import Mapping
import logging

import mlx.core as mx
import mlx.nn as nn

from mlx_tinker.types import GeneratedSequence, SampleInput, SampleOutput, SamplingParams

logger = logging.getLogger(__name__)


def _has_kv_cache_support(model: nn.Module) -> bool:
    """Check if a model supports KV cache (i.e., is a real mlx-lm model)."""
    if not hasattr(model, "layers") or len(model.layers) == 0:
        return False
    return hasattr(model, "make_cache") or model.__class__.__module__.startswith("mlx_lm.")


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


def _make_mlx_sampler(sp: SamplingParams):
    """Build an mlx-lm sampler matching the request's sampling params."""
    from mlx_lm.sample_utils import make_sampler

    top_k = sp.top_k if sp.top_k > 0 else 0
    return make_sampler(temp=sp.temperature, top_p=sp.top_p, top_k=top_k)


def _sampling_params_key(sp: SamplingParams) -> tuple:
    """Hashable compatibility key for grouping sample requests."""
    return (
        sp.temperature,
        sp.max_tokens,
        sp.seed,
        tuple(sp.stop_tokens or []),
        tuple(sp.stop_strings or []),
        sp.top_k,
        sp.top_p,
    )


def _sampling_params_key_from_mapping(data: Mapping[str, object] | None) -> tuple | None:
    """Build the same compatibility key from raw request data without validation."""
    if not isinstance(data, Mapping):
        return None

    stop_tokens = data.get("stop_tokens")
    stop_strings = data.get("stop_strings")
    if stop_tokens is not None and not isinstance(stop_tokens, (list, tuple)):
        return None
    if stop_strings is not None and not isinstance(stop_strings, (list, tuple)):
        return None

    return (
        data.get("temperature", 1.0),
        data.get("max_tokens", 256),
        data.get("seed", 0),
        tuple(stop_tokens or []),
        tuple(stop_strings or []),
        data.get("top_k", -1),
        data.get("top_p", 1.0),
    )


class InferenceBackend:
    """Token generation using mlx-lm, with per-request sampling parameters."""

    def __init__(self, max_kv_cache_size: int | None = None) -> None:
        self.max_kv_cache_size = max_kv_cache_size

    @staticmethod
    def _seed_rng(sp: SamplingParams) -> None:
        """Honor per-request seed without changing the public API."""
        if sp.seed is not None:
            mx.random.seed(sp.seed)

    def sample_batch(
        self,
        model: nn.Module,
        tokenizer: object,
        requests: list[SampleInput],
    ) -> list[SampleOutput]:
        """Generate samples for multiple compatible requests in one backend call."""
        if not requests:
            return []

        if len(requests) == 1:
            return [self.sample(model, tokenizer, requests[0])]

        if any(request.prompt_logprobs for request in requests):
            return [self.sample(model, tokenizer, request) for request in requests]

        if not _has_kv_cache_support(model):
            return [self.sample(model, tokenizer, request) for request in requests]

        first_key = _sampling_params_key(requests[0].sampling_params)
        if any(_sampling_params_key(request.sampling_params) != first_key for request in requests[1:]):
            return [self.sample(model, tokenizer, request) for request in requests]

        sp = requests[0].sampling_params
        self._seed_rng(sp)
        flat_prompts: list[list[int]] = []
        request_slices: list[tuple[int, int]] = []
        for request in requests:
            start = len(flat_prompts)
            prompt_tokens = request.prompt.get_tokens()
            for _ in range(request.num_samples):
                flat_prompts.append(list(prompt_tokens))
            request_slices.append((start, len(flat_prompts)))

        sequences = self._sample_prompt_batch(model, flat_prompts, sp)
        outputs: list[SampleOutput] = []
        for start, end in request_slices:
            outputs.append(SampleOutput(sequences=sequences[start:end], prompt_logprobs=None))
        return outputs

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
        self._seed_rng(sp)
        # model_path is routing-only in MLXBackend; batching depends on the resolved model call.
        use_batched_rollouts = request.num_samples > 1

        if _has_kv_cache_support(model):
            if use_batched_rollouts:
                try:
                    sequences = self._sample_with_batch_generator(
                        model,
                        prompt_tokens,
                        sp,
                        request.num_samples,
                    )
                except Exception:
                    logger.warning(
                        "batched sampling failed; falling back to sequential generation",
                        exc_info=True,
                    )
                    sequences = self._sample_with_generate_step(
                        model,
                        prompt_tokens,
                        sp,
                        request.num_samples,
                    )
            else:
                sequences = self._sample_with_generate_step(
                    model,
                    prompt_tokens,
                    sp,
                    request.num_samples,
                )
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

    def _sample_prompt_batch(
        self,
        model: nn.Module,
        prompts: list[list[int]],
        sp: SamplingParams,
    ) -> list[GeneratedSequence]:
        """Generate one output sequence for each prompt in a prompt batch."""
        from mlx_lm.generate import BatchGenerator

        sampler = _make_mlx_sampler(sp)
        max_tokens = [sp.max_tokens] * len(prompts)
        samplers = [sampler] * len(prompts)

        generator = BatchGenerator(
            model,
            stop_tokens=set(sp.stop_tokens or []),
            max_kv_size=self.max_kv_cache_size,
        )
        uids = generator.insert(prompts, max_tokens=max_tokens, samplers=samplers)
        by_uid = {
            uid: GeneratedSequence(stop_reason="length", tokens=[], logprobs=[]) for uid in uids
        }

        try:
            while responses := generator.next():
                for response in responses:
                    token_id = int(response.token)
                    sequence = by_uid[response.uid]
                    sequence.tokens.append(token_id)
                    if response.logprobs is not None:
                        sequence.logprobs.append(response.logprobs.reshape(-1)[token_id].item())
                    else:
                        sequence.logprobs.append(0.0)
                    if response.finish_reason is not None:
                        sequence.stop_reason = response.finish_reason
        finally:
            generator.close()

        return [by_uid[uid] for uid in uids]

    def _sample_with_batch_generator(
        self,
        model: nn.Module,
        prompt_tokens: list[int],
        sp: SamplingParams,
        num_samples: int,
    ) -> list[GeneratedSequence]:
        """Batch identical prompts to avoid repeated prefill per rollout."""
        prompts = [list(prompt_tokens) for _ in range(num_samples)]
        return self._sample_prompt_batch(model, prompts, sp)

    def _sample_with_generate_step(
        self,
        model: nn.Module,
        prompt_tokens: list[int],
        sp: SamplingParams,
        num_samples: int,
    ) -> list[GeneratedSequence]:
        """Use mlx-lm's generate_step for models with KV cache."""
        from mlx_lm.generate import generate_step

        sampler = _make_mlx_sampler(sp)
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
                max_kv_size=self.max_kv_cache_size,
            ):
                token_id = token.item() if hasattr(token, "item") else int(token)
                log_probs_all = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
                token_logprob = log_probs_all.reshape(-1)[token_id].item()

                generated_tokens.append(token_id)
                generated_logprobs.append(token_logprob)

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
                    sequences.append(
                        GeneratedSequence(
                            stop_reason="stop",
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
