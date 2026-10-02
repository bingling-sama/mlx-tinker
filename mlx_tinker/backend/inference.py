"""Inference / sampling backend using mlx-lm generation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
import inspect
import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from mlx_tinker.backend.transcript_cache import TranscriptPrefixCacheManager
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


def _encode_to_single_token(tokenizer: Any, s: str) -> int | None:
    if not hasattr(tokenizer, "encode") or not callable(tokenizer.encode):
        return None
    try:
        token_ids = tokenizer.encode(s, add_special_tokens=False)
    except TypeError:
        try:
            token_ids = tokenizer.encode(s)
        except Exception:
            return None
    except Exception:
        return None

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    elif not isinstance(token_ids, (list, tuple)):
        try:
            token_ids = list(token_ids)
        except Exception:
            return None

    if len(token_ids) == 1:
        try:
            return int(token_ids[0])
        except (ValueError, TypeError):
            return None
    return None


def _extract_eos_tokens(tokenizer: Any) -> list[int]:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return []
    if isinstance(eos_id, int):
        return [eos_id]
    if isinstance(eos_id, (list, tuple, set)):
        result = []
        for x in eos_id:
            try:
                result.append(int(x))
            except (ValueError, TypeError):
                pass
        return result
    return []


def resolve_stop_tokens(
    tokenizer: Any,
    stop_tokens: list[int] | tuple[int, ...] | set[int] | None = None,
    stop_strings: list[str] | tuple[str, ...] | None = None,
) -> list[int]:
    """Resolve token IDs for stopping, including tokenizer EOS, common end tokens, and stop_strings."""
    resolved: set[int] = set(stop_tokens or [])

    # 1. Include tokenizer.eos_token_id
    for tok_id in _extract_eos_tokens(tokenizer):
        resolved.add(tok_id)

    # 2. Include common end tokens (e.g. Qwen <|im_end|>, <|endoftext|>)
    for s in ("<|im_end|>", "<|endoftext|>"):
        tid = _encode_to_single_token(tokenizer, s)
        if tid is not None:
            resolved.add(tid)

    # 3. Include client-provided stop_strings
    if stop_strings:
        for s in stop_strings:
            if isinstance(s, str) and s:
                tid = _encode_to_single_token(tokenizer, s)
                if tid is not None:
                    resolved.add(tid)

    return sorted(resolved)


def prepare_sample_request(request: SampleInput, tokenizer: Any) -> SampleInput:
    """Ensure stop_tokens in request.sampling_params includes EOS tokens and encoded stop_strings."""
    sp = request.sampling_params
    resolved_stop_tokens = resolve_stop_tokens(
        tokenizer,
        stop_tokens=sp.stop_tokens,
        stop_strings=sp.stop_strings,
    )
    if resolved_stop_tokens == (sp.stop_tokens or []):
        return request
    new_sp = sp.model_copy(update={"stop_tokens": resolved_stop_tokens})
    return request.model_copy(update={"sampling_params": new_sp})


def _sampling_params_key_from_mapping(data: Mapping[str, object] | None) -> tuple | None:
    """Build the same compatibility key from raw request data without validation."""
    if not isinstance(data, Mapping):
        return None

    stop_tokens = (
        list(data.get("stop_tokens") or [])
        if isinstance(data.get("stop_tokens"), (list, tuple))
        else []
    )
    stop_strings = (
        list(data.get("stop_strings") or [])
        if isinstance(data.get("stop_strings"), (list, tuple))
        else []
    )

    raw_stop = data.get("stop")
    if raw_stop is not None:
        if isinstance(raw_stop, str):
            if raw_stop and raw_stop not in stop_strings:
                stop_strings.append(raw_stop)
        elif isinstance(raw_stop, (list, tuple)):
            for item in raw_stop:
                if isinstance(item, str):
                    if item and item not in stop_strings:
                        stop_strings.append(item)
                elif isinstance(item, int):
                    if item not in stop_tokens:
                        stop_tokens.append(item)
        elif isinstance(raw_stop, int):
            if raw_stop not in stop_tokens:
                stop_tokens.append(raw_stop)

    return (
        data.get("temperature", 1.0),
        data.get("max_tokens", 256),
        data.get("seed", 0),
        tuple(stop_tokens),
        tuple(stop_strings),
        data.get("top_k", -1),
        data.get("top_p", 1.0),
    )


class InferenceBackend:
    """Token generation using mlx-lm, with per-request sampling parameters."""

    def __init__(
        self,
        max_kv_cache_size: int | None = None,
        kv_cache_bits: int | None = None,
        kv_cache_group_size: int = 64,
        quantized_kv_start: int = 0,
        transcript_cache: TranscriptPrefixCacheManager | None = None,
    ) -> None:
        self.max_kv_cache_size = max_kv_cache_size
        self.kv_cache_bits = kv_cache_bits
        self.kv_cache_group_size = kv_cache_group_size
        self.quantized_kv_start = quantized_kv_start
        self.transcript_cache = transcript_cache

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
        namespace: str | None = None,
    ) -> list[SampleOutput]:
        """Generate samples for multiple compatible requests in one backend call."""
        if not requests:
            return []

        requests = [prepare_sample_request(r, tokenizer) for r in requests]

        if len(requests) == 1:
            return [self.sample(model, tokenizer, requests[0], namespace=namespace)]

        if any(request.prompt_logprobs for request in requests):
            return [self.sample(model, tokenizer, request, namespace=namespace) for request in requests]

        if self._should_use_transcript_cache(model, namespace):
            return [self.sample(model, tokenizer, request, namespace=namespace) for request in requests]

        if self.kv_cache_bits is not None:
            return [self.sample(model, tokenizer, request, namespace=namespace) for request in requests]

        if not _has_kv_cache_support(model):
            return [self.sample(model, tokenizer, request, namespace=namespace) for request in requests]

        first_key = _sampling_params_key(requests[0].sampling_params)
        if any(_sampling_params_key(request.sampling_params) != first_key for request in requests[1:]):
            return [self.sample(model, tokenizer, request, namespace=namespace) for request in requests]

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
        namespace: str | None = None,
    ) -> SampleOutput:
        """Generate token sequences from the model.

        Uses mlx-lm's generate_step for real models with KV cache support,
        falls back to a simple autoregressive loop for simpler models.
        """
        request = prepare_sample_request(request, tokenizer)
        prompt_tokens = request.prompt.get_tokens()
        sp = request.sampling_params
        self._seed_rng(sp)
        # model_path is routing-only in MLXBackend; batching depends on the resolved model call.
        use_batched_rollouts = (
            request.num_samples > 1
            and self.kv_cache_bits is None
            and not self._should_use_transcript_cache(model, namespace)
        )

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
                        namespace=namespace,
                    )
            else:
                sequences = self._sample_with_generate_step(
                    model,
                    prompt_tokens,
                    sp,
                    request.num_samples,
                    namespace=namespace,
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
            while True:
                if hasattr(generator, "next_generated"):
                    responses = generator.next_generated()
                else:
                    res = generator.next()
                    if isinstance(res, tuple):
                        responses = res[1]
                    else:
                        responses = res
                if not responses:
                    break
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
        namespace: str | None = None,
    ) -> list[GeneratedSequence]:
        """Use mlx-lm's generate_step for models with KV cache."""
        from mlx_lm.generate import generate_step

        sampler = _make_mlx_sampler(sp)
        sequences: list[GeneratedSequence] = []

        for _ in range(num_samples):
            generated_tokens: list[int] = []
            generated_logprobs: list[float] = []
            prompt_cache, prompt_tail = self._prepare_prompt_cache(model, prompt_tokens, namespace)
            prompt_array = mx.array(prompt_tail)
            saved_chunk_lengths: set[int] = set()
            stop_tokens_set = set(sp.stop_tokens or [])

            for token, logprobs in self._iter_generate_step(
                generate_step,
                prompt_array,
                model,
                sp,
                sampler,
                prompt_cache=prompt_cache,
            ):
                token_id = token.item() if hasattr(token, "item") else int(token)
                log_probs_all = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
                token_logprob = log_probs_all.reshape(-1)[token_id].item()

                generated_tokens.append(token_id)
                generated_logprobs.append(token_logprob)
                self._checkpoint_transcript_prefixes(
                    namespace,
                    prompt_tokens,
                    generated_tokens,
                    prompt_cache,
                    saved_chunk_lengths,
                )

                if stop_tokens_set and token_id in stop_tokens_set:
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
            self._persist_final_transcript(namespace, prompt_tokens, generated_tokens, prompt_cache)

        return sequences

    def _is_cache_trimmable(self, model: nn.Module) -> bool:
        cached = getattr(model, "_is_trimmable_cache", None)
        if cached is not None:
            return cached
        if not hasattr(model, "make_cache"):
            return False
        try:
            from mlx_lm.models.cache import can_trim_prompt_cache

            c = model.make_cache()
            result = bool(can_trim_prompt_cache(c))
        except Exception:
            result = False
        try:
            model._is_trimmable_cache = result
        except Exception:
            pass
        return result

    def _should_use_transcript_cache(self, model: nn.Module, namespace: str | None) -> bool:
        return bool(
            namespace
            and self.transcript_cache is not None
            and self.transcript_cache.enabled
            and self.max_kv_cache_size is None
            and _has_kv_cache_support(model)
            and self._is_cache_trimmable(model)
        )

    def _prepare_prompt_cache(
        self,
        model: nn.Module,
        prompt_tokens: list[int],
        namespace: str | None,
    ) -> tuple[list[Any] | None, list[int]]:
        if not self._should_use_transcript_cache(model, namespace):
            return None, list(prompt_tokens)

        lookup = self.transcript_cache.lookup(namespace, prompt_tokens)
        prompt_cache = lookup.prompt_cache
        if prompt_cache is None:
            prompt_cache = make_prompt_cache(model, max_kv_size=self.max_kv_cache_size)
        tail = lookup.uncached_tail
        if not tail:
            tail = list(prompt_tokens)
            prompt_cache = make_prompt_cache(model, max_kv_size=self.max_kv_cache_size)
        return prompt_cache, tail

    def _iter_generate_step(
        self,
        generate_step,
        prompt_array: mx.array,
        model: nn.Module,
        sp: SamplingParams,
        sampler,
        *,
        prompt_cache: list[Any] | None,
    ):
        kwargs: dict[str, Any] = {
            "prompt": prompt_array,
            "model": model,
            "max_tokens": sp.max_tokens,
            "sampler": sampler,
            "max_kv_size": self.max_kv_cache_size,
            "kv_bits": self.kv_cache_bits,
            "kv_group_size": self.kv_cache_group_size,
            "quantized_kv_start": self.quantized_kv_start,
        }
        sig = inspect.signature(generate_step)
        if prompt_cache is not None and "prompt_cache" in sig.parameters:
            kwargs["prompt_cache"] = prompt_cache
        return generate_step(**kwargs)

    def _checkpoint_transcript_prefixes(
        self,
        namespace: str | None,
        prompt_tokens: list[int],
        generated_tokens: list[int],
        prompt_cache: list[Any] | None,
        saved_chunk_lengths: set[int],
    ) -> None:
        if not namespace or prompt_cache is None or self.transcript_cache is None:
            return

        transcript_tokens = prompt_tokens + generated_tokens
        current_length = len(transcript_tokens)
        checkpoint_length = (current_length // self.transcript_cache.chunk_size) * self.transcript_cache.chunk_size
        while checkpoint_length > 0 and checkpoint_length not in saved_chunk_lengths:
            checkpoint_tokens = transcript_tokens[:checkpoint_length]
            checkpoint_cache = copy.deepcopy(prompt_cache)
            if checkpoint_length < current_length:
                to_trim = current_length - checkpoint_length
                trimmed = trim_prompt_cache(checkpoint_cache, to_trim)
                if trimmed != to_trim:
                    break
            parent_length = checkpoint_length - self.transcript_cache.chunk_size
            parent_tokens = checkpoint_tokens[:parent_length] if parent_length > 0 else None
            self.transcript_cache.enqueue_persist(
                namespace,
                checkpoint_tokens,
                checkpoint_cache,
                checkpoint_reason="chunk",
                parent_tokens=parent_tokens,
            )
            saved_chunk_lengths.add(checkpoint_length)
            checkpoint_length -= self.transcript_cache.chunk_size

    def _persist_final_transcript(
        self,
        namespace: str | None,
        prompt_tokens: list[int],
        generated_tokens: list[int],
        prompt_cache: list[Any] | None,
    ) -> None:
        if not namespace or prompt_cache is None or self.transcript_cache is None:
            return
        transcript_tokens = prompt_tokens + generated_tokens
        if not transcript_tokens:
            return
        parent_length = (
            (len(transcript_tokens) // self.transcript_cache.chunk_size)
            * self.transcript_cache.chunk_size
        )
        if parent_length == len(transcript_tokens):
            parent_length -= self.transcript_cache.chunk_size
        parent_tokens = transcript_tokens[:parent_length] if parent_length > 0 else None
        self.transcript_cache.enqueue_persist(
            namespace,
            transcript_tokens,
            prompt_cache,
            checkpoint_reason="final",
            parent_tokens=parent_tokens,
        )

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
            stop_tokens_set = set(sp.stop_tokens or [])

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

                if stop_tokens_set and token_id in stop_tokens_set:
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
