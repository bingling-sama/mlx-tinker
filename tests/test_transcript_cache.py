from __future__ import annotations

from pathlib import Path
import importlib
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, make_prompt_cache
import numpy as np
import pytest

from mlx_tinker.api.openai_compat import register_openai_routes
from mlx_tinker.backend.inference import InferenceBackend
from mlx_tinker.backend.transcript_cache import TranscriptPrefixCacheManager
from mlx_tinker.types import EncodedTextChunk, ModelInput, SampleInput, SamplingParams
from tests.helpers import FakeTokenizer, TinyModelWithCache


class CountingTinyModel(TinyModelWithCache):
    def __init__(self) -> None:
        super().__init__()
        self.processed_tokens = 0
        mx.eval(self.parameters())

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def reset_counts(self) -> None:
        self.processed_tokens = 0

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        self.processed_tokens += int(x.shape[1])
        h = self.embed(x)
        if cache is not None:
            for idx, layer in enumerate(self.layers):
                h, _ = layer(h, cache[idx])
                kv = h.reshape(h.shape[0], 1, h.shape[1], h.shape[2])
                cache[idx].update_and_fetch(kv, kv)
        else:
            for layer in self.layers:
                h, _ = layer(h)
        return self.head(h)


class PrefixConsistentTinyModel(nn.Module):
    """Toy LM whose cache-backed and cold-prefill logits should stay numerically aligned."""

    def __init__(self, vocab_size: int = 32, dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.layers = [object()]
        mx.eval(self.parameters())

    def make_cache(self):
        return [KVCache()]

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        h = self.embed(x)
        if cache is None:
            return self.head(mx.cumsum(h, axis=1))

        prefix_sum = mx.zeros((h.shape[0], 1, h.shape[2]), dtype=h.dtype)
        if cache[0].offset > 0:
            _, values, *_ = cache[0].state
            prefix_sum = mx.sum(values[:, :, : cache[0].offset, :], axis=2)

        running = mx.cumsum(h, axis=1) + prefix_sum
        kv = h.reshape(h.shape[0], 1, h.shape[1], h.shape[2])
        cache[0].update_and_fetch(kv, kv)
        return self.head(running)


class ChatTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_kwargs):
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    def decode(self, tokens, **_kwargs):
        return super().decode(tokens)


def _manager(tmp_path: Path, max_bytes: int = 1 << 20) -> TranscriptPrefixCacheManager:
    return TranscriptPrefixCacheManager(tmp_path / "prefix_cache", max_bytes=max_bytes)


def _make_cache(length: int) -> list[KVCache]:
    cache = [KVCache()]
    keys = mx.arange(length * 2, dtype=mx.float32).reshape(1, 1, length, 2)
    values = (mx.arange(length * 2, dtype=mx.float32) + 100).reshape(1, 1, length, 2)
    cache[0].update_and_fetch(keys, values)
    return cache


def _request(tokens: list[int], *, max_tokens: int = 2, temperature: float = 0.0, num_samples: int = 1):
    return SampleInput(
        prompt=ModelInput(chunks=[EncodedTextChunk(tokens=tokens)]),
        sampling_params=SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=7),
        num_samples=num_samples,
    )


def test_manager_matches_exact_shorter_and_longer_prefixes(tmp_path):
    manager = _manager(tmp_path)
    manager.enqueue_persist("ns", [1, 2, 3, 4], _make_cache(4), "final")
    manager.enqueue_persist("ns", [1, 2, 3, 4, 5, 6, 7, 8], _make_cache(8), "final")
    manager.wait_for_idle()

    exact = manager.lookup("ns", [1, 2, 3, 4])
    shorter = manager.lookup("ns", [1, 2, 3, 4, 9, 10])
    longer = manager.lookup("ns", [1, 2, 3, 4, 5, 6])

    assert exact.cached_tokens == 3
    assert exact.uncached_tail == [4]
    assert shorter.cached_tokens == 4
    assert shorter.uncached_tail == [9, 10]
    assert longer.cached_tokens == 5
    assert longer.uncached_tail == [6]

    manager.close()


def test_manager_enforces_leaf_lru_eviction_and_namespace_invalidation(tmp_path):
    manager = _manager(tmp_path, max_bytes=1 << 30)
    manager.enqueue_persist("ns", [1, 2, 3, 4], _make_cache(4), "final")
    manager.wait_for_idle()
    manager.max_bytes = manager.total_bytes + 1
    manager.enqueue_persist("ns", [9, 8, 7, 6], _make_cache(4), "final")
    manager.wait_for_idle()

    assert manager.lookup("ns", [1, 2, 3, 4]).cached_tokens == 0
    assert manager.lookup("ns", [9, 8, 7, 6]).cached_tokens == 3

    manager.enqueue_persist("student:model-1:v0", [4, 5, 6], _make_cache(3), "final")
    manager.wait_for_idle()
    manager.invalidate_namespace_prefix("student:")
    assert manager.lookup("student:model-1:v0", [4, 5, 6]).cached_tokens == 0

    manager.close()


def test_manager_evicts_entries_idle_for_over_one_hour(monkeypatch, tmp_path):
    now = {"value": 1_000_000.0}
    monkeypatch.setattr("mlx_tinker.backend.transcript_cache.time.time", lambda: now["value"])

    manager = _manager(tmp_path)
    manager.enqueue_persist("ns", [1, 2, 3, 4], _make_cache(4), "final")
    manager.wait_for_idle()

    key_hash = manager._hash_tokens("ns", [1, 2, 3, 4])
    cache_file = manager._entries[key_hash].file_path
    assert cache_file.exists()

    now["value"] += manager.stale_after_seconds - 10
    warm = manager.lookup("ns", [1, 2, 3, 4])
    assert warm.cached_tokens == 3
    assert cache_file.exists()

    now["value"] += 20
    still_warm = manager.lookup("ns", [1, 2, 3, 4])
    assert still_warm.cached_tokens == 3
    assert cache_file.exists()

    now["value"] += manager.stale_after_seconds + 1
    expired = manager.lookup("ns", [1, 2, 3, 4])
    assert expired.cached_tokens == 0
    assert not cache_file.exists()
    assert manager.total_bytes == 0
    assert manager.last_eviction is not None
    assert manager.last_eviction["eviction_reason"] == "ttl"

    manager.close()


def test_disk_prefix_cache_preserves_logits_with_tiny_error_without_quantization(tmp_path):
    partial_namespace = "base:test-model:partial:kv_bits=None"
    exact_namespace = "base:test-model:exact:kv_bits=None"
    prefix_tokens = [1, 2, 3, 4]
    tail_tokens = [5, 6, 7]
    full_prompt = prefix_tokens + tail_tokens

    mx.random.seed(0)
    model = PrefixConsistentTinyModel()
    warm_manager = _manager(tmp_path / "warm")
    warm_inference = InferenceBackend(transcript_cache=warm_manager, kv_cache_bits=None)

    prefix_cache = make_prompt_cache(model)
    model(mx.array(prefix_tokens)[None], cache=prefix_cache)
    mx.eval([cache.state for cache in prefix_cache])
    warm_inference._persist_final_transcript(partial_namespace, prefix_tokens, [], prefix_cache)

    full_cache = make_prompt_cache(model)
    cold_logits = model(mx.array(full_prompt)[None], cache=full_cache)
    mx.eval(cold_logits, [cache.state for cache in full_cache])
    warm_inference._persist_final_transcript(exact_namespace, full_prompt, [], full_cache)
    warm_manager.close()

    restarted_manager = _manager(tmp_path / "warm")
    restarted_inference = InferenceBackend(transcript_cache=restarted_manager, kv_cache_bits=None)

    partial_cache, partial_tail = restarted_inference._prepare_prompt_cache(
        model, full_prompt, partial_namespace
    )
    partial_logits = model(mx.array(partial_tail)[None], cache=partial_cache)
    mx.eval(partial_logits, [cache.state for cache in partial_cache])

    exact_cache, exact_tail = restarted_inference._prepare_prompt_cache(
        model, full_prompt, exact_namespace
    )
    exact_logits = model(mx.array(exact_tail)[None], cache=exact_cache)
    mx.eval(exact_logits, [cache.state for cache in exact_cache])

    cold_tail_logits = np.array(cold_logits)[:, len(prefix_tokens) :, :]
    partial_tail_logits = np.array(partial_logits)
    cold_exact_logits = np.array(cold_logits)[:, -1:, :]
    exact_hit_logits = np.array(exact_logits)
    partial_max_diff = float(np.max(np.abs(partial_tail_logits - cold_tail_logits)))
    exact_max_diff = float(np.max(np.abs(exact_hit_logits - cold_exact_logits)))

    assert partial_tail == tail_tokens
    assert exact_tail == [full_prompt[-1]]
    assert restarted_manager.lookup(partial_namespace, full_prompt).cached_tokens == len(prefix_tokens)
    assert restarted_manager.lookup(exact_namespace, full_prompt).cached_tokens == len(full_prompt) - 1
    np.testing.assert_allclose(partial_tail_logits, cold_tail_logits, rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(exact_hit_logits, cold_exact_logits, rtol=0.0, atol=2e-7)
    assert partial_max_diff < 2e-7
    assert exact_max_diff < 2e-7

    restarted_manager.close()


def test_disk_cache_proves_multi_turn_reuse_across_restart(tmp_path):
    namespace = "base:test-model"
    model = CountingTinyModel()
    tokenizer = FakeTokenizer()

    warm_manager = _manager(tmp_path / "warm")
    warm_inference = InferenceBackend(transcript_cache=warm_manager)
    turn_one = warm_inference.sample(model, tokenizer, _request([1, 2, 3, 4], max_tokens=2), namespace=namespace)
    assistant_tokens = turn_one.sequences[0].tokens
    warm_manager.wait_for_idle()
    warm_manager.close()

    turn_two_prompt = [1, 2, 3, 4] + assistant_tokens + [7, 8, 9]

    cold_manager = _manager(tmp_path / "cold")
    cold_inference = InferenceBackend(transcript_cache=cold_manager)
    model.reset_counts()
    cold = cold_inference.sample(model, tokenizer, _request(turn_two_prompt, max_tokens=2), namespace=namespace)
    cold_processed = model.processed_tokens
    cold_manager.close()

    restarted_manager = _manager(tmp_path / "warm")
    restarted_inference = InferenceBackend(transcript_cache=restarted_manager)
    model.reset_counts()
    warm = restarted_inference.sample(
        model,
        tokenizer,
        _request(turn_two_prompt, max_tokens=2),
        namespace=namespace,
    )
    restarted_manager.wait_for_idle()
    warm_processed = model.processed_tokens

    assert warm.sequences[0].tokens == cold.sequences[0].tokens
    assert restarted_manager.last_lookup is not None
    assert restarted_manager.last_lookup["hit_tokens"] > 0
    assert warm_processed < cold_processed

    restarted_manager.close()


def test_all_sampled_branches_are_persisted(monkeypatch, tmp_path):
    namespace = "base:test-model"
    tokenizer = FakeTokenizer()
    model = CountingTinyModel()
    manager = _manager(tmp_path)
    inference = InferenceBackend(transcript_cache=manager)

    call_index = {"value": 0}
    branch_tokens = {
        1: [11, 12],
        2: [21, 22],
        3: [31],
    }
    prompt_lengths: list[int] = []

    def fake_generate_step(
        *,
        prompt,
        model,
        max_tokens,
        sampler,
        max_kv_size,
        kv_bits,
        kv_group_size,
        quantized_kv_start,
        prompt_cache=None,
    ):
        del sampler, max_kv_size, kv_bits, kv_group_size, quantized_kv_start
        prompt_lengths.append(int(prompt.shape[0]))
        if prompt_cache is None:
            prompt_cache = make_prompt_cache(model)
        if int(prompt.shape[0]) > 0:
            model(prompt[None], cache=prompt_cache)
            mx.eval([cache.state for cache in prompt_cache])
        call_index["value"] += 1
        for token_id in branch_tokens[call_index["value"]][:max_tokens]:
            model(mx.array([token_id])[None], cache=prompt_cache)
            mx.eval([cache.state for cache in prompt_cache])
            yield mx.array(token_id), mx.zeros(32)

    generate_module = importlib.import_module("mlx_lm.generate")
    monkeypatch.setattr(generate_module, "generate_step", fake_generate_step)

    request = _request([1, 2, 3], max_tokens=2, temperature=1.0, num_samples=2)
    first = inference.sample(model, tokenizer, request, namespace=namespace)
    manager.wait_for_idle()

    branch_two = first.sequences[1].tokens
    followup_prompt = [1, 2, 3] + branch_two + [7]
    second = inference.sample(
        model,
        tokenizer,
        _request(followup_prompt, max_tokens=1, temperature=1.0),
        namespace=namespace,
    )
    manager.wait_for_idle()

    assert first.sequences[0].tokens == [11, 12]
    assert first.sequences[1].tokens == [21, 22]
    assert second.sequences[0].tokens == [31]
    assert manager.last_lookup is not None
    assert manager.last_lookup["hit_tokens"] >= len([1, 2, 3] + branch_two)
    assert prompt_lengths[-1] < len(followup_prompt)

    manager.close()


def test_openai_chat_route_reuses_transcript_cache(monkeypatch, tmp_path):
    namespace = "base:test-model"
    model = CountingTinyModel()
    tokenizer = ChatTokenizer()
    manager = _manager(tmp_path)
    inference = InferenceBackend(transcript_cache=manager)
    prompt_lengths: list[int] = []

    def fake_generate_step(
        *,
        prompt,
        model,
        max_tokens,
        sampler,
        max_kv_size,
        kv_bits,
        kv_group_size,
        quantized_kv_start,
        prompt_cache=None,
    ):
        del sampler, max_kv_size, kv_bits, kv_group_size, quantized_kv_start
        prompt_lengths.append(int(prompt.shape[0]))
        if prompt_cache is None:
            prompt_cache = make_prompt_cache(model)
        if int(prompt.shape[0]) > 0:
            model(prompt[None], cache=prompt_cache)
            mx.eval([cache.state for cache in prompt_cache])
        for token_id in [7, 8][:max_tokens]:
            model(mx.array([token_id])[None], cache=prompt_cache)
            mx.eval([cache.state for cache in prompt_cache])
            yield mx.array(token_id), mx.zeros(32)

    generate_module = importlib.import_module("mlx_lm.generate")
    monkeypatch.setattr(generate_module, "generate_step", fake_generate_step)

    backend = SimpleNamespace(
        config=SimpleNamespace(
            base_model="test-model",
            max_kv_cache_size=None,
            kv_cache_bits=None,
            kv_cache_group_size=64,
            quantized_kv_start=0,
            checkpoints_base=tmp_path / "checkpoints",
        ),
        _base_model=model,
        _base_tokenizer=tokenizer,
        _ensure_base_model=lambda model_name=None: None,
        _base_namespace=lambda base_model: namespace,
        _student_namespace=lambda model_name: f"student:{model_name}",
        _path_namespace=lambda path, base_model: f"path:{path}",
        _validate_checkpoint_path=lambda path: Path(path),
        _load_sampling_model=lambda checkpoint_path, base_model=None: (model, tokenizer),
        models={},
        tokenizers={},
        inference=inference,
    )

    app = FastAPI()
    register_openai_routes(app, backend)
    client = TestClient(app)

    payload = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 2}
    first = client.post("/v1/chat/completions", json=payload)
    manager.wait_for_idle()
    second = client.post("/v1/chat/completions", json=payload)
    manager.wait_for_idle()

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["choices"][0]["message"]["content"] == second.json()["choices"][0]["message"]["content"]
    assert prompt_lengths[1] < prompt_lengths[0]
    assert manager.last_lookup is not None
    assert manager.last_lookup["hit_tokens"] > 0

    manager.close()
