"""Unit tests for the inference backend (sample)."""

import importlib
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_tinker.backend.inference import InferenceBackend
from mlx_tinker.types import (
    EncodedTextChunk,
    GeneratedSequence,
    ModelInput,
    SampleInput,
    SamplingParams,
)


class TinyLayer(nn.Module):
    """Single layer with KV cache compatibility."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, cache=None):
        return self.linear(x), cache


class TinyModel(nn.Module):
    """Minimal model compatible with mlx-lm generate_step (needs .layers)."""

    def __init__(self, vocab_size: int = 32, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = [TinyLayer(dim)]
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        h = self.embed(x)
        if cache is not None:
            for i, layer in enumerate(self.layers):
                h, cache[i] = layer(h, cache[i])
        else:
            for layer in self.layers:
                h, _ = layer(h)
        return self.head(h)


class FakeTokenizer:
    """Minimal tokenizer for testing."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t + 65) for t in tokens)


@pytest.fixture
def model():
    m = TinyModel(vocab_size=32, dim=16)
    mx.eval(m.parameters())
    m.eval()
    return m


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def inference():
    return InferenceBackend()


class TestSample:
    def test_same_seed_repeats_identical_simple_samples(self, model, tokenizer, monkeypatch):
        inference = InferenceBackend()
        inference_module = importlib.import_module("mlx_tinker.backend.inference")
        monkeypatch.setattr(inference_module, "_has_kv_cache_support", lambda _model: False)

        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=6, seed=7),
            num_samples=1,
        )

        first = inference.sample(model, tokenizer, request)
        second = inference.sample(model, tokenizer, request)

        assert first.sequences[0].tokens == second.sequences[0].tokens
        assert first.sequences[0].logprobs == second.sequences[0].logprobs

    def test_different_seeds_can_diverge_for_simple_samples(self, model, tokenizer, monkeypatch):
        inference = InferenceBackend()
        inference_module = importlib.import_module("mlx_tinker.backend.inference")
        monkeypatch.setattr(inference_module, "_has_kv_cache_support", lambda _model: False)

        request_a = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=8, seed=7),
            num_samples=1,
        )
        request_b = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=8, seed=9),
            num_samples=1,
        )

        result_a = inference.sample(model, tokenizer, request_a)
        result_b = inference.sample(model, tokenizer, request_b)

        assert result_a.sequences[0].tokens != result_b.sequences[0].tokens

    def test_basic_generation(self, model, tokenizer, inference):
        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=5),
            num_samples=1,
        )

        result = inference.sample(model, tokenizer, request)
        assert len(result.sequences) == 1
        assert len(result.sequences[0].tokens) <= 5
        assert result.sequences[0].stop_reason in ("length", "stop")

    def test_multiple_samples(self, model, tokenizer, inference):
        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=3),
            num_samples=3,
        )

        result = inference.sample(model, tokenizer, request)
        assert len(result.sequences) == 3

    def test_multiple_samples_uses_batched_generation(self, model, tokenizer, inference, monkeypatch):
        class FakeBatchGenerator:
            instances = []

            def __init__(self, _model, stop_tokens=None, max_kv_size=None):
                self.stop_tokens = stop_tokens
                self.max_kv_size = max_kv_size
                self.calls = 0
                self.closed = False
                self.prompts = None
                self.max_tokens = None
                self.samplers = None
                FakeBatchGenerator.instances.append(self)

            def insert(self, prompts, max_tokens=None, samplers=None, **_kwargs):
                self.prompts = prompts
                self.max_tokens = max_tokens
                self.samplers = samplers
                return [10, 11]

            def next(self):
                self.calls += 1
                if self.calls == 1:
                    return [
                        SimpleNamespace(
                            uid=10,
                            token=4,
                            logprobs=mx.array([-5.0, -4.0, -3.0, -2.0, -1.0]),
                            finish_reason=None,
                        ),
                        SimpleNamespace(
                            uid=11,
                            token=2,
                            logprobs=mx.array([-6.0, -5.0, -0.5, -3.0, -4.0]),
                            finish_reason=None,
                        ),
                    ]
                if self.calls == 2:
                    return [
                        SimpleNamespace(
                            uid=10,
                            token=0,
                            logprobs=mx.array([-0.25, -4.0, -5.0, -6.0, -7.0]),
                            finish_reason="stop",
                        ),
                        SimpleNamespace(
                            uid=11,
                            token=3,
                            logprobs=mx.array([-7.0, -6.0, -5.0, -0.75, -4.0]),
                            finish_reason="length",
                        ),
                    ]
                return []

            def close(self):
                self.closed = True

        generate_module = importlib.import_module("mlx_lm.generate")
        monkeypatch.setattr(generate_module, "BatchGenerator", FakeBatchGenerator)
        inference_module = importlib.import_module("mlx_tinker.backend.inference")
        monkeypatch.setattr(inference_module, "_has_kv_cache_support", lambda _model: True)

        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=2, stop_tokens=[0]),
            num_samples=2,
        )

        result = inference.sample(model, tokenizer, request)

        assert len(result.sequences) == 2
        assert result.sequences[0].tokens == [4, 0]
        assert result.sequences[0].stop_reason == "stop"
        assert result.sequences[0].logprobs == [-1.0, -0.25]
        assert result.sequences[1].tokens == [2, 3]
        assert result.sequences[1].stop_reason == "length"
        assert result.sequences[1].logprobs == [-0.5, -0.75]

        fake = FakeBatchGenerator.instances[0]
        assert fake.prompts == [[1, 2, 3], [1, 2, 3]]
        assert fake.max_tokens == [2, 2]
        assert len(fake.samplers) == 2
        assert fake.stop_tokens == {0}
        assert fake.max_kv_size is None
        assert fake.closed is True

    def test_multiple_samples_with_model_path_uses_batched_generation(
        self, model, tokenizer, inference, monkeypatch
    ):
        calls = {"batch": 0, "step": 0}

        def fake_batch(model, prompt_tokens, sp, num_samples):
            calls["batch"] += 1
            return [
                GeneratedSequence(stop_reason="length", tokens=[1], logprobs=[-0.1])
                for _ in range(num_samples)
            ]

        def fake_step(model, prompt_tokens, sp, num_samples):
            calls["step"] += 1
            raise AssertionError("Sequential generation should not be used after model resolution")

        monkeypatch.setattr(inference, "_sample_with_batch_generator", fake_batch)
        monkeypatch.setattr(inference, "_sample_with_generate_step", fake_step)
        inference_module = importlib.import_module("mlx_tinker.backend.inference")
        monkeypatch.setattr(inference_module, "_has_kv_cache_support", lambda _model: True)

        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=2, stop_tokens=[0]),
            num_samples=8,
            model_path="checkpoints/sampler-1",
        )

        result = inference.sample(model, tokenizer, request)

        assert len(result.sequences) == 8
        assert calls == {"batch": 1, "step": 0}

    def test_generate_step_forwards_max_kv_cache_size(self, model, tokenizer, monkeypatch):
        captured = {}

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
        ):
            captured["max_kv_size"] = max_kv_size
            captured["kv_bits"] = kv_bits
            captured["kv_group_size"] = kv_group_size
            captured["quantized_kv_start"] = quantized_kv_start
            yield mx.array(0), mx.array([-0.25, -4.0, -5.0])

        generate_module = importlib.import_module("mlx_lm.generate")
        monkeypatch.setattr(generate_module, "generate_step", fake_generate_step)
        inference_module = importlib.import_module("mlx_tinker.backend.inference")
        monkeypatch.setattr(inference_module, "_has_kv_cache_support", lambda _model: True)

        inference = InferenceBackend(
            max_kv_cache_size=77,
            kv_cache_bits=4,
            kv_cache_group_size=32,
            quantized_kv_start=5,
        )
        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=1, stop_tokens=[0]),
            num_samples=1,
        )

        result = inference.sample(model, tokenizer, request)

        assert result.sequences[0].tokens == [0]
        assert captured["max_kv_size"] == 77
        assert captured["kv_bits"] == 4
        assert captured["kv_group_size"] == 32
        assert captured["quantized_kv_start"] == 5

    def test_multiple_samples_falls_back_to_generate_step_when_kv_quant_enabled(
        self, model, tokenizer, monkeypatch
    ):
        calls = {"batch": 0, "step": 0}

        def fake_batch(model, prompt_tokens, sp, num_samples):
            calls["batch"] += 1
            raise AssertionError("BatchGenerator should be bypassed when kv quantization is enabled")

        def fake_step(model, prompt_tokens, sp, num_samples, namespace=None):
            calls["step"] += 1
            return [
                GeneratedSequence(stop_reason="length", tokens=[1], logprobs=[-0.1])
                for _ in range(num_samples)
            ]

        inference = InferenceBackend(kv_cache_bits=4)
        monkeypatch.setattr(inference, "_sample_with_batch_generator", fake_batch)
        monkeypatch.setattr(inference, "_sample_with_generate_step", fake_step)
        inference_module = importlib.import_module("mlx_tinker.backend.inference")
        monkeypatch.setattr(inference_module, "_has_kv_cache_support", lambda _model: True)

        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=2),
            num_samples=3,
        )

        result = inference.sample(model, tokenizer, request)

        assert len(result.sequences) == 3
        assert calls == {"batch": 0, "step": 1}

    def test_sample_batch_batches_multiple_requests(self, model, tokenizer, inference, monkeypatch):
        class FakeBatchGenerator:
            instances = []

            def __init__(self, _model, stop_tokens=None, max_kv_size=None):
                self.stop_tokens = stop_tokens
                self.max_kv_size = max_kv_size
                self.calls = 0
                self.closed = False
                self.prompts = None
                FakeBatchGenerator.instances.append(self)

            def insert(self, prompts, max_tokens=None, samplers=None, **_kwargs):
                self.prompts = prompts
                return [10, 11, 12]

            def next(self):
                self.calls += 1
                if self.calls == 1:
                    return [
                        SimpleNamespace(
                            uid=10,
                            token=4,
                            logprobs=mx.array([-5.0, -4.0, -3.0, -2.0, -1.0]),
                            finish_reason=None,
                        ),
                        SimpleNamespace(
                            uid=11,
                            token=2,
                            logprobs=mx.array([-6.0, -5.0, -0.5, -3.0, -4.0]),
                            finish_reason=None,
                        ),
                        SimpleNamespace(
                            uid=12,
                            token=1,
                            logprobs=mx.array([-6.0, -0.1, -0.5, -3.0, -4.0]),
                            finish_reason=None,
                        ),
                    ]
                if self.calls == 2:
                    return [
                        SimpleNamespace(
                            uid=10,
                            token=0,
                            logprobs=mx.array([-0.25, -4.0, -5.0, -6.0, -7.0]),
                            finish_reason="stop",
                        ),
                        SimpleNamespace(
                            uid=11,
                            token=3,
                            logprobs=mx.array([-7.0, -6.0, -5.0, -0.75, -4.0]),
                            finish_reason="length",
                        ),
                        SimpleNamespace(
                            uid=12,
                            token=0,
                            logprobs=mx.array([-0.2, -4.0, -5.0, -6.0, -7.0]),
                            finish_reason="stop",
                        ),
                    ]
                return []

            def close(self):
                self.closed = True

        generate_module = importlib.import_module("mlx_lm.generate")
        monkeypatch.setattr(generate_module, "BatchGenerator", FakeBatchGenerator)
        inference_module = importlib.import_module("mlx_tinker.backend.inference")
        monkeypatch.setattr(inference_module, "_has_kv_cache_support", lambda _model: True)

        requests = [
            SampleInput(
                prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
                sampling_params=SamplingParams(temperature=1.0, max_tokens=2, stop_tokens=[0]),
                num_samples=2,
            ),
            SampleInput(
                prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[9, 8, 7])]),
                sampling_params=SamplingParams(temperature=1.0, max_tokens=2, stop_tokens=[0]),
                num_samples=1,
            ),
        ]

        results = inference.sample_batch(model, tokenizer, requests)

        assert len(results) == 2
        assert [seq.tokens for seq in results[0].sequences] == [[4, 0], [2, 3]]
        assert [seq.tokens for seq in results[1].sequences] == [[1, 0]]

        fake = FakeBatchGenerator.instances[0]
        assert fake.prompts == [[1, 2, 3], [1, 2, 3], [9, 8, 7]]
        assert fake.stop_tokens == {0}
        assert fake.max_kv_size is None
        assert fake.closed is True

    def test_batch_generator_forwards_max_kv_cache_size(self, model, tokenizer, monkeypatch):
        class FakeBatchGenerator:
            instances = []

            def __init__(self, _model, stop_tokens=None, max_kv_size=None):
                self.max_kv_size = max_kv_size
                self.calls = 0
                FakeBatchGenerator.instances.append(self)

            def insert(self, prompts, max_tokens=None, samplers=None, **_kwargs):
                return list(range(10, 10 + len(prompts)))

            def next(self):
                self.calls += 1
                if self.calls == 1:
                    return [
                        SimpleNamespace(
                            uid=10,
                            token=0,
                            logprobs=mx.array([-0.25, -4.0, -5.0]),
                            finish_reason="stop",
                        ),
                        SimpleNamespace(
                            uid=11,
                            token=0,
                            logprobs=mx.array([-0.25, -4.0, -5.0]),
                            finish_reason="stop",
                        ),
                    ]
                return []

            def close(self):
                return None

        generate_module = importlib.import_module("mlx_lm.generate")
        monkeypatch.setattr(generate_module, "BatchGenerator", FakeBatchGenerator)
        inference_module = importlib.import_module("mlx_tinker.backend.inference")
        monkeypatch.setattr(inference_module, "_has_kv_cache_support", lambda _model: True)

        inference = InferenceBackend(max_kv_cache_size=99)
        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=2, stop_tokens=[0]),
            num_samples=2,
        )

        inference.sample(model, tokenizer, request)

        assert FakeBatchGenerator.instances[0].max_kv_size == 99

    def test_stop_token(self, model, tokenizer, inference):
        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=100, stop_tokens=[0]),
            num_samples=1,
        )

        result = inference.sample(model, tokenizer, request)
        seq = result.sequences[0]
        # If stop token was hit, last token should be 0 or length < max_tokens
        assert seq.stop_reason in ("stop", "length")

    def test_logprobs_returned(self, model, tokenizer, inference):
        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=3),
            num_samples=1,
        )

        result = inference.sample(model, tokenizer, request)
        seq = result.sequences[0]
        assert len(seq.logprobs) == len(seq.tokens)

    def test_prompt_logprobs(self, model, tokenizer, inference):
        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3, 4])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=2),
            num_samples=1,
            prompt_logprobs=True,
        )

        result = inference.sample(model, tokenizer, request)
        assert result.prompt_logprobs is not None
        assert len(result.prompt_logprobs) == 4  # same as prompt length

    def test_temperature_zero_deterministic(self, model, tokenizer, inference):
        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=0.0, max_tokens=5),
            num_samples=1,
        )

        r1 = inference.sample(model, tokenizer, request)
        r2 = inference.sample(model, tokenizer, request)
        assert r1.sequences[0].tokens == r2.sequences[0].tokens

    def test_max_tokens_one(self, model, tokenizer, inference):
        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(temperature=1.0, max_tokens=1),
            num_samples=1,
        )

        result = inference.sample(model, tokenizer, request)
        assert len(result.sequences) == 1
        assert len(result.sequences[0].tokens) == 1

    def test_resolve_stop_tokens_with_qwen_im_end_and_strings(self, inference):
        from mlx_tinker.backend.inference import resolve_stop_tokens

        class MockQwenTokenizer:
            eos_token_id = 151643

            def encode(self, text, add_special_tokens=False):
                if text == "<|im_end|>":
                    return [151645]
                if text == "<|endoftext|>":
                    return [151643]
                if text == "###":
                    return [9999]
                if text == "multi_word":
                    return [101, 102]
                return [1]

        tok = MockQwenTokenizer()
        # Default incorporates eos_token_id (151643) and common end token <|im_end|> (151645)
        resolved = resolve_stop_tokens(tok)
        assert 151643 in resolved
        assert 151645 in resolved

        # Client-provided stop_strings (single token encoded) are incorporated
        resolved_custom = resolve_stop_tokens(tok, stop_strings=["###", "multi_word"])
        assert 9999 in resolved_custom
        assert 101 not in resolved_custom

    def test_sampling_params_stop_field_normalization(self):
        # Single string
        sp1 = SamplingParams(stop="<|im_end|>")
        assert sp1.stop_strings == ["<|im_end|>"]

        # List of strings and token IDs
        sp2 = SamplingParams(stop=["<|im_end|>", 151645])
        assert sp2.stop_strings == ["<|im_end|>"]
        assert sp2.stop_tokens == [151645]

        # Single int
        sp3 = SamplingParams(stop=151645)
        assert sp3.stop_tokens == [151645]

    def test_early_stopping_on_qwen_im_end_with_generate_step(
        self, model, monkeypatch, inference
    ):
        class MockQwenTokenizer:
            eos_token_id = 151643

            def encode(self, text, add_special_tokens=False):
                if text == "<|im_end|>":
                    return [151645]
                return [1]

        tok = MockQwenTokenizer()

        def fake_generate_step(*, prompt, model, max_tokens, sampler, **kwargs):
            # Model emits 42, then Qwen's <|im_end|> (151645), then more tokens
            for token_id in [42, 151645, 99, 100, 101]:
                yield mx.array(token_id), mx.zeros(10)

        generate_module = importlib.import_module("mlx_lm.generate")
        monkeypatch.setattr(generate_module, "generate_step", fake_generate_step)
        inference_module = importlib.import_module("mlx_tinker.backend.inference")
        monkeypatch.setattr(inference_module, "_has_kv_cache_support", lambda _model: True)

        # Client requests max_tokens=64 with stop_strings=["<|im_end|>"]
        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(
                temperature=1.0, max_tokens=64, stop_strings=["<|im_end|>"]
            ),
            num_samples=1,
        )

        result = inference.sample(model, tok, request)
        assert len(result.sequences) == 1
        seq = result.sequences[0]
        # Generation must stop early at 151645 and NOT continue to 64 tokens
        assert seq.tokens == [42, 151645]
        assert seq.stop_reason == "stop"

    def test_early_stopping_simple_loop_on_qwen_im_end(
        self, model, monkeypatch, inference
    ):
        class MockQwenTokenizer:
            eos_token_id = 151643

            def encode(self, text, add_special_tokens=False):
                if text == "<|im_end|>":
                    return [151645]
                return [1]

        tok = MockQwenTokenizer()

        tokens_to_sample = iter([42, 151645, 99, 100])
        monkeypatch.setattr(
            "mlx_tinker.backend.inference._sample_token",
            lambda *args, **kwargs: next(tokens_to_sample),
        )
        inference_module = importlib.import_module("mlx_tinker.backend.inference")
        monkeypatch.setattr(inference_module, "_has_kv_cache_support", lambda _model: False)

        request = SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            sampling_params=SamplingParams(
                temperature=1.0, max_tokens=64, stop_strings=["<|im_end|>"]
            ),
            num_samples=1,
        )

        result = inference.sample(model, tok, request)
        assert len(result.sequences) == 1
        seq = result.sequences[0]
        assert seq.tokens == [42, 151645]
        assert seq.stop_reason == "stop"
