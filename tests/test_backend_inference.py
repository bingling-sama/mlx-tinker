"""Unit tests for the inference backend (sample)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_tinker.backend.inference import InferenceBackend
from mlx_tinker.types import (
    EncodedTextChunk,
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
