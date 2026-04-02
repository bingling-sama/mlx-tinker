"""Verify tinker SDK compatibility with mlx-tinker server.

Requires a running mlx-tinker server. Set environment:
    TINKER_BASE_URL=http://localhost:8010
    TINKER_API_KEY=tml-local

Run:
    TINKER_BASE_URL=http://localhost:8010 TINKER_API_KEY=tml-local \
        uv run pytest scripts/test_tinker_sdk_compat.py -v -s
"""

from __future__ import annotations

import asyncio
import os

import pytest
import tinker
import torch
from transformers import AutoTokenizer

MODEL = os.environ.get("MLX_TINKER_MODEL", "Qwen/Qwen3.5-4B")

pytestmark = pytest.mark.skipif(
    not os.environ.get("TINKER_BASE_URL"),
    reason="TINKER_BASE_URL not set (requires running mlx-tinker server)",
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def service_client():
    sc = tinker.ServiceClient()
    yield sc
    sc.holder.close()


class TestServiceClient:
    def test_session_created(self, service_client):
        assert service_client.holder._session_id
        print(f"  Session: {service_client.holder._session_id}")


class TestTrainingClient:
    @pytest.fixture(scope="class")
    def training_client(self, service_client):
        tc = asyncio.get_event_loop().run_until_complete(
            service_client.create_lora_training_client_async(
                base_model=MODEL, rank=8,
            )
        )
        return tc

    def test_training_client_created(self, training_client):
        assert training_client is not None

    def test_save_weights_returns_sampling_client(self, training_client):
        sampling = asyncio.get_event_loop().run_until_complete(
            training_client.save_weights_and_get_sampling_client_async(
                name="sdk_compat_test"
            )
        )
        assert sampling is not None

    def test_forward_backward(self, training_client, tokenizer):
        prompt = "Hello"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        # Minimal datum: 5 tokens
        tokens = prompt_ids[:4] + [tokenizer.eos_token_id or 0]
        n = len(tokens) - 1

        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:n]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(
                    torch.tensor(tokens[1 : n + 1], dtype=torch.long)
                ),
                "logprobs": tinker.TensorData.from_torch(
                    torch.zeros(n, dtype=torch.float32)
                ),
                "advantages": tinker.TensorData.from_torch(
                    torch.ones(n, dtype=torch.float32)
                ),
            },
        )

        async def run():
            fb_future = await training_client.forward_backward_async(
                [datum], loss_fn="ppo"
            )
            return await fb_future.result_async()

        result = asyncio.get_event_loop().run_until_complete(run())
        assert "loss:sum" in result.metrics
        print(f"  Loss: {result.metrics['loss:sum']:.4f}")

    def test_optim_step(self, training_client):
        async def run():
            future = await training_client.optim_step_async(
                tinker.AdamParams(learning_rate=1e-5)
            )
            return await future.result_async()

        result = asyncio.get_event_loop().run_until_complete(run())
        assert "learning_rate:unique" in result.metrics
        print(f"  LR: {result.metrics['learning_rate:unique']}")


class TestSamplingClient:
    @pytest.fixture(scope="class")
    def sampling_client(self, service_client):
        return asyncio.get_event_loop().run_until_complete(
            service_client.create_sampling_client_async(base_model=MODEL)
        )

    def test_sampling_client_created(self, sampling_client):
        assert sampling_client is not None

    def test_sample_generates_tokens(self, sampling_client, tokenizer):
        prompt = "What is 1+1?"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        chunk = tinker.EncodedTextChunk(tokens=list(prompt_ids), type="encoded_text")
        model_input = tinker.ModelInput(chunks=[chunk])
        params = tinker.SamplingParams(temperature=0.7, max_tokens=20)

        async def run():
            return await sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=params,
            )

        result = asyncio.get_event_loop().run_until_complete(run())
        seq = result.sequences[0]
        assert len(seq.tokens) > 0
        assert len(seq.logprobs) == len(seq.tokens)
        text = tokenizer.decode(seq.tokens)
        print(f"  Generated: {text[:80]!r}")
        print(f"  Tokens: {len(seq.tokens)}, logprobs: {len(seq.logprobs)}")

    def test_prompt_logprobs(self, sampling_client, tokenizer):
        """Teacher logprob extraction: include_prompt_logprobs=True."""
        prompt = "The answer is"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        chunk = tinker.EncodedTextChunk(tokens=list(prompt_ids), type="encoded_text")
        model_input = tinker.ModelInput(chunks=[chunk])
        params = tinker.SamplingParams(temperature=0.0, max_tokens=1)

        async def run():
            return await sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=params,
                include_prompt_logprobs=True,
                topk_prompt_logprobs=0,
            )

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.prompt_logprobs is not None
        assert len(result.prompt_logprobs) > 0
        print(f"  Prompt logprobs: {len(result.prompt_logprobs)} tokens")
