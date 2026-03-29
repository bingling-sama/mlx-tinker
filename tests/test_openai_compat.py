"""Tests for OpenAI-compatible API endpoints.

Uses mocked backend to test endpoint routing, response format,
and streaming without loading a real model.
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlx_tinker.api.openai_compat import register_openai_routes, router


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 128 for c in text[:10]]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(max(t % 26 + 65, 65)) for t in tokens)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


@pytest.fixture
def mock_backend():
    backend = MagicMock()
    backend.config.base_model = "test-model"
    backend.config.max_kv_cache_size = 123
    backend._base_model = MagicMock()
    backend._base_tokenizer = FakeTokenizer()
    backend._ensure_base_model = MagicMock()
    return backend


@pytest.fixture
def app(mock_backend):
    app = FastAPI()
    register_openai_routes(app, mock_backend)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def _mock_generate_step(prompt, model, max_tokens, sampler):
    """Yield fake tokens for testing."""
    import mlx.core as mx
    for i in range(min(3, max_tokens)):
        yield mx.array(i + 1), mx.zeros(128)


class TestChatCompletions:
    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_basic_response_format(self, mock_gen, client):
        mock_gen.return_value = [1, 2, 3]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert "message" in data["choices"][0]
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data
        assert data["usage"]["completion_tokens"] == 3

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_finish_reason_length(self, mock_gen, client):
        mock_gen.return_value = [1, 2, 3]  # No EOS token (eos=0)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
            },
        )
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "length"

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_finish_reason_stop(self, mock_gen, client):
        mock_gen.return_value = [1, 2, 0]  # Ends with EOS=0

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10,
            },
        )
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_model_field_echoed(self, mock_gen, client):
        mock_gen.return_value = [1]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "my-custom-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.json()["model"] == "my-custom-model"

    def test_streaming_smoke(self, client, monkeypatch):
        import mlx.core as mx
        import mlx_tinker.api.openai_compat as oai

        def fake_iter_generated_tokens(model, prompt_tokens, temperature, top_p, max_tokens):
            assert model is not None
            assert prompt_tokens
            assert temperature == 1.0
            assert top_p == 1.0
            assert max_tokens == 5
            yield mx.array(1), mx.zeros(8)
            yield mx.array(0), mx.zeros(8)

        monkeypatch.setattr(oai, "_iter_generated_tokens", fake_iter_generated_tokens)

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 5,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            events = [line[6:] for line in resp.iter_lines() if line.startswith("data: ")]

        assert events[-1] == "[DONE]"
        payloads = [json.loads(event) for event in events[:-1]]
        assert payloads[0]["object"] == "chat.completion.chunk"
        assert payloads[0]["choices"][0]["delta"]["content"] == "B"
        assert payloads[-1]["choices"][0]["finish_reason"] == "stop"

    def test_streaming_uses_configured_max_kv_size(self, client, monkeypatch):
        import mlx.core as mx

        captured = {}

        def fake_generate_step(*, prompt, model, max_tokens, sampler, max_kv_size):
            captured["max_kv_size"] = max_kv_size
            yield mx.array(0), mx.zeros(8)

        generate_module = importlib.import_module("mlx_lm.generate")
        monkeypatch.setattr(generate_module, "generate_step", fake_generate_step)

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            list(resp.iter_lines())

        assert captured["max_kv_size"] == 123


class TestCompletions:
    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_basic_response_format(self, mock_gen, client):
        mock_gen.return_value = [5, 6, 7]

        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello world", "max_tokens": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]
        assert "usage" in data

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_prompt_as_list(self, mock_gen, client):
        mock_gen.return_value = [1]

        resp = client.post(
            "/v1/completions",
            json={"prompt": ["first prompt", "ignored"], "max_tokens": 3},
        )
        assert resp.status_code == 200


class TestListModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][0]["owned_by"] == "mlx-tinker"


class TestBackendNotInitialized:
    def test_chat_completions_503(self):
        """Should return 503 when backend is None."""
        import mlx_tinker.api.openai_compat as oai

        app = FastAPI()
        app.include_router(router)
        # Reset _backend to None
        old = oai._backend
        oai._backend = None
        try:
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 503
        finally:
            oai._backend = old

    def test_completions_503(self):
        import mlx_tinker.api.openai_compat as oai

        app = FastAPI()
        app.include_router(router)
        old = oai._backend
        oai._backend = None
        try:
            client = TestClient(app)
            resp = client.post(
                "/v1/completions",
                json={"prompt": "hello"},
            )
            assert resp.status_code == 503
        finally:
            oai._backend = old
