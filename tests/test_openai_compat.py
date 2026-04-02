"""Contract tests for OpenAI-compatible API endpoints."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlx_tinker.api.openai_compat import register_openai_routes, router


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.last_template_kwargs = None

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 128 for c in text[:32]]

    def decode(self, tokens: list[int], **_kwargs) -> str:
        return "".join(chr(max(t % 26 + 65, 65)) for t in tokens)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.last_template_kwargs = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


class CapturingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.last_template_kwargs = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            **kwargs,
        }
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


@pytest.fixture
def mock_backend(tmp_path):
    backend = MagicMock()
    backend.config.base_model = "test-model"
    backend.config.max_kv_cache_size = 123
    backend.config.kv_cache_bits = 4
    backend.config.kv_cache_group_size = 64
    backend.config.quantized_kv_start = 0
    backend.config.checkpoints_base = tmp_path / "checkpoints"
    backend.config.checkpoints_base.mkdir()
    backend._base_model = MagicMock(name="base-model")
    backend._base_tokenizer = FakeTokenizer()
    backend._ensure_base_model = MagicMock()
    backend.models = {}
    backend.tokenizers = {}
    backend._load_sampling_model = MagicMock(return_value=(MagicMock(name="checkpoint-model"), FakeTokenizer()))
    backend._validate_checkpoint_path = MagicMock(side_effect=lambda path: Path(path))
    backend._base_namespace = MagicMock(side_effect=lambda model_name: f"base:{model_name}")
    backend._student_namespace = MagicMock(side_effect=lambda model_name: f"student:{model_name}")
    backend._path_namespace = MagicMock(side_effect=lambda path, base_model: f"path:{path}")
    backend.inference._prepare_prompt_cache = MagicMock(
        side_effect=lambda _model, prompt_tokens, _namespace: (None, list(prompt_tokens))
    )
    backend.inference._checkpoint_transcript_prefixes = MagicMock()
    backend.inference._persist_final_transcript = MagicMock()
    return backend


@pytest.fixture
def app(mock_backend):
    app = FastAPI()
    register_openai_routes(app, mock_backend)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def _sse_events(response) -> list[str]:
    return [line[6:] for line in response.iter_lines() if line.startswith("data: ")]


def _patch_decode(monkeypatch, text: str):
    import mlx_tinker.api.openai_compat as oai

    monkeypatch.setattr(oai, "_safe_decode", lambda _tokenizer, _tokens, skip_special=True: text)


class TestChatCompletions:
    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_basic_response_format(self, mock_gen, client):
        mock_gen.return_value = [1, 2, 3]

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["usage"]["completion_tokens"] == 3

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_max_completion_tokens_alias_is_used(self, mock_gen, client):
        mock_gen.return_value = [1]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 7,
            },
        )

        assert resp.status_code == 200
        assert mock_gen.call_args.args[5] == 7

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_stop_string_trims_chat_content(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(monkeypatch, "alpha STOP omega")

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stop": "STOP",
                "max_tokens": 8,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "alpha "
        assert data["choices"][0]["finish_reason"] == "stop"

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_stop_list_trims_chat_streaming_content(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(monkeypatch, "alpha STOP omega")

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stop": ["STOP", "omega"],
                "stream": True,
                "max_tokens": 8,
            },
        ) as resp:
            assert resp.status_code == 200
            events = _sse_events(resp)

        payloads = [json.loads(event) for event in events[:-1]]
        text = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
        assert text == "alpha "
        assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
        assert events[-1] == "[DONE]"

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_tool_choice_none_ignores_tool_call_parsing(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(
            monkeypatch,
            "<tool_call><function=get_weather><parameter=location>\"Tokyo\"</parameter></function></tool_call>",
        )

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                        },
                    }
                ],
                "tool_choice": "none",
                "max_tokens": 16,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "tool_calls" not in data["choices"][0]["message"]
        assert "<tool_call>" in data["choices"][0]["message"]["content"]

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_tool_choice_required_rejects_missing_tool_call(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(monkeypatch, "plain text")

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                        },
                    }
                ],
                "tool_choice": "required",
                "max_tokens": 16,
            },
        )

        assert resp.status_code == 400
        assert "tool_choice='required'" in resp.json()["detail"]

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_tool_choice_forced_function_requires_exact_name(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(
            monkeypatch,
            "<tool_call><function=other_tool><parameter=location>\"Tokyo\"</parameter></function></tool_call>",
        )

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "max_tokens": 16,
            },
        )

        assert resp.status_code == 400
        assert "get_weather" in resp.json()["detail"]

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_tool_choice_forced_function_accepts_exact_name(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(
            monkeypatch,
            "<tool_call><function=get_weather><parameter=location>\"Tokyo\"</parameter></function></tool_call>",
        )

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "max_tokens": 16,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_multiple_tool_calls_are_parsed(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(
            monkeypatch,
            (
                "<tool_call><function=get_weather><parameter=location>\"Tokyo\"</parameter></function></tool_call>\n"
                "<tool_call><function=get_time><parameter=timezone>\"UTC\"</parameter></function></tool_call>"
            ),
        )

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "weather and time?"}],
                "tools": [
                    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
                    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}},
                ],
                "max_tokens": 32,
            },
        )

        assert resp.status_code == 200
        tool_calls = resp.json()["choices"][0]["message"]["tool_calls"]
        assert [tc["function"]["name"] for tc in tool_calls] == ["get_weather", "get_time"]

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_streaming_emits_tool_call_chunks(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(
            monkeypatch,
            "<tool_call><function=get_weather><parameter=location>\"Tokyo\"</parameter></function></tool_call>",
        )

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
                ],
                "stream": True,
                "max_tokens": 16,
            },
        ) as resp:
            assert resp.status_code == 200
            events = _sse_events(resp)

        payloads = [json.loads(event) for event in events[:-1]]
        assert payloads[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_qwen_thinking_block_populates_reasoning_content_without_tags(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(monkeypatch, "<think>private scratchpad</think>Final answer")

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16},
        )

        assert resp.status_code == 200
        message = resp.json()["choices"][0]["message"]
        assert message["reasoning_content"] == "private scratchpad"
        assert message["content"] == "Final answer"

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_qwen_thinking_block_and_xml_tool_call_parse_together(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(
            monkeypatch,
            (
                "<think>private scratchpad</think>"
                '<tool_call><function=get_weather><parameter=location>"Tokyo"</parameter></function></tool_call>'
            ),
        )

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                        },
                    }
                ],
                "max_tokens": 16,
            },
        )

        assert resp.status_code == 200
        message = resp.json()["choices"][0]["message"]
        assert message["reasoning_content"] == "private scratchpad"
        assert message["content"] is None
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_truncated_qwen_thinking_block_does_not_leak_into_visible_text(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(monkeypatch, "<think>private scratchpad only")

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16},
        )

        assert resp.status_code == 200
        message = resp.json()["choices"][0]["message"]
        assert message["reasoning_content"] == "private scratchpad only"
        assert message["content"] == ""

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_orphan_qwen_thinking_close_does_not_emit_empty_reasoning_content(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(monkeypatch, "</think>\n\nFinal answer")

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16},
        )

        assert resp.status_code == 200
        message = resp.json()["choices"][0]["message"]
        assert "reasoning_content" not in message
        assert message["content"] == "Final answer"

    def test_response_format_forces_disable_thinking_and_uses_schema_processor(self, client, mock_backend, monkeypatch):
        import mlx_tinker.api.openai_compat as oai

        tokenizer = CapturingTokenizer()
        mock_backend._base_tokenizer = tokenizer
        sentinel = object()
        monkeypatch.setattr(oai, "_make_json_schema_processor", lambda *args, **kwargs: sentinel)
        monkeypatch.setattr(oai, "_safe_decode", lambda *_args, **_kwargs: '{"x":1}')

        captured = {}

        def fake_generate_tokens(
            model,
            tokenizer,
            prompt_tokens,
            temperature,
            top_p,
            max_tokens,
            namespace=None,
            logits_processor=None,
        ):
            captured["max_tokens"] = max_tokens
            captured["logits_processor"] = logits_processor
            return [1, 2, 0]

        monkeypatch.setattr(oai, "_generate_tokens", fake_generate_tokens)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "json"}],
                "max_completion_tokens": 9,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Answer",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                            "required": ["x"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        )

        assert resp.status_code == 200
        assert captured["max_tokens"] == 9
        assert captured["logits_processor"] is sentinel
        assert tokenizer.last_template_kwargs["enable_thinking"] is False
        assert resp.json()["choices"][0]["message"]["content"] == '{"x":1}'

    def test_response_format_rejects_tools_combo(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "json"}],
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "Answer", "schema": {"type": "object"}},
                },
            },
        )

        assert resp.status_code == 400
        assert "tools and response_format" in resp.json()["detail"]

    def test_response_format_rejects_invalid_schema_shape(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "json"}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "Answer"}},
            },
        )

        assert resp.status_code == 400
        assert "response_format.json_schema.schema" in resp.json()["detail"]

    def test_json_schema_processor_accepts_latest_token_before_masking(self, monkeypatch):
        import mlx_tinker.api.openai_compat as oai

        events = []

        class FakeTokenizerInfo:
            def __init__(self, vocab_size=16):
                self.vocab_size = vocab_size

            @classmethod
            def from_huggingface(cls, *_args, **_kwargs):
                return cls()

        class FakeCompiler:
            def __init__(self, tokenizer_info):
                self.tokenizer_info = tokenizer_info

            def compile_json_schema(self, schema_str):
                return {"schema": schema_str, "tokenizer_info": self.tokenizer_info}

        class FakeMatcher:
            last_instance = None

            def __init__(self, compiled):
                self.compiled = compiled
                self.accept_calls = 0
                self.terminated = False
                FakeMatcher.last_instance = self

            def accept_token(self, token_id):
                self.accept_calls += 1
                events.append(("accept", token_id))
                return self.accept_calls > 1

            def fill_next_token_bitmask(self, bitmask):
                events.append(("fill", bitmask))

            def reset(self):
                events.append(("reset",))

            def is_terminated(self):
                return self.terminated

        fake_xgrammar = types.SimpleNamespace(
            TokenizerInfo=FakeTokenizerInfo,
            GrammarCompiler=FakeCompiler,
            GrammarMatcher=FakeMatcher,
            allocate_token_bitmask=lambda *_args: "bitmask",
        )
        fake_kernel_module = types.SimpleNamespace(
            apply_token_bitmask_mlx=lambda bitmask, logits, vocab_size: (
                events.append(("apply", bitmask, vocab_size)),
                ("masked", logits),
            )[1]
        )

        monkeypatch.setitem(sys.modules, "xgrammar", fake_xgrammar)
        monkeypatch.setitem(sys.modules, "xgrammar.kernels.apply_token_bitmask_mlx", fake_kernel_module)
        monkeypatch.setattr(oai, "_unwrap_hf_tokenizer", lambda tokenizer: tokenizer)
        monkeypatch.setattr(oai, "_model_vocab_size", lambda _model: 16)

        processor = oai._make_json_schema_processor(object(), object(), {"type": "object"})
        result = processor([7], "logits")

        assert result == ("masked", "logits")
        assert events == [
            ("accept", 7),
            ("reset",),
            ("accept", 7),
            ("fill", "bitmask"),
            ("apply", "bitmask", 16),
        ]

    def test_unknown_tool_choice_is_rejected(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
                "tool_choice": "banana",
            },
        )

        assert resp.status_code == 400

    def test_streaming_uses_configured_max_kv_size(self, client, monkeypatch):
        import mlx.core as mx

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
        assert captured["kv_bits"] == 4
        assert captured["kv_group_size"] == 64
        assert captured["quantized_kv_start"] == 0


class TestCompletions:
    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_basic_response_format(self, mock_gen, client):
        mock_gen.return_value = [5, 6, 7]

        resp = client.post("/v1/completions", json={"prompt": "Hello world", "max_tokens": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]
        assert "usage" in data

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_prompt_as_list(self, mock_gen, client):
        mock_gen.return_value = [1]

        resp = client.post("/v1/completions", json={"prompt": ["first prompt", "ignored"], "max_tokens": 3})
        assert resp.status_code == 200

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_stop_string_trims_completion(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(monkeypatch, "alpha STOP omega")

        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello", "stop": "STOP", "max_tokens": 8},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["text"] == "alpha "
        assert data["choices"][0]["finish_reason"] == "stop"

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_streaming_completion_returns_sse(self, mock_gen, client, monkeypatch):
        mock_gen.return_value = [1, 2, 3]
        _patch_decode(monkeypatch, "hello STOP world")

        with client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "Hello", "stream": True, "stop": "STOP", "max_tokens": 8},
        ) as resp:
            assert resp.status_code == 200
            events = _sse_events(resp)

        payloads = [json.loads(event) for event in events[:-1]]
        text = "".join(p["choices"][0].get("text", "") for p in payloads)
        assert text == "hello "
        assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
        assert payloads[0]["object"] == "text_completion"
        assert events[-1] == "[DONE]"


class TestModelRouting:
    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_training_model_id_uses_in_memory_model(self, mock_gen, client, mock_backend):
        training_model = MagicMock(name="training-model")
        training_tokenizer = FakeTokenizer()
        mock_backend.models["training-id"] = training_model
        mock_backend.tokenizers["training-id"] = training_tokenizer
        mock_gen.return_value = [1]

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "training-id", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert resp.status_code == 200
        assert mock_gen.call_args.args[0] is training_model
        assert mock_gen.call_args.args[1] is training_tokenizer

    @patch("mlx_tinker.api.openai_compat._generate_tokens")
    def test_checkpoint_model_id_uses_loader(self, mock_gen, client, mock_backend):
        mock_gen.return_value = [1]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model:checkpoint-a",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert resp.status_code == 200
        assert mock_backend._load_sampling_model.called
        load_path, load_base = mock_backend._load_sampling_model.call_args.args
        assert load_base == "test-model"
        assert load_path.endswith("checkpoint-a")

    def test_missing_checkpoint_returns_404(self, client, mock_backend):
        mock_backend._load_sampling_model.side_effect = FileNotFoundError("missing")

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model:missing",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert resp.status_code == 404
        assert "Checkpoint not found" in resp.json()["detail"]


class TestListModels:
    def test_list_models_includes_base_training_and_checkpoint(self, client, mock_backend):
        mock_backend.models["training-id"] = MagicMock()
        mock_backend.tokenizers["training-id"] = FakeTokenizer()
        checkpoint_dir = Path(mock_backend.config.checkpoints_base) / "training-id" / "sampler" / "checkpoint-a"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "adapters.safetensors").write_text("stub")
        (checkpoint_dir / "config.json").write_text(
            json.dumps({"base_model": "test-model", "lora_config": {"rank": 8, "alpha": 16.0}})
        )

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        ids = {entry["id"] for entry in data["data"]}
        assert "test-model" in ids
        assert "training-id" in ids
        assert "test-model:training-id/sampler/checkpoint-a" in ids


class TestBackendNotInitialized:
    def test_chat_completions_503(self):
        import mlx_tinker.api.openai_compat as oai

        app = FastAPI()
        app.include_router(router)
        old = oai._backend
        oai._backend = None
        try:
            client = TestClient(app)
            resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
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
            resp = client.post("/v1/completions", json={"prompt": "hello"})
            assert resp.status_code == 503
        finally:
            oai._backend = old
