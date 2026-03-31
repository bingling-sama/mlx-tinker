from __future__ import annotations

import json

from fastapi.testclient import TestClient

from integrations.openclaw_rl_local.api_server import (
    OpenClawLocalProxy,
    _extract_tool_calls,
    _postprocess_response_text,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    }
]


class _FakeSequence:
    def __init__(self, tokens, logprobs, stop_reason="stop"):
        self.tokens = tokens
        self.logprobs = logprobs
        self.stop_reason = stop_reason


class _FakeSampleResponse:
    def __init__(self, sequence):
        self.sequences = [sequence]


class _FakeSamplingClient:
    def __init__(self, sequence):
        self.sequence = sequence
        self.calls = []

    async def sample_async(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeSampleResponse(self.sequence)


class _CapturingTokenizer:
    eos_token_id = 0

    def __init__(self, decoded_text: str):
        self.decoded_text = decoded_text
        self.last_template_messages = None
        self.last_template_kwargs = None

    def encode(self, text: str, add_special_tokens=False):
        del add_special_tokens
        return [ord(c) % 128 for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        del tokens, skip_special_tokens
        return self.decoded_text

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.last_template_messages = messages
        self.last_template_kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            **kwargs,
        }
        return "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)


def _make_proxy(decoded_text: str) -> tuple[OpenClawLocalProxy, TestClient, _CapturingTokenizer]:
    tokenizer = _CapturingTokenizer(decoded_text)
    sampling_client = _FakeSamplingClient(_FakeSequence(tokens=[1, 2, 3], logprobs=[-0.3, -0.4, -0.5]))
    proxy = OpenClawLocalProxy(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        served_model_name="qwen3.5-local",
        api_key=None,
    )
    proxy._build_model_input = lambda prompt_tokens: prompt_tokens  # type: ignore[method-assign]
    proxy._build_sampling_params = lambda body: body  # type: ignore[method-assign]
    return proxy, TestClient(proxy.app), tokenizer


def _sse_events(response) -> list[str]:
    return [line[6:] for line in response.iter_lines() if line.startswith("data: ")]


def test_extract_tool_calls_supports_bare_json_payload():
    content, tool_calls = _extract_tool_calls(
        json.dumps({"name": "get_weather", "arguments": {"location": "Tokyo"}})
    )

    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "Tokyo"}


def test_extract_tool_calls_supports_qwen_json_tag_payload():
    content, tool_calls = _extract_tool_calls(
        '<tool_call>{"name":"get_weather","arguments":{"location":"Tokyo"}}</tool_call>'
    )

    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "Tokyo"}


def test_extract_tool_calls_supports_qwen_xml_payload():
    content, tool_calls = _extract_tool_calls(
        '<tool_call><function=get_weather><parameter=location>"Tokyo"</parameter></function></tool_call>'
    )

    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "Tokyo"}


def test_extract_tool_calls_supports_kimi_payload():
    content, tool_calls = _extract_tool_calls(
        '<|tool_call_begin|>get_weather<|tool_call_argument_begin|>{"location":"Tokyo"}<|tool_call_end|>'
    )

    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "Tokyo"}


def test_extract_tool_calls_preserves_mixed_content():
    content, tool_calls = _extract_tool_calls(
        'Checking now.\n<tool_call>{"name":"get_weather","arguments":{"location":"Tokyo"}}</tool_call>\nStand by.'
    )

    assert len(tool_calls) == 1
    assert "Checking now." in content
    assert "Stand by." in content
    assert "<tool_call>" not in content


def test_extract_tool_calls_keeps_malformed_payload_as_plain_text():
    raw = '<tool_call>{"name": }</tool_call>'

    content, tool_calls = _extract_tool_calls(raw)

    assert content == raw
    assert tool_calls == []


def test_postprocess_response_text_strips_thinking_before_tool_call_parsing():
    content, tool_calls = _postprocess_response_text(
        '<think>private scratchpad</think><tool_call>{"name":"get_weather","arguments":{"location":"Tokyo"}}</tool_call>',
        parse_tool_calls=True,
    )

    assert content is None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"


def test_proxy_parses_tool_call_markup_when_tools_are_present():
    proxy, client, _tokenizer = _make_proxy(
        '<tool_call>{"name":"get_weather","arguments":{"location":"Tokyo"}}</tool_call>'
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-local",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": TOOLS,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["content"] is None
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "location": "Tokyo"
    }
    assert proxy.records_since(0)[0].response_text == ""


def test_proxy_keeps_json_like_output_as_plain_text_without_tools():
    _proxy, client, _tokenizer = _make_proxy('{"name":"get_weather","arguments":{"location":"Tokyo"}}')

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-local",
            "messages": [{"role": "user", "content": "weather?"}],
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "tool_calls" not in payload["choices"][0]["message"]
    assert payload["choices"][0]["message"]["content"] == '{"name":"get_weather","arguments":{"location":"Tokyo"}}'
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_proxy_keeps_tool_markup_as_plain_text_when_tool_choice_is_none():
    _proxy, client, tokenizer = _make_proxy(
        '<tool_call>{"name":"get_weather","arguments":{"location":"Tokyo"}}</tool_call>'
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-local",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": TOOLS,
            "tool_choice": "none",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "tool_calls" not in payload["choices"][0]["message"]
    assert payload["choices"][0]["message"]["content"] == (
        '<tool_call>{"name":"get_weather","arguments":{"location":"Tokyo"}}</tool_call>'
    )
    assert "tools" not in tokenizer.last_template_kwargs


def test_proxy_normalizes_replayed_tool_history_for_chat_template():
    _proxy, client, tokenizer = _make_proxy("Final answer")

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-local",
            "messages": [
                {"role": "developer", "content": "be helpful"},
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_prev",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location":"Tokyo"}'},
                        }
                    ],
                },
                {"role": "tool", "content": '{"temp_c":20}', "tool_call_id": "call_prev"},
                {"role": "user", "content": [{"type": "text", "text": "What happened?"}]},
            ],
            "tools": TOOLS,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    captured = tokenizer.last_template_messages
    assert captured[0]["role"] == "system"
    assert captured[2]["tool_calls"][0]["function"]["arguments"] == {"location": "Tokyo"}
    assert captured[3]["tool_call_id"] == "call_prev"
    assert captured[4]["content"] == "What happened?"


def test_streaming_tool_call_only_emits_tool_call_delta_and_done():
    _proxy, client, _tokenizer = _make_proxy(
        '<think>scratchpad</think><tool_call>{"name":"get_weather","arguments":{"location":"Tokyo"}}</tool_call>'
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-local",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": TOOLS,
            "stream": True,
            "max_tokens": 16,
        },
    ) as response:
        assert response.status_code == 200
        events = _sse_events(response)

    payloads = [json.loads(event) for event in events[:-1]]
    first_delta = payloads[0]["choices"][0]["delta"]
    assert first_delta["role"] == "assistant"
    assert "content" not in first_delta
    assert first_delta["tool_calls"][0]["function"]["name"] == "get_weather"
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1] == "[DONE]"


def test_streaming_tool_calls_include_residual_content_in_first_chunk():
    _proxy, client, _tokenizer = _make_proxy(
        'Checking now.\n<tool_call>{"name":"get_weather","arguments":{"location":"Tokyo"}}</tool_call>'
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-local",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": TOOLS,
            "stream": True,
            "max_tokens": 16,
        },
    ) as response:
        assert response.status_code == 200
        events = _sse_events(response)

    payloads = [json.loads(event) for event in events[:-1]]
    first_delta = payloads[0]["choices"][0]["delta"]
    assert first_delta["role"] == "assistant"
    assert first_delta["content"] == "Checking now."
    assert first_delta["tool_calls"][0]["function"]["name"] == "get_weather"
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1] == "[DONE]"
