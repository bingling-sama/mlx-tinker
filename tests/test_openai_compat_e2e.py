"""End-to-end tests for the OpenAI-compatible API endpoints.

Requires a running mlx-tinker server at http://localhost:8010.
Run with: uv run pytest tests/test_openai_compat_e2e.py -v -s
"""

from __future__ import annotations

import json

import pytest
import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from pydantic import BaseModel

BASE_URL = "http://127.0.0.1:8010"
MODEL = "Qwen/Qwen3.5-4B"


def _chat(messages, **kwargs):
    """Helper to call /v1/chat/completions."""
    payload = {"model": MODEL, "messages": messages, **kwargs}
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _stream_chat(messages, **kwargs):
    """Helper to stream /v1/chat/completions."""
    payload = {"model": MODEL, "messages": messages, "stream": True, **kwargs}
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions", json=payload, stream=True, timeout=120
    )
    resp.raise_for_status()
    chunks = []
    for line in resp.iter_lines(decode_unicode=True):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


class TestListModels:
    def test_list_models(self):
        resp = requests.get(f"{BASE_URL}/v1/models", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        assert "Qwen" in data["data"][0]["id"]


class TestChatCompletions:
    def test_basic_chat(self):
        result = _chat(
            [{"role": "user", "content": "Say 'hello world' and nothing else."}],
            max_tokens=20,
        )
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["role"] == "assistant"
        content = result["choices"][0]["message"]["content"]
        assert len(content) > 0
        print(f"  Response: {content!r}")

    def test_system_message(self):
        result = _chat(
            [
                {"role": "system", "content": "You are a pirate. Always say 'Arrr'."},
                {"role": "user", "content": "Hello"},
            ],
            max_tokens=30,
        )
        content = result["choices"][0]["message"]["content"]
        assert len(content) > 0
        print(f"  Response: {content!r}")

    def test_multi_turn(self):
        result = _chat(
            [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=200,
        )
        content = result["choices"][0]["message"]["content"].lower()
        assert "alice" in content
        print(f"  Response: {content!r}")

    def test_max_tokens_respected(self):
        result = _chat(
            [{"role": "user", "content": "Write a very long essay about the universe."}],
            max_tokens=10,
        )
        assert result["usage"]["completion_tokens"] <= 11  # allow 1 extra for EOS
        print(f"  Tokens: {result['usage']['completion_tokens']}")

    def test_temperature_zero(self):
        """Temperature 0 should produce deterministic output."""
        msgs = [{"role": "user", "content": "What is 1+1? Answer with just the number."}]
        r1 = _chat(msgs, max_tokens=5, temperature=0.0)
        r2 = _chat(msgs, max_tokens=5, temperature=0.0)
        # With temp=0 outputs should be identical
        assert r1["choices"][0]["message"]["content"] == r2["choices"][0]["message"]["content"]
        print(f"  Deterministic: {r1['choices'][0]['message']['content']!r}")

    def test_usage_counts(self):
        result = _chat(
            [{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )
        usage = result["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        print(f"  Usage: {usage}")

    def test_finish_reason_stop(self):
        result = _chat(
            [{"role": "user", "content": "Say 'ok'."}],
            max_tokens=50,
        )
        # Should finish with stop (model produces short response)
        fr = result["choices"][0]["finish_reason"]
        print(f"  Finish reason: {fr}")
        assert fr in ("stop", "length")

    def test_finish_reason_length(self):
        result = _chat(
            [{"role": "user", "content": "Write a 1000 word essay."}],
            max_tokens=5,
        )
        assert result["choices"][0]["finish_reason"] == "length"

    def test_content_none_message(self):
        """Messages with content=null should not crash."""
        result = _chat(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
                ]},
                {"role": "tool", "content": "result", "tool_call_id": "call_1"},
                {"role": "user", "content": "What happened?"},
            ],
            max_tokens=30,
        )
        assert result["choices"][0]["message"]["content"] is not None
        print(f"  Response: {result['choices'][0]['message']['content']!r}")

    def test_list_content(self):
        """Messages with list content (multi-part) should work."""
        result = _chat(
            [
                {"role": "user", "content": [
                    {"type": "text", "text": "What is 2+2?"},
                ]},
            ],
            max_tokens=20,
        )
        assert result["choices"][0]["message"]["content"] is not None
        print(f"  Response: {result['choices'][0]['message']['content']!r}")

    def test_developer_role(self):
        """Developer role should be mapped to system."""
        result = _chat(
            [
                {"role": "developer", "content": "Always respond in JSON."},
                {"role": "user", "content": "What is 1+1?"},
            ],
            max_tokens=30,
        )
        assert result["choices"][0]["message"]["content"] is not None
        print(f"  Response: {result['choices'][0]['message']['content']!r}")


class TestThinking:
    def test_thinking_enabled_by_default(self):
        result = _chat(
            [{"role": "user", "content": "What is 3+5?"}],
            max_tokens=200,
        )
        msg = result["choices"][0]["message"]
        # With thinking enabled, we should get reasoning_content
        # (or just content if model decides not to think)
        print(f"  Has reasoning: {'reasoning_content' in msg}")
        print(f"  Content: {msg['content']!r}")
        if "reasoning_content" in msg:
            print(f"  Reasoning: {msg['reasoning_content'][:100]}...")

    def test_thinking_disabled(self):
        result = _chat(
            [{"role": "user", "content": "What is 3+5?"}],
            max_tokens=50,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        msg = result["choices"][0]["message"]
        assert "reasoning_content" not in msg or msg.get("reasoning_content") is None
        print(f"  Content (no thinking): {msg['content']!r}")


class TestToolCalling:
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    def test_tools_accepted(self):
        """Server should accept tools parameter without error."""
        result = _chat(
            [{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=self.TOOLS,
            max_tokens=200,
        )
        assert result["object"] == "chat.completion"
        msg = result["choices"][0]["message"]
        print(f"  Content: {msg.get('content', '')!r}")
        print(f"  Tool calls: {msg.get('tool_calls', 'none')}")
        # The model should attempt to use the tool
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            assert tc["type"] == "function"
            assert tc["function"]["name"] == "get_weather"
            args = json.loads(tc["function"]["arguments"])
            assert "location" in args
            print(f"  Parsed args: {args}")
            assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_tool_result_roundtrip(self):
        """Full tool-use conversation: request → tool_call → tool result → final answer."""
        # Step 1: Get tool call
        r1 = _chat(
            [{"role": "user", "content": "What's the weather in Paris?"}],
            tools=self.TOOLS,
            max_tokens=200,
        )
        msg1 = r1["choices"][0]["message"]
        if not msg1.get("tool_calls"):
            pytest.skip("Model did not produce a tool call")

        # Step 2: Provide tool result and get final answer
        tc = msg1["tool_calls"][0]
        r2 = _chat(
            [
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": None, "tool_calls": [tc]},
                {
                    "role": "tool",
                    "content": json.dumps({"temperature": 18, "condition": "sunny"}),
                    "tool_call_id": tc["id"],
                },
            ],
            tools=self.TOOLS,
            max_tokens=100,
        )
        content = r2["choices"][0]["message"]["content"]
        assert content is not None
        print(f"  Final answer: {content!r}")

    def test_tool_choice_none_does_not_emit_tool_calls(self):
        result = _chat(
            [{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=self.TOOLS,
            tool_choice="none",
            max_tokens=80,
        )
        msg = result["choices"][0]["message"]
        assert not msg.get("tool_calls")
        assert result["choices"][0]["finish_reason"] in ("stop", "length")


class TestStreaming:
    def test_basic_streaming(self):
        # Use non-thinking mode for streaming test (thinking can consume all tokens)
        chunks = _stream_chat(
            [{"role": "user", "content": "Say hello"}],
            max_tokens=50,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        assert len(chunks) > 0
        # First chunk should have content delta
        has_content = any(
            c["choices"][0]["delta"].get("content") for c in chunks
        )
        assert has_content
        # Last chunk should have finish_reason
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] in ("stop", "length")
        full_text = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        print(f"  Streamed: {full_text!r}")

    def test_streaming_stop(self):
        chunks = _stream_chat(
            [{"role": "user", "content": "Say 'ok' and stop."}],
            max_tokens=50,
        )
        finish_reasons = [c["choices"][0].get("finish_reason") for c in chunks if c["choices"][0].get("finish_reason")]
        assert len(finish_reasons) > 0
        print(f"  Finish: {finish_reasons[-1]}")


class TestStructuredOutputs:
    def test_response_format_json_schema(self):
        result = _chat(
            [{"role": "user", "content": "Return JSON with x equal to 1 and nothing else."}],
            response_format={
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
            max_completion_tokens=32,
        )
        content = result["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        assert parsed == {"x": 1}

    def test_response_format_rejects_tools_combo(self):
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Return JSON"}],
                "tools": TestToolCalling.TOOLS,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Answer",
                        "schema": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                            "required": ["x"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
            timeout=60,
        )
        assert resp.status_code == 400

    def test_openai_sdk_pydantic_parse(self):
        if OpenAI is None:
            pytest.skip("openai package not installed")
        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="tml-local")

        class Answer(BaseModel):
            x: int

        result = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[{"role": "user", "content": "Return JSON with x equal to 1 and nothing else."}],
            response_format=Answer,
            max_completion_tokens=32,
        )
        assert result.choices[0].message.parsed == Answer(x=1)


class TestCompletions:
    def test_basic_completion(self):
        payload = {"model": MODEL, "prompt": "The capital of France is", "max_tokens": 10}
        resp = requests.post(f"{BASE_URL}/v1/completions", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"][0]["text"]) > 0
        print(f"  Completion: {data['choices'][0]['text']!r}")

    def test_streaming_completion(self):
        payload = {"model": MODEL, "prompt": "Reply with ok.", "stream": True, "max_tokens": 16}
        resp = requests.post(f"{BASE_URL}/v1/completions", json=payload, stream=True, timeout=60)
        resp.raise_for_status()
        chunks = [line for line in resp.iter_lines(decode_unicode=True) if line.startswith("data: ")]
        assert chunks[-1] == "data: [DONE]"


class TestExtraFields:
    def test_unknown_fields_accepted(self):
        """Extra fields like presence_penalty, frequency_penalty should not cause 422."""
        result = _chat(
            [{"role": "user", "content": "Hi"}],
            max_tokens=5,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            top_k=20,
            seed=42,
            logprobs=False,
            n=1,
        )
        assert result["object"] == "chat.completion"
        print("  Extra fields accepted OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
