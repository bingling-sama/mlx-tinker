"""OpenAI-compatible proxy that fronts the local sampling client."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .adapter import LocalSamplingClientAdapter

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_QWEN_JSON_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_QWEN_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_KIMI_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*([a-zA-Z0-9_.-]+)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*"
    r"(?:<\|tool_call_argument_end\|>\s*)?<\|tool_call_end\|>",
    re.DOTALL,
)
_KIMI_TOOL_CALL_SECTION_RE = re.compile(
    r"<\|tool_calls_section_(?:begin|end)\|>",
    re.DOTALL,
)


@dataclass(frozen=True)
class ProxyTrainingRecord:
    trace_id: str
    task_id: str | None
    session_key: str | None
    messages: tuple[dict[str, Any], ...]
    prompt_text: str
    prompt_tokens: tuple[int, ...]
    response_text: str
    response_tokens: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    tool_calls: tuple[dict[str, Any], ...]
    created_at: float


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return " ".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        msg = dict(message)
        if msg.get("role") == "developer":
            msg["role"] = "system"
        msg["content"] = _flatten_content(msg.get("content"))
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    normalized_calls.append(tool_call)
                    continue
                tool_call_copy = dict(tool_call)
                function = tool_call_copy.get("function")
                if isinstance(function, dict):
                    function_copy = dict(function)
                    arguments = function_copy.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            function_copy["arguments"] = json.loads(arguments)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            function_copy["arguments"] = {}
                    tool_call_copy["function"] = function_copy
                normalized_calls.append(tool_call_copy)
            msg["tool_calls"] = normalized_calls
        normalized.append(msg)
    return normalized


def _new_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": str(name), "arguments": arguments},
    }


def _tool_call_from_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name") or payload.get("name")
        arguments = function.get("arguments", payload.get("arguments"))
    else:
        name = payload.get("name")
        arguments = payload.get("arguments")
    if not name:
        return None
    return _new_tool_call(str(name), arguments)


def _stringify_json_arguments(arguments_text: str) -> str:
    try:
        return json.dumps(json.loads(arguments_text), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        return arguments_text


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    parts: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if start < cursor:
            continue
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    remaining = "".join(parts)
    remaining = _KIMI_TOOL_CALL_SECTION_RE.sub("", remaining)
    remaining = re.sub(r"\n{3,}", "\n\n", remaining)
    return remaining.strip()


def _extract_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    stripped = text.strip()
    if not stripped:
        return "", []
    tool_calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []

    for match in _KIMI_TOOL_CALL_RE.finditer(stripped):
        tool_calls.append(_new_tool_call(match.group(1).strip(), _stringify_json_arguments(match.group(2).strip())))
        spans.append(match.span())

    for match in _QWEN_XML_TOOL_CALL_RE.finditer(stripped):
        func_name = match.group(1).strip()
        body = match.group(2).strip()
        arguments: dict[str, Any] = {}
        for param_match in re.finditer(
            r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
            body,
            re.DOTALL,
        ):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            try:
                arguments[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, TypeError, ValueError):
                arguments[param_name] = param_value
        tool_calls.append(_new_tool_call(func_name, arguments))
        spans.append(match.span())

    for match in _QWEN_JSON_TOOL_CALL_RE.finditer(stripped):
        payload_text = match.group(1).strip()
        try:
            payload = json.loads(payload_text)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        tool_call = _tool_call_from_payload(payload)
        if tool_call is None:
            continue
        tool_calls.append(tool_call)
        spans.append(match.span())

    if tool_calls:
        return _remove_spans(stripped, spans), tool_calls

    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        return stripped, []
    tool_call = _tool_call_from_payload(payload)
    if tool_call is None:
        return stripped, []
    return "", [tool_call]


def _strip_thinking(text: str) -> str:
    stripped = _THINK_RE.sub("", text)
    return stripped.replace("<think>", "").replace("</think>", "").strip()


def _postprocess_response_text(text: str, *, parse_tool_calls: bool) -> tuple[str | None, list[dict[str, Any]]]:
    if not parse_tool_calls:
        return text, []
    content, tool_calls = _extract_tool_calls(_strip_thinking(text))
    if tool_calls:
        return content or None, tool_calls
    return content, []


def _apply_chat_template_with_fallbacks(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
) -> str:
    optional_items: list[tuple[str, Any]] = []
    if tools is not None:
        optional_items.append(("tools", tools))

    current_items = list(optional_items)
    while True:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        kwargs.update({key: value for key, value in current_items})
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            if not current_items:
                raise
            current_items.pop()


class OpenClawLocalProxy:
    def __init__(
        self,
        *,
        sampling_client: LocalSamplingClientAdapter,
        tokenizer: Any,
        served_model_name: str,
        api_key: str | None = None,
        host: str = "127.0.0.1",
        port: int = 30000,
        max_completion_tokens: int = 192,
    ) -> None:
        self._sampling_client = sampling_client
        self._tokenizer = tokenizer
        self.served_model_name = served_model_name
        self.api_key = api_key
        self.host = host
        self.port = port
        self.max_completion_tokens = max_completion_tokens
        self._lock = threading.RLock()
        self._records: list[ProxyTrainingRecord] = []
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.app = self._build_app()

    def record_cursor(self) -> int:
        with self._lock:
            return len(self._records)

    def records_since(self, cursor: int) -> list[ProxyTrainingRecord]:
        with self._lock:
            return list(self._records[cursor:])

    def update_sampling_client(self, sampling_client: LocalSamplingClientAdapter) -> None:
        with self._lock:
            self._sampling_client = sampling_client

    def start(self) -> None:
        if self._thread is not None:
            return
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        self._server = server
        self._thread = threading.Thread(target=server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="OpenClaw Local Proxy")
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.get("/v1/models")
        async def models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.served_model_name,
                        "object": "model",
                        "owned_by": "mlx-tinker",
                        "created": int(time.time()),
                    }
                ],
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: str | None = Header(default=None),
            x_session_id: str | None = Header(default=None),
            x_task_id: str | None = Header(default=None),
        ):
            owner: OpenClawLocalProxy = request.app.state.owner
            body = await request.json()
            await owner._check_auth(authorization)
            result = await owner._handle_request(
                body=body,
                session_key=x_session_id or body.get("session_id"),
                task_id=x_task_id or body.get("task_id"),
            )
            if body.get("stream"):
                return StreamingResponse(owner._stream_response(result), media_type="text/event-stream")
            return JSONResponse(result["response"])

        return app

    async def _check_auth(self, authorization: str | None) -> None:
        if not self.api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        if authorization.split(" ", 1)[1].strip() != self.api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    async def _handle_request(
        self,
        *,
        body: dict[str, Any],
        session_key: str | None,
        task_id: str | None,
    ) -> dict[str, Any]:
        messages = _normalize_messages(list(body.get("messages", [])))
        tools = body.get("tools")
        parse_tool_calls = tools is not None and body.get("tool_choice") != "none"
        template_tools = tools if parse_tool_calls else None
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt_text = _apply_chat_template_with_fallbacks(
                self._tokenizer,
                messages,
                tools=template_tools,
            )
        else:
            prompt_text = "\n".join(
                f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages
            )
        try:
            prompt_tokens = list(self._tokenizer.encode(prompt_text, add_special_tokens=False))
        except TypeError:
            prompt_tokens = list(self._tokenizer.encode(prompt_text))

        with self._lock:
            sampling_client = self._sampling_client

        response = await sampling_client.sample_async(
            prompt=self._build_model_input(prompt_tokens),
            num_samples=1,
            sampling_params=self._build_sampling_params(body),
            include_prompt_logprobs=False,
            topk_prompt_logprobs=0,
        )

        sequence = response.sequences[0]
        response_tokens = list(sequence.tokens)
        response_logprobs = [float(value) for value in (sequence.logprobs or [])]
        try:
            response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)
        except TypeError:
            response_text = self._tokenizer.decode(response_tokens)
        response_content, tool_calls = _postprocess_response_text(
            response_text,
            parse_tool_calls=parse_tool_calls,
        )
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"

        record = ProxyTrainingRecord(
            trace_id=trace_id,
            task_id=task_id,
            session_key=session_key,
            messages=tuple(messages),
            prompt_text=prompt_text,
            prompt_tokens=tuple(prompt_tokens),
            response_text=response_content or "",
            response_tokens=tuple(response_tokens),
            response_logprobs=tuple(response_logprobs),
            tool_calls=tuple(tool_calls),
            created_at=time.time(),
        )
        with self._lock:
            self._records.append(record)

        assistant_message: dict[str, Any] = {"role": "assistant", "content": response_content}
        if tool_calls:
            assistant_message["tool_calls"] = list(tool_calls)

        return {
            "response": {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", self.served_model_name),
                "choices": [
                    {
                        "index": 0,
                        "message": assistant_message,
                        "finish_reason": "tool_calls" if tool_calls else (sequence.stop_reason or "stop"),
                        "logprobs": {
                            "content": [
                                {"token": "", "logprob": lp, "top_logprobs": []}
                                for lp in response_logprobs
                            ]
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": len(response_tokens),
                    "total_tokens": len(prompt_tokens) + len(response_tokens),
                },
            }
        }

    async def _stream_response(self, result: dict[str, Any]):
        response = result["response"]
        message = response["choices"][0]["message"]
        content = message.get("content")
        tool_calls = message.get("tool_calls") or []

        if tool_calls:
            for index, tool_call in enumerate(tool_calls):
                delta: dict[str, Any] = {
                    "role": "assistant" if index == 0 else None,
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                            },
                        }
                    ],
                }
                if index == 0 and content:
                    delta["content"] = content
                delta = {key: value for key, value in delta.items() if value is not None}
                chunk = {
                    "id": response["id"],
                    "object": "chat.completion.chunk",
                    "created": response["created"],
                    "model": response["model"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        else:
            chunk = {
                "id": response["id"],
                "object": "chat.completion.chunk",
                "created": response["created"],
                "model": response["model"],
                "choices": [{"index": 0, "delta": {"content": content or ""}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        final_chunk = {
            "id": response["id"],
            "object": "chat.completion.chunk",
            "created": response["created"],
            "model": response["model"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": response["choices"][0]["finish_reason"]}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    @staticmethod
    def _build_model_input(prompt_tokens: list[int]):
        import tinker

        return tinker.ModelInput.from_ints(prompt_tokens)

    def _build_sampling_params(self, body: dict[str, Any]):
        import tinker

        requested_max_tokens = int(body.get("max_tokens") or self.max_completion_tokens)
        return tinker.SamplingParams(
            temperature=float(body.get("temperature", 0.6)),
            max_tokens=min(requested_max_tokens, self.max_completion_tokens),
            top_p=float(body.get("top_p", 0.95)),
            top_k=int(body.get("top_k", 50)),
            stop=body.get("stop"),
        )
