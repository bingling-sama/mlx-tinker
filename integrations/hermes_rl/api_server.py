"""FastAPI bridge that turns Hermes agent traffic into live RL samples."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from itertools import count
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import HermesRLConfig
from .data_formatter import TrainingSample

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"

_NON_STANDARD_BODY_KEYS = {
    "session_id",
    "session_done",
    "turn_type",
    "hermes_outer_turn_id",
    "hermes_step_index",
    "hermes_request_id",
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_KIMI_TC_RE = re.compile(
    r"<\|tool_call_begin\|>\s*([a-zA-Z0-9_.-]+)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>",
    re.DOTALL,
)
_QWEN_JSON_TC_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_QWEN_XML_TC_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_KIMI_TOOL_CALL_SECTION_RE = re.compile(
    r"<\|tool_calls_section_(?:begin|end)\|>",
    re.DOTALL,
)

LogicalKey = tuple[str, int]


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(part for part in parts if part)
    return str(content) if content is not None else ""


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        normalized = dict(message)
        if normalized.get("role") == "developer":
            normalized["role"] = "system"
        content = normalized.get("content")
        if not isinstance(content, str) and content is not None:
            normalized["content"] = _flatten_content(content)
        tool_calls = normalized.get("tool_calls")
        if isinstance(tool_calls, list):
            parsed_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    parsed_calls.append(tool_call)
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
                parsed_calls.append(tool_call_copy)
            normalized["tool_calls"] = parsed_calls
        out.append(normalized)
    return out


def _extract_logprobs(choice: dict[str, Any]) -> list[float]:
    lp_obj = choice.get("logprobs")
    if not isinstance(lp_obj, dict):
        return []
    content = lp_obj.get("content")
    if not isinstance(content, list):
        return []
    return [float(item.get("logprob", 0.0)) for item in content if isinstance(item, dict)]


def _resolve_enable_thinking(body: dict[str, Any]) -> bool:
    enable_thinking = True
    extra_body = body.get("extra_body")
    if isinstance(extra_body, dict):
        chat_template_kwargs = extra_body.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, dict) and "enable_thinking" in chat_template_kwargs:
            return bool(chat_template_kwargs["enable_thinking"])
        if "enable_thinking" in extra_body:
            return bool(extra_body["enable_thinking"])
    chat_template_kwargs = body.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict) and "enable_thinking" in chat_template_kwargs:
        return bool(chat_template_kwargs["enable_thinking"])
    if "enable_thinking" in body:
        return bool(body["enable_thinking"])
    return enable_thinking


def _apply_chat_template_with_fallbacks(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    tools=None,
    tool_choice=None,
) -> str:
    optional_items: list[tuple[str, Any]] = [("enable_thinking", enable_thinking)]
    if tools:
        optional_items.append(("tools", tools))
    if tool_choice is not None and tools:
        optional_items.append(("tool_choice", tool_choice))

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


def _strip_thinking(text: str) -> str:
    stripped = _THINK_RE.sub("", text)
    return stripped.replace("<think>", "").replace("</think>", "").strip()


def _split_thinking(text: str) -> tuple[str | None, str]:
    stripped = text.lstrip()
    full_block_match = re.match(r"^<think>\s*(.*?)\s*</think>\s*(.*)$", stripped, flags=re.DOTALL)
    if full_block_match:
        reasoning = full_block_match.group(1).strip() or None
        return reasoning, full_block_match.group(2).strip()

    truncated_block_match = re.match(r"^<think>\s*(.*)$", stripped, flags=re.DOTALL)
    if truncated_block_match:
        reasoning = truncated_block_match.group(1).strip() or None
        return reasoning, ""

    orphan_close_match = re.match(r"^(.*?)</think>\s*(.*)$", stripped, flags=re.DOTALL)
    if orphan_close_match:
        reasoning = orphan_close_match.group(1).replace("<think>", "").strip() or None
        return reasoning, orphan_close_match.group(2).strip()

    return None, text


def _extract_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    if not text:
        return "", []

    stripped = _strip_thinking(text)
    tool_calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []

    for index, match in enumerate(_KIMI_TC_RE.finditer(stripped)):
        tool_calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": (match.group(1) or "").strip() or "unknown_tool",
                    "arguments": _stringify_json_arguments((match.group(2) or "{}").strip()),
                },
            }
        )
        spans.append(match.span())

    for index, match in enumerate(_QWEN_XML_TC_RE.finditer(stripped), start=len(tool_calls)):
        func_name = (match.group(1) or "").strip() or "unknown_tool"
        body = (match.group(2) or "").strip()
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
        tool_calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
        spans.append(match.span())

    for index, match in enumerate(_QWEN_JSON_TC_RE.finditer(stripped), start=len(tool_calls)):
        try:
            payload = json.loads(match.group(1).strip())
        except Exception:
            continue
        name = payload.get("name") or payload.get("function", {}).get("name") or "unknown_tool"
        arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        tool_calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": str(name), "arguments": arguments},
            }
        )
        spans.append(match.span())

    if tool_calls:
        return _remove_spans(stripped, spans), tool_calls
    return stripped, []


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_step_index(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class _BaseServer:
    """Shared infrastructure for Hermes RL / OPD / combined proxy servers."""

    _TITLE = "Hermes Tinker Proxy"
    _SCORE_FILE = "scores.jsonl"

    def __init__(
        self,
        config: HermesRLConfig,
        output_queue: queue.Queue,
        submission_enabled: threading.Event,
        *,
        sampling_client=None,
        tokenizer=None,
    ) -> None:
        self.config = config
        self.output_queue = output_queue
        self.submission_enabled = submission_enabled
        self._sampling_client = sampling_client

        self._index_counter = count(0)
        self._group_counter = count(0)
        self._turn_counters: dict[str, int] = {}
        self._logical_turn_nums: dict[str, dict[LogicalKey, int]] = {}
        self._last_main_key: dict[str, LogicalKey] = {}
        self._seen_request_ids: dict[str, set[str]] = {}
        self._pending_turn_data: dict[str, dict[LogicalKey, dict[str, Any]]] = {}
        self._score_tasks: dict[str, dict[LogicalKey, asyncio.Task]] = {}
        self._pending_records: dict[str, dict[LogicalKey, dict[str, Any]]] = {}

        self._eval_scores: list[float] = []
        self._eval_scores_lock = threading.Lock()

        os.makedirs(config.record_dir, exist_ok=True)
        self._record_file = os.path.join(config.record_dir, "conversations.jsonl")
        self._prm_record_file = os.path.join(config.record_dir, self._SCORE_FILE)
        open(self._record_file, "w").close()
        open(self._prm_record_file, "w").close()

        self._tokenizer = tokenizer or self._load_tokenizer()
        self.app = self._build_app()
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        except Exception as exc:
            logger.error("[Server] failed to load tokenizer: %s", exc, exc_info=True)
            return None

    def _build_app(self) -> FastAPI:
        app = FastAPI(title=self._TITLE)
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_session_id: Optional[str] = Header(default=None),
            x_turn_type: Optional[str] = Header(default=None),
            x_session_done: Optional[str] = Header(default=None),
            x_hermes_outer_turn_id: Optional[str] = Header(default=None),
            x_hermes_step_index: Optional[str] = Header(default=None),
            x_hermes_request_id: Optional[str] = Header(default=None),
        ):
            owner: _BaseServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.submission_enabled.is_set():
                resumed = await asyncio.to_thread(owner.submission_enabled.wait, 300.0)
                if not resumed:
                    raise HTTPException(status_code=503, detail="submission paused (timeout)")

            body = await request.json()
            session_id = x_session_id or body.get("session_id") or "unknown"
            turn_type = (x_turn_type or body.get("turn_type") or "ignore").strip().lower()
            session_done = _truthy(x_session_done) or _truthy(body.get("session_done"))
            outer_turn_id = x_hermes_outer_turn_id or body.get("hermes_outer_turn_id") or "0"
            step_index = _coerce_step_index(
                x_hermes_step_index or body.get("hermes_step_index") or 0
            )
            request_id = (
                x_hermes_request_id
                or body.get("hermes_request_id")
                or f"hermes_proxy_{uuid.uuid4().hex}"
            )

            stream = bool(body.get("stream", False))
            result = await owner._handle_request(
                body,
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
                outer_turn_id=str(outer_turn_id),
                step_index=step_index,
                request_id=request_id,
            )
            if stream:
                return StreamingResponse(
                    owner._stream_response(result),
                    media_type="text/event-stream",
                )
            return JSONResponse(content=result["response"])

        return app

    async def _check_auth(self, authorization: Optional[str]) -> None:
        if not self.config.proxy_api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        if authorization.split(" ", 1)[1].strip() != self.config.proxy_api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    async def _forward_to_tinker(self, body: dict[str, Any]) -> dict[str, Any]:
        import tinker

        if self._sampling_client is None:
            raise HTTPException(status_code=503, detail="no sampling client available")
        if self._tokenizer is None:
            raise HTTPException(status_code=503, detail="no tokenizer available")

        messages = body.get("messages", [])
        normalized_messages = _normalize_messages(messages)
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        parse_tool_calls = tools is not None and tool_choice != "none"
        enable_thinking = _resolve_enable_thinking(body)
        temperature = float(body.get("temperature", 0.6))
        max_tokens = int(body.get("max_tokens") or 2048)
        stop = body.get("stop")

        prompt_text = _apply_chat_template_with_fallbacks(
            self._tokenizer,
            normalized_messages,
            enable_thinking=enable_thinking,
            tools=tools if parse_tool_calls else None,
            tool_choice=tool_choice if parse_tool_calls else None,
        )
        prompt_ids = self._tokenizer.encode(prompt_text, add_special_tokens=False)
        chunk = tinker.EncodedTextChunk(tokens=list(prompt_ids), type="encoded_text")
        model_input = tinker.ModelInput(chunks=[chunk])

        sampling_kwargs = dict(temperature=temperature, max_tokens=max_tokens, top_k=50, top_p=0.95)
        if stop is not None:
            sampling_kwargs["stop"] = stop
        sampling_params = tinker.SamplingParams(**sampling_kwargs)
        response = await self._sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
            include_prompt_logprobs=False,
            topk_prompt_logprobs=0,
        )

        sequence = response.sequences[0]
        raw_response_tokens = list(sequence.tokens)
        raw_response_logprobs = [float(lp) for lp in (sequence.logprobs or [])]
        response_text = self._tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        reasoning_content, visible_text = _split_thinking(response_text)
        if not enable_thinking:
            reasoning_content = None
        if parse_tool_calls:
            normalized_text, parsed_tool_calls = _extract_tool_calls(visible_text)
        else:
            normalized_text, parsed_tool_calls = visible_text, []

        logprob_content = [
            {"token": "", "logprob": lp, "top_logprobs": []} for lp in raw_response_logprobs
        ]
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": normalized_text if (normalized_text or not parsed_tool_calls) else None,
        }
        if reasoning_content is not None:
            assistant_message["reasoning_content"] = reasoning_content
        if parsed_tool_calls:
            assistant_message["tool_calls"] = parsed_tool_calls

        return {
            "id": f"chatcmpl-hermes-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", self.config.served_model_name),
            "choices": [
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": (
                        "tool_calls"
                        if parsed_tool_calls
                        else (sequence.stop_reason or "stop")
                    ),
                    "logprobs": {"content": logprob_content},
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(raw_response_tokens),
                "total_tokens": len(prompt_ids) + len(raw_response_tokens),
            },
            "_raw_prompt_ids": list(prompt_ids),
            "_raw_response_tokens": raw_response_tokens,
            "_raw_response_logprobs": raw_response_logprobs,
        }

    def _logical_key(self, outer_turn_id: str, step_index: int) -> LogicalKey:
        return (str(outer_turn_id), int(step_index))

    def _turn_num_for(self, session_id: str, logical_key: LogicalKey) -> int:
        session_turns = self._logical_turn_nums.setdefault(session_id, {})
        existing = session_turns.get(logical_key)
        if existing is not None:
            return existing
        next_turn_num = self._turn_counters.get(session_id, 0) + 1
        self._turn_counters[session_id] = next_turn_num
        session_turns[logical_key] = next_turn_num
        return next_turn_num

    def _buffer_record(
        self,
        session_id: str,
        logical_key: LogicalKey,
        record: dict[str, Any],
    ) -> None:
        self._pending_records.setdefault(session_id, {})[logical_key] = record

    def _flush_pending_record(
        self,
        session_id: str,
        logical_key: LogicalKey,
        next_state: dict[str, Any] | None,
    ) -> None:
        session_records = self._pending_records.get(session_id)
        if not session_records:
            return
        record = session_records.pop(logical_key, None)
        if record is None:
            return
        record["next_state"] = next_state
        try:
            with open(self._record_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
        if not session_records:
            self._pending_records.pop(session_id, None)

    def _append_score_record(self, record: dict[str, Any]) -> None:
        try:
            with open(self._prm_record_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def purge_record_files(self) -> None:
        for path in (self._record_file, self._prm_record_file):
            try:
                open(path, "w").close()
            except OSError:
                pass

    def drain_eval_scores(self) -> list[float]:
        with self._eval_scores_lock:
            scores = list(self._eval_scores)
            self._eval_scores.clear()
            return scores

    def reset_eval_scores(self) -> None:
        with self._eval_scores_lock:
            self._eval_scores.clear()

    def _on_next_state(
        self,
        session_id: str,
        logical_key: LogicalKey,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def _maybe_submit_ready_samples(self, session_id: str, **kwargs) -> None:
        raise NotImplementedError

    def _transition_previous_turn(
        self,
        session_id: str,
        current_key: LogicalKey,
        next_state: dict[str, Any],
    ) -> None:
        previous_key = self._last_main_key.get(session_id)
        self._last_main_key[session_id] = current_key
        if previous_key is None or previous_key == current_key:
            return
        previous_turn = self._pending_turn_data.get(session_id, {}).get(previous_key)
        if previous_turn is None:
            return

        self._flush_pending_record(session_id, previous_key, next_state)
        ns_text = _flatten_content(next_state.get("content"))
        ns_role = next_state.get("role", "user")
        logger.info(
            "%s[Server] session=%s turn=%d next_state role=%s len=%d%s",
            _GREEN,
            session_id,
            previous_turn["turn_num"],
            ns_role,
            len(ns_text),
            _RESET,
        )
        self._on_next_state(session_id, previous_key, previous_turn, next_state)

    def _tokenize_turn(self, messages, assistant_msg, tools, choice, output=None):
        normalized_messages = _normalize_messages(messages)
        prompt_text = self._tokenizer.apply_chat_template(
            normalized_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

        if output is not None and "_raw_response_tokens" in output:
            prompt_ids = output["_raw_prompt_ids"]
            response_ids = output["_raw_response_tokens"]
            response_logprobs = output["_raw_response_logprobs"]

            if len(response_logprobs) != len(response_ids):
                logger.error(
                    "[Server] raw logprobs len=%d != raw tokens len=%d; repairing",
                    len(response_logprobs),
                    len(response_ids),
                )
                if len(response_logprobs) > len(response_ids):
                    response_logprobs = response_logprobs[: len(response_ids)]
                else:
                    response_logprobs = response_logprobs + [0.0] * (
                        len(response_ids) - len(response_logprobs)
                    )

            response_text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
            return prompt_ids, response_ids, response_logprobs, prompt_text, response_text

        response_message = dict(assistant_msg)
        if response_message.get("content") is None:
            response_message["content"] = ""
        normalized_response = _normalize_messages([response_message])[0]
        full_text = self._tokenizer.apply_chat_template(
            normalized_messages + [normalized_response],
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        response_text = (
            full_text[len(prompt_text) :]
            if full_text.startswith(prompt_text)
            else full_text
        )
        prompt_ids = self._tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self._tokenizer(response_text, add_special_tokens=False)["input_ids"]
        response_logprobs = _extract_logprobs(choice)
        if len(response_logprobs) > len(response_ids):
            response_logprobs = response_logprobs[: len(response_ids)]
        elif len(response_logprobs) < len(response_ids):
            response_logprobs += [0.0] * (len(response_ids) - len(response_logprobs))
        return prompt_ids, response_ids, response_logprobs, prompt_text, response_text

    async def _handle_request(
        self,
        body: dict[str, Any],
        *,
        session_id: str,
        turn_type: str,
        session_done: bool,
        outer_turn_id: str,
        step_index: int,
        request_id: str,
    ) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        tools = body.get("tools")
        logical_key = self._logical_key(outer_turn_id, step_index)
        duplicate = request_id in self._seen_request_ids.setdefault(session_id, set())

        forward_body = {
            key: value
            for key, value in body.items()
            if key not in _NON_STANDARD_BODY_KEYS
        }
        forward_body["stream"] = False
        forward_body.pop("stream_options", None)
        forward_body["logprobs"] = True
        forward_body["top_logprobs"] = 1
        if "model" not in forward_body:
            forward_body["model"] = self.config.served_model_name

        output = await self._forward_to_tinker(forward_body)
        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        tool_calls = assistant_msg.get("tool_calls") or []
        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or ""

        logger.info(
            "%s[Server] [%s] session=%s outer=%s step=%d dup=%s prompt_msgs=%d%s",
            _YELLOW,
            turn_type,
            session_id,
            outer_turn_id,
            step_index,
            duplicate,
            len(messages),
            _RESET,
        )
        logger.info(
            "%s[Server] [%s] session=%s thinking=%d response:\n%s%s",
            _RED,
            turn_type,
            session_id,
            len(reasoning),
            content[:500],
            _RESET,
        )

        if turn_type == "main":
            if not duplicate:
                self._seen_request_ids[session_id].add(request_id)
                self._transition_previous_turn(session_id, logical_key, messages[-1])

            (
                prompt_ids,
                response_ids,
                response_logprobs,
                prompt_text,
                response_text,
            ) = self._tokenize_turn(
                messages,
                assistant_msg,
                tools,
                choice,
                output=output,
            )

            if response_ids or response_text.strip():
                if duplicate:
                    logger.info(
                        (
                            "[Server] duplicate request ignored for training "
                            "session=%s outer=%s step=%d request=%s"
                        ),
                        session_id,
                        outer_turn_id,
                        step_index,
                        request_id,
                    )
                else:
                    turn_num = self._turn_num_for(session_id, logical_key)
                    turn_data = {
                        "turn_num": turn_num,
                        "outer_turn_id": outer_turn_id,
                        "step_index": step_index,
                        "request_id": request_id,
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                        "response_logprobs": response_logprobs,
                        "prompt_text": prompt_text,
                        "response_text": response_text,
                        "messages": messages,
                        "tools": tools,
                        "has_next_state": False,
                    }
                    self._pending_turn_data.setdefault(session_id, {})[logical_key] = turn_data
                    self._buffer_record(
                        session_id,
                        logical_key,
                        {
                            "session_id": session_id,
                            "turn": turn_num,
                            "outer_turn_id": outer_turn_id,
                            "step_index": step_index,
                            "request_id": request_id,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "prompt_text": prompt_text,
                            "response_text": response_text,
                            "tool_calls": tool_calls or None,
                        },
                    )
                    logger.info(
                        "[Server] MAIN session=%s turn=%d outer=%s step=%d prompt=%d response=%d",
                        session_id,
                        turn_num,
                        outer_turn_id,
                        step_index,
                        len(prompt_ids),
                        len(response_ids),
                    )
                    self._maybe_submit_ready_samples(session_id)
            output["session_id"] = session_id
        else:
            logger.info("[Server] non-main session=%s -> skipped", session_id)

        if session_done:
            for pending_key in list(self._pending_turn_data.get(session_id, {}).keys()):
                self._flush_pending_record(session_id, pending_key, None)
            self._maybe_submit_ready_samples(session_id, force_finalize=True)
            self._last_main_key.pop(session_id, None)
            self._seen_request_ids.pop(session_id, None)

        output["session_id"] = session_id
        return {"response": output}

    async def _stream_response(self, result: dict[str, Any]):
        payload = result["response"]
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        base = {
            "id": payload.get("id", ""),
            "object": "chat.completion.chunk",
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", ""),
            "session_id": payload.get("session_id", ""),
        }
        tool_calls = message.get("tool_calls") or []
        content = message.get("content")
        if tool_calls:
            for index, tool_call in enumerate(tool_calls):
                delta: dict[str, Any] = {"role": "assistant"} if index == 0 else {}
                if index == 0 and content:
                    delta["content"] = content
                delta["tool_calls"] = [
                    {
                        "index": index,
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                        },
                    }
                ]
                yield (
                    "data: "
                    + json.dumps(
                        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
        else:
            delta = {"role": "assistant", "content": content or ""}
            yield (
                "data: "
                + json.dumps(
                    {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        yield (
            "data: "
            + json.dumps(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": choice.get("finish_reason", "stop"),
                        }
                    ],
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
        yield "data: [DONE]\n\n"

    def update_sampling_client(self, client) -> None:
        self._sampling_client = client
        logger.info("[Server] sampling client updated")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        cfg = uvicorn.Config(
            self.app,
            host=self.config.proxy_host,
            port=self.config.proxy_port,
            log_level="info",
        )
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        threading.Thread(target=self._print_ready, daemon=True).start()

    def _print_ready(self) -> None:
        time.sleep(3)
        banner = (
            f"\n{'=' * 60}\n"
            f"  {self._TITLE} ready\n"
            f"  {self.config.proxy_host}:{self.config.proxy_port}"
            f" -> {self.config.tinker_base_url}\n"
            f"  Method: {self.config.method}\n"
            f"  PRM/Teacher: mlx-tinker SamplingClient (m={self.config.prm_m})\n"
            f"{'=' * 60}\n"
        )
        logger.info("%s%s%s", _GREEN, banner, _RESET)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _safe_create_task(self, coro) -> None:
        task = asyncio.create_task(coro)
        task.add_done_callback(self._task_done_cb)

    @staticmethod
    def _task_done_cb(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[Server] background task failed: %s", exc, exc_info=exc)


class HermesRLServer(_BaseServer):
    """Binary PRM-scored live RL for Hermes sessions."""

    _TITLE = "Hermes-RL Tinker Proxy"
    _SCORE_FILE = "prm_scores.jsonl"

    def __init__(self, *args, prm_scorer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prm_scorer = prm_scorer
        self._session_effective: dict[str, int] = {}

    def _on_next_state(
        self,
        session_id: str,
        logical_key: LogicalKey,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
    ) -> None:
        if not self.prm_scorer:
            return
        ns_text = _flatten_content(next_state.get("content"))
        ns_role = next_state.get("role", "user")
        task = asyncio.create_task(
            self.prm_scorer.evaluate(
                turn_data["response_text"],
                ns_text,
                ns_role,
                session_id,
                turn_data["turn_num"],
            )
        )
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _task: self._maybe_submit_ready_samples(session_id))
        self._score_tasks.setdefault(session_id, {})[logical_key] = task
        turn_data["has_next_state"] = True

    def _maybe_submit_ready_samples(self, session_id: str, force_finalize: bool = False) -> None:
        pending = self._pending_turn_data.get(session_id, {})
        score_tasks = self._score_tasks.get(session_id, {})
        ordered_keys = sorted(pending.keys(), key=lambda key: pending[key]["turn_num"])
        for logical_key in ordered_keys:
            turn_data = pending[logical_key]
            task = score_tasks.get(logical_key)
            if self.prm_scorer:
                if task is not None and not task.done():
                    continue
                if task is None and not force_finalize:
                    continue

            pending.pop(logical_key, None)
            prm_result = None
            if task is not None and task.done():
                try:
                    prm_result = task.result()
                except Exception:
                    prm_result = None
                score_tasks.pop(logical_key, None)
            self._safe_create_task(self._submit_turn_sample(session_id, turn_data, prm_result))

    async def _submit_turn_sample(
        self,
        session_id: str,
        turn_data: dict[str, Any],
        prm_result: dict[str, Any] | None,
    ) -> None:
        score = prm_result["score"] if prm_result else 0.0
        with self._eval_scores_lock:
            self._eval_scores.append(score)

        response_ids = turn_data["response_ids"]
        exclude = not turn_data.get("has_next_state", False) or score == 0.0
        if (
            exclude
            and turn_data.get("has_next_state", False)
            and self._session_effective.get(session_id, 0) == 0
        ):
            exclude = False
            logger.info(
                "[Server] promoting session=%s neutral score to keep one effective sample",
                session_id,
            )

        loss_mask = [0] * len(response_ids) if exclude else [1] * len(response_ids)
        sample = TrainingSample(
            session_id=session_id,
            turn_num=turn_data["turn_num"],
            prompt_tokens=turn_data["prompt_ids"],
            response_tokens=response_ids,
            response_logprobs=turn_data["response_logprobs"],
            loss_mask=loss_mask,
            reward=score,
            prompt_text=turn_data.get("prompt_text", ""),
            response_text=turn_data.get("response_text", ""),
        )

        if not exclude:
            self._session_effective[session_id] = self._session_effective.get(session_id, 0) + 1

        if prm_result:
            self._append_score_record(
                {
                    "session_id": session_id,
                    "turn": turn_data["turn_num"],
                    "outer_turn_id": turn_data["outer_turn_id"],
                    "step_index": turn_data["step_index"],
                    "score": score,
                    "votes": prm_result.get("votes", []),
                    "representative": prm_result.get("representative", ""),
                }
            )

        index = next(self._index_counter)
        group_index = next(self._group_counter)
        logger.info(
            "[Server] submitted session=%s idx=%d turn=%d score=%.1f exclude=%s",
            session_id,
            index,
            turn_data["turn_num"],
            score,
            exclude,
        )
        await asyncio.to_thread(self.output_queue.put, (group_index, [sample]))


class HermesOPDServer(_BaseServer):
    """Live OPD for Hermes sessions using hindsight hints and teacher logprobs."""

    _TITLE = "Hermes-OPD Tinker Proxy"
    _SCORE_FILE = "opd_scores.jsonl"

    def __init__(self, *args, opd_scorer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.opd_scorer = opd_scorer

    def _on_next_state(
        self,
        session_id: str,
        logical_key: LogicalKey,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
    ) -> None:
        if not self.opd_scorer:
            return
        task = asyncio.create_task(
            self.opd_scorer.evaluate(
                response_text=turn_data["response_text"],
                next_state_text=_flatten_content(next_state.get("content")),
                next_state_role=next_state.get("role", "user"),
                turn_data=turn_data,
                tokenizer=self._tokenizer,
                normalize_fn=_normalize_messages,
                session_id=session_id,
                turn_num=turn_data["turn_num"],
            )
        )
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _task: self._maybe_submit_ready_samples(session_id))
        self._score_tasks.setdefault(session_id, {})[logical_key] = task
        turn_data["has_next_state"] = True

    def _maybe_submit_ready_samples(self, session_id: str, force_finalize: bool = False) -> None:
        pending = self._pending_turn_data.get(session_id, {})
        score_tasks = self._score_tasks.get(session_id, {})
        ordered_keys = sorted(pending.keys(), key=lambda key: pending[key]["turn_num"])
        for logical_key in ordered_keys:
            turn_data = pending[logical_key]
            task = score_tasks.get(logical_key)
            if task is None:
                if force_finalize:
                    pending.pop(logical_key, None)
                    if self.config.eval_mode:
                        with self._eval_scores_lock:
                            self._eval_scores.append(0.0)
                continue
            if not task.done():
                continue

            pending.pop(logical_key, None)
            score_tasks.pop(logical_key, None)
            try:
                result = task.result()
            except Exception as exc:
                logger.error(
                    "[Server] OPD task failed session=%s turn=%d: %s",
                    session_id,
                    turn_data["turn_num"],
                    exc,
                    exc_info=True,
                )
                if self.config.eval_mode:
                    with self._eval_scores_lock:
                        self._eval_scores.append(0.0)
                continue

            if self.config.eval_mode:
                eval_score = result.get("eval_score")
                if eval_score is not None:
                    with self._eval_scores_lock:
                        self._eval_scores.append(eval_score)

            if not result.get("accepted"):
                self._append_score_record(
                    {
                        "session_id": session_id,
                        "turn": turn_data["turn_num"],
                        "outer_turn_id": turn_data["outer_turn_id"],
                        "step_index": turn_data["step_index"],
                        "accepted": False,
                        "hint": "",
                        "hint_raw": result.get("hint_raw", ""),
                        "eval_raw": result.get("eval_raw", ""),
                    }
                )
                continue

            self._append_score_record(
                {
                    "session_id": session_id,
                    "turn": turn_data["turn_num"],
                    "outer_turn_id": turn_data["outer_turn_id"],
                    "step_index": turn_data["step_index"],
                    "accepted": True,
                    "hint": result.get("hint", ""),
                    "hint_len": len(result.get("hint", "")),
                    "hint_raw": result.get("hint_raw", ""),
                    "eval_raw": result.get("eval_raw", ""),
                }
            )
            self._safe_create_task(self._submit_turn_sample(session_id, turn_data, result))

    async def _submit_turn_sample(
        self,
        session_id: str,
        turn_data: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        response_ids = turn_data["response_ids"]
        teacher_lps = result.get("teacher_log_probs") or []
        if len(teacher_lps) > len(response_ids):
            teacher_lps = teacher_lps[: len(response_ids)]
        elif len(teacher_lps) < len(response_ids):
            teacher_lps = teacher_lps + [0.0] * (len(response_ids) - len(teacher_lps))

        sample = TrainingSample(
            session_id=session_id,
            turn_num=turn_data["turn_num"],
            prompt_tokens=turn_data["prompt_ids"],
            response_tokens=response_ids,
            response_logprobs=turn_data["response_logprobs"],
            loss_mask=[1] * len(response_ids),
            reward=1.0,
            prompt_text=turn_data.get("prompt_text", ""),
            response_text=turn_data.get("response_text", ""),
            teacher_logprobs=teacher_lps,
        )

        index = next(self._index_counter)
        group_index = next(self._group_counter)
        logger.info(
            "[Server] submitted OPD session=%s idx=%d turn=%d hint_len=%d",
            session_id,
            index,
            turn_data["turn_num"],
            len(result.get("hint", "")),
        )
        await asyncio.to_thread(self.output_queue.put, (group_index, [sample]))


class HermesCombineServer(_BaseServer):
    """Combined OPD + PRM training for Hermes sessions."""

    _TITLE = "Hermes-Combine Tinker Proxy"
    _SCORE_FILE = "combine_scores.jsonl"

    def __init__(self, *args, scorer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scorer = scorer

    def _print_ready(self) -> None:
        time.sleep(3)
        banner = (
            f"\n{'=' * 60}\n"
            f"  {self._TITLE} ready\n"
            f"  {self.config.proxy_host}:{self.config.proxy_port}"
            f" -> {self.config.tinker_base_url}\n"
            f"  Method: combine\n"
            f"  PRM/Teacher: mlx-tinker SamplingClient (m={self.config.prm_m})\n"
            f"  Weights: w_opd={self.config.w_opd} w_rl={self.config.w_rl}\n"
            f"{'=' * 60}\n"
        )
        logger.info("%s%s%s", _GREEN, banner, _RESET)

    def _on_next_state(
        self,
        session_id: str,
        logical_key: LogicalKey,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
    ) -> None:
        if not self.scorer:
            return
        task = asyncio.create_task(
            self.scorer.evaluate(
                response_text=turn_data["response_text"],
                next_state_text=_flatten_content(next_state.get("content")),
                next_state_role=next_state.get("role", "user"),
                turn_data=turn_data,
                tokenizer=self._tokenizer,
                normalize_fn=_normalize_messages,
                session_id=session_id,
                turn_num=turn_data["turn_num"],
            )
        )
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _task: self._maybe_submit_ready_samples(session_id))
        self._score_tasks.setdefault(session_id, {})[logical_key] = task
        turn_data["has_next_state"] = True

    @staticmethod
    def _is_valid_rl_score(score) -> bool:
        return score in (1, -1, 1.0, -1.0)

    def _maybe_submit_ready_samples(self, session_id: str, force_finalize: bool = False) -> None:
        pending = self._pending_turn_data.get(session_id, {})
        score_tasks = self._score_tasks.get(session_id, {})
        ordered_keys = sorted(pending.keys(), key=lambda key: pending[key]["turn_num"])
        for logical_key in ordered_keys:
            turn_data = pending[logical_key]
            task = score_tasks.get(logical_key)
            if task is None:
                if force_finalize:
                    pending.pop(logical_key, None)
                    with self._eval_scores_lock:
                        self._eval_scores.append(0.0)
                continue
            if not task.done():
                continue

            pending.pop(logical_key, None)
            score_tasks.pop(logical_key, None)
            try:
                result = task.result()
            except Exception as exc:
                logger.error(
                    "[Server] combined evaluation failed session=%s turn=%d: %s",
                    session_id,
                    turn_data["turn_num"],
                    exc,
                    exc_info=True,
                )
                with self._eval_scores_lock:
                    self._eval_scores.append(0.0)
                continue

            eval_score = result.get("eval_score")
            if eval_score is not None:
                with self._eval_scores_lock:
                    self._eval_scores.append(eval_score)

            opd_accepted = result.get("accepted")
            has_valid_rl = self._is_valid_rl_score(eval_score)
            hint = result.get("hint", "")

            if opd_accepted and has_valid_rl:
                self._append_score_record(
                    {
                        "session_id": session_id,
                        "turn": turn_data["turn_num"],
                        "outer_turn_id": turn_data["outer_turn_id"],
                        "step_index": turn_data["step_index"],
                        "type": "opd+rl",
                        "eval_score": eval_score,
                        "hint": hint,
                        "hint_len": len(hint),
                        "hint_raw": result.get("hint_raw", ""),
                        "eval_raw": result.get("eval_raw", ""),
                    }
                )
                self._safe_create_task(
                    self._submit_opd_sample(session_id, turn_data, result, reward=float(eval_score))
                )
            elif opd_accepted:
                self._append_score_record(
                    {
                        "session_id": session_id,
                        "turn": turn_data["turn_num"],
                        "outer_turn_id": turn_data["outer_turn_id"],
                        "step_index": turn_data["step_index"],
                        "type": "opd",
                        "eval_score": eval_score,
                        "hint": hint,
                        "hint_len": len(hint),
                        "hint_raw": result.get("hint_raw", ""),
                        "eval_raw": result.get("eval_raw", ""),
                    }
                )
                self._safe_create_task(
                    self._submit_opd_sample(session_id, turn_data, result, reward=0.0)
                )
            elif has_valid_rl:
                self._append_score_record(
                    {
                        "session_id": session_id,
                        "turn": turn_data["turn_num"],
                        "outer_turn_id": turn_data["outer_turn_id"],
                        "step_index": turn_data["step_index"],
                        "type": "rl",
                        "eval_score": eval_score,
                        "eval_raw": result.get("eval_raw", ""),
                    }
                )
                self._safe_create_task(
                    self._submit_rl_sample(session_id, turn_data, float(eval_score))
                )
            else:
                logger.info(
                    "[Server] no signal session=%s turn=%d",
                    session_id,
                    turn_data["turn_num"],
                )

    async def _submit_opd_sample(
        self,
        session_id: str,
        turn_data: dict[str, Any],
        result: dict[str, Any],
        *,
        reward: float,
    ) -> None:
        response_ids = turn_data["response_ids"]
        teacher_lps = result.get("teacher_log_probs") or []
        if len(teacher_lps) > len(response_ids):
            teacher_lps = teacher_lps[: len(response_ids)]
        elif len(teacher_lps) < len(response_ids):
            teacher_lps = teacher_lps + [0.0] * (len(response_ids) - len(teacher_lps))

        sample_type = "opd+rl" if reward != 0.0 else "opd"
        sample = TrainingSample(
            session_id=session_id,
            turn_num=turn_data["turn_num"],
            prompt_tokens=turn_data["prompt_ids"],
            response_tokens=response_ids,
            response_logprobs=turn_data["response_logprobs"],
            loss_mask=[1] * len(response_ids),
            reward=reward,
            prompt_text=turn_data.get("prompt_text", ""),
            response_text=turn_data.get("response_text", ""),
            teacher_logprobs=teacher_lps,
            sample_type=sample_type,
        )

        index = next(self._index_counter)
        group_index = next(self._group_counter)
        logger.info(
            "[Server] submitted %s session=%s idx=%d turn=%d reward=%.1f",
            sample_type,
            session_id,
            index,
            turn_data["turn_num"],
            reward,
        )
        await asyncio.to_thread(self.output_queue.put, (group_index, [sample]))

    async def _submit_rl_sample(
        self,
        session_id: str,
        turn_data: dict[str, Any],
        eval_score: float,
    ) -> None:
        response_ids = turn_data["response_ids"]
        response_logprobs = turn_data["response_logprobs"]
        if len(response_logprobs) > len(response_ids):
            response_logprobs = response_logprobs[: len(response_ids)]
        elif len(response_logprobs) < len(response_ids):
            response_logprobs = response_logprobs + [0.0] * (
                len(response_ids) - len(response_logprobs)
            )

        sample = TrainingSample(
            session_id=session_id,
            turn_num=turn_data["turn_num"],
            prompt_tokens=turn_data["prompt_ids"],
            response_tokens=response_ids,
            response_logprobs=response_logprobs,
            loss_mask=[1] * len(response_ids),
            reward=eval_score,
            prompt_text=turn_data.get("prompt_text", ""),
            response_text=turn_data.get("response_text", ""),
            teacher_logprobs=list(response_logprobs),
            sample_type="rl",
        )

        index = next(self._index_counter)
        group_index = next(self._group_counter)
        logger.info(
            "[Server] submitted RL session=%s idx=%d turn=%d score=%.1f",
            session_id,
            index,
            turn_data["turn_num"],
            eval_score,
        )
        await asyncio.to_thread(self.output_queue.put, (group_index, [sample]))


OpenClawRLServer = HermesRLServer
OpenClawOPDServer = HermesOPDServer
OpenClawCombineServer = HermesCombineServer
