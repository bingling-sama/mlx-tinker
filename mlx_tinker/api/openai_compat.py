"""OpenAI-compatible endpoints: /v1/chat/completions, /v1/completions."""

from __future__ import annotations

import json
import logging
import re
import inspect
import time
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_tinker.api.lora_catalog import LoraCatalogService
from mlx_tinker.backend.mlx_backend import MLXBackend

logger = logging.getLogger(__name__)

router = APIRouter()

# Backend reference — set by register_openai_routes()
_backend: MLXBackend | None = None

# ---------------------------------------------------------------------------
# Pydantic models (extra="allow" for forward-compat with new OpenAI fields)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    model_config = {"extra": "allow"}
    role: str
    content: str | list | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str = "default"
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = 4096
    max_completion_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    response_format: dict | None = None


class CompletionRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str = "default"
    prompt: str | list[str]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    stop: list[str] | str | None = None


def register_openai_routes(app, backend: MLXBackend) -> None:
    """Register OpenAI-compatible routes on the FastAPI app."""
    global _backend
    _backend = backend
    app.include_router(router)


def _get_model_and_tokenizer_and_namespace(model_name: str | None = None):
    """Resolve model by name. Supports:

    - None / base model name → base model (default)
    - "base_model:checkpoint_name" → LoRA from sampler checkpoint
    - Tinker model_id → in-memory training model
    """
    if _backend is None:
        raise HTTPException(status_code=503, detail="Backend not initialized")

    if model_name and ":" in model_name:
        # LoRA checkpoint pattern: "Qwen/Qwen3.5-4B:openclaw_local_step_0001"
        base_part, checkpoint_name = model_name.rsplit(":", 1)
        checkpoint_path = str(_backend.config.checkpoints_base / checkpoint_name)
        try:
            resolved_path = _backend._validate_checkpoint_path(checkpoint_path)
            namespace = _backend._path_namespace(resolved_path, base_part or None)
            model, tokenizer = _backend._load_sampling_model(checkpoint_path, base_part or None)
            return model, tokenizer, namespace
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_name} ({e})")

    if model_name and model_name in _backend.models:
        # In-memory Tinker training model
        return (
            _backend.models[model_name],
            _backend.tokenizers[model_name],
            _backend._student_namespace(model_name),
        )

    # Default: base model
    _backend._ensure_base_model()
    if _backend._base_model is None or _backend._base_tokenizer is None:
        raise HTTPException(status_code=503, detail="Base model not loaded")
    return (
        _backend._base_model,
        _backend._base_tokenizer,
        _backend._base_namespace(_backend.config.base_model),
    )


# ---------------------------------------------------------------------------
# Tool-call parsing: Qwen3.5 <tool_call> XML → OpenAI tool_calls list
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)


def _parse_qwen_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse Qwen3.5 tool-call XML into OpenAI tool_calls. Returns (remaining, tool_calls)."""
    tool_calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        func_name = match.group(1).strip()
        body = match.group(2).strip()
        arguments = {}
        for param_match in re.finditer(
            r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL
        ):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            try:
                arguments[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                arguments[param_name] = param_value
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    remaining = _TOOL_CALL_RE.sub("", text).strip()
    return remaining, tool_calls


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------


def _text_content(msg: ChatMessage) -> str:
    if msg.content is None:
        return ""
    if isinstance(msg.content, list):
        parts = []
        for p in msg.content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(msg.content)


def _normalize_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert ChatMessage list to dicts suitable for apply_chat_template.

    Qwen3.5's Jinja template expects tool_call.arguments to be a dict.
    """
    result = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role, "content": _text_content(m)}
        if m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        if m.role == "tool":
            name = getattr(m, "name", None)
            if name:
                d["name"] = name
        if m.tool_calls is not None:
            normalized_calls = []
            for tc in m.tool_calls:
                tc_copy = dict(tc) if isinstance(tc, dict) else tc
                if isinstance(tc_copy, dict) and "function" in tc_copy:
                    func = dict(tc_copy["function"])
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            func["arguments"] = {}
                    tc_copy = {**tc_copy, "function": func}
                normalized_calls.append(tc_copy)
            d["tool_calls"] = normalized_calls
        if d["role"] == "developer":
            d["role"] = "system"
        result.append(d)
    return result


def _normalized_stop_sequences(stop: list[str] | str | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [s for s in stop if isinstance(s, str) and s]


def _apply_stop_sequences(text: str, stop: list[str] | str | None) -> tuple[str, bool]:
    stop_sequences = _normalized_stop_sequences(stop)
    if not stop_sequences:
        return text, False

    earliest_idx: int | None = None
    for seq in stop_sequences:
        idx = text.find(seq)
        if idx >= 0 and (earliest_idx is None or idx < earliest_idx):
            earliest_idx = idx

    if earliest_idx is None:
        return text, False
    return text[:earliest_idx], True


def _request_max_tokens(request: ChatCompletionRequest | CompletionRequest) -> int:
    max_completion_tokens = getattr(request, "max_completion_tokens", None)
    if max_completion_tokens is not None:
        return int(max_completion_tokens)
    if request.max_tokens is None:
        raise HTTPException(status_code=400, detail="max_tokens must not be null")
    return int(request.max_tokens)


def _supports_response_format(response_format: dict | None) -> bool:
    return response_format is not None


def _extract_json_schema(response_format: dict | None) -> dict[str, Any] | str | None:
    if response_format is None:
        return None
    if response_format.get("type") != "json_schema":
        raise HTTPException(status_code=400, detail="Only response_format type 'json_schema' is supported")

    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        raise HTTPException(status_code=400, detail="response_format.json_schema must be an object")

    schema = json_schema.get("schema")
    if not isinstance(schema, (dict, str)):
        raise HTTPException(
            status_code=400,
            detail="response_format.json_schema.schema must be an object or JSON schema string",
        )
    return schema


def _normalize_tool_choice(
    tool_choice: str | dict | None,
    tools: list[dict] | None,
) -> tuple[str, str | None]:
    tool_names = {
        tool.get("function", {}).get("name")
        for tool in (tools or [])
        if isinstance(tool, dict)
    }

    if tool_choice is None or tool_choice == "auto":
        return "auto", None
    if tool_choice == "none":
        return "none", None
    if tool_choice == "required":
        if not tools:
            raise HTTPException(status_code=400, detail="tool_choice='required' requires tools")
        return "required", None
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise HTTPException(status_code=400, detail="tool_choice dict must have type='function'")
        function = tool_choice.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            raise HTTPException(status_code=400, detail="tool_choice.function.name is required")
        forced_name = str(function["name"])
        if not tools:
            raise HTTPException(status_code=400, detail="tool_choice function requires tools")
        if forced_name not in tool_names:
            raise HTTPException(status_code=400, detail=f"Unknown tool_choice function: {forced_name}")
        return "function", forced_name

    raise HTTPException(status_code=400, detail=f"Unsupported tool_choice: {tool_choice!r}")


def _validate_tool_choice_result(
    tool_choice_mode: str,
    forced_tool_name: str | None,
    tool_calls: list[dict[str, Any]],
) -> None:
    if tool_choice_mode == "none":
        return
    if tool_choice_mode == "required" and not tool_calls:
        raise HTTPException(status_code=400, detail="tool_choice='required' was not satisfied by the model output")
    if tool_choice_mode == "function":
        if not tool_calls:
            raise HTTPException(
                status_code=400,
                detail=f"tool_choice function '{forced_tool_name}' was not satisfied by the model output",
            )
        invalid_names = [
            tc.get("function", {}).get("name")
            for tc in tool_calls
            if tc.get("function", {}).get("name") != forced_tool_name
        ]
        if invalid_names:
            raise HTTPException(
                status_code=400,
                detail=f"tool_choice function '{forced_tool_name}' was not satisfied by the model output",
            )


def _prepare_template_messages(
    messages: list[ChatMessage],
    tool_choice_mode: str,
    forced_tool_name: str | None,
) -> list[dict[str, Any]]:
    normalized = _normalize_messages(messages)
    if tool_choice_mode == "required":
        normalized = [
            {
                "role": "system",
                "content": "You must answer by calling one of the provided tools.",
            },
            *normalized,
        ]
    elif tool_choice_mode == "function" and forced_tool_name is not None:
        normalized = [
            {
                "role": "system",
                "content": f"You must answer by calling the tool named '{forced_tool_name}'.",
            },
            *normalized,
        ]
    return normalized


def _apply_chat_template_with_fallbacks(
    tokenizer,
    messages_dicts: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    tools: list[dict] | None,
    tool_choice: str | dict | None,
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
            return tokenizer.apply_chat_template(messages_dicts, **kwargs)
        except TypeError:
            if not current_items:
                raise
            current_items.pop()


def _unwrap_hf_tokenizer(tokenizer):
    candidate = getattr(tokenizer, "_tokenizer", tokenizer)
    return candidate


def _model_vocab_size(model) -> int | None:
    for attr_chain in (
        ("config", "vocab_size"),
        ("args", "vocab_size"),
        ("vocab_size",),
    ):
        current = model
        try:
            for attr in attr_chain:
                current = getattr(current, attr)
        except AttributeError:
            continue
        if isinstance(current, int):
            return current
    return None


def _postprocess_chat_output(
    full_text: str,
    *,
    stop: list[str] | str | None,
    enable_thinking: bool,
    parse_tool_calls: bool,
    tool_choice_mode: str,
    forced_tool_name: str | None,
    generated_tokens: list[int],
    eos_token_id: int,
) -> tuple[dict[str, Any], str]:
    stopped_text, stop_hit = _apply_stop_sequences(full_text, stop)
    reasoning_content = None
    content = stopped_text
    if enable_thinking:
        reasoning_content, content = _split_thinking(stopped_text)

    tool_calls: list[dict[str, Any]] = []
    content_after_tools = content
    if parse_tool_calls:
        content_after_tools, tool_calls = _parse_qwen_tool_calls(content)
        _validate_tool_choice_result(tool_choice_mode, forced_tool_name, tool_calls)

    has_tool_calls = len(tool_calls) > 0
    message: dict[str, Any] = {"role": "assistant"}
    if has_tool_calls:
        message["content"] = content_after_tools or None
        message["tool_calls"] = tool_calls
    else:
        message["content"] = content
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content

    finish_reason = _finish_reason(
        generated_tokens,
        eos_token_id,
        has_tool_calls=has_tool_calls,
        stop_hit=stop_hit,
    )
    return message, finish_reason


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------


def _iter_generated_tokens(
    model,
    prompt_tokens: list[int],
    temperature: float,
    top_p: float,
    max_tokens: int,
    namespace: str | None = None,
    logits_processor=None,
):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature, top_p=top_p)
    prompt_cache = None
    prompt_tail = list(prompt_tokens)
    saved_chunk_lengths: set[int] = set()
    generated_tokens: list[int] = []
    if _backend is not None:
        prompt_cache, prompt_tail = _backend.inference._prepare_prompt_cache(
            model,
            prompt_tokens,
            namespace,
        )
    prompt_array = mx.array(prompt_tail)
    max_kv_size = getattr(_backend.config, "max_kv_cache_size", None) if _backend else None
    kv_bits = getattr(_backend.config, "kv_cache_bits", None) if _backend else None
    kv_group_size = getattr(_backend.config, "kv_cache_group_size", 64) if _backend else 64
    quantized_kv_start = getattr(_backend.config, "quantized_kv_start", 0) if _backend else 0

    kwargs: dict[str, Any] = {}
    if logits_processor is not None:
        generate_sig = inspect.signature(generate_step)
        if "logits_processors" in generate_sig.parameters:
            kwargs["logits_processors"] = [logits_processor]
        elif "logits_processor" in generate_sig.parameters:
            kwargs["logits_processor"] = logits_processor
    else:
        generate_sig = inspect.signature(generate_step)
    if prompt_cache is not None and "prompt_cache" in generate_sig.parameters:
        kwargs["prompt_cache"] = prompt_cache

    try:
        for token, logprobs in generate_step(
            prompt=prompt_array,
            model=model,
            max_tokens=max_tokens,
            sampler=sampler,
            max_kv_size=max_kv_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
            **kwargs,
        ):
            token_id = token.item() if hasattr(token, "item") else int(token)
            generated_tokens.append(token_id)
            if _backend is not None:
                _backend.inference._checkpoint_transcript_prefixes(
                    namespace,
                    prompt_tokens,
                    generated_tokens,
                    prompt_cache,
                    saved_chunk_lengths,
                )
            yield token, logprobs
    finally:
        if _backend is not None:
            _backend.inference._persist_final_transcript(
                namespace,
                prompt_tokens,
                generated_tokens,
                prompt_cache,
            )


def _generate_tokens(
    model,
    tokenizer,
    prompt_tokens,
    temperature,
    top_p,
    max_tokens,
    namespace: str | None = None,
    logits_processor=None,
):
    generated_tokens = []
    for token, _ in _iter_generated_tokens(
        model, prompt_tokens, temperature, top_p, max_tokens, namespace, logits_processor
    ):
        token_id = token.item() if hasattr(token, "item") else int(token)
        generated_tokens.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
    return generated_tokens


def _finish_reason(generated_tokens, eos_token_id, has_tool_calls=False, stop_hit=False):
    if has_tool_calls:
        return "tool_calls"
    if stop_hit:
        return "stop"
    if generated_tokens and generated_tokens[-1] == eos_token_id:
        return "stop"
    return "length"


def _usage(prompt_tokens, generated_tokens):
    return {
        "prompt_tokens": len(prompt_tokens),
        "completion_tokens": len(generated_tokens),
        "total_tokens": len(prompt_tokens) + len(generated_tokens),
    }


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


def _safe_decode(tokenizer, tokens, skip_special=True):
    if skip_special:
        try:
            return tokenizer.decode(tokens, skip_special_tokens=True)
        except TypeError:
            pass
    return tokenizer.decode(tokens)


# ---------------------------------------------------------------------------
# /v1/chat/completions (non-streaming)
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    _model, tokenizer, namespace = _get_model_and_tokenizer_and_namespace(request.model)
    if request.tools and request.response_format is not None:
        raise HTTPException(status_code=400, detail="tools and response_format cannot be used together")

    tool_choice_mode, forced_tool_name = _normalize_tool_choice(request.tool_choice, request.tools)
    parse_tool_calls = request.tools is not None and tool_choice_mode != "none"
    request_tools = request.tools if parse_tool_calls else None
    messages_dicts = _prepare_template_messages(request.messages, tool_choice_mode, forced_tool_name)
    max_tokens = _request_max_tokens(request)

    # Resolve enable_thinking
    enable_thinking = not _supports_response_format(request.response_format)
    extra_body = getattr(request, "extra_body", None) or {}
    if isinstance(extra_body, dict):
        ctk = extra_body.get("chat_template_kwargs", {})
        if enable_thinking and isinstance(ctk, dict) and "enable_thinking" in ctk:
            enable_thinking = ctk["enable_thinking"]
        elif enable_thinking and "enable_thinking" in extra_body:
            enable_thinking = extra_body["enable_thinking"]

    # Format prompt
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = _apply_chat_template_with_fallbacks(
            tokenizer,
            messages_dicts,
            enable_thinking=enable_thinking,
            tools=request_tools,
            tool_choice=request.tool_choice if parse_tool_calls else None,
        )
    else:
        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages_dicts)

    prompt_tokens = tokenizer.encode(prompt_text)

    # Optional xgrammar constrained decoding
    logits_processor = None
    schema = _extract_json_schema(request.response_format)
    if schema is not None:
        logits_processor = _make_json_schema_processor(_model, tokenizer, schema)

    if request.stream:
        return StreamingResponse(
            _stream_chat_response(
                _model,
                tokenizer,
                prompt_tokens,
                request,
                namespace=namespace,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                logits_processor=logits_processor,
                parse_tool_calls=parse_tool_calls,
                tool_choice_mode=tool_choice_mode,
                forced_tool_name=forced_tool_name,
            ),
            media_type="text/event-stream",
        )

    generated_tokens = _generate_tokens(
        _model,
        tokenizer,
        prompt_tokens,
        request.temperature,
        request.top_p,
        max_tokens,
        namespace,
        logits_processor,
    )
    completion_text = _safe_decode(tokenizer, generated_tokens)
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    message, finish_reason = _postprocess_chat_output(
        completion_text,
        stop=request.stop,
        enable_thinking=enable_thinking,
        parse_tool_calls=parse_tool_calls,
        tool_choice_mode=tool_choice_mode,
        forced_tool_name=forced_tool_name,
        generated_tokens=generated_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )

    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": _usage(prompt_tokens, generated_tokens),
    }


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------


@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    _model, tokenizer, namespace = _get_model_and_tokenizer_and_namespace(request.model)
    max_tokens = _request_max_tokens(request)
    prompt_text = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    prompt_tokens = tokenizer.encode(prompt_text)
    if request.stream:
        return StreamingResponse(
            _stream_completion_response(
                _model,
                tokenizer,
                prompt_tokens,
                request,
                max_tokens=max_tokens,
                namespace=namespace,
            ),
            media_type="text/event-stream",
        )
    generated_tokens = _generate_tokens(
        _model, tokenizer, prompt_tokens, request.temperature, request.top_p, max_tokens, namespace,
    )
    completion_text = _safe_decode(tokenizer, generated_tokens)
    completion_text, stop_hit = _apply_stop_sequences(completion_text, request.stop)
    resp_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    return {
        "id": resp_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "text": completion_text,
            "index": 0,
            "finish_reason": _finish_reason(generated_tokens, tokenizer.eos_token_id, stop_hit=stop_hit),
        }],
        "usage": _usage(prompt_tokens, generated_tokens),
    }


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


@router.get("/v1/models")
async def list_models():
    ts = int(time.time())
    models = []

    if _backend is not None:
        base = _backend.config.base_model
        models.append({
            "id": base,
            "object": "model",
            "created": ts,
            "owned_by": "mlx-tinker",
        })

        # List in-memory Tinker training models
        for model_id in _backend.models:
            models.append({
                "id": model_id,
                "object": "model",
                "created": ts,
                "owned_by": "mlx-tinker",
                "metadata": {"type": "lora_training"},
            })

        # List persisted LoRA sampler checkpoints on disk
        for item in await LoraCatalogService(_backend).list_openai_export_models():
            if not item.relative_path:
                continue
            created_at = item.created_at or ""
            try:
                created_ts = int(datetime.fromisoformat(created_at).timestamp())
            except ValueError:
                created_ts = ts
            models.append({
                "id": item.openai_model_id,
                "object": "model",
                "created": created_ts,
                "owned_by": "mlx-tinker",
                "metadata": {"type": "lora_checkpoint", "checkpoint": item.relative_path},
            })

    return {"object": "list", "data": models}


# ---------------------------------------------------------------------------
# Streaming — full non-streaming generation, then emit chunks
# ---------------------------------------------------------------------------
# This approach generates the full response first (to correctly parse
# <think> and <tool_call> blocks), then emits it as SSE chunks.
# This avoids the complexity of streaming tool-call parsing.


async def _stream_chat_response(
    model,
    tokenizer,
    prompt_tokens,
    request,
    *,
    namespace: str | None,
    max_tokens: int,
    enable_thinking=True,
    logits_processor=None,
    parse_tool_calls=True,
    tool_choice_mode="auto",
    forced_tool_name: str | None = None,
):
    """SSE streaming generator for chat completions.

    Generates the complete response first to correctly handle <think> and
    <tool_call> blocks, then streams the result as chunks.
    """
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Generate full response
    generated_tokens = _generate_tokens(
        model,
        tokenizer,
        prompt_tokens,
        request.temperature,
        request.top_p,
        max_tokens,
        namespace,
        logits_processor,
    )
    full_text = _safe_decode(tokenizer, generated_tokens)
    message, finish_reason = _postprocess_chat_output(
        full_text,
        stop=request.stop,
        enable_thinking=enable_thinking,
        parse_tool_calls=parse_tool_calls,
        tool_choice_mode=tool_choice_mode,
        forced_tool_name=forced_tool_name,
        generated_tokens=generated_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    has_tool_calls = bool(message.get("tool_calls"))

    # Emit chunks
    ts = int(time.time())

    if has_tool_calls:
        content_after_tools = message.get("content")
        tool_calls = message["tool_calls"]
        # Emit tool_calls as delta chunks (OpenAI streaming tool call format)
        for i, tc in enumerate(tool_calls):
            # First chunk: role + tool call start
            delta: dict[str, Any] = {}
            if i == 0:
                delta["role"] = "assistant"
                if content_after_tools:
                    delta["content"] = content_after_tools
            delta["tool_calls"] = [{
                "index": i,
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                },
            }]
            chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": ts,
                "model": request.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk with finish_reason
        chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": ts,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    else:
        # Stream content character-by-character (or in small chunks)
        text_to_stream = message.get("content") or ""
        CHUNK_SIZE = 4  # characters per chunk for streaming feel
        for i in range(0, len(text_to_stream), CHUNK_SIZE):
            piece = text_to_stream[i : i + CHUNK_SIZE]
            delta: dict[str, Any] = {"content": piece}
            if i == 0:
                delta["role"] = "assistant"
            chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": ts,
                "model": request.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": ts,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


async def _stream_completion_response(model, tokenizer, prompt_tokens, request, *, max_tokens: int, namespace: str | None):
    resp_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    generated_tokens = _generate_tokens(
        model,
        tokenizer,
        prompt_tokens,
        request.temperature,
        request.top_p,
        max_tokens,
        namespace,
    )
    full_text = _safe_decode(tokenizer, generated_tokens)
    completion_text, stop_hit = _apply_stop_sequences(full_text, request.stop)
    finish_reason = _finish_reason(generated_tokens, tokenizer.eos_token_id, stop_hit=stop_hit)
    ts = int(time.time())

    chunk_size = 4
    for i in range(0, len(completion_text), chunk_size):
        piece = completion_text[i : i + chunk_size]
        chunk = {
            "id": resp_id,
            "object": "text_completion",
            "created": ts,
            "model": request.model,
            "choices": [{"text": piece, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    final_chunk = {
        "id": resp_id,
        "object": "text_completion",
        "created": ts,
        "model": request.model,
        "choices": [{"text": "", "index": 0, "finish_reason": finish_reason}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# xgrammar constrained decoding (optional)
# ---------------------------------------------------------------------------


def _make_json_schema_processor(model, tokenizer, json_schema):
    try:
        import xgrammar as xgr
        from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="response_format json_schema requires xgrammar to be installed",
        )

    import mlx.core as mx

    schema_str = json.dumps(json_schema) if isinstance(json_schema, dict) else json_schema
    try:
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            _unwrap_hf_tokenizer(tokenizer),
            vocab_size=_model_vocab_size(model),
            stop_token_ids=getattr(tokenizer, "eos_token_id", None),
        )
        compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = compiler.compile_json_schema(schema_str)
        matcher = xgr.GrammarMatcher(compiled)
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    except Exception as exc:
        logger.exception("Failed to initialize response_format json_schema processor")
        raise HTTPException(status_code=400, detail=f"Invalid response_format json_schema: {exc}") from exc

    def processor(tokens, logits):
        if len(tokens) == 0:
            return logits

        last_token = tokens[-1].item() if hasattr(tokens[-1], "item") else int(tokens[-1])
        accepted = matcher.accept_token(last_token) if not matcher.is_terminated() else False
        if not accepted:
            matcher.reset()
            matcher.accept_token(last_token)

        if matcher.is_terminated():
            return logits

        matcher.fill_next_token_bitmask(bitmask)
        mask_input = mx.array(bitmask.numpy()) if hasattr(bitmask, "numpy") else bitmask
        return apply_token_bitmask_mlx(mask_input, logits, tokenizer_info.vocab_size)

    return processor
