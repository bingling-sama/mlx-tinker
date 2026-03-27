"""OpenAI-compatible endpoints: /v1/chat/completions, /v1/completions."""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_tinker.backend.mlx_backend import MLXBackend

logger = logging.getLogger(__name__)

router = APIRouter()

# Backend reference — set by register_openai_routes()
_backend: MLXBackend | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    stop: list[str] | str | None = None


class CompletionRequest(BaseModel):
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


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _backend is None:
        raise HTTPException(status_code=503, detail="Backend not initialized")

    _backend._ensure_base_model()
    tokenizer = _backend._base_tokenizer
    model = _backend._base_model

    # Format messages using tokenizer's chat template
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = "\n".join(f"{m.role}: {m.content}" for m in request.messages)

    prompt_tokens = tokenizer.encode(prompt_text)

    if request.stream:
        return StreamingResponse(
            _stream_chat_response(model, tokenizer, prompt_tokens, request),
            media_type="text/event-stream",
        )

    # Non-streaming
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
    generated_tokens = []
    prompt_array = mx.array(prompt_tokens)

    for token, _ in generate_step(
        prompt=prompt_array,
        model=model,
        max_tokens=request.max_tokens,
        sampler=sampler,
    ):
        token_id = token.item()
        generated_tokens.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break
        if len(generated_tokens) >= request.max_tokens:
            break

    completion_text = tokenizer.decode(generated_tokens)
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion_text},
                "finish_reason": "stop"
                if generated_tokens and generated_tokens[-1] == tokenizer.eos_token_id
                else "length",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(generated_tokens),
            "total_tokens": len(prompt_tokens) + len(generated_tokens),
        },
    }


@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    if _backend is None:
        raise HTTPException(status_code=503, detail="Backend not initialized")

    _backend._ensure_base_model()
    tokenizer = _backend._base_tokenizer
    model = _backend._base_model

    prompt_text = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    prompt_tokens = tokenizer.encode(prompt_text)

    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
    generated_tokens = []
    prompt_array = mx.array(prompt_tokens)

    for token, _ in generate_step(
        prompt=prompt_array,
        model=model,
        max_tokens=request.max_tokens,
        sampler=sampler,
    ):
        token_id = token.item()
        generated_tokens.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break
        if len(generated_tokens) >= request.max_tokens:
            break

    completion_text = tokenizer.decode(generated_tokens)
    resp_id = f"cmpl-{uuid.uuid4().hex[:12]}"

    return {
        "id": resp_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "text": completion_text,
                "index": 0,
                "finish_reason": "stop"
                if generated_tokens and generated_tokens[-1] == tokenizer.eos_token_id
                else "length",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(generated_tokens),
            "total_tokens": len(prompt_tokens) + len(generated_tokens),
        },
    }


@router.get("/v1/models")
async def list_models():
    base = _backend.config.base_model if _backend else "unknown"
    return {
        "object": "list",
        "data": [
            {
                "id": base,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx-tinker",
            }
        ],
    }


async def _stream_chat_response(model, tokenizer, prompt_tokens, request):
    """SSE streaming generator for chat completions."""
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    prompt_array = mx.array(prompt_tokens)
    count = 0

    for token, _ in generate_step(
        prompt=prompt_array,
        model=model,
        max_tokens=request.max_tokens,
        sampler=sampler,
    ):
        token_id = token.item()
        count += 1

        if token_id == tokenizer.eos_token_id:
            # Final chunk
            chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            break

        token_text = tokenizer.decode([token_id])
        chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        if count >= request.max_tokens:
            chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            break

    yield "data: [DONE]\n\n"
