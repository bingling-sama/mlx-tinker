# MLX-Tinker

Tinker API-compatible backend for Mac (Apple Silicon / Metal) using MLX and mlx-lm.

## Project overview
- Three-layer architecture: API (FastAPI) → Engine (asyncio) → Backend (MLX)
- v0.1 targets Qwen3.5-9B with QLoRA finetuning
- Single-process, unified memory — no distributed workers

## Commands
- Install: `uv sync --all-extras`
- Run server: `uv run python -m mlx_tinker --model Qwen/Qwen3.5-9B`
- Tests: `uv run pytest tests/ -k "not stress and not cookbook"`
- Stress tests: `uv run pytest tests/stress/ -m stress`
- Cookbook tests: `uv run pytest tests/cookbook/ -m cookbook`
- Lint: `uv run ruff check mlx_tinker/`
- Format: `uv run ruff format mlx_tinker/`

## Architecture
- `mlx_tinker/api/` — FastAPI server, Tinker endpoints + OpenAI-compat
- `mlx_tinker/engine/` — Async polling loop with barrier-aware batching
- `mlx_tinker/backend/` — MLX compute: training, inference, QLoRA, checkpointing
- `mlx_tinker/db/` — SQLModel ORM with async SQLite (WAL mode)
- `mlx_tinker/types.py` — Shared enums and Tinker-compatible types
- `mlx_tinker/config.py` — Pydantic config

## Key patterns
- All training/sample endpoints return FutureResponse; client polls retrieve_future
- Engine runs 100ms cycles, batches compatible requests, respects barriers (optim_step blocks)
- Backend uses nn.value_and_grad for forward_backward, accumulates grads until optim_step
- QLoRA: 4-bit quantized base + LoRA adapters via mlx-lm tuner utilities
