#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
MLX_TINKER_PORT="${MLX_TINKER_PORT:-8010}"
OPENCLAW_PROXY_PORT="${OPENCLAW_PROXY_PORT:-30000}"

echo "Starting mlx-tinker on port ${MLX_TINKER_PORT}"
uv run python -m mlx_tinker --model "$MODEL_NAME" --port "$MLX_TINKER_PORT" &
MLX_PID=$!

cleanup() {
  kill "$MLX_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 3

echo "Launching local OpenClaw RL trainer"
uv run python -m integrations.openclaw_rl_local.run \
  --model-name "$MODEL_NAME" \
  --runtime podman \
  --mlx-tinker-base-url "http://127.0.0.1:${MLX_TINKER_PORT}" \
  --proxy-port "$OPENCLAW_PROXY_PORT"
