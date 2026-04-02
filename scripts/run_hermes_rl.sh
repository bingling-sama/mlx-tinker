#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_DIR="${HERMES_DIR:-$ROOT_DIR/.external/hermes-agent}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
TINKER_HOST="${TINKER_HOST:-127.0.0.1}"
TINKER_PORT="${TINKER_PORT:-8010}"
TINKER_API_KEY="${TINKER_API_KEY:-tml-local}"

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-30050}"
PROXY_API_KEY="${PROXY_API_KEY:-hermes-local}"
METHOD="${METHOD:-rl}"
RECORD_DIR="${RECORD_DIR:-$ROOT_DIR/workspace_reports/hermes_rl/records}"

mkdir -p "$RECORD_DIR"

cleanup() {
  if [[ -n "${PROXY_PID:-}" ]]; then
    kill "$PROXY_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${TINKER_PID:-}" ]]; then
    kill "$TINKER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "$ROOT_DIR"

echo "[hermes-rl] starting mlx-tinker backend on ${TINKER_HOST}:${TINKER_PORT}"
uv run python -m mlx_tinker \
  --model "$MODEL_NAME" \
  --host "$TINKER_HOST" \
  --port "$TINKER_PORT" \
  --kv-cache-bits 4 \
  --quantized-kv-start 4096 \
  --prefix-cache-disk-limit-gb 8 \
  --max-batch-size 8 \
  --cycle-ms 2 \
  >/tmp/hermes_mlx_tinker.log 2>&1 &
TINKER_PID=$!

echo "[hermes-rl] starting Hermes RL bridge on ${PROXY_HOST}:${PROXY_PORT}"
TINKER_BASE_URL="http://${TINKER_HOST}:${TINKER_PORT}" \
TINKER_API_KEY="$TINKER_API_KEY" \
HERMES_RL_PROXY_API_KEY="$PROXY_API_KEY" \
HERMES_RL_RECORD_DIR="$RECORD_DIR" \
uv run python -m integrations.hermes_rl.run \
  --method "$METHOD" \
  --model-name "$MODEL_NAME" \
  --served-model-name hermes-local \
  --proxy-host "$PROXY_HOST" \
  --proxy-port "$PROXY_PORT" \
  --proxy-api-key "$PROXY_API_KEY" \
  --tinker-base-url "http://${TINKER_HOST}:${TINKER_PORT}" \
  --tinker-api-key "$TINKER_API_KEY" \
  >/tmp/hermes_rl_bridge.log 2>&1 &
PROXY_PID=$!

export HERMES_RL_ENABLED=1
export HERMES_RL_PROXY_BASE_URL="http://${PROXY_HOST}:${PROXY_PORT}/v1"

echo
echo "[hermes-rl] stack ready"
echo "  mlx-tinker:  http://${TINKER_HOST}:${TINKER_PORT}"
echo "  RL proxy:    ${HERMES_RL_PROXY_BASE_URL}"
echo "  model alias: hermes-local"
echo
echo "Logs:"
echo "  tail -f /tmp/hermes_mlx_tinker.log"
echo "  tail -f /tmp/hermes_rl_bridge.log"
echo

if [[ $# -gt 0 ]]; then
  echo "[hermes-rl] launching Hermes with supplied args"
  exec uv run --directory "$HERMES_DIR" python run_agent.py \
    --model hermes-local \
    --base_url "$HERMES_RL_PROXY_BASE_URL" \
    --api_key "$PROXY_API_KEY" \
    "$@"
fi

echo "Run Hermes manually with:"
echo "  HERMES_RL_ENABLED=1 HERMES_RL_PROXY_BASE_URL=${HERMES_RL_PROXY_BASE_URL} \\"
echo "  uv run --directory \"$HERMES_DIR\" python run_agent.py \\"
echo "    --model hermes-local --base_url \"$HERMES_RL_PROXY_BASE_URL\" --api_key \"$PROXY_API_KEY\""
echo
echo "Press Ctrl+C to stop the stack."
wait
