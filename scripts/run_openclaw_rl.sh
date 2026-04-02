#!/usr/bin/env bash
# Run upstream OpenClaw-RL against local mlx-tinker using a single live model.
#
# mlx-tinker runs natively on Metal for inference/training. Docker is only used
# for the OpenClaw gateway/proxy process.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MODEL="${MLX_TINKER_MODEL:-Qwen/Qwen3.5-4B}"
PORT="${MLX_TINKER_PORT:-8010}"
GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-mlx-tinker-local}"
MAX_WAIT="${MLX_TINKER_MAX_WAIT:-60}"
RUN_ID="${OPENCLAW_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OPENCLAW_RUN_DIR:-$PROJECT_DIR/workspace_reports/openclaw_upstream/runs/$RUN_ID}"
REUSE_EXISTING="${MLX_TINKER_REUSE_EXISTING:-0}"
OPENCLAW_RL_DIR="${OPENCLAW_RL_DIR:-$PROJECT_DIR/.external/openclaw-rl}"
OPENCLAW_RL_REPO_URL="${OPENCLAW_RL_REPO_URL:-https://github.com/ojus1/OpenClaw-RL.git}"
OPENCLAW_RL_REF="${OPENCLAW_RL_REF:-codex/qwen35-openclaw-tinker}"

LOG_DIR="$RUN_DIR/logs"
META_DIR="$RUN_DIR/meta"
RECORD_DIR="$RUN_DIR/records"
MLX_DB_PATH="$RUN_DIR/run.db"
MLX_CHECKPOINT_DIR="$RUN_DIR/checkpoints"
MLX_TINKER_LOG="$LOG_DIR/mlx-tinker.log"
TRAIN_LOG="$LOG_DIR/train.log"

mkdir -p "$LOG_DIR" "$META_DIR" "$RECORD_DIR" "$MLX_CHECKPOINT_DIR"

bash "$PROJECT_DIR/scripts/bootstrap_openclaw_rl.sh"
OPENCLAW_RL_COMMIT="$(git -C "$OPENCLAW_RL_DIR" rev-parse HEAD)"

export TINKER_BASE_URL="http://localhost:${PORT}"
export TINKER_API_KEY="tml-local"
export OPENCLAW_GATEWAY_TOKEN="$GATEWAY_TOKEN"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

{
    echo "date=$(date -Iseconds)"
    echo "model=$MODEL"
    echo "port=$PORT"
    echo "run_dir=$RUN_DIR"
    echo "db=$MLX_DB_PATH"
    echo "checkpoints=$MLX_CHECKPOINT_DIR"
    echo "records=$RECORD_DIR"
    echo "wandb_disabled=$WANDB_DISABLED"
    echo "openclaw_rl_dir=$OPENCLAW_RL_DIR"
    echo "openclaw_rl_repo_url=$OPENCLAW_RL_REPO_URL"
    echo "openclaw_rl_ref=$OPENCLAW_RL_REF"
    echo "openclaw_rl_commit=$OPENCLAW_RL_COMMIT"
} > "$META_DIR/run.env"
printf '%q ' "$@" > "$META_DIR/upstream_args.sh"
printf '\n' >> "$META_DIR/upstream_args.sh"

MLX_TINKER_PID=""

cleanup() {
    if [[ -n "$MLX_TINKER_PID" ]] && kill -0 "$MLX_TINKER_PID" >/dev/null 2>&1; then
        echo "Stopping mlx-tinker pid=$MLX_TINKER_PID"
        kill "$MLX_TINKER_PID" >/dev/null 2>&1 || true
        wait "$MLX_TINKER_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

wait_for_service() {
    local name="$1"
    local url="$2"
    shift 2
    local elapsed=0

    while [[ "$elapsed" -lt "$MAX_WAIT" ]]; do
        if curl -sf "$@" "$url" >/dev/null 2>&1; then
            echo "  $name ready (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    echo "  ERROR: $name failed to start within ${MAX_WAIT}s" >&2
    return 1
}

echo "=== Run Directory ==="
echo "  $RUN_DIR"

echo "=== mlx-tinker ==="
if curl -sf "http://localhost:${PORT}/api/v1/healthz" >/dev/null 2>&1; then
    if [[ "$REUSE_EXISTING" == "1" ]]; then
        echo "  Reusing existing mlx-tinker on port ${PORT}"
    else
        echo "  ERROR: port ${PORT} already has a running mlx-tinker instance." >&2
        echo "  Set MLX_TINKER_REUSE_EXISTING=1 to reuse it, or choose another MLX_TINKER_PORT." >&2
        exit 1
    fi
else
    echo "  Starting mlx-tinker natively (model: ${MODEL})..."
    nohup uv run python -m mlx_tinker \
        --model "$MODEL" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --db "$MLX_DB_PATH" \
        --checkpoints "$MLX_CHECKPOINT_DIR" \
        > "$MLX_TINKER_LOG" 2>&1 &
    MLX_TINKER_PID=$!
    wait_for_service "mlx-tinker" "http://localhost:${PORT}/api/v1/healthz"
fi

echo "=== OpenClaw gateway ==="
if curl -sf -H "Authorization: Bearer $GATEWAY_TOKEN" "http://localhost:18789/healthz" >/dev/null 2>&1; then
    echo "  Already running on port 18789"
else
    echo "  Starting OpenClaw Docker gateway..."
    (cd .external/openclaw && docker compose up -d openclaw-gateway) 2>&1 | sed 's/^/  /'
    wait_for_service \
        "OpenClaw gateway" \
        "http://localhost:18789/healthz" \
        -H "Authorization: Bearer $GATEWAY_TOKEN"
fi

echo "=== OpenClaw-RL (upstream) ==="
echo "  TINKER_BASE_URL=$TINKER_BASE_URL"
echo "  WANDB_DISABLED=$WANDB_DISABLED"
echo "  dependency: $OPENCLAW_RL_REPO_URL#$OPENCLAW_RL_REF ($OPENCLAW_RL_COMMIT)"
echo "  args: $*"
echo ""

cd "$OPENCLAW_RL_DIR"
uv run python openclaw-tinker/run.py \
    --model-name "$MODEL" \
    --teacher-model-name "$MODEL" \
    --proxy-host 0.0.0.0 \
    --proxy-port 30000 \
    --record-dir "$RECORD_DIR" \
    "$@" 2>&1 | tee "$TRAIN_LOG"
