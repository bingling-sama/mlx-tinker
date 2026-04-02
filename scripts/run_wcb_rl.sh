#!/usr/bin/env bash
# Run upstream OpenClaw-RL on WildClawBench tasks against local mlx-tinker.
#
# This script orchestrates:
#   1. mlx-tinker (native Metal inference/training on $PORT)
#   2. openclaw-tinker RL proxy+trainer (port 30000, waits for training samples)
#   3. WCB benchmark episodes (Docker containers with rl-training-headers extension)
#
# The trainer blocks on drain_output_queue(). Each WCB episode routes LLM calls
# through the proxy, which collects training samples. Once batch_size samples
# accumulate, the trainer processes a training step.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Configuration ──────────────────────────────────────────────────────
MODEL="${MLX_TINKER_MODEL:-Qwen/Qwen3.5-4B}"
PORT="${MLX_TINKER_PORT:-8020}"
MAX_WAIT="${MLX_TINKER_MAX_WAIT:-120}"
RUN_ID="${OPENCLAW_RUN_ID:-wcb_rl_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OPENCLAW_RUN_DIR:-$PROJECT_DIR/workspace_reports/openclaw_upstream/runs/$RUN_ID}"
REUSE_EXISTING="${MLX_TINKER_REUSE_EXISTING:-0}"
OPENCLAW_RL_DIR="${OPENCLAW_RL_DIR:-$PROJECT_DIR/.external/openclaw-rl}"
OPENCLAW_RL_REPO_URL="${OPENCLAW_RL_REPO_URL:-https://github.com/ojus1/OpenClaw-RL.git}"
OPENCLAW_RL_REF="${OPENCLAW_RL_REF:-codex/qwen35-openclaw-tinker}"

WCB_DIR="$PROJECT_DIR/.external/wildclaw-bench"
RL_DIR="$OPENCLAW_RL_DIR"
RL_EXT_PATH="$RL_DIR/extensions/rl-training-headers"
PROXY_MODELS_CONFIG="$WCB_DIR/my_api_proxy.json"

LOG_DIR="$RUN_DIR/logs"
META_DIR="$RUN_DIR/meta"
RECORD_DIR="$RUN_DIR/records"
MLX_DB_PATH="$RUN_DIR/run.db"
MLX_CHECKPOINT_DIR="$RUN_DIR/checkpoints"
MLX_TINKER_LOG="$LOG_DIR/mlx-tinker.log"
TRAIN_LOG="$LOG_DIR/train.log"
DRIVER_DIR="$RUN_DIR/driver_batches"

mkdir -p "$LOG_DIR" "$META_DIR" "$RECORD_DIR" "$MLX_CHECKPOINT_DIR" "$DRIVER_DIR"

bash "$PROJECT_DIR/scripts/bootstrap_openclaw_rl.sh"
OPENCLAW_RL_COMMIT="$(git -C "$RL_DIR" rev-parse HEAD)"

export TINKER_BASE_URL="http://localhost:${PORT}"
export TINKER_API_KEY="tml-local"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

# ── Parse arguments ────────────────────────────────────────────────────
# Split our flags (--task, --tasks-file, --episodes-per-step) from upstream flags (passed through)
TASK_PATH=""
TASKS_FILE=""
EPISODES_PER_STEP=4
WCB_MODEL="mlx-tinker/openclaw-tinker-proxy"
UPSTREAM_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK_PATH="$2"; shift 2 ;;
        --tasks-file)
            TASKS_FILE="$2"; shift 2 ;;
        --episodes-per-step)
            EPISODES_PER_STEP="$2"; shift 2 ;;
        --wcb-model)
            WCB_MODEL="$2"; shift 2 ;;
        *)
            UPSTREAM_ARGS+=("$1"); shift ;;
    esac
done

# Build task list from --task or --tasks-file
TASK_LIST=()
if [[ -n "$TASKS_FILE" ]]; then
    if [[ ! -f "$TASKS_FILE" ]]; then
        echo "ERROR: Tasks file not found: $TASKS_FILE" >&2
        exit 1
    fi
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        if [[ ! -f "$line" ]]; then
            echo "WARNING: Task file not found, skipping: $line" >&2
            continue
        fi
        TASK_LIST+=("$(cd "$(dirname "$line")" && pwd)/$(basename "$line")")
    done < "$TASKS_FILE"
    echo "Loaded ${#TASK_LIST[@]} tasks from $TASKS_FILE"
elif [[ -n "$TASK_PATH" ]]; then
    if [[ ! -f "$TASK_PATH" ]]; then
        echo "ERROR: Task file not found: $TASK_PATH" >&2
        exit 1
    fi
    TASK_LIST+=("$(cd "$(dirname "$TASK_PATH")" && pwd)/$(basename "$TASK_PATH")")
else
    echo "ERROR: --task <path> or --tasks-file <path> is required" >&2
    exit 1
fi

# ── Save run metadata ─────────────────────────────────────────────────
{
    echo "date=$(date -Iseconds)"
    echo "model=$MODEL"
    echo "port=$PORT"
    echo "run_dir=$RUN_DIR"
    echo "num_tasks=${#TASK_LIST[@]}"
    echo "episodes_per_step=$EPISODES_PER_STEP"
    echo "wcb_model=$WCB_MODEL"
    echo "db=$MLX_DB_PATH"
    echo "checkpoints=$MLX_CHECKPOINT_DIR"
    echo "records=$RECORD_DIR"
    echo "openclaw_rl_dir=$RL_DIR"
    echo "openclaw_rl_repo_url=$OPENCLAW_RL_REPO_URL"
    echo "openclaw_rl_ref=$OPENCLAW_RL_REF"
    echo "openclaw_rl_commit=$OPENCLAW_RL_COMMIT"
} > "$META_DIR/run.env"
printf '%q ' "${UPSTREAM_ARGS[@]}" > "$META_DIR/upstream_args.sh"
printf '\n' >> "$META_DIR/upstream_args.sh"

# ── Process management ─────────────────────────────────────────────────
MLX_TINKER_PID=""
TRAINER_PID=""

cleanup() {
    echo ""
    echo "=== Cleanup ==="
    if [[ -n "${TRAINER_PID:-}" ]] && kill -0 "$TRAINER_PID" 2>/dev/null; then
        echo "  Stopping trainer pid=$TRAINER_PID"
        kill "$TRAINER_PID" 2>/dev/null || true
        wait "$TRAINER_PID" 2>/dev/null || true
    fi
    if [[ -n "${MLX_TINKER_PID:-}" ]] && kill -0 "$MLX_TINKER_PID" 2>/dev/null; then
        echo "  Stopping mlx-tinker pid=$MLX_TINKER_PID"
        kill "$MLX_TINKER_PID" 2>/dev/null || true
        wait "$MLX_TINKER_PID" 2>/dev/null || true
    fi
    echo "  Done"
}
trap cleanup EXIT

wait_for_service() {
    local name="$1" url="$2"; shift 2
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

# ── 1. Start mlx-tinker ───────────────────────────────────────────────
echo "=== Run Directory ==="
echo "  $RUN_DIR"
echo ""

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

# ── 2. Start RL trainer + proxy (background) ──────────────────────────
echo ""
echo "=== OpenClaw-RL Trainer + Proxy ==="
echo "  TINKER_BASE_URL=$TINKER_BASE_URL"
echo "  Dependency: $OPENCLAW_RL_REPO_URL#$OPENCLAW_RL_REF ($OPENCLAW_RL_COMMIT)"
echo "  Upstream args: ${UPSTREAM_ARGS[*]}"
echo ""

cd "$RL_DIR"
nohup uv run python openclaw-tinker/run.py \
    --model-name "$MODEL" \
    --teacher-model-name "$MODEL" \
    --proxy-host 0.0.0.0 \
    --proxy-port 30000 \
    --record-dir "$RECORD_DIR" \
    "${UPSTREAM_ARGS[@]}" \
    > "$TRAIN_LOG" 2>&1 &
TRAINER_PID=$!
cd "$PROJECT_DIR"

echo "  Trainer PID: $TRAINER_PID"
echo "  Waiting for proxy to be ready..."
wait_for_service "RL proxy" "http://localhost:30000/healthz"

# ── 3. Launch WCB episodes in batches ─────────────────────────────────
echo ""
echo "=== WCB Episode Loop ==="
echo "  Tasks: ${#TASK_LIST[@]}"
echo "  Model: $WCB_MODEL"
echo "  Episodes per batch: $EPISODES_PER_STEP"
echo ""

BATCH_NUM=0
TASK_IDX=0
NUM_TASKS=${#TASK_LIST[@]}
while kill -0 "$TRAINER_PID" 2>/dev/null; do
    BATCH_NUM=$((BATCH_NUM + 1))
    BATCH_DIR="$DRIVER_DIR/batch${BATCH_NUM}"
    mkdir -p "$BATCH_DIR"

    echo "--- Batch $BATCH_NUM ($EPISODES_PER_STEP episodes) ---"

    for i in $(seq 1 "$EPISODES_PER_STEP"); do
        # Check if trainer is still alive before launching another episode
        if ! kill -0 "$TRAINER_PID" 2>/dev/null; then
            echo "  Trainer finished, stopping episode launches"
            break 2
        fi

        # Round-robin task selection
        CURRENT_TASK="${TASK_LIST[$((TASK_IDX % NUM_TASKS))]}"
        TASK_IDX=$((TASK_IDX + 1))
        TASK_BASENAME=$(basename "$CURRENT_TASK" .md)

        echo "  Episode ${BATCH_NUM}.${i} [${TASK_BASENAME}] starting..."
        cd "$WCB_DIR"
        python3 eval/run_batch.py \
            --task "$CURRENT_TASK" \
            --model "$WCB_MODEL" \
            --models-config "$PROXY_MODELS_CONFIG" \
            --rl-extension "$RL_EXT_PATH" \
            > "$BATCH_DIR/run${i}_${TASK_BASENAME}.log" 2>&1 || true
        cd "$PROJECT_DIR"
        echo "  Episode ${BATCH_NUM}.${i} [${TASK_BASENAME}] done"
    done

    # Brief pause between batches to let trainer process samples
    if kill -0 "$TRAINER_PID" 2>/dev/null; then
        echo "  Waiting 5s for trainer to process samples..."
        sleep 5
    fi
done

# ── 4. Wait for trainer to finish ─────────────────────────────────────
echo ""
echo "=== Waiting for trainer to complete ==="
if kill -0 "$TRAINER_PID" 2>/dev/null; then
    wait "$TRAINER_PID" || true
fi

echo ""
echo "=== Run Complete ==="
echo "  Run dir:    $RUN_DIR"
echo "  Train log:  $TRAIN_LOG"
echo "  Records:    $RECORD_DIR"
echo "  Checkpoints: $MLX_CHECKPOINT_DIR"
echo "  OpenClaw-RL: $OPENCLAW_RL_REPO_URL#$OPENCLAW_RL_REF ($OPENCLAW_RL_COMMIT)"
echo ""

# Print summary of collected data
CONV_LINES=$(wc -l < "$RECORD_DIR/conversations.jsonl" 2>/dev/null || echo 0)
PRM_LINES=$(wc -l < "$RECORD_DIR/prm_scores.jsonl" 2>/dev/null || echo 0)
echo "  conversations.jsonl: $CONV_LINES entries"
echo "  prm_scores.jsonl:    $PRM_LINES entries"
