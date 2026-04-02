#!/usr/bin/env bash
# Full WCB RL experiment: baseline eval → RL training → post-train eval.
#
# Runs all non-coding, non-search WCB tasks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MODEL="${MLX_TINKER_MODEL:-Qwen/Qwen3.5-2B}"
PORT="${MLX_TINKER_PORT:-8020}"
MAX_WAIT="${MLX_TINKER_MAX_WAIT:-300}"
RUN_ID="${OPENCLAW_RUN_ID:-wcb_full_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$PROJECT_DIR/workspace_reports/openclaw_upstream/runs/$RUN_ID"
TASKS_FILE="$PROJECT_DIR/configs/wcb_rl_tasks.txt"
RESUME="${WCB_RESUME:-1}"
MAX_STEPS="${WCB_MAX_STEPS:-32}"
EVAL_PARALLEL="${WCB_EVAL_PARALLEL:-1}"
OPENCLAW_RL_DIR="${OPENCLAW_RL_DIR:-$PROJECT_DIR/.external/openclaw-rl}"
OPENCLAW_RL_REPO_URL="${OPENCLAW_RL_REPO_URL:-https://github.com/ojus1/OpenClaw-RL.git}"
OPENCLAW_RL_REF="${OPENCLAW_RL_REF:-codex/qwen35-openclaw-tinker}"

WCB_DIR="$PROJECT_DIR/.external/wildclaw-bench"
CATEGORIES="01_Productivity_Flow 03_Social_Interaction 05_Creative_Synthesis 06_Safety_Alignment"

EVAL_BASE_CONFIG="$WCB_DIR/my_api_eval_base.json"
EVAL_RL_CONFIG="$WCB_DIR/my_api_eval_rl.json"

LOG_DIR="$RUN_DIR/logs"
META_DIR="$RUN_DIR/meta"
CKPT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$META_DIR" "$RUN_DIR/eval_before" "$RUN_DIR/eval_after"

bash "$PROJECT_DIR/scripts/bootstrap_openclaw_rl.sh"
OPENCLAW_RL_COMMIT="$(git -C "$OPENCLAW_RL_DIR" rev-parse HEAD)"
{
    echo "openclaw_rl_dir=$OPENCLAW_RL_DIR"
    echo "openclaw_rl_repo_url=$OPENCLAW_RL_REPO_URL"
    echo "openclaw_rl_ref=$OPENCLAW_RL_REF"
    echo "openclaw_rl_commit=$OPENCLAW_RL_COMMIT"
} > "$META_DIR/openclaw_rl.env"

MLX_PID=""
cleanup() {
    if [[ -n "${MLX_PID:-}" ]] && kill -0 "$MLX_PID" 2>/dev/null; then
        echo "  Stopping mlx-tinker pid=$MLX_PID"
        kill "$MLX_PID" 2>/dev/null || true
        wait "$MLX_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_mlx() {
    if curl -sf "http://localhost:${PORT}/api/v1/healthz" >/dev/null 2>&1; then
        echo "  mlx-tinker already running on port $PORT"
        return
    fi
    echo "  Starting mlx-tinker (model: $MODEL, port: $PORT)..."
    nohup uv run python -m mlx_tinker \
        --model "$MODEL" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --checkpoints "$CKPT_DIR" \
        > "$LOG_DIR/mlx-tinker.log" 2>&1 &
    MLX_PID=$!
    local elapsed=0
    while [[ "$elapsed" -lt "$MAX_WAIT" ]]; do
        curl -sf "http://localhost:${PORT}/api/v1/healthz" >/dev/null 2>&1 && echo "  Ready (${elapsed}s)" && return 0
        sleep 2; elapsed=$((elapsed + 2))
    done
    echo "  ERROR: mlx-tinker failed to start" >&2; exit 1
}

run_eval() {
    local label="$1" config="$2" out_dir="$3"
    local model_id safe_model
    model_id="$(jq -r '.providers["mlx-tinker"].models[0].id' "$config")"
    safe_model="$(python3 - "$model_id" <<'PY'
import re, sys
model = sys.argv[1]
print(re.sub(r'[^a-zA-Z0-9.\-_]', '_', model.rsplit('/', 1)[-1]))
PY
)"
    echo ""
    echo "============================================================"
    echo "  EVAL: $label"
    echo "============================================================"
    for cat in $CATEGORIES; do
        local cat_dir="$WCB_DIR/tasks/$cat"
        local log_path="$out_dir/${cat}.log"
        [[ -d "$cat_dir" ]] || continue
        local -a task_files=()
        while IFS= read -r task_file; do
            task_files+=("$task_file")
        done < <(find "$cat_dir" -maxdepth 1 -name '*task_*.md' | sort -V)
        local n_tasks="${#task_files[@]}"
        echo ""
        echo "--- $cat ($n_tasks tasks) ---"
        if [[ ${#task_files[@]} -eq 0 ]]; then
            echo "  No tasks found, skipping"
            continue
        fi
        touch "$log_path"
        if [[ "$RESUME" != "1" ]]; then
            echo "  Running category via run_batch --parallel $EVAL_PARALLEL" | tee -a "$log_path"
            (
                cd "$WCB_DIR"
                python3 eval/run_batch.py \
                    --category "$cat" \
                    --parallel "$EVAL_PARALLEL" \
                    --model "mlx-tinker/${model_id}" \
                    --models-config "$config"
            ) >> "$log_path" 2>&1 || true
            echo "  Done (log: $log_path)"
            continue
        fi
        for task_file in "${task_files[@]}"; do
            local task_name existing
            task_name="$(basename "$task_file" .md)"
            existing=""
            if [[ "$RESUME" == "1" ]]; then
                existing="$(find "$WCB_DIR/output/$cat/$task_name" -maxdepth 2 -type f -path "*/${safe_model}_*/score.json" 2>/dev/null | sort | tail -1 || true)"
            fi
            if [[ -n "$existing" ]]; then
                echo "  Skipping $task_name (resume found $existing)" | tee -a "$log_path"
                continue
            fi
            echo "  Running $task_name" | tee -a "$log_path"
            (
                cd "$WCB_DIR"
                python3 eval/run_batch.py \
                    --task "$task_file" \
                    --model "mlx-tinker/${model_id}" \
                    --models-config "$config"
            ) >> "$log_path" 2>&1 || true
        done
        echo "  Done (log: $log_path)"
    done
}

summarize_phase() {
    local phase="$1"
    local phase_dir="$RUN_DIR/$phase"
    local found_any=0
    [[ -d "$phase_dir" ]] || return 0

    while IFS= read -r logf; do
        local cat_name score_paths
        cat_name="$(basename "$logf" .log)"
        score_paths="$(python3 - "$logf" <<'PY'
import re, sys
from pathlib import Path

log_path = Path(sys.argv[1])
pattern = re.compile(r'Grading results written to .+?(/.+?/score\.json)')
seen = set()

for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
    match = pattern.search(line)
    if match:
        path = match.group(1)
        if path not in seen:
            seen.add(path)
            print(path)
PY
)"
        [[ -n "$score_paths" ]] || continue
        found_any=1
        SCORE_PATHS="$score_paths" python3 - "$cat_name" <<'PY'
import json, sys
import os
from pathlib import Path

cat_name = sys.argv[1]
paths = [Path(line.strip()) for line in os.environ["SCORE_PATHS"].splitlines() if line.strip()]
scores = []
scored = 0

for path in paths:
    if not path.exists():
        continue
    try:
        score = float(json.loads(path.read_text()).get("overall_score", 0) or 0)
    except Exception:
        continue
    scores.append(score)
    if score != 0:
        scored += 1

if scores:
    avg = round(sum(scores) / len(scores), 4)
    print(f"  {cat_name}: avg={avg} scored={scored}/{len(scores)}")
PY
    done < <(find "$phase_dir" -name "*.log" | sort)

    if [[ "$found_any" == "0" ]]; then
        echo "  No scored tasks recorded yet"
    fi
}

echo "=== WCB Full RL Experiment ==="
echo "  Model: $MODEL"
echo "  Run ID: $RUN_ID"
echo "  Tasks file: $TASKS_FILE ($(wc -l < "$TASKS_FILE") tasks)"
echo "  Categories: $CATEGORIES"
echo "  Resume mode: $RESUME"
echo "  Eval parallelism: $EVAL_PARALLEL"
echo "  OpenClaw-RL: $OPENCLAW_RL_REPO_URL#$OPENCLAW_RL_REF ($OPENCLAW_RL_COMMIT)"
echo ""

# ── Phase 1: Baseline eval ─────────────────────────────────────────────
echo "=== Phase 1: Baseline Eval ==="
start_mlx
run_eval "BEFORE (base $MODEL)" "$EVAL_BASE_CONFIG" "$RUN_DIR/eval_before"

# Stop mlx-tinker between phases to free memory
if [[ -n "${MLX_PID:-}" ]] && kill -0 "$MLX_PID" 2>/dev/null; then
    kill "$MLX_PID" 2>/dev/null; wait "$MLX_PID" 2>/dev/null || true; MLX_PID=""
    sleep 2
fi

# ── Phase 2: RL Training ───────────────────────────────────────────────
echo ""
echo "=== Phase 2: RL Training ==="
OPENCLAW_RUN_ID="${RUN_ID}_train" \
OPENCLAW_RUN_DIR="$RUN_DIR/train" \
MLX_TINKER_MODEL="$MODEL" \
MLX_TINKER_PORT="$PORT" \
MLX_TINKER_MAX_WAIT="$MAX_WAIT" \
bash scripts/run_wcb_rl.sh \
    --tasks-file "$TASKS_FILE" \
    --method rl \
    --max-steps "$MAX_STEPS" \
    --batch-size 1 \
    --loss-fn ppo \
    --max-context-tokens 4096 \
    2>&1 | tee "$LOG_DIR/train.log"

# Copy the final checkpoint to the main checkpoints dir for eval
TRAIN_CKPT_DIR="$RUN_DIR/train/checkpoints"
if [[ -d "$TRAIN_CKPT_DIR" ]]; then
    # Extract LoRA-only weights for inference
    LAST_CKPT=$(ls -d "$TRAIN_CKPT_DIR"/step_* 2>/dev/null | sort | tail -1)
    if [[ -n "$LAST_CKPT" ]]; then
        STEP_NAME=$(basename "$LAST_CKPT")
        LORA_DIR="$CKPT_DIR/${STEP_NAME}_lora"
        echo "  Extracting LoRA weights: $LAST_CKPT → $LORA_DIR"
        uv run python -c "
import os, json
from safetensors.torch import load_file, save_file
src = '$LAST_CKPT/model.safetensors'
dst = '$LORA_DIR'
os.makedirs(dst, exist_ok=True)
w = load_file(src)
lora = {k: v for k, v in w.items() if 'lora' in k.lower()}
save_file(lora, os.path.join(dst, 'model.safetensors'))
json.dump({'step': '$STEP_NAME'}, open(os.path.join(dst, 'metadata.json'), 'w'))
print(f'Extracted {len(lora)} LoRA keys to {dst}')
"
        # Update the RL eval config with the correct checkpoint name
        LORA_NAME="${STEP_NAME}_lora"
        cat > "$EVAL_RL_CONFIG" <<RLJSON
{
  "providers": {
    "mlx-tinker": {
      "baseUrl": "http://host.docker.internal:${PORT}/v1",
      "apiKey": "\${MY_PROXY_API_KEY}",
      "api": "openai-completions",
      "models": [
        {
          "id": "$MODEL:$LORA_NAME",
          "name": "$MODEL (RL $STEP_NAME)"
        }
      ]
    }
  }
}
RLJSON
        echo "  Eval config updated: $MODEL:$LORA_NAME"
    fi
fi

# ── Phase 3: Post-train eval ───────────────────────────────────────────
echo ""
echo "=== Phase 3: Post-train Eval ==="
start_mlx
run_eval "AFTER (RL $MODEL)" "$EVAL_RL_CONFIG" "$RUN_DIR/eval_after"

# ── Summary ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  EXPERIMENT COMPLETE"
echo "============================================================"
echo "  Run dir: $RUN_DIR"
echo ""
echo "  Collecting scores..."

for phase in eval_before eval_after; do
    echo ""
    echo "--- $phase ---"
    summarize_phase "$phase"
done
