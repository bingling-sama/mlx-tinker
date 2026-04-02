#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OPENCLAW_RL_DIR="${OPENCLAW_RL_DIR:-$PROJECT_DIR/.external/openclaw-rl}"
OPENCLAW_RL_REPO_URL="${OPENCLAW_RL_REPO_URL:-https://github.com/ojus1/OpenClaw-RL.git}"
OPENCLAW_RL_REF="${OPENCLAW_RL_REF:-codex/qwen35-openclaw-tinker}"

mkdir -p "$(dirname "$OPENCLAW_RL_DIR")"

echo "=== OpenClaw-RL Dependency ==="
echo "  Dir:    $OPENCLAW_RL_DIR"
echo "  Repo:   $OPENCLAW_RL_REPO_URL"
echo "  Branch: $OPENCLAW_RL_REF"

if [[ ! -d "$OPENCLAW_RL_DIR/.git" ]]; then
    echo "  Cloning fork branch..."
    git clone --branch "$OPENCLAW_RL_REF" --single-branch "$OPENCLAW_RL_REPO_URL" "$OPENCLAW_RL_DIR"
else
    if [[ -n "$(git -C "$OPENCLAW_RL_DIR" status --porcelain)" ]]; then
        echo "  ERROR: existing OpenClaw-RL checkout is dirty: $OPENCLAW_RL_DIR" >&2
        echo "  Commit, stash, or reset changes before continuing." >&2
        exit 1
    fi

    echo "  Fetching fork branch..."
    git -C "$OPENCLAW_RL_DIR" fetch "$OPENCLAW_RL_REPO_URL" "$OPENCLAW_RL_REF"

    fetched_sha="$(git -C "$OPENCLAW_RL_DIR" rev-parse FETCH_HEAD)"
    current_sha="$(git -C "$OPENCLAW_RL_DIR" rev-parse HEAD)"

    if [[ "$current_sha" == "$fetched_sha" ]]; then
        :
    elif git -C "$OPENCLAW_RL_DIR" merge-base --is-ancestor "$current_sha" "$fetched_sha"; then
        git -C "$OPENCLAW_RL_DIR" checkout -B "$OPENCLAW_RL_REF" "$fetched_sha" >/dev/null
    else
        echo "  ERROR: existing OpenClaw-RL checkout diverges from $OPENCLAW_RL_REPO_URL#$OPENCLAW_RL_REF" >&2
        echo "  Refusing to overwrite clean but divergent history." >&2
        exit 1
    fi
fi

OPENCLAW_RL_COMMIT="$(git -C "$OPENCLAW_RL_DIR" rev-parse HEAD)"
echo "  Commit: $OPENCLAW_RL_COMMIT"
