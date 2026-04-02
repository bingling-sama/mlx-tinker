#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_REPO_DIR="${OPENCLAW_REPO_DIR:-$ROOT_DIR/.external/openclaw}"
OPENCLAW_CONFIG_DIR="${OPENCLAW_CONFIG_DIR:-$HOME/.openclaw}"
OPENCLAW_WORKSPACE_DIR="${OPENCLAW_WORKSPACE_DIR:-$OPENCLAW_CONFIG_DIR/workspace}"
OPENCLAW_MODELS_FILE="${OPENCLAW_MODELS_FILE:-$OPENCLAW_CONFIG_DIR/models.local.json}"

"$ROOT_DIR/scripts/setup_podman_machine.sh"

mkdir -p "$OPENCLAW_CONFIG_DIR" "$OPENCLAW_WORKSPACE_DIR" "$(dirname "$OPENCLAW_REPO_DIR")"

if [[ ! -d "$OPENCLAW_REPO_DIR/.git" ]]; then
  git clone --depth=1 https://github.com/openclaw/openclaw "$OPENCLAW_REPO_DIR"
else
  git -C "$OPENCLAW_REPO_DIR" pull --ff-only
fi

python3 "$ROOT_DIR/scripts/render_openclaw_provider_config.py" --runtime podman >"$OPENCLAW_MODELS_FILE"

echo
echo "Prepared OpenClaw Podman checkout at: $OPENCLAW_REPO_DIR"
echo "Rendered local provider config at:   $OPENCLAW_MODELS_FILE"
echo
echo "Next commands:"
echo "  cd \"$OPENCLAW_REPO_DIR\""
echo "  ./scripts/podman/setup.sh --container"
echo "  OPENCLAW_PODMAN_ENV=\"$OPENCLAW_REPO_DIR/openclaw.podman.env\" ./scripts/run-openclaw-podman.sh launch"
echo
echo "Then merge the rendered models config into ~/.openclaw/openclaw.json under the top-level 'models' key."
