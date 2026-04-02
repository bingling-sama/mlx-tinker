#!/usr/bin/env bash
set -euo pipefail

if ! command -v podman >/dev/null 2>&1; then
  echo "podman is required" >&2
  exit 1
fi

if ! podman machine list --format json >/dev/null 2>&1; then
  echo "Unable to query podman machine state" >&2
  exit 1
fi

MACHINE_NAME="${PODMAN_MACHINE_NAME:-podman-machine-default}"
MACHINE_LIST_JSON="$(podman machine list --format json)"
MACHINE_CONFIG_DIR="${HOME}/.config/containers/podman/machine/applehv"
MACHINE_IGN_PATH="${MACHINE_CONFIG_DIR}/${MACHINE_NAME}.ign"

repair_ignition_if_needed() {
  if [[ ! -f "${MACHINE_IGN_PATH}" ]]; then
    return 0
  fi

  python3 - "${MACHINE_IGN_PATH}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
ignition = data.get("ignition", {})
config = ignition.get("config", {})
replace = config.get("replace")

if isinstance(replace, dict) and not replace.get("source"):
    config.pop("replace", None)
    if not config:
        ignition.pop("config", None)
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Repaired malformed ignition config at {path}")
PY
}

if ! python3 -c 'import json, sys; machine = sys.argv[1]; machines = json.loads(sys.argv[2]); raise SystemExit(0 if any(item.get("Name") == machine for item in machines) else 1)' \
  "$MACHINE_NAME" "$MACHINE_LIST_JSON"; then
  echo "Initializing Podman machine: ${MACHINE_NAME}"
  podman machine init "${MACHINE_NAME}"
fi

repair_ignition_if_needed

if ! podman info >/dev/null 2>&1; then
  echo "Starting Podman machine: ${MACHINE_NAME}"
  podman machine start "${MACHINE_NAME}"
fi

for _ in $(seq 1 60); do
  if podman info >/dev/null 2>&1; then
    echo "Podman machine is ready."
    exit 0
  fi

  sleep 2
done

echo "Podman machine did not become ready in time." >&2
podman machine inspect "${MACHINE_NAME}" >&2 || true
exit 1
