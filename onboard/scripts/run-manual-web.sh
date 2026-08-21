#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/onboard/.manual-venv"
MANIFEST_PATH="${DROBOT_MANUAL_MANIFEST:-${REPO_ROOT}/hardware/robot-runtime/four-leg.toml}"
SERIAL_PORT="${DROBOT_MANUAL_SERIAL_PORT:-auto}"
HTTP_BIND="${DROBOT_MANUAL_BIND:-0.0.0.0}"
HTTP_PORT="${DROBOT_MANUAL_PORT:-8080}"
DEMO_MODE="${DROBOT_MANUAL_DEMO:-true}"

if [[ ! -x "${VENV_DIR}/bin/drobot-four-leg-web" ]]; then
  echo "Manual dashboard runtime is not installed. Run:" >&2
  echo "  bash ${REPO_ROOT}/onboard/scripts/install-manual-runtime.sh" >&2
  exit 1
fi

ARGS=(
  --manifest "${MANIFEST_PATH}"
  --http-bind "${HTTP_BIND}"
  --http-port "${HTTP_PORT}"
  --allow-remote
  --no-browser
)
case "${DEMO_MODE,,}" in
  true|1|yes) ARGS+=(--demo) ;;
  false|0|no) ARGS+=(--port "${SERIAL_PORT}") ;;
  *) echo "DROBOT_MANUAL_DEMO must be true or false." >&2; exit 2 ;;
esac

exec "${VENV_DIR}/bin/drobot-four-leg-web" "${ARGS[@]}" "$@"
