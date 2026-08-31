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
FALLBACK_DEMO="${DROBOT_MANUAL_FALLBACK_DEMO:-false}"
RL_MODEL="${DROBOT_RL_MODEL:-${REPO_ROOT}/onboard/models/parallel-walking-v24-padded-feet-forward-bias/model_3248.onnx}"
RL_IMU_AXIS_MAP="${DROBOT_RL_IMU_AXIS_MAP:-+x,+y,+z}"

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
  --rl-model "${RL_MODEL}"
  --rl-imu-axis-map "${RL_IMU_AXIS_MAP}"
)
case "${DEMO_MODE,,}" in
  true|1|yes) ARGS+=(--demo) ;;
  false|0|no)
    case "${FALLBACK_DEMO,,}" in
      true|1|yes)
        if [[ "${SERIAL_PORT}" == "auto" ]]; then
          if ! compgen -G '/dev/ttyUSB*' >/dev/null \
            && ! compgen -G '/dev/ttyACM*' >/dev/null; then
            echo "Servo adapter is missing; keeping port ${HTTP_PORT} available in demo mode." >&2
            ARGS+=(--demo)
          else
            ARGS+=(--port "${SERIAL_PORT}")
          fi
        elif [[ ! -e "${SERIAL_PORT}" ]]; then
          echo "Servo adapter ${SERIAL_PORT} is missing; keeping port ${HTTP_PORT} available in demo mode." >&2
          ARGS+=(--demo)
        else
          ARGS+=(--port "${SERIAL_PORT}")
        fi
        ;;
      false|0|no) ARGS+=(--port "${SERIAL_PORT}") ;;
      *) echo "DROBOT_MANUAL_FALLBACK_DEMO must be true or false." >&2; exit 2 ;;
    esac
    ;;
  *) echo "DROBOT_MANUAL_DEMO must be true or false." >&2; exit 2 ;;
esac

exec "${VENV_DIR}/bin/drobot-four-leg-web" "${ARGS[@]}" "$@"
