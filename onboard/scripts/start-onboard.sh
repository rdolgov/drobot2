#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
VENV_ACTIVATE="${REPO_ROOT}/onboard/.venv/bin/activate"
WORKSPACE_SETUP="${REPO_ROOT}/onboard/ros2_ws/install/setup.bash"
MANIFEST_PATH="${DROBOT_MANIFEST:-${REPO_ROOT}/hardware/robot-runtime/four-leg.toml}"
SERIAL_PORT="${DROBOT_SERIAL_PORT:-auto}"
HTTP_BIND="${DROBOT_HTTP_BIND:-0.0.0.0}"
HTTP_PORT="${DROBOT_HTTP_PORT:-8080}"
CONTROL_TOKEN="${DROBOT_CONTROL_TOKEN:-}"
DEMO="false"

if [[ "${1:-}" == "--demo" ]]; then
  DEMO="true"
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--demo]" >&2
  exit 2
fi

for required_file in "${ROS_SETUP}" "${VENV_ACTIVATE}" "${WORKSPACE_SETUP}" "${MANIFEST_PATH}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file is missing: ${required_file}" >&2
    echo "Run onboard/scripts/install-pi.sh first." >&2
    exit 1
  fi
done

source "${ROS_SETUP}"
source "${VENV_ACTIVATE}"
source "${WORKSPACE_SETUP}"

LAUNCH_ARGS=(
  "manifest:=${MANIFEST_PATH}"
  "serial_port:=${SERIAL_PORT}"
  "http_bind:=${HTTP_BIND}"
  "http_port:=${HTTP_PORT}"
  "demo:=${DEMO}"
)
if [[ -n "${CONTROL_TOKEN}" ]]; then
  LAUNCH_ARGS+=("control_token:=${CONTROL_TOKEN}")
fi

exec ros2 launch drobot_onboard onboard.launch.py "${LAUNCH_ARGS[@]}"
