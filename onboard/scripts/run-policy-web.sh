#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/onboard/.policy-venv"
MODEL_PATH="${REPO_ROOT}/onboard/models/parallel-walking-v30-symmetry-gated-robust-straight-crawl/model_5000.onnx"
HTTP_BIND="${DROBOT_POLICY_BIND:-0.0.0.0}"
HTTP_PORT="${DROBOT_POLICY_PORT:-8090}"
CONTROL_TOKEN="${DROBOT_POLICY_TOKEN:-}"
IMU_BACKEND="${DROBOT_POLICY_IMU:-bno085}"
IMU_AXIS_MAP="${DROBOT_POLICY_IMU_AXIS_MAP:-+x,+y,+z}"

if [[ ! -x "${VENV_DIR}/bin/drobot-policy-web" ]]; then
  echo "Policy runtime is not installed. Run:" >&2
  echo "  bash ${REPO_ROOT}/onboard/scripts/install-policy-runtime.sh" >&2
  exit 1
fi

ARGS=(
  --model "${MODEL_PATH}"
  --imu "${IMU_BACKEND}"
  --imu-axis-map "${IMU_AXIS_MAP}"
  --bind "${HTTP_BIND}"
  --port "${HTTP_PORT}"
)
if [[ -n "${CONTROL_TOKEN}" ]]; then
  ARGS+=(--control-token "${CONTROL_TOKEN}")
fi

exec "${VENV_DIR}/bin/drobot-policy-web" "${ARGS[@]}" "$@"
