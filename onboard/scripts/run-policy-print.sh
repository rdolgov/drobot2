#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/onboard/.policy-venv"
MODEL_PATH="${REPO_ROOT}/onboard/models/parallel-walking-v23-higher-speed-straight-residual-crawl/model_1500.onnx"

if [[ ! -x "${VENV_DIR}/bin/drobot-policy-print" ]]; then
  echo "Policy runtime is not installed. Run:" >&2
  echo "  bash ${REPO_ROOT}/onboard/scripts/install-policy-runtime.sh" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Policy model is missing: ${MODEL_PATH}" >&2
  exit 1
fi

exec "${VENV_DIR}/bin/drobot-policy-print" --model "${MODEL_PATH}" "$@"
