#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/onboard/.policy-venv"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

PYTHON_TAG="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
MACHINE="$(uname -m)"
if [[ "${PYTHON_TAG}" == "cp314" && "${MACHINE}" == "aarch64" ]]; then
  python -m pip install \
    "https://github.com/adafruit/lgpio-python-wheels/raw/main/wheels/lgpio-0.2.2.0-cp314-cp314-linux_aarch64.whl"
fi

python -m pip install --editable "${REPO_ROOT}/onboard/policy-runtime[bno085]"

echo
echo "Drobot print-only policy runtime installed."
echo "Test without hardware input:"
echo "  bash ${REPO_ROOT}/onboard/scripts/run-policy-print.sh --imu level --duration-s 5"
echo "Read the BNO085 and print targets:"
echo "  bash ${REPO_ROOT}/onboard/scripts/run-policy-print.sh --imu bno085"

