#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/onboard/.manual-venv"

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

python -m pip install --editable "${REPO_ROOT}/hardware/test-apps/one-leg-testbed"
python -m pip install --editable "${REPO_ROOT}/onboard/policy-runtime[bno085]"
python -m pip install --editable "${REPO_ROOT}/hardware/test-apps/four-leg-dashboard"

echo
echo "Drobot manual/IK dashboard runtime installed."
echo "Safe demo mode:"
echo "  DROBOT_MANUAL_DEMO=true bash ${REPO_ROOT}/onboard/scripts/run-manual-web.sh"
echo "Hardware mode after connecting and checking the servo adapter:"
echo "  DROBOT_MANUAL_DEMO=false DROBOT_MANUAL_SERIAL_PORT=/dev/ttyUSB0 bash ${REPO_ROOT}/onboard/scripts/run-manual-web.sh"
