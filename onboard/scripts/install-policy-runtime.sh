#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/onboard/.policy-venv"
MODEL_PATH="${REPO_ROOT}/onboard/models/parallel-walking-v22-low-speed-residual-crawl/model_500.onnx"
MODEL_METADATA="${REPO_ROOT}/onboard/models/parallel-walking-v22-low-speed-residual-crawl/model_500.json"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" || ! -f "${MODEL_METADATA}" ]]; then
  echo "The onboard model is missing from this checkout." >&2
  echo "For a sparse checkout, run: git sparse-checkout add onboard" >&2
  exit 1
fi
if head -n 1 "${MODEL_PATH}" | grep -q '^version https://git-lfs'; then
  if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Installing Git LFS so the ONNX model can be downloaded."
    sudo apt-get update
    sudo apt-get install -y git-lfs
  fi
  git -C "${REPO_ROOT}" lfs install
  git -C "${REPO_ROOT}" lfs pull \
    --include="onboard/models/parallel-walking-v22-low-speed-residual-crawl/model_500.onnx"
fi
python3 - "${MODEL_PATH}" "${MODEL_METADATA}" <<'PY'
import hashlib
import json
import pathlib
import sys

model = pathlib.Path(sys.argv[1])
metadata = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
actual = hashlib.sha256(model.read_bytes()).hexdigest()
expected = metadata["onnx_sha256"]
if actual != expected:
    raise SystemExit(f"ONNX SHA-256 mismatch: expected {expected}, got {actual}")
print(f"Verified ONNX model: {actual}")
PY

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
if [[ -f /boot/firmware/config.txt ]] \
  && ! grep -qE '^[[:space:]]*dtoverlay=i2c-gpio,.*bus=8([,[:space:]]|$)' \
    /boot/firmware/config.txt; then
  echo "WARNING: the BNO085 software I2C bus is not configured."
  echo "Before using the real IMU, run:"
  echo "  bash ${REPO_ROOT}/onboard/scripts/configure-bno085-i2c.sh"
  echo "  sudo reboot"
fi
echo "Test without hardware input:"
echo "  bash ${REPO_ROOT}/onboard/scripts/run-policy-print.sh --imu level --duration-s 5"
echo "Read the BNO085 and print targets:"
echo "  bash ${REPO_ROOT}/onboard/scripts/run-policy-print.sh --imu bno085"
echo "Start the phone/computer dashboard:"
echo "  bash ${REPO_ROOT}/onboard/scripts/run-policy-web.sh --imu bno085"
