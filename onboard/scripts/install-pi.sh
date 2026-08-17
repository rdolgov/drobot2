#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
VENV_DIR="${REPO_ROOT}/onboard/.venv"
WORKSPACE_DIR="${REPO_ROOT}/onboard/ros2_ws"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS 2 ${ROS_DISTRO_NAME} was not found at ${ROS_SETUP}." >&2
  echo "Install ROS 2 first, then rerun this script." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "Python 3.11 or newer is required; found ${PYTHON_VERSION}." >&2
  exit 1
fi

source "${ROS_SETUP}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel colcon-common-extensions
python -m pip install --editable "${REPO_ROOT}/hardware/test-apps/one-leg-testbed"
python -m pip install --editable "${REPO_ROOT}/hardware/test-apps/four-leg-dashboard"

colcon --log-base "${WORKSPACE_DIR}/log" build \
  --symlink-install \
  --base-paths "${WORKSPACE_DIR}/src" \
  --build-base "${WORKSPACE_DIR}/build" \
  --install-base "${WORKSPACE_DIR}/install"

echo
echo "Drobot onboard package installed."
echo "Start in demo mode:"
echo "  bash ${REPO_ROOT}/onboard/scripts/start-onboard.sh --demo"
echo "Start with hardware:"
echo "  DROBOT_SERIAL_PORT=/dev/ttyUSB0 bash ${REPO_ROOT}/onboard/scripts/start-onboard.sh"
