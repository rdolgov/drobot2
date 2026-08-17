#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SERVICE_TEMPLATE="${REPO_ROOT}/onboard/systemd/drobot-onboard.service"
SERVICE_TARGET="/etc/systemd/system/drobot-onboard.service"
ENV_TEMPLATE="${REPO_ROOT}/onboard/systemd/drobot-onboard.env.example"
ENV_TARGET="/etc/default/drobot-onboard"
SERVICE_USER="${SUDO_USER:-${USER}}"
START_NOW="false"

if [[ "${1:-}" == "--start" ]]; then
  START_NOW="true"
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--start]" >&2
  exit 2
fi

if [[ ! -f "${SERVICE_TEMPLATE}" ]]; then
  echo "Service template not found: ${SERVICE_TEMPLATE}" >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/onboard/scripts/start-onboard.sh" ]]; then
  echo "start-onboard.sh is missing." >&2
  exit 1
fi

TEMP_SERVICE="$(mktemp)"
trap 'rm -f -- "${TEMP_SERVICE}"' EXIT
sed \
  -e "s|__DROBOT_USER__|${SERVICE_USER}|g" \
  -e "s|__DROBOT_REPO_ROOT__|${REPO_ROOT}|g" \
  "${SERVICE_TEMPLATE}" > "${TEMP_SERVICE}"

sudo install -m 0644 "${TEMP_SERVICE}" "${SERVICE_TARGET}"
if [[ ! -f "${ENV_TARGET}" ]]; then
  sudo install -m 0600 "${ENV_TEMPLATE}" "${ENV_TARGET}"
fi
sudo usermod -aG dialout "${SERVICE_USER}"
sudo systemctl daemon-reload
sudo systemctl enable drobot-onboard.service

if [[ "${START_NOW}" == "true" ]]; then
  sudo systemctl start drobot-onboard.service
fi

echo "Installed ${SERVICE_TARGET} for user ${SERVICE_USER}."
echo "Runtime overrides: ${ENV_TARGET}"
echo "Log out and back in if dialout membership was newly added."
if [[ "${START_NOW}" != "true" ]]; then
  echo "The service was enabled but not started."
  echo "Start it with: sudo systemctl start drobot-onboard"
fi
