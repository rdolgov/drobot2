#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SERVICE_TEMPLATE="${REPO_ROOT}/onboard/systemd/drobot-manual-web.service"
SERVICE_TARGET="/etc/systemd/system/drobot-manual-web.service"
ENV_TARGET="/etc/default/drobot-manual-web"
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
if [[ ! -x "${REPO_ROOT}/onboard/.manual-venv/bin/drobot-four-leg-web" ]]; then
  echo "Run onboard/scripts/install-manual-runtime.sh first." >&2
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
  sudo install -m 0600 \
    "${REPO_ROOT}/onboard/systemd/drobot-manual-web.env.example" \
    "${ENV_TARGET}"
fi
sudo usermod -aG dialout "${SERVICE_USER}"
sudo systemctl daemon-reload
sudo systemctl enable drobot-manual-web.service
if [[ "${START_NOW}" == "true" ]]; then
  sudo systemctl restart drobot-manual-web.service
fi

echo "Installed ${SERVICE_TARGET} for ${SERVICE_USER}."
echo "Runtime settings: ${ENV_TARGET}"
echo "Open http://$(hostname).local:8080/."
echo "The tracked default is demo mode with no servo output."
if [[ "${START_NOW}" != "true" ]]; then
  echo "Start it with: sudo systemctl start drobot-manual-web"
fi
