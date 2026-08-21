#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SERVICE_TEMPLATE="${REPO_ROOT}/onboard/systemd/drobot-policy-web.service"
SERVICE_TARGET="/etc/systemd/system/drobot-policy-web.service"
ENV_TARGET="/etc/default/drobot-policy-web"
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
if [[ ! -x "${REPO_ROOT}/onboard/.policy-venv/bin/drobot-policy-web" ]]; then
  echo "Run onboard/scripts/install-policy-runtime.sh first." >&2
  exit 1
fi

TEMP_SERVICE="$(mktemp)"
TEMP_ENV="$(mktemp)"
trap 'rm -f -- "${TEMP_SERVICE}" "${TEMP_ENV}"' EXIT
sed \
  -e "s|__DROBOT_USER__|${SERVICE_USER}|g" \
  -e "s|__DROBOT_REPO_ROOT__|${REPO_ROOT}|g" \
  "${SERVICE_TEMPLATE}" > "${TEMP_SERVICE}"

if [[ ! -f "${ENV_TARGET}" ]]; then
  cp "${REPO_ROOT}/onboard/systemd/drobot-policy-web.env.example" "${TEMP_ENV}"
  sudo install -m 0600 "${TEMP_ENV}" "${ENV_TARGET}"
elif sudo grep -q '^DROBOT_POLICY_TOKEN=' "${ENV_TARGET}"; then
  sudo grep -v '^DROBOT_POLICY_TOKEN=' "${ENV_TARGET}" > "${TEMP_ENV}" || true
  sudo install -m 0600 "${TEMP_ENV}" "${ENV_TARGET}"
fi
sudo install -m 0644 "${TEMP_SERVICE}" "${SERVICE_TARGET}"
sudo usermod -aG i2c "${SERVICE_USER}"
sudo systemctl daemon-reload
sudo systemctl enable drobot-policy-web.service
if [[ "${START_NOW}" == "true" ]]; then
  sudo systemctl restart drobot-policy-web.service
fi

echo "Installed ${SERVICE_TARGET} for ${SERVICE_USER}."
echo "Runtime settings: ${ENV_TARGET}"
echo "Open http://$(hostname).local:8090/."
if [[ "${START_NOW}" != "true" ]]; then
  echo "Start it with: sudo systemctl start drobot-policy-web"
fi
