#!/usr/bin/env bash
set -euo pipefail

BOOT_CONFIG="${DROBOT_BOOT_CONFIG:-/boot/firmware/config.txt}"
SOFTWARE_BUS="${DROBOT_BNO085_I2C_BUS:-8}"
GPIO_DELAY_US="${DROBOT_BNO085_I2C_DELAY_US:-2}"
SERVICE_ENV="${DROBOT_MANUAL_SERVICE_ENV:-/etc/default/drobot-manual-web}"

if [[ ! -f "${BOOT_CONFIG}" ]]; then
  echo "Raspberry Pi boot config was not found at ${BOOT_CONFIG}." >&2
  exit 1
fi
for value in "${SOFTWARE_BUS}" "${GPIO_DELAY_US}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "Software-bus and delay values must be non-negative integers." >&2
    exit 2
  fi
done
if (( SOFTWARE_BUS == 0 || GPIO_DELAY_US == 0 )); then
  echo "Software-bus and delay values must be greater than zero." >&2
  exit 2
fi

OVERLAY="dtoverlay=i2c-gpio,bus=${SOFTWARE_BUS},i2c_gpio_sda=2,i2c_gpio_scl=3,i2c_gpio_delay_us=${GPIO_DELAY_US}"
BACKUP_PATH="${BOOT_CONFIG}.drobot-bno085-backup"
sudo cp --preserve=mode,ownership,timestamps "${BOOT_CONFIG}" "${BACKUP_PATH}"

# GPIO 2/3 cannot be owned by hardware and software I2C simultaneously. The
# software controller honors BNO085 clock stretching and keeps the existing
# four-wire connection on physical header pins 3 and 5.
if grep -qE '^[[:space:]]*dtparam=i2c_arm=' "${BOOT_CONFIG}"; then
  sudo sed -i \
    's/^[[:space:]]*dtparam=i2c_arm=.*/dtparam=i2c_arm=off/' \
    "${BOOT_CONFIG}"
else
  printf '\ndtparam=i2c_arm=off\n' | sudo tee -a "${BOOT_CONFIG}" >/dev/null
fi
sudo sed -i '/^[[:space:]]*dtparam=i2c_arm_baudrate=/d' "${BOOT_CONFIG}"
if grep -qE "^[[:space:]]*dtoverlay=i2c-gpio,.*bus=${SOFTWARE_BUS}([,[:space:]]|$)" \
  "${BOOT_CONFIG}"; then
  sudo sed -i \
    "s|^[[:space:]]*dtoverlay=i2c-gpio,.*bus=${SOFTWARE_BUS}.*|${OVERLAY}|" \
    "${BOOT_CONFIG}"
else
  printf '\n# BNO085 clock-stretching-compatible software I2C\n%s\n' "${OVERLAY}" \
    | sudo tee -a "${BOOT_CONFIG}" >/dev/null
fi

if [[ -f "${SERVICE_ENV}" ]]; then
  if grep -qE '^[[:space:]]*DROBOT_BNO085_I2C_BUS=' "${SERVICE_ENV}"; then
    sudo sed -i \
      "s/^[[:space:]]*DROBOT_BNO085_I2C_BUS=.*/DROBOT_BNO085_I2C_BUS=${SOFTWARE_BUS}/" \
      "${SERVICE_ENV}"
  else
    printf '\nDROBOT_BNO085_I2C_BUS=%s\n' "${SOFTWARE_BUS}" \
      | sudo tee -a "${SERVICE_ENV}" >/dev/null
  fi
fi

echo "Configured BNO085 software I2C on GPIO 2/3 as /dev/i2c-${SOFTWARE_BUS}."
echo "Boot-config backup: ${BACKUP_PATH}"
echo "Reboot the Pi before using the IMU: sudo reboot"
