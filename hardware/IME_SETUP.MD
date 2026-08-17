# BNO085 IMU Setup on Raspberry Pi 5

This guide records the working BNO085 setup on `pi5-dog` so it can be repeated after an OS or environment rebuild.

## Tested configuration

- Raspberry Pi 5, 64-bit (`aarch64`)
- Ubuntu Raspberry Pi kernel
- Python 3.14 virtual environment: `~/bno085-env`
- Adafruit Blinka 9.2.0
- Adafruit CircuitPython BNO08x 1.3.3
- `lgpio` 0.2.2.0
- BNO085 I2C address: `0x4A`

## Wiring

| BNO085 | Raspberry Pi 5 |
| --- | --- |
| VIN | 3.3 V |
| GND | GND |
| SDA | GPIO 2, physical pin 3 |
| SCL | GPIO 3, physical pin 5 |

## Enable I2C and install system tools

```bash
sudo raspi-config nonint do_i2c 0
sudo apt update
sudo apt install -y i2c-tools python3-venv
```

## Create the virtual environment

```bash
python3 -m venv ~/bno085-env
source ~/bno085-env/bin/activate
python -m pip install --upgrade pip
python -m pip install adafruit-blinka adafruit-circuitpython-bno08x
```

Activate it again in future shells with:

```bash
source ~/bno085-env/bin/activate
```

## Python 3.14 `lgpio` installation

On 64-bit Raspberry Pi OS/Ubuntu, PyPI may try and fail to build `adafruit-lgpio` under Python 3.14. Install Adafruit's prebuilt CPython 3.14 ARM64 wheel instead:

```bash
python -m pip install https://github.com/adafruit/lgpio-python-wheels/raw/main/wheels/lgpio-0.2.2.0-cp314-cp314-linux_aarch64.whl
```

Do not use this exact wheel on a non-`aarch64` system or with a Python version other than 3.14. Check first with:

```bash
uname -m
python --version
```

## Device permissions

Blinka needs access to both `/dev/gpiochip0` and `/dev/i2c-1`. Add the login user to the device groups:

```bash
sudo /usr/sbin/usermod -aG dialout,i2c "$USER"
```

Log out completely and reconnect so the new group membership takes effect. A reboot is not required. Verify after reconnecting:

```bash
id
```

The output must include both `dialout` and `i2c`.

## Verify the hardware

Check that the sensor appears at `0x4A`:

```bash
i2cdetect -y 1
```

Test Blinka and scan the bus using short commands that are safe to paste into a terminal:

```bash
python -c "import board; print('GPIO access OK')"
python -c "import board,busio; i=busio.I2C(board.SCL,board.SDA); print(i.scan())"
```

The scan should print `[74]`; decimal 74 is hexadecimal `0x4A`.

## Read the BNO085

Save the following as `bno085_test.py` or run it from a Python editor:

```python
import time

import board
import busio
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GAME_ROTATION_VECTOR,
    BNO_REPORT_GYROSCOPE,
)
from adafruit_bno08x.i2c import BNO08X_I2C

i2c = busio.I2C(board.SCL, board.SDA)
imu = BNO08X_I2C(i2c, address=0x4A)

imu.enable_feature(BNO_REPORT_ACCELEROMETER)
imu.enable_feature(BNO_REPORT_GYROSCOPE)
imu.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)

time.sleep(1)
print("BNO085 connected. Press Ctrl+C to stop.")

while True:
    print(
        "accel:", imu.acceleration,
        " gyro:", imu.gyro,
        " quaternion:", imu.game_quaternion,
    )
    time.sleep(0.2)
```

Run it with:

```bash
source ~/bno085-env/bin/activate
python bno085_test.py
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'lgpio'`

Install the matching `lgpio` wheel described above.

### `lgpio.error: 'can not open gpiochip'`

Check `id`. The current login session must include `dialout`. Also check:

```bash
ls -l /dev/gpiochip0
```

### Permission error opening `/dev/i2c-1`

Check `id`. The current login session must include `i2c`. Also check:

```bash
ls -l /dev/i2c-1
```

### Sensor not found

Run `i2cdetect -y 1`, recheck power and wiring, and confirm whether the board uses address `0x4A` or `0x4B`. Update the `address=` argument if necessary.

### I2C frequency warning

Blinka may warn that I2C frequency is not settable from Python. This is benign on Linux; the working code leaves out the `frequency=400000` argument.
