# Drobot one-leg ST3215 testbed

This folder is a standalone Python test bench for one Drobot leg containing
three Feetech ST3215 bus servos:

| Testbed number | Joint | Default bus ID |
| ---: | --- | ---: |
| 1 | hip abduction | 1 |
| 2 | hip flexion | 2 |
| 3 | knee | 3 |

It discovers a USB serial adapter, assigns unique IDs with one motor connected
at a time, configures conservative position-control settings, records a
physical neutral pose, reports telemetry, and provides an interactive
selected-motor controller.

The implementation uses the pure-Python
[`ftservo-python-sdk` 2.0.0](https://pypi.org/project/ftservo-python-sdk/)
package from Feetech's
[`FTServo_Python`](https://github.com/ftservo/FTServo_Python) repository. Its
workflow also follows the current LeRobot practices of configuring one isolated
motor at a time, disabling torque for calibration, and using unique IDs on a
shared bus. See the
[LeRobot SO-101 motor setup](https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx)
and
[Feetech motor bus implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/feetech.py).
This testbed does **not** require the full LeRobot package.

## Safety first

The ST3215 can apply enough torque to pinch fingers, break printed parts, or
throw an unsupported leg.

- Clamp or suspend the leg before enabling torque.
- Keep a physical power switch or supply cutoff within reach.
- Do not use the USB connection as the servo power source.
- Use the correct external supply for the **12 V ST3215 variant** and connect
  the adapter and servo power grounds as required by the adapter.
- Turn external servo power **off before connecting, disconnecting, or
  daisy-chaining motors**.
- During ID assignment, connect exactly one motor. New motors commonly share
  ID 1; connecting several factory-default motors creates an ID collision.
- Start without a payload and with the provided 30% torque limit.
- Treat `Ctrl+C`/`quit` as a software stop only. A crashed process or broken
  USB cable cannot replace a physical power cutoff.

The software begins with torque off. Arming a motor first commands its current
measured position and only then enables torque, reducing the chance of a sudden
jump. Each subsequent target change is limited to 5 degrees by default.

## Hardware assumptions

- Three Feetech ST3215 serial bus servos using the SMS/STS protocol.
- One compatible half-duplex Feetech/Waveshare bus-servo USB adapter.
- External servo power appropriate for the exact ST3215 voltage variant.
- A data/power daisy chain wired according to the adapter and servo manuals.
- Python 3.11 or newer.

The code does not silently support STS3212, ST3215-HS, PWM servos, or a mixed
bus of different protocol families.

## Platform setup guides

- Windows 10/11 PowerShell: [`WINDOWS.md`](WINDOWS.md)
- macOS: continue with the installation section below

## macOS installation

Open Terminal:

```bash
cd hardware/test-apps/one-leg-testbed

python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

The dependency is a platform-independent Python wheel and communicates through
`pyserial`. No Linux-only device API is used.

Connect only the USB adapter and list ports:

```bash
drobot-leg ports
```

Typical macOS names include:

```text
/dev/tty.usbmodem575E0031751
/dev/cu.usbmodem575E0031751
/dev/tty.usbserial-110
/dev/cu.usbserial-110
```

When exactly one likely USB serial adapter is present, `--port auto` selects
it. With multiple adapters, pass the exact port:

```bash
drobot-leg --port /dev/tty.usbmodem575E0031751 ports
```

If no port appears, first check the data-capable USB cable and System
Information > USB. A driver may be required for some third-party USB-to-serial
chipsets; use the driver specified by the adapter manufacturer.

## 1. Create your local configuration

The tracked example remains the reviewed source. Make an ignored local copy:

```bash
cp leg.example.toml leg.toml
```

The example uses:

- 1,000,000 baud;
- 30% (`300/1000`) torque limit;
- speed value 350 (approximately 30.8 degrees per second);
- acceleration value 10;
- maximum 5-degree target change per command;
- deliberately conservative bench-test angle limits.

Use the local file in every command:

```bash
drobot-leg --config leg.toml ports
```

The four assembled robot legs also have tracked, hardware-specific profiles and
Shared helper scripts and verified profiles under
[`../../robot-runtime/`](../../robot-runtime/README.md). Use those profiles
directly when working with the verified ID ranges 1-3, 4-6, 7-9, or 10-12:

```powershell
..\..\robot-runtime\scripts\show-status.ps1 -Leg 4 -Port COM4
..\..\robot-runtime\scripts\start-web-control.ps1 -Leg 4 -Port COM4
```

The ignored root-level files remain useful for scratch experiments. The
tracked `../../robot-runtime/servos/leg-N.toml` and
`../../robot-runtime/servos/calibration-leg-N.json` pairs are the
shared canonical values for the assembled four-leg robot.

Do not widen a joint's limits until the installed horn, printed geometry,
cables, and test fixture have been checked through that motion.

## 2. Assign motor IDs

All three new motors may initially answer as ID 1. Configure them separately:

```bash
drobot-leg --config leg.toml \
  --port /dev/tty.usbmodem575E0031751 \
  setup-ids
```

For each prompted joint:

1. Turn external servo power off.
2. Connect only that one servo.
3. Restore power and press Enter.
4. Type `ASSIGN` when the script names the target ID.
5. Turn external power off again before disconnecting it.

The code disables torque, unlocks the servo configuration, writes the new ID,
verifies the new ID responds, then locks configuration again. It deliberately
does not rewrite baud-rate EEPROM; all motors and the adapter are expected to
use the configured 1,000,000 baud.

To assign one motor manually:

```bash
drobot-leg --config leg.toml --port /dev/tty.usbmodem575E0031751 \
  assign-id --current-id 1 --new-id 2
```

## 3. Assemble and verify the three-motor bus

With power off, connect the three uniquely numbered motors in the one-leg
daisy chain. Restore power, then scan IDs 1 through 3:

```bash
drobot-leg --config leg.toml --port /dev/tty.usbmodem575E0031751 \
  scan --id-start 1 --id-end 3
```

Exactly three responses should appear. Stop if an ID is missing or duplicated.

## 4. Apply conservative motor settings

Support the leg so it cannot hit the bench:

```bash
drobot-leg --config leg.toml --port /dev/tty.usbmodem575E0031751 \
  configure
```

Type `CONFIGURE` after reviewing the warning. This:

- verifies IDs 1-3;
- disables torque;
- selects position mode;
- sets the configured torque limit and acceleration;
- leaves every motor disarmed.

It does not change PID coefficients, protection thresholds, homing offset,
firmware, or baud rate.

## 5. Record the neutral pose

The local calibration file maps zero degrees to the real assembled leg:

```bash
drobot-leg --config leg.toml --port /dev/tty.usbmodem575E0031751 \
  --calibration calibration.json \
  capture-centers
```

The script disables all three motors. Manually place the leg in the intended
neutral pose, support it, and press Enter. The current encoder ticks are saved
to the ignored `calibration.json`. They are not written into the servo's EEPROM.

If a commanded positive direction moves the wrong way, stop, set that motor's
`direction` in `leg.toml` to `-1`, and capture centers again.

### Persistently center a servo near raw 2048

`capture-centers` changes only `calibration.json`; it does not move an encoder
boundary. If a required joint range would cross raw 0/4095, support the
torque-off leg at neutral and use Feetech's persistent middle-position function
on one addressed motor:

```powershell
.\.venv\Scripts\drobot-leg.exe --config .\leg.toml --port COM5 `
  set-middle --motor 1
```

Type `CENTER` only after confirming the selected motor and neutral pose. Repeat
for motors 2 and 3 when required, then run `capture-centers` again. The command
disables torque, updates the servo's internal position correction, verifies the
new position is near raw 2048, and relocks configuration. Unlike
`capture-centers`, this is a persistent servo-memory update.

## 6. Read telemetry

```bash
drobot-leg --config leg.toml --port /dev/tty.usbmodem575E0031751 \
  --calibration calibration.json \
  status
```

The output includes raw position, calibrated angle, speed, voltage,
temperature, estimated current, torque state, and the responding model number.
The servo's current/load feedback is diagnostic telemetry, not a calibrated
force or joint-torque measurement.

## 7. Control the selected motor

```bash
drobot-leg --config leg.toml --port /dev/tty.usbmodem575E0031751 \
  --calibration calibration.json \
  control
```

The console starts with every motor disarmed:

```text
select 1
arm
+ 1
+ 1
status
set 5
disarm

select 2
arm
- 2
status
disarm

disarm-all
quit
```

`set 20` is rejected when it would jump more than the configured 5 degrees
from the previous target. Move in small increments. A target outside that
motor's `min_deg`/`max_deg` range is also rejected.

Only the selected motor receives `arm`, `set`, and nudge commands. Selecting
another motor does not change torque state: a previously armed motor keeps
holding its last target until you explicitly `disarm` it. This lets you build
a three-joint pose deliberately, but also means you should check and disarm
each motor you no longer need. `quit`, normal exit, and `Ctrl+C` make a
best-effort attempt to disable every motor armed by that session.

## Local browser controller

The localhost-only browser UI shows each joint's live measured position,
requested destination, torque state, voltage, temperature, current, and raw
encoder position. Each joint has a full-range slider, exact numeric angle
entry, zero button, and 1/5-degree nudges.

Install the current editable package after pulling or changing the web tool:

```powershell
Set-Location hardware\test-apps\one-leg-testbed
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

With the supported leg clear and the physical cutoff in reach, start:

```powershell
.\.venv\Scripts\drobot-leg-web.exe --config .\leg.toml --port COM4 `
  --calibration .\calibration.json
```

The server binds only to `127.0.0.1` and opens
`http://127.0.0.1:8765/` on the same Windows machine. It is not a remote-control
service and is not exposed to the local network.

The UI accepts any destination inside that joint's `min_deg`/`max_deg` range.
A large destination is ramped at 30 degrees per second through small internal
steps no larger than `max_command_step_deg`; the browser does not bypass the
underlying target checks. Before arming, the user must confirm that the fixture
is secure, the area is clear, and the physical cutoff is ready.

The browser sends a liveness heartbeat while any motor is armed. If heartbeat
updates stop for three seconds, the server disarms all motors. Closing the page,
pressing **DISARM ALL**, `Ctrl+C` in the server terminal, and normal server exit
also request a complete disarm. None can replace cutting external servo power
after a USB, process, operating-system, or wiring failure.

To preview the complete UI without opening a serial port or commanding
hardware:

```powershell
.\.venv\Scripts\drobot-leg-web.exe --config .\leg.toml `
  --calibration .\calibration.json --demo
```

The local `leg.toml` remains the source of truth for joint direction and range.
The browser never writes configuration or calibration files.

## Four-leg configuration recorded 2026-08-02

All twelve motors have persistent IDs and each three-motor leg was configured,
centered, and tested individually. The verified map, calibration snapshots,
PowerShell helpers, repeatable calibration procedure, exact reproduction
commands, and hardware-test limitations are maintained in
[`../../robot-runtime/README.md`](../../robot-runtime/README.md).

Legs 2-4 were each exercised unloaded with 15-degree guarded ramps. Positive
motion was operator-confirmed as hip abduction outward, hip flexion forward,
and knee forward after applying the tracked direction mappings. Every test
returned near zero and ended with telemetry reporting torque OFF. These were
basic direction checks, not payload, endurance, stall, thermal, cable-wear, or
complete-robot collision tests.

## Physical range run recorded 2026-07-28

The current local testbed was centered at raw tick `2048` on all three motors
and physically exercised while fixed to the wall. The tested local
configuration used hip abduction `-45 to +45 deg`, hip flexion
`-90 to +90 deg`, and knee `-120 to +120 deg`; hip flexion and knee use
`direction = -1`. The operator reported that the unloaded leg moved freely
through this useful range.

This is an operator-observed isolated-fixture result, not an endurance,
payload, current, thermal, or complete-robot collision qualification. The
machine-local `leg.toml` and `calibration.json` remain ignored. A reproducible
simulation snapshot of their kinematic values is maintained in
`cad/drobot_cad/urdf/one_leg_wall_testbed.py` and exercised by
`simulation/isaac/run_one_leg_wall.py`.

## Configuration reference

`leg.toml` owns the testbed mapping and software limits:

| Field | Meaning |
| --- | --- |
| `number` | Human-facing selector 1-3 |
| `name` | Stable joint name |
| `id` | Persistent servo bus ID |
| `direction` | `1` or `-1` conversion between joint degrees and encoder direction |
| `min_deg`, `max_deg` | Software target limits around the captured neutral pose |
| `torque_limit` | ST3215 live torque-limit register, 0-1000 |
| `speed` | Feetech `WritePosEx` speed value |
| `acceleration` | Feetech acceleration value, 1-254 |
| `max_command_step_deg` | Largest accepted target change in one console command |

The software angle limits do not create physical stops. The bare ST3215 can
rotate through 360 degrees, while the assembled joint is restricted by its
horn, printed parts, cables, neighboring links, and fixture.

## Validation

Run the hardware-independent checks:

```bash
python3 -m pytest
python3 -m ruff check src tests
python3 -m compileall -q src
```

These tests verify configuration validation, unique motor numbering/IDs,
angle conversion, calibration persistence, arm-before-command behavior,
current-position hold before torque enable, per-command movement limits, and
explicit disarming. They do not communicate with real motors.

## Known limitations

- One unloaded wall-mounted range run has been completed, but no endurance,
  payload, current, thermal, or cable-wear run has been completed.
- The tracked example and four-leg profile ranges are bench-control limits,
  not complete-robot mechanical or collision-qualified limits.
- USB disconnect, interpreter termination, or operating-system failure may
  prevent software disarm; use a physical power cutoff.
- Browser heartbeat and page-close disarm are best-effort software safeguards,
  not an emergency stop.
- The program controls one selected motor at a time and is not a gait
  controller.
- It does not synchronize three targets into one bus transaction.
- It does not perform load, endurance, thermal, power-supply, or cable-routing
  validation.
- It assumes all three motors already use the same configured baud rate.
