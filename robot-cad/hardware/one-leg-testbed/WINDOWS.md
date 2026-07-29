# Windows USB setup and first-movement guide

This guide configures and tests one Drobot leg containing three Feetech ST3215
serial bus servos from Windows PowerShell. It covers USB detection, assigning
IDs 1-3, recording the assembled leg's neutral position, reading telemetry,
and making the first small movements.

The software has been checked on Windows without motors connected. The example
joint limits and motor settings are conservative starting values, not
hardware-validated mechanical limits.

## Before connecting a motor

You need:

- Windows 10 or 11;
- 64-bit Python 3.11 or newer;
- a compatible half-duplex Feetech/Waveshare bus-servo USB adapter;
- a data-capable USB cable;
- an external supply appropriate for the exact ST3215 voltage variant;
- a physical servo-power cutoff within reach; and
- a clamp or fixture that prevents the leg from striking the desk.

Do not power three ST3215 servos from the computer's USB port. USB carries the
control connection; the servos require their appropriate external supply.
Follow the adapter manufacturer's instructions for servo power and shared
ground wiring.

Turn external servo power off before connecting, disconnecting, or
daisy-chaining motors. Keep hands away from horns and linkages whenever torque
may be enabled.

## 1. Install the testbed

Open PowerShell in the repository and run:

```powershell
Set-Location robot-cad\hardware\one-leg-testbed

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

Copy-Item .\leg.example.toml .\leg.toml
```

The commands use the virtual environment directly, so PowerShell script
execution policy does not need to be changed and activation is optional.

If `py -3.11` is not found, install a current 64-bit Python release with the
Windows Python launcher, then open a new PowerShell window and confirm:

```powershell
py -3.11 --version
```

## 2. Find the USB adapter

Connect the USB adapter, initially without any servo connected, and run:

```powershell
.\.venv\Scripts\drobot-leg.exe ports
```

The output should include a Windows port such as:

```text
COM5
  USB Serial Port (COM5)
  USB VID:PID=...
```

Use the displayed port in the rest of this guide. Replace `COM5` if Windows
assigned a different number.

If no port appears:

1. Try a known data-capable USB cable and another USB port.
2. Open Device Manager and inspect **Ports (COM & LPT)** and **Other devices**.
3. Install only the driver recommended by the USB adapter manufacturer.
4. Disconnect and reconnect the adapter, then run `drobot-leg ports` again.

Windows can assign a different COM number after changing USB sockets, so list
ports again whenever a previously working port disappears.

## 3. Review the motor map

The local `leg.toml` maps the three physical testbed motors:

| Testbed number | Joint | Bus ID |
| ---: | --- | ---: |
| 1 | hip abduction | 1 |
| 2 | hip flexion | 2 |
| 3 | knee | 3 |

It also defines conservative angle, torque, speed, acceleration, and
per-command movement limits. Keep the initial limits unchanged for the first
unloaded test.

## 4. Assign IDs with one motor connected

New motors may all use ID 1. Never connect multiple factory-default motors
while assigning IDs because identical IDs collide on the shared bus.

Start the guided setup:

```powershell
.\.venv\Scripts\drobot-leg.exe --config .\leg.toml --port COM5 setup-ids
```

For each prompted motor:

1. Turn external servo power off.
2. Connect only the named motor to the adapter.
3. Restore external servo power.
4. Press Enter and type `ASSIGN` when requested.
5. Turn external power off before disconnecting that motor.

The program assigns IDs 1, 2, and 3 according to `leg.toml`. It disables torque
before changing an ID and verifies that the motor responds at the new ID.

To assign one isolated motor manually:

```powershell
.\.venv\Scripts\drobot-leg.exe --config .\leg.toml --port COM5 `
  assign-id --current-id 1 --new-id 2
```

The backtick continues a PowerShell command onto the next line. The same
command can be written on one line without it.

## 5. Connect and scan all three motors

Turn external power off, daisy-chain the three uniquely numbered motors, and
support the leg in its fixture. Restore power and scan:

```powershell
.\.venv\Scripts\drobot-leg.exe --config .\leg.toml --port COM5 `
  scan --id-start 1 --id-end 3
```

Proceed only when exactly IDs 1, 2, and 3 answer. If an ID is missing, power
off and check its cable and previous assignment. If IDs collide, return to the
one-motor assignment procedure.

## 6. Apply conservative position-control settings

With the leg supported:

```powershell
.\.venv\Scripts\drobot-leg.exe --config .\leg.toml --port COM5 configure
```

Review the warning and type `CONFIGURE`. This selects position mode and applies
the configured torque limit and acceleration. All motors remain disarmed when
the command finishes.

## 7. Record the assembled neutral pose

Run:

```powershell
.\.venv\Scripts\drobot-leg.exe --config .\leg.toml --port COM5 `
  --calibration .\calibration.json capture-centers
```

The program disables torque on all three motors. Support the leg, manually
place all joints at the intended neutral pose, and then press Enter. Encoder
centers are stored locally in the ignored `calibration.json`; the procedure
does not write homing offsets to servo EEPROM.

If the required range crosses raw encoder boundary 0/4095, a local calibration
file is not sufficient. With torque off and the supported joint physically at
neutral, persistently make that motor logical tick 2048:

```powershell
.\.venv\Scripts\drobot-leg.exe --config .\leg.toml --port COM5 `
  set-middle --motor 1
```

Review the named motor and type `CENTER`. Repeat for another motor only while
the complete leg remains at neutral, then run `capture-centers` again. This
updates the selected servo's internal position correction and is distinct from
the local JSON calibration.

## 8. Check telemetry before moving

```powershell
.\.venv\Scripts\drobot-leg.exe --config .\leg.toml --port COM5 `
  --calibration .\calibration.json status
```

Confirm that all three motors respond and that voltage and temperature look
reasonable for the unpowered-to-powered bench setup. Current feedback is
diagnostic telemetry, not a calibrated measurement of joint torque.

## 9. Make the first small movement

Clamp or suspend the leg, clear the mechanism, and keep the physical power
cutoff within reach:

```powershell
.\.venv\Scripts\drobot-leg.exe --config .\leg.toml --port COM5 `
  --calibration .\calibration.json control
```

The controller starts with all motors disarmed. The prompt always shows the
selected motor number and name:

```text
[#1 hip_abduction]>
```

Try one degree at a time:

```text
select 1
arm
+ 1
status
- 1
disarm

select 2
arm
+ 1
status
disarm

select 3
arm
- 1
status
disarm

disarm-all
quit
```

`arm` first commands the selected motor's measured position and then enables
torque. `set`, `+`, and `-` affect only the selected motor. A previously armed
motor continues holding its last target until explicitly disarmed.

The default configuration rejects targets outside each joint's conservative
range and rejects a target change larger than five degrees. Repeated commands
can still move a joint into a physical obstruction, so watch the mechanism and
use small increments.

If positive motion goes in the wrong physical direction:

1. Run `disarm-all` and `quit`.
2. Turn external servo power off.
3. Change that motor's `direction` in `leg.toml` from `1` to `-1`.
4. Restore power and capture the neutral centers again.
5. Repeat the one-degree unloaded test.

## 10. Use the local browser controller

Reinstall the editable project once so Windows creates the
`drobot-leg-web.exe` entry point:

```powershell
Set-Location robot-cad\hardware\one-leg-testbed
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Start the controller:

```powershell
.\.venv\Scripts\drobot-leg-web.exe --config .\leg.toml --port COM4 `
  --calibration .\calibration.json
```

The program opens `http://127.0.0.1:8765/`. The page shows current and target
angles for all three joints. Check the hardware-ready confirmation, arm one
joint, then use its slider or type an exact destination angle. Destinations
inside the configured joint range are approached smoothly in bounded internal
steps.

The server listens on the local Windows machine only. It automatically disarms
after three seconds without a browser heartbeat, on normal page close, or when
the server receives `Ctrl+C`. Keep the physical external-power cutoff within
reach because software disarm cannot cover every process, USB, operating-system,
or wiring failure.

To check the interface with simulated motors and no serial-port access:

```powershell
.\.venv\Scripts\drobot-leg-web.exe --config .\leg.toml `
  --calibration .\calibration.json --demo
```

## Emergency and normal stopping

For a normal software stop:

```text
disarm-all
quit
```

`Ctrl+C`, `quit`, and normal program exit make a best-effort attempt to disable
every motor armed during that session. They cannot guarantee torque-off after
a USB disconnect, interpreter crash, wiring failure, or operating-system
failure. Use the physical external-power cutoff for an emergency.

## Windows troubleshooting

### Access to COM5 is denied

Close serial terminals, Arduino tools, vendor utilities, or another
`drobot-leg` process using that port. Disconnect and reconnect the adapter if
the previous process exited unexpectedly.

### More than one USB serial adapter is found

Pass the exact port explicitly:

```powershell
.\.venv\Scripts\drobot-leg.exe --config .\leg.toml --port COM5 `
  scan --id-start 1 --id-end 3
```

### The adapter appears, but no motor answers

Power off before inspecting wiring. Check external servo power, adapter mode,
shared ground, connector orientation, data continuity, configured one-megabit
baud rate, and the expected SMS/STS protocol. Do not repeatedly move connectors
while the servo supply is live.

### A motor jumps or approaches a collision

Cut external servo power. Verify that the horn was installed near the intended
neutral position, then repeat neutral capture. Review `direction`, reduce the
joint's configured range, and restart with one-degree changes.

### PowerShell cannot find `drobot-leg.exe`

Confirm installation using the virtual environment's Python:

```powershell
.\.venv\Scripts\python.exe -m pip show drobot-one-leg-testbed
.\.venv\Scripts\python.exe -m drobot_leg_testbed --help
```

## Software-only validation

These checks do not energize or communicate with motors:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp .pytest-tmp-windows
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check src tests
.\.venv\Scripts\python.exe -m compileall -q src tests
```

Passing these checks confirms the software contracts, not the mechanical
limits, power integrity, thermal behavior, or safety of the physical leg.
