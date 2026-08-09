# Four-leg configuration and calibration

This directory is the tracked source of truth for the four assembled Drobot
leg profiles. Each profile controls one three-servo leg at a time with the
one-leg testbed CLI or localhost browser tool.

The files record the hardware setup initially verified on 2026-08-02 local
time and updated after the four-leg rewire on 2026-08-08. The center ticks were
recaptured from the manually positioned whole-robot neutral pose at
2026-08-09 00:04:22 UTC. They are hardware-specific snapshots, not universal
values for another robot or a leg whose horns have been reinstalled.

## Verified motor map

| Leg | Hip abduction | Hip flexion | Knee | Directions | Center ticks |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | ID 1 | ID 2 | ID 3 | `-1, +1, +1` | `2131, 2183, 1005` |
| 2 | ID 4 | ID 5 | ID 6 | `+1, +1, +1` | `2022, 2162, 2052` |
| 3 | ID 7 | ID 8 | ID 9 | `-1, -1, -1` | `2020, 2052, 2046` |
| 4 | ID 10 | ID 11 | ID 12 | `+1, +1, +1` | `1998, 2049, 2048` |

The intended positive joint directions are hip abduction outward, hip flexion
forward, and knee forward. After the 2026-08-08 rewire, the left-side Leg 1
and Leg 3 profiles were reversed in software so movement in those intended
directions increases the reported angle. Recheck all six affected joints with
a guarded 15-degree ramp before using leg-level commands. The 2026-08-09
software capture updated all twelve centers without writing servo EEPROM;
Leg 1 knee ID 3 moved from center tick 2048 to 1005. Earlier unloaded checks
reported model 777, approximately 12.2-12.3 V, 30-37 C, and torque OFF after
each test. This is not a payload, stall, endurance, thermal, cable-wear, or
complete-robot collision qualification.

All profiles currently use:

- 1,000,000 baud;
- torque limit `300/1000` (30%);
- speed `350` servo steps/s, approximately 30.8 degrees/s;
- acceleration `10`;
- a 5-degree maximum change per interactive console command; and
- software ranges of +/-45 degrees abduction, +/-90 degrees flexion, and
  +/-120 degrees knee.

Software ranges do not replace physical stops. Recheck horn placement, cable
slack, fixture clearance, and link interference before using the full range.

## One-time software setup

Run these commands from `robot-cad\hardware\one-leg-testbed` in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The helper scripts call the executables inside `.venv` directly, so activating
the virtual environment is optional.

## Safety sequence before every hardware command

1. Turn external servo power off before connecting, disconnecting, or swapping
   a leg.
2. Connect only the intended leg and support it so torque-off joints cannot
   fall.
3. Restore the correct external servo power and keep its physical cutoff within
   reach.
4. Keep hands, cables, and tools outside the complete joint sweep.
5. Confirm the selected leg number and COM port before continuing.

USB, browser heartbeat, `Ctrl+C`, and software disarm are not emergency stops.

## Read status without enabling torque

```powershell
.\config\show-status.ps1 -Leg 4 -Port COM4
```

This reads positions, voltage, temperature, current, model, and torque state.
It does not arm or move a motor.

## Reapply safe motor settings

Use this after replacing a motor or if its position-mode settings may have
changed:

```powershell
.\config\configure-leg.ps1 -Leg 4 -Port COM4
```

The script scans the selected leg's three expected IDs, asks for the exact word
`CONFIGURE`, writes position mode, torque limit, speed, and acceleration, and
then checks telemetry. It leaves torque disabled and does not command motion.

## Recalibrate neutral in software

Use software-only calibration when the assembled neutral pose changed but the
needed range does not cross the encoder boundary at raw 0/4095:

```powershell
.\config\calibrate-leg.ps1 -Leg 4 -Port COM4
```

The script performs a read-only status check, asks for `CALIBRATE`, disables
torque, and pauses while you support and manually place all three joints at
neutral. Press Enter at the CLI prompt to overwrite the selected tracked
`calibration-leg-N.json`, then the script verifies near-zero angles and torque
OFF.

This default operation does not change servo EEPROM or firmware. Review and
commit the updated calibration JSON if the new physical neutral should become
the shared canonical value.

## Persistently reset a servo midpoint

Use the persistent middle-position operation only when a required joint range
would cross raw 0/4095, a replacement servo lost its reference, or a status
check shows neutral near the encoder boundary. The leg must already be
physically supported at the intended neutral pose.

Reset all three selected-leg motors, then recapture software centers:

```powershell
.\config\calibrate-leg.ps1 -Leg 4 -Port COM4 -SetServoMiddle
```

Reset only one motor selector when the other two references are already valid:

```powershell
.\config\calibrate-leg.ps1 -Leg 4 -Port COM4 `
  -SetServoMiddle -MiddleMotor 2
```

This mode requires the additional phrase `CENTER-PERSISTENT`. It disables
torque and makes each selected motor's current physical pose logical tick 2048
using the Feetech middle-position command. That is a persistent servo-memory
write. The script then performs the normal three-joint software center capture.

Do not use persistent midpoint reset merely to correct a reversed motion.
Direction is controlled by `direction = 1` or `direction = -1` in the leg TOML
profile.

## Start the local browser controller

```powershell
.\config\start-web-control.ps1 -Leg 4 -Port COM4
```

The controller binds only to `127.0.0.1:8765`, starts disarmed, and uses a
three-second browser-heartbeat timeout. The page can show current position and
ramp to any destination inside the selected joint's configured range.

To connect all four configured legs on one bus and see IDs 1-12 together, use
the separate [`../../test-apps/`](../../test-apps/README.md) four-leg
commissioning dashboard.

Preview the UI without opening COM4 or communicating with motors:

```powershell
.\config\start-web-control.ps1 -Leg 4 -Demo
```

## Equivalent direct commands

The scripts are convenience wrappers. For Leg 4, the underlying commands are:

```powershell
.\.venv\Scripts\drobot-leg.exe --config .\config\leg-4.toml --port COM4 `
  --calibration .\config\calibration-leg-4.json status

.\.venv\Scripts\drobot-leg.exe --config .\config\leg-4.toml --port COM4 `
  --calibration .\config\calibration-leg-4.json capture-centers

.\.venv\Scripts\drobot-leg-web.exe --config .\config\leg-4.toml --port COM4 `
  --calibration .\config\calibration-leg-4.json
```

Change both occurrences of `4` to select another leg profile.

## Files

- `leg-N.toml` owns bus settings, servo IDs, joint directions, and software
  angle limits.
- `calibration-leg-N.json` owns the measured raw tick corresponding to zero
  degrees for each assembled joint.
- Root-level `leg*.toml` and calibration JSON files remain ignored scratch
  copies for local experiments.
- The Python package under `../src/` owns transport, safety checks, console
  control, and the localhost browser application.

## Validation

The tracked configuration package is validated without opening a serial port
by loading every TOML/JSON pair through the production parser. The repository
unit, lint, format, and compile checks also remain the software validation
source. Hardware observations above apply only to the unloaded guarded tests
that were actually performed.
