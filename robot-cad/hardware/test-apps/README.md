# Drobot hardware test applications

This directory is the shared home for local commissioning and diagnostic web
applications. The first app is a four-leg dashboard that controls the twelve
configured ST3215 servos on one shared data bus while reusing the proven
transport, calibration, angle conversion, and bounded-motion logic from
`../one-leg-testbed`.

## Four-leg dashboard

The local dashboard provides:

- all four legs and IDs 1-12 on one screen;
- calibrated current, commanded, and destination angles;
- exact angle entry, full configured-range sliders, and 5/15-degree quick
  tests;
- explicit per-joint arming and disarming;
- intentional three-motor leg arming, zeroing, and disarming;
- global return-to-zero for already armed motors, a guarded **CENTER ALL 12**
  command, and a persistent **DISARM ALL 12** control;
- guarded capture of the current torque-free pose as calibrated zero for all
  twelve motors, with timestamped backups of all four calibration files;
- voltage, temperature, diagnostic current, raw encoder, speed, torque state,
  model, and per-leg current summaries;
- a three-second browser-heartbeat auto-disarm;
- best-effort disarm on page close, telemetry/motion fault, `Ctrl+C`, or normal
  server exit; and
- a complete simulated mode that never opens a serial port.

The server binds only to `127.0.0.1`. It is not a LAN or internet remote-control
service.

## Source of truth

- `config/four-leg.toml` owns dashboard labels, monitoring attention
  thresholds, local HTTP settings, and references to each leg profile.
- `../one-leg-testbed/config/leg-N.toml` owns servo IDs, directions, torque
  limit, speed, acceleration, and angle ranges.
- `../one-leg-testbed/config/calibration-leg-N.json` owns the measured neutral
  encoder ticks.
- `src/drobot_hardware_test_apps/four_leg_control.py` owns bus/session safety,
  HTTP endpoints, telemetry summaries, and demo behavior.
- `src/drobot_hardware_test_apps/four_leg_static/` owns the browser UI.

The manifest deliberately labels the branches `Leg 1` through `Leg 4`. Update
only the `label` fields after physically confirming which ID group is front
left, front right, rear left, and rear right. Do not infer body placement from
bus IDs.

## Wiring and power prerequisites

Turn external servo power off before changing any connection.

The complete robot should follow the repository electrical plan:

- one separately fused power branch per three-servo leg;
- a shared passive data junction to the half-duplex ST3215 adapter;
- a common ground between servo supply and adapter; and
- the regular 12 V ST3215 variant on all twelve IDs.

Do not power twelve servos through USB. Do not route all four legs' power
through one leg's first small connector. Verify polarity, connector orientation,
data continuity, and branch fuses before applying power.

For the first combined-bus test, suspend or block the body so every foot is
clear of the floor and no leg can strike the bench. Keep a physical battery or
bench-supply cutoff within reach. Software disarm is not an emergency stop.

## Install

Open PowerShell in `robot-cad\hardware\test-apps`:

```powershell
.\install-test-apps.ps1
```

This reuses `../one-leg-testbed/.venv`, installs the existing one-leg package,
and then installs the shared test-app package. It does not open COM4 or
communicate with motors.

## Preview with twelve simulated motors

```powershell
.\start-four-leg-web.ps1 -Demo
```

The demo opens `http://127.0.0.1:8766/`, uses the real profiles/calibrations,
and exercises the same arming, ramping, target, watchdog, and UI code without
serial hardware.

## First complete-robot test

1. Power off and support the complete robot with all feet clear.
2. Verify the four fused power branches, shared data, common ground, and unique
   ID groups 1-3, 4-6, 7-9, and 10-12.
3. Restore power with the physical cutoff ready.
4. Start the app:

   ```powershell
   .\start-four-leg-web.ps1 -Port COM4
   ```

5. Type `CONNECT-12`. Startup requires all twelve configured IDs to respond
   and then writes torque OFF to every ID before opening the dashboard.
6. Before arming, confirm the page shows `12 / 12`, `0 / 12` armed, no
   unexpected-torque warning, plausible voltage, and reasonable temperatures.
7. Check the arming interlock. Arm only one joint. Use `+15°`, observe the
   intended physical direction, return with `ZERO`, and disarm it.
8. Repeat the three joints on one leg, then the other legs. Stop immediately on
   an unexpected direction, cable pull, collision, noise, heating, voltage
   warning, or sustained high current.
9. Only after individual checks should `ARM 3 MOTORS` be used to hold one
   supported leg. Do not arm all twelve for a floor-standing test until power,
   structure, support, and complete-robot collision behavior have separate
   validation.

## Center all twelve joints

**CENTER ALL 12** is the whole-robot neutral-position command. It requires the
support/clearance/cutoff checkbox and a second confirmation. The server then
arms every configured motor at its measured position and ramps all twelve
joints to calibrated `0°` using the normal 30-degree-per-second motion limit.

Calibrated `0°` comes from each `calibration-leg-N.json` center; it is not an
unconditional raw tick `2048` command. Torque remains armed after the motion so
the mechanism holds the neutral pose. Inspect the dashboard, then use
**DISARM ALL 12** when holding torque is no longer required. Because all four
legs can move and draw current together, use this only with the robot fully
supported, every sweep clear, branch power already checked, and the physical
cutoff ready.

## Capture a new all-leg zero pose

Use **CAPTURE ZERO ALL** when the assembled neutral pose has changed, such as
after reinstalling horns or rewiring mirrored legs:

1. Press **DISARM ALL 12** and verify the dashboard shows `0 / 12` armed.
2. Keep the robot supported and manually place all four legs in the desired
   neutral pose.
3. Check the support/clearance/cutoff box.
4. Press **CAPTURE ZERO ALL** and approve the second confirmation.
5. Verify every displayed current angle changes to approximately `0.00°` while
   torque remains off.
6. Arm and test one affected joint at `+15°` before using **CENTER ALL 12**.

The operation reads the current encoder tick for IDs 1-12, writes four updated
`calibration-leg-N.json` files, and first copies the previous files into a
timestamped `config/backups/` directory. It does not change servo EEPROM,
direction signs, IDs, speed, torque limit, or angle ranges. Demo mode exercises
the interaction in memory and never changes calibration files.

## Understanding “power OK”

The dashboard uses attention thresholds from `config/four-leg.toml`:

| Signal | Dashboard attention threshold | Meaning |
| --- | ---: | --- |
| Servo voltage | below 10.5 V | Repository's conservative 3S light-load warning |
| Servo voltage | above 12.6 V | Above the documented 12 V ST3215 supply maximum |
| Voltage spread | above 0.5 V | Inspect branch wiring, connectors, and voltage drop |
| Servo temperature | 60 C or above | Conservative pause-and-inspect UI threshold |
| Per-leg diagnostic current | 3000 mA or above | Inspect load and the first leg connector |

These are UI warnings, not firmware protection settings. Servo voltage does not
show individual LiPo cell balance. Servo current feedback is diagnostic and is
not a calibrated battery, branch, force, or joint-torque measurement. Use an
inline watt meter or clamp meter and per-cell checker for electrical validation.
The dashboard never declares the entire power system certified.

## Motion behavior

Arming first commands the motor's measured position and only then enables
torque. Browser destinations are converted with the tracked calibration and
range checked before acceptance. The background worker approaches them at the
configured 30 degrees per second through steps no larger than the existing
5-degree command limit.

The `ZERO` controls mean calibrated zero, not torque off. Always use the joint,
leg, or global disarm button when holding torque is no longer required.

## Direct command

After installation, the launcher is equivalent to:

```powershell
..\one-leg-testbed\.venv\Scripts\drobot-four-leg-web.exe `
  --manifest .\config\four-leg.toml --port COM4
```

## Software validation

These checks do not open a serial port:

```powershell
..\one-leg-testbed\.venv\Scripts\python.exe -m pytest
..\one-leg-testbed\.venv\Scripts\python.exe -m ruff check src tests
..\one-leg-testbed\.venv\Scripts\python.exe -m ruff format --check src tests
..\one-leg-testbed\.venv\Scripts\python.exe -m compileall -q src tests
```

Demo-mode browser validation proves the UI and software state machine execute.
It does not validate bus signal integrity, branch voltage drop, fuse behavior,
servo heating, mechanism clearance, payload capacity, standing, or walking.

## Known limitations

- The app controls test targets; it is not a synchronized gait controller.
- Twelve separate servo writes are not one atomic bus transaction.
- A USB disconnect, process crash, operating-system failure, or damaged data
  wire can prevent software disarm.
- Dashboard polling adds traffic to the same bus used for motion commands.
- Physical corner labels remain unassigned until the harness map is confirmed.
- Monitoring warnings are commissioning aids, not electrical certification.
