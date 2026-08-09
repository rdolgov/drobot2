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
- a guarded **WALK FORWARD** command that runs two deterministic, slow,
  four-beat crawl cycles with visible phase/progress and a dedicated
  **STOP + DISARM** control;
- voltage, temperature, diagnostic current, raw encoder, speed, torque state,
  model, and per-leg current summaries;
- a three-second browser-heartbeat auto-disarm;
- best-effort disarm on page close, telemetry/motion fault, `Ctrl+C`, or normal
  server exit; and
- a complete simulated mode that never opens a serial port.

The server binds only to `127.0.0.1`. It is not a LAN or internet remote-control
service.

## Source of truth

- `config/four-leg.toml` owns dashboard labels, body-corner mapping, the
  bounded crawl parameters, monitoring attention thresholds, local HTTP
  settings, and references to each leg profile.
- `../one-leg-testbed/config/leg-N.toml` owns servo IDs, directions, torque
  limit, speed, acceleration, and angle ranges.
- `../one-leg-testbed/config/calibration-leg-N.json` owns the measured neutral
  encoder ticks.
- `src/drobot_hardware_test_apps/four_leg_control.py` owns bus/session safety,
  HTTP endpoints, telemetry summaries, and demo behavior.
- `src/drobot_hardware_test_apps/crawl_gait.py` owns the dependency-free,
  deterministic crawl target equations mirrored from the Isaac runtime.
- `src/drobot_hardware_test_apps/four_leg_static/` owns the browser UI.

The walking manifest currently maps Leg 1 to front-left, Leg 2 to front-right,
Leg 3 to rear-left, and Leg 4 to rear-right. This preserves the confirmed
left-side assignment for Legs 1 and 3 and assumes the normal front-to-rear ID
ordering. Verify the map printed on the page against the physical harness
before enabling the first crawl. If the harness differs, update both `label`
and `corner` for the affected `[[legs]]` entries; every corner must appear
exactly once.

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
serial hardware. The header must show **DEMO / NO MOTOR OUTPUT**. Stop the demo
before starting a COM-port session; the launcher now refuses to share its HTTP
port with another dashboard process.

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
6. Before arming, confirm the header shows **HARDWARE / COM4**, then confirm
   the page shows `12 / 12`, `0 / 12` armed, no unexpected-torque warning,
   plausible voltage, and reasonable temperatures. Never test physical motion
   from a page marked **DEMO / NO MOTOR OUTPUT**.
7. Check the arming interlock. Arm only one joint. Use `+15°`, observe the
   intended physical direction, return with `ZERO`, and disarm it.
8. Repeat the three joints on one leg, then the other legs. Stop immediately on
   an unexpected direction, cable pull, collision, noise, heating, voltage
   warning, or sustained high current.
9. Only after individual checks should `ARM 3 MOTORS` be used to hold one
   supported leg. Do not arm all twelve for a floor-standing test until power,
   structure, support, and complete-robot collision behavior have separate
   validation.

## Slow crawl-forward command

**WALK FORWARD** is a fixed two-cycle choreography, not an AI policy. It uses
the same quasi-static target equations and physical dimensions as the tracked
Isaac crawl: rear-right, front-right, rear-left, then front-left. Each foot is
unloaded, lifted 10 mm, advanced through a 15 mm stride, lowered, and returned
to support before the next foot moves. Each cycle takes 20 seconds, for a
40-second default command.

Before the command is accepted, the server requires:

- all 12 IDs online and all motors disarmed;
- no voltage, temperature, current, or unexpected-torque warning;
- every measured joint within 35 degrees of calibrated zero;
- every sampled gait target inside that motor's configured range;
- the support/clearance/corner-map/cutoff checkbox and a second confirmation.

For the first physical attempt, support the body on blocks with the feet barely
touching or just clear of the floor. Confirm the displayed corner map, press
**WALK FORWARD**, and be ready to use **STOP + DISARM** or the physical power
cutoff. The app first ramps to the crawl stance, settles, then runs two cycles.
It finishes holding the crawl stance with torque still enabled; support the
robot before pressing **STOP + DISARM**. Increase stride, lift, cycle count, or
speed only after the default motion is confirmed on the assembled robot.

The tracked defaults live in `[crawl]` in `config/four-leg.toml`. They are
deliberately capped to a 5-30 mm stride, 5-20 mm lift, 12-60 second period, and
one to four cycles. Manual joint, zero, and centering commands are locked while
the crawl is active. Browser-heartbeat loss, a motion/telemetry fault, any
disarm request, page close, or **STOP + DISARM** cancels the gait and attempts
to remove torque from all 12 motors.

## Isaac validation

The dashboard gait was validated on 2026-08-08 against the tracked floating
quadruped USD and the rated `0.980665 Nm` ST3215 torque profile. The dashboard
equations are parity-tested against
`simulation/isaac/_quadruped_runtime.py` over 81 samples per cycle. The full
physics reproduction command, run from the repository root, is:

```powershell
& C:\isaacsim\python.bat robot-cad\simulation\isaac\run_crawl.py `
  --usd robot-cad\exports\isaac\quadruped_robot_floating.usdc `
  --headless --gait-mode quasi-static --torque-cap rated `
  --cycles 2 --period 20 --stride 0.015 --lift 0.010 `
  --weight-shift-forward 0.030 --weight-shift-lateral 0 `
  --stance-down 0.310 --stance-fore-aft 0.025 `
  --abduction-deg 0 --start-z 0.460 --review-phase 0.11 `
  --report robot-cad\hardware\test-apps\validation\isaac-slow-crawl.json `
  --screenshot robot-cad\reviews\isaac-dashboard-slow-crawl.png
```

The rated-torque run passed the existing, unchanged crawl thresholds: 21.97 mm
forward travel, 1.28 mm lateral drift, 2.00 degrees maximum body tilt, 6.27 mm
maximum loaded support-tip slip, 99.64% expected support contact, 0.097 rad
maximum joint tracking error, and completed rear-right, front-right,
rear-left, and front-left steps. The machine-readable report is
[`validation/isaac-slow-crawl.json`](validation/isaac-slow-crawl.json), and the
review image is
[`../../reviews/isaac-dashboard-slow-crawl.png`](../../reviews/isaac-dashboard-slow-crawl.png).

An initial one-cycle trial moved forward 11.12 mm and completed all four steps
with similarly stable contact, but correctly failed the unchanged 20 mm
forward-distance gate. The production button therefore runs two cycles instead
of weakening the acceptance threshold or increasing stride/lift authority.

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
Isaac validation checks the same joint targets against the simulated robot and
contact model. Neither validates bus signal integrity, branch voltage drop,
fuse behavior, servo heating, printed-foot friction, mechanism clearance,
payload capacity, or safe walking on the assembled hardware.

## Known limitations

- Twelve separate servo writes are not one atomic bus transaction.
- The crawl is open-loop position choreography. It has no foot-contact,
  balance, IMU, slip, or obstacle feedback and must not be treated as an
  autonomous walking controller.
- A USB disconnect, process crash, operating-system failure, or damaged data
  wire can prevent software disarm.
- Dashboard polling adds traffic to the same bus used for motion commands.
- The configured ID-to-corner map (Leg 1 front-left, 2 front-right, 3
  rear-left, 4 rear-right) must still be confirmed against the physical
  harness before the first crawl.
- Monitoring warnings are commissioning aids, not electrical certification.
