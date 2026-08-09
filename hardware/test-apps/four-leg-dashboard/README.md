# Drobot hardware test applications

This directory is the shared home for local commissioning and diagnostic web
applications. The first app is a four-leg dashboard that controls the twelve
configured ST3215 servos on one shared data bus while reusing the proven
transport, calibration, angle conversion, and bounded-motion logic from
`../one-leg-testbed` and reads physical-robot state from
[`../../robot-runtime/`](../../robot-runtime/README.md).

## Four-leg dashboard

The local dashboard provides:

- all four legs and IDs 1-12 on one screen;
- calibrated current, commanded, and destination angles;
- exact angle entry, full configured-range sliders, and 5/15-degree quick
  tests;
- explicit per-joint arming and disarming;
- intentional three-motor leg arming, zeroing, and disarming;
- directly accessible toolbar **CENTER ALL 12** and **DISARM ALL 12** commands,
  plus a **SETTINGS** dialog for infrequent **ZERO ARMED** and
  **CAPTURE ZERO ALL** maintenance commands;
- guarded capture of the current torque-free pose as calibrated zero for all
  twelve motors, with timestamped backups of all four calibration files;
- direct **SET GAIT START STANCE** and **TEST DISTRIBUTED CRAWL** commands
  with visible phase/progress and a dedicated
  **STOP + DISARM** control;
- a permanent red error log directly below the toolbar; command, connection,
  and server faults remain in browser storage until explicitly cleared from
  **SETTINGS**;
- voltage, temperature, diagnostic current, raw encoder, speed, torque state,
  model, and per-leg current summaries;
- a 1.5-second browser-heartbeat auto-disarm;
- best-effort disarm on page close, telemetry read/motion fault, `Ctrl+C`, or normal
  server exit; and
- a complete simulated mode that never opens a serial port.

The server binds only to `127.0.0.1`. It is not a LAN or internet remote-control
service.

## Source of truth

- `../../robot-runtime/four-leg.toml` owns dashboard labels, body-corner mapping, the
  bounded crawl parameters, monitoring attention thresholds, local HTTP
  settings, and references to each leg profile.
- `../../robot-runtime/servos/leg-N.toml` owns servo IDs, directions, torque
  limit, speed, acceleration, and angle ranges.
- `../../robot-runtime/servos/calibration-leg-N.json` owns the measured neutral
  encoder ticks.
- `src/drobot_hardware_test_apps/four_leg_control.py` owns bus/session safety,
  HTTP endpoints, telemetry summaries, and demo behavior.
- `src/drobot_hardware_test_apps/crawl_gait.py` owns the dependency-free
  foot-space inverse kinematics and deterministic crawl equations.
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

Open PowerShell in `hardware\test-apps`:

```powershell
.\install-test-apps.ps1
```

This reuses `one-leg-testbed/.venv`, installs the existing one-leg package,
and then installs the four-leg dashboard package. It does not open COM4 or
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

5. Startup opens COM4 directly, requires all twelve configured IDs to respond,
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

## Distributed-push crawl V5

The previous manually proposed three-joint sequence was not a gait: it did not
lift and replace every foot or produce a coordinated support-leg push. V5 uses
foot-space inverse kinematics and repeats this order:

1. rear right;
2. front right;
3. rear left; and
4. front left.

Each foot step has eight phases: weight transfer, lift, swing, lower,
touchdown, weight return, all-feet push, and settle. Only one foot is airborne.
After it lands, all four planted feet move rearward by one quarter of the
stride. Four such pushes translate the body while returning every foot target
to the next periodic starting position.

The start stance is computed rather than hardcoded. The tracked geometry uses
`300 mm` downward reach, `25 mm` fore/aft separation, no extra abduction, and
the physically verified knee convention: front knees negative/downward and
rear knees positive/downward. **SET GAIT START STANCE** moves to the exact pose
used at gait time zero without beginning the crawl.

**TEST DISTRIBUTED CRAWL** runs two approximately 6.67-second cycles with a
`112 mm` stride, `14 mm` lift, and `16 mm` fore/aft body-weight transfer. This
triples the trajectory rate from the prior 20-second baseline. The lateral
shift is zero because the simulated robot completed every foot unload without
it and because an unverified open-loop lateral body command adds a physical
fall mode. The controller uses smooth interpolation throughout the phase table.

The exact equations, phase fractions, sign convention, simulation evidence,
and physical trial ladder are in
[`specs/hardware-gait-sequence-v1.md`](specs/hardware-gait-sequence-v1.md).
The filename is retained so existing documentation links remain stable. The V4
and older documents remain historical evidence.

Before either whole-robot command is accepted, the server requires all 12 IDs
online and every computed target inside the configured joint limits. The
command arms any currently disarmed motors at their measured positions before
ramping. Browser-heartbeat loss, a telemetry read or motion exception, any
disarm request, page close, or **STOP + DISARM** cancels motion and attempts to
remove torque from all twelve motors. Display-only voltage, temperature,
voltage-spread, and diagnostic-current warnings remain visible but do not stop
the command.

The tracked defaults live in `[crawl]` in `../../robot-runtime/four-leg.toml`.
The parser permits a 5-120 mm stride, 5-25 mm lift, 5-60 second period, and one
to four cycles. The current `cycles = 2` setting is intentional and matches the
two-cycle Isaac validation.

## Isaac validation status

The hardware and Isaac implementations use matching distributed-push
equations. Before the physical stride was doubled, the selected two-cycle
`28 mm` baseline was run against the floating 4.526 kg Isaac robot at the
sustainable rated `0.980665 Nm` per-joint cap:

```powershell
& C:\isaacsim\python.bat simulation\isaac\run_crawl.py `
  --usd simulation\exports\isaac\quadruped_robot_floating.usdc `
  --headless --gait-mode distributed-push --torque-cap rated `
  --cycles 2 --period 20 --stride 0.028 --lift 0.014 `
  --weight-shift-forward 0.016 --weight-shift-lateral 0 `
  --stance-down 0.305 --stance-fore-aft 0.025 `
  --abduction-deg 0 --start-z 0.455 `
  --min-forward-displacement 0.010 `
  --report hardware\test-apps\four-leg-dashboard\validation\isaac-distributed-push-v5-selected.json `
  --screenshot hardware\test-apps\four-leg-dashboard\validation\isaac-distributed-push-v5-selected.png
```

Result: **PASS under the selected assumptions**. It moved forward `60.0 mm`,
reached `1.42 degrees` maximum body tilt, held maximum joint error to `0.100
rad`, limited support-tip slip to `3.67 mm`, and completed unload/touchdown for
all four feet. The retained report is
[`validation/isaac-distributed-push-v5-selected.json`](validation/isaac-distributed-push-v5-selected.json).

The same run at the short-duration stall/peak cap of `2.941995 Nm` also passed,
but moved `59.73 mm` rather than farther. Its peak joint speed increased from
`1.448` to `1.842 rad/s`; the extra simulated torque reserve did not improve
travel. That comparison is retained at
[`validation/isaac-distributed-push-max-power-control.json`](validation/isaac-distributed-push-max-power-control.json).
Simulation remains a geometry and dynamics screen, not permission for an
untethered first physical run. The current `112 mm`, 6.67-second physical-trial
profile and its `300 mm` nominal downward reach have not been simulated; these
changes were requested after this retained baseline.

## Center all twelve joints

Press **CENTER ALL 12** in the toolbar for the whole-robot neutral-position
command. A click sends the command immediately. The server then
arms every configured motor at its measured position and ramps all twelve
joints to calibrated `0°` using the configured 270-degree-per-second motion limit.

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
3. Open **SETTINGS** in the toolbar.
4. Press **CAPTURE ZERO ALL**. The command starts immediately.
5. Verify every displayed current angle changes to approximately `0.00°` while
   torque remains off.
6. Arm and test one affected joint at `+15°` before using **CENTER ALL 12**.

The operation reads the current encoder tick for IDs 1-12, writes four updated
`calibration-leg-N.json` files, and first copies the previous files into a
timestamped `config/backups/` directory. It does not change servo EEPROM,
direction signs, IDs, speed, torque limit, or angle ranges. Demo mode exercises
the interaction in memory and never changes calibration files.

## Permanent error history

Dashboard command failures, connection failures, and reported server faults are
listed in red immediately below the toolbar. A later successful command does not
replace them, and the browser retains the list across page refreshes. Open
**SETTINGS** and press **CLEAR ERROR LOG** after the underlying problem has been
resolved. Repeated identical messages are stored once so a disconnected server
does not flood the panel.

## Understanding “power OK”

The dashboard uses attention thresholds from `../../robot-runtime/four-leg.toml`:

| Signal | Dashboard attention threshold | Meaning |
| --- | ---: | --- |
| Servo voltage | below 11.0 V | Earlier warning for supply sag under the higher torque cap |
| Servo voltage | above 12.6 V | Above the documented 12 V ST3215 supply maximum |
| Voltage spread | above 0.3 V | Earlier branch-wiring and voltage-drop warning |
| Servo temperature | 55 C or above | Earlier pause-and-inspect threshold |
| Per-leg diagnostic current | 2500 mA or above | Earlier load and connector warning |

These are display-only software thresholds, not firmware protection settings.
Temperature, voltage, voltage-spread, and diagnostic-current warnings remain
visible but do not block a command or automatically disarm motors. Actual
telemetry read exceptions, motion exceptions, the browser heartbeat watchdog,
and explicit stop/disarm commands still disarm. Servo voltage does not show
individual LiPo cell balance. Servo current feedback is diagnostic and is not
a calibrated battery, branch, force, or joint-torque measurement. Use an inline
watt meter or clamp meter and per-cell checker for electrical validation. The
dashboard never declares the entire power system certified.

## Motion behavior

Arming first commands the motor's measured position and only then enables
torque. Browser destinations are converted with the tracked calibration and
range checked before acceptance. The background worker can approach them at up
to the configured 270 degrees per second through command steps no larger than
15 degrees. Each servo profile uses speed register `3400`, acceleration `254`,
and the existing 90% torque limit.

The `ZERO` controls mean calibrated zero, not torque off. Always use the joint,
leg, or global disarm button when holding torque is no longer required.

## Direct command

From `hardware\test-apps\four-leg-dashboard`, the launcher is equivalent to:

```powershell
..\one-leg-testbed\.venv\Scripts\drobot-four-leg-web.exe `
  --manifest ..\..\robot-runtime\four-leg.toml --port COM4
```

## Software validation

Run these checks from `hardware\test-apps\four-leg-dashboard`; they do not open
a serial port:

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
