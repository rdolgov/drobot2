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
  **STOP + STABLE HOLD** control; **DISARM ALL 12** remains the explicit way
  to remove torque;
- a guarded V20 **START RL WALK** using the real BNO085, live calibrated
  position/velocity feedback for all 12 joints, a model-declared forward
  command range (`0.040-0.100 m/s` for V20), a custom `1-60 s` duration,
  bounded 60 Hz policy targets and synchronous 12-motor writes,
  normal-completion transition to calibrated center with torque holding, fault
  disarm, and a dedicated **STOP RL + DISARM**;
- a recent red error log directly below the toolbar; active bus and RL faults
  remain visible, resolved state-backed faults clear on refresh, and other
  entries expire after ten minutes (with an immediate **SETTINGS** clear);
- voltage, temperature, diagnostic current, raw encoder, speed, torque state,
  model, and per-leg current summaries;
- a browser heartbeat sent every 0.7 seconds on an independent request path,
  with a visible warning after 20 seconds but no effect on gait or torque;
- best-effort disarm on telemetry read/motion fault, `Ctrl+C`, or normal server
  exit; and
- a complete simulated mode that never opens a serial port.

The desktop launcher binds only to `127.0.0.1`. The Raspberry Pi launcher under
`onboard/scripts/run-manual-web.sh` deliberately enables trusted-LAN access so
the same page can be used from a phone or computer. Never expose this motor
control page to the public internet.

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
- `src/drobot_hardware_test_apps/rl_policy_control.py` adapts the shared ONNX
  policy runtime to the single motor-owning dashboard session.
- `src/drobot_hardware_test_apps/four_leg_static/` owns the browser UI.

The real-policy rollout and every enforced guard are documented in
[`specs/rl-walk-v1.md`](specs/rl-walk-v1.md).

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

## Rectangular flat-support crawl V9 slow

The current gait uses foot-space inverse kinematics and repeats this order:

1. rear right;
2. front right;
3. rear left; and
4. front left.

Each foot step has eight phases: weight transfer, lift, swing, lower, firm
plant, weight return, all-feet push, and settle. Only one foot is airborne.
After it lands, all four planted feet move rearward by one quarter of the
stride. Four such pushes translate the body while returning every foot target
to the next periodic starting position.

The start stance is computed rather than hardcoded. The rectangular shoe's PLA
face is `30 mm` beyond the distal fork axis, and its recommended bonded tread
adds `1 mm`. The gait therefore uses a `159.896689 mm` proximal link and a
`190.896689 mm` effective knee-to-contact length.

The `100 x 60 mm` sole is flat rather than a rocker. Gait V9 keeps every planted
leg at `hip flexion + knee = 0 degrees`, so the distal link and shoe normal
point vertically down throughout weight transfer, swing, touchdown, and the
four-foot push. The previous 3 mm fixed-X support extension was removed because
it deliberately tipped the three planted soles. The swing leg returns to zero
sole pitch before the firm-plant phase.

Because the robot has only two pitch joints per leg, it cannot independently
command swing-foot X, height, and pitch. V9 prioritizes the requested 90-degree
flatness for every loaded shoe; only the unloaded swing shoe may pitch. The
smaller 25 mm lift and 60 mm stride reduce swing excursion relative to V8. Zero
hip abduction and zero lateral shift preserve full 3D flatness for support.

The tracked nominal stance is `329.341 mm` deep with `80 mm` front/rear separation
from each hip and no extra abduction. **SET GAIT START STANCE** moves to the
exact phase-zero flat-sole pose without beginning the crawl. **TEST DISTRIBUTED
CRAWL** repeats its 12-second cycle until **STOP + STABLE HOLD** is pressed. It
uses a `60 mm` stride, `25 mm` contact-centre lift, zero support extension, a
`15 mm` planted-foot push after each placement, and `6 mm` of fore/aft body
transfer. The dedicated crawl ramp is `60 degrees/s`, separate from the
`270 degrees/s` manual/RL ceiling. Each servo update advances by one fixed
50 ms control tick, so a slow telemetry read cannot create a catch-up burst.
Each horizontal swing happens only after the selected rectangular shoe has
lifted. The controller uses smooth interpolation throughout the phase table.
After **SET GAIT START STANCE** reaches `COMPLETE / HOLDING`, the distributed
crawl button becomes available without disarming. The server accepts this
transition only when all 12 motors remain armed at the settled gait targets;
partial or unrelated armed states are still rejected.

The exact equations, phase fractions, sign convention, simulation evidence,
and physical trial ladder are in
[`specs/hardware-gait-sequence-v1.md`](specs/hardware-gait-sequence-v1.md).
The filename is retained so existing documentation links remain stable. The V4
and older documents remain historical evidence.

Before either whole-robot command is accepted, the server requires all 12 IDs
online and every computed target inside the configured joint limits. The
command accepts a minor off-stance measured pose, arms all motors at their
current measured positions, and ramps toward the computed gait-start stance.
It no longer rejects walking solely because a measured joint is outside a
zero-centred start-tolerance window. This does not bypass the calibrated hard
joint-limit validation applied to every computed stance and gait target.
Browser-heartbeat loss or page close is warning-only. It does not cancel the
gait, change targets, or remove torque. Heartbeats continue independently when
a telemetry refresh fails and do not take the servo-bus session lock. A
telemetry read or motion exception, an explicit disarm request, `Ctrl+C`, or
normal server shutdown still attempts to remove torque from all twelve motors.
The normal crawl Stop command instead returns all four feet to the stable gait
stance and holds torque; use **DISARM ALL 12** when it is safe to remove support.
Display-only voltage, temperature, voltage-spread, and
diagnostic-current warnings remain visible but do not stop the command.

When `--port auto` is used, a USB disconnect/reconnect is recovered without a
manual service restart. The failed motion state is cleared, the adapter path is
resolved again (including `ttyACM0`/`ttyACM1` renumbering), all twelve IDs are
revalidated, and every motor is explicitly disarmed before telemetry resumes.
If the complete bus is not yet available, the API stays faulted and retries on
the next telemetry request rather than continuing with a partial robot. While
the adapter is absent, the browser disables motion controls and records one
deduplicated recovery fault instead of logging every telemetry retry or button
press.

The tracked defaults live in `[crawl]` in `../../robot-runtime/four-leg.toml`.
The parser permits a 5-120 mm stride, 5-80 mm lift, and 4-60 second period.
Both dashboard walking buttons loop continuously until **STOP + STABLE HOLD**,
a bus/motion fault, or process shutdown. Closing or backgrounding a mobile
browser does not stop walking or send a disarm request.

### Power and battery analytics

The dashboard estimates electrical power from each servo's diagnostic voltage
and current registers. It shows instantaneous and 60-second average/peak watts,
peak summed current, the lowest recent bus voltage, per-leg and per-motor power,
and sampled watt-hours. The chart is populated by the browser's telemetry poll,
so the energy figure covers observed dashboard time and is not a battery fuel
gauge. It also excludes Raspberry Pi and converter losses.

Use **RESET AT IDLE** after connecting each power source while all motors are
disarmed. Leave the robot idle for several telemetry samples, then run the same
supported gait with the bench supply and battery:

1. A large sag from the idle reference, especially together with a current
   spike and a stall warning, points toward battery internal resistance, BMS or
   connector limits, wiring loss, or inadequate peak-current delivery.
2. A high-current, low-speed motor with at least 8 degrees of tracking error is
   marked as a possible stall. Repeated IDs or one high-power leg suggest joint
   friction, mechanical interference, excess load, or a weak servo branch.
3. If the robot falls without meaningful voltage sag or stall signatures, move
   the same battery mass around while supported and inspect center of mass,
   lateral balance, shoe contact, and traction. Electrical telemetry cannot
   distinguish those mechanical causes by itself.

The tracked warning defaults are 0.6 V sag, 1200 mA per motor, 8 degrees of
tracking error, and raw speed at or below 20. These are diagnostic heuristics,
not certified protection thresholds. Hard joint limits, onboard fault handling,
explicit disarm controls, and the physical cutoff remain the actual safety
layers.

For the tracked 3S LiPo, the same idle reference drives a deliberately coarse
charge indicator: **FULL** at 4.10 V/cell or above, **GOOD** from 3.90 V/cell,
**LOW** from 3.70 V/cell, and **RECHARGE** below 3.70 V/cell. The displayed cell
value is only pack voltage divided by three; it cannot detect an imbalanced or
failing individual cell. Confirm a low/recharge result with a balance-plug cell
checker and follow the battery manufacturer's limits.
The value beside the status is the live measured pack voltage; the smaller line
retains the disarmed idle reference used to classify the charge state.

### Separate diagonal-pair mode

**TEST DIAGONAL PAIRS** starts a separate `/api/diagonal-pair-forward` routine;
it does not replace or redirect **TEST DISTRIBUTED CRAWL**. Front-left and
rear-right lift and advance together, followed by front-right and rear-left.
During each airborne phase the opposite diagonal remains on the exact flat-sole
branch. After the pair plants, all four targets push rearward by half a stride.

The mode shares the conservative 12-second cycle, 60 mm stride, 25 mm lift,
and 60 degrees/s crawl ramp until **STOP + STABLE HOLD** is pressed. Offline
geometry sampling found a periodic path with a `67.25 degree` maximum
hip-flexion target, `73.11 degree` maximum knee target, `144.1 degrees/s` peak
requested rate, and at least `19.43 mm` of long-edge clearance during
horizontal swing. These values fit the tracked hardware profiles and
command-rate cap.

Two diagonal contacts form a support line rather than the three-foot support
polygon used by the original crawl. The routine is therefore more sensitive to
center-of-mass error, compliance, and floor contact. It has not been run in
Isaac or on hardware. The full sequence and target equations are documented in
[`specs/diagonal-pair-gait-v1.md`](specs/diagonal-pair-gait-v1.md).

## Isaac validation status

The retained Isaac result is for the previous `24 mm` V8 stride, not the active
V9 60 mm hardware profile. It used two cycles with rectangular `100 x 60 x 6
mm` PLA collision boxes and `94 x 54 x 1 mm` tread boxes. The robot remained
upright and moved forward, but the strict result is **FAIL**:
peak tracking error was `0.223851 rad` and only the rear-right and rear-left
placements passed the per-foot force/contact-duration gate. The rigid model can
support the broad coplanar shoes on a diagonal pair, so this result is stable
preliminary evidence rather than proof that every physical shoe is loaded.

| Metric | V8 24 mm two-cycle baseline |
| --- | ---: |
| Forward displacement | 53.31 mm |
| Lateral drift | 0.33 mm |
| Maximum body tilt | 1.84 degrees |
| Minimum base height | 0.3738 m |
| Peak / RMS joint error | 0.2239 / 0.0416 rad |
| Maximum joint speed | 2.269 rad/s |
| Expected support contact | 87.56% |
| Maximum support slip | 5.34 mm |
| Contact-verified placements | rear right, rear left |

The report and screenshot are
[`validation/isaac-rectangular-flat-crawl-v8.json`](validation/isaac-rectangular-flat-crawl-v8.json)
and
[`validation/isaac-rectangular-flat-crawl-v8.png`](validation/isaac-rectangular-flat-crawl-v8.png).
The proxy omits the CAD-estimated `70.237 g` mass of each PLA shoe, adhesive
compliance, 2 mm corner radii, and physical backlash.

The evidence below is retained historical V5/TPU evidence and must not be
attributed to V8.

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
After the shoe update, one peak/stall-cap quick check used the then-current `112 mm`
stride, `60 mm` lift, `6.67 s` period, `357 mm` downward reach, and a rigid
`54 x 48 x 48 mm` capsule approximation for each TPU shoe. The run stayed
upright, moved `115.61 mm` forward, and reached `3.83 degrees` maximum tilt,
but the result is **FAIL**, not a validated gait: maximum joint error was
`0.481 rad`, support slip was `34.54 mm`, and the final front-left step did not
meet the contact-duration gate. The report is
[`validation/isaac-tpu-shoe-crawl-quick.json`](validation/isaac-tpu-shoe-crawl-quick.json).
The proxy omits TPU compliance, shoe mass, vents, and changing rocker contact.
Simulation remains a geometry and dynamics screen, not permission for an
untethered first physical run.

The first physical trial with the TPU shoes and `112 mm` stride fell forward.
The subsequent `56 mm` profile fell backward. For that historical V5 work, the
rounded shoes were treated as possible energy-storage elements and the
simulator gave each proxy a separate provisional compliant TPU contact
material. The selected assumptions were static friction `1.05`, dynamic
friction `0.85`, restitution `0.03`,
contact stiffness `8000 N/m`, and contact damping `45 N s/m`. These are tuning
assumptions, not measured properties of the printed shoe.

The comparison screened shorter strides, lower lift, slower periods, lateral
weight transfer, and two compliance/damping ranges. The slow 12-second trials
and the 8 mm lateral-shift trial collapsed. The least unstable result used a
`40 mm` stride, `40 mm` lift, 8-second period, and `12 mm` fore/aft transfer.
It stayed upright, moved `50.60 mm` forward, reached `7.10 degrees` maximum
tilt, limited maximum joint error to `0.296 rad`, and limited support slip to
`11.10 mm`. It is still a strict **FAIL** because `0.296 rad` exceeds the
`0.15 rad` tracking limit and rear-left did not maintain the required
three-foot support duration. The comparison used Isaac's short-duration
peak/stall torque cap; the physical profiles retain their configured 90%
limit. This was the next candidate before the rigid shoe superseded it; it was
never a validated autonomous gait.

Historical flexible-shoe evidence:

- [`validation/isaac-tpu-flex-crawl-40mm-8s-moderate.json`](validation/isaac-tpu-flex-crawl-40mm-8s-moderate.json)
- [`validation/isaac-tpu-flex-crawl-40mm-8s-moderate.png`](validation/isaac-tpu-flex-crawl-40mm-8s-moderate.png)

That historical TPU comparison selected a `40 mm` profile. The V8 rectangular
profile that followed used a 96 mm stride and 4-second period, but the hardware
trial showed 20-39 degree tracking error and unstable motion. The active V9
profile is deliberately more conservative: 60 mm stride, 25 mm lift, 12-second
period, longer contact/settle phases, and a 60 degrees/s crawl ramp. It has not
been run in Isaac or tested on hardware yet; no result is claimed from the
older reports.

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

## Automatic RL trial recordings

Every port-8080 RL walk now records the exact policy inputs and outputs plus
lower-rate motor diagnostics without adding disk access to the 60 Hz control
thread. Open **Settings → Trial recordings** to name, download, or delete a
finalized trial. Data is stored under
`~/.local/share/drobot2/rl-recordings` by default; pass `--recordings-dir` to
change it. The format, field meanings, dropped-sample behavior, analysis uses,
and future ROS 2/rosbag2 boundary are documented in
[`docs/rl-recordings.md`](docs/rl-recordings.md).

## Understanding “power OK”

The dashboard uses attention thresholds from `../../robot-runtime/four-leg.toml`:

| Signal | Dashboard attention threshold | Meaning |
| --- | ---: | --- |
| Servo voltage | below 11.0 V | Earlier warning for supply sag under the higher torque cap |
| Servo voltage | above 12.6 V | Above the documented 12 V ST3215 supply maximum |
| Voltage spread | above 0.3 V | Earlier branch-wiring and voltage-drop warning |
| Servo temperature | 55 C or above | Begin a five-second RL confirmation window; a normal re-read clears it |
| Servo temperature | 65 C or above | Immediate critical RL stop and disarm |
| Per-leg diagnostic current | 2500 mA or above | Earlier load and connector warning |

These are software thresholds, not firmware protection settings. During an RL
walk, a `55-64 C` sample is prioritized for repeated reads and stops the run
only after it remains high for five seconds with at least three high samples. A
normal sample resets the confirmation, while `65 C` or above immediately stops
and disarms. Voltage, voltage-spread, and diagnostic-current warnings remain
display-only. Actual telemetry read exceptions, motion exceptions, and explicit
stop/disarm commands still disarm. Browser-heartbeat age is diagnostic only.
Servo voltage does not show individual LiPo cell balance. Servo current feedback
is diagnostic and is not a calibrated battery, branch, force, or joint-torque
measurement. Use an inline watt meter or clamp meter and per-cell checker for
electrical validation. The dashboard never declares the entire power system
certified.

## Motion behavior

Arming first commands the motor's measured position and only then enables
torque. Browser destinations are converted with the tracked calibration and
range checked before acceptance. The background worker can approach manual and
RL destinations at up to the configured 270 degrees per second through command
steps no larger than 15 degrees. The hardcoded crawl has its own 60 degrees/s
ceiling and advances from a fixed 50 ms gait tick to avoid catch-up bursts.
Each servo profile uses speed register `3400`, acceleration `254`, and the
existing 90% torque limit.

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
