# Isaac Sim 6.0 quadruped checks

This is the owning guide for the robot's Isaac import, articulation, sensors,
gravity checks, scripted gait, and reinforcement-learning handoff. Start with
the committed world for learning and inspection; rebuild the USD assets only
after the URDF changes.

## How the files connect

The simulation has one direction of data flow:

```text
robot_cad/urdf/quadruped_robot.py
                 |
                 v
exports/urdf/quadruped_robot.urdf
                 |
                 v
simulation/isaac/import_quadruped.py
          |                         |
          v                         v
fixed robot USDC             floating robot USDC
          |                         |
          |                         v
          |          simulation/isaac/create_manual_world.py
          |                         |
          |                         v
          |       exports/isaac/quadruped_robot_manual_world.usda
          |                         |
          v                         v
 camera validation       manual control, IMU, standing,
                         crawl, PPO training, PPO evaluation
```

The URDF defines the links, inertias, collision shapes, joint axes and limits,
and actuator limits. The importer turns that description into a USD
articulation and adds the Isaac camera, IMU, and self-collision filters. The
manual-world builder references the floating articulation and adds the floor,
gravity, contact material, starting height, rated servo cap, and standing
targets. Runtime scripts then open that same world; they do not construct a
different robot.

Use the two imported variants for different questions:

| Variant | Base behavior | Use it for |
| --- | --- | --- |
| fixed | base is attached to the world | joint structure and camera validation |
| floating | base responds to gravity and contact | standing, gait, IMU, manual control, and RL |
| one-leg wall testbed | the physical 76 mm hip plate is fixed flush to a vertical wall | isolated three-joint range, wall clearance, gravity load, and direct joint control |

All robot coordinates are SI units with `+X` forward, `+Y` left, and `+Z` up.
Joint positions are radians in Python even though USD angular-drive target
attributes are authored in degrees.

## How a Python command launches Isaac

Use `C:\isaacsim\python.bat`, not the project's ordinary Python, for every
script that imports `isaacsim`. `python.bat` selects Isaac's bundled Python and
its Omniverse extension paths. Each entry-point then:

1. parses its command-line options;
2. constructs `SimulationApp`, which starts Kit/Isaac in headless or GUI mode;
3. imports Omniverse APIs after the app exists;
4. opens or creates a USD stage;
5. runs physics and reads the articulation or sensors;
6. writes a JSON report and returns a non-zero process exit code on failure;
7. closes `SimulationApp`.

You do not need to launch Isaac separately before using these commands. A GUI
script such as `open_articulation.py` starts Isaac and opens the requested
world for you. Double-clicking the USDA in Isaac is also possible, but the
launcher is more reproducible because it verifies the robot path, 12 joints,
13 imported rigid bodies, camera, IMU, and rated effort cap first.

Pure NumPy contract tests do use ordinary project Python because they do not
start the simulator:

```powershell
python -m pytest `
  tests\test_quadruped_isaac_runtime.py `
  tests\test_quadruped_imu_observation.py `
  tests\test_quadruped_rl_contract.py
```

## One-leg wall testbed

The isolated wall setup answers the same question as the physical home test
without the quadruped body, other legs, ground, or stair contacts. Its fixture
frame uses `+X` forward along the wall, `+Y` away from the wall, and `+Z` up.
The exact printable body-side mount is shown with its 76 x 76 mm plate back
face flush to the vertical wall. The moving hip and two arm links reuse the
quadruped URDF geometry, collision proxies, joint frames, masses, and
inertias. There is deliberately no virtual foot sphere.

The tracked model records the locally calibrated 2026-07-28 physical setup:

| Motor | Logical testbed range | Encoder direction | Positive model motion |
| --- | ---: | ---: | --- |
| 1 hip abduction | `-45 to +45 deg` | `+1` | leg moves away from the wall |
| 2 hip flexion | `-90 to +90 deg` | `-1` | hanging leg moves toward wall-frame `+X` |
| 3 knee | `-120 to +120 deg` | `-1` | distal link bends toward wall-frame `+X` at zero flexion |

All three captured center ticks are `2048`. These are physically exercised
isolated-testbed ranges, not automatically approved whole-robot limits. Body
interference, four-leg cable routing, support contact, loaded current, and
continuous thermal behavior still require separate checks.

Generate, import, and sweep the explicit artifacts from `robot-cad`:

```powershell
$projectPython = '.\.venv\Scripts\python.exe'
$urdfTool = 'C:\Users\roman\.codex\plugins\cache\text-to-cad\cad\0.3.9\skills\urdf\scripts\urdf'
$isaacPython = 'C:\isaacsim\python.bat'

& $projectPython $urdfTool `
  robot_cad\urdf\one_leg_wall_testbed.py `
  -o exports\urdf\one_leg_wall_testbed.urdf

& $isaacPython simulation\isaac\import_one_leg_wall.py `
  --urdf exports\urdf\one_leg_wall_testbed.urdf `
  --output exports\isaac\one_leg_wall_testbed.usdc `
  --report simulation\isaac\output\one-leg-wall\import_report.json

& $isaacPython simulation\isaac\run_one_leg_wall.py `
  --gravity both `
  --screenshot reviews\isaac-one-leg-wall-range.png
```

The automatic runner verifies the local `leg.toml` ranges/directions and
`calibration.json` centers when those ignored machine-local files exist. It
then checks each joint at two degrees inside both hard endpoints and checks
two combined poses, first with zero gravity and then with Earth gravity.
The default `0.8825985 N*m` effort cap is 30% of published stall torque to
mirror the hardware register value of 300 only nominally; the ST3215 register
is not a calibrated linear joint-torque sensor.

For direct viewport control of the same articulation:

```powershell
& $isaacPython simulation\isaac\run_one_leg_wall.py --interactive
```

Use `1`, `2`, or `3` to select the joint, hold `Up`/`Down` to move its target,
press `Z` to zero it, `G` to toggle gravity, `R` to reset all targets, `C` to
print state, and `X` or `Esc` to save the session report and exit. Wall
contact, hard joint limits, velocity limits, moving-link self-collision, drive
gains, and the selected effort cap remain active.

The 2026-07-28 Isaac 6.0.1 run imported one articulation root, four rigid
bodies, three revolute drives, and two filtered moving-pivot pairs. With wall
contact active, the isolated leg produced:

| Test | Zero gravity | Earth gravity |
| --- | ---: | ---: |
| hip abduction toward wall, target `-43 deg` | stopped at `-10.250 deg` | stopped at `-10.250 deg` |
| hip abduction away from wall, target `+43 deg` | `42.999 deg` | `41.377 deg` after 2 s |
| hip flexion, targets `+/-88 deg` | max error `0.0019 deg` | max error `1.4932 deg` after 2 s |
| knee, targets `+/-118 deg` | max error `0.0038 deg` | max error `0.3519 deg` after 2 s |
| combined `(30, 60, -90) deg` | max error `0.0020 deg` | max error `1.0537 deg` after 2 s |

The reverse combined pose also stopped only on inward hip abduction. Repeating
the zero-gravity sweep with `--disable-wall-contact` made all eight endpoint
and combined poses pass; the worst error was `0.00382 deg`. This diagnostic
proves the negative-abduction stop is the flush wall envelope, not a frozen
Isaac drive. Keep wall contact enabled for the physical fixture; use the
option only to compare a clamp, stand, or wall-edge setup that gives the leg
clearance behind the plate.

Two follow-up probes on the unchanged full quadruped fixed asset also passed:
front-left hip flexion reached `40 deg` with `0.00093 deg` error and the knee
reached `70 deg` with `0.00196 deg` error, both reaching 90% motion in about
`0.383 s` under zero gravity. The prior full-scene failure therefore did not
show that Isaac was unable to move the arm. The stair gate remains a loaded
floating-base support/contact problem: its IK targets were already inside the
old limits, while the previous run saturated rated effort, lost three-foot
support, slipped, and tipped before achieving the requested foot lift.

Local reports are:

- `simulation/isaac/output/one-leg-wall/import_report.json`
- `simulation/isaac/output/one-leg-wall/range_report.json`
- `simulation/isaac/output/one-leg-wall/range_report_no_wall_contact.json`
- `simulation/isaac/output/motor-physics-audit/wall-followup-hip-flexion.json`
- `simulation/isaac/output/motor-physics-audit/wall-followup-knee.json`

The reviewed render is `reviews/isaac-one-leg-wall-range.png`.

## First interactive run

From the `robot-cad` directory in a fresh clone:

```powershell
git lfs pull
$isaacPython = 'C:\isaacsim\python.bat'
if (-not (Test-Path -LiteralPath $isaacPython)) {
  throw "Isaac Sim Python was not found at $isaacPython"
}

& $isaacPython simulation\isaac\open_articulation.py `
  --world exports\isaac\quadruped_robot_manual_world.usda `
  --report simulation\isaac\output\manual-open.json
```

The committed `quadruped_robot_manual_world.usda` references
`quadruped_robot_floating.usdc` in the same `exports/isaac` directory. Git LFS
must therefore download both files, and they must stay together.

Once the window opens:

1. wait for the robot and floor to appear;
2. press **Play** if the timeline is paused;
3. open **Physics > Articulation Inspector**;
4. select `/World/Robot`;
5. move one joint by a small amount and observe the corresponding leg;
6. stop the timeline before closing the application.

The launcher applies the conservative standing target once, then stops sending
commands. The Inspector therefore owns subsequent target changes. Add
`--onboard-camera` to begin in the mounted camera view. Run with
`--headless --smoke-seconds 2` when only the automated
open/physics/sensor check is needed.

## Script map

Every maintained simulation source file is indexed here. Files beginning with
an underscore are imported helpers and are not launched directly.

| Source | What it does | Reads | Writes or changes |
| --- | --- | --- | --- |
| `scripts/setup_isaac_rl.ps1` | Installs and verifies the pinned PPO dependencies in Isaac's Python | `simulation/isaac/rl/requirements.txt` | Isaac Python environment |
| `simulation/isaac/import_quadruped.py` | Imports fixed or floating URDF, authors camera/IMU/self-collision metadata, validates structure, optionally flattens to one USDC | generated URDF and its meshes | imported USD plus import JSON |
| `simulation/isaac/import_one_leg_wall.py` | Imports the physically configured fixed wall-mounted one-leg URDF and preserves wall contact while filtering two intentional moving-pivot overlaps | generated one-leg URDF and its meshes | monolithic one-leg USDC plus import JSON |
| `simulation/isaac/run_one_leg_wall.py` | Runs zero/Earth-gravity endpoint and combined-pose sweeps or opens direct 1/2/3 joint viewport control | one-leg USDC and optional local hardware config/calibration | range or interactive JSON and optional PNG |
| `simulation/isaac/create_manual_world.py` | Builds the portable gravity/contact world around the floating asset | floating USDC | manual-world USDA plus JSON |
| `simulation/isaac/open_articulation.py` | Starts the GUI or a headless smoke run and leaves manual GUI control unopposed | manual-world USDA | optional open-run JSON |
| `simulation/isaac/validate_quadruped.py` | Checks fixed structure or floating standing stability, joint tracking, effort, tilt, and height | fixed or floating USD | validation JSON and optional PNG |
| `simulation/isaac/run_crawl.py` | Commands the scripted crawl or contact-verified quasi-static sequence | floating USD | gait metrics JSON and review PNG |
| `simulation/isaac/validate_camera.py` | Renders known colored targets through the mounted RTX camera and validates RGB/depth buffers and orientation | fixed USDC | camera JSON and onboard PNG |
| `simulation/isaac/validate_imu.py` | Holds the stand, samples the body IMU, and validates gravity, quaternion, rates, and the nine-value IMU block | manual-world USDA | IMU JSON |
| `simulation/isaac/_quadruped_runtime.py` | Owns joint names, servo torque/speed constants, leg inverse kinematics, stance, crawl, and target ordering | no simulator asset | Python values used by validation, gait, and RL |
| `simulation/isaac/_imu_observation.py` | Converts Isaac IMU frames into hardware-oriented body axes and a finite nine-value observation | IMU frame dictionaries | NumPy observation arrays |
| `simulation/isaac/rl/quadruped_walk_v1.yaml` | Versions the task, action scale, reward, reset, termination, and PPO settings | edited by the trainer and evaluator | no generated data |
| `simulation/isaac/rl/requirements.txt` | Pins Gymnasium, Stable-Baselines3, TensorBoard, and YAML packages | setup script | no generated data |
| `simulation/isaac/rl/_rl_contract.py` | Pure NumPy definition of the 48-value observation, reward terms, and termination decisions | state arrays and YAML values | observations and scalar reward breakdown |
| `simulation/isaac/rl/_quadruped_rl_env.py` | Implements Gymnasium `reset()`/`step()` against the real Isaac articulation and IMU | manual world and task config | actions, observations, rewards, episode state |
| `simulation/isaac/rl/train_ppo.py` | Creates `SimulationApp`, wraps the environment, trains/resumes PPO, checkpoints, and records provenance | YAML, manual world, optional checkpoint | model ZIP, TensorBoard, monitor CSV, training JSON |
| `simulation/isaac/rl/play_ppo.py` | Reloads a saved policy and runs deterministic evaluation episodes | YAML, manual world, model ZIP | evaluation JSON and optional PNG |
| `simulation/isaac/rl/record_ppo.py` | Records one deterministic flat-policy episode through an external or onboard RTX camera | YAML, manual world, model ZIP | H.264 MP4, thumbnail PNG, recording JSON |
| `simulation/isaac/rl/stairs/__init__.py` | Marks the stair experiment as a separate Python source package | no runtime input | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v1.yaml` | Versions the stair geometry, terrain inputs, curriculum, reward, reset, termination, action, and PPO settings | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v2.yaml` | Corrects the stair task with a close reset, navigation observations, physical-height reward, mastery curriculum, bounded exploration, and progress watchdog | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/_stair_geometry.py` | Defines the crack-free stacked collision layers for the four-step staircase | stair YAML values | pure Python geometry facts |
| `simulation/isaac/rl/stairs/_stair_rl_contract.py` | Defines analytic terrain sampling, 57/60-value observation contracts, progress gates, curriculum goals, physical-height reward terms, and failure reasons | walking observation and stair YAML values | observations and scalar contract results |
| `simulation/isaac/rl/stairs/_quadruped_stairs_env.py` | Extends the walking environment with exact DOF order, two physics updates per action, stair-relative state, curriculum, and episode metrics | stair world and task config | actions, observations, rewards, episode state |
| `simulation/isaac/rl/stairs/_policy_transfer.py` | Strictly copies 11 tensors and expands two input matrices from a verified 48/12 ELU flat policy, with no skipped tensors | source and target policy tensors | transferred policy state and report |
| `simulation/isaac/rl/stairs/_run_support.py` | Creates/verifies schema-2 model, config, world/dependency, environment, PPO-mode, transfer, and resume contracts | model, YAML, composed world files, runtime/PPO contracts | adjacent `.contract.json` manifests |
| `simulation/isaac/rl/stairs/create_stairs_world.py` | Sublayers the validated manual world and authors static stair collision layers | base world and stair YAML | stair-world USDA and static validation JSON |
| `simulation/isaac/rl/stairs/train_stairs_ppo.py` | Trains, transfers, or resumes the separate stairs PPO policy with timestep/mastery curriculum, checkpoint manifests, live stair metrics, and automatic no-progress abort reports | stair YAML/world and optional model | model ZIPs, manifests, TensorBoard, monitor CSV, progress watchdog, training JSON |
| `simulation/isaac/rl/stairs/evaluate_stairs_ppo.py` | Runs deterministic stair episodes at a pinned curriculum level and verifies the model contract | stair YAML/world, model, manifest | evaluation JSON and optional PNG |
| `simulation/isaac/rl/stairs/record_stairs_ppo.py` | Records one deterministic stair-policy episode through an external or onboard RTX camera | stair YAML/world, model, manifest | H.264 MP4, thumbnail PNG, recording JSON |
| `simulation/isaac/experiments/stair_feasibility/__init__.py` | Marks the scripted real-stair feasibility experiment as a separate Python package | no runtime input | no generated data |
| `simulation/isaac/experiments/stair_feasibility/real_stair_feasibility.yaml` | Versions the isolated block geometry, scripted motion, contact model, rated torque cap, and pass/fail thresholds | edited feasibility configuration | no generated data |
| `simulation/isaac/experiments/stair_feasibility/_contract.py` | Defines pure IK targets, support-triangle margin, configuration validation, and trial gates | feasibility YAML and measured metrics | analytic targets and gate failures |
| `simulation/isaac/experiments/stair_feasibility/run_real_stair_feasibility.py` | Runs non-RL front-foot placement against `100-196 mm` risers under floating-base dynamics | YAML and floating robot USDC | JSON report and per-height screenshots |
| `simulation/isaac/experiments/stair_feasibility/_manual_control.py` | Defines safe selectable-leg foot targets, held-key motion, IK rejection, and reset state without simulator imports | feasibility YAML and keyboard state | named joint-position targets |
| `simulation/isaac/experiments/stair_feasibility/manual_180mm_stair.py` | Opens a GUI beside one `180 mm` riser for direct keyboard leg control with all floating-base physics active | feasibility YAML, floating robot USDC, and keyboard input | live viewport and manual-session JSON |
| `simulation/isaac/experiments/stair_feasibility/manual_180mm_motor_angles.py` | Selects the direct numbered-motor mode of the interactive `180 mm` runner | command-line arguments and shared runner | live motor-angle control panel and session JSON |
| `simulation/isaac/experiments/stair_feasibility/probe_motor_physics.py` | Measures one fixed- or floating-base motor with zero gravity and no environment contacts | curated robot USDC and drive parameters | sampled response JSON |

## What happens during one commanded step

The scripted gait and RL environment both command position targets, not raw
mesh transforms:

```text
desired body/foot motion
        -> leg inverse kinematics or policy action scaling
        -> 12 named joint-position targets
        -> URDF/Isaac angular drives
        -> PhysX forces, limited to the ST3215 rated torque
        -> rigid-body motion and contacts
        -> articulation state + body IMU
        -> metrics, next observation, reward, or termination
```

The policy action is limited, clamped to URDF joint limits, and rate-limited
using the ST3215 no-load speed. PhysX still decides whether the commanded pose
is achievable under mass, gravity, contact, friction, and the effort cap.
This is why a target trajectory is not evidence that the robot can walk: the
reported body displacement, support contacts, slip, tilt, and joint error are
the result that matters.

## Output and version-control policy

- Editable simulation code, YAML, dependency pins, and this guide are committed.
- Curated handoff assets in `exports/isaac/` are committed through Git LFS.
- Repeatable run products under `simulation/isaac/output/` are ignored:
  imported scratch USD, JSON logs, PPO checkpoints, TensorBoard events, and
  local probes can be large or machine-specific.
- Intentional review PNGs belong in `reviews/` and are committed through Git
  LFS only when they communicate a specific reviewed result.
- A `PASS` report means the checks coded by that runner passed. A smoke test
  means the pipeline executed; it does not mean a walking policy converged.

For debugging, open the JSON report before changing thresholds. It records the
resolved source asset, simulator version, structural counts, options, metrics,
status, and traceback on failure.

## Rebuild the imported Isaac assets

These runners use the generated URDF as the physical source of truth and must
be launched with Isaac Sim's bundled Python. `monolithic` produces one
self-contained binary USDC, which is the preferred handoff format:

```powershell
$isaacPython = 'C:\isaacsim\python.bat'
$urdf = (Resolve-Path 'exports\urdf\quadruped_robot.urdf').Path

& $isaacPython simulation\isaac\import_quadruped.py `
  --urdf $urdf `
  --output-dir simulation\isaac\output\monolithic\fixed `
  --mode fixed `
  --asset-layout monolithic `
  --report simulation\isaac\output\monolithic\fixed-import.json

& $isaacPython simulation\isaac\import_quadruped.py `
  --urdf $urdf `
  --output-dir simulation\isaac\output\monolithic\floating `
  --mode floating `
  --asset-layout monolithic `
  --report simulation\isaac\output\monolithic\floating-import.json
```

Each import report must show one articulation root, 13 rigid bodies, 12
revolute joints, 12 angular drives, one RTX camera, and one IMU sensor. The
camera and IMU fixed URDF joints are merged into `base_link`, preserving the
proven 13-body articulation. Use the report's `root_usd` value for validation.
`--asset-layout packaged` remains available when a multi-file, multi-physics
payload package is specifically needed; keep that entire generated directory.

Build a scratch manual world from the newly imported floating asset:

```powershell
$floatingReport = Get-Content `
  simulation\isaac\output\monolithic\floating-import.json | ConvertFrom-Json

& $isaacPython simulation\isaac\create_manual_world.py `
  --usd $floatingReport.root_usd `
  --output simulation\isaac\output\monolithic\quadruped_robot_manual_world.usda `
  --report simulation\isaac\output\monolithic\manual-world.json
```

Open that scratch world with `open_articulation.py` and run validation before
replacing the curated files in `exports/isaac/`. Promotion is deliberately not
automatic: a new URDF can import successfully while still having incorrect
mass, collision, joint orientation, sensor orientation, or standing behavior.

Self-collision is enabled by default. The importer preserves collision between
different legs and between non-adjacent robot links, while authoring exactly
12 filtered pairs for directly connected joint neighbors whose CAD envelopes
overlap at their pivots. Use `--disable-self-collision` only as a troubleshooting
comparison; it permits legs to pass through each other.

```powershell
& $isaacPython simulation\isaac\validate_quadruped.py `
  --usd <fixed-root.usda> `
  --mode fixed `
  --torque-cap rated `
  --report simulation\isaac\output\fixed-rated.json `
  --screenshot reviews\isaac-fixed-rated.png

& $isaacPython simulation\isaac\validate_quadruped.py `
  --usd <floating-root.usda> `
  --mode floating `
  --torque-cap rated `
  --report simulation\isaac\output\standing-rated.json `
  --screenshot reviews\isaac-standing-rated.png

& $isaacPython simulation\isaac\run_crawl.py `
  --usd <floating-root.usda> `
  --headless `
  --torque-cap rated `
  --cycles 4 `
  --report simulation\isaac\output\crawl-rated.json `
  --screenshot reviews\isaac-crawl-rated.png
```

The contact-verified quasi-static trial uses a straighter stance and explicit
weight transfer before each foot is lifted:

```powershell
& $isaacPython simulation\isaac\run_crawl.py `
  --usd <floating-root.usda> `
  --headless `
  --gait-mode quasi-static `
  --torque-cap rated `
  --cycles 2 `
  --period 20 `
  --stride 0.015 `
  --lift 0.010 `
  --weight-shift-forward 0.030 `
  --weight-shift-lateral 0 `
  --stance-down 0.310 `
  --stance-fore-aft 0.025 `
  --abduction-deg 0 `
  --start-z 0.460 `
  --review-phase 0.11 `
  --report simulation\isaac\output\quasistatic-rated.json `
  --screenshot reviews\isaac-quasistatic-rated.png
```

In this mode, all four distal links have ground-filtered contact tracking.
Acceptance requires the three support contacts to remain loaded, each swing
foot to unload and clear the floor, each touchdown to persist, and planted
fork tips to stay within the configured slip limit.

Repeat the standing and crawl runs with `--torque-cap stall` only as a short
peak-torque comparison. The rated profile is the sustainable ST3215 test.
Both runners accept `--effort-limit-nm` for an explicit custom cap.

The default acceptance thresholds are deliberately strict: ten seconds of
stable standing, four crawl cycles, at least 20 mm forward travel, less than
50 mm lateral drift, less than 25 degrees body tilt, and less than 0.15 rad
maximum joint error. Do not weaken them merely to turn a shuffle into a pass.

The present model has no printed feet. Its distal fork-tip collision contacts
are intentionally recorded as an approximation in every report, so a simulated
pass does not replace printed-foot, floor-friction, or endurance testing.

## Mounted camera

The LeKiwi-compatible Arducam model is represented by `camera_link` plus the
massless `camera_optical_frame`. The Isaac asset contains a camera at:

```text
/World/Robot/Geometry/base_link/lekiwi_camera
```

The default simulation profile is 640 x 480 RGB and depth at 30 Hz, with a
95-degree horizontal field of view. Validate the actual RTX render buffers:

```powershell
& C:\isaacsim\python.bat simulation\isaac\validate_camera.py `
  --usd exports\isaac\quadruped_robot_fixed.usdc `
  --output reviews\isaac-lekiwi-camera-rgb.png `
  --report simulation\isaac\output\camera-validation.json
```

The validator uses Isaac Sim 6.0's
`isaacsim.sensors.experimental.rtx.CameraSensor`, reads both `rgb` and
`distance_to_image_plane`, checks the mounted orientation, and writes an
onboard RGB frame.

## Body IMU

The exact Adafruit BNO085 board is mounted on the electronics tray with its
sensing element at the body centre. Its URDF and Isaac axes are the robot body
axes: `+X` forward, `+Y` left, and `+Z` up. The Isaac prim is:

```text
/World/Robot/Geometry/base_link/body_imu
```

Validate the experimental-physics sensor, gravity direction, quaternion, and
nine-value walking-policy observation:

```powershell
& C:\isaacsim\python.bat simulation\isaac\validate_imu.py `
  --world exports\isaac\quadruped_robot_manual_world.usda `
  --settle-seconds 3 `
  --sample-seconds 1 `
  --report simulation\isaac\output\imu-validation.json
```

The output order is body angular velocity, projected unit gravity, and linear
acceleration divided by `9.81 m/s^2`. Concatenate that block with commands,
joint state, and previous actions for walking training. Add measured bias,
latency, vibration, quantization, and dropout as domain randomization before
claiming sim-to-real fidelity.

## PPO reinforcement-learning task

The executable `Drobot-Quadruped-Walk-v1` task is under
`simulation/isaac/rl/`. It uses the same manual world, 13-body articulation,
rated ST3215 effort cap, and mounted IMU validated above.

Install the tested Isaac Python dependencies once:

```powershell
.\scripts\setup_isaac_rl.ps1
```

Run a short end-to-end smoke test:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\train_ppo.py `
  --smoke-test `
  --output-dir simulation\isaac\output\rl\smoke-v1
```

Run the configured 500,000-step training job:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\train_ppo.py `
  --output-dir simulation\isaac\output\rl\ppo-walk-v1
```

The version-1 policy has 12 normalized joint-offset actions and a 48-value
hardware-reproducible observation: command (3), IMU (9), joint-position error
(12), normalized joint velocity (12), and previous action (12). Base linear
velocity is simulator-only reward information and is not exposed to the
policy. The camera is present for evaluation but is not a training input.

Full configuration, reward terms, termination conditions, checkpoints,
TensorBoard, evaluation commands, limitations, and the Isaac Lab migration
rationale are documented in
[`docs/rl-training.md`](../../docs/rl-training.md).

## Separate stair-climbing PPO task

`Drobot-Quadruped-Stairs-v1` lives under
`simulation/isaac/rl/stairs/` and writes separate models under
`simulation/isaac/output/rl/ppo-stairs-*`. It composes a fixed four-step
staircase over the validated manual world: four `40 mm` rises, `230 mm`
treads, `1.00 m` width, and a `0.50 m` top platform.

The policy has 12 normalized joint-offset actions and 57 observations: the
48-value walking input plus eight analytic forward terrain-height deltas and
one goal-distance value. The MLP uses separate `256 x 256` ELU actor and
critic paths. Physics runs at 120 Hz, control at 60 Hz, and this environment
explicitly performs two physics updates per action. Its curriculum moves the
success goal from one through four stairs while keeping the whole staircase
present. The last 12 base-observation values describe the action just applied;
the action-rate reward compares that action with a separately saved prior
action. Completion earns `+400`, chosen to beat discounted stationary
loitering at `gamma = 0.995`.

Generate the stair world, then run the tested transfer smoke:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\create_stairs_world.py

& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --smoke-test `
  --initialize-from-flat simulation\isaac\output\rl\ppo-walk-v1-2m\drobot_walk_ppo_final.zip `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v1-smoke
```

Recorded on 2026-07-27, world generation passed with four static collision
layers, expected robot/sensor prim counts, and both composed dependency hashes.
The corrected 512-step smoke run passed in 13.950 seconds, saved a
164,633-parameter model plus schema-2 manifest, copied 11 flat-policy tensors,
expanded exactly two input layers, and skipped none. A two-episode
deterministic level-1 reload passed manifest and loaded-algorithm checks but
completed `0/2` stairs: both episodes tipped and mean highest step was `0.0`.
The corrected level-1 recording passed the same checks and encoded 660 H.264
frames at `960 x 540`, 30 FPS, but timed out at 22 seconds without reaching a
stair. These validate the pipeline, not stair learning.

The v1 full run was stopped at 964,608 steps after a five-episode audit of its
950k checkpoint found `0/5` first-step reaches. Mean forward displacement was
`0.735 m`, lateral drift reached `0.410-0.504 m`, and three episodes left the
corridor despite a high mean return. This proves the v1 reward/curriculum
optimized approach walking without stair contact; its checkpoints are retained
for diagnosis, not continued as a stair solution.

`Drobot-Quadruped-Stairs-v2` is the corrected experiment. It resets the base at
`x=0.18-0.24 m`, only `0.31-0.37 m` before the first riser; adds normalized
lateral position and heading sine/cosine for a 60-value observation; rewards
actual base-height change at `150 x delta-z`; reduces inherited exploration to
an initial standard deviation near `0.50`; tightens the approach corridor; and
advances curriculum levels only after a 70% recent success rate. A progress
watchdog writes `progress_watchdog.json`, requires at least three physically
elevated first-step reaches by 100k steps, and returns
`ABORTED_NO_PROGRESS` if the gate fails.

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\create_stairs_world.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v2.yaml `
  --report simulation\isaac\output\rl\ppo-stairs-v2\world_report.json

& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v2.yaml `
  --initialize-from-flat simulation\isaac\output\rl\ppo-walk-v1-2m\drobot_walk_ppo_final.zip `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v2
```

The v2 world-generation and 512-step transfer smoke paths passed on
2026-07-27. The smoke is pipeline validation only; the 100k progress gate is
the first behavioral decision point.

The source map, complete commands, reward/termination formulas, manifest
rules, measured smoke results, evaluation guidance, and sim-to-real limits are
owned by [`docs/rl-stairs/README.md`](../../docs/rl-stairs/README.md), with
the corrected experiment recorded separately in
[`docs/rl-stairs-v2/README.md`](../../docs/rl-stairs-v2/README.md).

## Real-stair feasibility gate

Before more stair PPO training, run the isolated non-RL foot-placement test:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\experiments\stair_feasibility\run_real_stair_feasibility.py `
  --config simulation\isaac\experiments\stair_feasibility\real_stair_feasibility.yaml `
  --output-dir simulation\isaac\output\stair-feasibility-v1 `
  --headless
```

It places one `280 mm`-deep block in front of the floating robot and commands
scripted inverse-kinematics front-foot placement at `100`, `140`, `180`, and
`196 mm` riser heights. It evaluates ideal joint-limit reachability, physical
foot lift, collision, three-foot support, slip, body stability, tracking error,
and rated-torque saturation. No RL model is loaded.

The 2026-07-27 run returned `FAIL` for all four heights and set
`curriculum_authorized` to `false`. Ideal targets fit the URDF hard limits, but
the robot achieved only `16.0`, `20.8`, `26.2`, and `36.7 mm` of foot lift,
respectively; none cleared the edge or contacted the tread. The taller cases
lost support and tipped into the block. This blocks further stair PPO training
until foot contact, weight transfer, and sustainable actuator capability are
revised and the rated-torque physical gate passes.

The complete contract, measured table, report locations, exact tests,
limitations, and revision order are owned by
[`docs/stair-feasibility/README.md`](../../docs/stair-feasibility/README.md).

To challenge the scripted-controller result directly, launch the interactive
`180 mm` scene:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\experiments\stair_feasibility\manual_180mm_stair.py
```

Click the viewport, select a leg with `1-4`, and hold `W/S` for
forward/backward, `E/D` for up/down, or `Q/A` for hip abduction. `R` resets the
floating robot, `Space` pauses or resumes physics, `C` prints the measured
state, and `X` or `Esc` saves and exits. Other legs stay at their last commanded
targets but remain fully dynamic; the script does not teleport links or fix
the base. Gravity, collisions, friction, hard joint limits, speed limits, and
the `0.980665 N m` rated effort cap remain active.

The optional `--torque-profile stall` comparison applies `2.941995 N m`.
Use it only as a short simulator diagnostic: stall torque is not a sustainable
hardware operating point. The session report is written to
`simulation/isaac/output/stair-feasibility-manual-180mm/session.json`.
Its `PASS` status means the interactive runner executed; tread-contact and
stability fields record what the human-controlled attempt actually achieved.

For direct motor targets instead of foot-space IK:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\experiments\stair_feasibility\manual_180mm_motor_angles.py
```

Type a motor number `1-12` and press `Enter`, then hold `Up` or `Down` to
change that motor's target angle. `Backspace` edits the entered number and `Z`
sets the selected motor target to zero. The panel continuously displays the
selected motor number and full joint name, target angle, measured angle, and
the live gravity setting. This distinguishes joint tracking from movement of
the floating body when gravity is manually disabled.

For a contact-free check of the same drive, run:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\experiments\stair_feasibility\probe_motor_physics.py `
  --base-mode floating `
  --joint front_left_hip_abduction `
  --target-deg 20 `
  --duration-s 1.5
```

The probe explicitly uses zero gravity and omits the floor and stair while
retaining self-collision, joint and speed limits, drive gains, and the selected
effort cap. Its default JSON output is under
`simulation/isaac/output/motor-physics-audit/`.

## Manual articulation

The portable handoff files are:

- `exports/isaac/quadruped_robot_fixed.usdc`
- `exports/isaac/quadruped_robot_floating.usdc`
- `exports/isaac/quadruped_robot_manual_world.usda`

The world references the floating USDC beside it, so keep those two files in
the same directory. It already contains Earth gravity, a high-friction floor,
the conservative standing targets, and the sustainable `0.980665 N·m`
ST3215 cap.

```powershell
& C:\isaacsim\python.bat simulation\isaac\open_articulation.py `
  --world exports\isaac\quadruped_robot_manual_world.usda `
  --onboard-camera
```

When Isaac opens, press **Play**, open **Physics > Articulation Inspector**,
select `/World/Robot`, and move the named joints. The launcher sets the initial
stand once and then leaves control to Isaac; it does not continuously overwrite
Inspector commands. Omit `--onboard-camera` to start with the external orbit
view; the onboard camera remains selectable from the viewport camera menu.
