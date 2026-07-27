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
