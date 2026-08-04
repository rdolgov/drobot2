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
| `simulation/isaac/rl/_quadruped_rl_env.py` | Implements Gymnasium `reset()`/`step()` against the real Isaac articulation and IMU, with a pre-physics-play hook for task-specific contact tensor registration | manual world and task config | actions, observations, rewards, episode state |
| `simulation/isaac/rl/train_ppo.py` | Creates `SimulationApp`, wraps the environment, trains/resumes PPO, checkpoints, and records provenance | YAML, manual world, optional checkpoint | model ZIP, TensorBoard, monitor CSV, training JSON |
| `simulation/isaac/rl/play_ppo.py` | Reloads a saved policy and runs deterministic evaluation episodes | YAML, manual world, model ZIP | evaluation JSON and optional PNG |
| `simulation/isaac/rl/record_ppo.py` | Records one deterministic flat-policy episode through an external or onboard RTX camera | YAML, manual world, model ZIP | H.264 MP4, thumbnail PNG, recording JSON |
| `simulation/isaac/rl/foot_lift/__init__.py` | Marks the supported single-foot-lift task as a separate Python source package | no runtime input | no generated data |
| `simulation/isaac/rl/foot_lift/quadruped_foot_lift_v1.yaml` | Versions the supported 190 mm lift gate, 200 mm IK reference, measured hardware cap, residual action, reward, termination, and PPO settings | edited by foot-lift runners | no generated data |
| `simulation/isaac/rl/foot_lift/quadruped_foot_lift_v2_balance.yaml` | Versions the unsupported 190 mm gate, five-stage clearance curriculum, 300 mm stance, 45 mm diagonal weight transfer, support-triangle reward, and measured hardware cap | edited by foot-lift runners | no generated data |
| `simulation/isaac/rl/foot_lift/quadruped_foot_lift_v3_rear_right_balance.yaml` | Mirrors the unsupported balance task onto the rear-right leg needed next in the stair sequence, retaining the strict 190 mm hold gate and measured hardware cap | edited by foot-lift runners | no generated data |
| `simulation/isaac/rl/foot_lift/_foot_lift_contract.py` | Defines the 56-value observation, smooth lift reference, reward, explicit failure reasons, and strict 190 mm hold gate | walking observation, measured skill state, and YAML values | observations and scalar contract results |
| `simulation/isaac/rl/foot_lift/_quadruped_foot_lift_env.py` | Applies the raise-forward IK reference, optional torso pose hold, lift curriculum, bounded PPO residuals, support-triangle measurement, foot-tip measurements, and episode metrics | manual world and foot-lift config | actions, observations, rewards, and supported/unsupported lift metrics |
| `simulation/isaac/rl/foot_lift/train_foot_lift_ppo.py` | Runs smoke/full/final-stage PPO training, optional flat or same-shape skill transfer, lift curriculum, checkpointing, and schema-2 model packaging | foot-lift YAML/world and optional initializer | model ZIP, manifest, monitor/TensorBoard data, and training report |
| `simulation/isaac/rl/foot_lift/train_foot_lift_v3_rear_right_190mm_small.ps1` | Runs the bounded fresh 512-step rear-right 190 mm final-stage training job | V3 foot-lift YAML and manual world | V3 model, manifest, checkpoints, and training report |
| `simulation/isaac/rl/foot_lift/evaluate_foot_lift_ppo.py` | Verifies the model manifest and runs deterministic supported-lift episodes | foot-lift YAML/world, model, and manifest | evaluation JSON and optional PNG |
| `simulation/isaac/rl/foot_lift/record_foot_lift_ppo.py` | Records one deterministic supported lift with a best-lift thumbnail | foot-lift YAML/world, model, and manifest | H.264 MP4, thumbnail PNG, and recording JSON |
| `simulation/isaac/rl/stairs/__init__.py` | Marks the stair experiment as a separate Python source package | no runtime input | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v1.yaml` | Versions the stair geometry, terrain inputs, curriculum, reward, reset, termination, action, and PPO settings | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v2.yaml` | Corrects the stair task with a close reset, navigation observations, physical-height reward, mastery curriculum, bounded exploration, and progress watchdog | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v3.yaml` | Applies the exercised one-leg joint ranges and nominal 30%-of-stall torque cap while widening the stair action box | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v4.yaml` | Corrects flat-policy action scaling and matches its effective 120 Hz action cadence | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v5.yaml` | Defines the frozen-flat residual controller, staged 10-40 mm worlds, physical fork-tip shaping, strict elevation/hold success, and measured hardware limits | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v6_180mm.yaml` | Defines the fixed-250 mm-depth height curriculum through the exact 180 mm stage, 68-value foot-sequence input, strict four-foot landing gate, and measured hardware limits | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v7_vl53l5cx.yaml` | Reuses the exact 180 x 250 mm world and measured hardware limits while replacing analytic terrain input with a 15 Hz noisy/latent 8 x 8 VL53L5CX raycast contract | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v8_single_tread_placement.yaml` | Defines the blind five-phase front-left placement curriculum, exact 180 x 250 mm geometry, force-backed tread/support gate, slip threshold, and measured hardware cap | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v9_front_pair_placement.yaml` | Composes front-left/front-right skills around an incenter-targeted, force/margin/velocity-gated mixed-height body transfer; current conservative run passes transfer and reaches 163 mm right lift | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v10_front_right_single_tread_placement.yaml` | Mirrors the blind force-backed 180 x 250 mm single-tread curriculum for the independently verified front-right leg | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v13_front_right_stabilized_lift.yaml` | Isolates a strict supported front-right 190 mm lift beside the exact 180 x 250 mm tread, with no custom friction | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v14_front_pair_right_then_left.yaml` | Composes verified right-foot placement with a snapshot-stabilized left-foot lift/placement mastery curriculum; current bounded run reaches the 100 mm gate and isolates post-transfer body-clearance loss | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v15_front_left_stabilized_lift.yaml` | Mirrors the strict supported-lift curriculum for the front-left leg and gates 190 mm clearance, three-foot support, slip, and upright hold beside the exact 250 mm tread | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v16_front_pair_proprioceptive_support.yaml` | Replays the verified right-foot placement, performs a force/margin-gated body transfer, then trains a support-only PPO residual through progressive left-foot lift and contact stages; RGB remains disabled | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v22_front_pair_torque_aware.yaml` | Slows the transferred left-foot lift to 3 s and extends the placed front-right support target by 50 mm under the real-test torque cap; exact tread depth remains 250 mm | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v23_front_pair_support_regulation.yaml` | Adds stance load, composite-COM error, and requested-effort saturation observations for live front-pair support training | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v24_front_pair_conservative_support.yaml` | Pins a strict randomized 190 mm front-left transfer drill with reward ordering that makes late drift/falls worse than early physical success | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v25_front_pair_load_sharing.yaml` | Tests direct stance-height load redistribution during the transferred 190 mm lift; retained as a rejected ablation | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v26_front_pair_smoothed_load_sharing.yaml` | Low-pass filters and bounds the rejected load-sharing ablation for a same-seed causal comparison | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v36_transfer_support_residual.yaml` | Isolates a two-second post-transfer support catch before the rear-right swing on the exact 180 x 250 mm stair, with a compact nine-action residual and measured hardware effort cap | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v37_joint_clearance_support.yaml` | Stages the post-transfer rear-pitch correction and joins the frozen V17/V35 rear-right swing with V36 support control on the exact 180 x 250 mm stair | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v42_com_margin_rear_right_landing.yaml` | Clips the rear-right landing balance target into the live support polygon at a positive margin while preserving the measured effort cap and exact 180 x 250 mm stair | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v43_compliant_rear_right_touchdown.yaml` | Adds bounded rear-right tread-load feedback for compliant touchdown diagnosis | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v44_early_contact_rear_right_landing.yaml` | Accepts force-backed post-clearance tread contact, latches the landing reference during the strict hold, and applies searched asymmetric support reach plus pitch regulation | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v45_rear_left_transfer.yaml` | Extends the accepted three-foot prefix only far enough to expose the rear-right-to-rear-left transfer for the next COM search | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/quadruped_stairs_v46_rear_right_sidestep.yaml` | Repositions the landed rear-right foot outward/deeper on the same 250 mm tread before exposing the rear-left COM transfer | edited by stair runners | no generated data |
| `simulation/isaac/rl/stairs/_stair_geometry.py` | Defines the crack-free stacked collision layers for the four-step staircase | stair YAML values | pure Python geometry facts |
| `simulation/isaac/rl/stairs/_vl53l5cx_contract.py` | Defines pure 8 x 8 ray geometry, 15 Hz mode validation, bounded noise/dropout, three-lane compression, and 24 depth observation fields | sensor YAML values and NumPy grids | rays and normalized depth observations |
| `simulation/isaac/rl/stairs/_vl53l5cx_sensor.py` | Samples 64 PhysX closest-hit rays, rotates the body sensor into world space, and applies update cadence, hold, and latency | live base pose and sensor config | raw/noisy grids, hit paths, and 24-value depth observation |
| `simulation/isaac/rl/stairs/_stair_rl_contract.py` | Defines selectable analytic or hardware-shaped terrain input, observation contracts through the 95-value support-regulation form, placement phases/contact gates, measured-state transfer re-anchoring, three- and four-foot load correction math, progress gates, strict stair/foot goals, physical-height reward terms, and failure reasons | walking/terrain/placement observation and stair YAML values | observations and scalar contract results |
| `simulation/isaac/rl/stairs/_quadruped_stairs_env.py` | Extends walking with exact DOF order, dynamic swing/support residual scales, pre-play force tracking, five-phase placement, mass-weighted whole-robot COM regulation, staged pitch correction, opt-in four-foot transfer preload, instrumented stance load-sharing ablations, per-leg slip and reachable-reference metrics, optional foot-friction sensitivity, selectable analytic/VL53L5CX perception, curriculum, and episode gates | stair world and task config | actions, observations, rewards, contact/sensor and episode state |
| `simulation/isaac/rl/stairs/_placement_phase_training.py` | Replays verified precursor phases and composes a frozen full-action base plus a second compact masked residual before exposing a bounded trainable target residual | stair environment and verified PPO policies | phase-local observations, actions, resets, and training metrics |
| `simulation/isaac/rl/stairs/_policy_transfer.py` | Strictly expands the verified 48/12 ELU flat policy and can rescale output means to preserve physical actions across action boxes | source and target policy tensors | transferred policy state and report |
| `simulation/isaac/rl/stairs/_run_support.py` | Creates/verifies schema-2 model, config, world/dependency, environment, PPO-mode, transfer, and resume contracts | model, YAML, composed world files, runtime/PPO contracts | adjacent `.contract.json` manifests |
| `simulation/isaac/rl/stairs/create_stairs_world.py` | Sublayers the validated manual world and authors static stair collision layers | base world and stair YAML | stair-world USDA and static validation JSON |
| `simulation/isaac/rl/stairs/train_stairs_ppo.py` | Trains, transfers, or resumes the separate stairs PPO policy, including verified nested frozen-base residual composition and placement-aware observation sizing, with curriculum, checkpoint manifests, live metrics, and no-progress abort reports | stair YAML/world and optional models | model ZIPs, manifests, TensorBoard, monitor CSV, progress watchdog, training JSON |
| `simulation/isaac/rl/stairs/train_stairs_v3_ppo.py` | Runs the generic stair trainer with hardware-informed v3 config/output defaults while preserving all trainer overrides | v3 stair YAML/world and optional model | v3 model ZIPs, manifests, TensorBoard, monitor CSV, progress watchdog, training JSON |
| `simulation/isaac/rl/stairs/train_stairs_v4_ppo.py` | Selects the transfer-safe v4 stair defaults | v4 stair YAML/world and optional model | v4 training outputs |
| `simulation/isaac/rl/stairs/train_stairs_v5_ppo.py` | Selects the residual-policy v5 defaults and staged-height configuration | v5 stair YAML/world and optional model | v5 training outputs |
| `simulation/isaac/rl/stairs/train_stairs_v6_180mm_ppo.py` | Selects the v6 exact-180 mm task and release output defaults | v6 stair YAML/world and optional model | v6 training outputs |
| `simulation/isaac/rl/stairs/train_stairs_v7_vl53l5cx_ppo.py` | Selects the v7 exact-180 mm task with cheap multi-zone ToF input | v7 stair YAML/world and optional model | v7 sensor-policy training outputs |
| `simulation/isaac/rl/stairs/train_stairs_v16_front_pair_ppo.py` | Selects the v16 blind front-pair task, verified v10 right-foot precursor, 140 mm left-leg entry stage, and support-only PPO action mask | v16 config, stair world, and v10 model | v16 model ZIPs, manifests, checkpoints, and training report |
| `simulation/isaac/rl/stairs/train_stairs_v17_single_foot_190mm_ppo.py` | Fine-tunes the verified isolated front-left policy directly at the strict 190 mm lift-and-hold gate using the exact 250 mm tread scene | v15 config and tracked v15 model | v17 model ZIPs, manifests, checkpoints, and training report |
| `simulation/isaac/rl/stairs/search_stairs_transfer_pose.py` | Searches timing, support extension, COM feedback, friction, and effort-cap variants from either a cached target phase or the full continuous front-pair sequence | v16/V22 config, stair world, and verified per-leg policies | ranked JSON search report |
| `simulation/isaac/rl/stairs/search_post_transfer_support.py` | Searches bounded constant support actions from the exact dynamic post-transfer snapshot before PPO training | V36 config, stair world, and verified precursor policies | ranked JSON action-search report |
| `simulation/isaac/rl/stairs/train_stairs_v36_transfer_support_small.ps1` | Runs the bounded 4,096-step post-transfer-only V36 PPO experiment with the successful search action as actor-mean initialization | V36 config, stair world, and verified V10/V17 policies | V36 model, contract, checkpoints, and training report |
| `simulation/isaac/rl/stairs/train_stairs_v37_joint_clearance_support_small.ps1` | Defines the corrected bounded V37 joint swing/support training job with full-strength V36 support authority; this corrected wrapper has not yet been run | V37 config and verified V10/V17/V35/V36 policies | prospective V37 model and training reports |
| `simulation/isaac/rl/stairs/train_stairs_v39_rear_right_landing_small.ps1` | Runs the rejected bounded 512-step rear-right landing support-residual pilot | V39 config and verified V10/V17/V35/V38 policies | local model, contract, and training report |
| `simulation/isaac/rl/stairs/train_stairs_v40_rear_right_swing_landing_small.ps1` | Runs the rejected bounded 1,024-step compact rear-right swing-residual landing pilot | V40 config and verified V10/V17/V35/V38 policies | local model, contract, and training report |
| `simulation/isaac/rl/stairs/train_stairs_v42_com_margin_rear_right_landing_small.ps1` | Runs the bounded V42 support-margin residual ablation | V42 config and verified V10/V17/V35/V38 policies | local model, contract, and training report |
| `simulation/isaac/rl/stairs/train_stairs_v44_early_contact_rear_right_landing_small.ps1` | Runs the accepted bounded 512-step V44 compact support-residual pilot | V44 config and verified V10/V17/V35/V38 policies | model, contract, checkpoints, and training report |
| `simulation/isaac/rl/stairs/search_rear_right_landing.py` | Restores one verified rear-right handoff snapshot and searches geometry, per-front-foot support reach, swing actions, pitch feedback, touchdown, and staged body-shift variants without relaxing the 190 mm/margin/effort gates | V39-V44 configs and verified composed policies | ranked local JSON reports and optional success-gated phase video/thumbnail |
| `simulation/isaac/rl/stairs/search_rear_left_transfer_support_actions.py` | Restores the exact V45 rear-left transfer boundary, audits constant support-hip authority or deterministically evaluates a 12-action PPO, and optionally records an external-camera failure clip | V45 config, phase snapshot, and optional PPO model | ranked JSON authority/evaluation report and optional MP4/PNG |
| `simulation/isaac/rl/stairs/search_rear_left_transfer_com.py` | Replays the accepted V44 three-foot prefix and searches the next rear-left transfer COM deltas from one cached boundary state | V45 config and verified V10/V17/V35/V44 policies | ranked local JSON report |
| `simulation/isaac/rl/stairs/search_rear_right_post_landing_sidestep.py` | Restores the exact accepted landing, re-lifts rear-right from the tread, searches bounded outward/forward replacements, and can require a separate final force-backed load before saving the next-phase snapshot | V46 config and exact V45 landing snapshot | ranked sidestep or force-backed-foothold report and rear-left transfer snapshot |
| `simulation/isaac/rl/stairs/search_rear_left_progressive_preload.py` | Restores an exact rear-left-transfer boundary, searches small four-foot COM/load/attitude increments, measures actual composite-COM progress so slip cannot masquerade as transfer, and supports bounded traction sensitivity | V46/V48 config and exact transfer snapshot | ranked V47-V49 preload reports and progress snapshots |
| `simulation/isaac/rl/stairs/PROGRESSIVE_PRELOAD_SEARCH.md` | Records the V47-V49 controller contract, exact commands, force-backed foothold, attitude and traction sensitivities, slip-proof progress gate, and the reason another PPO run is gated | exact V46/V48 snapshots and search reports | reproducible preload diagnosis |
| `simulation/isaac/rl/stairs/REAR_RIGHT_LANDING_SEARCH.md` | Records V39-V48 inputs, exact small-run commands, measured rejects, accepted V44 landing/V46 sidestep/V48 force-backed settle evidence, transfer failures, assumptions, and next gate | training/evaluation/search reports | reproducible landing and transfer diagnosis |
| `simulation/isaac/rl/stairs/validate_vl53l5cx_stairs.py` | Verifies live 64-ray PhysX hits, 15 Hz cadence, latency, held observations, and writes an 8 x 8 review heatmap | v7 config and stair world | sensor validation JSON and PNG |
| `simulation/isaac/rl/stairs/distill_successful_stairs.py` | Collects physically successful stochastic rollouts and behavior-clones their residual actions into the actor mean | v5 world, model, manifest | distilled model, rebound manifest, and collection report |
| `simulation/isaac/rl/stairs/evaluate_stairs_ppo.py` | Runs deterministic stair episodes at a pinned curriculum level, verifies model hashes, and can compose a full base, compact frozen swing residual, and mapped support residual per leg | stair YAML/world, primary/per-leg models, manifests | evaluation JSON and optional PNG |
| `simulation/isaac/rl/stairs/record_stairs_ppo.py` | Records deterministic or stochastic stair episodes with the same nested per-leg policy composition, strict whole-stair or intermediate placement-success search, and a close external placement view | stair YAML/world, models, manifests | H.264 MP4, thumbnail PNG, optional trajectory, recording JSON |
| `simulation/isaac/rl/stairs/search_stairs_v98_front_left_retained_tread_actions.py` | Searches bounded front-hip/knee action pairs from the retained first-tread snapshot to confirm a low-slip force-backed authority corridor | V97 retained-tread snapshot and measured effort cap | ranked local JSON report and best-action snapshot |
| `simulation/isaac/rl/parallel_stairs/__init__.py` | Registers the external pure-parallel Isaac Lab stair task and its RSL-RL agent configuration | no runtime input | no generated data |
| `simulation/isaac/rl/parallel_stairs/agents/__init__.py` | Marks the pure-parallel RSL-RL agent configuration package | no runtime input | no generated data |
| `simulation/isaac/rl/parallel_stairs/agents/rsl_rl_ppo_cfg.py` | Defines the 70-input, 256-by-256 PPO actor/critic and bounded training schedule | task registration | checkpoints and TensorBoard events |
| `simulation/isaac/rl/parallel_stairs/exact_stairs_terrain.py` | Generates the exact one-direction four-step 180 mm rise by 250 mm tread mesh and approach origin | terrain generator config | runtime triangle mesh |
| `simulation/isaac/rl/parallel_stairs/pure_stairs_env_cfg.py` | Pins the 128-environment robot, actuator cap, friction, 15 Hz VL53L5CX-style rays, and fixed stair geometry | floating robot USDC | runtime scene configuration |
| `simulation/isaac/rl/parallel_stairs/pure_stairs_env.py` | Implements all-joint pure actions, deployable IMU/joint/load/depth observations, physical fork-tip lift/tread reward, and non-scripted success/failure gates | pure task config and simulator sensors | observations, rewards, metrics, and resets |
| `simulation/isaac/rl/parallel_stairs/train_pure_parallel_stairs.py` | Registers and launches 128-way RSL-RL PPO training without gait phases, IK, or scripted leg order | task and agent config | checkpoints, parameters, and TensorBoard events |
| `simulation/isaac/rl/parallel_stairs/evaluate_pure_parallel_stairs.py` | Runs a bounded many-environment deterministic comparison without RGB frame capture | task, agent config, checkpoint, and step count | exact episode totals |
| `simulation/isaac/rl/parallel_stairs/widen_rsl_rl_checkpoint.py` | Duplicates hidden units and divides downstream weights to widen both PPO MLPs without changing their outputs | 256-by-256 RSL-RL checkpoint | verified 512-by-512 checkpoint with reset optimizer moments |
| `simulation/isaac/rl/parallel_stairs/two_mode_gaussian.py` | Implements an exact two-component whole-action Gaussian PPO distribution whose deterministic action selects a learned component rather than averaging modes | unchanged 70-value actor observation | sampled actions, exact mixture log probability, and committed deterministic action |
| `simulation/isaac/rl/parallel_stairs/transplant_two_mode_checkpoint.py` | Converts a widened single-Gaussian checkpoint into two antisymmetrically separated modes while preserving its average action | 512-by-512 RSL-RL checkpoint | verified two-mode training initializer |
| `simulation/isaac/rl/parallel_stairs/force_two_mode_checkpoint.py` | Forces either learned mixture component for diagnostic deterministic evaluation | trained two-mode checkpoint | evaluation-only component checkpoint |
| `simulation/isaac/rl/parallel_stairs/persistent_mode_policy.py` | Implements reward-trained discrete and continuous episode commitments, exact PPO replay likelihoods, and deterministic JIT deployment with commitment reset | unchanged 70-value actor observation | coherent whole-episode exploration and deployable actions |
| `simulation/isaac/rl/parallel_stairs/transplant_persistent_bias_checkpoint.py` | Expands a two-mode checkpoint with zero-centered learned 12-joint episode biases while preserving its deterministic control outputs | trained two-mode checkpoint | persistent-bias PPO initializer |
| `simulation/isaac/rl/parallel_stairs/optimize_episode_bias_cem.py` | Runs a reward-ranked multi-population diagonal CEM over held 12-joint episode biases, with antithetic samples, winner-centered refinement, randomized resets, and optional repeated randomized evaluations of every candidate | unchanged deterministic 70-input sensor policy and strict environment reward | baked deterministic checkpoints and a complete JSON search report |
| `Drobot-Pure-Stairs-Low25-To37-HardBias-Stand-Rise10-Hip-Direct` | Tests a symmetric four-support, 10 mm body-rise precursor from lower hardware-representable resets | IMU, depth, joints, previous action, and foot loads | strict body-rise checkpoint telemetry |
| `Drobot-Pure-Stairs-Low25-To37-HardBias-ThreeSupport-Rise10-Hip-Direct` | Relaxes the same pure-PPO body-rise precursor to any three verified supports | same 70 deployable actor inputs | three-support rise telemetry |
| `Drobot-Pure-Stairs-Low25-To37-HardBias-Upright-Rise10-Hip-Direct` | Uses the literal non-failing upright + held 10 mm body-rise outcome without a contact-pattern gate | same 70 deployable actor inputs | ungated extension checkpoint telemetry |
| `Drobot-Pure-Stairs-FullFold-TwoSupport-Rise5-Hip-Direct` | Starts fully folded at 0.30 m and scores a settled, held 5 mm body rise with at least two supports | same 70 deployable actor inputs | corrected full-fold rise telemetry and PPO checkpoints |
| `Drobot-Pure-Stairs-Yaw45-FullFold-TwoSupport-Rise5-Hip-Direct` | Bridges the fully folded policy to a 45-degree approach while keeping the existing ToF aimed at the stair and increasing bounded hip-abduction authority | same 70 deployable actor inputs | held-out 45-degree rise telemetry and PPO checkpoints |
| `Drobot-Pure-Stairs-Yaw67p5-FullFold-TwoSupport-Rise5-Hip-Direct` | Tests the retained yaw-45 checkpoint at a 67.5-degree full-fold approach with the same corrected rise gate | same 70 deployable actor inputs | held-out transfer telemetry and rejected continuation evidence |
| `Drobot-Pure-Stairs-Yaw90-FullFold-TwoSupport-Rise5-Hip-Direct` | Tests the gradual-yaw checkpoint at a fully sideways full-fold reset with the ToF still aimed at the stair | same 70 deployable actor inputs | lateral held-out telemetry and short PPO continuation evidence |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-Hip-Direct` | Requires any foot to clear 50 mm for four post-settle control steps while three other feet remain force-loaded from the fully folded sideways reset | same 70 deployable actor inputs | strict symmetric unload telemetry |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift10-Hip-Direct` | Raises the same force-backed lateral unload gate to 100 mm for six control steps | same 70 deployable actor inputs | staged lift telemetry and held-out comparisons |
| `Drobot-Pure-Stairs-Yaw90-FoldBridge-Foot-Lift5-Hip-Direct` | Samples the full neutral-to-folded 0.46-to-0.30 m reset range at yaw 90 and adds a symmetric relative-force unload signal before the 50 mm gate | same 70 deployable actor inputs; no selected leg or reference motion | fold-transfer PPO checkpoints |
| `Drobot-Pure-Stairs-Yaw90-FoldBridge-Foot-Lift5-Consolidate-Hip-Direct` | Uses long low-entropy PPO batches on the same mixed-fold 50 mm task to move sampled successes into the policy mean | same 70 deployable actor inputs and 256-by-256 actor | deterministic full-fold promotion candidates |
| `Drobot-Pure-Stairs-Yaw90-FoldBridge-Foot-Lift5-Wide512-Hip-Direct` | Continues the same pure-PPO bridge after a function-preserving 256-to-512 hidden-width transplant | same 70 inputs, symmetric reward, and no reference motion | controlled capacity A/B checkpoints |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-Wide512-Hip-Direct` | Evaluates the widened actor from the exact fully folded sideways reset | same 512-by-512 actor and strict force-backed 50 mm gate | deterministic capacity promotion evidence |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-Coupled-Wide512-Hip-Direct` | Couples lift and relative-force unload reward on the same unnamed foot from the exact folded reset | symmetric physical outcomes; no selected leg, phase, or reference action | mode-consolidation PPO checkpoints |
| `Drobot-Pure-Stairs-Yaw90-FoldTail75-Foot-Lift5-Wide512-Hip-Direct` | Restricts resets to 75%-to-fully-folded and biases samples toward the endpoint | same symmetric pure-RL reward and 512-by-512 actor | endpoint-focused curriculum checkpoints |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-TwoMode-Hip-Direct` | Learns two whole-action Gaussian modes so valid unnamed-foot strategies need not be averaged into one action | same 70 deployable inputs; no selected leg, phase, or reference motion | multimodal PPO checkpoints and forced-component diagnostics |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-GRU-Hip-Direct` | Adds a 128-state GRU so the sensor-only actor can coordinate a persistent multi-step unload | same 70 deployable inputs and symmetric physical reward | recurrent PPO checkpoints |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-SuccessDominant-GRU-Hip-Direct` | Removes the per-step four-foot support income that made waiting competitive with terminal success and doubles the held-lift payoff | same recurrent sensor-only actor | controlled reward-economics evidence |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-SensorAsym-GRU-Hip-Direct` | Adds bounded symmetric joint and lateral reset differences visible through existing joint/load/IMU sensors | no leg identifier or non-deployable observation | sensor-conditioned mode-selection evidence |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-PersistentMode-Hip-Direct` | Samples one learned discrete whole-action mode per episode and retains it until reset | same 70 deployable inputs and symmetric physical reward | episode-consistent mode PPO checkpoints |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-PersistentBias-Hip-Direct` | Samples one learned 12-joint bias per episode plus small per-step action noise | same 70 deployable inputs; no leg identifier, phase, or reference motion | coherent episode-exploration PPO checkpoints |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-PersistentBias-Consolidate-Hip-Direct` | Gives the one-time latent choice 32-times policy-gradient/KL credit while removing entropy and reducing learning rate | same sensor-only actor and exact full-fold sideways reset | controlled latent-credit comparison |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-PersistentBias-CEM-Robust-Hip-Direct` | Retests the accepted 50 mm CEM center with 0.02 rad joint and 0.015 m lateral reset variability | same 70 deployable actor inputs and 512-by-512 persistent-bias actor | stronger-reset validation and conservative PPO continuation |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift7p5-PersistentBias-CEM-Robust-Hip-Direct` | Adds a five-step 75 mm bridge gate under the stronger reset distribution | same pure sensor policy; no selected leg, gait, IK, or reference motion | rare out-of-search 75 mm authority evidence |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift10-PersistentBias-CEM-Robust-Hip-Direct` | Tests the same reward-ranked episode-bias search at a six-step 100 mm gate | same pure sensor policy and real-test effort cap | rare search-only 100 mm authority evidence |
| `Drobot-Pure-Stairs-Yaw90-FoldBridge-Foot-Lift10-Hip-Direct` | Applies the same fold-distribution bridge and symmetric unload signal at the 100 mm gate | same 70 deployable actor inputs; no selected leg or reference motion | 10 cm transfer candidates for exact full-fold evaluation |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift14-Hip-Direct` | Defines the 140 mm intermediate force-backed lift stage without prescribing a swing leg | same 70 deployable actor inputs | next-stage PPO checkpoints |
| `Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift19-Hip-Direct` | Defines the required 190 mm force-backed lift from the fully folded sideways stance | same 70 deployable actor inputs | final lift-stage PPO checkpoints |
| `Drobot-Pure-Stairs-Sideways-TwoSupport-Rise5-Hip-Direct` | Rotates the robot 90 degrees, aims the existing ToF sensor toward the stair, and expands bounded hip-abduction authority | same 70 deployable actor inputs | lateral hip-leverage telemetry and PPO checkpoints |
| `simulation/isaac/rl/parallel_stairs/zero_agent_pure_parallel_stairs.py` | Runs a vectorized zero-action task smoke for scene and throughput validation | pure task config | console metrics |
| `simulation/isaac/rl/parallel_stairs/play_pure_parallel_stairs.py` | Loads a pure-parallel checkpoint for deterministic evaluation and optional short RGB recording | task, agent config, and checkpoint | H.264 MP4 |
| `simulation/isaac/rl/parallel_stairs/run_stair_rl_workflow.ps1` | Runs the current folded-sideways 7.5 cm checkpoint as one visible robot, a bounded five-robot GUI fine-tune, or a long 128-robot headless continuation | packaged checkpoint or latest workflow checkpoint | visual inspection and resumable RSL-RL checkpoints |

### Hands-on current-model workflow

Run these commands from `robot-cad` in order. The first command opens one
isolated robot with the newest checkpoint produced by this workflow (or the
current packaged 7.5 cm model before the first training run) and keeps running
until the Isaac Sim window is closed:

```powershell
& .\simulation\isaac\rl\parallel_stairs\run_stair_rl_workflow.ps1 `
  -Mode test
```

The second command opens the Kit GUI, shows five physics replicas, performs
five PPO iterations, saves the resulting checkpoint, and exits. Five robots
use one trajectory minibatch because RSL-RL's persistent-state generator
requires the environment count to be divisible by the minibatch count.

```powershell
& .\simulation\isaac\rl\parallel_stairs\run_stair_rl_workflow.ps1 `
  -Mode train-visible `
  -Iterations 5 `
  -NumEnvs 5
```

The third command automatically resumes the newest checkpoint produced by a
prior `train-visible` or `train-headless` workflow run, disables visualizers,
and continues with 128 parallel replicas. Change `-Iterations` for the desired
training duration.

```powershell
& .\simulation\isaac\rl\parallel_stairs\run_stair_rl_workflow.ps1 `
  -Mode train-headless `
  -Iterations 500 `
  -NumEnvs 128
```

All three modes retain the exact 180 mm rise by 250 mm tread, literal 0.30 m
fully folded yaw-90 reset, 0.8825985 N m effort cap, and 70-value deployable
IMU/joint/load/depth actor input. `-Checkpoint <path>` explicitly selects a
different initializer. Training outputs remain under
`logs/rsl_rl/drobot_pure_stairs_yaw90_fullfold_foot_lift7p5_persistent_bias_cem_robust_hip_180x250_direct/`.
Parallel replicas may intentionally share display coordinates because
`replicate_physics=True` isolates their dynamics. The one-robot test hides all
other render prims, so shared-coordinate replicas cannot clutter that review.

The pure-stairs PPO remains intentionally compact. Its deployable actor is
`70 -> 256 -> 256 -> 12` with `87,064` trainable parameters including the
Gaussian standard deviation. The training-only critic has `84,225`, for
`171,289` trainable parameters in total. The roughly `2.1 MB` RSL-RL
checkpoint also contains both normalizers and optimizer state, so checkpoint
size is not the actor's deployed memory footprint.

The recurrent comparison uses `GRU(70, 128) -> 256 -> 256 -> 12`. Its saved
actor state has `178,923` values and the training-only critic has `176,084`,
for `355,007` model-state values total. This is still smaller than the
`512 x 512` feed-forward comparison actor, so the recurrent result is not a
large-model capacity test.

The persistent-bias comparison uses `70 -> 512 -> 512 -> 50`: two mode logits,
two 12-action control means, and two learned 12-joint bias centers. It has
`324,706` trainable actor parameters (`324,917` saved values including the
70-value normalizer buffers), about `1.30 MB` as FP32 deployment weights. Its
full RSL-RL training checkpoint is `7,511,983` bytes because it also contains
the critic, normalizers, and optimizer state. This is already a comfortable
deployment size; the current evidence points to episode-level credit and
symmetry consolidation rather than insufficient network capacity.

The 2026-08-03 folded-sideways bridge round added a symmetric relative-force
unload reward and sampled the full `0.46` to `0.30 m` reset range without a
selected swing leg. Three 128-way PPO pilots accumulated `1,536,000`
transitions. Their stochastic success totals were `443/2,859`, `625/3,831`,
and `1,228/7,391`; the last pilot restarted exploration standard deviation at
`0.20` and let zero-entropy PPO anneal it toward `0.19`. Exact deterministic
full-fold evaluation still returned `0/41` for model 566, `0/45` for the
annealed midpoint model 615, and `0/51` for final model 665. Therefore none is
promoted to the 100 mm task and no improvement video is claimed. The evidence
favors a controlled `512 x 512` actor comparison next while retaining the
current actor as the control; it does not yet show that capacity, rather than
policy-mean consolidation, is the root cause.

The controlled width comparison then widened both hidden layers from `256` to
`512` with a function-preserving checkpoint transplant. Maximum actor and
critic output errors were `5.96e-7` and `1.53e-5`, respectively. Over `819,200`
new transitions the wider bridge policy reached `1,312/7,455` stochastic
successes (`17.60%`), only modestly above the matched narrow control's
`1,228/7,391` (`16.61%`). Deterministic exact-full-fold evaluations remained
`0/45` at model 764 and `0/48` at model 714.

Three endpoint experiments added another `1,228,800` transitions. The biased
75%-to-full-fold tail produced `43/1,782` stochastic successes but `0/47`
deterministic successes. Direct exact-full-fold PPO produced `40/2,384`, and a
same-foot coupled lift-and-unload reward produced `40/1,961`; their early
deterministic checkpoints were still `0/50` and `0/48`. These literal endpoint
successes demonstrate that the 50 mm lift is physically learnable, while the
ordinary Gaussian policy mean continues to settle into a stable no-lift
stance. The width hypothesis is therefore rejected for this stage, no model is
promoted to 100 mm, and the next experiment should address the multimodal
policy distribution without adding a selected leg, gait phase, reference
action, or non-deployable observation. The third-person diagnostic is archived
as `reviews/ppo-stairs-fold-tail75-wide512-model813-seed1238.mp4`.

The next pure-PPO round tested that hypothesis with `2,048,000` additional
parallel transitions. A two-component whole-action Gaussian produced
`79/4,040` strict stochastic successes (`1.96%`) over `819,200` transitions,
but learned-selector checkpoints and both individually forced components were
all deterministic zero-success. A 128-state GRU then produced `31/1,936`
(`1.60%`) over `409,600` transitions; models 24, 44, and 49 were also
deterministic zero-success. This rejects per-control-step mixture selection and
sensor history alone as solutions to the policy-mean collapse.

Reward auditing found that the old exact-fold task paid `0.25` per supported
foot on every control step, making a full stationary episode competitive with
the delayed `+200` success under discounting. A controlled continuation removed
that stationary support income and raised terminal success to `+400`. It
produced `27/1,863` (`1.45%`) over `409,600` transitions, while four screened
checkpoints remained deterministic zero-success. A final continuation increased
independent symmetric reset joint offsets from `0.01` to `0.04 rad` so the same
joint/load/IMU sensors could expose real-hardware-like asymmetry without a leg
identifier. It produced `34/2,324` (`1.46%`) over `409,600` transitions; models
103, 122, 137, and 147 again scored zero deterministic successes. No checkpoint
is promoted to 100 mm. The honest post-training third-person attempts are
archived as
`reviews/ppo-stairs-success-dominant-gru-model98-seed1263.mp4` and
`reviews/ppo-stairs-sensor-asym-gru-model147-seed1270.mp4`.

The 2026-08-03 episode-commitment round then added `1,024,000` substantive
parallel transitions (`1,048,576` including three smoke iterations) from the
literal `0.30 m` fully folded, 90-degree sideways reset. The discrete
persistent-mode run produced `44/2,468` strict stochastic successes (`1.78%`)
and explored `112.6 mm` maximum foot clearance, but four deterministic screens
and forced checks of both components remained at zero. The continuous
persistent-bias run improved stochastic exploration to `52/2,003` (`2.60%`),
with an early cumulative peak of `17/231` (`7.36%`), but six deterministic
screens remained at zero. Scaling the one-time commitment credit by 32 produced
`27/1,048` (`2.58%`) and did not consolidate the sampled behavior into the
policy center. The latest model 961 completed `0/29` strict episodes in the
30-second deterministic playback. It is not promoted to 100 or 190 mm. The
ordinary third-person evidence is archived as
`reviews/ppo-stairs-persistent-bias-model961-seed1297-30s.mp4`; it shows a
stable folded stance with small motion but no meaningful unload-and-lift
sequence. The next controlled experiment should optimize the persistent
12-joint episode distribution directly from ranked whole-episode returns (for
example an elite/CEM-style reward-only update), then fine-tune the resulting
deterministic center with sensor feedback.

That reward-only search is now implemented and screened. Three successive
128-environment, two-population CEM runs added `2,457,600` whole-episode search
transitions: 40 generations from the early persistent-bias checkpoint, 20
winner-centered nominal refinements, and 20 winner-centered refinements with
the configured `0.01 rad` joint and `0.01 m` lateral reset variability. The
three reports contain `204`, `75`, and `47` first-episode strict successes; the
randomized run also produced the first successful population center. No
selected leg, gait phase, reference motion, IK, or simulator-only actor input
was added: every candidate remained the deterministic 70-input sensor policy
plus one reward-selected 12-joint bias held for its full episode.

Four fresh randomized screens selected robust population 0 at `19/583`
strict 50 mm held-foot successes (`3.26%`), narrowly ahead of population 1 at
`18/568` (`3.17%`). This is the first nonzero deterministic-policy result from
the literal fully folded sideways reset, but it remains fragile and is not a
stair-climbing result. The accepted 30.00-second third-person recording is
`reviews/ppo-stairs-robust-cem-pop0-env10-seed1351-30s.mp4`; filmed environment
10 itself crossed the strict gate, while all rendered environments totaled
`26/638`. Visual motion is still an unload/lift exploration with occasional
destabilization: there is no 100 or 190 mm promotion, tread placement, weight
transfer, step, or climb. The packaged checkpoint and reports are under
`simulation/isaac/models/ppo-stairs-pure-cem-fullfold-yaw90-seed1341/`.

The subsequent robustness and height-bridge round added `2,662,400`
substantive transitions. Conservative PPO at 50 mm produced `46/3,901`
training successes, but its screened model 940 scored `27/547` and model 1000
scored `23/545`, both below the unchanged source at `34/509`; it was rejected.
A local 50 mm CEM was also rejected after fresh screens of source `14/487`,
population 1 `10/492`, and best sample `13/515`. Direct 100 mm search found
only one source screen success (`1/155`) and one search success among `5,339`
completed episodes; all fresh candidates were zero, proving rare authority but
not a deployable behavior.

The 75 mm bridge search used 128 environments, two populations, 20 generations,
and `614,400` transitions. It recorded `15/5,603` strict successes and one
successful population center during search. Packaged population 1 then achieved
the first out-of-search bridge success: `0/165` followed by `1/143` across two
fresh seeds (`1/308` pooled). The 30-second ordinary batch recording at
`reviews/ppo-stairs-cem-7p5cm-pop1-env96-seed1452-30s.mp4` contains a strict
success in environment 72, but the third-person camera followed environment 96,
which did not pass; it is not claimed as visual proof. There is still no
repeatable 100 or 190 mm lift, tread placement, weight transfer, step, or climb.

A replicated-candidate consolidation then added `1,228,800` transitions. Each
of 32 candidate biases per generation was evaluated in four independently
reset physics replicas before ranking. Across 40 generations, eight candidate
groups produced a strict first-episode pass, but every passing group was only
`1/4`; neither population center passed. The continuation is rejected and the
packaged bridge model is unchanged. Isaac Lab intentionally permits isolated
`replicate_physics=True` environments to share renderer coordinates, which
made the old many-environment video look like overlapping robots even though
their dynamics did not interact. New interactive review uses one visible
robot, and the packaged report records this visual limitation explicitly.

| `simulation/isaac/models/ppo-walk-v1-2m/` | Tracks the frozen flat-walking dependency used by v5 residual control | validated flat PPO ZIP | release dependency |
| `simulation/isaac/models/ppo-stairs-pure-parallel-v1-lifthold-seed1059/agent.yaml` | Captures the exact RSL-RL PPO configuration for the packaged lift-hold checkpoint | saved training parameters | no generated data |
| `simulation/isaac/models/ppo-stairs-pure-parallel-v1-lifthold-seed1059/env.yaml` | Captures the exact Isaac Lab environment configuration for the packaged lift-hold checkpoint | saved training parameters | no generated data |
| `simulation/isaac/models/ppo-stairs-pure-parallel-v1-lifthold-seed1059/` | Packages iteration-400/600 pure PPO checkpoints, hashes, metrics, and the explicit partial-result report | physical-tip/lift-hold training chain | evaluation checkpoint package; rare tread contact, no complete climb |
| `simulation/isaac/models/ppo-stairs-pure-parallel-v2-hip-seed1066/` | Packages the iteration-260/280 hip-authority checkpoints, resolved configs, hashes, reset comparison, and explicit partial-result report | low/forward/sideways pure-PPO comparison and consolidation | evaluation checkpoint package; two-contact exploration, no complete climb |
| `simulation/isaac/models/ppo-stairs-pure-width105-low25-seed1129/` | Packages the strongest 0.42 m, 25%-fold reset checkpoint, exact configs, hash, and Low25/Low50 metrics | gradual lower-reset pure-PPO curriculum | stochastic four-support landing checkpoint; no deterministic landing or climb |
| `simulation/isaac/models/ppo-stairs-pure-low25-to37-seed1135/` | Packages the mixed 0.42-to-0.40 m reset winner, exact configs, hash, reset-bin audit, and fixed-Low37 rejection | correlated lower-reset pure-PPO bridge | harder-half deterministic tread success; no fixed-pose repeatability or climb |
| `simulation/isaac/models/ppo-stairs-pure-low25-to37-hard-bias-seed1145/` | Packages the 75%-hard-half continuation winner, exact configs, hash, paired two-seed comparison, and RGB review | harder-sample pure-PPO continuation | 5/983 pooled deterministic successes; still no repeatable step, transfer, or climb |
| `simulation/isaac/models/ppo-stairs-pure-cem-fullfold-yaw90-seed1341/` | Packages the robust reward-ranked episode-bias checkpoint, full CEM report, hashes, four-seed screen, and accepted 30-second selected-environment review | exact 180 mm x 250 mm folded-sideways 50 mm held-lift task | 19/583 strict deterministic-policy successes; no tread placement or climb |
| `simulation/isaac/models/ppo-stairs-pure-cem-7p5cm-seed1441/` | Packages the experimental 75 mm bridge center, full CEM report, hashes, fresh screens, and ordinary 30-second review | exact 180 mm x 250 mm folded-sideways task with stronger reset variability | 1/308 fresh strict successes; no visual pass, tread placement, or climb |
| `simulation/isaac/models/ppo-stairs-v5-10mm-four-step/` | Tracks the source-equivalent shallow-stair policy, schema-2 manifest, and deterministic evaluation | v5 config/world and source policy | evaluable release package |
| `simulation/isaac/models/ppo-stairs-v6-180mm-25cm-small/` | Tracks the bounded 180 mm x 250 mm evaluation policy, schema-2 manifest, training/evaluation/recording reports, and explicit failure result | v6 config/world and bounded policy | non-deployable evaluation package |
| `simulation/isaac/models/ppo-stairs-v6-180mm-25cm-balance-small/` | Tracks the 512-step 180 mm x 250 mm stair smoke model initialized from unsupported balance, exact shared-prefix transfer report, 0/5 evaluation, and recording | v6 config/world and v2 balance policy | non-deployable transfer evaluation package |
| `simulation/isaac/models/ppo-stairs-v7-vl53l5cx-180mm-small/` | Tracks the 512-step VL53L5CX sensor-policy smoke model, schema-2 manifest, training/evaluation/recording reports, and explicit failure result | v7 config/world and transferred v6 policy | non-deployable sensor pipeline package |
| `simulation/isaac/models/ppo-stairs-v8-180mm-25cm-single-foot-placement-small/` | Tracks the 2,048-step fixed-tread placement policy, schema-2 manifest, 5/5 force-backed evaluation, recording, and contact/slip evidence | v8 config/world and unsupported-balance initializer | single-foot placement evaluation package, not a complete climb |
| `simulation/isaac/models/ppo-stairs-v10-180mm-25cm-front-right-placement-small/` | Tracks the mirrored 2,048-step front-right policy, schema-2 manifest, 5/5 force-backed evaluation, recording, and contact/slip evidence | v10 config/world and unsupported-balance initializer | front-right placement evaluation package, not a complete climb |
| `simulation/isaac/models/ppo-stairs-v13-front-right-190mm-lift-small/` | Tracks the 8,192-step direct front-right lift policy, schema-2 manifest, 3/3 strict evaluation, recording, and 205 mm lift evidence | v13 config/world and direct PPO policy | supported front-right lift package, not a complete climb |
| `simulation/isaac/models/ppo-stairs-v15-front-left-190mm-lift-small/` | Tracks the mirrored 8,192-step front-left policy, schema-2 manifest, 5/5 strict evaluation, recording, and 205-207 mm lift evidence | v15 config/world and transferred/direct PPO policy | supported front-left lift package, not a complete climb |
| `simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/` | Tracks the 4,096-step strict-stage V15 fine-tune, schema-2 manifest, 5/5 deterministic evaluation, recording, trajectory, and 205-208 mm lift evidence | v15 config/world and tracked v15 initializer | isolated 190 mm balance gate, not a complete climb |
| `simulation/isaac/models/ppo-stairs-v27-support-abduction-190mm-small/` | Tracks the 1,024-step three-action support-abduction residual, 2/5 fresh evaluation, successful recording, and rejected load-sharing A/B | v24 config, verified front-right prefix, and frozen v17 swing policy | experimental transferred 190 mm lift; not robust or a complete climb |
| `simulation/isaac/models/ppo-stairs-v36-post-transfer-catch-small/` | Tracks the 4,096-step nine-action support catch, bounded initialization search, fresh 65/80-second evaluations, recording, and explicit precursor/lift limitations | V36 config, verified precursor/swing policies, and dynamic transfer snapshot | stable seed-832 post-transfer catch; not a 190 mm lift, tread landing, or complete climb |
| `simulation/isaac/models/ppo-stairs-v44-early-contact-rear-right-landing-small/` | Tracks the 512-step compact support policy, schema-2 manifest, deterministic search acceptance, fresh seed-870 force-backed landing evaluation, recording report, and exact limitations | V44 config and verified V10/V17/V35/V38 composition | accepted first-tread rear-right landing package; not rear-left transfer or a complete climb |
| `simulation/isaac/models/ppo-stairs-v45-rear-left-dynamic-transfer-4096/` | Tracks the rejected 4,096-step exact-snapshot 12-action transfer policy, deterministic failed replay, and controller-authority evidence | V45 config, accepted V44 boundary snapshot, and real-test effort cap | diagnostic policy package; not a successful transfer or climb |
| `simulation/isaac/models/ppo-foot-lift-v1-190mm-small/` | Tracks the 512-step supported foot-lift smoke model, schema-2 manifest, three-episode evaluation, recording report, and strict 190 mm success | foot-lift config/world and supported policy | supported-skill evaluation package, not unsupported balance proof |
| `simulation/isaac/models/ppo-foot-lift-v2-balance-190mm-small/` | Tracks the 512-step unsupported foot-lift smoke model, schema-2 manifest, 5/5 evaluation, screenshot report, recording report, and strict 190 mm success | v2 balance config/manual world | unsupported flat-ground clearance evaluation package |
| `simulation/isaac/models/ppo-foot-lift-v3-rear-right-190mm-small/` | Tracks the fresh 512-step rear-right balance policy, schema-2 manifest, 5/5 evaluation, and successful recording | V3 balance config/manual world | rear-right flat-ground 190 mm clearance package; not stair-transfer evidence |
| `simulation/isaac/models/ppo-foot-lift-v3-rear-right-190mm-seed941-small/` | Tracks an independent fresh 512-step rear-right policy, schema-2 manifest, 5/5 evaluation, and seed-943 recording | V3 balance config/manual world | reproduced rear-right flat-ground 190 mm clearance; not stair-transfer evidence |
| `simulation/isaac/models/ppo-stairs-v46-rear-right-sidestep-transfer-4096/` | Tracks the accepted seed-937 rear-right sidestep boundary and rejected 4,096-step rear-left transfer policy with deterministic failure video | V46 config, exact landing snapshot, and real-test effort cap | diagnostic package; not a successful transfer or climb |
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

V4/V5 correct the v3 transfer scale and cadence, then place the stair policy
as a learned residual over the frozen flat gait. V5 retains the real-test
`0.8825985 N m` effort cap and joint ranges, adds physical fork-tip lift/tread
metrics, and provides `10`, `20`, `30`, and `40 mm` height stages. A 20,480-step
four-step run followed by success-conditioned stochastic evaluation produced
one complete four-step `10 mm` climb in 81 attempts. Exact replay recorded all
four feet on step four, `43.5829 mm` maximum base rise, `1.39269 m` travel,
`12.852 deg` maximum tilt, and the required `0.5 s` hold. The packaged policy
mean still scored `0/10` deterministic successes, and no taller stage
completed. This is a verified shallow-stair simulator episode, not convergence
or hardware-transfer evidence.

V6 keeps every tread at the requested `250 mm` depth and adds height stages
through the exact four-step `180 mm` task. It expands the observation to 68
values with sequenced physical-foot progress, retains the measured
`0.8825985 N m` cap, and requires all four stairs plus all four feet on the
landing. A bounded `20,480`-step `10 mm` entry transfer followed by `10,240`
steps on `180 mm` produced no success. The final training peak was `172.196 mm`
body rise at stair one; the tracked model scored `0/10` deterministically and
never passed stair one. This validates the exact-world/training/recording
pipeline while reinforcing, not overturning, the existing rated-torque
feasibility failure.

V7 keeps the same exact `180 mm` rise, `250 mm` tread, and measured
`0.8825985 N m` hardware cap, but removes the policy's perfect analytic
terrain-height samples. It casts an `8 x 8` VL53L5CX-style grid from a
`40 deg` downward mount at the real full-grid limit of `15 Hz`, adds bounded
range noise, `5%` modeled zone dropout, and one-frame latency, then median
compresses it to 24 left/center/right row depths. The policy grows from 68 to
84 values. IMU and joint feedback remain inputs; RGB camera pixels are only
used for the review recording.

The 512-step transfer smoke and live sensor validator passed. The validator
delivered new samples only at control frames `8`, `16`, and `24`; all rays hit
the first two authored stair layers. The deterministic four-step evaluation
still scored `0/1`, and the recording stopped at the seven-second progress
gate without reaching stair one. Review the honest run and depth grid at
https://drobot-stair-sensor-eval.romka.chatgpt.site. This validates the cheap
sensor/training/recording path, not locomotion or hardware readiness.

The fixed-geometry per-leg prerequisite is now independently verified before
adding vision. V8 places the front-left foot on the first `180 x 250 mm` tread
in `5/5` force-backed episodes, and V10 mirrors that result for the front-right
foot in `5/5`. V13 and V15 separately hold the right and left feet above
`190 mm` under the measured `0.8825985 N m` effort cap (`3/3` and `5/5`
strict episodes). All four policies report `analytic_height_profile`,
`rgb_camera_policy_input: false`, and no custom traction material.

V16 is a deliberately separate front-pair composition experiment. Its new
closed-loop transfer uses the mass-weighted COM of all 13 rigid links and a
bounded target inside the three-foot support triangle. The right-to-left
transfer now finishes with `103.99 mm` support margin, `13.48 mm` COM-target
error, `23.42 N` tread load, and `4.58 deg` tilt. A 4,096-step support-residual
run then produced three consecutive stable `140 mm` successes with
`153.44-153.83 mm` lift, `14.6-14.9 mm` slip, and at most `5.0 deg` tilt.
The sequential `190 mm` stage is still unsolved: a gentle rear-leg thrust is
stable but reaches about `150 mm`, while a stronger thrust reaches `170 mm`
and tips. A higher-friction sensitivity reached only `162 mm` and still tipped,
so more traction is not the default next change. The remaining problem is
dynamic body elevation and support recovery after the first front foot is on
the stair, not terrain perception.

V17 isolates the user's prerequisite again and confirms the hardware model can
raise one front foot high enough while balanced. It fine-tunes the V15 policy
for 4,096 additional steps directly at the strict `190 mm` stage beside the
exact `250 mm` tread. All 12 recent training episodes passed, followed by a
`5/5` deterministic evaluation with `205.55-207.75 mm` lift, `0.5 s` hold,
`3.27 mm` maximum support slip, `2.33 deg` maximum tilt, and at least
`43.31 mm` final support margin. The policy is camera-blind; the external
camera is only for the review video. This proves isolated lift and balance,
not the second-foot transfer or a complete climb.

V35 then demonstrated that the composed sequence can physically raise the
rear-right foot `192.3 mm` beside the exact `180 x 250 mm` tread while staying
upright for a 65-second run, but the foot did not land and the run did not
complete a stair. V36 isolates the unstable state immediately after the
front-left-to-rear-right transfer. A bounded 64-action search found one
two-second support catch where the zero action tipped at `1.75 s`; that action
initialized a compact nine-action, 95-observation PPO policy. The 4,096-step
seed-840 run completed 27 cached target holds and saved model SHA-256
`4b45219e9e2d2d071c4373f152278e8e5fb67a9d00a5d74a469af875b3dffef8`.

Independent seed-832 composition completed both precursor transfers and stayed
upright through 65- and 80-second horizons with no failure. Maximum tilt was
`16.95 deg`, but the rear-right foot plateaued at `150.05 mm` physical lift;
seed 841 tipped in the older precursor transfer before V36 activated. V36 is
therefore a post-transfer stabilization milestone, not proof of the requested
190 mm lift, tread contact, or stair climbing. The next run must jointly train
this support catch with the frozen V35 swing policy and reward clearance plus
upright hold before introducing lowering/contact. The policy remains RGB-free:
IMU/proprioception, contact/load, COM/support state, phase, and analytic stair
geometry are inputs; the camera only records review evidence.

V37 adds that joint composition without retraining the already verified swing
policies: V17 supplies the full rear-right swing baseline, the compact V35 actor
adds a bounded `0.5` swing residual, and V36 controls the other nine joints at
the full action authority on which it was trained. Rear-stance pitch correction
is held off for two seconds, then restored through a two-second smooth blend.
The seed-832 controller diagnostic stayed upright for the full `65 s` horizon
with no failure and `15.33 deg` worst tilt, but reached only about `155.5 mm`
rear-right rise. Its report is tracked as
`ppo-stairs-v36-post-transfer-catch-small/evaluation_v37_joint_composition_seed832.json`.
The primary manifest check is explicitly marked skipped because no exact V37
model existed; every composed dependency hash and observation adapter passed.
This isolates the mixed-height pose/reference reach as the problem and is not a
V37 training success. The corrected full-authority V37 training wrapper remains
unrun.

The exact diagnostic command was:

```powershell
& C:\isaacsim\python.bat simulation/isaac/rl/stairs/evaluate_stairs_ppo.py `
  --config simulation/isaac/rl/stairs/quadruped_stairs_v37_joint_clearance_support.yaml `
  --model simulation/isaac/models/ppo-stairs-v36-post-transfer-catch-small/drobot_stairs_ppo_final.zip `
  --episodes 1 --seed 832 --device cpu --active-steps 1 `
  --placement-level left-center-tread-load --episode-seconds 65 `
  --maximum-lateral-deviation-m 0.30 `
  --leg-model front_right=simulation/isaac/models/ppo-stairs-v10-180mm-25cm-front-right-placement-small/drobot_stairs_ppo_final.zip `
  --leg-model front_left=simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
  --leg-model rear_right=simulation/isaac/models/ppo-stairs-v36-post-transfer-catch-small/drobot_stairs_ppo_final.zip `
  --leg-residual-support-only rear_right --leg-compact-action rear_right `
  --leg-residual-scale rear_right=1.0 `
  --leg-base-model rear_right=simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
  --leg-base-swing-only rear_right `
  --leg-base-residual-model rear_right=simulation/isaac/models/ppo-stairs-v35-rear-right-190mm-lift-small/drobot_stairs_ppo_final.zip `
  --leg-base-residual-scale rear_right=0.5 `
  --leg-base-residual-swing-only rear_right `
  --leg-base-residual-compact-action rear_right `
  --zero-action-leg rear_left --allow-unverified-model `
  --report simulation/isaac/output/rl/ppo-stairs-v37-joint-clearance-support-4096-seed842/evaluation-staged-baseline-seed832.json
```

V3 then separates raw rear-leg capability from that stair-pose transfer. A
fresh `512`-step policy raised the rear-right foot on flat ground while balancing
on the other three physical feet. Independent seed-13291 evaluation passed
`5/5`: every episode held the strict `190 mm` gate for `0.75 s`, maximum lift
was `199.85-203.66 mm`, worst tilt was `2.12 deg`, and support-foot lift stayed
below `4.99 mm`. The recording replay reached `200.81 mm` with `1.73 deg` worst
tilt. There is no RGB policy input; the external camera only generated the MP4.
This proves clearance and simple-pose balance, not mixed-height transfer,
landing, or ascent. The stair experiments remain fixed at `250 mm` tread depth.

V38 reintroduces the mixed-height pose but keeps the outcome deliberately
bounded: front-right and front-left first complete their verified tread
placements, an analytic composite-COM transfer unloads rear-right, and a small
support-residual PPO policy is scored only on a physical `190 mm` rear-right
lift and no-fall hold. A cached-state target search added `20 mm` forward and
`40 mm` toward rear-left, changing the old rear transfer from a negative margin
to about `40 mm` positive margin. The uninterrupted replay required a
rear-right-specific `3 N` preload gate held for `0.50 s`; other legs keep the
global `5 N` threshold, and the unloaded gate remains `<= 1 N`.

Two `1,024`-step transfer-residual pilots were run and rejected: both actors
slightly reduced next-foot preload and timed out before unload. The accepted
controller therefore keeps the positive-margin analytic transfer and trains
PPO only after that handoff. The final bounded training ran for `512` steps at
seed `847`. Independent seed `848` passed model and loaded-algorithm checks,
completed both transfers, raised rear-right `217.319 mm`, and terminated with
no failure. Rear-transfer completion margin was `38.519 mm`; maximum body tilt
was `11.2504 deg` under the exact `0.8825985 N m` effort cap.

This is not a stair-climb claim. Rear-right was not lowered onto the tread,
rear-left did not move, and the controller did not ascend even one complete
stair. Maximum simulated support slip was `57.582 mm`, and requested PD effort
was at or above `95%` of the cap in `42.29%` of samples. The next experiment
must isolate rear-right lowering/contact, then mirror an independently searched
positive-margin rear-left transfer. Real rubber-pad friction and compliance
should be measured before changing traction parameters; RGB vision is not the
current bottleneck. The policy uses IMU/proprioception, joint state,
contact/load, composite COM/support state, phase, previous action, and analytic
stair geometry. The external camera is recording-only.

The exact bounded training command is:

```powershell
& simulation\isaac\rl\stairs\train_stairs_v38_rear_right_lift_small.ps1 `
  -OutputDir simulation\isaac\output\rl\ppo-stairs-v38-rear-right-190mm-512-seed847 `
  -Seed 847
```

The accepted policy, exact config, reports, limitations, and full evaluation
command are packaged under
`simulation/isaac/models/ppo-stairs-v38-rear-right-190mm-small/`. The accepted
external-camera video is tracked as
`reviews/ppo-stairs-v38-rear-right-190mm-lift-seed848.mp4` and hosted privately
at https://drobot-design-review.romka.chatgpt.site.

V39-V41 then isolated rear-right landing without changing the exact
`180 x 250 mm` stair or `0.8825985 N m` effort cap. The `512`-step V39 support
pilot preserved `218.095 mm` lift but produced a `351.313 N` edge contact and
`19.9904 deg` tip. The `1,024`-step V40 swing pilot stayed at `10.7597 deg` and
`37.939 mm` margin but stopped at world X `0.318371 m` with no tread load. A
bounded V41 search reached world X `0.557554 m`, still short of clearing the
`12.5 mm` foot radius past the `0.550 m` riser, and corner contact tipped the
body. A sequenced body-shift test stayed upright but exposed rear-hip tracking
of only about `0.68 rad` against a `1.83 rad` reference under the measured cap.
No landing pilot is promoted. Commands, assumptions, local report names, and
the traction-versus-vision conclusion are recorded in
[`simulation/isaac/rl/stairs/REAR_RIGHT_LANDING_SEARCH.md`](rl/stairs/REAR_RIGHT_LANDING_SEARCH.md).

V42-V44 then corrected the support geometry before contact instead of asking a
post-impact residual to recover. V42 clips the analytic balance target to a
positive support-polygon margin; V43 instruments compliant tread-load
feedback; and the accepted V44 controller uses asymmetric front support reach,
bounded IMU pitch regulation, and a post-clearance contact latch. The 512-step
seed-869 PPO run completed in `77.254 s`. Because that horizon ends before
touchdown, acceptance comes from a fresh seed-870 evaluation: rear-right held
force-backed tread contact for `45/45` frames after `217.990 mm` lift, with all
three support feet loaded, `39.443 mm` minimum support margin, `14.000 mm`
maximum support slip, and `10.238 N` maximum tread load. The exact stair remains
`180 mm` rise by `250 mm` tread and the applied effort cap remains
`0.8825985 N m`.

This advances the first-tread sequence through rear-right but is still not a
complete stair. V45 now caches the exact accepted rear-right landing boundary
and trains all 12 joint residuals only on the next COM transfer. The bounded
seed-875 run completed 4,096 steps in `135.744 s`, but completed `0` transfers
and lifted rear-left only `15.136 mm`. A deterministic seed-876 replay tipped
after 215 steps as COM-target error grew from `128.454 mm` to `153.372 mm`.
All 27 constant loaded-support hip-abduction probes also failed. The next
target is a post-landing rear-right sidestep-and-settle stage that physically
widens the support polygon before rear-left unloading. The policy remains
camera-blind: the external camera only records review video, while inference
uses IMU/proprioception, joint, contact/load, COM/support, phase, previous
action, and known fixed-stair geometry.

V46-V49 then tested that prerequisite without changing the hardware-informed
contract. The original V46 outward replacement moved rear-right `9.215 mm` but
retained only `2.301 N`. V48 instead found a `33.739 N` force-backed settle at
`11.457 deg`, although the foot moved `5.203 mm` inward and the future
rear-left support margin remained `-94.730 mm`. A nominal `5 mm` forward /
`5 mm` lateral preload from that stronger contact appeared to gain about
`14 mm` margin, but actual composite-COM progress was below `0.4 mm`, rear-
right load fell to `0 N`, and the feet slipped about `15 mm`. The V49 search
now requires measured COM progress and reports continuous rear-right load.
Even a deliberately high `1.8/1.5` static/dynamic friction sensitivity with
`max` combine produced only `0.312 mm` useful COM progress and lost the same
contact. The next intervention is therefore calibrated wide/compliant rubber
foot geometry plus active normal-force retention, not RGB/depth vision or a
larger PPO budget. Exact commands and hashes are in
[`simulation/isaac/rl/stairs/PROGRESSIVE_PRELOAD_SEARCH.md`](rl/stairs/PROGRESSIVE_PRELOAD_SEARCH.md).

The accepted model, exact config, fresh evaluation, search evidence, recording
report, and limitations are packaged under
`simulation/isaac/models/ppo-stairs-v44-early-contact-rear-right-landing-small/`.
The accepted 331-frame video is a phase-local seed-870 replay that completes
the strict landing hold. A separate four-attempt reset-to-contact recording
search found no full-prefix success, so the site and package label that
instability instead of presenting the phase clip as an end-to-end climb.

The rejected V45 model, exact snapshot, 4,096-step training report,
deterministic evaluation, controller audits, and negative-evidence recording
are packaged under
`simulation/isaac/models/ppo-stairs-v45-rear-left-dynamic-transfer-4096/`.

The source map, complete commands, reward/termination formulas, manifest
rules, measured smoke results, evaluation guidance, and sim-to-real limits are
owned by [`docs/rl-stairs/README.md`](../../docs/rl-stairs/README.md), with
the corrected experiment recorded separately in
[`docs/rl-stairs-v2/README.md`](../../docs/rl-stairs-v2/README.md), and the
hardware-informed residual result in
[`docs/rl-stairs-v5/README.md`](../../docs/rl-stairs-v5/README.md), with the
full-size bounded result in
[`docs/rl-stairs-v6-180mm/README.md`](../../docs/rl-stairs-v6-180mm/README.md),
and the multi-zone ToF experiment in
[`docs/rl-stairs-v7-vl53l5cx/README.md`](../../docs/rl-stairs-v7-vl53l5cx/README.md).
The corrected composite-COM training contract, V19-V21 transfer sensitivities,
and the traction-versus-vision recommendation are recorded in
[`docs/rl-stairs-v19-v21-transfer-audit/README.md`](../../docs/rl-stairs-v19-v21-transfer-audit/README.md).

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
