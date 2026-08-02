# Supported and unsupported 190 mm single-foot-lift PPO skills

## Unsupported balance result

V2 removes the virtual torso hold and passes the requested flat-ground gate.
The final `512`-step PPO smoke checkpoint completed `5/5` deterministic
episodes with measured lift `197.064-199.411 mm`, maximum tilt `2.535 deg`,
maximum support-foot lift `4.860 mm`, maximum planned XY torso displacement
below the `80 mm` gate, and a continuous `0.75 s` hold. The recorded episode
reached `199.369 mm` with `2.270 deg` maximum tilt and completed the same gate.

This is an unsupported simulation result: `base_support.mode` is `none` and
the three stance feet remain physical contacts. It is still a short PPO
pipeline run around an analytic weight-shift/lift reference, not convergence
or hardware proof.

This experiment isolates the first stair prerequisite: raise the front-left
foot by `190 mm` while a virtual external support holds the settled torso pose.
This deliberately isolates leg authority from unsupported balance. It removes
the staircase, forward locomotion, and terrain perception. Analytic
two-link IK supplies a smooth raise-forward foot reference while PPO learns
bounded residuals on all 12 joints for whole-body stability.

The final stair requirement remains `180 mm` rise with `250 mm` tread depth.
This task does not climb or perceive the stair; it tests the clearance skill
needed before tread placement.

## Source of truth

| File | Purpose |
| --- | --- |
| `simulation/isaac/rl/foot_lift/quadruped_foot_lift_v1.yaml` | Target, timing, measured hardware limits, reward, gates, and PPO settings |
| `simulation/isaac/rl/foot_lift/quadruped_foot_lift_v2_balance.yaml` | Unsupported stance, five-stage lift curriculum, diagonal weight transfer, support margin, final gates, and PPO settings |
| `simulation/isaac/rl/foot_lift/_foot_lift_contract.py` | Pure observation, reward, success, and failure contract |
| `simulation/isaac/rl/foot_lift/_quadruped_foot_lift_env.py` | Isaac articulation, IK reference, PPO residuals, foot measurement, and metrics |
| `simulation/isaac/rl/foot_lift/train_foot_lift_ppo.py` | PPO training and model manifest |
| `simulation/isaac/rl/foot_lift/evaluate_foot_lift_ppo.py` | Deterministic evaluation and screenshot |
| `simulation/isaac/rl/foot_lift/record_foot_lift_ppo.py` | H.264 evaluation recording |

## Task contract

- Swing leg: `front_left`.
- Required measured lift: `0.19 m` relative to the settled foot tip. The IK
  reference commands `0.20 m` to cover the measured position-tracking error.
- Initial stance: `290 mm` crouch, preserving leg reach for the lift arc.
- Reference: ramp from `0.50 s` through `3.00 s`, commanding `200 mm` lift and moving
  forward `110 mm`, then hold. A purely vertical target would require
  about `135 deg` of knee bend; the arc remains within the measured `120 deg`
  knee limit.
- Control: analytic IK reference plus bounded 12-joint PPO residuals.
- Initialization: zero residual mean around the analytic reference.
- Support: an explicit `pose_hold` training harness fixes the torso. This is
  supported-leg evidence, not proof of independent three-foot balance.
- Hardware profile: measured joint directions/ranges and `0.8825985 N m`
  effort cap from the one-leg test.
- Policy inputs: the 48-value IMU/joint/prior-action walking contract plus
  desired/measured lift, lift error/progress, base height/drift, and support
  foot lift. RGB and terrain vision are excluded.
- Success: measured `190 mm` lift held `0.75 s`, body tilt at most `12 deg`,
  base height and planar drift within `10 mm`, and each support
  foot within `25 mm` of its settled height.

V2 changes the operating point to a `300 mm` stance and a `45 mm` rear-right
diagonal weight-transfer reference. Its policy keeps the same `56` inputs and
12 actions, but the swing residual is tightly bounded so PPO cannot cancel the
analytic lift. The curriculum stages are `20`, `50`, `90`, `140`, and
`190 mm`. A signed margin from the measured torso projection to the current
three-foot support triangle supplies a coordinate-invariant balance reward.
The planned diagonal shift is accepted up to `80 mm`; arbitrary drift still
terminates at `150 mm`.

## Reproduction

From `robot-cad`, run:

```powershell
python -m pytest tests/test_quadruped_foot_lift_rl_contract.py -q

& C:\isaacsim\python.bat `
  simulation/isaac/rl/foot_lift/train_foot_lift_ppo.py `
  --smoke-test

& C:\isaacsim\python.bat `
  simulation/isaac/rl/foot_lift/evaluate_foot_lift_ppo.py `
  --episodes 3 `
  --screenshot reviews/ppo-foot-lift-v1-190mm-evaluation.png

& C:\isaacsim\python.bat `
  simulation/isaac/rl/foot_lift/record_foot_lift_ppo.py

$source = 'simulation/isaac/output/rl/ppo-foot-lift-v1-190mm'
$release = 'simulation/isaac/models/ppo-foot-lift-v1-190mm-small'
New-Item -ItemType Directory -Force $release | Out-Null
Copy-Item "$source/drobot_foot_lift_ppo_final.zip*" $release -Force
Copy-Item "$source/*report.json" $release -Force
```

Run and package the final unsupported smoke policy:

```powershell
& C:\isaacsim\python.bat `
  simulation/isaac/rl/foot_lift/train_foot_lift_ppo.py `
  --config simulation/isaac/rl/foot_lift/quadruped_foot_lift_v2_balance.yaml `
  --final-stage-only `
  --smoke-test `
  --seed 12190 `
  --output-dir simulation/isaac/output/rl/ppo-foot-lift-v2-balance-190mm-zero-small

& C:\isaacsim\python.bat `
  simulation/isaac/rl/foot_lift/evaluate_foot_lift_ppo.py `
  --config simulation/isaac/rl/foot_lift/quadruped_foot_lift_v2_balance.yaml `
  --model simulation/isaac/output/rl/ppo-foot-lift-v2-balance-190mm-zero-small/drobot_foot_lift_ppo_final.zip `
  --episodes 5 `
  --seed 12191 `
  --report simulation/isaac/output/rl/ppo-foot-lift-v2-balance-190mm-zero-small/evaluation_report.json

& C:\isaacsim\python.bat `
  simulation/isaac/rl/foot_lift/record_foot_lift_ppo.py `
  --config simulation/isaac/rl/foot_lift/quadruped_foot_lift_v2_balance.yaml `
  --model simulation/isaac/output/rl/ppo-foot-lift-v2-balance-190mm-zero-small/drobot_foot_lift_ppo_final.zip `
  --seed 12192 `
  --video reviews/ppo-foot-lift-v2-balance-190mm-evaluation.mp4 `
  --thumbnail reviews/ppo-foot-lift-v2-balance-190mm-recording.png `
  --report simulation/isaac/output/rl/ppo-foot-lift-v2-balance-190mm-zero-small/recording_report.json
```

## Validation status

Validated on Isaac Sim `6.0.1` with the real-test `0.8825985 N m` effort cap:

- focused pure contract tests: `8 passed`;
- PPO smoke training: status `PASS`, `512` steps; this is pipeline validation,
  not convergence evidence;
- deterministic evaluation: `3/3` successful supported lifts, measured range
  `192.65-197.49 mm`, maximum tilt `1.47 deg`, no failure reasons;
- recording: status `PASS`, `108` H.264 frames at `960 x 540`, `30 FPS`;
  measured lift `194.05 mm`, final lift `193.92 mm`, `0.75 s` hold, maximum
  tilt `0.38 deg`, maximum support-foot lift `6.05 mm`, no failure reasons.

The result establishes supported single-leg clearance in simulation. It does
not establish unsupported three-foot balance or a stair climb.

### Fresh rear-right seed-941 validation

The unchanged V3 rear-right task was rerun after the stair-transfer diagnosis
to answer the simpler capability question independently. The `512`-step
seed-941 smoke run completed its training episode at `196.700 mm` lift and
`2.362 deg` maximum tilt. Fresh deterministic seed-942 evaluation passed
`5/5`, with `201.006-204.345 mm` maximum lift, `2.218 deg` worst tilt, and no
failure reason. The seed-943 recording reached `202.907 mm`, retained a
positive `1.271 mm` minimum support-triangle margin, and held the strict gate
for `0.75 s`.

This run uses the `one-leg-real-test-2026-07-28` joint directions, limits, and
`0.8825985 N m` effort cap. Policy input is camera-blind: IMU, joint state,
previous action, lift target/progress, and body state. The external camera only
records the MP4. The tracked package is
`simulation/isaac/models/ppo-foot-lift-v3-rear-right-190mm-seed941-small/`.
The model SHA-256 is
`7f3ccb0a159140de47eb99d8ad71c0eeabf3692a6dd712e36c44206c4e9d279c`.

## Generated outputs

| Artifact | Path |
| --- | --- |
| Small PPO checkpoint and manifest | `simulation/isaac/models/ppo-foot-lift-v1-190mm-small/` |
| Three-episode evaluation report | `simulation/isaac/models/ppo-foot-lift-v1-190mm-small/evaluation_report.json` |
| Recording report | `simulation/isaac/models/ppo-foot-lift-v1-190mm-small/recording_report.json` |
| Review video | `reviews/ppo-foot-lift-v1-190mm-evaluation.mp4` |
| Best-lift thumbnail | `reviews/ppo-foot-lift-v1-190mm-evaluation.png` |
| Unsupported PPO checkpoint, manifest, and reports | `simulation/isaac/models/ppo-foot-lift-v2-balance-190mm-small/` |
| Unsupported review video | `reviews/ppo-foot-lift-v2-balance-190mm-evaluation.mp4` |
| Unsupported evaluation image | `reviews/ppo-foot-lift-v2-balance-190mm-evaluation.png` |
| Fresh rear-right seed-941 package | `simulation/isaac/models/ppo-foot-lift-v3-rear-right-190mm-seed941-small/` |
| Fresh rear-right seed-943 video | `reviews/ppo-foot-lift-v3-rear-right-190mm-fresh-seed943.mp4` |
| Fresh rear-right seed-943 thumbnail | `reviews/ppo-foot-lift-v3-rear-right-190mm-fresh-seed943.png` |

## Assumptions and limitations

- The IK reference supplies the nominal lift trajectory; PPO is learning joint
  residuals around a supported torso, not discovering the path from scratch.
- The next curriculum stage must change `base_support.mode` to `none` and add
  weight transfer before claiming unsupported balance.
- The virtual distal contact remains the provisional `12.5 mm` sphere.
- Servo effort is capped, but current, voltage sag, backlash, compliance, and
  temperature are not simulated.
- A short PPO run validates the pipeline and may expose feasibility; it is not
  convergence or hardware proof.
- The support-triangle calculation uses the base-link origin as the simulated
  torso projection; a hardware controller should use a calibrated center of
  mass estimate and contact detection.
- RGB remains recording-only. The unsupported policy still uses IMU, joints,
  prior action, and task state; it does not use camera or ToF input.
