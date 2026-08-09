# Hardware-profiled 180 mm stair PPO v6

## Unsupported-balance transfer follow-up

The verified unsupported `190 mm` flat-ground balance policy is now a
supported stair initializer. `train_stairs_ppo.py --initialize-from-balance`
copies only the shared `48`-value IMU/joint/prior-action prefix from the
`56`-value balance policy into the `68`-value v6 stair policy. The eight
lift-specific source columns are deliberately dropped, all 20 stair-specific
input columns start at zero, and output weights/biases are rescaled per joint
to preserve the source policy's physical action mean under the larger stair
residual scales.

The exact `180 mm` rise and `250 mm` tread smoke integration passed its
`512`-step pipeline, but locomotion still failed. Deterministic evaluation was
`0/5`; two episodes reached stair index one, the best base elevation gain was
`18.003 mm`, and no foot reached the first tread. Failure counts were two base
clearance, one body tip, and two each for no forward and no foot-tread
progress. The recorded attempt reached stair index one and failed base
clearance with `14.911 deg` maximum tilt.

This proves the transfer plumbing, not a stair climb. The next policy needs an
explicit per-foot lift/placement reference or curriculum that invokes the
validated clearance primitive in the required foot sequence; weight transfer
encoded only in a flat single-leg actor is not enough for four-step locomotion.

## Status

V6 defines and runs the requested four-step Isaac Sim task with exactly
`180 mm` rise and `250 mm` tread depth. Static world generation, bounded PPO
training, model packaging, contract verification, deterministic evaluation,
and H.264 recording all passed their respective pipelines. The locomotion
objective did not pass.

The final `10,240`-step run completed `40` training episodes with `0`
successes. Its best body elevation gain was `172.196 mm`, and the body reached
only the first of four stairs. The packaged policy then completed `0/10`
deterministic evaluations; the best evaluated body rise was `103.513 mm`, and
the maximum stair index was one. The reviewed recording is
`reviews/ppo-stairs-v6-180mm-25cm-small-training.mp4`. The same clip and result
JSON are on the private Sites review page:
https://drobot-stairs-v3-smoke-20260731.romka.chatgpt.site.

This is an honest small-training result for evaluation, not a converged policy
or hardware-deployment checkpoint.

## What changed

- `quadruped_stairs_v6_180mm.yaml` keeps tread depth fixed at `0.25 m` while
  providing `10`, `20`, `30`, `40`, `60`, `80`, `100`, `120`, `140`, `150`,
  `160`, and `180 mm` height stages.
- The final world contains four `180 mm` static collision layers. Its exposed
  top surfaces are `0.18`, `0.36`, `0.54`, and `0.72 m` high, and every tread
  begins `0.25 m` after the previous one.
- The real-test profile remains authoritative: `0.8825985 N m` effort cap,
  hip-abduction limits of `-45..45 deg`, hip-flexion limits of
  `-90..90 deg`, knee limits of `-120..120 deg`, and `120 Hz` physics/control.
- The strict success gate requires the goal position, at least 90% of the
  expected base elevation, all four feet on the active landing, and a
  continuous `0.5 s` hold.
- Four normalized foot-to-tread progress values and a four-value next-foot
  one-hot were added, expanding the stair observation from 60 to 68 values.
  The placement sequence is front-left, front-right, rear-left, rear-right.
- Training can transfer a smaller stair actor to the 68-value input. Existing
  policy/value input columns are copied exactly and new columns start at zero.
- A stalled episode now fails when it lacks either forward progress or any
  meaningful foot-to-tread progress after seven seconds.
- `train_stairs_v6_180mm_ppo.py` selects the v6 config, final `180 mm` stage,
  and output defaults while preserving command-line overrides.

## Policy inputs and camera use

The policy does **not** use camera images. It consumes simulated IMU values,
joint positions/velocities, prior actions, navigation state, analytic
simulator terrain-height samples, foot-to-tread progress, and the next-foot
target. The external RTX camera is used only to record the evaluation video.

The analytic height samples are not a hardware sensor implementation. A real
deployment still needs camera/depth processing or another terrain estimator
that reproduces this observation contract.

## Source of truth

- `simulation/isaac/rl/stairs/quadruped_stairs_v6_180mm.yaml` owns geometry,
  height stages, hardware values, inputs, reward, termination, curriculum, and
  PPO settings.
- `simulation/isaac/rl/stairs/_stair_rl_contract.py` owns pure observation,
  strict-goal, foot-progress, reward, and curriculum functions.
- `simulation/isaac/rl/stairs/_quadruped_stairs_env.py` owns Isaac physics,
  physical fork-tip sampling, sequence tracking, success, and episode metrics.
- `simulation/isaac/rl/stairs/train_stairs_ppo.py` owns strict transfer,
  training, checkpoint manifests, and reports.
- `simulation/isaac/rl/stairs/train_stairs_v6_180mm_ppo.py` owns v6 launcher
  defaults.
- Editable YAML and Python are authoritative. USD worlds, ZIP models, reports,
  and media are generated artifacts.

## Reproduce

Generate the fixed-depth entry and final worlds:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\create_stairs_world.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v6_180mm.yaml `
  --height-stage 10mm `
  --report simulation\isaac\output\rl\stairs-v6-010mm-world-report.json

& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\create_stairs_world.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v6_180mm.yaml `
  --height-stage 180mm `
  --report simulation\isaac\output\rl\stairs-v6-180mm-world-report.json
```

Run the bounded `10 mm` entry transfer, keeping `250 mm` treads:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v6_180mm.yaml `
  --height-stage 10mm `
  --fixed-active-steps 4 `
  --total-timesteps 20000 `
  --initialize-from-stairs `
  simulation\isaac\models\ppo-stairs-v5-10mm-four-step\drobot_stairs_ppo_initialized.zip `
  --seed 6310 `
  --output-dir `
  simulation\isaac\output\rl\ppo-stairs-v6-010mm-25cm-four-step-20k
```

Fine-tune that policy on the exact final staircase:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\train_stairs_v6_180mm_ppo.py `
  --fixed-active-steps 4 `
  --total-timesteps 10000 `
  --initialize-from-stairs `
  simulation\isaac\output\rl\ppo-stairs-v6-010mm-25cm-four-step-20k\drobot_stairs_ppo_final.zip `
  --seed 6480 `
  --output-dir `
  simulation\isaac\output\rl\ppo-stairs-v6-180mm-25cm-four-step-10k
```

Evaluate and record the tracked release package:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v6_180mm.yaml `
  --height-stage 180mm `
  --model `
  simulation\isaac\models\ppo-stairs-v6-180mm-25cm-small\drobot_stairs_ppo_initialized.zip `
  --active-steps 4 `
  --episodes 10 `
  --seed 6480 `
  --report `
  simulation\isaac\models\ppo-stairs-v6-180mm-25cm-small\evaluation_report.json

& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\record_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v6_180mm.yaml `
  --height-stage 180mm `
  --model `
  simulation\isaac\models\ppo-stairs-v6-180mm-25cm-small\drobot_stairs_ppo_initialized.zip `
  --seed 6480 `
  --active-steps 4 `
  --camera-view external `
  --fps 30 `
  --width 960 `
  --height 540 `
  --video reviews\ppo-stairs-v6-180mm-25cm-small-training.mp4 `
  --thumbnail reviews\ppo-stairs-v6-180mm-25cm-small-training.png `
  --report `
  simulation\isaac\models\ppo-stairs-v6-180mm-25cm-small\recording_report.json
```

## Validation performed

Validation on 2026-07-31:

- focused stair contract tests: `20 passed, 2 skipped`;
- full repository regression suite: `196 passed, 2 skipped`;
- Ruff on the changed stair scripts and contract tests: passed;
- Python compilation for the changed stair scripts: passed;
- `10 mm x 250 mm` and `180 mm x 250 mm` world generation: passed with
  four static collision layers and no stair rigid bodies;
- `10 mm` entry PPO: `20,480` actual steps, `0/10` completed-episode
  successes, maximum stair one, maximum body rise `14.071 mm`;
- exact `180 mm` PPO: `10,240` actual steps, `0/40` successes, maximum stair
  one, maximum body rise `172.196 mm`;
- tracked release packaging and exact same-shape parameter transfer: passed;
- release deterministic evaluation: model/algorithm contracts passed,
  `0/10` successes, maximum stair one, best body rise `103.513 mm`;
- release recording: contracts passed, 75 H.264 frames at `960 x 540` and
  30 FPS; the recorded attempt reached stair one and failed base clearance;
- private Sites page build and its two rendered/asset tests: passed; deployment
  version 3 succeeded.

The compact machine-readable result is
`reviews/ppo-stairs-v6-180mm-25cm-results.json`. Full reports and the release
manifest are in `simulation/isaac/models/ppo-stairs-v6-180mm-25cm-small/`.

## Limitations and next work

The earlier rated-torque scripted feasibility study failed at `180 mm` and set
`curriculum_authorized` to false. This user-requested bounded RL run does not
overturn that gate: it also failed to clear the first riser reliably. Raising
the simulated torque was deliberately not used to create an artificial pass.

The run is far too short to claim convergence, uses one simulated environment,
and lacks friction/mass/latency/battery randomization. Foot placement uses
simulated geometry rather than measured force, and analytic terrain input has
no robot-side equivalent yet. No four-leg hardware stair test was run.

Before more full-height PPO or any deployment, revise sustainable actuator
margin, body/leg geometry, contact and weight transfer, then rerun the scripted
rated-torque feasibility gate. Only after that passes should training proceed
through the fixed-depth height curriculum with a hardware-reproducible terrain
sensor and robust deterministic acceptance rate.

## Balance-transfer reproduction and artifacts

```powershell
& C:\isaacsim\python.bat `
  simulation/isaac/rl/stairs/train_stairs_ppo.py `
  --config simulation/isaac/rl/stairs/quadruped_stairs_v6_180mm.yaml `
  --height-stage 180mm `
  --fixed-active-steps 4 `
  --initialize-from-balance `
  simulation/isaac/models/ppo-foot-lift-v2-balance-190mm-small/drobot_foot_lift_ppo_final.zip `
  --smoke-test `
  --seed 13191 `
  --output-dir simulation/isaac/output/rl/ppo-stairs-v6-180mm-25cm-balance-small

& C:\isaacsim\python.bat `
  simulation/isaac/rl/stairs/evaluate_stairs_ppo.py `
  --config simulation/isaac/rl/stairs/quadruped_stairs_v6_180mm.yaml `
  --height-stage 180mm `
  --model simulation/isaac/output/rl/ppo-stairs-v6-180mm-25cm-balance-small/drobot_stairs_ppo_final.zip `
  --active-steps 4 `
  --episodes 5 `
  --seed 13192 `
  --report simulation/isaac/output/rl/ppo-stairs-v6-180mm-25cm-balance-small/evaluation_report.json

& C:\isaacsim\python.bat `
  simulation/isaac/rl/stairs/record_stairs_ppo.py `
  --config simulation/isaac/rl/stairs/quadruped_stairs_v6_180mm.yaml `
  --height-stage 180mm `
  --model simulation/isaac/output/rl/ppo-stairs-v6-180mm-25cm-balance-small/drobot_stairs_ppo_final.zip `
  --active-steps 4 `
  --seed 13192 `
  --video reviews/ppo-stairs-v6-180mm-25cm-balance-transfer.mp4 `
  --thumbnail reviews/ppo-stairs-v6-180mm-25cm-balance-transfer-recording.png `
  --report simulation/isaac/output/rl/ppo-stairs-v6-180mm-25cm-balance-small/recording_report.json
```

The packaged checkpoint, manifest, and reports are under
`simulation/isaac/models/ppo-stairs-v6-180mm-25cm-balance-small/`. The review
video and images are `reviews/ppo-stairs-v6-180mm-25cm-balance-transfer.*`.
