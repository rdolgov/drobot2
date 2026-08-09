# Hardware-informed stair PPO v3

## Status

This is a separate simulator experiment that promotes selected 2026-07-28
one-leg bench observations into an Isaac stair-training profile. It does not
change the flat-walking, stair v1, or stair v2 contracts.

The short run recorded for this change is a pipeline smoke training. It is not
a converged stair-climbing policy, a full-quadruped hardware validation, or
authorization to test stairs on the physical robot.

## What changed

`simulation/isaac/rl/stairs/quadruped_stairs_v3.yaml` records the local
one-leg test configuration:

- the neutral encoder center is tick `2048`;
- exercised joint ranges are hip abduction `-45 to +45 deg`, hip flexion
  `-90 to +90 deg`, and knee `-120 to +120 deg`;
- hip flexion and knee encoder directions are reversed relative to raw
  positive register motion;
- torque limit register `300/1000` is modeled as 30% of published stall
  torque, or `0.8825985 N*m`;
- speed register `350` is retained as provenance but is not converted to
  rad/s because the bench run did not calibrate that relationship.

The generic Isaac environment now applies optional task-specific joint limits
and effort caps after the articulation opens. The applied values are included
in the saved environment/model manifest. Existing tasks omit the profile and
retain their previous authored limits and rated effort cap.

The v3 action box is wider than v2 so PPO can use the newly available
sagittal range. Exploration starts lower (`log_std=-0.90`) to reduce violent
first updates. Geometry, 40 mm risers, observations, rewards, mastery
curriculum, and the 100k progress gate remain inherited from v2.

## Source of truth

- The machine-local inputs read on 2026-07-28 were later promoted to the
  tracked `hardware/robot-runtime/servos/leg-1.toml` and
  `hardware/robot-runtime/servos/calibration-leg-1.json` shared profiles.
- `conversations/2026-07-28-one-leg-wall-isaac-revisit.md` is the durable
  evidence summary.
- `simulation/isaac/rl/stairs/quadruped_stairs_v3.yaml` is the committed
  learning configuration.
- `simulation/isaac/rl/stairs/train_stairs_v3_ppo.py` is the dedicated entry
  point.

The encoder directions are documented for transfer tooling but are not
applied to Isaac joint coordinates. The URDF already authors left/right joint
axes in a common robot-space command convention.

## Reproduce

Run from the repository root:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\train_stairs_v3_ppo.py `
  --smoke-test `
  --total-timesteps 4096 `
  --initialize-from-flat `
  simulation\isaac\output\rl\ppo-walk-v1-2m\drobot_walk_ppo_final.zip
```

Evaluate the first curriculum step:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v3.yaml `
  --model simulation\isaac\output\rl\ppo-stairs-v3\drobot_stairs_ppo_final.zip `
  --episodes 2 `
  --active-steps 1 `
  --screenshot reviews\ppo-stairs-v3-hardware-smoke.png `
  --report simulation\isaac\output\rl\ppo-stairs-v3\evaluation_report.json
```

Record one deterministic first-step episode:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\record_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v3.yaml `
  --model simulation\isaac\output\rl\ppo-stairs-v3\drobot_stairs_ppo_final.zip `
  --active-steps 1 `
  --video reviews\ppo-stairs-v3-hardware-smoke.mp4 `
  --thumbnail reviews\ppo-stairs-v3-hardware-smoke-recording.png `
  --report simulation\isaac\output\rl\ppo-stairs-v3\recording_report.json
```

## Validation and outputs

Validation performed on 2026-07-31:

- focused stair contract tests: `13 passed, 1 skipped`;
- repository test suite: `188 passed, 1 skipped`;
- Ruff on the changed Python files: passed;
- Python `compileall` on the changed Python files: passed;
- Isaac 6.0.1 smoke training: passed, 4,096 timesteps in `117.90 s`;
- saved PPO/model manifest contract verification: passed;
- two deterministic first-step evaluation episodes: pipeline passed, policy
  success `0/2`;
- deterministic MP4 recording and thumbnail encoding: passed.

The smoke training completed 23 episodes. No episode reached step one, and
the maximum base elevation gain was `0.00126481 m`. Both deterministic
evaluation episodes terminated because the body tipped, with a mean return of
`-2.73427`, mean forward displacement of `-0.00602 m`, and maximum body tilt
of `48.1632 deg`. The recorded episode ran for 31 control steps (`0.5167 s`),
moved forward `0.00349 m`, and tipped before reaching the stair.

These are baseline failure metrics, not evidence of learned climbing. The
exact machine-readable summary is
`reviews/ppo-stairs-v3-hardware-smoke-results.json`.

Durable review files:

- `reviews/ppo-stairs-v3-hardware-smoke.mp4`;
- `reviews/ppo-stairs-v3-hardware-smoke.png`;
- `reviews/ppo-stairs-v3-hardware-smoke-recording.png`;
- `reviews/ppo-stairs-v3-hardware-smoke-results.json`.

Run products and the smoke model under
`simulation/isaac/output/rl/ppo-stairs-v3/` remain ignored because they are
diagnostic artifacts, not release models.

## Limitations

The promoted ranges were exercised on one unloaded wall-mounted leg. They do
not prove cable clearance, self-collision clearance, support capacity,
current margin, voltage sag, or thermal endurance on the complete 4.53 kg
quadruped. The torque-register fraction is a nominal simulator mapping, not a
calibrated torque measurement. The robot still uses simulation-only spherical
fork-tip contacts instead of physical grippy feet.
