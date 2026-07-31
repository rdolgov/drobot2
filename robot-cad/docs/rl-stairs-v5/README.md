# Hardware-informed residual stair PPO v5

## Status

The v5 experiment produced one verified Isaac Sim climb over four `10 mm`
steps. The stochastic policy traveled `1.39269 m`, raised the base by as much
as `43.5829 mm`, placed all four feet on step four, stayed below `12.852 deg`
body tilt, and held the landing goal for `0.5 s` without a failure condition.
The reviewed recording is
`reviews/ppo-stairs-v5-10mm-four-step-success.mp4`, with a private hosted
handoff at https://drobot-stairs-v3-smoke-20260731.romka.chatgpt.site.

This is narrow proof that the updated policy can walk the shallow simulated
staircase. It is not a converged controller: the success appeared once in 81
stochastic attempts, and the packaged policy mean completed `0/10` fresh
deterministic episodes. No `20`, `30`, or `40 mm` stage completed, and no
quadruped hardware test was run.

## What changed

V4 first corrected two transfer errors found while auditing the v3 failure:
flat-policy output means are rescaled when action boxes differ, and the stair
task uses the flat controller's actual `120 Hz` action cadence. V5 then uses a
frozen, tracked flat-walking PPO model with a learned 12-value stair residual.
The residual keeps the measured one-leg profile from the 2026-07-28 test:

- `0.8825985 N m` effort cap, modeled from register `300/1000` and published
  stall torque;
- hip-abduction limits `-45..45 deg`, hip-flexion `-90..90 deg`, and knee
  `-120..120 deg`;
- `120 Hz` physics and control;
- a `24 s` horizon so the torque-limited rear-leg transfer can finish on the
  landing.

The environment now measures the physical fork-tip poses, rewards lift and
new tread placement, subtracts the stationary velocity-tracking baseline,
terminates stalled approaches, and requires base elevation plus a continuous
landing hold for success. Height stages generate `10`, `20`, `30`, and
`40 mm` worlds. The dedicated v5 launcher, same-shape stair transfer,
fixed-active-step training, initialization-only packaging, stochastic success
distillation, and exact episode-history recording are separate reproducible
paths rather than hidden local edits.

## Source of truth

- `simulation/isaac/rl/stairs/quadruped_stairs_v5.yaml` owns the hardware,
  residual-policy, terrain, reward, termination, curriculum, and PPO values.
- `simulation/isaac/rl/stairs/train_stairs_v5_ppo.py` selects v5 defaults.
- `simulation/isaac/rl/stairs/train_stairs_ppo.py` implements training,
  transfer, fixed levels, and initialization-only packaging.
- `simulation/isaac/rl/stairs/_quadruped_stairs_env.py` owns runtime physics,
  residual composition, fork-tip metrics, and the success predicate.
- `simulation/isaac/rl/stairs/distill_successful_stairs.py` collects only
  physically successful stochastic trajectories before fitting the actor
  mean.
- `simulation/isaac/rl/stairs/record_stairs_ppo.py` can replay the preceding
  seeded episodes so policy RNG and PhysX reset history remain exact.
- `simulation/isaac/models/ppo-walk-v1-2m/` and
  `simulation/isaac/models/ppo-stairs-v5-10mm-four-step/` contain the tracked
  base and stair policy packages.

The editable YAML and Python are authoritative. USD worlds, model ZIPs,
manifests, reports, and review media are generated artifacts.

## Reproduce

Generate a staged world from `robot-cad`:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\create_stairs_world.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v5.yaml `
  --height-stage 10mm `
  --report simulation\isaac\output\rl\ppo-stairs-v5-10mm\world_report.json
```

Run the bounded four-step training used for the recorded source policy:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\train_stairs_v5_ppo.py `
  --height-stage 10mm `
  --fixed-active-steps 4 `
  --total-timesteps 20000 `
  --initialize-from-stairs `
  simulation\isaac\output\rl\ppo-stairs-v5-10mm-distilled-gated\drobot_stairs_ppo_distilled.zip `
  --output-dir `
  simulation\isaac\output\rl\ppo-stairs-v5-10mm-four-step-full
```

Evaluate the tracked release package deterministically:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v5.yaml `
  --height-stage 10mm `
  --model `
  simulation\isaac\models\ppo-stairs-v5-10mm-four-step\drobot_stairs_ppo_initialized.zip `
  --active-steps 4 `
  --episodes 10 `
  --report `
  simulation\isaac\models\ppo-stairs-v5-10mm-four-step\evaluation_report.json
```

Replay and record the verified stochastic episode. The 80 precursor episodes
are necessary because both policy sampling and contact/reset history affect
the result:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\record_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v5.yaml `
  --height-stage 10mm `
  --model `
  simulation\isaac\models\ppo-stairs-v5-10mm-four-step\drobot_stairs_ppo_initialized.zip `
  --seed 314 `
  --stochastic `
  --policy-seed 314 `
  --skip-episodes 80 `
  --search-success-episodes 1 `
  --active-steps 4 `
  --video reviews\ppo-stairs-v5-10mm-four-step-success.mp4 `
  --thumbnail reviews\ppo-stairs-v5-10mm-four-step-success.png `
  --report simulation\isaac\output\rl\ppo-stairs-v5-replay\recording_report.json
```

## Validation and measured results

Validation performed on 2026-07-31:

- focused stair contract tests: `18 passed, 2 skipped`;
- full repository regression suite: `191 passed, 2 skipped`;
- Ruff and Python compilation on the changed stair scripts: passed;
- `10` and `20 mm` staged world generation and dependency hashing: passed;
- v5 four-step PPO run: `20,480` actual steps, pipeline/model save passed;
- stochastic success collection: `1/81`, with `1,872` demonstration steps;
- success-conditioned actor distillation: pipeline passed, but the distilled
  mean completed `0/10` deterministic episodes;
- tracked source-equivalent model policy-state comparison: maximum absolute
  tensor difference `0.0`;
- tracked release model manifest and algorithm verification: passed;
- tracked release deterministic evaluation: `0/10` successes;
- exact stochastic replay: all 80 precursor episode lengths matched the
  collection report, then seed `394` completed the task in `15.6 s`;
- H.264 recording: `468` frames, `960 x 540`, `30 FPS`, `10,716,982` bytes.

The compact machine-readable record is
`reviews/ppo-stairs-v5-10mm-four-step-results.json`. The model directory also
contains its full deterministic evaluation report and schema-2 manifest.

## Limitations and next work

The `10 mm` steps total only `40 mm`; this does not establish capability on a
normal building stair. The success rate is too low for deployment, and the
policy mean is not usable. The terrain samples are analytic simulator values,
not an onboard depth/camera estimator. Fork tips use simulated contact geometry
without the final physical foot material, wiring, battery sag, bus latency, or
thermal effects. The hardware profile comes from one unloaded wall-mounted
leg, not a loaded four-leg stair trial.

Before hardware use, training needs a robust deterministic success rate,
progressive validation at `20`, `30`, and `40 mm`, perception replacement for
analytic terrain, randomized friction/mass/latency, actuator-current and
temperature limits, and a tethered low-rise physical test with an emergency
stop. Do not deploy this checkpoint to the robot as-is.
