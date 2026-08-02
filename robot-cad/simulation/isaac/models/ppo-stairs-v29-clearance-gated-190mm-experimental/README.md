# V29 clearance-gated 190 mm front-pair experiment

This package is an **experimental checkpoint, not a verified stair-climbing
policy**. It combines the verified V10 front-right placement, the V29
four-foot COM/support transfer gate, the frozen V17 front-left swing policy,
and a PPO residual limited to the three front-left swing joints.

## Physical contract

- Stair rise: 0.18 m
- Stair tread depth: 0.25 m
- Required measured front-left clearance before forward advance: 0.190 m
- Clearance timeout: 2.0 s at the apex reference
- Applied joint effort cap: 0.8825985 N m
- Policy inputs: IMU, joint state, contact/load state, computed COM/support
  state, prior action, and the known analytic stair profile
- Camera: recording only; RGB pixels are not policy or gate inputs

## Training and selection

- Training seed: 590
- Small-run budget: 4,096 target-leg PPO steps with live V10 precursor replay
- Selected checkpoint: 3,072 steps
- Selected model SHA-256:
  `d5897b9cbfd42f95141beaac4418f31419bf7a9e2d49cadadd8a3977ec5ff048`
- Config SHA-256:
  `0a41d1405b58259642c140121212d8beae91a7f0d39024d9c5db9bff7d0430e0`

The checkpoint passed a three-episode screening batch (3/3, seed 600), but
failed the independent five-episode acceptance batch (1/5, seed 610). The four
failed acceptance episodes ended as upright `swing_clearance_timeout` events
instead of lateral corridor/tip failures. This is useful safety progress, but
it does not meet the 5/5 front-pair promotion gate and must not yet be extended
to rear-leg or full-stair sequencing.

The representative recording is acceptance episode 4: 218.4 mm front-left
lift, 8.3 mm maximum support slip, 10.4 degrees maximum body tilt, and a 0.02 s
clearance-gate hold. The recording report includes the three preceding timeout
episodes so the video is not presented as aggregate success evidence.

## Files

- `drobot_stairs_ppo_3072_steps.zip`: selected PPO residual checkpoint
- `drobot_stairs_ppo_3072_steps.zip.contract.json`: exact model manifest
- `training_report.json`: full 4,096-step training report
- `eval_checkpoint3072_seed600.json`: 3/3 screening report
- `evaluation_report_checkpoint3072_seed610.json`: 1/5 acceptance report
- `recording_report_seed610.json`: exact recorded rollout/search report

## Reproduce the acceptance evaluation

Run from `robot-cad` with Isaac Sim installed at `C:\isaacsim`:

```powershell
& 'C:\isaacsim\python.bat' simulation/isaac/rl/stairs/evaluate_stairs_ppo.py `
  --config simulation/isaac/rl/stairs/quadruped_stairs_v29_preunload_com_gate.yaml `
  --model simulation/isaac/models/ppo-stairs-v29-clearance-gated-190mm-experimental/drobot_stairs_ppo_3072_steps.zip `
  --episodes 5 --seed 610 --device cpu --active-steps 1 `
  --placement-level left-supported-190mm-lift `
  --maximum-lateral-deviation-m 0.20 `
  --leg-model front_right=simulation/isaac/models/ppo-stairs-v10-180mm-25cm-front-right-placement-small/drobot_stairs_ppo_final.zip `
  --leg-model front_left=simulation/isaac/models/ppo-stairs-v29-clearance-gated-190mm-experimental/drobot_stairs_ppo_3072_steps.zip `
  --leg-base-model front_left=simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
  --leg-base-swing-only front_left `
  --leg-residual-scale front_left=0.50 `
  --leg-residual-swing-only front_left
```
