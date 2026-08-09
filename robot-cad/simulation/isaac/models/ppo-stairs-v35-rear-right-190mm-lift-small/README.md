# V35 rear-right 190 mm lift milestone

This package is a deliberately small 512-step PPO milestone for the third leg
in the first-tread sequence. It proves that, after the two front feet are on the
stair and the center-of-mass transfer has unloaded rear-right, a compact
three-output residual policy can raise that foot past the measured 190 mm
clearance gate without a fall.

It is **not** a full stair-climb result. Rear-right has not yet achieved the
force-backed tread landing, rear-left has not moved, and the reported stair
success rate is therefore correctly zero.

## Physical and sensing contract

- Stair tread depth: exactly `0.250 m`
- Stair rise: `0.180 m`
- Clearance gate: measured rear-right foot lift `>= 0.190 m`
- Applied joint-effort cap: `0.8825985 N m`, from the real one-leg test
- Policy action: three rear-right swing-joint residuals
- Policy observations: IMU/proprioception, joint state, force/contact loads,
  computed COM/support state, previous action, and analytic stair geometry
- Camera: external recording only; no RGB pixels enter policy inference

## Result

- Training budget: 512 PPO steps, seed 832
- Maximum measured rear-right lift during training: `0.192259 m`
- Independent model-contract verification: PASS
- Independent PPO-algorithm verification: PASS
- Independent evaluation: clearance gate released, no clearance timeout, no
  failure reasons through the 65-second episode
- Maximum body tilt in independent evaluation: `17.091 deg`
- Full climb / rear-right tread landing: not yet achieved

The evaluation video is
[ppo-stairs-v35-rear-right-190mm-lift.mp4](../../../../reviews/ppo-stairs-v35-rear-right-190mm-lift.mp4).

## Files

- `drobot_stairs_ppo_final.zip` — compact rear-right PPO policy
- `drobot_stairs_ppo_final.zip.contract.json` — model/environment/algorithm
  integrity contract
- `initial_compact_lift_512_steps.zip` and its contract — verified compact
  190 mm lift initialization used by the small training script
- `quadruped_stairs_v35_full_first_tread.yaml` — exact task configuration
- `training_report.json` — 512-step phase-training evidence
- `evaluation_report_fresh_seed832.json` — independent deterministic evaluation
- `recording_report.json` — deterministic external-camera recording evidence

## Reproduce evaluation

Run from `robot-cad`:

```powershell
& 'C:\isaacsim\python.bat' simulation/isaac/rl/stairs/evaluate_stairs_ppo.py `
  --config simulation/isaac/models/ppo-stairs-v35-rear-right-190mm-lift-small/quadruped_stairs_v35_full_first_tread.yaml `
  --model simulation/isaac/models/ppo-stairs-v35-rear-right-190mm-lift-small/drobot_stairs_ppo_final.zip `
  --episodes 1 --seed 832 --device cpu --active-steps 1 `
  --placement-level left-center-tread-load `
  --maximum-lateral-deviation-m 0.30 `
  --leg-model front_right=simulation/isaac/models/ppo-stairs-v10-180mm-25cm-front-right-placement-small/drobot_stairs_ppo_final.zip `
  --leg-model front_left=simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
  --leg-model rear_right=simulation/isaac/models/ppo-stairs-v35-rear-right-190mm-lift-small/drobot_stairs_ppo_final.zip `
  --leg-base-model rear_right=simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
  --leg-base-swing-only rear_right `
  --leg-residual-scale rear_right=0.5 `
  --leg-residual-swing-only rear_right `
  --leg-compact-action rear_right `
  --zero-action-leg rear_left
```

## Integrity

- Model SHA-256: `21bc20d289f31031d037548de7ae0554e534eac0d0133d0379b52f4d5d601270`
- Config SHA-256: `5de61fe07f450fd97c20d6b4cf08a417b15eb0ce4f37f4c8c59a0f05c1ec6806`
- Video SHA-256: `9f223bef5ff34cf89a9b05cb89f812516ac17f9a268137c6e5ac19e6ed3c2fcf`
