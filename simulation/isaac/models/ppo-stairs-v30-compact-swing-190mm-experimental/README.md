# V30 compact front-left swing policy (experimental)

This package is the first stair-placement PPO in this repository whose policy
has only the three outputs it can physically use: front-left hip abduction,
hip flexion, and knee. The phase wrapper expands those three values onto the
robot's 12-joint command before applying a bounded residual to the frozen V17
front-left base policy.

The simulation contract is unchanged from V29:

- stair tread depth: 0.25 m
- stair rise: 0.18 m
- measured front-left clearance gate: 0.19 m
- real-test joint effort cap: 0.8825985 N m
- verified V10 front-right placement and four-foot COM/support transfer prefix
- policy input: IMU/proprioception, joint state, contact/load and computed
  COM/support state, prior action, and the known analytic stair profile
- RGB camera: recording only, not a policy input

## Result

The selected 3,072-step checkpoint scored 2/3 on checkpoint screening (seed
660) and 3/5 on the independent acceptance run (seed 670). The three accepted
episodes reached 217.9--220.5 mm front-left lift. The two failures reached only
162.8 and 169.2 mm, then stopped upright at the two-second clearance timeout
instead of advancing or tipping.

This improves the independent V29 result from 1/5 to 3/5, but it is not robust
enough to promote. Rear-leg training remains blocked until the front pair
passes 5/5 on a fresh seed.

The representative acceptance recording is
`reviews/ppo-stairs-v30-compact-swing-190mm-experimental.mp4`. It is the first
episode of the independent seed-670 sequence: 219.4 mm lift, 14.1 mm maximum
support slip, and 11.0 degrees maximum body tilt.

## Package contents

- `drobot_stairs_ppo_3072_steps.zip`: selected three-output PPO residual
- `drobot_stairs_ppo_3072_steps.zip.contract.json`: exact model/world/config
  manifest (SHA-256
  `3b6ed33cee4904d6264ed634f554eb0efee6eee9452cc910890b12df9d8ecc7e`)
- `training_report.json`: full 4,096-step live-prefix training report
- `eval_checkpoint*_seed660.json`: matched checkpoint screens
- `evaluation_report_checkpoint3072_seed670.json`: independent 3/5 acceptance
- `recording_report_seed670.json`: exact recorded rollout and video metadata

## Reproduce the acceptance evaluation

Run from the repository root with Isaac Sim installed at `C:\isaacsim`:

```powershell
& 'C:\isaacsim\python.bat' simulation/isaac/rl/stairs/evaluate_stairs_ppo.py `
  --config simulation/isaac/rl/stairs/quadruped_stairs_v29_preunload_com_gate.yaml `
  --model simulation/isaac/models/ppo-stairs-v29-clearance-gated-190mm-experimental/drobot_stairs_ppo_3072_steps.zip `
  --episodes 5 --seed 670 --device cpu --active-steps 1 `
  --placement-level left-supported-190mm-lift `
  --maximum-lateral-deviation-m 0.20 `
  --leg-model front_right=simulation/isaac/models/ppo-stairs-v10-180mm-25cm-front-right-placement-small/drobot_stairs_ppo_final.zip `
  --leg-model front_left=simulation/isaac/models/ppo-stairs-v30-compact-swing-190mm-experimental/drobot_stairs_ppo_3072_steps.zip `
  --leg-base-model front_left=simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
  --leg-base-swing-only front_left `
  --leg-residual-scale front_left=0.50 `
  --leg-residual-swing-only front_left `
  --leg-compact-action front_left
```
