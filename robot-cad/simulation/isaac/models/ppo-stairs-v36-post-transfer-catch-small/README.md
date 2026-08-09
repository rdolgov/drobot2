# V36 post-transfer support catch

This package is a bounded 4,096-step PPO experiment for one narrow part of the
stair sequence: keep the body upright for two seconds immediately after the
front-left-to-rear-right weight transfer. It uses the exact `180 mm` rise,
`250 mm` tread depth, and measured `0.8825985 N m` actuator effort cap.

The policy does not use RGB. Its 95-value input is IMU, joint state, previous
action, contact/load, COM/support, phase, and analytic stair-geometry data. The
external camera is used only for the review recording.

## Contents

- `drobot_stairs_ppo_final.zip`: 183,315-parameter, nine-action support policy
- `drobot_stairs_ppo_final.zip.contract.json`: schema-2 model contract
- `quadruped_stairs_v36_transfer_support_residual.yaml`: exact task config
- `training_report.json`: 4,096-step seed-840 training provenance
- `action_search_report_seed840.json`: bounded 64-candidate initialization search
- `evaluation_report_fresh_seed832.json`: independent 65-second composed run
- `evaluation_report_seed832_80s.json`: extended 80-second composed run
- `evaluation_report_fresh_seed841.json`: fresh-seed precursor failure
- `recording_report.json`: H.264 recording and episode provenance
- `RESULTS.md`: measured result and limitations

The review media are tracked separately at
[`reviews/ppo-stairs-v36-post-transfer-catch-65s.mp4`](../../../../reviews/ppo-stairs-v36-post-transfer-catch-65s.mp4)
and
[`reviews/ppo-stairs-v36-post-transfer-catch-65s.png`](../../../../reviews/ppo-stairs-v36-post-transfer-catch-65s.png).

## Reproduce training

From the repository root in PowerShell:

```powershell
& simulation\isaac\rl\stairs\train_stairs_v36_transfer_support_small.ps1 `
  -OutputDir simulation\isaac\output\rl\ppo-stairs-v36-post-transfer-catch-4096-seed840
```

The wrapper invokes Isaac Sim 6.0.1, Stable-Baselines3 PPO, seed `840`, a
dynamic cached phase snapshot, `4096` control steps, and post-transfer-only
support training. It composes the tracked V10 front-right and V17 front-left
precursor policies while learning a compact nine-joint support residual.

## Reproduce the bounded action search

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_post_transfer_support.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v36_transfer_support_residual.yaml `
  --precursor-model front_right=simulation\isaac\models\ppo-stairs-v10-180mm-25cm-front-right-placement-small\drobot_stairs_ppo_final.zip `
  --precursor-model front_left=simulation\isaac\models\ppo-stairs-v17-single-foot-190mm-small\drobot_stairs_ppo_final.zip `
  --target-leg rear_right `
  --seed 840 `
  --report simulation\isaac\output\rl\ppo-stairs-v36-post-transfer-action-search-seed840.json `
  --headless
```

## Reproduce composed evaluation

Use the normal stair evaluator with V10/V17 as the precursor policies, V35 as
the rear-right swing policy, and this package as the rear-right post-transfer
policy. Pin seed `832`, one episode, the first active tread, and `65` seconds.
The evaluator verifies every adjacent model contract before stepping physics.
The exact resolved paths and hashes are preserved in
`evaluation_report_fresh_seed832.json`.

This package is an evaluation artifact, not a deployable walking controller.
