# V45 results and reproduction

## Result

The seed-875 run completed exactly `4,096` PPO steps in `135.744 s`, using 20
terminated episodes plus one final reset from the same cached physical
snapshot. It completed `0` rear-right-to-rear-left transfers. Maximum
rear-left physical lift was only `15.136 mm`; minimum support margin was
`-102.399 mm`, maximum support slip was `88.638 mm`, maximum COM-target error
was `151.523 mm`, and minimum upright cosine was `0.937176`.

The deterministic seed-876 replay also failed: after 215 control steps
(`3.583 s`) it terminated with `body_tipped`. COM-target error changed from
`128.454 mm` to `153.372 mm`, best support margin was still `-44.601 mm`, and
maximum body tilt was `20.790 deg`. The external recording contains 107 frames
at 30 fps and 960 x 540. Camera pixels were not policy inputs.

The controller audit tested all 27 constant combinations of the three loaded
support hip-abduction actions. None completed the transfer. A rear-right
foothold-offset search found that `5 mm` was the only nonzero tested offset
that preserved the V44 landing; offsets of `10-30 mm` tipped before contact.
The accepted `5 mm` reference retained `217.990 mm` rear-right lift and all
`45/45` contact-hold frames but did not materially widen the physical foothold.

## Reproduce

From the repository root, reproduce and save the exact transfer boundary:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\search_rear_left_transfer_com.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v45_rear_left_transfer.yaml `
  --seed 870 --forward-deltas-m 0.000 --lateral-deltas-m 0.000 `
  --pitch-profiles 0.080:0.025 --front-only-values false `
  --transfer-durations-seconds 8.0 `
  --save-transfer-snapshot simulation\isaac\models\ppo-stairs-v45-rear-left-dynamic-transfer-4096\phase_snapshot_seed870.json `
  --report simulation\isaac\models\ppo-stairs-v45-rear-left-dynamic-transfer-4096\zero_action_transfer_probe_seed870.json
```

Train the bounded policy:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v45_rear_left_transfer.yaml `
  --output-dir simulation\isaac\models\ppo-stairs-v45-rear-left-dynamic-transfer-4096 `
  --seed 875 --total-timesteps 4096 `
  --phase-train-leg rear_left --phase-train-transfer `
  --phase-snapshot simulation\isaac\models\ppo-stairs-v45-rear-left-dynamic-transfer-4096\phase_snapshot_seed870.json `
  --fixed-placement-level left-center-tread-load `
  --ppo-learning-rate 0.0001 --ppo-initial-log-std -0.3 `
  --ppo-entropy-coefficient 0.001 --device cuda
```

Replay it deterministically and record the external camera:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\search_rear_left_transfer_support_actions.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v45_rear_left_transfer.yaml `
  --phase-snapshot simulation\isaac\models\ppo-stairs-v45-rear-left-dynamic-transfer-4096\phase_snapshot_seed870.json `
  --policy-model simulation\isaac\models\ppo-stairs-v45-rear-left-dynamic-transfer-4096\drobot_stairs_ppo_final.zip `
  --seed 876 --maximum-seconds 5.0 `
  --record-video reviews\ppo-stairs-v45-rear-left-dynamic-transfer-eval-seed876.mp4 `
  --record-thumbnail reviews\ppo-stairs-v45-rear-left-dynamic-transfer-eval-seed876.png `
  --record-fps 30 --record-width 960 --record-height 540 `
  --report simulation\isaac\models\ppo-stairs-v45-rear-left-dynamic-transfer-4096\evaluation_report_seed876.json
```

Re-run the deterministic 27-action authority audit:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\search_rear_left_transfer_support_actions.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v45_rear_left_transfer.yaml `
  --phase-snapshot simulation\isaac\models\ppo-stairs-v45-rear-left-dynamic-transfer-4096\phase_snapshot_seed870.json `
  --seed 874 --maximum-seconds 5.0 `
  --report simulation\isaac\models\ppo-stairs-v45-rear-left-dynamic-transfer-4096\constant_support_action_search_seed874.json
```

## Limitation and next target

This snapshot begins only after a successful phase-local V44 landing, so it
does not prove reset-to-boundary prefix reliability. The current failure is
support geometry and COM regulation, not a missing view of a known fixed stair.
The next experiment should insert an explicit post-landing rear-right sidestep
and settle stage, verify a positive support polygon around the desired COM,
then train rear-left unloading/lift. More vision or longer end-to-end PPO is
not the next justified change.
