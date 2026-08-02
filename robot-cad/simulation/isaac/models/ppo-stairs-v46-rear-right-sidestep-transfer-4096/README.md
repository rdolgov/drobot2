# V46 rear-right sidestep and rear-left transfer diagnostic

V46 starts from the accepted first-tread rear-right landing, lifts and
repositions that foot outward on the same `250 mm` tread, then asks PPO to
perform the next rear-left COM transfer. It retains the exact `180 mm` rise,
`250 mm` tread, and `0.8825985 N m` real-test effort cap. RGB remains
recording-only.

## Measured result

The accepted seed-937 sidestep commanded `10 mm` outward and `15 mm` forward.
It produced `9.215 mm` physical outward displacement, `7.033 mm` measured
re-clearance above the already elevated tread pose, `65.201 mm` minimum
replacement support margin, `2.301 N` final tread load, `15.999 mm` maximum
support slip, and `11.461 deg` maximum tilt. The exact next-phase snapshot is
included in this package.

The following transfer remained infeasible with this controller geometry:

- zero action tipped after `87` steps;
- the `4,096`-step seed-939 PPO run completed `0` transfers and reached only
  `13.425 mm` rear-left motion;
- deterministic seed-940 replay tipped after `93` steps;
- COM-target error increased from `126.584` to `133.259 mm`;
- final support margin was `-101.073 mm`; maximum tilt was `20.144 deg`.

The model is retained as rejected evidence, not a climbing policy. Its
SHA-256 is
`9b8cf6dec9f2da8f43dab9fb65f72b4a32999340d43356cf748c3a8498164977`.
The accepted sidestep report SHA-256 is
`f1d3622add4f020caaec7c26fea75e1eb20cf1c4c67e91aa7c87c1193cb8c67d`;
the exact phase snapshot SHA-256 is
`7545b2c0370e6c58f753487409b3f9a27c1876dbcf4e114b33872823623d1be7`.

## Reproduction

The accepted sidestep search:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_rear_right_post_landing_sidestep.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v46_rear_right_sidestep.yaml `
  --phase-snapshot simulation\isaac\models\ppo-stairs-v45-rear-left-dynamic-transfer-4096\phase_snapshot_seed870.json `
  --seed 937 `
  --report simulation\isaac\output\rl\rear-right-post-landing-sidestep-v46-seed937.json `
  --output-phase-snapshot simulation\isaac\output\rl\rear-left-transfer-snapshot-v46-seed937.json
```

The bounded PPO run:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v46_rear_right_sidestep.yaml `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v46-rear-left-transfer-4096 `
  --seed 939 --total-timesteps 4096 `
  --phase-train-leg rear_left --phase-train-transfer `
  --phase-snapshot simulation\isaac\output\rl\rear-left-transfer-snapshot-v46-seed937.json `
  --fixed-placement-level left-center-tread-load `
  --ppo-learning-rate 0.0001 --ppo-initial-log-std -0.3 `
  --ppo-entropy-coefficient 0.001 --device cuda
```

The recorded deterministic failure:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_rear_left_transfer_support_actions.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v46_rear_right_sidestep.yaml `
  --phase-snapshot simulation\isaac\output\rl\rear-left-transfer-snapshot-v46-seed937.json `
  --policy-model simulation\isaac\output\rl\ppo-stairs-v46-rear-left-transfer-4096\drobot_stairs_ppo_final.zip `
  --seed 940 --maximum-seconds 5.0 `
  --record-video reviews\ppo-stairs-v46-rear-right-sidestep-transfer-eval-seed940.mp4 `
  --record-thumbnail reviews\ppo-stairs-v46-rear-right-sidestep-transfer-eval-seed940.png `
  --record-fps 30 --record-width 960 --record-height 540 `
  --report simulation\isaac\output\rl\ppo-stairs-v46-rear-left-transfer-4096\evaluation_report_seed940.json
```

## Next gate

The sidestep adds lateral foothold width but does not reduce the roughly
`98 mm` lateral COM demand enough. The next controller should first settle all
four loaded feet and shift the body/COM into a safe preload corridor before
unloading rear-left. More RGB vision is not justified for this known fixed
geometry; traction remains a hardware-calibration variable, but the immediate
simulation failure is COM/reference sequencing under the measured effort cap.
