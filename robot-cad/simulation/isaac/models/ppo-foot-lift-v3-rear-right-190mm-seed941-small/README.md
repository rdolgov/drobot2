# Fresh rear-right 190 mm balance policy (seed 941)

This package is the requested simplified prerequisite: lift the rear-right
foot at least `190 mm`, hold it for `0.75 s`, and remain upright on the other
three physical feet. The floating body is not pose-pinned. The policy uses
IMU, joint state, previous action, lift progress, and body state; RGB is used
only to record the review video.

## Result

- Fresh PPO smoke run: `512` steps, seed `941`, `18.014 s`.
- Completed training episode: `196.700 mm` lift, `2.362 deg` maximum tilt,
  `0.75 s` hold, no failure reason.
- Independent deterministic seed `942`: `5/5` successes, `201.006-204.345 mm`
  maximum lift, `2.218 deg` worst tilt, and no failure reason.
- Independent recorded seed `943`: `202.907 mm` lift, `2.083 deg` worst tilt,
  `1.271 mm` minimum measured support-triangle margin, and `0.75 s` hold.
- Applied real-test effort cap: `0.8825985 N m` per joint.
- Model SHA-256:
  `7f3ccb0a159140de47eb99d8ad71c0eeabf3692a6dd712e36c44206c4e9d279c`.

One of the five evaluation episodes briefly measured a `-0.358 mm` geometric
support-triangle margin while remaining dynamically upright and completing all
no-fall gates. The recorded episode retained a positive margin.

## Exact commands

From `robot-cad`:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_quadruped_foot_lift_rl_contract.py -q `
  --basetemp .pytest_tmp_v47_foot_lift

& C:\isaacsim\python.bat simulation\isaac\rl\foot_lift\train_foot_lift_ppo.py `
  --config simulation\isaac\rl\foot_lift\quadruped_foot_lift_v3_rear_right_balance.yaml `
  --output-dir simulation\isaac\output\rl\ppo-foot-lift-v3-rear-right-190mm-fresh-seed941 `
  --total-timesteps 512 --seed 941 --device cpu `
  --smoke-test --final-stage-only

& C:\isaacsim\python.bat simulation\isaac\rl\foot_lift\evaluate_foot_lift_ppo.py `
  --config simulation\isaac\rl\foot_lift\quadruped_foot_lift_v3_rear_right_balance.yaml `
  --model simulation\isaac\output\rl\ppo-foot-lift-v3-rear-right-190mm-fresh-seed941\drobot_foot_lift_ppo_final.zip `
  --episodes 5 --seed 942 --device cpu `
  --report simulation\isaac\output\rl\ppo-foot-lift-v3-rear-right-190mm-fresh-seed941\evaluation_report_seed942.json

& C:\isaacsim\python.bat simulation\isaac\rl\foot_lift\record_foot_lift_ppo.py `
  --config simulation\isaac\rl\foot_lift\quadruped_foot_lift_v3_rear_right_balance.yaml `
  --model simulation\isaac\output\rl\ppo-foot-lift-v3-rear-right-190mm-fresh-seed941\drobot_foot_lift_ppo_final.zip `
  --seed 943 --device cpu --fps 30 --width 960 --height 540 `
  --video reviews\ppo-foot-lift-v3-rear-right-190mm-fresh-seed943.mp4 `
  --thumbnail reviews\ppo-foot-lift-v3-rear-right-190mm-fresh-seed943.png `
  --report simulation\isaac\output\rl\ppo-foot-lift-v3-rear-right-190mm-fresh-seed941\recording_report_seed943.json
```

## Scope

This is a short feasibility run around a 200 mm analytic raise-forward IK
reference plus bounded PPO residuals; it is not a converged from-scratch gait.
There is no stair in this scene. It proves modeled rear-foot clearance and
simple three-foot balance, not the mixed-height support transfer, tread
landing, traction robustness, sim-to-real behavior, or a full stair climb.
