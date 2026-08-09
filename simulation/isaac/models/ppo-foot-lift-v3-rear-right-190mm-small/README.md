# Rear-right 190 mm balance policy

This package records the bounded V3 prerequisite experiment: raise the
rear-right foot at least `190 mm` for `0.75 s` while balancing on the other
three physical feet. The floating base is not pose-pinned. The policy uses IMU,
joint position/velocity, prior action, lift target/progress, and base drift; RGB
is recording-only. The exercised hardware profile caps every joint at
`0.8825985 N m` and retains the real-test joint limits.

## Result

- Training: fresh zero-mean PPO, `512` steps, seed `13190`; its completed
  episode reached `197.35 mm`, held `0.75 s`, and stayed below `2.34 deg` tilt.
- Independent evaluation: seed `13291`, `5/5` strict successes,
  `199.85-203.66 mm` maximum lift, `2.12 deg` worst tilt, and no failure reason.
- Recording: `200.81 mm` lift, `1.73 deg` worst tilt, `175` frames at
  `960 x 540`, 30 FPS.
- Model SHA-256:
  `376d1e02e09ea9d5d25eacdc38bf46432c8ed9bf6b3efc65cccdbec6214e636b`.

## Exact reproduction

From the repository root:

```powershell
& simulation/isaac/rl/foot_lift/train_foot_lift_v3_rear_right_190mm_small.ps1

& C:\isaacsim\python.bat simulation/isaac/rl/foot_lift/evaluate_foot_lift_ppo.py `
  --config simulation/isaac/rl/foot_lift/quadruped_foot_lift_v3_rear_right_balance.yaml `
  --model simulation/isaac/output/rl/ppo-foot-lift-v3-rear-right-190mm-small/drobot_foot_lift_ppo_final.zip `
  --episodes 5 --seed 13291 --device cpu `
  --report simulation/isaac/output/rl/ppo-foot-lift-v3-rear-right-190mm-small/evaluation_report.json

& C:\isaacsim\python.bat simulation/isaac/rl/foot_lift/record_foot_lift_ppo.py `
  --config simulation/isaac/rl/foot_lift/quadruped_foot_lift_v3_rear_right_balance.yaml `
  --model simulation/isaac/output/rl/ppo-foot-lift-v3-rear-right-190mm-small/drobot_foot_lift_ppo_final.zip `
  --seed 13291 --device cpu --fps 30 --width 960 --height 540 `
  --video reviews/ppo-foot-lift-v3-rear-right-190mm-evaluation.mp4 `
  --thumbnail reviews/ppo-foot-lift-v3-rear-right-190mm-evaluation.png `
  --report simulation/isaac/output/rl/ppo-foot-lift-v3-rear-right-190mm-small/recording_report.json
```

## Scope and limitations

This run intentionally contains no stair. It isolates the user's kinematic and
balance question before returning to the exact `180 mm` rise / `250 mm` tread
stair task. It does not prove the front-pair-to-rear-right mixed-height transfer,
tread contact, lowering, traction robustness, sim-to-real performance, or a
complete climb. The current distal fork-tip contacts also remain approximations
for the unmodeled printed feet.
