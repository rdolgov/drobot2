# V17 isolated 190 mm front-foot balance

This package is the deliberately simplified prerequisite for stair climbing.
The floating quadruped starts with four feet on the floor beside the exact
`180 mm` rise, `250 mm` tread staircase. It unloads the front-left foot and
must hold at least `190 mm` measured lift for `0.5 s` without losing the other
three contacts, slipping, or tipping.

The policy was fine-tuned for 4,096 additional PPO steps from the verified V15
checkpoint with the measured real-test joint directions, `+/-120 deg` knee
range, and `0.8825985 N m` effort cap. All 12 recent training episodes passed;
lift ranged from `205.86` to `208.55 mm`, maximum support slip was `3.33 mm`,
and maximum body tilt was `2.42 deg`.

The independent deterministic evaluation passed `5/5` episodes:

| Metric | Result |
| --- | ---: |
| measured front-left lift | `205.55-207.75 mm` |
| required lift hold | `0.50 s` in every episode |
| maximum body tilt | `2.33 deg` |
| maximum support-foot slip | `3.27 mm` |
| minimum final support margin | `43.31 mm` |

The recording is
`reviews/ppo-stairs-v17-single-foot-190mm-lift-success.mp4`; its exact seeded
trajectory and thumbnail are stored beside it. The recorded seed reached
`207.40 mm` with `2.11 deg` tilt and `3.18 mm` support slip.

This policy is camera-blind. It uses the IMU, joint state, previous action,
placement-reference state, and the simulation's known analytic stair profile.
The external camera is used only to render the review video.

This result proves isolated single-foot lift and balance. It does **not** prove
the second front foot can repeat the motion after the opposite front foot is
already loaded on the stair, and it is not a full stair climb.

## Reproduce training

Run from `robot-cad` with Isaac Sim 6.0.1:

```powershell
& 'C:\isaacsim\python.bat' `
  simulation/isaac/rl/stairs/train_stairs_v17_single_foot_190mm_ppo.py
```

The wrapper defaults to the strict `190 mm` level, a 4,096-step fine-tune,
seed `271`, CPU PPO, and the tracked V15 initializer. Every option can be
overridden with the generic trainer flags.

## Reproduce evaluation

```powershell
& 'C:\isaacsim\python.bat' `
  simulation/isaac/rl/stairs/evaluate_stairs_ppo.py `
  --config simulation/isaac/rl/stairs/quadruped_stairs_v15_front_left_stabilized_lift.yaml `
  --model simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
  --episodes 5 `
  --seed 272 `
  --device cpu `
  --active-steps 1 `
  --placement-level front-left-stabilized-190mm-lift-hold `
  --maximum-lateral-deviation-m 0.20
```
