# Front-left 190 mm stabilized lift

This package is a deliberately simplified stair prerequisite. The policy
starts with all four feet on the floor beside the exact 180 mm-rise,
250 mm-depth stair, unloads the front-left foot, raises it at least 190 mm,
and holds the lift for 0.5 seconds without losing three-foot support.

It does not claim a stair placement or a complete climb. The separate v14
right-foot-place/left-foot-lift experiment proved that the leg can exceed
190 mm after transfer, but the body then lost clearance. That isolates the
next problem as load transfer and support control, not leg range.

## Verified result

- PPO training: 8,192 steps, 28/28 successful completed episodes.
- Final 190 mm mastery gate: 10/10 recent successes, 205.91-208.19 mm lift.
- Strict deterministic evaluation: 5/5 successes, 205.37-207.10 mm lift.
- Evaluation hold: 0.5 seconds in every episode.
- Evaluation support contact fraction: 1.0 in every episode.
- Evaluation maximum support-foot motion: 3.37 mm.
- Evaluation maximum body tilt: 2.34 degrees.
- Evaluation failures: none.
- Custom foot-friction override: disabled.
- Terrain input: analytic stair height profile; RGB camera is not a policy input.

The review video is
`reviews/ppo-stairs-v15-front-left-190mm-lift-success.mp4`.

## Reproduce the strict evaluation

Run from the repository root with Isaac Sim 6.0.1:

```powershell
& 'C:\isaacsim\python.bat' `
  simulation/isaac/rl/stairs/evaluate_stairs_ppo.py `
  --config simulation/isaac/rl/stairs/quadruped_stairs_v15_front_left_stabilized_lift.yaml `
  --model simulation/isaac/models/ppo-stairs-v15-front-left-190mm-lift-small/drobot_stairs_ppo_final.zip `
  --episodes 5 `
  --seed 196 `
  --device cpu `
  --active-steps 1 `
  --maximum-lateral-deviation-m 0.20
```
