# V27 front-pair 190 mm support-abduction experiment

This package tests the simplified prerequisite requested before extending to a
full climb: replay the verified front-right placement, transfer onto three
support feet, then raise the front-left foot at least `190 mm` beside the exact
`180 mm` rise, `250 mm` tread staircase without falling.

The 1,024-step PPO run starts from V24, freezes the verified V17 swing policy,
and gives the learned residual authority over only the three support-leg
hip-abduction joints. The residual scale is `0.15`; RGB and range pixels are
not policy inputs. The robot configuration retains the real-test `+/-120 deg`
knee range and `0.8825985 N m` effort cap.

## Result

The independent deterministic evaluation passed `2/5` fresh randomized
episodes. This is a useful successful-motion sample, but it is **not** a robust
policy and is not promoted as a solved front-pair transfer or full stair climb.

| Episode | Result | Front-left lift | Support slip | Peak tilt |
| ---: | :---: | ---: | ---: | ---: |
| 1 | pass | `222.1 mm` | `15.3 mm` | `10.5 deg` |
| 2 | fail | `181.5 mm` | `20.3 mm` | `21.0 deg` |
| 3 | fail | `179.2 mm` | `28.8 mm` | `20.0 deg` |
| 4 | pass | `217.4 mm` | `8.9 mm` | `10.9 deg` |
| 5 | fail | `168.2 mm` | `28.6 mm` | `18.4 deg` |

The first successful evaluation rollout is recorded at
`reviews/ppo-stairs-v27-support-abduction-1024-success.mp4`. It is a
`21.4 s`, `960 x 540`, `30 fps` H.264 video with an exact trajectory and
thumbnail stored beside it.

## Load-sharing ablation

A deterministic stance-height load-sharing controller was also tested and
rejected. On the same seed-450 reset sequence, the frozen no-load-sharing
control passed `2/3`, while a smoothed controller failed `3/3` even though its
largest correction was only `2.72 mm`. The unsmoothed controller reached at
least `190.7 mm` in `5/5` seeds but tipped in every late hold. The diagnostic
reports are in `ablation/`.

This A/B evidence points to lateral support geometry and transfer-state
variation, not missing foot reach. The next experiment should widen or
reposition the rear support feet and regulate the lateral COM target before
adding camera input or increasing friction again.

## Reproduce evaluation

Run from `robot-cad` with Isaac Sim 6.0.1:

```powershell
& 'C:\isaacsim\python.bat' `
  simulation/isaac/rl/stairs/evaluate_stairs_ppo.py `
  --config simulation/isaac/rl/stairs/quadruped_stairs_v24_front_pair_conservative_support.yaml `
  --model simulation/isaac/models/ppo-stairs-v27-support-abduction-190mm-small/drobot_stairs_ppo_final.zip `
  --episodes 5 --seed 470 --device cpu --active-steps 1 `
  --placement-level left-supported-190mm-lift `
  --maximum-lateral-deviation-m 0.20 `
  --leg-model front_right=simulation/isaac/models/ppo-stairs-v10-180mm-25cm-front-right-placement-small/drobot_stairs_ppo_final.zip `
  --leg-model front_left=simulation/isaac/models/ppo-stairs-v27-support-abduction-190mm-small/drobot_stairs_ppo_final.zip `
  --leg-base-model front_left=simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
  --leg-base-swing-only front_left `
  --leg-residual-support-abduction-only front_left `
  --leg-residual-scale front_left=0.15
```
