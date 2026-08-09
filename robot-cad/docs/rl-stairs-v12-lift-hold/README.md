# Front-right 190 mm lift-and-hold curriculum

## Outcome

This experiment isolates the next mixed-height prerequisite on the fixed
`180 mm` rise and `250 mm` tread. A verified front-left policy first places
and loads the tread, the deterministic transfer unloads front-right, and PPO
may correct only the nine support-leg joints around the frozen front-right
base policy.

The `2,048`-step small training run completed with one physical prefix replay,
seven cached post-transfer resets, and zero cache restore failures. Stochastic
cached training episodes raised front-right `222-234 mm`, but they lost one
support contact and eventually failed base clearance. The selected `1,536`
step deterministic checkpoint is therefore evaluation-only, not a successful
or deployable stair policy.

Strict full-sequence seed-`195` evaluation measured:

| Metric | Result |
| --- | ---: |
| Front-left lift and force placement | `207.10 mm`, `25.39 N` retained |
| Transfer support margin | `80.97 mm` |
| Front-right transfer load | `0.00 N` (explicitly unloaded) |
| Front-right lift before termination | `142.08 mm` |
| Maximum support slip | `17.67 mm` (below the `25 mm` measurable gate) |
| Minimum support contact fraction | `1.00` |
| Lateral position at termination | `202.12 mm` |
| Result | **FAIL**: `left_stair_corridor` at the strict `200 mm` gate |

The evidence does not justify more traction or camera work yet. The support
feet did not show measurable slip, RGB was not a policy input, and the fixed
geometry was already known. The next change should directly stabilize the
post-transfer body/reference trajectory before the right-foot apex.

## New training contract

- `quadruped_stairs_v12_front_right_lift_hold.yaml` selects
  `swing_lift_hold` only for front-right; front-left still requires force-backed
  tread contact.
- A lift success requires at least `190 mm`, all three support loads above the
  contact threshold, at least `15 mm` support-triangle margin, upright body,
  and a `0.50 s` hold.
- `_placement_phase_training.py` replays the verified precursor once, captures
  the post-transfer articulation/controller state, and restores that state for
  repeated target-leg PPO episodes.
- The cached snapshot preserves completed-foot progression. This is necessary
  for restored episodes to reward front-right rather than incorrectly
  selecting front-left again.
- `record_stairs_ppo.py` now records the same frozen-base plus bounded-residual
  per-leg policy composition used by deterministic evaluation.

## Reproduction

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v12_front_right_lift_hold.yaml `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v12-front-right-190mm-lift-small `
  --total-timesteps 2048 --seed 195 --device cpu `
  --phase-train-leg front_right `
  --precursor-leg-model front_left=simulation\isaac\models\ppo-stairs-v8-180mm-25cm-single-foot-placement-small\drobot_stairs_ppo_final.zip `
  --phase-base-model simulation\isaac\models\ppo-stairs-v10-180mm-25cm-front-right-placement-small\drobot_stairs_ppo_final.zip `
  --phase-residual-scale 0.25 --phase-residual-support-only
```

The tracked review video is
`reviews/ppo-stairs-v12-front-right-190mm-lift-evaluation.mp4`. It intentionally
shows the strict failure and must not be described as a 190 mm success.
The strict evaluator/recorder invocation adds
`--maximum-lateral-deviation-m 0.20`; the tracked YAML retains the wider
training corridor bound by the checkpoint manifest.
