# V2 stair-policy architecture

## Observation

The actor and critic receive 60 `float32` values:

1. the existing 48 command, IMU, joint-state, and previous-action values;
2. eight terrain-height deltas sampled from `-0.10` through `+0.60 m`;
3. normalized forward distance to the current curriculum goal;
4. lateral world-Y offset normalized by the staircase half-width;
5. sine of heading error from world `+X`;
6. cosine of heading error from world `+X`.

The last three navigation values make accumulated sideways drift and yaw
observable. V1 penalized lateral position without exposing it to the policy,
which made reliable closed-loop correction impossible.

The PPO network remains separate actor and critic MLPs:

```text
60 observations
  +-- actor:  60 -> 256 ELU -> 256 ELU -> 12 action means
  `-- critic: 60 -> 256 ELU -> 256 ELU -> scalar value
```

The model contains 166,169 trainable parameters.

## Reward

The critical climbing terms are:

- `60 x delta-world-X` for forward progress;
- `150 x delta-base-Z` for actual physical elevation;
- `40 x delta-analytic-terrain-height` for entering a higher tread region;
- `400` after holding the curriculum goal for 0.5 seconds.

Actual base-height gain is signed. Rising is rewarded and losing the gained
height is penalized, which avoids rewarding vertical oscillation on flat
ground as if it were stair ascent.

Path terms include:

- `-8 x world-Y^2`;
- `-0.50 x body-lateral-velocity^2`;
- `-0.60 x (1 - cos(heading-error))`;
- failure outside `|world Y| <= 0.30 m`.

The remaining upright, clearance, rate, action, and joint-speed terms are
defined in the v2 YAML and reported separately by the environment.

## Mastery curriculum

All four physical steps remain present. Training begins with the goal on step
one. A level advances only after at least 20 episodes exist in the recent
40-episode window and success rate is at least 70%. Outcomes are cleared after
advancement so each new level must demonstrate its own mastery.

Unlike the v1 timestep schedule, training duration alone cannot move the goal
to step two, three, or four.

## Progress watchdog

The trainer writes `progress_watchdog.json` every 10,000 steps. At 100,000
steps it checks:

- at least 40 completed episodes;
- at least three episodes with `highest_step_reached >= 1` and actual base
  elevation of at least `0.02 m`;
- those physically elevated reaches are at least 2% of completed episodes.

If any condition fails, the callback stops PPO, saves
`drobot_stairs_ppo_aborted.zip` plus its manifest, writes
`training_report.json` with status `ABORTED_NO_PROGRESS`, and exits with code
3. After passing the first gate, 300,000 steps without a new stair level or a
successful episode triggers the stagnation abort.

The same metrics are written to TensorBoard under `stair/`.

## Limitations

Terrain samples still come from known simulator geometry rather than camera or
depth perception. The task uses one fixed staircase and one Isaac environment.
Passing the progress gate proves early simulated stair interaction only; it
does not prove convergence, generalization, or safe hardware deployment.

