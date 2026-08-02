# Rear-right first-tread landing search (V39-V41)

## Scope and immutable inputs

This bounded follow-up starts from the accepted V38 mixed-height state: the
front-right and front-left feet are force-backed on the first tread and the
rear-right foot has completed the positive-margin transfer. It tests only the
rear-right advance/lower/contact phase. It does not move rear-left or claim a
completed stair.

- Stair rise: exactly `0.180 m`
- Stair tread depth: exactly `0.250 m`
- Applied joint-effort cap: `0.8825985 N m` from the real leg test
- Rear-right clearance gate: measured foot-tip rise `>= 0.190 m`
- Transfer/placement margin gate: `>= 0.015 m`
- Upright gate: body tilt `<= 12 deg`
- Policy input: IMU/proprioception, joint state, contact/load, composite COM,
  support state, phase, previous action, and analytic stair geometry
- Camera: recording only; no RGB pixels enter training or inference

The cached-state search composes frozen V17 swing, `0.5 * V35` compact swing,
and `1.0 * V38` compact support policies. Each candidate restores the same
rear-right phase snapshot.

## Small training runs

V39 trains only a compact nine-action support residual around the accepted
swing composition:

```powershell
& simulation\isaac\rl\stairs\train_stairs_v39_rear_right_landing_small.ps1 `
  -OutputDir simulation\isaac\output\rl\ppo-stairs-v39-rear-right-landing-512-seed851 `
  -Seed 851
```

The run completed `512` PPO steps in `76.27 s`; model SHA-256 is
`1174ed3b5ce6991f4a48f878d32031354bd6f93d02e2ec6a9da3331742e4db9f`.
Independent seed-848 replay preserved the V38 handoff and reached
`218.095 mm`, but rear-right struck the tread edge at `351.313 N` and the body
tipped to `19.9904 deg`. V39 is rejected.

V40 freezes V38 support and trains only a compact three-action rear-right swing
residual:

```powershell
& simulation\isaac\rl\stairs\train_stairs_v40_rear_right_swing_landing_small.ps1 `
  -OutputDir simulation\isaac\output\rl\ppo-stairs-v40-rear-right-swing-landing-1024-seed856 `
  -Seed 856
```

The run completed `1,024` PPO steps in `89.25 s`; model SHA-256 is
`74369c0f9a465c5e1394836632acbcbc7b0a92130e669dc5c76cee5967227c08`.
Cached seed-857 evaluation stayed upright at `10.7597 deg`, held
`37.939 mm` support margin, and raised `223.978 mm`, but the foot stopped at
world X `0.318371 m` with zero tread load. V40 is rejected.

## Bounded geometry and controller search

The deterministic harness is:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_rear_right_landing.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v41_staged_rear_right_landing.yaml `
  --seed 849 --candidate-start 0 --candidate-limit 16 `
  --report simulation\isaac\output\rl\rear-right-landing-search.json
```

Run additional candidate slices by changing `--candidate-start` and
`--candidate-limit`. The search keeps the physical gate, positive margin,
effort cap, and camera-blind contract unchanged.

Measured negative evidence:

- Direct `0.380-0.400 m` swing references reached only about
  `0.543-0.545 m` world X; the tread begins at `0.550 m`.
- Staged world-anchor release plus a `70 mm` lift-forward waypoint reached
  `0.557554 m`, but only `7.554 mm` of the `12.5 mm` foot radius cleared the
  riser. Contact peaked at `330.772 N` and tilt reached `20.6903 deg`.
- Slower lowering and higher touchdown targets did not help because the corner
  collision occurred during forward advance.
- Stronger IMU pitch correction reduced one run to `17.0625 deg`, but pulled
  reach back to `0.542627 m`; the strongest setting also violated the strict
  support-margin gate.
- A `+10 mm` post-clearance body command produced up to `+20.103 mm` base and
  `+34.102 mm` COM motion, proving the frame sign. When sequenced before swing,
  the robot stayed at `10.3434 deg` with `39.244 mm` margin and raised
  `223.833 mm`, but the torque-capped rear hip tracked only about `0.68 rad` of
  a `1.83 rad` reference and the foot did not advance onto the tread.

Local reports are written below `simulation/isaac/output/rl/` and intentionally
remain ignored build artifacts. Key reports include
`ppo-stairs-v39-rear-right-landing-search-seed849.json`,
`ppo-stairs-v41-lift-forward-search-seed849.json`,
`ppo-stairs-v41-pitch-feedback-search-seed849.json`, and
`ppo-stairs-v41-split-body-shift-8s-replay-seed849.json`.

## Conclusion and next gate

The accepted V38 policy already proves the requested simplified skill:
rear-right clears `0.190 m` and holds without falling. V39-V41 do not prove
landing. The dominant first-tread landing limitation is rear-leg motion under
three-foot support at the measured effort cap, followed by real/sim traction
calibration. Fixed-stair vision is not the present bottleneck.

Before longer PPO training, reduce rear-hip load (mass distribution), increase
verified actuator/gear torque without changing the safety contract, or author a
hardware-backed support posture with more hip authority. Then train a
phase-local body-shift-plus-swing policy and require force-backed contact on the
same `250 mm` tread before mirroring rear-left.
