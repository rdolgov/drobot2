# V47 progressive rear-left preload

V47 tests the next stair-climbing prerequisite after the independently
verified rear-right `190 mm` foot lift: can the mixed-height robot load all four
feet and move composite COM into the future three-foot support polygon before
rear-left unload?

## Immutable physical contract

- stair rise: `0.180 m`
- stair tread depth: `0.250 m`
- actuator effort cap: `0.8825985 Nm`
- source boundary: `rear-left-transfer-snapshot-v46-seed937.json`
- policy camera input: none
- controller inputs: joint/base state, IMU attitude/rates, force-backed foot
  loads, composite COM, and known analytic stair geometry

The external camera remains recording-only. V47 does not need better vision to
identify the fixed stair; the measured problem is contact/load/attitude during
weight transfer.

## Controller change

`reanchor_inter_leg_transfer_snapshot()` preserves the physical articulation
and load-bearing PD targets while restarting base, composite-COM, and four-leg
reference origins at a measured settled state. The optional
`four_foot_preload_load_sharing` controller computes zero-sum vertical
corrections toward equal measured normal-load fractions. It is active only
before unload and only when explicitly enabled by the V47 search.

The first strict target search was:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_rear_left_progressive_preload.py `
  --seed 946 --maximum-stages 1 `
  --target-deltas-m 0.005:-0.005,0.010:-0.005,0.005:-0.010,0.010:-0.010 `
  --durations-seconds 5.0,8.0 `
  --pitch-feedback-modes off,front_only,all `
  --maximum-body-tilt-deg 12.0 `
  --report simulation/isaac/output/rl/rear-left-progressive-preload-v47-stage1-seed946.json
```

The seed-946 report predates four-foot load-sharing injection, so its executed
grid contains 24 pitch/timing/target combinations. Strict pass count was zero.
It is retained because it exposes the contact-loss/slip mechanism.

The focused load-sharing diagnostic was:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_rear_left_progressive_preload.py `
  --seed 950 --maximum-stages 1 `
  --target-deltas-m 0.005:-0.005 --durations-seconds 5.0 `
  --pitch-feedback-modes off --load-sharing-gains-m 0.060 `
  --maximum-body-tilt-deg 15.5 `
  --report simulation/isaac/output/rl/rear-left-progressive-preload-v47-diagnostic-seed950.json
```

It remained below the simulator's `20 deg` fall threshold and ended with:

- all-foot normal loads: `38.331, 19.342, 27.564, 1.462 N`;
- completed-foot tread loads: `38.331, 27.564, 1.462 N`;
- support margin: `-95.425 mm`, improved by `7.658 mm`;
- target error: `9.095 mm`;
- maximum foot slip: `11.742 mm`;
- maximum tilt: `14.989 deg`;
- final base speed/body rate: `0.0185 m/s`, `0.0630 rad/s`.

This is better contact behavior but still fails the strict `12 deg` gate and
is `110.425 mm` short of the required `+15 mm` rear-left-unload margin.

## Re-anchored second-stage result

The relaxed first-increment diagnostic and saved state were reproduced with:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_rear_left_progressive_preload.py `
  --seed 952 --maximum-stages 1 `
  --target-deltas-m 0.005:-0.005 --durations-seconds 5.0 `
  --pitch-feedback-modes off --load-sharing-gains-m 0.060 `
  --maximum-body-tilt-deg 15.5 `
  --report simulation/isaac/output/rl/rear-left-progressive-preload-v47-stage1-seed952.json `
  --save-transfer-snapshot simulation/isaac/output/rl/rear-left-transfer-snapshot-v47-stage1-seed952.json
```

Report SHA-256:
`81854d2acfbf7f952b4d53f58bcb6391ea79862f45f518e16886d94fe12599c2`.
Snapshot SHA-256:
`96c300a16ed14328115547bccd21ff429dcbf6a879e9408da7ebcfd9b1fd0fe4`.

The second-stage direction search restored that state and tested 12 bounded
target/pitch candidates:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_rear_left_progressive_preload.py `
  --phase-snapshot simulation/isaac/output/rl/rear-left-transfer-snapshot-v47-stage1-seed952.json `
  --seed 953 --maximum-stages 1 `
  --target-deltas-m=-0.005:-0.005,0.000:-0.005,-0.0025:-0.0025,0.000:-0.0025,-0.005:0.000,0.000:0.000 `
  --durations-seconds 5.0 --pitch-feedback-modes off,front_only `
  --load-sharing-gains-m 0.060 --maximum-body-tilt-deg 15.5 `
  --report simulation/isaac/output/rl/rear-left-progressive-preload-v47-stage2-search-seed953.json
```

All 12 were rejected. Minimum candidate tilt was `17.771 deg`; best margin was
`-88.998 mm`. Report SHA-256:
`6896a9382bae1daa7816d3df237278de11eddb95814b610e4dbc98fceb4a86a7`.

## Decision

Do not spend another PPO budget on this boundary yet. The robot has already
proved a `202.907 mm` rear-right flat-ground lift, so leg clearance is not the
blocker. The next experiment should improve the force-backed rear-right
foothold and mixed-height stance—more normal load, deeper tread placement,
and measured foot friction/compliance—then repeat this exact preload gate.
Vision can follow after the known-geometry controller is physically stable.
