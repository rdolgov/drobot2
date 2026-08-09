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

## V48 force-backed foothold and attitude search

V48 first searched for a materially loaded rear-right contact instead of
optimizing outward displacement. The accepted deterministic command was:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_rear_right_post_landing_sidestep.py `
  --seed 964 --outward-offsets-m 0.005 --forward-offsets-m 0.015 `
  --relative-apex-lifts-m 0.060 `
  --minimum-physical-outward-displacement-m=-0.006 `
  --minimum-rear-right-tread-load-n 15.0 `
  --report simulation/isaac/output/rl/rear-right-force-backed-foothold-v48-seed964.json `
  --save-transfer-snapshot simulation/isaac/output/rl/rear-left-transfer-snapshot-v48-force-backed-seed964.json
```

This is a force-backed inward settle, not a successful sidestep. The physical
rear-right foot moved `5.203 mm` inward, but held `33.739 N` tread load with
`12.048 mm` maximum slip, `11.457 deg` maximum tilt, and `65.390 mm` minimum
replacement support margin. The future rear-left three-foot support margin was
still negative at `-94.730 mm`. Report SHA-256:
`610724d4a3a2e42865f64a662fcf28b172275049e06fb1c0a901241830b0f5f5`.
Saved snapshot SHA-256:
`c24b78fb494615c1395ebe0a2cbede0bde174455cc40f8180424825883391deb`.

The first nominal `5 mm` forward / `5 mm` lateral preload from this stronger
boundary appeared to improve support margin by `14.125 mm`, but the composite
COM actually moved only `[0.722, 0.175] mm`, rear-right load fell to `0 N`,
maximum foot slip reached `15.269 mm`, and body tilt reached `13.518 deg`.
V48 roll/pitch attitude searches did not recover the lost normal force. The
best gentle roll-hold point remained at `13.59 deg`; aggressive leveling
increased pitch and slip.

## V49 traction sensitivity and slip-proof progress gate

The search now reports measured composite-COM progress along the requested
increment and requires at least `50%` progress before a state may settle. This
prevents moving support feet from being mis-ranked as successful COM transfer.
It also records explicit acceptance-gate failures and the minimum rear-right
load over the whole rollout.

The high-traction sensitivity run is:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_rear_left_progressive_preload.py `
  --phase-snapshot simulation/isaac/output/rl/rear-left-transfer-snapshot-v48-force-backed-seed964.json `
  --seed 970 --maximum-stages 1 --target-deltas-m 0.005:-0.005 `
  --durations-seconds 5.0 --pitch-feedback-modes off `
  --pitch-feedback-gains-m 0.080 --roll-feedback-gains-m 0.0 `
  --load-sharing-gains-m 0.0001 --load-sharing-maximum-correction-m 0.0001 `
  --minimum-final-rear-right-load-n 5.0 --maximum-body-tilt-deg 12.0 `
  --maximum-all-foot-slip-m 0.022 --minimum-balance-progress-fraction 0.50 `
  --static-friction 1.8 --dynamic-friction 1.5 --friction-combine-mode max `
  --report simulation/isaac/output/rl/rear-left-traction-sensitivity-v49-seed970.json
```

Compared with the conservative rubber-pad model (`1.2` static, `1.0` dynamic,
average combine), the higher-traction model did not improve the mechanism. It
produced only `0.312 mm` useful COM progress (`4.41%` of the command), rear-right
load still fell to `0 N`, maximum slip was `15.489 mm`, and tilt reached
`13.547 deg`. The apparent support-margin gain was `14.318 mm`, confirming it
was dominated by contact geometry/slip rather than commanded COM motion.
Report SHA-256:
`48415d4fcda247da5c6a3607ecf35806d6b180f35d45a4568a659d3880f1056c`.

Therefore the next physical/simulation change should be a wider compliant
rubber foot pad with calibrated collision/contact geometry and explicit
normal-force retention. Increasing the friction coefficient alone is not
enough. RGB/depth vision remains deferred because the failure occurs on exact,
known `180 x 250 mm` geometry with complete proprioceptive/contact state.
