# Rear-right first-tread landing search (V39-V45)

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

V42-V44 supersede that negative conclusion with a force-backed landing; the
remaining warning about torque, traction calibration, and full-staircase proof
still applies.

## V42 support-margin and V43 touchdown ablations

V42 clips the analytic COM target into the current support polygon at the
configured positive margin and adds separate front-foot reach corrections.
Its bounded support-residual run is reproducible with:

```powershell
& simulation\isaac\rl\stairs\train_stairs_v42_com_margin_rear_right_landing_small.ps1 `
  -OutputDir simulation\isaac\output\rl\ppo-stairs-v42-com-margin-rear-right-landing-1024-seed865 `
  -Seed 865
```

That seed did not complete the landing. V43 added a tread-load feedback
correction targeting `15 N`, with `0.0005 m/N` gain and `40 mm` cap. Same-seed
replays showed that correction was reactive by one physics frame: at seed 862,
V42 lost both front contacts at step 556 and hit the tread at step 560 with
`65.12 N` and `15.657 deg` tilt. V43 applied `25.06 mm` correction only after
that initial impulse and did not change the failure.

The extended deterministic search therefore varied support reach and pitch
feedback before contact. Front-left `30 mm` plus front-right `90 mm` support
reach restored all stance contacts. A `0.255` pitch gain with `0.080 m` maximum
correction reduced the first seed-862 rear-right contact to `1.496 N` at
`11.026 deg`. These distances are commanded support references, not CAD or
link-length edits.

## Accepted V44 landing

V44 allows a physical tread contact after the clearance gate to become the
placement contact. It latches the reference at first valid contact instead of
continuing to advance the swing reference while the live environment verifies
contact, upright, support, slip, and margin gates for the full `0.75 s` hold.

Candidate 91 at seed 862 completed all `45/45` contact-hold frames:

- physical rear-right lift: `218.873 mm`
- minimum support margin: `39.660 mm`
- minimum support-contact fraction: `1.0`
- maximum rear-right tread load: `18.485 N`
- maximum support slip: `13.809 mm`

The bounded PPO job is:

```powershell
& simulation\isaac\rl\stairs\train_stairs_v44_early_contact_rear_right_landing_small.ps1 `
  -OutputDir simulation\isaac\output\rl\ppo-stairs-v44-early-contact-rear-right-landing-512-seed869 `
  -Seed 869
```

It completed exactly `512` steps in `77.254 s`. The horizon ends before the
rear-right contact phase, so its zero completed episodes are not landing
evidence. A separate fresh evaluation at seed 870 composed the trained V44
support policy with the verified V10/V17/V35 policies and completed `45/45`
hold frames with `217.990 mm` lift, `39.443 mm` margin, all three support
contacts, `14.000 mm` slip, and `10.238 N` maximum tread load. The first
accepted contact was `6.789 N` at `10.681 deg` body tilt.

The tracked package is
`simulation/isaac/models/ppo-stairs-v44-early-contact-rear-right-landing-small/`.
Local raw search/training outputs remain ignored. The external camera is used
only by the recording paths in `record_stairs_ppo.py` and
`search_rear_right_landing.py`; RGB remains absent from the 95-value policy
observation.

The accepted phase-local seed-870 camera replay records candidate 91 for 331
frames at 30 fps and completes the same `45/45` hold at step 662. The strict
reset-to-contact recorder separately rejected four prefix attempts
(`body_tipped`, two `body_transfer_failed`, then `body_tipped`). The published
video is therefore labeled phase-local and is not evidence that the complete
three-foot prefix is robust from reset.

## V45 rear-left transfer probe

V45 adds `rear_left` to the placement sequence only to expose and search the
rear-right-to-rear-left transfer. It does not claim a rear-left lift. The
bounded probe is:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_rear_left_transfer_com.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v45_rear_left_transfer.yaml `
  --seed 870 --forward-deltas-m 0.000,0.020 `
  --lateral-deltas-m=-0.020,0.000 `
  --minimum-support-margin-m 0.015 --maximum-body-tilt-deg 12 `
  --report simulation\isaac\output\rl\ppo-stairs-v45-rear-left-transfer-grid-retry-seed870.json
```

All four prefix attempts terminated with `body_tipped` before the rear-left
inter-leg-transfer snapshot became trainable. This is negative next-stage
evidence, not a V44 landing failure. The next controller should explicitly
regulate pitch/body rate and COM during the newly loaded rear-right transition,
then search rear-left unloading. Longer end-to-end PPO or RGB vision is not yet
justified; hardware traction/compliance measurements remain useful for
sim-to-real calibration, but the immediate simulated failure is transfer
attitude rather than unseen geometry.

## V45 exact-snapshot transfer training

V45 now waits for a low-rate rear-right landing state, allows per-next-leg
transfer residual authority, supports per-leg pitch feedback, and can ramp a
bounded body-relative outward foothold offset during swing advance. The source
also supports an exact phase snapshot and clipped potential-difference COM
progress reward, so every short PPO episode starts from the same verified
physical boundary without replaying the fragile prefix.

The fine rear-right offset search used candidate 91 with offsets `5, 10, 15,
20, 25, 30 mm`. Only `5 mm` retained the complete landing: `217.990 mm` lift,
all `45/45` hold frames, `39.443 mm` minimum support margin, `14.000 mm`
maximum support slip, and `13.819 N` maximum tread load. Offsets at or above
`10 mm` tipped before contact. The selected 5 mm reference did not materially
widen the physical foothold, so it is not presented as a transfer fix.

The exact boundary snapshot has SHA-256
`587ffc5e447e8f36f877490dae7525848529480304565cc7cc1c04e7a1143f85`.
Its COM starts at `[0.487750, 0.067301, 0.337153] m` and the analytic target is
`[0.567750, -0.033200, 0.337153] m`: an `80.0 mm` forward and `100.5 mm`
lateral move. Zero action still tipped, and all 27 constant loaded-support
hip-abduction combinations worsened the final COM error.

The bounded dynamic policy was trained with:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v45_rear_left_transfer.yaml `
  --output-dir simulation\isaac\models\ppo-stairs-v45-rear-left-dynamic-transfer-4096 `
  --seed 875 --total-timesteps 4096 `
  --phase-train-leg rear_left --phase-train-transfer `
  --phase-snapshot simulation\isaac\models\ppo-stairs-v45-rear-left-dynamic-transfer-4096\phase_snapshot_seed870.json `
  --fixed-placement-level left-center-tread-load `
  --ppo-learning-rate 0.0001 --ppo-initial-log-std -0.3 `
  --ppo-entropy-coefficient 0.001 --device cuda
```

It completed 4,096 steps in `135.744 s`, with `0` successful transfers and
only `15.136 mm` maximum rear-left lift. The deterministic seed-876 replay
tipped after 215 control steps; COM-target error increased from `128.454 mm`
to `153.372 mm` and maximum tilt reached `20.790 deg`. The external-camera
clip has 107 frames at 30 fps, but RGB remains absent from the policy.

This rejects longer training on the same boundary/controller geometry. The
next justified stage is a post-landing rear-right sidestep-and-settle controller
that creates a physically wider support polygon before asking rear-left to
unload. Validate positive margin and COM capture first, then train the 190 mm
rear-left lift. Better traction remains a sim-to-real calibration item; better
vision is not needed for this known fixed 180 x 250 mm stair failure.

## V46 rear-right sidestep and fresh lift-capability split

V46 rewinds only the placement state machine after the accepted V44/V45
rear-right landing, preserving the exact physical articulation state. It then
re-lifts rear-right from the tread and searches a small outward/deeper
replacement before caching the rear-left transfer boundary. The selected
seed-937 command was `10 mm` outward, `15 mm` forward, and `60 mm` relative
apex. It completed in `438` steps with:

- `9.215 mm` physical outward displacement;
- `7.033 mm` measured re-clearance from the already elevated tread pose;
- `65.201 mm` minimum replacement support margin;
- `2.301 N` final force-backed rear-right tread load;
- `15.999 mm` maximum support slip and `11.461 deg` maximum tilt.

The report SHA-256 is
`f1d3622add4f020caaec7c26fea75e1eb20cf1c4c67e91aa7c87c1193cb8c67d`.
The saved rear-left transfer snapshot SHA-256 is
`7545b2c0370e6c58f753487409b3f9a27c1876dbcf4e114b33872823623d1be7`.
Its requested COM move remains `80.0 mm` forward and about `98.1 mm` lateral.

The exact 4,096-step phase PPO command is recorded in
`simulation/isaac/models/ppo-stairs-v46-rear-right-sidestep-transfer-4096/README.md`.
Seed 939 completed the pipeline in `129.338 s`, but achieved `0` transfers and
only `13.425 mm` rear-left motion. Deterministic seed 940 tipped after `93`
steps; COM-target error grew from `126.584` to `133.259 mm`, final support
margin was `-101.073 mm`, and maximum tilt was `20.144 deg`. The external
camera recorded 46 frames at 30 FPS; RGB was not a policy input.

To separate raw leg ability from this mixed-height failure, a fresh `512`-step
flat-ground rear-right policy was trained at seed 941. Independent seed-942
evaluation passed `5/5` with `201.006-204.345 mm` lift and `2.218 deg` worst
tilt. Therefore the modeled robot can raise the foot above the 180 mm riser;
the unresolved stair problem is the loaded mixed-height COM transfer and
controller sequence. The next stage should add an all-four-feet settle/preload
phase before rear-left unload. More RGB vision is not the immediate need.
