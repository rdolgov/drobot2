# Rear-right first-tread landing search (V39-V45)

> **Superseded contact interpretation (2026-08-02):** V44-V49 used a
> `RigidPrim` contact sensor attached to the entire distal leg link. The
> reported `StepLayer_01` force could therefore come from the shin, distal arm,
> riser, or tread edge rather than the foot sole on the tread top. V53's
> geometry-qualified tread-top gate gives every completed foot in the V48
> snapshot `0 N` qualified load, so all earlier "force-backed tread" claims in
> this document are historical false positives. The flat-ground 190 mm lift
> result remains valid. See V53-V54 below for the corrected result and next
> action.

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

## V47 progressive four-foot preload

V47 implements that next gate without changing the exact `180 mm` rise,
`250 mm` tread, or measured `0.8825985 Nm` effort cap. It preserves the active
load-bearing PD targets, re-anchors each analytic COM increment at measured
base/joint/composite-COM state, and optionally applies bounded zero-sum
four-foot load sharing before rear-left unload.

The strict `12 deg` seed-946 search tested 24 combinations of `5-10 mm`
increments, `5/8 s` timing, and off/front-only/all pitch feedback. None passed.
The apparent best margin gains lost rear-right tread load or exceeded the slip
and attitude gates, so they are negative evidence rather than PPO
initialization states.

With four-foot load sharing at `0.060 m` proportional gain, `0.020 m` maximum
correction, and `0.50` smoothing, seed 950 ended with measured total loads
`[38.331, 19.342, 27.564, 1.462] N` in simulator foot order, `11.742 mm`
maximum slip, `0.0185 m/s` base speed, and `0.0630 rad/s` body rate. Support
margin improved only from `-103.083` to `-95.425 mm`, and tilt reached
`14.989 deg`; the strict 12-degree gate correctly rejected it.

An explicitly relaxed `15.5 deg` seed-952 diagnostic held that first increment
for `0.40 s` without a simulator failure, but a fresh seed-953 search from its
saved state rejected all 12 second-increment candidates. The best second-stage
margin remained `-88.998 mm` and the least candidate tilt was `17.771 deg`.
Therefore the first increment is a short-horizon diagnostic, not a stable
transfer or training boundary. No additional transfer PPO was trained because
the analytic safety prerequisite did not pass. See
`PROGRESSIVE_PRELOAD_SEARCH.md` for exact commands and hashes.

## V48 force-backed rear-right settle

The V46 outward candidate preserved geometry but ended at only `2.301 N` rear-
right load. V48 therefore reuses the exact landing snapshot and searches for a
deeper force-backed settle, while reporting physical displacement honestly.
Seed 964 accepted a `5 mm` nominal outward, `15 mm` forward, and `60 mm`
relative-apex command under a relaxed `-6 mm` displacement floor and a strict
`15 N` final-load floor.

The foot physically settled `5.203 mm` inward, so this result is not called a
sidestep. It completed in `418` steps with `33.739 N` final rear-right tread
load, `12.048 mm` maximum support slip, `11.457 deg` maximum tilt, all stance
contacts retained, and `65.390 mm` minimum replacement support margin. The
cached next-transfer margin remains negative at `-94.730 mm`, so rear-left PPO
is still gated. Exact command, hashes, preload response, and the higher-
traction V49 sensitivity are recorded in `PROGRESSIVE_PRELOAD_SEARCH.md`.

## V53-V54 true tread-top gate and first-foot reset

V53 keeps the raw per-step-layer contact force for diagnosis but qualifies a
load as tread support only when the sampled foot bottom is horizontally inside
the exposed `250 mm` tread top (with a `5 mm` inset) and vertically within
`15 mm` of its `180 mm` top surface. Snapshot restore, placement completion,
and transfer gates now consume only this qualified load. This is still a
camera-blind policy: RGB is used only for external evidence recording.

The V48 snapshot was audited with:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_rear_left_progressive_preload.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v46_rear_right_sidestep.yaml `
  --phase-snapshot simulation\isaac\output\rl\rear-left-transfer-snapshot-v48-force-backed-seed964.json `
  --seed 994 --candidate-target-deltas-m 0.0025,-0.0025 `
  --candidate-durations-seconds 1.0 --candidate-pitch-feedback-modes off `
  --candidate-pitch-feedback-gains-m 0.08 --candidate-roll-feedback-gains-m 0.0 `
  --candidate-load-sharing-gains-m 0.0001 --maximum-stages 1 `
  --report simulation\isaac\output\rl\rear-left-tread-top-restore-audit-v53-seed994.json
```

Restore correctly failed: the three formerly completed feet had qualified
tread-top loads `[0, 0, 0] N` while the whole-distal-link sensor still reported
raw step-layer loads `[13.152, 29.781, 35.391] N`. The report SHA-256 is
`16eea4613acc2d9f3d311b93fa34e58cb0898468f6a21bc6d5d954360c73d05a`.
A fresh full-prefix seed-995 audit also rejected all four precursor attempts
before rear-right with `phase_training_precursor_timeout`; report SHA-256 is
`64d3d5a8b989aa2ad7dc398ab3d5dd36a80f23730681a37f84d129366cd0412a`.

The corrected task was reset to only front-right placement and trained for
four bounded PPO trials. The fixed-level command for the final trial was:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v10_front_right_single_tread_placement.yaml `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v54-true-tread-top-quarter-front-right-1024-seed1000 `
  --total-timesteps 1024 --curriculum-total-timesteps 1024 `
  --fixed-placement-level quarter-tread-load --seed 1000 --device cpu `
  --ppo-learning-rate 0.00003 --ppo-initial-log-std -3.5 `
  --ppo-entropy-coefficient 0.0005
```

The seed-1000 run completed 1,024 PPO steps in `30.09 s`. It remained upright
(`2.775 deg` maximum tilt), raised front-right `199.719 mm`, kept support slip
to `3.450 mm`, and retained all three support contacts. The final foot was at
`x=0.598395 m`, `z=0.195517 m`, but qualified tread-top load remained `0 N`;
therefore placement success was `0`. The report SHA-256 is
`c05f7bf5b66d3b3c2b31f0eb3557c90ced4a01ab24775930df0f341c403bd4b2`.

The bounded placement sweep is consistent:

- seed 996, near edge: `192.165 mm` lift, `3.073 deg` tilt, `4.015 mm`
  slip, final `x=0.530991 m` (19 mm short of the riser), `0 N` top load;
- seed 998, center: `200.445 mm` lift, `3.470 deg` tilt, `3.410 mm` slip,
  final `x=0.616453 m`, `z=0.195723 m`, `0 N` top load;
- seed 999, center with `120 mm` landing request: `205.254 mm` lift,
  `3.223 deg` tilt, `3.528 mm` slip, final `x=0.628114 m`,
  `z=0.197053 m`, `0 N` top load;
- seed 1000, quarter tread: `199.719 mm` lift, `2.775 deg` tilt,
  `3.450 mm` slip, final `x=0.598395 m`, `z=0.195517 m`, `0 N` top load.

The archived training-report SHA-256 values for seeds 996, 998, 999, and 1000
are respectively `07f485f7c2e3043d21778fe8c6941d63aedb461295bba0e18416cb49c9ece73d`,
`383f0b1cabdd8afc60c494f3ae66048557f200703c9df2ffff0dd56f98378396`,
`e1ab4907b4ef73959716f8eb1453ba2bb2c613380ba3e5de0a0ace87d8c9d552`,
and `c05f7bf5b66d3b3c2b31f0eb3557c90ced4a01ab24775930df0f341c403bd4b2`.

This corrected evidence rules out traction and vision as the immediate first-
plant bottleneck: the support feet barely slip, the body stays upright, stair
geometry is known, and the swing sole never reaches the tread top. The next
training stage must first move the torso/support polygon forward to create a
reachable descend-and-load pose, then train the first-foot plant. Hardware
should add a dedicated sole load sensor or isolated foot rigid body so shin or
riser force cannot satisfy a foothold gate. Vision becomes important later for
unknown stair localization; friction tuning becomes important only after a
true top contact begins to slip. No V54 success video is published because no
strict tread-top placement passed; the existing seed-943 external-camera video
remains evidence only for the independently reproduced 190 mm lift capability.

## V55-V62 stance, hip, body-position, and loaded-contact sweep

The next bounded sweep keeps the exact `180 mm` rise, `250 mm` tread, measured
`0.8825985 Nm` effort cap, and camera-blind 81-value policy observation. A new
`--first-tread-profile` selector applies reproducible posture/approach variants
to training, evaluation, and recording. `probe_first_tread_profiles.py` runs
the same analytic placement reference with a zero PPO residual and reports the
strict geometry-qualified top load separately from raw distal-link contact.

The representative probe command was:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\probe_first_tread_profiles.py `
  --profile forward-preposition-load `
  --placement-level quarter-tread-load --seed 1012 `
  --report simulation\isaac\output\rl\first-tread-profile-v61-forward-preposition-load-seed1012.json
```

The profile findings were:

- fully folded at a measured-limit-safe `119 deg` knee: only `75.128 mm`
  front-right lift, `-81.709 mm` support margin, and `41.693 deg` tilt;
- `220 mm` low crouch: no measurable lift, `-53.813 mm` support margin, and
  `1.127 rad` swing tracking error under the hardware effort cap;
- sideways `90 deg`: `9.465 mm` lift, `-50.880 mm` margin, `46.460 mm` slip;
- diagonal `45 deg`: `2.254 mm` lift, `-77.140 mm` margin, `39.130 deg` tilt;
- angled `20 deg`: `216.304 mm` lift but `-171.098 mm` margin and `272.626 mm`
  backward motion;
- forward baseline: stable `199.850 mm` lift, `-2.249 mm` margin, `3.411 mm`
  slip, and `0 N` qualified top load;
- forward pre-position (`30 mm` closer with the same world foothold): stable
  `198.628 mm` lift, `-1.566 mm` margin, `3.415 mm` slip, and `0 N` top load;
- a `130 mm` landing-lift command ended at `z=195.180 mm`, preserved a
  `+1.648 mm` minimum margin, but still carried `0 N`;
- a `110 mm` landing-lift command produced the first true top contact at
  `8.870 N`, but the abrupt load transfer slid the body backward and tipped;
- doubling the lower phase to `3.0 s` still tipped and is rejected.

Thus the low/folded and sideways/hip-first hypotheses are negative evidence.
Moving the torso closer is useful because it keeps the lift stable and improves
the support boundary, while a deeper landing proves the foot can physically
load the tread. The current problem is retaining three-foot support during the
first 10-12 N load transient.

Two 1,024-step CPU PPO smoke trainings were run. The loaded-contact command was:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v10_front_right_single_tread_placement.yaml `
  --first-tread-profile forward-preposition-load `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v61-forward-preposition-load-1024-seed1013 `
  --total-timesteps 1024 --curriculum-total-timesteps 1024 `
  --fixed-placement-level quarter-tread-load --seed 1013 --device cpu `
  --ppo-learning-rate 0.00003 --ppo-initial-log-std -3.5 `
  --ppo-entropy-coefficient 0.0005
```

Training completed in `32.123 s` and observed `10.302 N` qualified load, but
its rollout tipped and the optimizer hit the `0.03` target-KL stop. Independent
seed-1022 evaluation used three episodes: all reached `11.464-12.559 N` true
top load and `197.914-198.932 mm` lift, but strict placement remained `0/3`.
One episode ran the full `14 s` with `4.991 deg` maximum tilt, `18.182 mm`
maximum slip, and `-1.481 mm` minimum support margin; two moved about
`305-311 mm` backward and tipped. This is a contact-capable training boundary,
not a reliable stair policy and not a completed climb.

The exact stable-contact replay was recorded with:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\record_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v10_front_right_single_tread_placement.yaml `
  --first-tread-profile forward-preposition-load `
  --model simulation\isaac\output\rl\ppo-stairs-v61-forward-preposition-load-1024-seed1013\drobot_stairs_ppo_final.zip `
  --seed 1022 --device cpu --active-steps 1 `
  --placement-level quarter-tread-load --camera-view external `
  --fps 30 --width 960 --height 540 `
  --video reviews\ppo-stairs-v61-forward-preposition-load-contact-eval-seed1022.mp4 `
  --thumbnail reviews\ppo-stairs-v61-forward-preposition-load-contact-eval-seed1022.png `
  --report simulation\isaac\output\rl\ppo-stairs-v61-forward-preposition-load-record-seed1022.json
```

The 420-frame H.264 MP4 is `16,602,005` bytes with SHA-256
`38954f339916d29bd8c80efc2b07734b61641c45df0f1c2580c0106d407792a1`.
The three-episode evaluation report SHA-256 is
`4f6c23b2a5a98dfcb7beac1f62fa4637c9d2c30cf18444cefafb710fbfeb9275`.
RGB remains recording-only. The next run should train a short, contact-triggered
support/COM residual that acts only around touchdown; neither additional vision
nor higher friction addresses the observed first-load instability yet.

## V63-V75 contact-triggered support release and first strict foothold

V63-V73 tested the touchdown transient directly. Releasing only half of the
forward stance bias still tipped at `12.268 N`; releasing the full forward bias
could remain upright, but seed sensitivity exposed a front-left support-foot
unload. Centered, forward-only, faster-release, and front-left-preload variants
did not remove that failure. The decisive change starts forward support release
during the `advance` phase, before contact, while leaving lateral support bias
in place. This keeps the three stance feet loaded as the swing foot descends.

The zero-residual V74 acceptance probe used:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\probe_first_tread_profiles.py `
  --profile forward-preposition-load-advance-forward-floor `
  --placement-level quarter-tread-load --seed 1018 `
  --report simulation\isaac\output\rl\first-tread-profile-v74-advance-forward-floor-seed1018.json
```

It completed the first strict, geometry-qualified front-right foothold in
`741` control steps (`12.35 s`): `207.154 mm` maximum lift, `11.472 N` maximum
true tread-top load, `19.919 mm` maximum support slip, `8.677 deg` maximum
tilt, and no failure reason. Minimum post-touchdown loads were `10.315 N`
front-left, `32.921 N` rear-left, and `32.352 N` rear-right. The report SHA-256
is `e9582511ba9c871ac48c239d166aa602c896f42f38c05e09b261e24040fefaab`.

A deliberately small all-joint residual PPO was then trained from the same
reference:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v10_front_right_single_tread_placement.yaml `
  --first-tread-profile forward-preposition-load-advance-forward-floor `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v75-first-strict-foothold-2048-seed1025 `
  --total-timesteps 2048 --curriculum-total-timesteps 2048 `
  --fixed-placement-level quarter-tread-load --seed 1025 --device cpu `
  --ppo-learning-rate 0.000001 --ppo-initial-log-std -5.5 `
  --ppo-entropy-coefficient 0.0
```

Training completed `2,048` steps in `60.550 s`. Fresh evaluation used:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v10_front_right_single_tread_placement.yaml `
  --first-tread-profile forward-preposition-load-advance-forward-floor `
  --model simulation\isaac\output\rl\ppo-stairs-v75-first-strict-foothold-2048-seed1025\drobot_stairs_ppo_final.zip `
  --episodes 5 --seed 1026 --device cpu --active-steps 1 `
  --placement-level quarter-tread-load `
  --report simulation\isaac\output\rl\ppo-stairs-v75-first-strict-foothold-eval-5ep-seed1026.json
```

Four of five deterministic fresh episodes completed the required foothold.
Those four lifted `209.003-211.115 mm`, carried `12.608-16.531 N` qualified
tread load, and stayed below `10.047 deg` tilt. One episode tipped before true
contact. Successful episodes showed `28.178-33.478 mm` maximum whole-episode
support-foot motion, above the `25 mm` measurable-slip diagnostic, so V75 is a
useful first-foot policy but not yet the robust boundary for adding another
leg. The V74 zero-residual controller remains the cleaner strict boundary. The
five-episode report SHA-256 is
`0bb46699886e05e06ba2cbead862fc162aca199e3bdaa172abadde293597f351`.

The accepted seed-1026 policy replay was recorded with:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\record_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v10_front_right_single_tread_placement.yaml `
  --first-tread-profile forward-preposition-load-advance-forward-floor `
  --model simulation\isaac\output\rl\ppo-stairs-v75-first-strict-foothold-2048-seed1025\drobot_stairs_ppo_final.zip `
  --seed 1026 --device cpu --active-steps 1 `
  --placement-level quarter-tread-load `
  --search-placement-success-episodes 5 --camera-view external `
  --fps 30 --width 960 --height 540 `
  --video reviews\ppo-stairs-v75-first-strict-foothold-eval-seed1026.mp4 `
  --thumbnail reviews\ppo-stairs-v75-first-strict-foothold-eval-seed1026.png `
  --report simulation\isaac\output\rl\ppo-stairs-v75-first-strict-foothold-record-seed1026.json
```

The 337-frame H.264 MP4 is `14,783,668` bytes with SHA-256
`87e4578c2198abaf74151cd736e9ebd08f465989ef4e4ee5340af02c23b6c06f`.
The recording reaches `211.115 mm` lift and `12.728 N` tread load without a
failure. It uses an external camera only for evidence; the policy remains
camera-blind and consumes IMU, joint/proprioceptive, contact/load, prior-action,
and known analytic stair-geometry values. The stair stays exactly `180 mm`
rise by `250 mm` tread and the applied joint effort stays capped at
`0.8825985 N m`. This is one-foot placement, not a completed stair climb. The
next bounded task should restore the cleaner V74 boundary and train front-left
placement while explicitly penalizing any loss of front-right tread load.

## V80 simplified single-foot 190 mm lift

Before adding a second stair foothold, V80 isolates the user's hardware-level
reach question: raise front-left at least `190 mm`, hold it for `0.50 s`, keep
the three support feet loaded, and do not tip. The task reuses the V15
single-foot environment and current real-test hardware profile. The staircase
remains exactly `180 mm` rise by `250 mm` tread, and applied effort remains
capped at `0.8825985 N m`.

The fresh 2,048-step fine-tune used:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\train_stairs_v80_single_foot_190mm_ppo.py
```

The wrapper resumes the verified V17 isolated-lift policy, starts directly at
`front-left-stabilized-190mm-lift-hold`, and preserves the saved PPO algorithm
contract. All six recent training episodes passed. Their lift range was
`203.626-208.060 mm`, maximum tilt was `2.758 deg`, and maximum support slip
was `3.382 mm`.

Independent deterministic evaluation used a fresh seed:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v15_front_left_stabilized_lift.yaml `
  --model simulation\isaac\output\rl\ppo-stairs-v80-single-foot-190mm-2048-seed1030\drobot_stairs_ppo_final.zip `
  --episodes 5 --seed 1031 --device cpu --active-steps 1 `
  --placement-level front-left-stabilized-190mm-lift-hold `
  --maximum-lateral-deviation-m 0.20 `
  --report simulation\isaac\output\rl\ppo-stairs-v80-single-foot-190mm-eval-5ep-seed1031.json
```

All `5/5` episodes passed with `204.900-208.077 mm` lift, no failure reasons,
`2.324 deg` maximum tilt, `3.337 mm` maximum support slip, and at least
`10.386 N` on every support foot. The accepted replay was recorded with:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\record_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v15_front_left_stabilized_lift.yaml `
  --model simulation\isaac\output\rl\ppo-stairs-v80-single-foot-190mm-2048-seed1030\drobot_stairs_ppo_final.zip `
  --seed 1031 --device cpu --active-steps 1 `
  --placement-level front-left-stabilized-190mm-lift-hold `
  --camera-view external --fps 30 --width 960 --height 540 `
  --video reviews\ppo-stairs-v80-single-foot-190mm-eval-seed1031.mp4 `
  --thumbnail reviews\ppo-stairs-v80-single-foot-190mm-eval-seed1031.png `
  --report simulation\isaac\output\rl\ppo-stairs-v80-single-foot-190mm-record-seed1031.json
```

The 166-frame H.264 MP4 is `7,221,002` bytes with SHA-256
`1f7800f75b7c38f93e545692d7da45d45f84c01c9d06874d91a8447b71e658ef`.
It reaches `207.761 mm` lift with `2.130 deg` tilt and `3.174 mm` support
slip. RGB is recording-only; the policy remains camera-blind and consumes IMU,
joint/proprioceptive state, previous action, contact/load, and known analytic
stair geometry. This validates isolated foot lift and balance only. It does
not validate a second-foot transfer or complete stair climbing.

## V81-V82 retained first-foot load and hip-assisted transfer

The next composition experiment starts with the V75 front-right foothold,
requires that completed foot to retain at least `5 N` of geometry-qualified
tread load, caps whole-support slip at `35 mm`, then trains a compact six-output
front-left transfer policy. The six outputs control front-left hip abduction,
hip flexion, and knee plus the three support-leg hip-abduction joints. This is
the smallest action space that can both use the swing hip and shift lateral
support. It keeps the exact `180 mm` rise by `250 mm` tread, the measured
`0.8825985 N m` applied effort cap, and camera-blind policy input.

The first 2,048-step V81 run used:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v14_front_pair_right_then_left.yaml `
  --first-tread-profile front-pair-preposition-load-advance-forward-floor `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v81-front-left-transfer-2048-seed1032 `
  --total-timesteps 2048 --curriculum-total-timesteps 2048 `
  --fixed-placement-level left-quarter-tread-load `
  --phase-train-leg front_left --phase-train-transfer `
  --precursor-leg-model front_right=simulation\isaac\output\rl\ppo-stairs-v75-first-strict-foothold-2048-seed1025\drobot_stairs_ppo_final.zip `
  --phase-residual-swing-support-abduction --phase-compact-residual-action `
  --seed 1032 --device cpu --ppo-learning-rate 0.00005 `
  --ppo-initial-log-std -4.0 --ppo-entropy-coefficient 0
```

V81 retained about `20 N` on the placed foot but reached only `7.232 N`
minimum front-left load and completed zero transfers. Fresh seed 1033 became
heavier at `26.623 N`, lifted only `6.510 mm`, and exceeded the slip limit at
`35.578 mm`. Inspection found that the phase wrapper rewarded balance and
support-margin progress but did not directly reward unloading the designated
swing foot.

V82 adds a measured swing-load-reduction term (`50` reward per newton, clipped
to `1 N` progress per control step) while retaining balance and support-margin
progress terms. The 4,096-step rerun was:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v14_front_pair_right_then_left.yaml `
  --first-tread-profile front-pair-preposition-load-advance-forward-floor `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v82-front-left-unload-reward-4096-seed1034 `
  --total-timesteps 4096 --curriculum-total-timesteps 4096 `
  --fixed-placement-level left-quarter-tread-load `
  --phase-train-leg front_left --phase-train-transfer `
  --precursor-leg-model front_right=simulation\isaac\output\rl\ppo-stairs-v75-first-strict-foothold-2048-seed1025\drobot_stairs_ppo_final.zip `
  --phase-residual-swing-support-abduction --phase-compact-residual-action `
  --seed 1034 --device cpu --ppo-learning-rate 0.00003 `
  --ppo-initial-log-std -2.5 --ppo-entropy-coefficient 0
```

Training improved minimum swing load to `5.145 N`, kept maximum support slip
at `34.951 mm`, and accumulated `95.679 N` of clipped load-reduction progress,
but still completed zero transfers. Fresh evaluation used three episodes:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v14_front_pair_right_then_left.yaml `
  --first-tread-profile front-pair-preposition-load-advance-forward-floor `
  --model simulation\isaac\output\rl\ppo-stairs-v75-first-strict-foothold-2048-seed1025\drobot_stairs_ppo_final.zip `
  --leg-model front_right=simulation\isaac\output\rl\ppo-stairs-v75-first-strict-foothold-2048-seed1025\drobot_stairs_ppo_final.zip `
  --zero-action-leg front_left `
  --transfer-model front_left=simulation\isaac\output\rl\ppo-stairs-v82-front-left-unload-reward-4096-seed1034\drobot_stairs_ppo_final.zip `
  --transfer-residual-swing-support-abduction front_left `
  --episodes 3 --seed 1036 --device cpu --active-steps 1 `
  --placement-level left-quarter-tread-load `
  --maximum-lateral-deviation-m 0.20 --episode-seconds 45 `
  --allow-unverified-model `
  --report simulation\isaac\output\rl\ppo-stairs-v82-front-left-unload-reward-4096-seed1034\evaluation_seed1036_3ep.json
```

One of three episodes reached transfer. It retained `20.010 N` on the first
tread with `31.576 mm` maximum support slip, but the left foot remained at
`7.979 N` and lifted only `6.707 mm`; the other two exceeded the `35 mm` slip
cap before the first foothold completed. V82 is therefore an improved training
signal but not a generalizing transfer policy. The representative failure was
recorded with the same composition plus `--skip-episodes 1` to
`reviews/ppo-stairs-v82-front-left-transfer-eval-seed1036-ep2.mp4`. The MP4 is
The web-review H.264 transcode preserves all 633 frames and is `309,237`
bytes with SHA-256
`48c3d7b09919da355a510e3791163dc50c494cc7f15e8413e95526be33ff1a98`.

The V82 model SHA-256 is
`5069f46744036cc92b52a3dcf21af3cab7fa0ebc0d33b747c47a8dfabdb5eeea`;
the three-episode report SHA-256 is
`729faccd40ccfe899182660d40d3db56a240b275bfb60853f6cc21a1f65da7ad`.
The next bounded experiment should train from a deterministic retained-foot
snapshot and expose measured swing-foot total load directly in the policy
observation; additional vision or friction is not yet the limiting factor.
