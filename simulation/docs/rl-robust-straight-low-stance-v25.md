# V25b robust straight low-stance residual crawl

## Status

V25b is the corrected revision of the V25 training profile. It is **not a
selected or deployed model**. As of 2026-08-30, reference-only Isaac comparisons
and three nominal-training investigations have been run, but no checkpoint from
them is acceptable for export or hardware use. The current V25b reward,
reference, mirrored randomization, residual constraints, mass model, and
neutral-policy bootstrap have not yet completed a new nominal training run.
V24 `model_3248` remains the
selected and deployed policy until a V25b candidate passes every gate below.

The task identifier remains
`Drobot-Commanded-Walk-Robust-Straight-Low-Stance-External-Rear-Payload-Direct`.
The V25 name is retained in code and artifact paths for compatibility; this
document uses V25b for the corrected configuration. V25b keeps the V24 50-value
actor observation and 12-value residual action shape, but no longer resumes the
learned V24 residual unchanged. It starts from a feature-preserving,
zero-deterministic-residual Beta bootstrap described below.

The assembled robot's measured total mass is `7 lb`, exactly
`3.17514659 kg` using `1 lb = 0.45359237 kg`. The external rear battery,
printed box, and lid remain a separately measured/modelled `0.523179545 kg`
pack. The URDF/CAD ledger overestimated solid printed-part mass, so V25b applies
a `0.650607608` nominal scale to the dry robot while preserving the rear pack
explicitly. This mass correction is part of the nominal model, not domain
randomization.

## Why another continuation is needed

The real robot can execute the slower hardcoded three-support-leg crawl, but the
learned policy still moves with excessive left/right motion and can drift left.
The investigation found several structural causes that reward scaling alone
could not repair:

- V24 randomizes one common effort factor and one common target-rate factor for
  all twelve servos. It never sees a weaker left side, one biased joint zero, or
  a one-frame timing difference.
- The original V25 code read Isaac Lab Warp root quaternions as WXYZ even though
  those tensors are XYZW. That corrupted episode yaw and reset attitude noise.
  V25b now uses XYZW consistently for yaw extraction, reset quaternion creation,
  and quaternion multiplication.
- The original simulation retained the `4.526139 kg` fully-solid CAD/URDF mass
  ledger rather than the measured `3.17514659 kg` assembled total. V25b scales
  dry bodies by `0.650607608` and then adds the `0.523179545 kg` rear pack.
- A body-frame forward reward can remain positive after the robot has rotated
  away from its original path, while instantaneous lateral penalties can punish
  the intended zero-mean side transfer of a stable crawl.
- Low lift, low transfer, speed-scaled amplitude, and an inherited asymmetric
  residual allowed a rear shoe—especially rear-left—to remain loaded instead
  of releasing cleanly.
- Making straightness costs arbitrarily large risks restoring the old
  stationary bent-knee optimum.

V25b therefore combines corrected mass and quaternion contracts, a world-path
heading loop, cycle-averaged straight-line scoring, phase-locked sway
compensation, mirrored asymmetric hardware randomization, a lower opposed
stance, full-amplitude foot clearance, and a continuous three-foot support
push. Progress remains valuable and jerk costs remain active.

## Stance and gait contract

V25b stays with the proven sequential crawl order and a reference-plus-residual
policy. It does **not** switch this position-servo robot directly to a flying
trot or copy cadence values from a torque-controlled Unitree/ANYmal.

| Setting | V25b value |
| --- | ---: |
| Reset/target body height | `0.367333494 m` |
| Opposed front/rear foot sweep | `92 mm` fore/aft |
| Approximate fore/aft foot-center footprint | `304 mm` |
| Nominal sagittal hip/knee angle | `0.613058309 rad` (`35.1260 deg`) |
| Reference stance-down dimension | `0.321674941 m` |
| Target pitch | `3 deg` nose-down, inherited from V24 |
| Crawl order | rear-right, front-right, rear-left, front-left |
| Phase offsets, FL/RL/FR/RR | `0.07, 0.32, 0.57, 0.82` |
| Effective stance duty | `0.8625` |
| Command curriculum start | `0.008-0.025 m/s` |
| Final trained command range | `0.005-0.040 m/s` |
| Command/horizon curriculum | `51,200` policy steps (`800` PPO updates) |
| Cadence | `0.12-0.62 Hz`, command-scaled |
| Reference amplitude | full at every nonzero trained speed |
| Reference stride | `45 mm` |
| Swing lift | `20 mm` |
| Phase forward shift | `10 mm` |
| Phase lateral shift | `6 mm` |
| Startup gait ramp | `1.5 s` |
| Abduction residual scales | `[0.20, 0.20, 0.20, 0.20]` |
| Hip/knee residual scales | `[0.05] * 8` |
| Joint target slew cap | `240 deg/s`, or `4 deg` per 60 Hz update |

The lower stance is modest: about `7.67 mm` below the previous 80 mm swept
stance. A deeper crouch was rejected as the starting point because it consumes
servo torque and thermal margin, especially with a rear battery. The first
draft used a visually aggressive 80 mm stride, but its `69.74 deg` hip request
exceeded both the approximately `60 deg` hardware limit and Isaac's `57 deg`
soft limit. The current 45 mm stride uses `20 mm` lift and a `10 mm` forward
phase transfer to unload each moving shoe. Cadence, rather than stride/lift
amplitude, expresses requested speed: `gait_stride_scale_min = 1.0` keeps full
clearance and propulsion throughout the trained `0.005-0.040 m/s` range.

The action order is joint-kind-major: four abductions, four sagittal hips, then
four knees. Abduction retains a `0.20` residual so the policy can compensate
lateral assembly and motor asymmetry. Sagittal hips and knees are limited to
`0.05` residual so an inherited or newly learned correction cannot erase the
reference's clearance or consume the remaining soft-limit margin. The smoothed
reference spreads support-foot propulsion through lift, swing, and lowering
instead of applying it as a short all-feet shove.

The higher `4 deg` packet cap is V25b-specific. The installed STS3215-C018 is
the 12 V model; Feetech specifies `0.222 s/60 deg`, approximately `270 deg/s`,
at 12 V. V25b requests at most `240 deg/s` and randomizes effective target-rate
capacity down to `158.4 deg/s`. Loaded hardware can still be slower, so limiter
backlog and effort remain selection gates rather than assumed capabilities.

### Why all knees do not point the same way

V25b keeps front and rear sagittal poses opposed: front feet are swept forward
and rear feet rearward around the body center. With the current body geometry,
the `120 mm` front/rear hip spacing plus two `92 mm` opposed sweeps retain
roughly a `304 mm` fore/aft foot-center support footprint. Sweeping every leg
the same direction would cancel the two sweeps from the footprint calculation
and leave roughly the `120 mm` hip spacing. That would shift the rear-loaded
body toward one support edge and create a fixed pitch bias before the policy
acts.

Lee and Meek's primary study of directional leg compliance found that fore/hind
leg orientation changes braking, propulsion, and pitch behavior; it does not
support treating an all-same-direction layout as a free increase in forward
thrust. Their mechanism is not identical to this rigid servo linkage, so V25b
uses the conclusion conservatively: keep a large opposed support polygon, then
let the reference's stance sweep create rearward foot motion and forward body
propulsion. A future experiment may expose a small fore/aft stance-offset
command, but it must be selected by pitch, effort, slip, and straightness—not
appearance alone.

## Straight-path control and reward

V25b records the episode-start world heading and measures progress and lateral
motion in that fixed path frame. Relative heading error drives the existing
yaw-command observation through a bounded proportional outer loop:

| Setting | Value |
| --- | ---: |
| Heading proportional gain | `1.50 s^-1` |
| Maximum yaw correction | `0.20 rad/s` |
| Straight corridor half-width | `8 mm` |
| Lateral-velocity normalization floor | `0.010 m/s` |
| Cycle-averaged normalized lateral-speed cost | `6.0` |
| Lateral-displacement cost | `60.0` |
| Corridor-excess cost | `10.0` |
| Yaw-rate cost | `22.0` |
| Heading-error normalizer | `3 deg` |
| Heading-error cost | `8.0` |
| Aligned-progress reward | `6.0` |
| Aligned-progress lateral sigma | `0.006 m/s` |
| Aligned-progress heading sigma | `4 deg` |
| Velocity/instant/sustained progress rewards | `6.0 / 8.0 / 12.0` |
| Stall/backward/overspeed costs | `14.0 / 18.0 / 8.0` |
| Analytic gait-reference reward | `5.0` |

The line objective is deliberately stricter than an instantaneous body-frame
penalty. For every environment, V25b compares path-frame lateral displacement
with the displacement one complete command-dependent gait period earlier. The
resulting cycle-averaged lateral velocity cannot be hidden by alternating
left/right steps or by turning the body so its new X axis follows a curved
path. During the first incomplete cycle, the same term uses accumulated
lateral tracking displacement divided by elapsed time.

A stable sequential crawl still needs a small, periodic body transfer toward
the support polygon. V25b samples the exact analytic reference and subtracts
only its phase-locked, zero-mean lateral displacement and velocity from the
instantaneous corridor and lateral tracking terms. The `6 mm` lateral phase
shift therefore is not rewarded as drift, while any error that accumulates
across a full cycle remains penalized. Straight-aligned forward reward is also
attenuated by cycle-averaged lateral speed and absolute heading error.

The actor contract does not gain a new observation. Simulation and deployment
both convert relative yaw error into the existing requested-yaw channel. On the
Pi, the BNO085 game-rotation quaternion supplies relative yaw and the reference
heading resets when a walk starts. Game-rotation yaw does not use a magnetometer
and can drift, so this is a short-trial heading hold rather than global
navigation. AprilTag or other external pose feedback remains the preferred
later outer loop for long straight paths and cross-track correction.

The inherited smoothness costs stay active: action rate `0.18`, action second
difference `0.90`, joint acceleration `0.30`, body linear acceleration `0.55`,
and body angular acceleration `0.45`. V25b adds a `0.20` soft-effort cost above
`85%` of each randomized joint's available effort.

Pose imitation is limited to `5.0`, while world-path progress, stall, reverse
motion, and cycle straightness remain dominant. This guards against both the
old stationary bent-knee optimum and the opposite failure mode of earning
forward reward through a fast curved path.

## Investigation history and rejected runs

Every run below is diagnostic only and must not seed deployment:

- `2026-08-30_19-30-14_manual-headless` and
  `2026-08-30_19-35-29_manual-headless` were the first two V24 continuations.
  They were run before the simulator audit was complete and are rejected as
  invalid evidence. Their configuration included the XYZW/WXYZ root-quaternion
  error, the `4.526139 kg` versus `3.17514659 kg` mass mismatch, and an
  insufficient low-lift/low-transfer reference that let rear feet remain
  planted. The first also exposed a reward loophole in which a high gait-pose
  reward could dominate real progress.
- `2026-08-30_20-16-31_manual-headless` was the corrected-mass/quaternion
  nominal baseline. It moved forward, but left drift grew as training
  continued and rear-left stopped releasing reliably. No checkpoint from the
  run is selected. That failure motivated the current phase-sway compensation,
  strict full-cycle lateral scoring, smaller sagittal residual, and neutral
  Beta bootstrap instead of another raw V24 continuation.

The current V25b code is downstream of all three runs. A new nominal run is
required before any robust-phase training, ONNX export, video-selection claim,
or physical trial.

## Physical randomization contract

The ranges below are deliberately narrower than values used on large
torque-controlled research robots. They are engineering priors for this
`3.17514659 kg`, 60 Hz serial-position-servo platform and should be tightened
when controlled hardware measurements are available. The table lists every
targeted V25b physical domain in the current configuration.

| Quantity | Distribution used by V25b |
| --- | --- |
| Measured nominal assembled mass | `3.17514659 kg` (`7 lb`) |
| Explicit external rear pack | `0.523179545 kg` |
| Static dry-robot CAD mass scale | `0.650607608` |
| Nominal share during robust phase | `25%` of environments |
| Mirroring | complementary left/right values with randomized side assignment |
| Whole-robot mass/inertia factor | `0.96-1.04` |
| Rear-loaded base/payload-body factor | `0.955-1.055` |
| Effective randomized base factor | `0.9168-1.0972` |
| Base COM jitter, X/Y/Z | `+/-8 / +/-10 / +/-6 mm` |
| Common effort capacity | `0.75-1.00` |
| Common target-rate capacity | `0.75-1.00` |
| Independent joint effort factor | `0.88-1.00` |
| Independent joint target-rate factor | `0.88-1.00` |
| Effective common times joint capacity | `0.66-1.00` |
| Independent stiffness factor | `0.85-1.15` |
| Independent damping factor | `0.85-1.15` |
| Per-foot static friction | `0.55-0.90` |
| Per-foot dynamic friction | `0.35-0.70`, clamped below static |
| Abduction target-zero bias | `+/-1 deg` per joint |
| Flexion/knee target-zero bias | `+/-1.5 deg` per joint |
| Command delay | `0-1` control step (`0-16.67 ms`) |
| Reset roll/pitch | `+/-2 deg` |
| Reset yaw | `+/-5 deg` |
| Gyroscope bias | `+/-0.020 rad/s` per axis |
| Projected-gravity noise | `0.006` standard deviation per axis |
| Linear-acceleration noise | `0.020 g` standard deviation per axis |
| Persistent force X/Y/Z | up to `+/-0.10 / +/-0.40 / 0 N` |
| Persistent torque roll/pitch/yaw | up to `+/-0.025 / +/-0.015 / +/-0.040 N m` |

The static dry scale corrects the CAD ledger; it is applied once to dry links,
while the authored base reconstructs its scaled dry portion plus the explicit
rear pack. The `0.96-1.04` whole-robot factor then covers remaining total-mass
uncertainty. The base/payload-body factor is additionally applied to the
rear-loaded base, giving the tabled product range. Nominal environments use all
three factors at their nominal values.

The common actuator factors model battery/supply conditions. Independent
factors and target biases model motor, linkage, assembly, and calibration
asymmetry. Each
left/right joint or foot pair receives complementary samples around the range
midpoint, with a random choice of which side gets each value. This keeps the
distribution unbiased while still requiring unequal left/right corrections.
This is a symmetry treatment, not a hard same-action constraint: sequential
crawl phases and mirrored joint frames require different instantaneous signs.
The neutral Beta seed removes the old policy's preferred residual direction,
paired randomization removes a preferred weak side, and cycle scoring judges
whether the resulting motion is unbiased over a complete gait.

Nominal environments retain V24's padded-foot approximation: static/dynamic
friction `0.75/0.55`, restitution `0.01`, contact stiffness `7000`, and damping
`85`. Robust environments sample each foot separately inside the tabled ranges;
left/right samples are paired around the range midpoint to avoid a preferred
turn direction. The chosen friction envelope remains an engineering prior until
a simple pull test measures the actual Velcro-like pad and floor. V25b does not
yet train on rough terrain; broad unmeasured terrain randomization now could
make the policy unnecessarily conservative.

## Training workflow

Do not use `-Fresh`, and do not continue the learned V24 Beta action mean
unchanged. V25b uses a purpose-built transfer seed. It preserves V24
`model_3248` actor feature layers, replaces each alpha/beta output-row pair with
their average so the deterministic residual is exactly zero for every
observation, preserves state-dependent concentration, zeros the stale critic
output head, clears Adam moments, and restores the configured `7.5e-5` learning
rate. The generated checkpoint deliberately remains `model_3248.pt` so the
existing interrupted-run curriculum fallback remains valid. A `.bootstrap.json`
sidecar records source and destination SHA-256 hashes and the transformation.

Generate the neutral seed without launching Isaac Sim or training:

```powershell
& .\simulation\isaac\rl\parallel_walking\prepare_v25_bootstrap.ps1
```

The default destination is
`simulation/isaac/models/parallel-walking-v25-neutral-bootstrap/model_3248.pt`.
The generator is strict: it locks the selected V24 source to SHA-256
`e9c521fbd9f63ea0c9329bc3487a44be5f9dbc58530e9f7749eeba37635b37d2`,
checks checkpoint keys, tensor shapes/dtypes, optimizer parameter ordering, and
destination filename, and refuses replacement unless `-Force` is explicit.

First adapt the transferred V24 feature representation for exactly 350 PPO
updates with nominal physics while the lower stance, smooth reference, and
heading controller are new:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet robust-straight-low-stance-external-rear-payload `
  -Iterations 350 -NumEnvs 128 -Seed 2501 -V25Phase nominal `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v25-neutral-bootstrap\model_3248.pt
```

Do not automatically start the robust phase after iteration 350. First evaluate
the nominal checkpoints and select one that has positive progress, no growing
cycle drift, clean release/touchdown from all four legs, and acceptable support
and smoothness. Only then continue that explicit selected checkpoint with
robust randomization:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet robust-straight-low-stance-external-rear-payload `
  -Iterations 1250 -NumEnvs 128 -Seed 2502 -V25Phase robust `
  -Checkpoint <selected-nominal-model.pt>
```

The nominal phase explicitly fixes combined base mass/inertia scale, COM jitter,
common effort, and common target-rate scale to their nominal values; the
per-joint, friction, delay, sensor, and wrench domains are also inactive.

`-V25Phase auto` selects nominal below 350 completed V25b updates and robust
afterward at the start of each invocation; it does not switch domains midway
through one long job. Keep the initial job at 350 updates. The workflow writes
`model_N.pt.curriculum.json` beside every V25b checkpoint. That sidecar stores
V25b policy steps independently of the inherited absolute RSL-RL iteration
number; the `32xx-35xx` checkpoint filename is not itself a curriculum age.
If a sidecar is missing, the workflow falls back to `N - 3248` and warns. This
prevents the first V25 resume from jumping directly to the final command range.
Copy the curriculum sidecar with a checkpoint if it is moved outside its log
directory and may be used for further training.

## Zero-action reference comparisons

Reference-only evaluation forces all twelve policy actions to zero, so it tests
the analytic crawl without credit or blame going to inherited residuals. These
numbers are not trained-policy results and do not select a deployable model.

At a `0.015 m/s` command, three nominal episodes were run for each lateral
phase-shift value:

| Lateral phase shift | Mean forward speed | Reported lateral displacement |
| ---: | ---: | ---: |
| `0 mm` | `0.00861 m/s` | `3.93 mm` |
| `6 mm` | `0.00870 m/s` | `2.21 mm` |
| `12 mm` | `0.00907 m/s` | `15.45 mm` |

The `6 mm` reference is selected for V25b configuration because it gave the
best lateral compromise while retaining essentially the same forward speed.
The `12 mm` transfer was slightly faster but introduced much larger alternating
diagonal/yaw bias; maximizing reference-only speed is not the objective.

The current default swing lift is `20 mm`. A separate `25 mm` zero-action
candidate reached `0.01055 m/s`, but its three/four-foot support fraction fell
to `0.878`. That is below the provisional `0.90` support gate,
so `25 mm` is not selected. The `20 mm` default remains the starting reference;
lift can be reconsidered only with matched release, support, effort, and
straightness evidence.

## Evaluation and selection gates

Evaluate held-out seeds at `0.005`, `0.015`, `0.030`, and `0.040 m/s`.
Use at least ten uninterrupted 30-second episodes per command, including
nominal and randomized cases. Selection is based on the worst cases as well as
the mean; the highest checkpoint number is not automatically preferred.

```powershell
& .\simulation\isaac\rl\parallel_walking\evaluate_walking_sustained.ps1 `
  -CommandSet robust-straight-low-stance-external-rear-payload `
  -Checkpoint <candidate-model.pt> -ForwardSpeed 0.015 `
  -Seconds 30 -Episodes 10 -Seed 2511 -DomainMode nominal

& .\simulation\isaac\rl\parallel_walking\evaluate_walking_sustained.ps1 `
  -CommandSet robust-straight-low-stance-external-rear-payload `
  -Checkpoint <candidate-model.pt> -ForwardSpeed 0.015 `
  -Seconds 30 -Episodes 10 -Seed 2512 -DomainMode randomized
```

Repeat both modes with additional held-out seeds. A one-environment evaluator
keeps its sampled motor/material domain across episodes, so changing the seed
and explicitly selecting nominal versus randomized physics is necessary.

The evaluator reports path-frame forward/lateral displacement, lateral error
per forward meter, mean cycle-averaged lateral speed, mean/RMS heading error,
per-joint effort/rate factors, control delay, target biases, soft-effort-limit
occupancy, maximum effort, per-leg release/touchdowns, support, limiter backlog,
backward steps, and joint/body acceleration.

A candidate advances only if all of these provisional gates pass:

- zero falls and no sustained-stall episode across the held-out set;
- positive path-frame progress in every episode;
- at least one qualified swing/release and touchdown from every leg in every
  episode after startup; a planted rear-left leg is an automatic rejection;
- mean speed within `60-135%` of command at `0.015-0.040 m/s`;
- mean absolute lateral error no greater than `0.05 m` per forward meter, with
  no episode above `0.10 m/m`;
- no checkpoint-to-checkpoint growth in cycle-averaged lateral speed or
  monotonic left/right displacement hidden by phase sway;
- mean final heading error no greater than `5 deg`, with no episode above
  `10 deg`;
- at least `90%` three/four-foot support through `0.030 m/s`, and at least
  `80%` at `0.040 m/s`;
- backward-step fraction below `20%` at `0.005 m/s` and below `10%` at higher
  commands;
- mean target-limiter gap no greater than `0.02 rad`;
- soft-effort-limit exceedance below `10%` of joint-control samples;
- joint RMS acceleration no greater than `10 rad/s^2`, and no matched-command
  joint/body acceleration more than `15%` worse than V24 `model_3248`;
- no visual two-leg hopping, foot crossing, repeated shoe-edge loading, or
  high-frequency yaw correction in a recorded preview.

These are selection gates, not claimed results. No V25b checkpoint currently
passes them because no post-V25b-bootstrap run exists. If no future checkpoint
passes, keep V24 deployed and revise one identified mismatch at a time. Do not
relax all gates merely to obtain a faster video.

After simulation selection, the first physical run must use a charged supply,
support harness/spotter, emergency stop, automatic telemetry recording, and a
`0.005 m/s` command. Perform a stationary low-stance hold, then a 3-second walk,
then a 10-second walk. Battery tests come only after the power-supply trials are
stable.

## Export contract after selection

Do not export until a checkpoint passes the gates. The V25b sidecar must embed
the low neutral pose, smoothed reference table, and heading-hold parameters:

```powershell
& C:\isaacsim\python.bat `
  .\simulation\isaac\rl\parallel_walking\export_policy_onnx.py `
  --checkpoint <selected-model.pt> --output <selected-model.onnx> `
  --training-task Drobot-Commanded-Walk-Robust-Straight-Low-Stance-External-Rear-Payload-Direct `
  --training-profile robust-straight-low-stance-external-rear-payload `
  --gait-clock-mode speed_scaled --gait-pattern distributed_support_crawl `
  --gait-duty-factor 0.8625 --gait-phase-offsets 0.07 0.32 0.57 0.82 `
  --gait-standstill-deadband-m-s 0.002 `
  --gait-speed-min-m-s 0.005 --gait-speed-max-m-s 0.040 `
  --gait-frequency-min-hz 0.12 --gait-frequency-max-hz 0.62 `
  --gait-stride-scale-min 1.0 --action-mode gait_residual `
  --residual-action-scale 0.05 `
  --residual-action-scales 0.20 0.20 0.20 0.20 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 `
  --reference-sample-count 2048 `
  --reference-start-ramp-s 1.5 --reference-stride-m 0.045 `
  --reference-lift-m 0.020 --reference-weight-shift-forward-m 0.010 `
  --reference-weight-shift-lateral-m 0.006 `
  --reference-stance-fore-aft-m 0.092 `
  --reference-stance-down-m 0.3216749408 `
  --reference-smooth-support-push `
  --neutral-sagittal-angle-rad 0.6130583087 `
  --heading-hold-enabled --heading-hold-kp-s 1.5 `
  --heading-hold-max-correction-rad-s 0.20 `
  --target-velocity-limit-rad-s 4.1887902048 --max-target-step-deg 4 `
  --forward-speed-min-m-s 0.005 --forward-speed-max-m-s 0.040 `
  --recommended-forward-speed-m-s 0.005
```

## Primary technical sources

- Tan et al., [Sim-to-Real: Learning Agile Locomotion for Quadruped Robots](https://arxiv.org/abs/1804.10332): actuator and latency modeling, measured dynamics randomization, and a reference gait with learned residual correction.
- Hwangbo et al., [Learning Agile and Dynamic Motor Skills for Legged Robots](https://arxiv.org/abs/1901.08652): COM/kinematic uncertainty, actuator-history modeling, sensor noise, and smoothness costs.
- Kumar et al., [RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034): per-motor strength, payload/friction variation, and adaptation from recent state/action history.
- Margolis and Agrawal, [Walk These Ways](https://proceedings.mlr.press/v205/margolis23a.html): policies conditioned on cadence, body height, stance, and swing parameters, with progress and auxiliary locomotion objectives composed together.
- Yang et al., [Fast and Efficient Locomotion via Learned Gait Transitions](https://proceedings.mlr.press/v164/yang22d.html): walking, trotting, and flying-trot regimes emerge at different speeds. V25b adopts speed-dependent cadence but stays in a supported crawl regime.
- Mittal et al., [Symmetry Considerations for Learning Task Symmetric Robot Policies](https://arxiv.org/abs/2403.04359): mirror data augmentation/loss can reduce asymmetric artifacts without forcing mechanically incorrect identical outputs.
- Xie et al., [Dynamics Randomization Revisited](https://arxiv.org/abs/2011.02404): randomization is neither automatically necessary nor sufficient; other controller choices matter and unnecessary ranges can obscure the real transfer issue.
- Jain et al., [Hierarchical Reinforcement Learning for Quadruped Locomotion](https://arxiv.org/abs/1905.08926): real path following uses a low-rate higher-level pose/path signal above fast onboard locomotion control.
- Lee and Meek, [Directionally compliant legs influence the intrinsic pitch behaviour of a trotting quadruped](https://doi.org/10.1098/rspb.2004.3014): fore/hind leg direction changes propulsion, braking, and pitch behavior.
- Zeng et al., [Leg Trajectory Planning for Quadruped Robots with High-Speed Trot Gait](https://doi.org/10.3390/app9071508): continuous spline/Bezier foot paths reduce trajectory acceleration and endpoint impact.
- Feetech, [STS3215-C018 product page](https://www.feetech.cn/en/525603.html) and [12 V specification](https://cdn.robotshop.com/media/F/Fit/RB-Fit-155/pdf/feetech_12v_30kg_cm_magnetic_encoding_servo_sts321_specification_pdf.pdf): specify `0.222 s/60 deg` no-load speed, `10 kg cm` rated torque, `30 kg cm` stall torque, and a maximum `1 ms` position-update interval. These bound, but do not guarantee, motion under load.
- Feetech, [communication protocol manual](https://files.seeedstudio.com/wiki/robotics/Actuator/feetech/Communication_Protocol_Manual.pdf): documents the half-duplex serial protocol, default `1 Mbps` communication rate, and present-position register at addresses `56-57`.

The servo's `1 ms` internal update specification does not mean a twelve-servo
USB/half-duplex control loop can obtain twelve fresh positions every millisecond;
request/response bytes, return delay, adapter latency, and software scheduling
still apply. It does show that the earlier `15 Hz` feedback behavior was a
software/bus implementation limit, not an STS3215 position-register limit.

The papers use different robots, actuators, and control rates. Their principles
motivate V25b, but their absolute speed and randomization ranges are not copied
onto this hardware.

## Known limitations and likely V26 work

- V25b has no explicit history encoder. If asymmetric dynamics remain hard to
  infer from instantaneous joint/IMU state, add a short observation history or
  RMA-style adaptation module rather than widening randomization indefinitely.
- Relative BNO085 game-rotation yaw drifts. Use AprilTag/visual pose or another
  global heading source for long path tracking.
- Small link-length mismatch, pad compliance, real servo deadband/backlash, and
  measured voltage are not yet explicit actor inputs/randomizations; the
  per-foot friction range is still unmeasured.
- A walking-trot transition should be a separately gated policy/config after
  the supported crawl reaches the `0.040 m/s` straightness and effort gates.
