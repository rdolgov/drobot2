# V30 symmetry-gated robust straight crawl

## Status

V30 is an implemented **research profile**, not a release model. Two nominal
continuations were trained and evaluated on 2026-08-31. The best diagnostic
checkpoint is substantially straighter and remains smooth and fall-free in
nominal simulation, but it fails the exact-contact, all-four-release, support,
and sustained-effort gates. On 2026-08-31, it was exported and installed on the
Raspberry Pi at the user's explicit request for a guarded physical trial. This
does not change its rejected/research status, and V24 remains installed as the
immediate rollback policy.

Identifiers:

- task: `Drobot-Commanded-Walk-Symmetry-Gated-Robust-Straight-Crawl-External-Rear-Payload-Direct`;
- command set: `symmetry-gated-robust-straight-crawl-external-rear-payload`;
- experiment: `drobot_commanded_walk_v30_symmetry_gated_robust_straight_crawl_external_rear_payload`.

V30's purpose is narrow: remove systematic left/right and leg-order bias while
retaining a stable one-foot-at-a-time crawl, then make that behavior robust to
the measured rear pack, an imperfectly level assembly, unequal servos, changing
battery voltage, and the new Velcro-like shoe pads. It is not a request to make
the robot dynamic at any cost.

## Implemented V30 configuration

The implementation preserves the deployable 50-value actor observation and
the low opposed stance. It adds:

- speed-normalized cycle lateral error and a `0.20 + 0.80 * alignment` gate on
  positive tracking/progress rewards; reverse progress is never discounted;
- unit-Huber corridor and heading costs, a `10 mm` corridor, `5 deg` heading
  normalization, and a `4 deg` heading-alignment sigma;
- left/right data augmentation during nominal training, with mirror loss off;
- independent rather than artificially complementary actuator/zero-offset
  asymmetries for the later robust stage;
- `+/-15 mm` lateral rear-payload COM jitter, `+/-0.040 N m` roll torque,
  `+/-1.5 deg` abduction-zero error, and a correlated `0.88-1.05` supply
  effort/rate proxy;
- common-plus-per-foot Velcro-like traction randomization inherited from V28;
- a moderate `-12 mm` empirical lateral reference correction;
- `25 mm` forward load transfer for rear-leg swings, selected after controlled
  `20/25/30 mm` ablations;
- a cadence ceiling of `0.85 Hz` and matched analytic command ceiling of
  `0.039 m/s`, without lowering or lengthening the already constrained stance.

Historical V25-V29 profiles keep their prior reward and reference behavior.

## 2026-08-31 training and selection result

The first nominal run continued V29 `model_4572.pt` for 750 PPO updates with
4,096 environments and seed 3017:

```text
logs/rsl_rl/drobot_commanded_walk_v30_symmetry_gated_robust_straight_crawl_external_rear_payload/
  2026-08-31_01-15-14_manual-headless/
```

At 0.015 m/s, its best straightness checkpoint was `model_4825.pt`: zero
falls, `0.01432 m/s` path speed, `0.01236 m/m` lateral error, and
`8.60 rad/s2` RMS joint acceleration. It still achieved only `0.5922` exact
scheduled contact, `0.8667` three/four-foot support, and `0.75` all-four
release cycles. Later checkpoints improved exact contact only to about `0.61`
while lateral error regressed to `0.041-0.050 m/m`.

The rear-swing transfer ablation on `model_4825.pt` compared 20, 25, and 30 mm.
The 25 mm reference improved exact contact to `0.6161`, support to `0.8878`,
and speed to `0.01469 m/s`, with `9.09 rad/s2` joint acceleration. The 30 mm
case reached `0.6194` exact contact and `0.8917` support but increased meaningful
reverse samples to `0.3094` and acceleration to `10.76 rad/s2`. Therefore the
implementation selects 25 mm, not the largest forward shift.

The controlled 25 mm continuation started from first-run `model_4825.pt` and
trained for 500 PPO updates with 4,096 environments and seed 3029:

```text
logs/rsl_rl/drobot_commanded_walk_v30_symmetry_gated_robust_straight_crawl_external_rear_payload/
  2026-08-31_07-15-11_manual-headless/
```

Deterministic `model_5000.pt` was the best checkpoint. At seed 4921 it produced
zero falls, `0.01431 m/s` speed, `0.02815 m/m` lateral error, `8.56 rad/s2`
joint acceleration, and `0.1411` meaningful reverse samples. Held-out seeds
4937 and 4951 remained fall-free at `0.01480-0.01514 m/s`,
`0.02009-0.03615 m/m` lateral error, and `8.49-8.80 rad/s2` acceleration.

The rejection is unambiguous: exact scheduled contact remained
`0.6056-0.6183`, support about `0.8806-0.8844`, and all-four-cycle success
`0.75-0.8889`. The dominant error is front-left anchor loss during the
rear-right swing (`0.226-0.302` of that phase); the smaller remaining error is
front-right anchor loss during rear-left swing. Rated-effort exceedance also
remained `0.428-0.439` of joint timesteps. Robust continuation was deliberately
stopped. Export and Pi deployment were later performed only for the explicitly
requested guarded physical trial.

The 10-second diagnostic preview is:

```text
logs/rsl_rl/drobot_commanded_walk_v30_symmetry_gated_robust_straight_crawl_external_rear_payload/
  2026-08-31_07-15-11_manual-headless/videos/play/rl-video-step-0.mp4
```

This result validates the straightness/reward redesign but not the complete
controller. The next iteration should alter rear-right load transfer or phase
support geometry specifically; widening every randomization range or merely
training this branch longer is not justified by the plateau.

## Evidence leading to V30

The repository already contains several useful failures. V30 should change the
credit assignment and robustness curriculum, not merely increase every reward
coefficient.

| Profile | Useful result | Reason it is not a release |
| --- | --- | --- |
| V25/V26 | Corrected the mass ledger, quaternion convention, world-path scoring, and lower opposed stance. | A scheduled rear-left shoe often remained loaded; exact mirror loss did not solve it. |
| V27 | Permitted asymmetric residuals and provided smooth force-based unloading discovery. | `model_3549.pt` still failed deterministic rear-foot release. A soft force score alone accepted partial unloading. |
| V28 | Reached about `0.0105-0.01095 m/s` at a `0.015 m/s` command with zero falls and low joint acceleration. | `model_3774.pt` had only about `0.447` exact scheduled-contact identity and `0.8128` three/four-foot support. The count-only support reward could accept the wrong airborne foot. |
| V29 Branch A/B | Added identity-aware contact scoring and a smooth discovery gate. | Exact topology remained about `0.426-0.444`; one branch also drifted `0.085-0.105 m/m`. |
| V29 Branch C | Improved exact topology to `0.5511` at `model_4073.pt`, with `0.01489 m/s` speed and no falls. | All-four cycles were only `0.8333`, support `0.8189`, lateral error `0.0809 m/m`, and meaningful reverse motion `0.2167`. The stance offset and final-leg cycle event introduced avoidable asymmetry. |
| V29 symmetric continuation | The run directory contains checkpoints through `model_4572.pt`. | There is no recorded held-out acceptance report for that endpoint in this document. Training return or checkpoint existence is not acceptance. |

The real robot adds a second signal: with the rear pack and adhesive Velcro-like
pads installed, the observed policy can drift left, wobble, or even move
backward. That does not prove one cause. Plausible contributors include lateral
COM error, a slightly tilted body or IMU, unequal joint zeros or servo rates,
different pad contact heights, direction-dependent pad friction, and voltage
sag under load. V30 must expose those causes independently enough that a good
training return cannot hide which assumption produced it.

## Mechanical decisions retained

### Keep the lateral-sequence crawl

Retain the physical swing order:

```text
RR -> FR -> RL -> FL
```

This is a cyclic rotation of a lateral-sequence four-beat crawl. At low speed,
one named foot swings while the other three form the support polygon. The order
also matches the hardcoded crawl that has been substantially more stable on the
real robot than the current RL policy. The V28/V29 failure was not evidence that
two-leg support is preferable; it was evidence that a count-only reward could
mistake the wrong three feet for the intended support triangle.

Classic creeping-gait analysis shows why a one-foot-at-a-time pattern is the
appropriate starting regime when static stability matters. See McGhee and
Frank, [On the stability properties of quadruped creeping gaits](https://doi.org/10.1016/0025-5564(68)90090-4).
The exact `RR -> FR -> RL -> FL` rotation remains a robot-specific engineering
choice, validated against this robot's rear payload and prior hardcoded walk.

V30 should randomize the starting quarter-cycle uniformly. It must apply the
same reward definition and coefficient to each physical leg. No single event
reward may be paid only after the fixed final `FL` phase.

### Keep opposed front/rear leg geometry

Retain the `92 mm` outward sagittal sweep: front feet forward of their hip axes,
rear feet rearward. With approximately `120 mm` between front and rear hip axes,
the nominal fore/aft contact span is:

```text
120 + 92 + 92 = 304 mm
```

Pointing all four sagittal legs the same direction translates the front and rear
contacts together and trends toward the roughly `120 mm` hip-axis spacing. It
therefore discards about `184 mm` of nominal support span at the moment the
rear-mounted battery most needs pitch margin. Same-direction knees also do not
guarantee forward thrust: forward impulse comes from a planted contact sweeping
backward relative to the body, not from the visual direction of the knee.

Directional leg orientation can change pitch, braking, and propulsion, but it
is not a free thrust gain; see Lee and Meek,
[Directionally compliant legs influence the intrinsic pitch behaviour of a trotting quadruped](https://doi.org/10.1098/rspb.2004.3014).
A future dynamic trot may be a separate mode after the crawl passes its gates.
It should not replace the supported crawl in V30.

### The present lower stance is already near the useful joint envelope

V30 should initially preserve the current geometry:

| Quantity | Current value |
| --- | ---: |
| Opposed foot sweep | `92 mm` |
| Stance-down dimension | `0.3216749408 m` |
| Approximate base height | `0.3673 m` |
| Difference from the earlier 80 mm stance | about `7.67 mm` lower |
| Body pitch target | `2 deg` nose-down |
| Reference stride / lift | `46 / 24 mm` |
| Command-scaled cadence | `0.12-0.85 Hz` |
| Analytic speed ceiling | `46 mm * 0.85 Hz = 0.0391 m/s` |
| Isaac soft hip limit | approximately `+/-57 deg` |
| Worst sampled reference hip target | approximately `-55.90 deg` |
| Remaining reference-only hip margin | approximately `1.10 deg` |

The stance is visibly lower than the earlier profile, but it is not safe to
keep lowering it blindly. The 46/24 mm reference already approaches the hip
soft limit, and V29's hip residual can request about another `2.06 deg` before
the final clamp. A deeper crouch also raises knee/hip holding torque and heat,
which is especially important with the rear payload and low battery headroom.

For V30, obtain a more assertive look and stronger push through correct load
transfer, knee clearance, and cadence rather than more hip sweep. A later
reference-only sweep may compare `90/92/94 mm` stance and small body-height
offsets, but it must reject any geometry that increases target clipping,
limiter backlog, above-rated effort dwell, or shoe-edge loading. If a lower
stance is selected, stride or hip residual authority must be reduced to restore
joint margin.

## V30 reward redesign

### Separate discovery signals from hard acceptance

PPO needs a graded signal while a loaded shoe begins to unload, but a smooth
score must never redefine a partly loaded shoe as a successful step. During the
scheduled swing of foot `i`, use a discovery quality such as:

```text
q_release_i = exp(-(F_i / sigma_release)^2)
```

Start with `sigma_release = 2 N`. A zero-load shoe approaches one; a heavily
loaded shoe approaches zero. Use the signed local term `2*q_release_i - 1` so a
firmly loaded scheduled swing is penalized rather than merely receiving no
bonus. The hard evaluator remains unchanged: the named shoe must be below
`1 N` for at least five consecutive `60 Hz` updates, about `83 ms`.

For each scheduled stance foot, compute a smooth contact-quality term around
the same contact threshold. The phase quality is the weakest required element:

```text
q_phase = min(q_release_of_named_swing,
              q_stance_of_each_named_anchor)
```

During transfer/plant/settle, all four stance qualities participate. Positive
progress may use a small discovery floor such as
`0.02 + 0.98*q_phase`. Backward, lateral, yaw, effort, slip, impact, and
smoothness penalties must **not** be multiplied by this floor; otherwise the
policy could hide bad motion by intentionally breaking contact.

### Gate symmetrically over all four legs

The previous current-cycle completion event was inevitably paid after the last
fixed `FL` swing and gave that phase different temporal credit. V30 should use
three complementary signals instead:

1. Apply the same local `q_phase` gate during every leg's quarter-cycle.
2. Accumulate one normalized quality `Q_RR`, `Q_FR`, `Q_RL`, and `Q_FL` over a
   completed cycle. Penalize the spread between the best and worst leg.
3. Form a balanced cycle score from the minimum or a narrow soft minimum. Apply
   that completed-cycle score uniformly to the next cycle, while retaining the
   current local phase gate so an alternating good/bad-cycle exploit cannot
   receive full current credit.

This is reward symmetry, not a command that left and right actuators emit
identical numbers. The robot may need asymmetric actions to cancel an
asymmetric payload, pad, joint zero, or servo. V26 showed that a strong mirror
loss can preserve the wrong contact. V30 should therefore avoid a hard mirror
loss. It should instead use balanced phase starts, left/right-mirrored domain
pairs, identical per-leg rewards, and held-out signed-drift tests.

Symmetry-aware RL research supports augmentation or equivariant priors when the
task and dynamics actually share that symmetry; see Mittal et al.,
[Symmetry Considerations for Learning Task Symmetric Robot Policies](https://arxiv.org/abs/2403.04359).
The centered nominal model is approximately symmetric, but each randomized or
physical robot need not be. That distinction is why V30 uses symmetry in the
training distribution and score, not an unconditional equality constraint on
actions.

### Reward straight travel without teaching the robot to stand still

Measure path motion in the episode-start world frame. At each complete gait
cycle, calculate forward displacement `dx`, lateral displacement `dy`, and
heading change `dpsi`. Positive progress should be aligned with the original
line and gated by valid four-leg contact. Penalize:

- cycle-net lateral displacement;
- accumulated corridor excess beyond `5 mm`;
- cycle-net yaw and heading error;
- backward displacement and meaningful reverse samples;
- asymmetric left/right cycle quality;
- sustained stall even if lateral error is nearly zero.

The known `6 mm` analytic side-to-side load transfer is a within-cycle motion.
Evaluate net line error at matched cycle phase, or subtract the analytic phase
offset, so the reward does not suppress the weight shift needed to lift a foot.
Never call a nearly stationary policy "straight": speed, positive distance,
all-four release, and support topology remain simultaneous hard gates.

The current 50-value actor observes the heading-corrected yaw command, IMU,
joint state, and previous action, but not global lateral position. World-path
reward can select a low-bias open-loop gait and the heading loop can correct
yaw. It cannot make the policy recover from a pure sideways translation that produces no observable
tilt or yaw. Long-path return-to-line behavior needs an AprilTag, visual
odometry, or another pose source in a slower outer steering loop.

### Keep smoothness and actuator headroom explicit

Retain action-rate, action-second-difference, joint-acceleration, body linear
and angular acceleration, support-foot slip, and touchdown-impact terms. These
are the current practical anti-jerk signals. Do not gain speed by relaxing all
of them together. Increase the speed curriculum mainly through cadence up to
the `0.0391 m/s` analytic ceiling.

Treat the documented `10 kg cm` (`0.980665 N m`) rating as the beginning of a
sustained-effort cost, not the hard instantaneous limit. The current
`2.6477955 N m` transient simulation cap is 90% of the documented `30 kg cm`
stall figure and is only an engineering approximation. A simulation torque
cap is not a thermal or current model; per-joint mean, RMS, peak, and
above-rated dwell must remain evaluation outputs.

## Targeted physical randomization

Randomization begins only after a nominal policy passes. Keep at least 25% of
robust environments at the exact nominal model so robustness cannot erase the
working reference. Generate left/right perturbations in mirrored pairs where
the parameter has a meaningful mirror, and use several evaluator processes or
seeds because one process can retain an episode-consistent hardware draw.

| Domain | Proposed V30 distribution or experiment | Reason |
| --- | --- | --- |
| Dry robot mass/inertia | `0.96-1.04` of corrected dry model | Printed-part and fastener uncertainty without returning to the incorrect fully-solid CAD mass. |
| Rear payload mass | `0.955-1.055` of `0.523179545 kg` | Battery, holder, lid, wire, and measurement uncertainty. |
| Rear payload COM | `+/-8 / +/-15 / +/-6 mm` in X/Y/Z | Rear placement and lateral mounting error can create pitch or left/right bias. |
| Common voltage/drive factor | correlated effort and rate, `0.88-1.05` around the 12 V model | Approximate a charged-to-lower-headroom 3S supply without treating unsafe undervoltage as locomotion. |
| Within-episode voltage sag | later, slow `0-8%` common droop plus bounded effort-correlated transient droop | A static draw cannot represent a pack that sags during a high-load support phase. Add only after telemetry bounds it. |
| Individual servo effort | `0.90-1.05` | Motor/gear/controller variation. |
| Individual servo rate | `0.88-1.05` | A delayed leg can distort contact timing even when torque is adequate. |
| Stiffness/damping | `0.85-1.15` per joint | Closed-loop response mismatch. |
| Joint-zero bias | `+/-1.5 deg` abduction and hip/knee | Assembly and horn indexing error. |
| Command delay | `0-1` control update | Packet/USB timing variation without recreating catch-up bursts. |
| Velcro-like pad friction | common static/dynamic `0.40-0.90 / 0.25-0.70`, then per-foot multiplier `0.85-1.15` | Common floor/pad material plus wear or installation asymmetry. |
| Pad height/compliance | small measured per-foot contact-height and compliance range | A pad can change load transfer before gross slip appears. Do not guess a wide range. |
| Reset attitude | roll/pitch `+/-2 deg`, yaw `+/-5 deg` | Small initial placement and body tilt. |
| Fixed IMU mounting error | proposed roll/pitch bias up to about `+/-1 deg`, after measurement | Distinguishes a tilted sensor from a tilted chassis. |
| Ground plane tilt | proposed paired roll slope up to `+/-1.5 deg` and pitch slope up to `+/-1 deg` | Tests whether the policy responds to observable tilt rather than memorizing a left correction. |
| Persistent body wrench | force `+/-0.10 / +/-0.40 / 0 N`; torque `+/-0.040 / +/-0.015 / +/-0.040 N m` | Cable drag, shifted wiring, and unmodeled assembly load. |
| IMU noise/bias | gyro `+/-0.020 rad/s`, projected gravity `0.006`, acceleration `0.020 g` | Bounded sensor error. |

These ranges are priors, not measurements. Narrow them when real data are
available. [Dynamics Randomization Revisited](https://arxiv.org/abs/2011.02404)
shows why wider randomization is not automatically better: an unnecessarily
broad domain can hide a bad controller or make the optimization problem harder.

### Velcro-like traction caveat

The installed material is an adhesive Velcro-like pad, not foam. It can change
static friction, sliding friction, contact height, edge compliance, and possibly
fore/aft versus lateral grip. A single Coulomb coefficient cannot capture all
of that. The present common-plus-per-foot friction ranges are reasonable
discovery priors, but V30 should not claim traction robustness until a simple
tilt or horizontal pull test is repeated for fore/aft and lateral directions on
the actual floor. If left and right pads differ, replace them or measure the
difference before asking the policy to compensate a large defect.

### Battery voltage, torque, and charge caveat

Low charge does **not** make the battery lighter or move its COM. It reduces
electrical headroom and can reduce loaded servo speed or available torque,
increase phase lag, and cause controller/bus resets if voltage sags far enough.
The Feetech STS3215-C018 page specifies `0.222 s/60 deg`, `10 kg cm` rated
torque, `30 kg cm` stall torque, and `2.7 A` stall current at `12 V`; it also
lists input-voltage, current, load, speed, and position feedback. See the
[manufacturer product page](https://www.feetech.cn/en/525603.html).

The correlated rate/effort factor is therefore a proxy, not a calibrated motor
curve. The `11.0-12.6 V` interpretation around the 12 V nominal model is an
engineering training envelope, not a license to operate at any voltage within
the servo's broad electronics range. Pack-level voltage under load is not an
exact state-of-charge measurement. Hardware must retain an independent,
pack-appropriate low-voltage stop. Training should never reward walking through
unsafe undervoltage, brownout, or sustained stall.

Record voltage and current with joint position, target, temperature, IMU, and
control-loop timing. Compare a charged bench supply and battery at the same
command before attributing a fall to balance. A battery-only failure accompanied
by voltage droop and target lag points to electrical/actuator mismatch; a
similar failure at steady voltage points more strongly toward geometry,
traction, contact timing, or policy bias.

## Staged training plan

### Stage 0: freeze contracts and establish reference feasibility

Before PPO, evaluate the analytic reference with zero residual at `0.005`,
`0.015`, `0.030`, and `0.039 m/s`. Confirm:

- the actual scheduled-contact cache and reward use the same controller tick;
- each named foot releases while the other named feet support;
- no nominal target exceeds the soft joint envelope;
- target-limiter backlog stays below the eventual gate;
- no common stance offset silently cancels the `2 deg` body-pitch geometry;
- per-joint effort telemetry distinguishes sustainable and transient load.

If the reference cannot meet basic topology, fix the trajectory or load
transfer before training. PPO should not spend millions of samples undoing a
known kinematic inconsistency.

### Stage 1: nominal symmetry-gated crawl

Train with physical randomization disabled, uniformly randomized quarter-cycle
starts, no final-leg event, and no hard action mirror loss. Begin at
`0.008-0.020 m/s`; expand gradually to `0.039 m/s` only after contact topology
improves. Screen deterministic actor means frequently. A short A/B can compare
a neutral V24/V25-compatible bootstrap against the best held-out V29 checkpoint;
do not assume the later checkpoint is better merely because its iteration
number is larger.

Stop and revise one identified term if exact topology, per-leg balance, or
lateral error fails to improve across several screens. Do not compensate by
simultaneously increasing contact, speed, and straightness weights.

### Stage 2: held-out nominal selection

Evaluate several late checkpoints, not only the endpoint, with deterministic
actor means at all four speeds and multiple seeds. Use complete gait cycles and
matched phase endpoints for line metrics. Select a checkpoint only if every
nominal hard gate below passes.

### Stage 3: targeted robust continuation

Continue the selected nominal checkpoint with the targeted distributions above
and at least 25% exact nominal environments. Introduce uncertainty in groups:

1. mass, payload COM, reset tilt, and joint-zero bias;
2. common voltage/rate/effort and individual servo response;
3. common and per-foot traction, measured pad height/compliance, and delay;
4. only then bounded sensor bias, slope, and persistent wrench.

Keep a regression panel of exact nominal environments through every stage. If a
new group destroys nominal topology, back out that group and identify whether
the range or controller observation is inadequate.

### Stage 4: held-out domain and corner evaluation

Use random seeds and manually selected corners not seen in training:

- left and right COM offsets as a mirrored pair;
- left-weak/right-strong and right-weak/left-strong servo pairs;
- high/mid/lower-safe voltage factors;
- high/low common traction plus one lower-traction foot at a time;
- positive/negative roll slope and IMU mounting bias;
- nominal physics as a mandatory regression case.

If the 50-value instantaneous actor succeeds nominally but cannot respond to
persistent servo/friction domains, stop broadening randomization. Add a short
proprioceptive history or RMA-style adaptation latent in a separately versioned
policy/runtime. RMA specifically motivates inferring payload, friction, and
motor differences from recent state/action history; see Kumar et al.,
[Rapid Motor Adaptation for Legged Robots](https://www.roboticsproceedings.org/rss17/p011.pdf).

### Stage 5: visual and guarded hardware validation

Record video only after numerical gates pass. Inspect the named swing foot,
three-foot support triangle, shoe-edge loading, toe scuff, hopping, body yaw,
and visible action jerk. Then use a harness or spotter, emergency stop, automatic
telemetry recording, and a charged bench supply:

1. center stance and low-stance hold;
2. `0.005 m/s` for three seconds;
3. `0.005 m/s` for ten seconds;
4. `0.015 m/s` for three and then ten seconds;
5. only after supply success, repeat with a fully charged battery;
6. repeat at lower but still safe battery voltage only with voltage/current
   telemetry and thermal limits active.

Stopping must return to the stable centered stance. A browser or telemetry
dropout must not be treated as evidence that a gait is safe.

## Acceptance gates

These are simultaneous gates, not values to trade away for a better-looking
video.

| Category | Required before robust training or deployment |
| --- | --- |
| Falls/stalls | Zero falls, nonfinite states, and sustained-stall episodes across the held-out set. |
| Progress | Positive path-frame progress in every moving episode; mean speed `60-135%` of command from `0.015-0.039 m/s`. |
| Four-leg operation | `100%` complete all-four release cycles after startup; every physical leg passes the `<1 N` for five-tick release criterion. |
| Contact identity | Mean exact scheduled-contact topology at least `0.90`, no leg/phase below `0.85`. |
| Support | At least `0.90` three/four-foot support through `0.030 m/s` and `0.80` at `0.039 m/s`; no repeated two-leg hopping. |
| Straightness | Mean absolute lateral error at most `0.05 m/m`, no episode above `0.10 m/m`; mean final heading error at most `5 deg`, no episode above `10 deg`. |
| Signed bias | No monotonic left/right drift hidden by zero mean across seeds; mirrored domain pairs must each pass individually, with aggregate signed lateral bias no more than `0.02 m/m`. |
| Reverse motion | Below `20%` meaningful reverse samples at `0.005 m/s` and below `10%` at higher commands. |
| Target tracking | Mean target-limiter gap at most `0.02 rad`; no repeated soft-limit clipping. |
| Effort | Above-rated effort occupancy below `10%` of joint-control samples, with per-joint mean/RMS/peak reviewed and no sustained near-stall hold. |
| Smoothness | Joint RMS acceleration at most `10 rad/s2`; matched-command joint/body acceleration no more than `15%` worse than V24 `model_3248.pt`. |
| Visual | No foot crossing, repeated shoe-edge loading, toe dragging, yaw chatter, collapse on stop, or visible high-frequency jerk. |
| Robust domains | Every nominal gate passes again; every selected voltage, traction, COM, tilt, and servo corner passes the same falls, topology, support, straightness, effort, and smoothness gates. |

For a 20-second `0.015 m/s` trial, `0.05 m/m` corresponds to roughly `15 mm`
lateral error over the commanded `0.30 m` path. Report both meters-per-meter and
absolute displacement so a very slow policy cannot make the ratio look good.

## Deployment rule

Do not promote V30 as the release default merely because training completes.
Promotion requires:

1. a selected checkpoint and reproducible run metadata;
2. complete deterministic nominal and randomized evaluation reports;
3. a reviewed simulation video;
4. an ONNX parity check against the selected checkpoint;
5. a guarded bench-supply hardware trial with telemetry;
6. an explicit human decision to promote the model.

V30 therefore remains a research artifact. The user-requested guarded trial is
installed at
`onboard/models/parallel-walking-v30-symmetry-gated-robust-straight-crawl/model_5000.onnx`
(SHA-256 `c93e9e58349326a56fa5846c72fd0c4ba8da3babf12dc2ce2bebdaf2fe01d564`).
V24 `model_3248.onnx` remains on the Pi for immediate rollback.

## Primary technical sources

- Tan et al., [Sim-to-Real: Learning Agile Locomotion for Quadruped Robots](https://roboticsproceedings.org/rss14/p10.pdf): actuator/latency modeling, reference motion, dynamics randomization, and staged real-hardware transfer.
- Hwangbo et al., [Learning Agile and Dynamic Motor Skills for Legged Robots](https://arxiv.org/abs/1901.08652): actuator-history modeling, COM/kinematic uncertainty, sensor noise, and smoothness-aware learned locomotion.
- Kumar et al., [RMA: Rapid Motor Adaptation for Legged Robots](https://www.roboticsproceedings.org/rss17/p011.pdf): online inference of changing payload, friction, and motor/terrain conditions from recent proprioceptive behavior.
- Xie et al., [Dynamics Randomization Revisited](https://arxiv.org/abs/2011.02404): evidence that targeted controller and model choices matter more than indiscriminately widening every randomization range.
- Mittal et al., [Symmetry Considerations for Learning Task Symmetric Robot Policies](https://arxiv.org/abs/2403.04359): symmetry-aware data augmentation and loss design, with the important requirement that the task transformation be valid.
- Margolis and Agrawal, [Walk These Ways](https://proceedings.mlr.press/v205/margolis23a.html): independent conditioning of cadence, body height, stance, and foot-swing characteristics.
- Shi et al., [Terrain-Aware Quadrupedal Locomotion via Reinforcement Learning](https://arxiv.org/abs/2310.04675): a parameterized trajectory generator combined with learned correction.
- McGhee and Frank, [On the stability properties of quadruped creeping gaits](https://doi.org/10.1016/0025-5564(68)90090-4): static-stability rationale for low-speed one-foot-at-a-time creeping gaits.
- Lee and Meek, [Directionally compliant legs influence the intrinsic pitch behaviour of a trotting quadruped](https://doi.org/10.1098/rspb.2004.3014): evidence that leg orientation changes pitch, braking, and propulsion rather than acting as a simple thrust switch.
- Feetech, [STS3215-C018 product specification](https://www.feetech.cn/en/525603.html) and [communication protocol](https://files.seeedstudio.com/wiki/robotics/Actuator/feetech/Communication_Protocol_Manual.pdf): 12 V speed/torque/current figures, feedback fields, serial protocol, and position register details.

The cited robots, actuators, and control rates differ from Drobot. Their methods
motivate the design; their numeric ranges are not copied as proof that this
hardware will transfer safely.
