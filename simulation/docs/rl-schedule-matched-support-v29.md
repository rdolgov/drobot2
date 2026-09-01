# V29 schedule-matched support straight crawl

## Status

V29 is an experimental Isaac Lab continuation that corrects a contact-topology
loophole found during deterministic V28 evaluation. It keeps the low opposed
stance, rear-payload model, straight-path objective, four-leg release contract,
and anti-jerk terms, but now rewards the **identities** of the contacting feet
rather than only counting how many feet touch the ground.

It is **not selected, exported, previewed as a successful candidate, or
deployed**. The policy installed on the Raspberry Pi is unchanged. Three V29
branches have been rejected. A fourth nominal continuation with a symmetric
reward contract is currently running; training return alone is not a release
result.

Identifiers:

- Task: `Drobot-Commanded-Walk-Schedule-Matched-Support-Straight-Crawl-External-Rear-Payload-Direct`
- Command set: `schedule-matched-support-straight-crawl-external-rear-payload`
- Experiment: `drobot_commanded_walk_v29_schedule_matched_support_straight_crawl_external_rear_payload`
- Bootstrap: V28 `model_3774.pt` from
  `2026-08-30_22-43-35_manual-headless`

## Why V29 exists

V28 reported about 81% three/four-foot support and completed almost every
all-four-release cycle. More detailed phase-conditioned diagnostics showed that
the apparently good support count did not prove a statically supported crawl:

- during a scheduled rear-leg swing, that rear foot could remain planted;
- one of the three scheduled anchor feet could unload instead;
- the contact count still remained three, so the old reward paid full support
  credit;
- the robot could therefore learn the wrong contact permutation while satisfying
  the aggregate support metric.

The old implementation was equivalent to:

```text
support_ok = number_of_airborne_feet <= 1
```

V29 uses:

```text
support_ok = actual_contact_mask == scheduled_contact_mask
```

During a swing this means the named swing foot is released and the other three
named feet are planted. During weight transfer, firm plant, weight return,
all-feet push, and settle, all four feet must be planted. This strict Boolean
identity match is logged and remains the acceptance metric.

Training also receives a smooth discovery signal based on each desired stance
foot's contact-force quality and each desired swing foot's release quality. The
weakest required contact controls the positive-progress multiplier, with a
`0.02` floor. This provides a gradient near the contact threshold without
relaxing deterministic evaluation: a partially unloaded or wrong foot still
does not count as an exact topology match.

The V29 support terms are:

| Term | Value |
| --- | ---: |
| Exact scheduled-contact reward | `8.0` |
| Missing scheduled-stance-foot penalty | `12.0` |
| More-than-one-airborne-foot penalty | `24.0` |
| Positive-progress discovery floor | `0.02` |
| Scheduled stance force-quality width | `1 N` |

The earlier causal all-four-release gate is retained independently. Every
physical foot must still remain below 1 N for five consecutive 60 Hz updates in
each complete gait cycle.

## Research rationale

The design uses a structured crawl reference and learned residuals, followed by
targeted rather than indiscriminate randomization:

- [Tan et al., Sim-to-Real Learning of Agile Locomotion](https://roboticsproceedings.org/rss14/p10.html)
  motivates explicit actuator, latency, and dynamics variation for transfer.
- [Rapid Motor Adaptation](https://arxiv.org/abs/2107.04034) demonstrates why
  latent hardware and terrain differences eventually benefit from an adaptation
  mechanism inferred from recent robot behavior.
- [Dynamics Randomization Revisited](https://arxiv.org/abs/2011.02404) supports
  staging and targeting uncertainty instead of assuming a wider distribution is
  always safer.
- [Walk These Ways](https://proceedings.mlr.press/v205/margolis23a.html) treats
  cadence, body height, pose, and foot motion as separable gait dimensions.
- [Terrain-aware trajectory generators](https://arxiv.org/html/2310.04675)
  support the use of an analytic foot trajectory plus learned correction rather
  than asking PPO to discover the entire gait from scratch.
- [Symmetry-aware reinforcement learning](https://hybrid-robotics.berkeley.edu/publications/IROS2024_Symmetry_RL_LeggedLoco.pdf)
  motivates symmetry as a prior, not a hard action constraint, because the rear
  pack and real actuator mismatches are asymmetric.
- [McGhee and Frank's static-stability formulation](https://doi.org/10.1016/0025-5564(68)90090-4)
  motivates preserving a useful support polygon while only one foot moves.

The STS3215 simulation continues to distinguish sustainable rated load from a
short transient drive limit. Feetech specifies the 12 V C018 variant at
10 kg-cm rated torque and 30 kg-cm stall torque; see the
[manufacturer page](https://www.feetech.cn/en/525603.html) and
[STS3215 specification](https://cdn.robotshop.com/media/F/Fit/RB-Fit-155/pdf/feetech_12v_30kg_cm_magnetic_encoding_servo_sts321_specification_pdf.pdf).
V29 inherits the V28 `2.6477955 N m` transient cap, begins penalizing above
`0.980665 N m`, and evaluates per-joint mean, RMS, peak, and above-rated dwell.

## Straight-line objective and its limit

The target line is latched in the episode-start world frame. The objective
penalizes cycle-normalized lateral velocity, accumulated lateral displacement,
departure beyond a 5 mm corridor, yaw, and heading error while rewarding only
forward progress aligned to the initial line. The known analytic side-to-side
weight transfer is subtracted before corridor scoring so a temporary zero-mean
crawl sway is not mistaken for drift.

V29 inherits the V28 straightness coefficients:

| Reward or penalty | Coefficient |
| --- | ---: |
| Cycle-normalized lateral-speed penalty | `10.0` |
| Lateral displacement penalty | `72.0` |
| Corridor-excess penalty | `14.0` |
| Corridor half-width | `5 mm` |
| Heading-error penalty | `12.0` |
| Straight-aligned progress reward | `10.0` |
| Forward tracking / instant / sustained progress | `6 / 8 / 12` |
| Stall / backward / overspeed penalty | `14 / 18 / 8` |

This can teach an unbiased gait and correction from observable lean, yaw, IMU,
joint state, and previous action. It cannot detect constant sideways crab
velocity after acceleration has returned to zero, or make the actor return to a
line after a pure lateral translation while the body remains parallel:
cross-track velocity and world position are not present in the 50-value actor
observation. Actual
return-to-line control requires AprilTag pose, camera/visual odometry, wheel-free
odometry, or another localization estimate in a slower outer steering loop.
Adding an unobservable training-only position penalty can select lower-bias
behavior, but it cannot give the deployed policy information it does not have.

## Selected mechanical reference

V29 does not switch to a same-direction-knee stance or a two-leg dynamic gait.
It retains the stable one-leg crawl order `RR -> FR -> RL -> FL` and the V28
shoe geometry, then changes load-transfer timing and residual authority.

| Setting | Selected V29 value |
| --- | ---: |
| Opposed front/rear foot sweep | `92 mm` |
| Approximate sagittal foot-center span | `304 mm` |
| Stance-down dimension | `0.3216749408 m` |
| Requested body pitch | `2 deg` nose-down |
| Independent stance-center offset | `0 mm` |
| Reference stride / lift | `46 / 24 mm` |
| General / rear forward transfer | `15 / 20 mm` |
| Lateral transfer | `6 mm` |
| Contact transition at each swing endpoint | `8%` of airborne interval |
| Physical lift order | `RR -> FR -> RL -> FL` |
| Phase fractions | `0.20 / 0.15 / 0.20 / 0.20 / 0.10 / 0.08 / 0.02 / 0.05` |
| Phase names | transfer / lift / swing / lower / plant / return / push / settle |
| Duty factor | `0.8625` |
| Initial command range | `0.008-0.020 m/s` |
| Full command range | through `0.037 m/s` |
| Command curriculum | `128,000` policy steps |
| Cadence range | `0.12-0.80 Hz`, command-scaled |
| Abduction / hip / knee residual scales | `0.10 / 0.12 / 0.15` |
| Target slew limit | `240 deg/s`, or `4 deg` per 60 Hz update |

The phase schedule moves five percentage points from the redundant all-feet
push hold to pre-lift transfer. Total swing time and duty factor are unchanged.
The short contact transitions avoid demanding release at the zero-motion first
lift sample or forbidding touchdown until the zero-motion final lower sample.
The hip residual is raised from V28's `0.04` to `0.12`, which corresponds to
about 2.06 degrees around the analytic reference; final targets remain clamped
to the soft joint envelope and rate limited.

The earlier `-5 mm` independent stance-center candidate was rejected after a
source-level geometry audit. The +2 degree helper already supplies a complete
`+7.5265 mm` common foot-X bias. Subtracting 5 mm reduced the analytic geometry
to about +0.67 degrees while reset pose and posture reward still requested +2
degrees. The corrected reference uses zero independent offset so its neutral
joint pose, root pose, pitch target, simulator table, and eventual export
contract agree.

### Why all knees do not point the same way

The opposed stance places the front feet forward of their hip axes and the rear
feet behind theirs. With 120 mm between the front and rear hip axes and 92 mm
outward sweep at both ends, the approximate sagittal support span is:

```text
120 + 92 + 92 = 304 mm
```

A same-direction arrangement translates the front and rear contact centers in
the same direction and trends back toward the roughly 120 mm hip-axis spacing.
That smaller fore/aft polygon is unfavorable for a robot with no ankle control
and an external rear pack. It also does not inherently produce more thrust:
forward impulse comes from a planted foot moving backward relative to the body,
not from the knee's visual orientation. A faster trot may be trained later as a
separate mode, after the sequential crawl meets support and effort gates.

## Geometry and offset ablations

These deterministic three-by-ten-second evaluations used zero policy residual
at `0.015 m/s`. They isolate the analytic reference and are not deployable
policy results.

### Forward-transfer sweep at zero stance offset

| General / rear transfer | Exact topology | Three/four support | All-four cycles | Speed | Meaningful reverse fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `8 / 10 mm` | `0.4175` | `0.7675` | `0.75` | `0.00650 m/s` | `0.395` |
| `10 / 14 mm` | `0.4358` | `0.7408` | `1.00` | `0.00826 m/s` | not retained |
| `12 / 18 mm` | `0.4433` | `0.7158` | `1.00` | `0.01031 m/s` | not retained |
| `15 / 20 mm` | `0.4625` | `0.7225` | `1.00` | `0.01056 m/s` | not retained |

At `15 / 20 mm`, rear swing-foot persistence improved: scheduled rear swing
contact fell to approximately `0.255 / 0.355` for RL/RR, compared with
`0.458 / 0.582` at `8 / 10 mm`. Front scheduled-stance airborne rates remained
about `0.258 / 0.175`, so transfer alone did not solve the identity mismatch.

### Stance-center sweep with `15 / 20 mm` transfer

| Independent offset | Exact topology | Three/four support | Mean pitch |
| --- | ---: | ---: | ---: |
| `-15 mm` | `0.4017` | `0.6467` | `-0.0768 rad` |
| `-10 mm` | `0.5008` | `0.7250` | `-0.0541 rad` |
| `-6 mm` | `0.5442` | `0.7792` | `-0.0360 rad` |
| `-5 mm` | `0.5117` | `0.7483` | `-0.0314 rad` |
| `-4 mm` | `0.5267` | `0.7625` | `-0.0272 rad` |
| `-2.5 mm` | `0.5108` | `0.7458` | `-0.0206 rad` |
| `0 mm` | `0.4658` | `0.7075` | `-0.00863 rad` |
| `+5 mm` | `0.4217` | `0.7275` | `+0.0138 rad` |

This sweep was initially interpreted as favoring `-5 mm`, but it optimized
contact count while violating the +2 degree stance contract. It is retained as
a diagnostic record, not as the selected geometry. Zero independent offset is
now selected for internal consistency. Rear unloading is tuned with transfer
timing and learned residuals rather than globally translating the stance.

## V28 checkpoint under the selected V29 reference

Before retraining, transferred V28 `model_3774.pt` was evaluated deterministically
for three ten-second episodes at `0.015 m/s` under the final V29 geometry:

| Metric | Result |
| --- | ---: |
| Falls | `0/3` |
| Mean speed | `0.014409 m/s` |
| Mean lateral error | `0.07513 m/m` |
| Exact scheduled topology | `0.49556` |
| Three/four-foot support | `0.76167` |
| Complete all-four cycles | `1.00` |
| Meaningful reverse fraction | `0.23667` |
| Mean pitch | `-0.00939 rad` (`-0.54 deg`) |
| Joint RMS acceleration | `8.464 rad/s2` |
| Target-limiter gap | `0.0000301` |

This is a useful bootstrap, not a candidate. It misses the strict support,
straightness, and reverse-motion gates.

## Rejected V29 branches

### Branch A: hard topology gate

The first branch used the corrected contact identities with a hard progress
gate, the revised phase timing, the original `8 / 10 mm` transfers, zero stance
offset, a `0.04` hip residual, `24.0` missing-anchor penalty, `6.0` rearward-pitch
penalty, and a `3e-5` PPO learning rate. Run:

`2026-08-30_23-20-49_manual-headless`

It was stopped after approximately 142 updates. Deterministic three-by-ten-second
screens at models 3800, 3850, and 3900 remained around `0.439-0.444` exact
topology and `0.816-0.823` three/four support; all three completed only about
two-thirds of all-four cycles. At model 3800, speed was `0.0090 m/s`, lateral
error was `0.0657 m/m`, meaningful reverse fraction was `0.258`, and mean pitch
was `-0.00568 rad`. The hard discontinuity did not provide a useful learning
signal, so no checkpoint was selected.

### Branch B: smooth force gate, uncorrected geometry

The second branch replaced the hard gate with the smooth weakest-contact force
quality, increased hip authority to `0.10`, reduced the missing-anchor and
rearward-pitch penalties to `12.0 / 4.0`, and used a `5e-5` learning rate. It
still used the original reference transfer and zero independent stance offset.
Run:

`2026-08-30_23-32-38_manual-headless`

It was stopped after approximately 136 updates. Models 3800, 3850, and 3900
gave only `0.426-0.435` exact topology and `0.814-0.818` three/four support;
lateral error worsened from about `0.085` to `0.105 m/m`. Smooth shaping alone
could not correct a reference that kept presenting the wrong load distribution.

### Branch C: mismatched stance and phase-asymmetric cycle reward

The third branch combined the smooth force gate with `15 / 20 mm` transfer,
`-5 mm` stance-center offset, revised phase fractions, and
`0.10 / 0.12 / 0.15` residual authority. It started from V28 `model_3774.pt`,
used 4,096 environments and seed 2999, and kept randomization disabled:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet schedule-matched-support-straight-crawl-external-rear-payload `
  -Iterations 300 -NumEnvs 4096 -Seed 2999 -V25Phase nominal `
  -Checkpoint .\logs\rsl_rl\drobot_commanded_walk_v28_forward_biased_cycle_gated_four_leg_straight_crawl_external_rear_payload\2026-08-30_22-43-35_manual-headless\model_3774.pt
```

Run:

`2026-08-30_23-54-36_manual-headless`

It completed 300 updates through `model_4073.pt`. Deterministic three-by-ten-
second screens at `0.015 m/s`, seed 4901, were:

| Checkpoint | Exact topology | Three/four support | All-four cycles | Lateral m/m | Reverse fraction | Speed m/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `model_3925` | `0.5411` | `0.8050` | `0.8333` | `0.0777` | `0.2394` | `0.01461` |
| `model_4000` | `0.5450` | `0.8111` | `0.8333` | `0.0809` | `0.2267` | `0.01475` |
| `model_4073` | `0.5511` | `0.8189` | `0.8333` | `0.0809` | `0.2167` | `0.01489` |

There were no falls and acceleration remained below `10 rad/s2`, but every
screen failed contact, all-four, lateral, reverse, and effort gates. Source
audit then found two causal design errors: the `-5 mm` offset cancelled most of
the +2 degree stance geometry, and the current-cycle release gate plus
120-point event always gave the largest progress credit after the fixed final
front-left swing. Branch C is rejected.

## Current symmetric nominal continuation

The corrected branch uses zero independent stance offset, removes the
leg-order-dependent cycle gate/event from training reward, maps a missing
scheduled anchor to zero soft gate quality, and reserves 8% of each end of the
airborne interval for smooth unload/touchdown contact. The strict all-four
release remains an acceptance metric. The analytic speed ceiling is reduced to
`0.037 m/s` because `46 mm * 0.80 Hz = 0.0368 m/s`.

It continues from Branch C `model_4073.pt`, which was the stronger bootstrap
under the corrected contract, with 4,096 environments and seed 3007:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet schedule-matched-support-straight-crawl-external-rear-payload `
  -Iterations 500 -NumEnvs 4096 -Seed 3007 -V25Phase nominal `
  -Checkpoint .\logs\rsl_rl\drobot_commanded_walk_v29_schedule_matched_support_straight_crawl_external_rear_payload\2026-08-30_23-54-36_manual-headless\model_4073.pt
```

Run: `2026-08-31_00-22-50_manual-headless`

**Result status: training in progress.** No checkpoint is selected from this
run until deterministic held-out evaluation is complete.

No V29 ONNX export, policy copy, Raspberry Pi update, or real motor command is
authorized by this in-progress run.

## Staged physical randomization

V29 nominal training intentionally leaves randomization inactive while the new
contact-identity objective and complete speed curriculum are learned. Broad
dynamics variation at this stage would make it difficult to distinguish a bad
contact contract from robustness. The workflow therefore keeps V29 nominal for
2,000 PPO updates (`128,000` policy steps), instead of inheriting the earlier
350-update switch that would enable randomization at only `22,400` steps.

Only after a nominal checkpoint passes should the robust phase retain 25%
nominal environments and sample:

| Domain | Planned inherited range |
| --- | --- |
| Dry-robot mass/inertia | `0.96-1.04` |
| Rear payload mass | `0.955-1.055` |
| Rear payload COM jitter X/Y/Z | `+/-8 / +/-10 / +/-6 mm` |
| Common supply effort/rate | correlated `0.92-1.05` |
| Per-joint effort | `0.90-1.05` |
| Per-joint target rate | `0.88-1.05` |
| Per-joint stiffness/damping | `0.85-1.15` |
| Abduction / hip-knee zero error | `+/-1 / +/-1.5 deg` |
| Command delay | `0-1` control update |
| Reset roll/pitch and yaw | `+/-2 / +/-5 deg` |
| Gyroscope bias | `+/-0.020 rad/s` |
| Projected-gravity / acceleration noise | `0.006 / 0.020 g` |
| Persistent force X/Y/Z | `+/-0.10 / +/-0.40 / 0 N` |
| Persistent roll/pitch/yaw torque | `+/-0.025 / +/-0.015 / +/-0.040 N m` |
| Common static / dynamic friction | `0.40-0.90 / 0.25-0.70` |
| Per-foot friction multiplier | `0.85-1.15` |

This distribution covers a slightly tilted assembly through reset lean,
persistent wrench, COM offset, fixed joint-zero error, unequal motor response,
and IMU bias. It does not hard-code a correction for the current robot's left
drift. If instantaneous IMU and joint observations cannot identify a persistent
physical variant, the next architecture should add short observation history
or an RMA-style adaptation latent rather than widening randomization again.

Before final robust selection, add or measure narrower targeted domains for
fixed IMU mounting roll/pitch, small persistent ground slope, per-leg contact
height, contact compliance, and modest servo deadband. Randomized evaluation
must use multiple process seeds because mass, friction, COM, motor, zero, and
latency draws are currently fixed at environment construction; multiple episode
resets inside one evaluator process are not independent hardware domains.

## Selection sequence and hard gates

1. Complete the nominal continuation and screen several late checkpoints with
   deterministic actor means at `0.015 m/s`.
2. Require zero falls, complete all-four release in every complete cycle, at
   least `0.90` exact scheduled-contact topology, and at least `0.90`
   three/four-foot support.
3. Require mean lateral error at most `0.05 m/m`, worst-episode error at most
   `0.10 m/m`, meaningful reverse fraction below `0.10`, joint RMS acceleration
   at most `10 rad/s2`, and target-limiter gap at most `0.02`.
4. Require measured speed within `0.009-0.02025 m/s` for the `0.015 m/s`
   screen, then repeat held-out nominal seeds at `0.005`, `0.015`, `0.030`, and
   `0.037 m/s`.
5. Inspect phase-conditioned scheduled-stance failures and per-joint effort;
   the evaluator reports an active-swing by missing-anchor matrix because
   aggregate support and effort averages are insufficient.
6. Start robust continuation only if nominal gates pass, then repeat nominal
   and randomized evaluations.
7. Record video only for a numerically credible candidate and inspect foot
   identity, crossing, edge loading, hopping, yaw chatter, and visible jerk.
8. Export and deploy only after numerical and visual selection.

A result that is fast but misses its named support feet is rejected. A result
that is straight because it barely moves is also rejected. The selected policy
must satisfy support, progress, straightness, smoothness, and actuator-headroom
criteria together.
