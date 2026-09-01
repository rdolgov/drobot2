# V28 forward-biased, cycle-gated straight crawl

## Status

V28 is a rejected Isaac Lab research profile for a smoother, faster low-stance
sequential crawl that stays close to the episode-start straight line and proves
meaningful release by all four feet. Its final current-source continuation
exposed a contact-identity reward bug and led to the V29 profile.

It is **not selected, exported, recorded as a successful preview, or deployed**.
The policy already installed on the Raspberry Pi is unchanged.

Identifiers:

- Task: `Drobot-Commanded-Walk-Forward-Biased-Cycle-Gated-Four-Leg-Straight-Crawl-External-Rear-Payload-Direct`
- Command set: `forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload`
- Experiment: `drobot_commanded_walk_v28_forward_biased_cycle_gated_four_leg_straight_crawl_external_rear_payload`

The first nominal run is diagnostic because it predates the current causal
cycle gate, all-four event reward, support weights, and corrected transient
motor cap. None of its checkpoints can be exported without a current-source
continuation and fresh held-out evaluation.

## Research basis

V28 uses an analytic foot trajectory with learned residual corrections, a
nominal-first then targeted-randomization curriculum, and an explicit
one-leg-at-a-time crawl contract. This direction is based on:

- [Tan et al., Sim-to-Real Learning of Agile Locomotion](https://arxiv.org/html/1804.10332): model actuator limits and latency, randomize uncertain dynamics, and validate transfer-sensitive behavior.
- [Rapid Motor Adaptation](https://arxiv.org/html/2107.04034): payload, friction, and motor variation benefit from adaptation inferred from robot behavior.
- [Dynamics Randomization Revisited](https://arxiv.org/html/2011.02404): broader randomization is not automatically better, so uncertainty should be targeted and staged.
- [Walk These Ways](https://proceedings.mlr.press/v205/margolis23a.html): cadence, stance height, body pose, and swing geometry are useful independent gait dimensions.
- [Terrain-aware trajectory generators](https://arxiv.org/html/2310.04675): a structured foot trajectory plus learned corrections provides useful clearance and timing without preventing adaptation.
- [Symmetry-aware RL](https://hybrid-robotics.berkeley.edu/publications/IROS2024_Symmetry_RL_LeggedLoco.pdf): symmetry is a useful prior, but hard mirrored actions are inappropriate for an asymmetric rear payload and imperfect hardware.
- [McGhee and Frank's static-stability formulation](https://doi.org/10.1016/0025-5564(68)90090-4): a slow crawl should keep the projected center of mass inside a useful support polygon while one foot moves.

The present actor has no history encoder. Targeted randomization can teach a
low-bias, disturbance-tolerant response using IMU and joint feedback. If
persistent hardware differences cannot be inferred from those instantaneous
signals, a short observation history or RMA-style adaptation module is the
next architectural step.

## Robot, payload, and motor model

| Quantity | Value |
| --- | ---: |
| Measured assembled mass | `3.17514659 kg` (`7 lb`) |
| Explicit rear battery, box, and lid | `0.523179545 kg` |
| External pack center | `(-0.1315, 0.0, 0.0500) m` |
| External pack size | `(0.043, 0.144, 0.068) m` |
| Dry-CAD mass correction | `0.650607608` |
| STS3215 rated torque at 12 V | `0.980665 N m` (`10 kg cm`) |
| STS3215 documented stall torque | `2.941995 N m` (`30 kg cm`) |
| Hardware torque-limit register | `900 / 1000` |
| Nominal simulated transient cap | `2.6477955 N m` |
| Control frequency | `60 Hz` |

Feetech documents the C018 servo for 12 V, 10 kg-cm rated torque, 30 kg-cm
stall torque, and 0.222 s/60-degree no-load speed. See the
[manufacturer product page](https://www.feetech.cn/en/525603.html) and
[STS3215 specification](https://cdn.robotshop.com/media/F/Fit/RB-Fit-155/pdf/feetech_12v_30kg_cm_magnetic_encoding_servo_sts321_specification_pdf.pdf).

The model deliberately separates short-duration capacity from continuous
load. Ninety percent of documented stall torque approximates the configured
hardware cap. The reward begins penalizing effort above rated torque, currently
`0.37037037` of that cap. This permits a brief support transient without
teaching the policy to hold near stall and overheat a servo. Battery/supply
randomization scales rate and torque together.

An earlier V28 draft incorrectly used rated torque as the hard actuator cap.
A static-hold audit then showed all four knees effectively clipped at 100% and
both rear hip motors near 95% even without policy actions. That was a modeling
artifact: rated torque is a sustainable target, not the servo's instantaneous
limit. Evaluation now reports mean, RMS, peak, and above-rated occupancy for
every joint rather than hiding them in one aggregate.

## Stance and gait contract

| Setting | V28 |
| --- | ---: |
| Opposed front/rear foot sweep | `92 mm` |
| Approximate sagittal foot-center span | `304 mm` |
| Stance-down dimension | `0.3216749408 m` |
| Requested body pitch | `2 deg` nose-down |
| Common forward stance bias | `7.526541 mm` |
| Reference stride | `46 mm` |
| Swing lift | `24 mm` |
| General / rear forward transfer | `8 / 10 mm` |
| Lateral phase transfer | `6 mm` |
| Duty factor | `0.8625` |
| Phase offsets, FL/RL/FR/RR | `0.07 / 0.32 / 0.57 / 0.82` |
| Physical lift order | `RR -> FR -> RL -> FL` |
| Startup ramp | `1.5 s` |
| Trained speed range | `0.005-0.045 m/s` |
| Nominal-start command range | `0.008-0.030 m/s` |
| Cadence range | `0.12-0.80 Hz`, command-scaled |
| Target slew limit | `240 deg/s`, or `4 deg` per 60 Hz update |

The stance is approximately 7.67 mm lower than the earlier 80 mm swept stance.
The upper command becomes more assertive mainly through cadence rather than an
unsafe sagittal envelope. Residual action scales in joint-kind-major order are
`0.10` for abduction, `0.04` for sagittal hips, and `0.15` for knees. Small hip
authority preserves the remaining soft-limit margin; knees retain enough
authority to unload a weak or mis-zeroed leg.

### Why the knees remain opposed

With 120 mm between front and rear hip axes and 92 mm outward sweep at both
ends, the approximate foot-center span is:

```text
120 + 92 + 92 = 304 mm
```

Pointing every leg in the same sagittal direction would translate front and
rear feet similarly and reduce their separation toward the 120 mm hip spacing.
That is especially unfavorable with the rear-mounted battery. Same-direction
knees do not inherently create more thrust; propulsion comes from planted feet
sweeping backward relative to the body. The repository also retains a previous
same-direction hardware profile that fell in Isaac and was superseded.

A faster dynamic trot can be a separate later mode. It should not replace the
stable sequential crawl until actuator, battery, and contact behavior are
validated.

## Straight-line objective

V28 latches its target line in the episode-start world frame. A robot that
yaws left cannot redefine its own body X direction as correct forward travel.

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
| Analytic gait-reference reward | `5.0` |

Lateral velocity is averaged over a command-dependent complete gait cycle.
The known 6 mm phase-locked weight transfer is subtracted before corridor and
instantaneous lateral scoring. Necessary zero-mean crawl sway is therefore
allowed, but drift that accumulates between cycles is penalized. A heading
controller with gain `1.5 s^-1` and a `0.20 rad/s` bound feeds yaw error back
through the existing command observation.

The 50-value actor observation contains command, gait clock, IMU angular
velocity, projected gravity, linear acceleration, joint positions, joint
velocities, and prior action. It does not contain world position or cross-track
displacement. The policy can learn unbiased walking and react to lean/yaw, but
cannot know that it was translated sideways while remaining parallel to the
line. True return-to-line behavior requires AprilTag, camera odometry, or
another localization source in a higher-level steering loop.

## Four-leg release and support gate

V27's differentiable force score accepted partially loaded shoes and could
award transfer frames where no foot was scheduled to swing. Stochastic Beta
tails could release a rear foot even when the deployable deterministic mean did
not. V27 reached `model_3549.pt`, but deterministic rear-foot release failed;
no V27 checkpoint was exported or deployed.

V28 requires each scheduled swing shoe to remain below 1 N for five consecutive
60 Hz updates, approximately 83 ms. All four feet must qualify in a complete
cycle. Current-cycle progress is scaled by:

```text
0.02 + 0.98 * (qualified_feet / 4)^4
```

| Qualified feet | Progress multiplier |
| ---: | ---: |
| 0 | `0.020` |
| 1 | `0.024` |
| 2 | `0.081` |
| 3 | `0.330` |
| 4 | `1.000` |

The gate is causal: a good previous cycle cannot give full motion credit to a
bad current cycle. The fourth qualifying foot earns a one-time `120.0` event.
A narrow force sigmoid remains only as discovery shaping.

Support shaping reinforces one moving leg, not longer swing time:

- scheduled stance reward: `3.5`;
- exact three-foot swing / four-foot transfer reward: `5.0`;
- more-than-one-airborne-foot penalty: `18.0`;
- scheduled-swing reward: `10.0`;
- least-active-foot reward: `8.0`.

Smoothness remains a first-class objective: action rate `0.18`, action second
difference `0.90`, joint acceleration `0.30`, body linear acceleration `0.55`,
body angular acceleration `0.45`, support slip `0.60`, and touchdown impact
`0.18`. These directly limit acceleration and action curvature; they are a
practical discrete-time jerk proxy rather than a literal third-derivative
sensor.

## Targeted robust randomization

Randomization is inactive during `-V25Phase nominal`. The later robust phase
retains 25% nominal environments and applies episode-consistent uncertainty to
the rest.

| Domain | Robust range |
| --- | --- |
| Whole dry-robot mass/inertia | `0.96-1.04` |
| Rear payload mass | `0.955-1.055` |
| Rear payload COM jitter X/Y/Z | `+/-8 / +/-10 / +/-6 mm` |
| Common supply effort/rate | correlated `0.92-1.05` |
| Per-joint effort | `0.90-1.05` |
| Per-joint rate | `0.88-1.05` |
| Per-joint stiffness/damping | `0.85-1.15` |
| Abduction zero error | `+/-1 deg` |
| Hip/knee zero error | `+/-1.5 deg` |
| Command delay | `0-1` control update |
| Reset roll/pitch and yaw | `+/-2 deg`, `+/-5 deg` |
| Gyroscope bias | `+/-0.020 rad/s` |
| Projected-gravity / acceleration noise | `0.006`, `0.020 g` |
| Persistent force X/Y/Z | `+/-0.10 / +/-0.40 / 0 N` |
| Persistent roll/pitch/yaw torque | `+/-0.025 / +/-0.015 / +/-0.040 N m` |
| Common static / dynamic friction | `0.40-0.90 / 0.25-0.70` |
| Per-foot friction multiplier | `0.85-1.15` |

Rear COM jitter, fixed joint-zero errors, unequal actuator response, persistent
body wrench, reset lean, and IMU bias cover plausible assembly tilt rather than
baking in one known left/right correction. The same common draw scales torque
and rate because supply voltage affects both. A common surface draw covers all
four Velcro-like pads becoming slippery together; per-foot multipliers cover
wear and installation differences. Friction remains an engineering prior until
a tilt or pull test measures it.

## Initial nominal run and rejection

The first run used 4,096 environments, seed 2828, 350 updates, and V24
`model_3248.pt` as its bootstrap:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload `
  -Iterations 350 -NumEnvs 4096 -Seed 2828 -V25Phase nominal `
  -Checkpoint .\logs\rsl_rl\drobot_commanded_walk_v24_padded_feet_forward_bias_external_rear_payload\2026-08-30_17-15-44_manual-headless\model_3248.pt
```

Output:

`logs/rsl_rl/drobot_commanded_walk_v28_forward_biased_cycle_gated_four_leg_straight_crawl_external_rear_payload/2026-08-30_22-10-44_manual-headless`

It produced checkpoints `model_3250.pt` through `model_3597.pt` in about 761 s.
The run predates the corrected current-cycle gate and motor cap.

Five deterministic 20-second nominal episodes at `0.015 m/s` gave:

| Metric | `model_3500` | `model_3525` | Required before robust |
| --- | ---: | ---: | ---: |
| Falls | `0` | `0` | `0` |
| Mean speed | `0.011868` | `0.011881 m/s` | `0.009-0.02025` |
| All-four cycles | `24/25` | `24/25` | `100%` |
| Mean lateral error | `0.0263` | `0.0178 m/m` | `<=0.05` |
| Three/four-foot support | `0.8475` | `0.8478` | `>=0.90` |
| Backward sample fraction | `0.228` | `0.226` | inspect with deadband |
| Joint RMS acceleration | `7.447` | `7.229 rad/s2` | `<=10` |

Both are rejected. They miss one all-four cycle and do not maintain the desired
support topology. The old effort comparison is not retained as a gate because
that run conflated rated torque with peak actuator capacity.

## Reference-only ablations

The following three-by-ten-second comparisons forced learned residuals to zero
at `0.015 m/s`. They test the reference, not a deployable policy.

| Geometry | Support | Speed | Mean pitch | All-four cycles |
| --- | ---: | ---: | ---: | ---: |
| `92 mm`, `2 deg`, `24 mm` lift, rated hard cap | `0.8583` | `0.008465` | `-0.00787 rad` | `6/6` |
| `85 mm`, `0 deg`, `24 mm` | `0.9100` | `0.01064` | `-0.04466 rad` | `0/6` |
| `85 mm`, `1 deg`, `24 mm` | `0.8750` | `0.00977` | `-0.02389 rad` | `1/6` |
| `92 mm`, `1 deg`, `24 mm` | `0.8861` | `0.00991` | `-0.02736 rad` | `0/6` |
| `92 mm`, `2 deg`, `20 mm` | `0.9072` | `0.007247` | `-0.00784 rad` | `1/6` |
| Previous row plus `10/12/8 mm` transfer | `0.8728` | `0.007717` | not recorded | `2/6` |
| Selected geometry, corrected transient cap | `0.7628` | `0.006725` | `-0.00653 rad` | `4/6` |

Narrowing the stance, reducing pitch, reducing lift, or increasing transfer
improved one metric only by degrading release, speed, straightness, or another
support property. The selected 92 mm / 2 degree / 24 mm geometry is retained,
but the corrected-cap baseline proves that the actor must improve its support
timing rather than merely copy the analytic trajectory.

## Current-source continuation and final rejection

V28 `model_3525.pt` was continued for 250 nominal updates under the corrected
current-source reward and actuator model:

`logs/rsl_rl/drobot_commanded_walk_v28_forward_biased_cycle_gated_four_leg_straight_crawl_external_rear_payload/2026-08-30_22-43-35_manual-headless`

The run produced `model_3774.pt`. Deterministic three-by-twenty-second screens
of late checkpoints from `model_3675.pt` through `model_3774.pt` had zero falls,
but remained at only `0.804-0.809` three/four-foot support and approximately
`0.933` complete all-four cycles. Mean speed was `0.0105-0.01095 m/s`, mean
lateral error `0.028-0.037 m/m`, meaningful reverse fraction `0.220-0.239`,
mean pitch about `-0.007 rad`, and joint RMS acceleration
`6.14-6.75 rad/s2`. About 45% of joint effort samples remained above the rated
torque threshold. No late checkpoint met all selection gates.

A detailed deterministic three-by-ten-second audit of `model_3774.pt` exposed
why the aggregate support number was misleading:

| Diagnostic | Result |
| --- | ---: |
| Three/four-foot support | `0.8128` |
| Exactly two / three / four feet planted | `0.1872 / 0.3822 / 0.4306` |
| Complete all-four release cycles | `0.933` |
| Swing-phase three/four support | `0.7286` |
| Transfer-phase three/four support | `0.9010` |
| Transfer-phase all-four support | `0.5813` |
| Exact scheduled contact identities | approximately `0.447` |
| Scheduled-stance airborne FL / FR | `0.196 / 0.276` |
| Scheduled-stance airborne RL / RR | `0.030 / 0.025` |
| Rear swing foot still contacting RL / RR | `0.483 / 0.452` |

The V28 `three_foot_support` reward checked only that no more than one foot was
airborne. It did not check **which** foot was airborne. A scheduled rear swing
foot could stay planted while a front anchor lifted, yet still receive full
support credit. This was a reward-definition bug, not evidence that the servo
itself was limited to a low contact rate.

V28 is therefore closed as a rejected research stage. It was not exported or
deployed. The identity-aware reward, load-transfer geometry sweeps, and ongoing
continuation are documented in
[`rl-schedule-matched-support-v29.md`](rl-schedule-matched-support-v29.md).

## Original training and selection sequence

1. Continue `model_3525.pt` in nominal mode under the causal gate, corrected
   transient cap, stronger support topology, and stronger straight-path terms.
2. Evaluate held-out nominal seeds at `0.005`, `0.015`, `0.030`, and
   `0.045 m/s`.
3. Require zero falls/stalls, positive progress in every episode, 100% complete
   all-four release cycles, every leg releasing in every episode, at least 90%
   three/four-foot support through `0.030 m/s` and 80% at `0.045 m/s`, mean
   lateral error at most `0.05 m/m`, worst at most `0.10 m/m`, and joint RMS
   acceleration at most `10 rad/s2`.
4. Inspect per-joint mean/RMS/peak effort and above-rated dwell instead of using
   one misleading aggregate threshold.
5. Only after nominal success, continue the selected checkpoint with
   `-V25Phase robust` and repeat both nominal and randomized evaluation.
6. Record video only for a numerically credible candidate. Inspect foot
   crossing, hopping, edge loading, yaw chatter, and visible jerk.
7. Export only after selection. The V28 exporter rejects incomplete task,
   stance, pitch, phase, cadence, residual, reference, and heading-hold
   metadata.

The first eventual hardware trial must use a charged bench supply, harness or
spotter, emergency stop, and automatic telemetry recording. Start with a stance
hold, then `0.005 m/s` for three seconds, then ten seconds. Battery trials come
only after a stable power-supply trial.
