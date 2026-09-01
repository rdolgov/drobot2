# Parallel commanded walking

This task trains a pure PPO policy across many Isaac Lab environments. It starts every
robot in the validated symmetric, four-foot neutral stance. V18 provides an analytic
diagonal-pair gait reference and clock, while PPO remains responsible for balance and
the final joint commands.

The policy observation has 50 hardware-reproducible values:

- commanded forward velocity, lateral velocity, and yaw rate (3)
- sine and cosine of the gait clock (2)
- body IMU angular velocity, projected gravity, and linear acceleration (9)
- joint position error, normalized joint velocity, and previous action (36)

The flat-ground actor deliberately does not consume the depth sensor. Depth is useful for
stairs and obstacles, but a featureless plane contributes no useful terrain information.
The actor is a two-layer 256x256 MLP using deployable IMU and joint state. During training
only, the critic also sees simulator base velocity, base height, and foot contacts. Those
privileged values are never required by the deployed actor.

## V21 low-speed external rear-payload profile

`low-speed-external-rear-payload` is the retraining profile created from the
real-walk timing investigation. It continues from selected V20 weights but
resets its own command curriculum: commands expand from `0.04-0.10 m/s` down
to `0.005-0.10 m/s` over 38,400 policy steps. Cadence scales from 0.35 to
1.25 Hz and reference stride from 35% to 100%, so a tiny speed request no
longer drives the old full-rate shuffle. The sustained-stall threshold also
scales with command speed, while V20's rear payload, straightness, slip,
touchdown, joint/body acceleration, and action-acceleration objectives remain.

Start the continuation with:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet low-speed-external-rear-payload -Iterations 800 -NumEnvs 128
```

The clean-checkout fallback is V20 `model_900.pt`. The workflow resets the V21
curriculum offset even though it bootstraps that checkpoint. Evaluate candidate
checkpoints independently at `0.005`, `0.01`, `0.02`, `0.04`, and `0.08 m/s`
before export or hardware deployment. The controller/timing baseline and exact
acceptance rationale are in
`debug/walk-trials/20260826T020512958721Z-7a4955cb/ANALYSIS.md`.

Export only a selected V21 checkpoint with its trained cadence contract:

```powershell
& C:\isaacsim\python.bat `
  .\simulation\isaac\rl\parallel_walking\export_policy_onnx.py `
  --checkpoint <selected-model.pt> `
  --output <selected-model.onnx> `
  --training-task Drobot-Commanded-Walk-Low-Speed-External-Rear-Payload-Direct `
  --training-profile low-speed-external-rear-payload `
  --gait-clock-mode speed_scaled `
  --gait-standstill-deadband-m-s 0.002 `
  --gait-speed-min-m-s 0.005 --gait-speed-max-m-s 0.10 `
  --gait-frequency-min-hz 0.35 --gait-frequency-max-hz 1.25 `
  --forward-speed-min-m-s 0.005 --forward-speed-max-m-s 0.10 `
  --recommended-forward-speed-m-s 0.005
```

The generated JSON sidecar is required at deployment. It prevents the runtime
from applying speed-scaled cadence to a fixed-clock model or accepting a speed
outside that model's training range.

## V22 deployment-matched residual crawl

`low-speed-crawl-external-rear-payload` is the smoothness-first candidate
created from the August 29 real V20 trial. Unlike V20/V21's direct-action
diagonal trot, it starts with the exact known-good distributed-push hardware
crawl and gives PPO only a 25% residual correction around that reference. The
base gait schedules one airborne foot at a time in the stable rear-right,
front-right, rear-left, front-left order. Its 86.25% effective stance duty and
explicit contact rewards favor at least three planted feet and penalize
two-or-more airborne feet.

The environment applies the same `2 degrees / 60 Hz` target cap as the Pi
(`120 deg/s`) and penalizes requested targets that run ahead of that limiter.
Cadence scales from `0.06` to `0.30 Hz` over `0.003-0.015 m/s`, while the full
50 mm step length is retained so speed changes by cadence instead of tiny foot
motions. Action, joint, body acceleration, slip, touchdown, tilt, yaw, and
lateral motion costs are stronger than V20. Training starts in
`0.008-0.015 m/s` and expands down to `0.003 m/s` over 51,200 policy steps.

Start a fresh residual-policy run with:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet low-speed-crawl-external-rear-payload `
  -Iterations 1000 -NumEnvs 128 -Fresh
```

Export only a candidate that passes sustained evaluation at `0.003`, `0.005`,
`0.01`, and `0.015 m/s`, has three-foot support, low limiter gap, no falls or
stalls, and visibly smooth motion. The V22 export must include:

```powershell
--training-task Drobot-Commanded-Walk-Low-Speed-Crawl-External-Rear-Payload-Direct `
--training-profile low-speed-crawl-external-rear-payload `
--gait-clock-mode speed_scaled --gait-pattern distributed_support_crawl `
--gait-duty-factor 0.8625 --gait-phase-offsets 0.07 0.32 0.57 0.82 `
--gait-standstill-deadband-m-s 0.002 `
--gait-speed-min-m-s 0.003 --gait-speed-max-m-s 0.015 `
--gait-frequency-min-hz 0.06 --gait-frequency-max-hz 0.30 `
--gait-stride-scale-min 1.0 --action-mode gait_residual `
--residual-action-scale 0.25 --reference-sample-count 2048 `
--reference-start-ramp-s 1.5 --reference-stride-m 0.050 `
--reference-lift-m 0.016 --reference-weight-shift-forward-m 0.006 `
--target-velocity-limit-rad-s 2.0943951024 --max-target-step-deg 2 `
--forward-speed-min-m-s 0.003 --forward-speed-max-m-s 0.015 `
--recommended-forward-speed-m-s 0.005
```

The ONNX sidecar embeds the same 2,048-sample joint reference table used in
Isaac. Deployment reconstructs `reference + 0.25 * policy residual`, then
applies the same 120 deg/s target limiter. The sidecar is therefore part of the
model and must always be copied with the ONNX file.

## V23 higher-speed straight residual crawl

`higher-speed-straight-crawl-external-rear-payload` continues from the selected
V22 checkpoint without changing its 50-value observation or 12-value residual
action contract. It expands the trained command range to `0.005-0.050 m/s` and
scales the sequential-crawl cadence from `0.12` to `0.75 Hz` around a 65 mm
reference stride. The single-foot swing order, 86.25% stance duty, 25% policy
residual, 60 Hz controller, and 120 deg/s joint-target limiter remain intact.

V23 adds separate costs for lateral velocity, total lateral displacement,
leaving a 20 mm path corridor, yaw rate, and accumulated heading error. The
strong V22 action, joint, and body-acceleration costs remain enabled. Continue
training from the bundled V22 model with:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet higher-speed-straight-crawl-external-rear-payload `
  -Iterations 1200 -NumEnvs 128 -Seed 2301
```

Evaluate candidate checkpoints at `0.005`, `0.015`, `0.030`, and `0.050 m/s`.
Selection must consider actual speed, fall/stall rate, lateral displacement,
final heading error, yaw travel, three-foot support, target-limiter gap, and
joint/body acceleration together; highest checkpoint number is not itself a
selection criterion.

## V24 padded-feet forward-bias residual crawl

`padded-feet-forward-bias-external-rear-payload` continues from selected V23
checkpoint 1500. It models the adhesive Velcro-like sole pads as potentially
lower-friction and modestly softer than bare printed tread, retains the measured
external rear payload, and randomizes effective servo effort and target rate
from 70-100% to cover reasonable supply-voltage sag. It shifts the analytic
support target forward by 18 mm, targets a mild 3 degree nose-down pitch, and
explicitly penalizes motion opposite the requested forward command.

Train the continuation with:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet padded-feet-forward-bias-external-rear-payload `
  -Iterations 1000 -NumEnvs 128 -Seed 2401
```

Evaluate checkpoints at `0.005`, `0.015`, `0.030`, and `0.050 m/s`. Compare
signed progress and backward-step fraction alongside drift, pitch, support,
target-limiter backlog, and acceleration. A low battery may reduce real servo
torque and speed but does not alter battery mass or center of mass; recharge
before the first real V24 comparison and never operate below its safe cutoff.

The selected release is model 3248. Its deployment sidecar deliberately limits
commands to `0.005-0.030 m/s` and recommends `0.005 m/s`; the policy was trained
with the full `0.005-0.050 m/s` cadence range, but the highest command was not
reliably achievable under simulated target-rate sag. See
`simulation/docs/rl-padded-feet-forward-bias-v24.md` for rejected continuations,
fixed-command results, hashes, and review artifacts.

## V28 forward-biased cycle-gated straight crawl

`forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload`
keeps the statically stable RR/FR/RL/FL crawl but uses a 92 mm opposed swept
stance, 2-degree nose-down target, 46 mm stride, 24 mm lift, and cadence up to
0.80 Hz. It requires every foot to stay below 1 N for five consecutive control
updates in every complete cycle. World-frame drift, cycle-averaged lateral
velocity, heading error, more-than-one airborne foot, acceleration, and
sustained effort above the STS3215 rated torque are explicitly penalized.

The simulated motor separates the physical 90%-register transient cap from the
10 kg-cm continuous/rated threshold. Robust training correlates common torque
and rate with supply, then adds per-joint mismatch, joint-zero error, rear-COM
uncertainty, persistent lean/wrench, IMU noise/bias, delay, and common plus
per-foot Velcro-pad friction.

Train nominal adaptation first, using an explicitly selected checkpoint:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload `
  -Checkpoint <selected-checkpoint.pt> `
  -Iterations 250 -NumEnvs 4096 -Seed 2829 -V25Phase nominal
```

Do not start the robust phase or export ONNX until held-out deterministic runs
have zero falls, complete all-four release in every cycle, and meet support,
straightness, progress, acceleration, and per-joint effort gates. The profile
is currently experimental and has not replaced the Raspberry Pi policy. See
`simulation/docs/rl-forward-biased-cycle-gated-v28.md` for research, exact
configuration, rejected checkpoints, ablations, and selection criteria.

## V29 schedule-matched support straight crawl

`schedule-matched-support-straight-crawl-external-rear-payload` corrects a V28
reward loophole: the old support score counted planted feet but did not require
the scheduled swing foot and three named anchors to have the right contact
identities. V29 requires an exact contact-mask match, adds a missing-anchor
penalty and a smooth weakest-contact discovery gate, and retains the strict
five-tick, below-1-N all-four release test.

The current corrected branch also removes a leg-order-dependent cycle payout
that always favored the final front-left swing, maps a completely missing
anchor to zero progress quality, and adds short unload/touchdown transition
windows for smooth contact.

The selected reference keeps the stable opposed 92 mm stance and
`RR -> FR -> RL -> FL` order. It uses `15 / 20 mm` general/rear forward
transfer, the internally matched +2-degree stance with no independent center
offset, a 46 mm stride, 24 mm lift, 8% contact transitions, and phase fractions
`0.20 / 0.15 / 0.20 / 0.20 / 0.10 / 0.08 / 0.02 / 0.05`. Residual authority is
`0.10 / 0.12 / 0.15` for abduction/hip/knee. Commands begin at
`0.008-0.020 m/s` and expand toward the analytic `0.037 m/s` ceiling over
128,000 policy steps.

The active symmetric nominal experiment continues from rejected diagnostic
checkpoint V29 `model_4073.pt`:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet schedule-matched-support-straight-crawl-external-rear-payload `
  -Iterations 500 -NumEnvs 4096 -Seed 3007 -V25Phase nominal `
  -Checkpoint .\logs\rsl_rl\drobot_commanded_walk_v29_schedule_matched_support_straight_crawl_external_rear_payload\2026-08-30_23-54-36_manual-headless\model_4073.pt
```

Training and checkpoint evaluation are still in progress. Do not export or
deploy V29 unless deterministic held-out evaluation passes exact contact
identity, all-four release, support, progress, straightness, acceleration,
target-limiter, and per-joint effort gates. The current Raspberry Pi policy is
unchanged. See
[`../../../docs/rl-schedule-matched-support-v29.md`](../../../docs/rl-schedule-matched-support-v29.md)
for the reward bug, research basis, geometry sweeps, rejected branches,
randomization plan, and exact gates.

## V30 symmetry-gated robust straight crawl

`symmetry-gated-robust-straight-crawl-external-rear-payload` is the implemented
V30 research profile. It keeps the low opposed `RR -> FR -> RL -> FL` crawl and
50-value deployment observation, adds speed-normalized straight-progress
gating, Huber path costs, nominal left/right data augmentation without a hard
mirror loss, and targeted physical domains for lateral COM, assembly tilt,
servo zeros/rates/strength, battery sag, and Velcro-like traction. Cadence may
reach `0.85 Hz` (`0.039 m/s` with the 46 mm stride); the stance is not lowered
further because the analytic hip reference is already near its soft envelope.

The selected reference uses a `-12 mm` empirical lateral transfer and `25 mm`
forward transfer for rear-leg swings. The latter was selected over 20 and
30 mm because it improved diagonal-anchor support without the 30 mm case's
larger reversal and acceleration.

Two nominal continuations were trained on 2026-08-31. The best diagnostic is:

```text
logs/rsl_rl/drobot_commanded_walk_v30_symmetry_gated_robust_straight_crawl_external_rear_payload/
  2026-08-31_07-15-11_manual-headless/model_5000.pt
```

It is **rejected as a release policy**. Across fixed-speed held-out trials it was
fall-free, walked `0.0143-0.0151 m/s`, stayed within `0.020-0.036 m/m` lateral
error, and held RMS joint acceleration below `8.9 rad/s2`. Exact scheduled
contact remained only `0.606-0.618`, support about `0.881-0.884`, all-four
release `0.75-0.889`, and above-rated effort occupancy about `0.43`. The main
failure is front-left anchor loss during rear-right swing. Robust training was
not performed. On 2026-08-31, `model_5000.pt` was exported and selected on the
Pi at the user's explicit request for a guarded physical trial; V24 remains
installed for rollback, and this trial does not constitute acceptance.

The diagnostic video is under the same run at
`videos/play/rl-video-step-0.mp4`. See
[`../../../docs/rl-symmetry-gated-robust-straight-v30.md`](../../../docs/rl-symmetry-gated-robust-straight-v30.md)
for research, exact ranges, ablations, evaluation results, and next steps.

## V25 robust straight low-stance residual crawl

`robust-straight-low-stance-external-rear-payload` is a not-yet-trained V24
`model_3248` continuation. It adds episode-start world-path scoring and bounded
relative-yaw heading hold, mirrored per-joint strength/rate/zero-offset
randomization, 0-1 frame command delay, a modest 92 mm opposed low stance, and
a smooth distributed support push. Its joint-limit-checked 45 mm stride trains
over `0.005-0.040 m/s`; the sequential RR/FR/RL/FL contact order remains intact.

Run a 350-update nominal adaptation before enabling the robust distribution:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet robust-straight-low-stance-external-rear-payload `
  -Iterations 350 -NumEnvs 128 -Seed 2501 -V25Phase nominal

& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet robust-straight-low-stance-external-rear-payload `
  -Iterations 1250 -NumEnvs 128 -Seed 2502 -V25Phase robust
```

The workflow persists V25 curriculum age beside each checkpoint, independent
of the inherited absolute model iteration. Do not use `-Fresh`. See
`simulation/docs/rl-robust-straight-low-stance-v25.md` for exact ranges,
research rationale, export parameters, gates, and current status.

## V20 external rear-payload walking

V20 corrects the battery installation used by V19. The 144 x 68 mm holder
face is centered on the outside of the 170 x 100 mm rear plate, while its
43 mm depth projects behind the chassis. The nominal 523.18 g payload center
is `(-131.5, 0, 50) mm` in the base-link frame. This moves the combined base
COM to approximately `(-32.42, 0, 52.12) mm` and exposes the rear box as a
visible collision proxy in Isaac Sim.

The selected `model_900.pt` uses a conservative one-iteration continuation
from V19 at a 5e-5 PPO learning rate. At a 0.05 m/s command it completed three
30-second trials with zero falls and stalls, all four legs active, 0.0358 m/s
actual speed, 0.132 m mean lateral drift, and 14.388 rad/s2 joint RMS
acceleration. Longer continuations were rejected because they gained speed at
the expense of drift and jerk. See
`simulation/docs/rl-external-rear-payload-walking-v20.md` for the complete
placement, rejected-run, evaluation, artifact, and reproduction record.

## V19 smooth rear-payload walking

V19 continues V18 with the measured 416 g rear battery plus a 107.18 g
CAD-volume estimate for its printed box and lid. The task models a nominal
523.18 g rear assembly and randomizes the uncertain payload over approximately
450--600 g with small COM offsets. Its reward directly penalizes joint, body
linear, and body angular acceleration in addition to action differences, foot
slip, and touchdown impact. Speed commands are deliberately low at
0.04--0.10 m/s, but sustained progress and four per-leg touchdown metrics keep
the solution from becoming a stationary bent-knee pose.

The selected `model_899.pt` completed three deterministic 30-second trials at
a 0.05 m/s command with zero falls, no stalled five-second windows, and all four
legs active. It averaged 0.0457 m/s, 0.204 m lateral drift, 14.653 rad/s2 joint
RMS acceleration, and 0.484 m/s2 body linear RMS acceleration. See
`simulation/docs/rl-smooth-rear-payload-walking-v19.md` for the complete mass,
reward, training, evaluation, and reproduction record.

## V18 coordinated walking

V18 fixes the stationary bent-knee behavior seen in the V17 review. Its reward makes
commanded forward speed and sustained displacement the dominant objective, explicitly
penalizes two-second stalls, and adds action-rate plus action-acceleration costs for
smoothness. A 0.8-second gait clock schedules diagonal pairs: front-left with rear-right,
then front-right with rear-left. Per-leg touchdown and scheduled-contact metrics ensure
that a policy cannot score well by dragging or ignoring a leg.

The selected `model_299.pt` completed three deterministic, uninterrupted 30-second
trials with zero falls and zero stalled five-second windows. It averaged 4.789 m forward
at 0.160 m/s. Every leg remained active (38--51 touchdowns per trial), with 80--95%
scheduled-contact agreement. Mean absolute lateral displacement was 0.570 m, which is
the main remaining weakness.

The selected checkpoint is
`simulation/isaac/models/parallel-walking-v18-coordinated/model_299.pt`. Later
checkpoints were rejected because additional training increased drift and action
saturation; a hard joint-action pairing experiment was also rejected because it caused
the moving policy to freeze.

## V17 rectangular-shoe smooth-walking training

V17 retrains the sustained forward policy for the 2026-08-13 flat rectangular
shoe: a 100 x 60 x 6 mm PLA sole, a 94 x 54 x 1 mm bonded tread, and a 70.237 g
CAD mass estimate per shoe. The old spherical fork-tip collisions are disabled.
The nominal pose uses the hardware controller's 80 mm fore/aft flat-sole stance.

The actor input and output contracts remain unchanged from V16. The objective
adds second-difference action smoothing, planted-foot slip, touchdown impact,
qualified touchdown, and stronger lateral/yaw regulation. This makes the new
policy deployable through the same 48-value IMU/joint interface while asking it
to use the wider contact patch smoothly instead of learning around point feet.

The superseded V17 workflow initialized from its selected rectangular-shoe
checkpoint unless `-Fresh` was supplied. Its continuations are stored under
`logs/rsl_rl/drobot_commanded_walk_forward_v17d_rectangular_smooth_direct/`;
the earlier V17a/V17b/V17c experiment directories are retained as training
evidence but are no longer searched automatically.

## V16 sustained-walking baseline

V15 could move during an eight-second episode but its Gaussian action mean became heavily
saturated and it settled into a fixed pose after roughly ten seconds without a reset. V16
addresses that failure directly:

- the actor uses a native bounded Beta distribution, so every sampled and deployed action
  stays in `[-1, 1]`
- the timeout curriculum ramps from 8 to 32 seconds over 1,000 PPO iterations
- a two-second rolling-speed reward preserves forward motion and penalizes sustained stalls
- checkpoint selection uses uninterrupted 30-second evaluation, not training return alone
- preview uses an asset-following third-person camera so the complete robot stays visible

The selected `model_250.pt` was chosen from a 1,000-iteration, 128-environment run. In three
uninterrupted 30-second evaluations it had `0/3` falls, no stalled five-second windows, mean
forward displacement `4.443 m`, mean final five-second speed `0.142 m/s`, and mean lateral
drift `1.604 m`. The drift is the next locomotion issue to improve, but the previous
post-ten-second stop is gone.

## 1. Visible five-robot training

Start a new forward-only policy:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_visible.ps1 -Fresh
```

The defaults are five visible robots and 20 PPO iterations. Omit `-Fresh` on later runs to
continue the newest selected V18 checkpoint, falling back to the bundled coordinated
policy for the first continuation. `-Fresh` creates a separate run; it does not erase an
existing model. Override the defaults with `-Iterations` and `-NumEnvs`.

## 2. Headless parallel training

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -Iterations 500 -NumEnvs 128
```

This resumes the newest accepted forward checkpoint automatically. The repository bundles
the selected coordinated rectangular-shoe policy at
`simulation/isaac/models/parallel-walking-v18-coordinated/model_299.pt`, which is the clean-checkout
fallback. To continue that exact policy explicitly:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v18-coordinated\model_299.pt `
  -Iterations 500 -NumEnvs 128
```

The main improvement signals are:

- `Metrics/min_rolling_forward_speed_m_s`: should remain above `0.04 m/s`
- `Metrics/sustained_stall_rate`: should approach zero
- `Metrics/mean_velocity_error_m_s`: should decrease
- `Metrics/mean_commanded_speed_m_s`: should approach `0.15 m/s`
- `Metrics/net_forward_displacement_m`: should keep increasing with the horizon
- `Metrics/net_lateral_displacement_m`: should stay close to zero
- `Metrics/current_episode_horizon_s`: ramps from 8 to 32 seconds
- `Metrics/action_saturation_rate`: should stay low rather than approach one
- `Metrics/qualified_touchdowns_per_episode`: should be non-zero as steps emerge
- `Metrics/qualified_touchdowns_<leg>`: all four values should remain non-zero
- `Metrics/fall_rate`: should approach zero
- `Mean reward`: should rise, but is secondary to physical displacement and stalls

Evaluate a checkpoint without reset-dependent transients:

```powershell
& .\simulation\isaac\rl\parallel_walking\evaluate_walking_sustained.ps1 `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v18-coordinated\model_299.pt `
  -Seconds 30 -Episodes 3
```

## 3. Preview one robot

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 -Command forward
```

For a continuous preview that resets only on a fall, not on a training timeout:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -Command forward -NoTimeLimit
```

Close Isaac Sim to stop an unlimited preview. To record exactly 30 seconds with the
asset-following camera:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -Command forward -NoTimeLimit -RecordSeconds 30
```

The selected coordinated review clip is
`simulation/reviews/parallel-walking-v18-coordinated-model299-30s.mp4`
(1,800 frames at 60 fps). The older V16 clip remains available for comparison.
Underscore-prefixed calibration and workflow directories are ignored during automatic
checkpoint selection.

## Later: backward and turns

Once forward walking and heading retention are stable, initialize the directional curriculum:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet directional `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v18-coordinated\model_299.pt `
  -Iterations 500 -NumEnvs 128
```

After that first transfer, repeat the command without `-Checkpoint` to resume the newest
directional run. Preview learned commands with `-Command left`, `right`, or `backward` and
`-CommandSet directional`. Left and right mean yaw turns with a small forward velocity; the
reserved lateral command remains zero until a strafing curriculum is intentionally added.
