# V20 real-walk trial 20260829T204215017854Z-299dd310

## Conclusion

This trial clears the previously suspected Raspberry Pi scheduling problem but
exposes three larger control-contract mismatches. The most important is the
startup pose: the dashboard requires all twelve joints at calibrated zero,
while V20 was trained from a crouched neutral pose with the hip-flexion and
knee joints at approximately `+/-30 degrees`. The policy therefore begins from
an observation about 30 degrees outside its nominal state and immediately
tries to enter the crouch while its 1.25 Hz gait clock is already moving.

The second mismatch is target dynamics. Training permits approximately 4.38
degrees of target motion per 60 Hz step, while the physical runtime now permits
at most 2 degrees per step. Even after the 1.2-second startup interval, the
policy remains ahead of the limiter on several hip-flexion and knee joints, and
the loaded hardware then trails the limited targets further.

The third issue is gait choice. V20 is a fixed-cadence diagonal-pair trot with a
65% duty factor. It is not a slow, three-support-leg crawl. The user's stable
hardcoded crawl provides strong evidence that a sequential learned crawl is a
better target for this heavy, rear-loaded robot than trying to make the current
trot arbitrarily slow.

Do not raise the real-hardware target-step cap or silently slow V20's gait
clock. Both would change the dynamics seen by a policy that was not trained for
them. Correct the RL startup stance first, then train a deployment-matched
low-speed policy.

## Repository artifacts and privacy processing

This directory contains:

- `20260829T204215017854Z-299dd310.zip`: the original automatic controller
  recording;
- `IMG_0834.MOV`: a repository-safe copy containing only the HEVC picture and
  primary AAC audio;
- `video-contact-sheet.jpg`: one-frame-per-second overview;
- `rl-motion-contact-sheet.jpg`: four-frame-per-second view around the RL run;
- `analyze_trial.mjs`: dependency-free numerical analysis of an extracted
  recording; and
- `analysis-metrics.json`: reproducible full-precision results from that script.

The source phone file contained precise QuickTime location, location accuracy,
device, software, timestamp, positional-audio, and timed-metadata tracks. The
repository copy was stream-copied with all global metadata and chapters removed
and only stream 0 (video) plus the primary audio stream retained. A second
FFprobe inspection found two streams only and no location, device, or creation
metadata. The original attachment was not modified.

The extracted JSON telemetry was also scanned for GPS/location fields, local IP
addresses, SSIDs, passwords, and tokens; none were found. The archive's normal
Raspberry Pi repository paths under `/home/rd/drobot2` are retained.

SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| Sanitized `IMG_0834.MOV` | `f45d2262fb5c2fcaf69f5859a6b41b77cbbd68020371f3e5b6bc1ec85d6bd40f` |
| Recording ZIP | `99ecb1fb4fcccf6ba43e981006672d2aa20024d7840b8f23e4df6dfde8edcc06` |

The phone recording began at `20:42:10Z`. Controller recording began at
`20:42:15.017854Z` and ended at `20:42:20.023017Z`, so the internal trial maps
to approximately 5.018 through 10.023 seconds in the video.

## Trial identity

| Item | Value |
| --- | ---: |
| Model | V20 `model_900.onnx` |
| Model SHA-256 | `f787669d33115f117ac691c1ba7bb145fa6a726c6a5d22aec7aecc9e284d6529` |
| Requested command | `0.04 m/s` forward, zero lateral/yaw |
| Duration | `5.0 s` |
| Samples | `300` |
| Gait clock | fixed `1.25 Hz` / `0.8 s` period |
| Clock cycles during trial | `6.25` |
| Recorded total robot mass | `3.175 kg` |
| Rear payload model | `0.52318 kg` at `[-0.1315, 0, 0.05] m` |
| Result | completed, no software fault |

## What the controller fixes accomplished

The real-time loop is no longer the leading explanation for this walk:

| Signal | This trial | Earlier baseline |
| --- | ---: | ---: |
| Achieved policy rate | `60.10 Hz` | nominal average `60 Hz` |
| Policy interval p50 / p95 / max | `16.76 / 20.16 / 22.62 ms` | p95 about `34 ms` |
| Intervals below 10 ms | `1` | `93` |
| Intervals above 25 ms | `0` | `85` |
| Missed deadlines | `0` | catch-up behavior present |
| Distinct joint feedback | `299` samples / `59.92 Hz` | about `15.1 Hz` |
| Joint sample age p95 | `10.77 ms` | synchronous blocking path |
| Dropped samples / events | `0 / 0` | `0 / 0` |

The corrected scheduler, background encoder feedback, and synchronized writes
are working. Feedback did not reach the requested 100 Hz, but approximately 60
Hz with a 10.8 ms p95 age is sufficient to show that the observed gait failure
is not another 15 Hz feedback or catch-up-burst problem.

Sparse round-robin diagnostics reported `11.6-12.0 V`, `33-41 C`, and a maximum
sampled current of `429 mA`. They show no voltage collapse or thermal event in
this five-second run. They are too sparse to rule out short current peaks, but
power is not the primary explanation for this trial.

## Startup-pose mismatch

The simulation reset pose is:

- all hip-abduction joints: `0 degrees`;
- front hip flexion / knees: `+30 / -30 degrees`;
- rear hip flexion / knees: `-30 / +30 degrees`; and
- reset joint noise: only `+/-0.015 rad`, or about `+/-0.86 degrees`.

The recording begins with every real joint within 0.4 degrees of calibrated
zero. The flexion/knee state is therefore roughly 35 reset-noise widths away
from training. The first inference illustrates the consequence:

| Joint group | Initial measured | First policy request | First limited command |
| --- | ---: | ---: | ---: |
| Front hip flexion | about `0 degrees` | about `+30 degrees` | about `+2 degrees` |
| Rear hip flexion | about `0 degrees` | about `-30 degrees` | about `-2 degrees` |
| Front-left knee | `-0.4 degrees` | `-30.0 degrees` | `-2.4 degrees` |
| Rear-left knee | `0.0 degrees` | `+14.9 degrees` | `+2.0 degrees` |
| Front-right knee | `0.0 degrees` | `-47.8 degrees` | `-2.0 degrees` |
| Rear-right knee | `+0.3 degrees` | `+9.1 degrees` | `+2.3 degrees` |

The asymmetric knee requests are not a motor-direction error: they are the
model's response to an out-of-distribution initial observation. The video shows
the matching behavior as the body drops into a crouch and starts stepping at
the same time.

The dashboard should replace **CENTER ALL 12 is RL-ready** with a distinct
**PREPARE RL STANCE** operation. It should move all joints together from center
to the model-declared neutral pose, wait until feedback is close and settled,
then reset action history and gait phase before starting inference. The neutral
pose belongs in model metadata rather than being duplicated in UI code.

## Target tracking and policy behavior

Training and deployment currently use different target slew limits:

- Isaac environment: `4.5836625 rad/s`, approximately `262.6 deg/s` or `4.38
  degrees` per 60 Hz action;
- physical runtime: at most `2 degrees` per update, approximately `120 deg/s`
  at a steady 60 Hz.

Across 285 of 300 samples, at least one raw policy target was more than 2
degrees ahead of the limited target. This is not only startup. After the
training profile's 1.2-second gait ramp, the limiter gap exceeded 2 degrees on
39-47% of hip-flexion samples and 41-55% of knee samples for several joints.

Steady-period p95 target-to-measured errors were:

| Group | p95 absolute error range | Estimated target-following lag |
| --- | ---: | ---: |
| Hip abduction | `6.6-10.4 degrees` | `167-200 ms` |
| Hip flexion | `9.8-27.4 degrees` | `200-333 ms` |
| Knee | `16.9-35.1 degrees` | `300-367 ms` |

The lag estimate is a periodic cross-correlation diagnostic, not a pure motor
transport delay. It combines the servo, mechanical load, target limiter, and
closed-loop policy response. Nevertheless, correlations of `0.80-0.94` show a
consistent delayed version of the commanded gait rather than random encoder
noise.

The policy also spends substantial time at its normalized action boundary:

- front-right hip abduction: `50.7%` of samples at `|action| >= 0.95`;
- rear-left hip abduction: `27.7%`;
- front-left knee: `31.0%`; and
- rear-left knee: `36.7%`.

That asymmetric saturation with zero lateral/yaw command is evidence that V20
is fighting the real state instead of executing a comfortable nominal gait.
The V20 simulator report had already accepted some residual lateral drift; the
real startup and actuator mismatches amplify it.

## IMU and video evidence

The synchronized video shows repeated body rocking and heading oscillation,
not a single clean turn. The internal measurements agree:

| Measurement | Result |
| --- | ---: |
| Roll range | `-5.05` to `+3.07 degrees` |
| Pitch range | `-2.89` to `+7.63 degrees` |
| Absolute yaw-rate p95 / max | `1.073 / 1.422 rad/s` |
| Integrated net yaw | about `+0.4 degrees` |
| Integrated absolute yaw travel | about `137.8 degrees` |
| Accelerometer norm p95 / max | `13.06 / 18.47 m/s2` |
| Consecutive acceleration-vector change p95 | `4.05 m/s2` |

Near-zero net yaw with very large absolute yaw travel is the numerical signature
of the visible left-right wobble. The robot did not simply choose the wrong
heading; it repeatedly corrected past it.

## Recommended implementation order

### 1. Correct deployment startup before another model judgment

Add a model-specific RL-ready stance and a two-stage start:

1. arm and hold calibrated center;
2. smoothly move to the metadata-declared V20 neutral pose at a conservative
   all-joint rate;
3. require measured position near that pose and low measured joint velocity for
   a short settling interval;
4. reset policy action history and phase to zero; and
5. start the timed policy run.

Keep normal completion's smooth return to calibrated center and torque hold.
Keep the 2-degree hardware cap for now; raising it would make this already-fast
trial more violent. Repeat only a short supported V20 trial after the startup
fix. That isolates startup from retraining.

### 2. Make the training actuator contract match deployment

For the next policy, separate the physical servo velocity used in observations
from the target slew limit. Train with the actual `2 degrees / 60 Hz` target
cap, and penalize the gap between the policy-requested target and the processed
target so the network learns executable commands.

Use controlled real pose sweeps to estimate unloaded and loaded response before
choosing simulated actuator delay/gain distributions. The correlations above
are useful bounds but should not be copied directly as a pure 200-350 ms delay.
Randomize torque/voltage capability, actuator response, sensor age, shoe
friction, payload mass, and rear COM within measured ranges.

### 3. Replace fixed-cadence V20 at low speed

The existing V21 profile already maps `0.005-0.10 m/s` to `0.35-1.25 Hz`. At a
`0.04 m/s` command it would use about `0.682 Hz` (a `1.47 s` period) and about a
`32.4 mm` reference stride, instead of V20's `1.25 Hz`, `0.8 s` period, and
55 mm reference stride. This is directionally correct, but V21 still inherits
the diagonal-pair trot.

For the user's smoothness-first goal, train and compare a sequential crawl
candidate using the now-successful hardcoded order:

1. rear right;
2. front right;
3. rear left; and
4. front left.

Use quarter-cycle offsets and a duty factor high enough that normally only one
foot swings while the other three remain support candidates. Replace the
diagonal-sync objective with three-support/contact-margin objectives. Retain
straightness, yaw-rate, body-tilt, action-rate, action-acceleration, joint/body
acceleration, slip, and touchdown penalties. Speed should remain secondary.

### 4. Gate deployment on real-contract metrics

Do not select a checkpoint only because it survives in Isaac. A candidate
should be rejected if it exhibits persistent action saturation, large
requested-to-applied target gaps, excessive tracking lag, yaw oscillation, or
two-leg support behavior inconsistent with the chosen crawl contract. Run the
same metrics at `0.005`, `0.01`, `0.02`, and `0.04 m/s`, then begin physical
evaluation with a stationary RL-stance hold followed by a short supported run.

## Current change and verification status

The follow-up implementation replaces the zero-centered RL start interlock with
a model-declared **PREPARE RL STANCE** operation. V20 metadata now carries its
actual `+/-30 degree` flexion/knee neutral, action scales, 45 deg/s preparation
ramp, 0.5-second settle, five-degree measured-position tolerance, and two-degree
packet cap. Start also requires one synchronized all-motor feedback sample with
low encoder speed. The policy observation and output conversion use the same
metadata rather than duplicated constants.

The new V22 training profile implements the recommended deployment-matched
residual crawl. It uses the exact stable distributed-push RR -> FR -> RL -> FL
hardware sequence as its base target and limits PPO to a 25% correction around
that sequence. Cadence is 0.06-0.30 Hz for `0.003-0.015 m/s` commands, the full
50 mm step remains available at low speed, and target slew is exactly 120 deg/s
(`2 degrees / 60 Hz`). Reward terms favor three-foot support and balanced
four-leg participation while penalizing excess airborne feet, target-limiter
backlog, acceleration, slip, touchdown, tilt, yaw, and lateral motion. The
exported metadata carries the same reference table so the Pi does not mistake
the learned residual for a complete joint target. Training/evaluation results
and deployment status are documented separately after qualification; this
report remains the immutable before-fix trial analysis.

## Qualification outcome

The final V22 formulation uses the distributed crawl as a deterministic base
and learns only a 25%-scale residual. A fresh 1,000-iteration run completed,
and checkpoint 500 was selected over checkpoints 750 and 999 because it was
smoother and drifted less at the same command. Across three 30-second episodes
at each of `0.003`, `0.005`, `0.010`, and `0.015 m/s`, it had zero falls,
zero stalls, zero action saturation, and touchdowns from all four legs. Joint
RMS acceleration ranged from `4.29` to `8.36 rad/s2`, compared with about
`14.39 rad/s2` for the selected V20 simulation checkpoint. Three-or-four-foot
support ranged from `96.3%` at the minimum command to `80.2%` at the maximum.

The full training search, rejected formulations, exact metrics, selected
artifacts, and disclosed low-speed overshoot are in
`simulation/docs/rl-low-speed-residual-crawl-v22.md`. The new Isaac result is a
qualified commissioning candidate, not proof of real-floor stability; the
first physical trial should remain supported at `0.003 m/s` and be reviewed
with the automatic IMU/joint recording.
