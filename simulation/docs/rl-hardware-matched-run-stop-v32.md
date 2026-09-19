# V32 hardware-matched run and graceful stop

## Status

V32 is the corrective continuation after the first V31 real-robot trial. It is
not yet a deployment policy. V30 remains the known slow fallback, and the
rejected V31 model remains available only for comparison.

Identifiers:

- task: `Drobot-Commanded-Run-Hardware-Matched-Stop-External-Rear-Payload-Direct`;
- command set: `hardware-matched-run-stop-external-rear-payload`;
- experiment: `drobot_commanded_run_v32_hardware_matched_stop_external_rear_payload`;
- bootstrap: V31 `model_1650.pt`.

The full input-trace analysis is in
`debug/walk-trials/20260902T040348286640Z-9f4db9ba/ANALYSIS.md`.

## Why V31 failed on the robot

V31 looked fast and straight in Isaac Lab, but its candidate never passed the
sustained-effort gate. The real trace clarified the mechanism. At a requested
`0.08 m/s`, the policy drove all hip and knee targets to the `240 deg/s` slew
limit. Loaded encoder motion reached about `77-115 deg/s`; periodic target
following lagged `0.27-0.47 s`. That is a large part of the `0.90-2.40 Hz` V31
gait cycle, so intended diagonal pairs were no longer diagonal in physical
time. The result was alternating leg motion with little propulsion.

Source inspection found a second contract error before V32 qualification:
V31 inherited V28's roughly `35 deg`, `2 deg` nose-down reset/default stance,
while its ONNX sidecar and the Raspberry Pi converted direct actions around
the calibrated symmetric `30 deg` center stance. Because V31 uses direct
actions, the same network output therefore represented different absolute
joint targets in Isaac Lab and on the robot. V32 explicitly uses the `30 deg`,
`0.375 m`, level center geometry for simulation reset, direct-action zero,
stopped hold, model export, and hardware startup. The preliminary V32
continuation made before this discovery is marked rejected and will not be
evaluated or exported.

The trace did not indicate reversed joints or an IMU sign error. Every joint
had positive delayed correlation, joint and IMU samples were fresh, total gait
phase was correct, and the real approximately `+g` stationary acceleration
matches Isaac Lab's accelerometer convention.

The policy callback also contended with servo-bus reads and writes through the
main session lock. Its effective inference rate fell to `50.24 Hz`, with 65
missed policy deadlines. This is corrected separately in the onboard runtime;
training still assumes a 60 Hz inference contract and randomized executable
target/actuator response.

## Runtime correction

Policy inference now publishes a complete target vector to a dedicated
one-element latest-value buffer. The servo owner consumes the newest vector on
its output tick. It never replays an accumulated target queue, and inference no
longer waits for the USB bus.

New trial telemetry separates:

- observation read duration;
- ONNX inference duration;
- policy-to-output handoff duration;
- target snapshots replaced before physical output;
- physical group-write duration and actual target age; and
- desired versus sent servo targets for each output packet.

This makes a future slow bus distinguishable from a slow inference loop and
from mechanically delayed joints.

## Training contract

V32 retains the V31 run, smooth-or-abrupt stop, and stable four-foot hold. The
main differences are deliberately conservative:

| Parameter | V31 | V32 |
| --- | ---: | ---: |
| Command range | `0.05-0.20 m/s` | `0.04-0.12 m/s` |
| Extra low-band sampling | `0.05-0.10 m/s` | `0.04-0.08 m/s` |
| Gait frequency | `0.90-2.40 Hz` | `0.70-1.30 Hz` |
| Diagonal duty factor | `60%` | `68%` |
| Reference lift | `26 mm` | `22 mm` |
| Target slew | `240 deg/s` | `120 deg/s` |
| Target step at 60 Hz | `4 deg` | `2 deg` |
| Heading hold | `1.5 s^-1`, `0.20 rad/s` | `0.75 s^-1`, `0.12 rad/s` |
| Direct-action neutral | inherited approximately `35 deg` in simulation; `30 deg` in export | calibrated `30 deg` everywhere |

Robust training samples common target rates from `70-105%` and per-joint rates
from `78-108%` of the V32 limit, plus common/per-joint effort, `55-110%`
stiffness, `85-130%` damping, one frame of target delay, servo-zero offsets,
rear-payload uncertainty, Velcro-like surface variation, and persistent body
forces/torques. Nominal training pins the physical domains while the inherited
actor first adapts to the new motion contract.

V32 strengthens target-limiter, action-saturation, action/joint/body
acceleration, diagonal-sync, support-slip, touchdown-impact, and above-rated
effort costs. Positive speed and straight progress remain material so the new
costs cannot be satisfied by standing still.

## Candidate gates

Evaluate at `0.04`, `0.06`, `0.08`, and `0.10 m/s`. Each speed needs a
`0.60 s` ramped stop and an abrupt stop before robust continuation. Existing
fall, command tracking, per-leg release, paired contact, straightness, heading,
stop, acceleration, and effort gates remain mandatory. V32 also requires:

- run action saturation at or below `5%` of joint samples;
- run requested-to-processed limiter occupancy at or below `10%`; and
- run target-velocity-ceiling occupancy at or below `15%`.

These gates directly reject the V31 hardware failure instead of relying on a
visually convincing simulated run.

## First nominal continuation result

The corrected-neutral continuation from V31 `model_1650.pt` produced a useful
candidate pool through iteration 2149, but no checkpoint was accepted. At the
fixed `0.06 m/s` five-episode ramp-stop evaluation, iteration 1875 achieved:

- `0.06488 m/s` mean run speed, zero falls, and only `0.0299` absolute lateral
  distance per forward distance;
- scheduled release rates of `97.5% / 74.0% / 93.5% / 86.7%` for
  FL/RL/FR/RR and paired-release quality `0.860`;
- `0.62 s` mean settle time, `99.9%` stable hold, and `0.00060 m/s` final
  speed; and
- low simulated acceleration: `11.47 rad/s2` joint RMS, `0.354 m/s2` body
  linear RMS, and `2.72 rad/s2` body angular RMS.

It nevertheless failed the deployment-specific gates: `31.8%` of joint-time
was above rated effort, `7.05%` of run actions were saturated, `24.0%` of run
targets differed from their executable rate-limited target by more than
`0.5 deg`, and `30.3%` of run joint motion sat at the packet-rate ceiling.
Iteration 2050 was worse: `38.5%` effort occupancy, `7.70%` action saturation,
and only `65.1%` rear-left scheduled release.

Per-joint telemetry localized the failure. The two front hip-flexion actions
account for most action saturation, while every knee is target-limited for
`44.9-59.0%` of run samples and sits at the velocity ceiling for `55.2-66.4%`.
Effort above the rated threshold is concentrated in sagittal joints and also
persists in an asymmetric diagonal set during the stopped hold.

The next nominal stage therefore resumes from iteration 1875 with stronger
penalties for requested/executable target disagreement, saturation, action and
joint acceleration, doubled above-rated effort cost, and a continuous
rated-effort-normalized RMS cost. A quadratic stopped-action penalty complements
the Gaussian neutral reward so far-from-center terminal actions retain a useful
gradient. Paired and least-active-foot release rewards are strengthened at the
same time so lower effort cannot be obtained by dragging a leg or standing.

## Commands

Nominal continuation:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet hardware-matched-run-stop-external-rear-payload `
  -Checkpoint .\simulation\isaac\models\parallel-running-v31-fast-stop\model_1650.pt `
  -Iterations 500 -NumEnvs 4096 -Seed 3201 -V25Phase nominal
```

Example deterministic evaluation:

```powershell
& .\simulation\isaac\rl\parallel_walking\evaluate_run_stop.ps1 `
  -CommandSet hardware-matched-run-stop-external-rear-payload `
  -Checkpoint <candidate.pt> -Speed 0.06 `
  -RunSeconds 4 -RampSeconds 0.60 -HoldSeconds 3 `
  -Episodes 5 -DomainMode nominal -Output <report.json>
```

Set `-RampSeconds 0` for the abrupt-stop case. Only continue a passing nominal
checkpoint with `-V25Phase robust`; reaching the requested PPO iteration count
is not acceptance.

## Deployment boundary

Do not install a V32 checkpoint merely because training finishes. Export only
after the fixed-speed gates pass, preserve V30 as a selectable rollback, and
begin the next real test supported at the selected model's minimum speed for a
short duration. Compare the new `motor_output` timing events with encoder lag
before any free-standing run.
