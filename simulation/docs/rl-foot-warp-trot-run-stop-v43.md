# V43 foot-warp trot with residual stabilization

## Status

V43 is the first guarded hardware candidate produced after the failed V31
real-robot run. It is not yet hardware-validated. It replaces V31 as the
dashboard's default running-policy choice, while V30 remains available as the
known stable crawl fallback.

Identifiers:

- task: `Drobot-Commanded-Run-Foot-Warp-Trot-Residual-Stop-External-Rear-Payload`;
- command set: `foot-warp-trot-residual-run-stop-external-rear-payload`;
- experiment: `drobot_commanded_run_v43_foot_warp_trot_residual_stop_external_rear_payload`;
- selected checkpoint: `model_399.pt` from
  `2026-09-02_12-42-55_manual-headless`.

The first physical attempt must be supported or tethered, powered from the
bench supply, run at `0.10 m/s`, and limited to a few seconds. Passing the
Isaac gates does not authorize an unattended or free-running trial.

## Why V31 failed and why V43 is different

The V31 telemetry showed useful software timing but infeasible physical
targets. Its hips and knees repeatedly requested `240 deg/s`, while the loaded
robot achieved roughly `80-120 deg/s` and lagged the requested cycle by
`0.20-0.47 s`. The intended diagonal pairs therefore became physically out of
phase and mostly swung sideways.

Several direct-action and integrated-velocity continuations reduced individual
symptoms without producing a convincing combination of speed, straightness,
four-leg release, smoothness, and stop behavior. V43 instead starts from the
stable V40 `0.10 m/s` learned cycle and gives PPO only small residual authority.
This follows the same general principle as policies that modulate a trajectory
generator: preserve a useful periodic prior and learn corrections from IMU and
joint feedback rather than relearning every joint target from scratch.

Relevant background includes [policies modulating trajectory
generators](https://research.google/pubs/policies-modulating-trajectory-generators/),
[learning agile locomotion with actuator and latency
models](https://research.google/pubs/sim-to-real-learning-agile-locomotion-for-quadruped-robots/),
and [learned gait transitions](https://research.google/pubs/fast-and-efficient-locomotion-via-learned-gait-transitions/).

## Exact gait contract

The source V40 cycle is transformed in sagittal foot space rather than by
multiplying joint angles. This matters because joint-angle scaling distorted
the foot path and produced poor propulsion in the rejected V41/V42 trials.

V43 uses two exact 256-sample phase tables:

| Command endpoint | Fore/aft cycle scale | Cadence | Table |
| --- | ---: | ---: | --- |
| `0.10 m/s` | `1.00` | `1.15 Hz` | `v43_v40_low_foot_warp_duty0p46_scale1p00_lift0p030.json` |
| `0.14 m/s` | `1.20` | `1.60 Hz` | `v43_v40_low_foot_warp_duty0p46_scale1p20_lift0p030.json` |

The controller linearly blends the two tables using planar command speed. The
same interpolation, floor-indexed phase lookup, startup ramp, and integrated
gait clock are implemented in Isaac Lab and the Raspberry Pi runtime.

Additional fixed geometry and timing:

- diagonal pairs use phase offsets `(0.0, 0.5, 0.5, 0.0)`;
- duty factor is `0.46`;
- stance is crouched by `22 mm` relative to the extracted cycle;
- the smooth swing path receives `30 mm` of extra clearance;
- gait-reference authority ramps in over `1.20 s`;
- the gait clock parks inside the `0.004 m/s` standstill deadband;
- the command band is exactly `0.10-0.14 m/s` and the UI default is `0.12 m/s`.

Reference hashes:

- high table: `1ff1494a172124f1c8943e3bb99b8be75b41442f13187e6c8db3c93c8ebf5f69`;
- low table: `7ac46755e3554f2e5781f5c5a6fd2af5bbfbe96880ec374a5683eeef01274fea`.

The exporter refuses a V43 package if either table hash, the actor task,
cadence, phase layout, speed range, action semantics, or actuator limits differ
from the evaluated contract. It embeds both full tables in the ONNX JSON
sidecar; deployment does not regenerate or approximate the foot path.

## Policy and actuator contract

The actor keeps the deployable 50-input, 12-output bounded-Beta network. Its
observations are command, gait phase, IMU angular velocity, projected gravity,
IMU linear acceleration, joint position error, joint velocity, and the prior
controller-applied normalized target.

The phase table supplies the base target. PPO can correct it by only:

- `0.04` of the four abduction action envelopes; and
- `0.12` of the eight sagittal hip/knee envelopes.

At zero command residual authority is exactly zero, matching training. The
runtime then rate-limits all joints toward calibrated center instead of letting
the actor move the legs after the stop.

Target velocity is limited to `60 deg/s` on abduction and `180 deg/s` on
sagittal joints, with an additional `2 deg` per-packet limit at `60 Hz`.
Relative-yaw heading hold uses `0.75 s^-1` gain and at most `0.12 rad/s`
correction. Command observations use the trained `(10, 10, 10)` scaling.

## Reward and domain design

The gait table is deliberately dominant, but PPO is rewarded for improving it
under actuator and physical variation. The V43 objective includes:

- rolling and instantaneous forward-speed tracking;
- a narrow world-frame lateral corridor and aligned-progress reward;
- heading and yaw-rate penalties;
- hard four-leg release gating and paired diagonal release quality;
- gait-reference tracking with strong abduction magnitude suppression;
- action rate, action acceleration, joint acceleration, body linear/angular
  acceleration, slip, and touchdown penalties;
- coupled torque-speed operating-envelope cost and per-leg operating-point
  imbalance; and
- stopped velocity, position, upright stance, and four-foot contact terms.

Randomized environments vary individual actuator rate, strength, stiffness,
damping, one-step control latency, joint zero, traction, rear-payload pose,
body force/torque, and supply-strength proxies. The nominal anchor population
prevents the robust domain from trading away the clean baseline gait.

## Candidate selection

Before PPO, the blended zero-residual reference was fall-free and produced:

| Command | Mean speed | Final-window speed | Absolute lateral / forward | Final heading |
| --- | ---: | ---: | ---: | ---: |
| `0.10 m/s`, nominal | `0.1076 m/s` | `0.1178 m/s` | `0.0273` | `3.93 deg` |
| `0.12 m/s`, nominal | `0.1172 m/s` | `0.1270 m/s` | `0.0206` | `4.48 deg` |
| `0.14 m/s`, nominal | `0.1246 m/s` | `0.1348 m/s` | `0.0051` | `3.77 deg` |

The first PPO pass used 256 environments for 400 iterations with a 75% nominal
and 25% randomized bridge. At the fixed nominal `0.14 m/s` screen, model 399
reduced final heading from `3.65` to `1.74 deg` while retaining `0.1202 m/s`.
Against the same randomized seeds, it reduced low-speed path drift from
`0.0450` to `0.0233 m/m` and high-speed drift from `0.0728` to `0.0472 m/m`.
No screened episode fell.

A second 400-iteration continuation used 75% randomized environments. Models
425 through 798 all lost low-speed straightness or speed relative to model 399,
so that continuation is rejected. More training was not assumed to mean a
better policy.

Selected sustained screens:

| Case | Mean speed | Lateral / forward | Heading | RMS linear acc. | RMS angular acc. | RMS joint acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.10 m/s`, randomized (3) | `0.1017` | `0.0233` | `9.56 deg` | `1.122 m/s2` | `5.500 rad/s2` | `13.95 rad/s2` |
| `0.14 m/s`, nominal | `0.1202` | `0.0098` | `1.74 deg` | `0.895 m/s2` | `5.662 rad/s2` | `17.33 rad/s2` |
| `0.14 m/s`, randomized (3) | `0.1159` | `0.0472` | `7.55 deg` | `1.218 m/s2` | `6.762 rad/s2` | `17.39 rad/s2` |

## Phase-aligned graceful stop

Starting the `0.90 s` deceleration at an arbitrary gait phase settled only 25%
of eight nominal episodes because one diagonal transition could begin just as
the command shrank. Starting near phase `0.05` cycles changed the result to
8/8 settled, with 95.7% four-foot contact and 86.7% stable hold.

The selected normal stop therefore:

1. keeps the run command until gait phase is within `0.03` cycles of `0.05`;
2. waits at most `0.50 s` for that alignment;
3. applies a `0.90 s` cubic-smoothstep command ramp;
4. suppresses actor residuals at zero command; and
5. holds centered torque for `2.0 s`.

| Stop case at `0.14 m/s` | Settled | Stable hold | Four-foot hold | Mean settle | Final speed | Stop error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal (8) | `100%` | `86.7%` | `95.7%` | `1.465 s` | `0.0002 m/s` | `0.0175 m` |
| randomized (5) | `100%` | `86.4%` | `91.2%` | `1.307 s` | `0.0002 m/s` | `0.0296 m` |

The ONNX sidecar owns these values. The dashboard reads them rather than
hardcoding a V43-specific stop sequence.

## Remaining risks

V43 is substantially more executable than V31, but it is still a guarded
candidate:

- randomized `0.10 m/s` heading error reached `9.56 deg` in the three-seed
  screen;
- randomized stop evaluation had `13.64 deg` p95 joint tracking error versus
  the evaluator's conservative `12 deg` gate;
- roughly `37%` of joint-time remained above the continuous-rated effort soft
  threshold, although action saturation was zero and target limiting was about
  `5.5%`;
- the high phase table contains geometric segments faster than the configured
  target limit, so the evaluated limiter remains part of the behavior; and
- no simulator can establish the real Velcro contact, battery voltage, servo
  thermal response, assembly asymmetry, or cable drag exactly.

These warnings are why the first test begins at `0.10 m/s` on external power.
The automatic real-walk recorder should be downloaded together with a fixed
camera video before increasing speed. A fall, repeated shoe drag, rising
tracking delay, or strong left/right oscillation rejects the candidate without
trying to compensate by increasing the UI command.

## Exported artifact

The Raspberry Pi candidate is:

```text
onboard/models/parallel-running-v43-foot-warp-trot-residual/model_399.onnx
onboard/models/parallel-running-v43-foot-warp-trot-residual/model_399.json
```

Hashes and parity:

- checkpoint SHA-256: `2e60fa7dd7167024bee71ab77a2c399d8f0b22a399576ce7957037871fcb7af7`;
- ONNX SHA-256: `85e914c75c9f8f6e815233bccacb283757c64e50440b41e8b92373fdbd4bcb83`;
- 512-sample PyTorch/ONNX maximum absolute action error: `2.3841858e-7`;
- mean absolute action error: `2.3884544e-8`.

The parity report is
`simulation/isaac/models/parallel-running-v43-foot-warp-trot-residual/exported/model_399_onnx_parity.json`.

The selected checkpoint is preserved outside the training-log tree at:

```text
simulation/isaac/models/parallel-running-v43-foot-warp-trot-residual/model_399.pt
```

Re-export the complete ONNX package and its JSON sidecar with the versioned
wrapper below. It supplies every behavior-setting argument explicitly; the
exporter independently rejects a mismatched task, reference hash, cadence,
action semantic, speed range, or actuator limit.

```powershell
& .\simulation\isaac\rl\parallel_walking\export_v43_policy.ps1
```

Pass `-Checkpoint` or `-Output` only when deliberately evaluating another
checkpoint or writing a comparison artifact.

The 20-second review clip runs at `0.14 m/s` for 15 seconds, aligns the
deceleration to phase `0.05`, ramps for `0.90 s`, and holds centered for the
remaining frames:

```text
simulation/reviews/parallel-running-v43-foot-warp-trot-residual-model399-20s.mp4
```

It contains 1,200 H.264 frames at 60 fps and has SHA-256
`be4381e92905ab4d0e6122de7f2b0369d14091c2742e9da287401dfd33598a16`.
Reproduce it with:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -CommandSet foot-warp-trot-residual-run-stop-external-rear-payload `
  -Checkpoint <model_399.pt> -Command forward -ForwardSpeed 0.14 `
  -RunThenStop -RunSeconds 15 -StopRampSeconds 0.90 `
  -RampStartPhaseCycles 0.05 -RampStartPhaseToleranceCycles 0.03 `
  -MaximumPhaseAlignmentWaitSeconds 0.50 -RecordSeconds 20 -NoTimeLimit
```

No Raspberry Pi pull, service restart, CAN/serial access, or physical motion is
part of creating this candidate.

## Reproduce the training continuation

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet foot-warp-trot-residual-run-stop-external-rear-payload `
  -Iterations 400 -NumEnvs 256 -Seed 7975 -V25Phase bridge `
  -Checkpoint .\simulation\isaac\models\parallel-running-v43-foot-warp-trot-residual-bootstrap\model_0.pt
```

Do not select the final iteration automatically. Compare fixed seeds across
checkpoints with `evaluate_sustained_walking.py`, then run both aligned and
unaligned stop matrices with `evaluate_run_stop.py`. The rejected robust
continuation is retained under `2026-09-02_13-08-40_manual-headless` so its
regression remains auditable.

## First hardware trial

### Raspberry Pi deployment — 2026-09-05

V43 was uploaded to `/home/rd/drobot2` on `pi5-dog.local` at the user's
request. The deployment included the ONNX and JSON sidecar, four-leg dashboard
controller/static files, policy runtime contract/loader/loop, servo status-block
reader, launcher, and these instructions. The service environment now selects
`parallel-running-v43-foot-warp-trot-residual/model_399.onnx`; V30 remains the
alternate crawl selection. Existing motor calibration and IMU source/settings
were preserved.

The dashboard reported all motors disarmed before the service was stopped.
After installation, systemd reported `active/running`, zero restarts, and an
exit status of zero. Remote ONNX and sidecar SHA-256 values matched the local
package. No policy inference, automated test, browser validation, or physical
walking trial was run as part of this deployment.

The previous runtime and service environment are backed up outside the checkout:

```text
/home/rd/drobot-deployments/v43-20260905/before.tar.gz
/home/rd/drobot-deployments/v43-20260905/manual-web.env.before
```

To restore the previous installation, first ensure the robot is disarmed, then:

```bash
sudo systemctl stop drobot-manual-web.service
tar -xzf /home/rd/drobot-deployments/v43-20260905/before.tar.gz -C /home/rd/drobot2
sudo cp /home/rd/drobot-deployments/v43-20260905/manual-web.env.before /etc/default/drobot-manual-web
sudo systemctl start drobot-manual-web.service
```

The archive restores the previous runtime and V31 selection; V31 remains a
rejected running policy. Use the V30 selection for the known working crawl.

### Supported commissioning procedure

1. Keep the robot supported so every shoe can move without striking the stand.
2. Use the bench supply and verify all 12 motors and the BNO085 before arming.
3. Select `parallel-running-v43-foot-warp-trot-residual` in the RL panel.
4. Press **Prepare RL Stance** and wait for ready.
5. Set speed to `0.10 m/s` and duration to `3-5 s`.
6. Start a fixed external video, then start the policy.
7. Let normal completion exercise phase-aligned ramp and centered hold; use the
   explicit stop only for a problem because an emergency stop still disarms.
8. Download the automatically recorded trial before changing speed.

Only a clean supported trace should proceed to a tethered floor attempt. Do
not start the first trial at `0.12` or `0.14 m/s` merely because those values
are present in the trained range.
