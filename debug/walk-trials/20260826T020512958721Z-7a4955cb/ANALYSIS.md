# Wobbly-walk baseline 20260826T020512958721Z-7a4955cb

## Purpose and artifacts

This folder keeps the trial evidence in repository-safe form:

- `20260826T020512958721Z-7a4955cb.zip`: automatic dashboard recording, with
  Raspberry Pi absolute paths normalized to repository-relative paths;
- `IMG_0827.MOV`: corresponding phone video, stream-copied without its GPS,
  device, timestamp, or Core Media metadata/data tracks; and
- this report: analysis, controller findings, and the planned comparison.

The trial ran V20 `model_900.onnx` for 10 seconds with a `0.03 m/s`
forward command. The archive says the run completed without a software fault,
contains 600 samples and 78 diagnostic events, and dropped no samples or
events. The numerical samples and diagnostic events were not altered during
privacy sanitization. The phone video is 15.13 seconds at 30 fps and begins approximately
1.96 seconds before the controller recording. FFmpeg 9.0.1 was installed and
used to inspect the MOV metadata and extract review frames.

This is a useful **before-fix baseline**, but it is not a valid judgment of the
trained V20 operating point: V20 was trained from `0.04` through `0.10 m/s`,
while this trial requested `0.03 m/s` and still supplied the model's fixed
1.25 Hz gait clock.

## What the evidence shows

| Signal | Observation | Interpretation |
| --- | ---: | --- |
| Policy rows | 600 / 10 s | The recorder received the requested 60 Hz average row count. |
| Long policy intervals | 85 above 25 ms | Slow operations frequently delayed the nominal 16.7 ms loop. |
| Compressed intervals | 93 below 10 ms | The old deadline loop replayed iterations to catch up after delays. |
| Policy interval p95 | about 34 ms | Timing was not a steady 60 Hz even though the average was 60 Hz. |
| Distinct encoder samples | about 151 / 10 s | Physical joint feedback was only about 15.1 Hz in this software path. |
| Motor-output worker | 20 Hz | Desired policy targets were physically transmitted much more slowly than inference. |
| Gait clock | fixed 1.25 Hz | Cadence did not decrease with the very small forward command. |
| Hip tracking error p95 | about 27 degrees | Multiple joints could not follow the rapidly changing requested gait. |
| Knee tracking error p95 | about 30–37.5 degrees | Large lag was visible in both the internal data and the unstable video. |
| Sent-target rate p95 | up to about 795 deg/s | Fixed-per-call limiting plus catch-up iterations compressed changes in wall time. |
| Front-right abduction saturation | 50.7% of policy outputs | The policy was often at its action boundary and materially asymmetric. |
| IMU pitch | maximum about 13.3 degrees | The body pitched substantially during the short run. |
| IMU roll | about -6.1 to +6.1 degrees | The recorded side-to-side wobble is present in the internal signal. |
| IMU yaw rate | maximum about 2.11 rad/s | The trial contains strong heading disturbance despite a zero-yaw command. |
| Servo voltage | 11.5–11.9 V | Sparse diagnostics do not show a sustained voltage-collapse explanation. |
| Servo temperature | 31–36 C | No thermal event explains this trial. |

The video review sampled 100 frames across the recorded motion and detected the
body AprilTag in all 100. The tag is useful for later camera-pose estimation,
but this clip has no fixed ground tag or calibrated camera, so this report does
not claim metrically accurate world displacement from video alone.

## STS3215 feedback-rate correction

The observed 15.1 Hz is **not an STS3215 specification limit**. The
[manufacturer datasheet](https://files.seeedstudio.com/products/Feetech/108090023_STS3215-C001_Datasheet.pdf)
specifies a maximum position update interval of 1 ms and a default 1 Mbps
communication rate. The
[Feetech communication protocol](https://files.seeedstudio.com/wiki/robotics/Actuator/feetech/Communication_Protocol_Manual.pdf)
defines the position feedback starting at address 56 and the protocol's group
operations.

At 1 Mbps, an isolated eight-byte request plus eight-byte response occupies
roughly 160 microseconds of wire time because UART sends about ten bits per
byte. Turnaround, servo return delay, operating-system scheduling, the USB
adapter, and twelve separate request/response cycles add overhead. Even so,
15 Hz is a result of the previous synchronous multi-servo software path, not
the servo's fresh-data capability.

The controller now asks the SDK for one group position/speed read covering
registers 56–59 for all 12 IDs, with a sequential fallback after communication
failures. It also uses one group write for all targets so the four legs receive
the same command packet. The requested background feedback rate is 100 Hz; the
actual achieved rate must be measured on the Raspberry Pi and USB adapter after
deployment. No persistent servo return-delay register is changed by this work.

## Root causes addressed before retraining

1. **Invocation-count rate limiting.** The old policy limiter allowed one
   fixed 60 Hz-sized target step each time the loop function ran. Catch-up
   calls therefore advanced several target steps in only a few milliseconds.
2. **Catch-up bursts.** A late inference/read iteration left the next deadline
   in the past, causing immediate subsequent iterations instead of dropping the
   missed slot.
3. **Synchronous encoder reads in inference.** Reading physical feedback could
   block observation construction and inference.
4. **Different policy and motor rates.** Policy targets were generated at 60 Hz
   but the dashboard motion worker sent physical targets at 20 Hz.
5. **Sequential leg commands.** Twelve writes introduced an avoidable time skew
   between the first and last leg command.
6. **Unsupported command/cadence pair.** V20 received `0.03 m/s`, below its
   training range, while its clock continued at 1.25 Hz.

## Implemented controller changes

- The policy target limiter uses actual elapsed monotonic time and also caps a
  single delayed update at 5 degrees.
- Both policy and motor-output schedulers skip missed slots; neither replays a
  catch-up burst.
- Encoder polling runs in a background source. Inference consumes the latest
  complete cached sample and faults only when it becomes older than 120 ms.
- The RL motor-output worker runs at 60 Hz; the manual crawl keeps its existing
  slower worker rate.
- Position/speed feedback uses the SDK group-read path first, while full
  voltage/current/temperature diagnostics remain slower and round-robin.
- All pending servo targets are sent in one SDK synchronous write, aligning the
  start of motion across all four legs.
- New recordings add elapsed target time, cumulative missed deadlines, gait
  frequency, feedback-transport mode, the measured 3.175 kg robot mass, and the
  modeled external rear payload metadata.
- V20 metadata now explicitly declares a fixed 0.8-second gait period and a
  supported `0.04–0.10 m/s` command range. The UI follows that model metadata
  instead of offering an untrained `0.03 m/s` value.

## V21 low-speed retraining design

The new `low-speed-external-rear-payload` profile is deliberately separate
from V20. It bootstraps the selected `model_900.pt` weights but resets the new
command curriculum:

- initial command range: `0.04–0.10 m/s`;
- final command range: `0.005–0.10 m/s`;
- curriculum: 38,400 policy steps (600 PPO rollouts at 64 steps each);
- cadence: linearly scaled from 0.35 Hz at `0.005 m/s` to 1.25 Hz at
  `0.10 m/s`;
- stride: scaled from 35% to 100% of the 55 mm reference stride;
- standstill deadband: `0.002 m/s`; and
- sustained-motion threshold: the smaller of `0.025 m/s` or 50% of the
  requested command, so a correct very-slow gait is not punished as stalled.

The external 523.18 g rear battery/holder model, payload randomization,
straight-line penalties, acceleration/jerk penalties, foot-slip penalty, and
touchdown penalty remain active. The exported V21 JSON sidecar must declare
the same speed/cadence mapping. Deployment reads that declaration; it never
guesses a new cadence for an older model.

## Next comparison trial

Do not compare several variables at once. After deploying the controller-only
changes, first repeat a supported 3–5 second V20 run at its minimum supported
`0.04 m/s` command. Keep the physical cutoff in reach and use the same camera
view. Also record a short hardcoded-crawl reference under the same battery and
surface conditions.

For the corrected V20 trial, compare:

- output interval distribution and cumulative missed deadlines;
- distinct encoder timestamps and reported feedback transport;
- per-joint target/measurement tracking error;
- target velocity and action saturation;
- IMU roll, pitch, yaw rate, and linear acceleration; and
- voltage/current diagnostics, acknowledging that round-robin electrical data
  remain much sparser than joint feedback.

Train and evaluate V21 separately before offering commands below `0.04 m/s` on
real hardware. A candidate should pass multiple uninterrupted simulator trials
at `0.005`, `0.01`, `0.02`, `0.04`, and `0.08 m/s`, show all four legs active,
avoid stalls/falls, and improve acceleration and lateral/yaw metrics relative
to V20. Only then should it be exported, deployed, and tried on the supported
robot in short increments.

## Current verification status

The source/configuration/documentation changes have not been executed in
Isaac Sim, trained, deployed to the Raspberry Pi, or exercised on hardware as
part of this change. This report records the intended implementation and the
existing before-fix evidence; those later verification results must be added
without overwriting this baseline.
