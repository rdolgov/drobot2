# V23 higher-speed straight residual crawl

## Goal

V22 made the real robot substantially calmer by constraining PPO to a residual
around the proven sequential crawl, but its training range ends at 0.015 m/s.
Changing the web speed above that value is rejected or clipped and cannot make
the policy meaningfully faster. V23 is trained for an adjustable
0.005-0.050 m/s range while retaining the same stability-first gait structure.

The second objective is to reduce the leftward drift observed on the real
robot. Simulation cannot identify an individual real servo or geometry bias,
but it can stop treating a curved nominal gait as acceptable. Hardware
calibration and trial telemetry remain necessary if a repeatable left bias
persists after this model.

## Training design

- Initialize the actor and critic from selected V22 `model_500.pt`.
- Keep the 50-value policy observation contract unchanged so V22 weights and
  the Raspberry Pi runtime remain compatible.
- Train commands from 0.010-0.030 m/s initially, expanding to
  0.005-0.050 m/s over 51,200 policy steps (800 PPO iterations).
- Scale the reference cadence from 0.12 to 0.75 Hz with command speed.
- Use a 65 mm reference stride, reducing it to 65% only at the lowest command.
- Retain the sequential RR, FR, RL, FL swing order, 86.25% stance duty, and one
  active swing leg.
- Retain the 25% learned residual and the deployment-matched 2 degree target
  change per 60 Hz controller tick.
- Retain V22's action-rate, action-acceleration, joint-acceleration, body-
  acceleration, support-slip, impact, tilt, and three-foot-support objectives.

## Straightness reward

V23 raises the lateral-velocity and raw lateral-displacement costs and adds two
terms:

1. A path-corridor cost activates after the base moves more than 20 mm sideways
   from its episode start line. It is normalized by the corridor width so a
   several-centimeter drift is no longer numerically tiny.
2. A heading cost penalizes wrapped absolute yaw relative to the episode's
   starting heading, in addition to the existing instantaneous yaw-rate cost.

Absolute heading is used only by the training reward/critic-side evaluation;
it is not added to the deployed policy observation. This preserves the V22
ONNX interface. The actor can learn a more symmetric nominal gait from gyro,
gravity, acceleration, joint state, and action history, but it cannot perform
magnetometer-style absolute-heading hold. A later real-world adaptation layer
could use observed drift or an external heading reference if mechanical
asymmetry remains.

## Candidate acceptance

Evaluate saved checkpoints for 30 seconds at 0.005, 0.015, 0.030, and
0.050 m/s. Reject a candidate that falls, repeatedly stalls, loses the
three-foot-support pattern, saturates the residual actor, or materially raises
joint/body acceleration. Compare mean absolute lateral displacement, final
heading error, and accumulated yaw travel against V22 as well as against other
V23 checkpoints.

## Training and selection result

Training continued V22 checkpoint 500 for 1,200 PPO updates with 128 parallel
robots and seed 2301. The run ended at iteration 1699. The mature training
phase held the full 32-second horizon without fall, tilt, low-height, or base-
contact failures.

Checkpoints 1250, 1500, 1600, and 1699 were screened at high speed. Checkpoint
1500 was selected because later checkpoints gained little speed while drifting
more. Two deterministic 20-second episodes at each command produced:

| Command | Mean actual speed | Lateral displacement | Final heading error | >=3-foot support | Joint RMS acceleration | Falls / stalls |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.005 m/s` | `0.00598 m/s` | `0.00969 m` | `0.0349 rad` | `95.0%` | `5.22 rad/s2` | `0 / 0` |
| `0.015 m/s` | `0.01621 m/s` | `0.01284 m` | `0.0511 rad` | `83.5%` | `7.79 rad/s2` | `0 / 0` |
| `0.030 m/s` | `0.02509 m/s` | `0.01457 m` | `0.0341 rad` | `79.3%` | `11.85 rad/s2` | `0 / 0` |
| `0.050 m/s` | `0.03551 m/s` | `0.03991 m` | `0.0353 rad` | `76.0%` | `10.47 rad/s2` | `0 / 0` |

At the shared 0.015 m/s command, V22 checkpoint 500 drifted 0.02280 m in the
same two-episode evaluation while V23 drifted 0.01284 m, a 44% reduction. V22
had the smaller final heading angle in that comparison, so V23 should be
described as straighter in lateral path, not as absolute-heading hold. Drift
also grows at the 0.050 m/s maximum; hardware rollout should not begin there.

Selected artifacts:

- checkpoint: `simulation/isaac/models/parallel-walking-v23-higher-speed-straight-residual-crawl/model_1500.pt`;
- ONNX and metadata: `onboard/models/parallel-walking-v23-higher-speed-straight-residual-crawl/model_1500.onnx` and `model_1500.json`;
- 20-second 0.030 m/s review: `simulation/reviews/parallel-walking-v23-higher-speed-straight-residual-crawl-model1500-20s.mp4`;
- ONNX SHA-256: `28e9167ca08e817b86489c16a10d4d43b99364afa8ae985d641b5d45307f83cb`.

Real-robot rollout should start at 0.005 m/s on the power supply with a spotter
and emergency stop, then advance in small increments only after the automatic
telemetry recording remains stable. Simulation qualification does not prove
the real robot is safe, particularly with servo and geometry asymmetry.
