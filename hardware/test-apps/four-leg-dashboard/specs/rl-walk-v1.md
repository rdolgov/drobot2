# Guarded RL walk V1

## Scope

The port 8080 hardware dashboard can run the selected
`parallel-walking-v20-external-rear-payload/model_900.onnx` policy as a bounded, supported-robot
commissioning test. The four-leg dashboard remains the only process that opens
the Feetech serial bus. The policy runner supplies targets to that same session;
it does not create a second motor owner.

The initial rollout is deliberately limited to five seconds. The selected V20
model accepts its trained `0.04` through `0.10 m/s` forward range; future model
sidecars may declare a different range. It is not an unattended or floor-ready
autonomy mode.

## Observation and action contract

Inference runs at the model's recorded 60 Hz rate. Each 50-value observation
contains:

1. forward/lateral/yaw command and gait-clock sine/cosine (5);
2. BNO085 body gyro, projected gravity, and acceleration (9);
3. 12 live calibrated joint-position errors;
4. 12 joint velocities computed from encoder differences; and
5. 12 previous normalized policy actions.

An independent background source requests encoder feedback at 100 Hz and the
policy consumes its latest complete cached sample. Feedback is rejected after
120 ms. The actual achieved USB/servo rate is measured in recordings rather
than assumed from the requested polling rate.
The policy produces 12 normalized actions in Isaac order. The tracked contract
maps those actions to physical servo IDs 1-12 and converts them to bounded
semantic joint angles. Policy targets are limited from actual elapsed monotonic
time, with a five-degree maximum for any one delayed update, before the
existing servo-session ramp is applied.

While the policy is active, encoder position/speed uses the SDK's synchronous
group read for registers 56–59 across all 12 IDs, with a sequential fallback
after repeated group-read failures. Torque, voltage, temperature, current, and
stall diagnostics are staggered across the 12 motors instead of reading every
register for every motor in one policy update. Each motor remains covered by
the onboard diagnostics during the trial. Browser dashboard responses use the
last full electrical snapshot plus live policy targets and encoder positions;
normal full telemetry resumes immediately after the policy disarms. This keeps
browser refreshes from blocking inference.

The 60 Hz motor-output worker sends all pending targets with one synchronous
group write, so the four legs begin each update from the same bus packet. Both
the inference scheduler and motor scheduler skip missed slots instead of
replaying compressed catch-up updates.

## Start interlock

The browser requires the exact `START SUPPORTED RL TEST` confirmation generated
only after the operator checks that the robot is supported, all feet are clear,
and the physical power cutoff is ready. Start is rejected unless:

- the real model exists and the BNO085 returns a complete, finite sample;
- projected gravity says the body is within 41 degrees of upright;
- the serial bus has no active fault;
- no manual, crawl, diagonal-pair, or RL command is active;
- either all 12 motors begin disarmed or all 12 are already armed in a stable
  hold; partially armed states are rejected; and
- all 12 measured joints are inside the policy envelope with a five-degree
  commissioning margin.

After these checks, each motor is anchored to its measured position before
policy output starts. If all 12 were already armed, torque remains enabled
through the transition; otherwise they are armed at those measured positions.
The measured 12-joint vector becomes the policy's initial rate-limiter target,
preventing a jump from an assumed neutral pose.

## Completion and runtime stops

The operator selects a forward command inside the model-declared range and a
duration from `1` through `60 seconds`. Normal timed completion stops policy
output, keeps all 12 motors armed, and sets every desired semantic joint angle
to calibrated zero. The existing bounded motion worker ramps into the same pose
as **CENTER ALL 12**, then holds torque so the robot does not collapse when the
trial timer ends.

Every path below requests policy stop and disarms all 12 motors:

- **STOP RL + DISARM** or **DISARM ALL 12**;
- model, IMU, serial, telemetry, or joint-feedback exception;
- a control-loop output gap longer than 120 ms;
- missing armed motor or invalid/non-finite target;
- a target update exceeding five degrees;
- projected-gravity magnitude outside `[0.8, 1.2]`; or
- body tilt exceeding 60 degrees while the test is running;
- one verified servo temperature remaining at or above `55 C` for five seconds
  and at least three readings; or
- one critical servo-temperature reading at or above `65 C`.

The temperature verifier prioritizes repeated reads of a suspicious motor. Any
re-read below `55 C` clears the candidate without stopping, preventing one
corrupted telemetry packet from collapsing the robot. The five-second check is
non-blocking and does not pause the 60 Hz policy loop.

Browser heartbeat remains diagnostic only. A physical cutoff is still the
emergency stop because neither Wi-Fi nor software disarm is safety-rated.

## Raspberry Pi process ownership

Install the policy package into `onboard/.manual-venv` with
`onboard/scripts/install-manual-runtime.sh`. Stop the standalone print-only
policy service before real RL testing so the 8080 process is the only BNO085
reader:

```bash
sudo systemctl disable --now drobot-policy-web.service
sudo systemctl restart drobot-manual-web.service
```

Open `http://pi5-dog.local:8080/`, confirm hardware mode, `12 / 12` online,
either `0 / 12` or `12 / 12` armed, plausible voltage, and no fault. Never
start from a partially armed robot. Begin at `0.04 m/s` while the body is
supported and the feet cannot contact the bench.

## ROS 2 migration

`OnnxWalkingPolicy`, `WalkingPolicyLoop`, `ImuSource`, `JointStateSource`, and
`MotorSink` remain transport-independent. The ROS 2 version should replace the
local callbacks with `sensor_msgs/Imu`, `sensor_msgs/JointState`, a velocity
command, and timestamped policy targets. The single motor-owner node must retain
the same freshness, range, rate, tilt, arming, telemetry, and physical-interlock
guards.
