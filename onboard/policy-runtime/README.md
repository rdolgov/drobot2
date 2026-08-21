# Raspberry Pi walking-policy runtime

This package runs the selected V18 rectangular-shoe policy on a 64-bit
Raspberry Pi with ONNX Runtime. It is intentionally print-only: it reads the
BNO085, assembles the exact 50-value observation, runs deterministic inference
at 60 Hz, converts the 12 normalized actions into bounded/rate-limited joint
targets, and prints the targets with physical servo IDs. It never opens the
servo serial port or enables torque.

The runtime was exercised on `pi5-dog` with Ubuntu 26.04 ARM64, Python 3.14.4,
ONNX Runtime 1.29.0, and the BNO085 detected at I2C address `0x4A`.

The shared code is split into replaceable interfaces:

- `ImuSource`: BNO085 now, a ROS `sensor_msgs/Imu` adapter later;
- `JointStateSource`: neutral placeholder now, servo telemetry or ROS
  `sensor_msgs/JointState` later;
- `OnnxWalkingPolicy`: portable deterministic policy inference;
- `MotorSink`: JSON printing now, a separately guarded motor-command publisher
  later.

## Model

The deployable model is
`onboard/models/parallel-walking-v18/model_299.onnx`. It is exported from
`simulation/isaac/models/parallel-walking-v18-coordinated/model_299.pt` with
`simulation/isaac/rl/parallel_walking/export_policy_onnx.py`.

The accompanying JSON file records hashes and the full observation/action
ordering. The policy consumes:

1. velocity command (3);
2. gait-clock sine/cosine (2);
3. body-frame gyro, projected gravity, and acceleration (9);
4. joint position error, joint velocity, and previous action (36).

IMU data alone is not sufficient for safe closed-loop motor control. The
current script supplies the trained neutral joint pose and zero joint velocity
only to exercise the sensor-to-policy-to-print pipeline. Real motor actuation
must first replace `NeutralJointStateSource` with fresh 12-joint feedback.

## Install on `pi5-dog`

```bash
ssh rd@pi5-dog.local
cd ~/drobot2
git pull --ff-only
bash onboard/scripts/install-policy-runtime.sh
```

If this is a sparse checkout that currently contains only `hardware/`, add the
onboard directory first:

```bash
git sparse-checkout add onboard
```

The installer creates `onboard/.policy-venv` and installs ONNX Runtime plus the
BNO085 dependencies. Current ONNX Runtime releases provide CPython 3.14 ARM64
wheels; the installer also handles the documented CPython 3.14 `lgpio` wheel.

First verify inference without touching I2C:

```bash
bash onboard/scripts/run-policy-print.sh --imu level --duration-s 5
```

Then read the BNO085 at `0x4A` and print policy motor targets:

```bash
bash onboard/scripts/run-policy-print.sh --imu bno085
```

Stop with `Ctrl+C`. Each output line contains
`"mode":"PRINT_ONLY_NO_SERVO_WRITES"`, the physical servo ID, semantic joint
name, target degrees, and normalized action.

If the sensor board axes do not match robot forward/left/up, pass a signed
permutation. For example, if sensor `+y` is robot forward and sensor `-x` is
robot left:

```bash
bash onboard/scripts/run-policy-print.sh \
  --imu bno085 --imu-axis-map +y,-x,+z
```

Confirm the mounting transform while the robot is supported. When level,
projected gravity should be approximately `[0, 0, -1]`; rotations about each
body axis must have the expected sign before any policy output is connected to
motors.

## Browser dashboard

Start the live, print-only dashboard:

```bash
bash onboard/scripts/run-policy-web.sh --imu bno085
```

The command prints a tokenized URL such as
`http://pi5-dog.local:8090/?token=...`. Open it from a computer or phone on the
same LAN. The page shows live body-frame IMU values and all 12 policy targets,
and can start/stop inference or adjust the requested forward speed. It also
links to the existing manual crawl dashboard on port 8080.

For automatic startup, install the dedicated service:

```bash
bash onboard/scripts/install-policy-web-service.sh --start
sudo cat /etc/default/drobot-policy-web
```

Copy the token from the protected environment file into the dashboard URL.
The service remains print-only and does not own the servo bus.

The Pi currently reports that its power supply cannot provide 5 A and is
restricting peripheral power. Correct that before depending on an attached USB
servo adapter; the servo rail still requires its own correctly sized supply.

## Recommended ROS 2 boundary

Keep one process as the sole owner of the servo bus. A future policy node should
subscribe to `sensor_msgs/Imu`, `sensor_msgs/JointState`, and a velocity command;
then publish a timestamped 12-joint target message. The existing onboard motor
node should validate freshness, position/rate limits, heartbeat, arming state,
and the physical enable interlock before accepting that message. This preserves
the same policy core while keeping safety enforcement outside the neural
network.
