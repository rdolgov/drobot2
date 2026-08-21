# `pi5-dog` bring-up record

Last checked: 2026-08-21

This records the current Raspberry Pi state and the safe rollout path from
print-only policy inference to closed-loop walking. Do not place login
passwords, dashboard tokens, or Wi-Fi credentials in this repository.

## Current Pi baseline

| Item | Observed state |
| --- | --- |
| Host | `pi5-dog.local` |
| Board / OS | Raspberry Pi 5, Ubuntu 26.04 LTS arm64 |
| Python | 3.14.4 |
| IMU | BNO085 detected on I2C bus 1 at `0x4A` |
| Policy model | `parallel-walking-v18/model_299.onnx` |
| Model SHA-256 | `3a7e31fdc7a57c9ed17e4c4090d56e870a366d87b98bd9a7e0420298816afc77` |
| Policy runtime | ONNX Runtime 1.29.0, 60 Hz, deterministic Beta mean |
| Motor output | Disabled; policy targets are displayed/printed only |
| USB servo adapter | Not detected during this check |
| ROS 2 | Not installed during this check; Lyrical is the tracked Ubuntu 26.04 target |

The Pi reported that its supply could not provide 5 A and was restricting
peripheral power. Fix the Pi supply before depending on a USB servo adapter.
The servo rail must continue to use its own correctly sized supply with a
shared signal ground. A kernel update was also pending; reboot only during a
planned maintenance window.

## One-time policy setup

For a normal clone:

```bash
ssh rd@pi5-dog.local
cd ~/drobot2
git pull --ff-only
git lfs install
git lfs pull
bash onboard/scripts/install-policy-runtime.sh
```

For an existing sparse clone that does not include `onboard/`:

```bash
cd ~/drobot2
git sparse-checkout add onboard
git lfs install
git lfs pull
bash onboard/scripts/install-policy-runtime.sh
```

The installer verifies that the ONNX file is not a Git LFS pointer and checks
its SHA-256 against the tracked model metadata.

## Safe commands available now

Run inference with a synthetic level IMU:

```bash
bash onboard/scripts/run-policy-print.sh --imu level --duration-s 5
```

Run inference from the real BNO085 and print all 12 targets:

```bash
bash onboard/scripts/run-policy-print.sh --imu bno085
```

Start the LAN dashboard:

```bash
bash onboard/scripts/run-policy-web.sh --imu bno085
```

Open the tokenized URL printed by the command. The default port is 8090. The
page can start and stop inference, change the requested speed from 0 to
0.20 m/s, and show the live IMU plus all 12 motor targets. It cannot open the
servo bus or enable torque.

To run the dashboard automatically after boot:

```bash
bash onboard/scripts/install-policy-web-service.sh --start
sudo systemctl status drobot-policy-web
sudo cat /etc/default/drobot-policy-web
```

The last command shows the protected token needed in the browser URL. Do not
commit that token.

## Manual inverse-kinematics and crawl dashboard

The existing manual dashboard remains on port 8080. It includes calibrated
per-joint controls, inverse-kinematics-backed stance controls, distributed
crawl, diagonal-pair gait, telemetry, and STOP + DISARM. The intended topology
is:

```text
phone/computer browser
        |
        | Wi-Fi or Ethernet (HTTP now, ROS 2 later)
        v
Raspberry Pi 5 -- USB --> Feetech servo adapter -- servo bus --> 12 servos
        |
        `-- I2C --> BNO085
```

Connect the USB servo adapter to the Pi, not simultaneously to another
computer. Exactly one process must own the serial bus. After ROS 2 Lyrical and
the onboard package are installed, launch the manual dashboard with:

```bash
bash onboard/scripts/install-pi.sh
DROBOT_SERIAL_PORT=/dev/ttyUSB0 bash onboard/scripts/start-onboard.sh
```

Then open `http://pi5-dog.local:8080/`. Use `--demo` instead of the serial-port
variable to review the page without hardware.

## Controlled rollout to learned motor control

The policy must not be connected directly to the motors yet. Its observation
expects real joint position and joint velocity, while the current print-only
runtime deliberately inserts the neutral pose and zero velocity.

1. **Sensor validation:** confirm the BNO085 mounting transform and signs while
   the robot is supported. Level projected gravity should be near `[0,0,-1]`.
2. **Read-only joint feedback:** add all 12 servo positions and velocities to
   the observation without enabling torque. Reject stale, missing, or
   out-of-range feedback.
3. **Shadow mode:** compare policy targets with measured positions while the
   manual controller moves the robot. Log timing, clipping, rate limiting, and
   disagreement.
4. **Guarded actuation:** require a physical enable/interlock, heartbeat,
   posture limits, action-rate limits, current/temperature limits, and an
   immediate disarm path. Begin supported and at low power.
5. **ROS 2 split:** publish BNO085 data as `sensor_msgs/Imu`, servo feedback as
   `sensor_msgs/JointState`, and velocity commands separately. Publish
   timestamped policy targets to the single motor-owner node, which remains
   responsible for all safety checks.

This split lets the same policy core run from shell scripts today and behind
ROS 2 later, while preventing the browser, manual gait, and learned policy from
fighting over the servo bus.

## Verification recorded on 2026-08-21

- Installed the ARM64 policy environment from the repository on the Pi.
- Verified the tracked ONNX model hash.
- Ran the level-IMU print loop successfully.
- Read live BNO085 samples and produced 12 bounded, rate-limited policy targets.
- Confirmed that neither print mode nor the policy dashboard opens a servo port.
- Confirmed that no USB serial servo adapter and no `/opt/ros` installation
  were present at the time of inspection.
