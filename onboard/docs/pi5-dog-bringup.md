# `pi5-dog` bring-up record

Last checked: 2026-08-23

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
| Policy model | `parallel-walking-v20-external-rear-payload/model_900.onnx` |
| Model SHA-256 | `f787669d33115f117ac691c1ba7bb145fa6a726c6a5d22aec7aecc9e284d6529` |
| Policy runtime | ONNX Runtime 1.29.0, 60 Hz, deterministic Beta mean |
| Motor output | Disabled; policy targets are displayed/printed only |
| Manual/IK dashboard | Boot service on port 8080; hardware preference with safe demo fallback |
| USB servo adapter | QinHeng USB serial, stable by-id path when connected; currently removable |
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

Open the URL printed by the command. The default port is 8090. The page can
start and stop inference, change the requested speed from 0 to
0.20 m/s, and show the live IMU plus all 12 motor targets. It cannot open the
servo bus or enable torque.

When using the boot service, open `http://pi5-dog.local:8090/`. The tracked
service configuration does not require a token and should be used only on a
trusted local network.

To run the dashboard automatically after boot:

```bash
bash onboard/scripts/install-policy-web-service.sh --start
sudo systemctl status drobot-policy-web
```

The service is open to the local network by default. Optional token protection
can be enabled later through `/etc/default/drobot-policy-web`.

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
computer. Exactly one process must own the serial bus. The manual dashboard can
run without ROS 2. Install and start its safe simulated version with:

```bash
bash onboard/scripts/install-manual-runtime.sh
bash onboard/scripts/install-manual-web-service.sh --start
```

Then open `http://pi5-dog.local:8080/`. The service defaults to demo mode and
cannot command real motors. To enable hardware later, set
`DROBOT_MANUAL_DEMO=false` and the detected `DROBOT_MANUAL_SERIAL_PORT` in
`/etc/default/drobot-manual-web`, then restart the service while the robot is
supported and the physical cutoff is ready.

Keep `DROBOT_MANUAL_FALLBACK_DEMO=false` on the real robot. If the adapter or a
configured motor is missing, the service continues serving hardware diagnostics
with motion disabled and retries recovery during telemetry polling. It enables
controls only after verifying all 12 IDs and disarming them. Linux address reuse
is enabled for this service so an active browser
connection does not prevent port 8080 from reopening during a clean restart;
Windows retains exclusive port ownership for the desktop dashboard. Trusted-LAN
Pi pages do not require a control token. Motion POSTs carry a non-secret client
version so stale pages are told to reload rather than controlling the robot.

Brief browser or Wi-Fi polling failures are shown as a live connection warning
and are not stored in the recent error log. Active hardware and RL faults remain
visible, then clear automatically after the corresponding server state recovers.
Other command errors expire after ten minutes; Settings still provides an
immediate manual clear.

The page sends its heartbeat every 0.7 seconds on an independent request path.
A missing heartbeat or closed page becomes a visible warning after 20 seconds,
but it does not stop the gait, alter targets, or remove torque. The onboard gait
continues until **STOP + DISARM**, an actual controller/bus fault, process
shutdown, or the physical cutoff.

Both walking buttons run continuously until **STOP + DISARM**. Starting a gait
from a minor off-stance pose no longer fails the old zero-centred tolerance
check: the controller holds each measured position first and ramps to the gait
stance. This relaxation does not remove startup ID checks, computed joint-limit
checks, browser-heartbeat diagnostics, telemetry/motion fault handling, or the
physical-cutoff requirement.

Keep `DROBOT_MANUAL_SERIAL_PORT=auto` for replaceable compatible adapters. If
USB reconnect changes the Linux node from `ttyACM0` to `ttyACM1` (or similarly
for `ttyUSB`), the running dashboard closes the dead handle, resolves the new
device, verifies IDs 1-12, and disarms all motors before serving telemetry
again. It refuses partial-bus recovery.

For battery comparisons, disarm all motors, connect the selected source, press
**RESET AT IDLE**, and allow several telemetry samples before walking. Compare
the 60-second power chart, idle-to-load voltage sag, peak current, and possible
stall IDs using the same supported gait. Sag plus stall flags indicates an
electrical-delivery or mechanical-load problem; falling without those signals
is stronger evidence for added battery mass, center-of-mass shift, or contact
geometry. Dashboard watt-hours are sampled servo estimates and exclude the Pi.
The **FULL / GOOD / LOW / RECHARGE** label is a coarse 3S idle-voltage estimate;
use a balance-plug checker to verify each cell before charging or continued use.

The standalone service must be stopped before starting the ROS 2 onboard node:

```bash
sudo systemctl stop drobot-manual-web
```

## Controlled rollout to learned motor control

The port 8080 dashboard now includes the first guarded real-motor rollout. It
uses live calibrated joint position/velocity feedback and the real BNO085 while
keeping the manual dashboard as the only servo-bus owner. The standalone port
8090 service remains print-only and should be stopped before an 8080 RL test.

1. **Sensor validation:** confirm the BNO085 mounting transform and signs while
   the robot is supported. Level projected gravity should be near `[0,0,-1]`.
2. **Live joint feedback:** all 12 calibrated positions and encoder-derived
   velocities feed the observation; stale, missing, or out-of-range feedback
   stops and disarms the test.
3. **Bounded actuation:** the UI permits a supported `1-60 s` test at a custom
   `0.000-0.100 m/s`, initializes targets from the measured pose, and enforces
   joint, rate, tilt, and telemetry guards. Normal timed completion returns all
   12 joints to calibrated center and keeps torque holding; an explicit stop or
   policy/hardware fault still disarms. An RL temperature sample from `55-64 C`
   must persist across repeated reads for five seconds before disarm; `65 C` or
   above remains an immediate critical stop.
4. **Floor rollout:** remains future work. Review recorded timing, tracking,
   current, voltage sag, body motion, and support behavior before removing the
   fixture or raising the speed/duration limits.
5. **ROS 2 split:** publish BNO085 data as `sensor_msgs/Imu`, servo feedback as
   `sensor_msgs/JointState`, and velocity commands separately. Publish
   timestamped policy targets to the single motor-owner node, which remains
   responsible for all safety checks.

This split lets the same policy core run from shell scripts today and behind
ROS 2 later, while preventing the browser, manual gait, and learned policy from
fighting over the servo bus.

The exact V1 contract and stop conditions are in
`hardware/test-apps/four-leg-dashboard/specs/rl-walk-v1.md`.

## Verification recorded on 2026-08-21

- Installed the ARM64 policy environment from the repository on the Pi.
- Verified the tracked ONNX model hash.
- Ran the level-IMU print loop successfully.
- Read live BNO085 samples and produced 12 bounded, rate-limited policy targets.
- Confirmed that neither print mode nor the policy dashboard opens a servo port.
- Confirmed that no USB serial servo adapter and no `/opt/ros` installation
  were present at the time of inspection.
