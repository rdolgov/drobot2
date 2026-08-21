# Raspberry Pi ROS 2 onboard controller

The selected learned walking policy also has a standalone, print-only runtime
under [`policy-runtime/`](policy-runtime/README.md). Use it to read the BNO085
and print 12 bounded motor targets before connecting policy output to the servo
bus. Its core interfaces are designed to be reused by this ROS 2 package.

This area packages the existing Drobot motor session and browser dashboard as
a ROS 2 node for an onboard Raspberry Pi. The Pi owns the USB servo bus, serves
the control page to computers on the same network, publishes telemetry, and
accepts the same walking commands through ROS services or a ROS command topic.

The package does not copy calibration or gait equations. It imports:

- `hardware/test-apps/one-leg-testbed` for the Feetech transport and calibrated
  angle conversion;
- `hardware/test-apps/four-leg-dashboard` for the twelve-servo session, browser
  UI, continuous distributed crawl, and continuous diagonal-pair gait; and
- `hardware/robot-runtime/four-leg.toml` plus its four tracked servo profiles
  and calibration JSON files as the physical robot source of truth.

## Supported Pi baseline

Use a Raspberry Pi 4 or 5 with **Ubuntu Server 24.04 64-bit (arm64)** and **ROS
2 Jazzy**. ROS publishes Jazzy packages for Ubuntu 24.04 arm64, and that image's
Python version satisfies the repository's Python 3.11-or-newer requirement.

Official references:

- [ROS 2 Jazzy Ubuntu installation](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
- [ROS 2 Jazzy Ubuntu arm64 support](https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Install-Binary.html)
- [Ubuntu Server setup on Raspberry Pi](https://ubuntu.com/tutorials/how-to-install-ubuntu-on-your-raspberry-pi)

Raspberry Pi OS is not the tracked installation target because ROS 2 Jazzy's
official binary platform is Ubuntu 24.04. A source-built or containerized ROS
installation can be added later without changing the package interfaces.

## Layout

```text
onboard/
|-- ros2_ws/src/drobot_onboard/   ROS 2 ament_python package
|-- scripts/install-pi.sh         dependency install and colcon build
|-- scripts/start-onboard.sh      foreground launcher
`-- systemd/                      optional boot-service template
```

## 1. Prepare the Raspberry Pi

Install ROS 2 Jazzy using the official instructions, including
`ros-jazzy-ros-base`. Then install the local build and serial-account
prerequisites:

```bash
sudo apt update
sudo apt install -y git python3-venv ros-dev-tools avahi-daemon
sudo usermod -aG dialout "$USER"
```

Log out and back in after adding `dialout`. This permission is required for
`/dev/ttyUSB*` and `/dev/ttyACM*` devices.

Clone the repository on the Pi:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL ~/drobot2
cd ~/drobot2
```

## 2. Install from the clone

From the repository root:

```bash
bash onboard/scripts/install-pi.sh
```

The script:

1. checks that ROS 2 and Python 3.11+ are available;
2. creates `onboard/.venv` with access to the system ROS Python packages;
3. installs both existing hardware-control packages in editable mode; and
4. builds `drobot_onboard` into `onboard/ros2_ws/install` with symlink install.

The environment, build, install, and log directories stay local and are
ignored by Git.

## 3. Start without hardware first

Demo mode creates twelve in-memory motors and does not open a serial port:

```bash
bash onboard/scripts/start-onboard.sh --demo
```

From another computer on the same network, open:

```text
http://RASPBERRY_PI_HOSTNAME.local:8080/
```

If `.local` name resolution is unavailable, use the Pi's IP address, such as
`http://192.168.1.42:8080/`.

## 4. Start with the robot bus

Connect the Feetech USB adapter and identify its Linux device:

```bash
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

Then start the controller:

```bash
DROBOT_SERIAL_PORT=/dev/ttyUSB0 bash onboard/scripts/start-onboard.sh
```

`DROBOT_SERIAL_PORT=auto` is the default and works when only one likely USB
serial adapter is connected. Startup opens the bus, verifies servo IDs 1-12,
and leaves every motor disarmed before the web or ROS command interfaces become
available.

The browser exposes the same controls as the desktop dashboard, including:

- telemetry and permanent error display;
- manual per-joint arm, target, leg, and disarm controls;
- calibrated center and gait-start stance;
- continuous distributed crawl;
- continuous diagonal-pair gait; and
- **STOP + DISARM**.

Walking continues until stopped. Browser-started motion also stops on browser
heartbeat loss. ROS-started motion is kept alive by the onboard node until a
ROS stop/disarm request, process shutdown, bus fault, or telemetry/motion fault.

## HTTP web-service API

The browser uses the same JSON API available to other LAN clients. Important
routes are:

| Method and route | Action |
| --- | --- |
| `GET /api/state` | Read telemetry, motor state, gait phase, warnings, and faults |
| `POST /api/crawl-forward` | Start continuous distributed crawl |
| `POST /api/diagonal-pair-forward` | Start continuous diagonal-pair gait |
| `POST /api/crawl-stop` | Stop and disarm all motors |
| `POST /api/crawl-stance` | Move to the distributed gait stance |
| `POST /api/center-all` | Move all twelve joints to calibrated zero |
| `POST /api/disarm-all` | Disarm all motors |

Every API request except the initial page/assets requires the
`X-Control-Token` header. The browser receives a random per-process token
automatically. For external scripts, set a stable token of at least 16
characters before starting:

```bash
export DROBOT_CONTROL_TOKEN='replace-with-a-long-random-value'
bash onboard/scripts/start-onboard.sh
```

Example script calls:

```bash
curl -H "X-Control-Token: $DROBOT_CONTROL_TOKEN" \
  http://drobot.local:8080/api/state

curl -X POST -H "Content-Type: application/json" \
  -H "X-Control-Token: $DROBOT_CONTROL_TOKEN" \
  -d '{"safety_ack":true,"confirmation":"TEST DISTRIBUTED CRAWL"}' \
  http://drobot.local:8080/api/crawl-forward

curl -X POST -H "Content-Type: application/json" \
  -H "X-Control-Token: $DROBOT_CONTROL_TOKEN" -d '{}' \
  http://drobot.local:8080/api/crawl-stop
```

## ROS 2 interfaces

The launch file places the node in the `/drobot` namespace.

| Interface | Type | Purpose |
| --- | --- | --- |
| `/drobot/status` | `std_msgs/msg/String` | Complete dashboard snapshot encoded as JSON |
| `/drobot/events` | `std_msgs/msg/String` | Last-event changes |
| `/drobot/command` | `std_msgs/msg/String` | Programmatic command input |
| `/drobot/walk_distributed` | `std_srvs/srv/Trigger` | Start continuous one-leg-at-a-time crawl |
| `/drobot/walk_diagonal_pair` | `std_srvs/srv/Trigger` | Start continuous diagonal-pair gait |
| `/drobot/stop` | `std_srvs/srv/Trigger` | Stop gait and disarm all motors |
| `/drobot/disarm_all` | `std_srvs/srv/Trigger` | Disarm all motors |
| `/drobot/center_all` | `std_srvs/srv/Trigger` | Move all joints to calibrated zero |
| `/drobot/gait_stance` | `std_srvs/srv/Trigger` | Move to the distributed gait start stance |

Open a second Pi shell and source the environments:

```bash
cd ~/drobot2
source /opt/ros/jazzy/setup.bash
source onboard/.venv/bin/activate
source onboard/ros2_ws/install/setup.bash
```

Examples:

```bash
ros2 topic echo /drobot/status
ros2 service call /drobot/walk_distributed std_srvs/srv/Trigger '{}'
ros2 service call /drobot/walk_diagonal_pair std_srvs/srv/Trigger '{}'
ros2 service call /drobot/stop std_srvs/srv/Trigger '{}'
ros2 topic pub --once /drobot/command std_msgs/msg/String "{data: stop}"
```

Accepted command-topic values are `walk_distributed`, `walk_diagonal_pair`,
`stop`, `disarm_all`, `center_all`, and `gait_stance`. Prefer services when the
caller needs a success/failure response.

## Launch parameters

The launcher reads these environment variables and accepts matching ROS launch
arguments:

| Environment | Launch argument | Default |
| --- | --- | --- |
| `DROBOT_MANIFEST` | `manifest` | `hardware/robot-runtime/four-leg.toml` from the clone |
| `DROBOT_SERIAL_PORT` | `serial_port` | `auto` |
| `DROBOT_HTTP_BIND` | `http_bind` | `0.0.0.0` |
| `DROBOT_HTTP_PORT` | `http_port` | `8080` |
| `DROBOT_CONTROL_TOKEN` | `control_token` | Random token per process |

Direct launch example:

```bash
ros2 launch drobot_onboard onboard.launch.py \
  manifest:="$PWD/hardware/robot-runtime/four-leg.toml" \
  serial_port:=/dev/ttyUSB0 http_port:=8080
```

## Optional start at boot

After the foreground hardware command has worked, install the service:

```bash
bash onboard/scripts/install-systemd-service.sh
```

This enables the service for the current user but does not start it immediately.
Optional overrides, including a stable API token, are placed in
`/etc/default/drobot-onboard`. Start and inspect the service explicitly:

```bash
sudo systemctl start drobot-onboard
systemctl status drobot-onboard
journalctl -u drobot-onboard -f
```

To install and start in one operation, pass `--start`. Node shutdown uses
`SIGINT`, closes the HTTP server, disarms all twelve motors, and closes the
serial bus before systemd restarts or powers down.

## Updating the Pi from Git

```bash
cd ~/drobot2
git pull --ff-only
bash onboard/scripts/install-pi.sh
sudo systemctl restart drobot-onboard
```

Tracked calibration files remain under `hardware/robot-runtime/servos`. Review
calibration changes before pulling over a physical robot installation.

## Network boundary

The dashboard binds to `0.0.0.0` so another device on the LAN can reach it. It
uses the dashboard's per-process request token to reject unrelated API posts,
but it is not an internet-facing authentication system. Keep port 8080 on a
trusted robot network; do not forward it from a router. Set
`DROBOT_HTTP_BIND=127.0.0.1` when only SSH tunneling should reach the UI.

## Troubleshooting

- **Serial permission denied:** confirm the service user belongs to `dialout`,
  then log out/in or reboot.
- **More than one adapter found:** set `DROBOT_SERIAL_PORT` explicitly.
- **IDs missing at startup:** the node intentionally refuses partial startup;
  check robot power, common ground, adapter wiring, and IDs 1-12.
- **Dashboard unreachable:** check `systemctl status`, port 8080, the Pi's IP,
  and the local firewall.
- **Port already occupied:** set `DROBOT_HTTP_PORT` to another unused port or
  stop the older onboard process.
- **Code changed but behavior did not:** rerun `install-pi.sh` and restart the
  foreground process or systemd service.
