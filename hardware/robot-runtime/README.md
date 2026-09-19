# Shared real-robot runtime

This directory is the source of truth for configuration shared by physical
robot applications. Test applications consume these files but must not keep
their own canonical copies.

## Contents

- [`four-leg.toml`](four-leg.toml) maps the four physical legs to body corners
  and owns dashboard, monitoring, and crawl defaults.
- [`servos/`](servos/) contains each leg's bus settings, servo IDs, directions,
  joint limits, and measured neutral-center calibration.
- [`scripts/`](scripts/) provides guarded status, configuration, calibration,
  and one-leg browser launch helpers using the shared profiles.

## Verified motor map

| Leg | Corner | Servo IDs | Directions | Center ticks |
| ---: | --- | --- | --- | --- |
| 1 | Front left | `1, 2, 3` | `-1, +1, +1` | `2131, 2183, 1005` |
| 2 | Front right | `4, 5, 6` | `+1, +1, +1` | `2022, 2162, 2052` |
| 3 | Rear left | `7, 8, 9` | `-1, -1, -1` | `2020, 2052, 2046` |
| 4 | Rear right | `10, 11, 12` | `+1, +1, +1` | `1998, 2049, 2048` |

The centers were last captured from the assembled whole-robot neutral pose on
2026-08-09. They are specific to this robot and must be revalidated after horn,
servo, wiring, or linkage changes.

All existing three-motor leg profiles use 1,000,000 baud, a 90% torque limit, speed register 3400,
acceleration register 254, and a 15-degree maximum command step. The dashboard
manual/RL motion ceiling is 270 degrees/s. The hardcoded crawl separately uses
a 120 degrees/s ramp, 6-second cycle, 60 mm stride, and 25 mm lift so only one
leg advances slowly while three feet remain planted. Loaded joints may not
reach no-load speed, and software limits do not replace physical stops or
collision checks. See the
[Waveshare ST3215 documentation](https://www.waveshare.com/wiki/ST3215_Servo)
and the
[ST3215-C047 datasheet](https://files.seeedstudio.com/products/Feetech/108090003_FEETECH_ST-3215-C047-Datasheet.pdf)
for the vendor-level register and electrical context.

## New ankle neutral centers and directions

The four additional ST3215 servos were assigned IDs 13-16 and their neutral
positions were captured on the Pi on 2026-09-18 (Pacific time). The user
confirmed the corner mapping and positioned each sole perpendicular to the
line between its knee and ankle pivots while the robot was supported.

| Corner | Servo ID | Neutral tick | Direction |
| --- | ---: | ---: | --- |
| Front left | 13 | 2048 | +1 |
| Front right | 14 | 2982 | -1 |
| Rear left | 15 | 3631 | -1 |
| Rear right | 16 | 3423 | +1 |

[`servos/calibration-ankles.json`](servos/calibration-ankles.json) records these
centers and is also saved at the matching path under `/home/rd/drobot2` on the
Pi. ID 13 initially read 4085 at neutral, only 11 ticks from encoder wrap, so
its persistent midpoint was reset at that same physical pose to 2048. The
original capture was backed up under `servos/backups/` on the Pi.

The dashboard appends this ankle record to each original three-motor profile
in memory, giving four motor controls per leg. Positive ankle pitch raises the
sole edge facing the robot front, consistent with the URDF's positive ankle
axes at nominal abduction. The front-left direction was measured from a
user-applied forward-edge-up movement: ID 13 increased from 2048 to 2308.
The same physical movement at the front-right ankle decreased ID 14 from
2982 to 2784, giving direction -1. At the rear-left ankle, ID 15 decreased
from 3631 to 3410, also giving direction -1. At the rear-right ankle, ID 16
increased from 3423 to 3767, giving direction +1. All four direction samples
are recorded in the JSON with timestamps and encoder deltas. Manual travel
extrema are recorded below. The dashboard applies a five-degree inward margin
from these observations. Existing leg 1-4 profiles and calibrations remain unchanged.

The capture used torque-off commands and encoder reads, plus the existing
midpoint-setting routine for ID 13. Direction capture reads user-positioned
ankles. No powered movement tests or automated tests were run. All four ankle
torque-off commands were acknowledged. The Pi manual dashboard was left
stopped after the calibration captures.

## Recorded manual ankle travel

[`servos/ankle-travel-observations.json`](servos/ankle-travel-observations.json)
contains the completed manual recording from 2026-09-18 (Pacific time), also
saved at the matching path on the Pi. The 150.657-second recording contains
3,009 readings per ankle, zero read errors, and acknowledged torque-off
commands for all four ankles at completion.

| Corner | Servo ID | Observed minimum | Observed maximum |
| --- | ---: | ---: | ---: |
| Front left | 13 | -116.10 degrees | +114.26 degrees |
| Front right | 14 | -115.22 degrees | +115.66 degrees |
| Rear left | 15 | -121.90 degrees | +108.63 degrees |
| Rear right | 16 | -109.78 degrees | +121.55 degrees |

Angles are relative to the saved neutral centers with the measured direction
signs. IDs 14-16 crossed the encoder wrap point during the manual movement;
continuous angle tracking accounts for this. The JSON retains both raw tick
extrema and the raw ticks corresponding to angular extrema. Raw tick minimum
and maximum alone do not describe a continuous interval across encoder wrap.

The raw readings and session files are saved under
`debug/hardware-calibration/ankle-travel-20260919T041723005021Z/` on both the Pi
and Windows PC. These are observed hand-moved ranges in the calibration pose;
the capture itself applied no limits. The dashboard now uses a five-degree
inward margin as software limits; no servo EEPROM limits were changed.
These ranges do not establish clearance in every leg pose. No powered movement
tests or automated verification were run for the dashboard update.

## Continuous manual ankle travel recording

[`scripts/record-ankle-limits.py`](scripts/record-ankle-limits.py) runs on the
Pi and records all four ankle encoders while the operator moves them by hand,
one at a time. It refuses to start while the manual dashboard or onboard
service is active, disables ankle torque, and sends no position commands.

Run it with the manual dashboard stopped and the robot supported:

```bash
onboard/.manual-venv/bin/python hardware/robot-runtime/scripts/record-ankle-limits.py \
  --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034018-if00 \
  --calibration hardware/robot-runtime/servos/calibration-ankles.json \
  --output-dir debug/hardware-calibration/NEW_SESSION
```

Use a new output directory for each session. The default duration is 15
minutes with a target sampling rate of 20 Hz per ankle. `samples.csv` stores
the readings; `bounds.json` saves running raw tick and calibrated angle
extrema once per second. Angles unwrap successive readings across encoder
wrap. Create `stop.request` inside the output directory, or interrupt the
foreground process, to finish early. The recorder attempts to leave all four
ankles torque-off on exit.

Move slowly and stop before interference or cable tension. Recorded extrema
are observations, not validated safe limits, and are not automatically
applied to the runtime. Read errors and the final capture status are retained
with the results. This helper was added for the requested live recording;
automated tests and powered movement tests were not run.

## Sixteen-motor dashboard and walking policy

The dashboard exposes an ankle row (motor 4) per leg: FL ID 13, FR ID 14,
RL ID 15, RR ID 16. Arm, target, zero, telemetry and disarm include these
motors. Positive ankle angle means toe up. Whole-robot zero recapture is
disabled because changing an ankle center would invalidate its travel data.
Legacy three-motor crawl controls are disabled while ankles are configured.

Software ankle limits, in degrees: FL -111.10 to +109.26; FR -110.22 to
+110.66; RL -116.90 to +103.63; RR -104.78 to +116.55. These are observed
travel minus five degrees at each end, not a full-pose collision envelope.

The default RL model is [V2.3 transfer seed 2311](../../onboard/models/ankle-v23-transfer-seed2311/README.md).
It is experimental, with 82 observations and 16 actions. Its simulation
selection failed multiple criteria, including rear-left stepping. The new
NumPy hardware port has not been tested. No inference, builds, simulations,
UI validation or powered checks were run during this update.

Open `http://pi5-dog.local:8080`. PREPARE RL STANCE moves all 16 joints at
20 degrees/s into the training pose; RL START remains a separate action.
The supported first trial defaults to five seconds at 0.018 m/s. A normal
completion ramps to zero over one second and holds; explicit Stop disarms.
The runtime caps trials at 15 seconds and retains position, IMU, feedback,
temperature and output deadline guards. Ankle policy timing faults disarm.

## Setup and guarded commands

Install both hardware applications first:

```powershell
.\hardware\test-apps\install-test-apps.ps1
```

Then run helpers from the repository root:

```powershell
.\hardware\robot-runtime\scripts\show-status.ps1 -Leg 4 -Port COM4
.\hardware\robot-runtime\scripts\configure-leg.ps1 -Leg 4 -Port COM4
.\hardware\robot-runtime\scripts\calibrate-leg.ps1 -Leg 4 -Port COM4
.\hardware\robot-runtime\scripts\start-web-control.ps1 -Leg 4 -Port COM4
```

Calibration overwrites the selected shared `calibration-leg-N.json`. The
four-leg dashboard creates timestamped backups under `servos/backups/` before
persisting a multi-leg center capture; that generated backup directory is
ignored by Git.

USB disconnect, browser heartbeat, `Ctrl+C`, and software disarm are not
emergency stops. Support the robot, clear the full linkage sweep, and keep the
physical servo-power cutoff within reach before any command that may arm or
move a motor.
