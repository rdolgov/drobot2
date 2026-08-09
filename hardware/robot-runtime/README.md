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

All profiles use 1,000,000 baud, 90% torque limit, speed 700, acceleration 10,
and a five-degree maximum interactive command step. The dashboard motion ramp
is 90 degrees/s. Software limits do not replace physical stops or collision
checks.

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
