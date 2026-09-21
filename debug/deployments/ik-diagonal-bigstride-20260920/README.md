# Diagonal IK V2 — installed on Pi

Installed on 2026-09-20 at 16:27 UTC on `rd@pi5-dog.local` (`192.168.1.58`),
repository `/home/rd/drobot2`. The `drobot-manual-web` dashboard service was
restarted and systemd reported `active/running`, with `ExecMainStatus=0`.
No tests, compilation, simulation, UI validation or hardware checks were run.
No walk was commanded. See [deployment record](applied.json).

Rollback snapshot on the Pi:
`/home/rd/drobot2/debug/deployment-backups/ik-diagonal-20260920T162757391037Z`.

## What is included

- 100 mm diagonal strides, 35 mm lift and an 8-second reference cycle.
- FL + RR together, then FR + RL together, with full placement and settle.
- Two cycles, then torque hold; STOP finishes the current placement.
- BNO085 gravity and measured hip/knee feedback for bounded ankle pitch
  leveling. ±8-degree correction, existing ±33.25-degree ankle target cap,
  existing servo calibration/travel and power settings.
- Joint/IMU traces plus browser electrical telemetry in the existing
  Settings → Trial recordings download archive.

The nominal trajectory parameters are larger than those in the earlier
videos. Geometry arithmetic fits the retained reach/joint envelope; this
does not establish real balance, floor contact, or motor strength. A
pitch-only ankle cannot fully correct foot roll. See the
[V2 controller description](../../../hardware/test-apps/four-leg-dashboard/specs/ankle-ik-diagonal-v2.md)
and [geometry calculation](geometry-calculation.json).

`ik-diagonal-bigstride-v2-pi-baseline.zip` replaces only five runtime files: the dashboard
manifest, Python session controller, new IK core, browser JS and HTML.
Calibration/leg profiles, runtime sensor/recorder code and ankle policy code
are read-only prerequisites whose hashes must match. No calibration or model
weight file is replaced. The archive checksum is in `archive.sha256`.

The first installation attempt stopped before modifying runtime files because
the Pi has newer calibration for legs 1, 3 and 4 (2026-08-29, with 2–4 tick
center differences from the local copies). All 13 prerequisite files were
downloaded and compared. The reviewed bundle requires those exact Pi
calibration hashes and preserves the files in place; all other prerequisites
matched. The five replacement source files also matched the expected prior
hashes. See [comparison record](pi-prerequisite-review.json).

The original `ik-diagonal-bigstride-v2.zip` and
`deployment-manifest.prepared.json` retain the pre-upload package history.
Only prerequisite metadata changed in the installed bundle; the runtime
payload is identical. Any future source mismatch still aborts installation.

## Re-uploading this release

The release above was applied using SSH/SCP. These helper commands can be run
from this directory in PowerShell. The first uploads the archive/installer and checks only source
and prerequisite hashes; it does not replace runtime files or restart the app:

```powershell
.\upload-to-pi.ps1
```

To apply when the robot is supported and idle, or servo power is off:

```powershell
.\upload-to-pi.ps1 -Apply
```

Applying makes a timestamped source backup, stops the dashboard service,
replaces the five files atomically per file, then restarts it if it was
previously active. A failed installation restores replaced sources. The
service restart opens its normal hardware adapter and disarms on startup;
no walk or motion check is sent. There is no automated functional check.

Open `http://pi5-dog.local:8080/` and reload the page for client version 14.
The button is **DIAGONAL 100 MM / 2 CYCLES**. Starting requires disarmed motors.
It initializes the IMU, moves to the start pose, settles, and runs the two
cycles. First establish directions/clearance with the body supported. Floor
trials are a separate operator action with a clear area and power cutoff
ready. `LEVELING LIMITED` means the requested flat-pitch correction clipped.
The existing tracking timeout still stops a stalled sequence; do not bypass it
to force the other pair forward. Use Settings → Trial recordings to retain
the real run for diagnosis.

## Restore a previous source snapshot

For this installation, run on the Pi:

```bash
python3 ~/apply-ik-diagonal-v2.py --restore ~/drobot2/debug/deployment-backups/ik-diagonal-20260920T162757391037Z --apply
```

Restore rejects files modified after installation and preserves calibration.
It stops/restarts the dashboard as necessary and does not command walking.
