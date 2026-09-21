# Current IK failure recordings

Requested September 20, 2026. These are actual Isaac Sim renders of the
current 16-servo deterministic IK controller, using the unchanged V4 gait
parameters. They are not RL-policy footage or physical robot footage.

- [Diagonal FL + RR, 18.4 seconds](trial-2-diagonal.mp4) — start here.
- [Opposite diagonal FR + RL, 18.4 seconds](trial-4-diagonal.mp4).
- [Single-leg crawl, 16.0 seconds](trial-0-crawl.mp4).

Each clip shows the original attempt at real-time playback, the final lift
again at half speed, and a three-second freeze. The visible simulation clock
is the source time; it repeats during replay. All three are 1280×720 H.264,
30 frames/s, with foot clearance, vertical force, joint tracking error and
torque-limit counts. The camera looks from the front-right side.

In the FL + RR clip, watch 3.9–7.4 seconds of simulation time. Front-left
reaches approximately 25.5 mm clearance; rear-right remains in contact
(-0.6 mm simulated clearance, 1.3 N vertical force). The opposite support
feet carry about 17.9 N and 14.3 N, six joints reach the simulated torque cap,
and the largest joint tracking error is 9.2 degrees. The lift gate times out
at 7.4 seconds before forward swing. The opposite diagonal likewise lifts
its front foot while failing to clear its rear foot. Crawl unintentionally
unloads the front-left support foot while the selected rear-right stays down.

This identifies the observed failure as unsuccessful rear-foot unloading
and joint tracking under load. It does not establish whether the same torque
cap and loaded response match the physical servos. No actuator limits,
gait targets, calibration, or learned weights were changed for recording.

The six simulated trials reproduce the previous V4 timeout messages and
failure times. Only nominal-strength trials 0, 2 and 4 were filmed. All have
zero qualified steps. The simulation ends on failure, not an automatic reset
or a successful continuation. See [report.json](report.json),
[trace.jsonl](trace.jsonl) and the source snapshot in `provenance/`.

The source state is rendered without physics writes during capture. The
annotation reports selected/support roles from the requested trajectory,
while clearance and force are measured simulator values. Negative clearance
denotes small contact penetration. The diagnostic holds a failed trial's
last target; the dashboard uses its separate fault/disarm path.

MP4 codec, frame count and duration were read back with ffprobe; a frame was
decoded from the diagonal video and all three final frames were inspected.
Only the requested recording and its media checks were run; no training,
test suite or hardware check was launched.

Reproduction (use a fresh directory):

```powershell
C:/isaacsim/python.bat simulation/isaac/diagnostics/ankle_ik_walk.py --output simulation/reviews/ankle-ik-walk/NEW-VIDEO --seconds 40 --record
```
