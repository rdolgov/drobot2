# Larger diagonal IK sequence prepared for the Pi

Historical V2 design, requested and deployed September 20, 2026 without
automated verification. The operator subsequently reported backward drift.
The current change is [V3 support push](ankle-ik-diagonal-v3-support-push.md).
Earlier V1 videos and test results do not validate either revision.

## Sequence

`[ankle_ik_diagonal]` selects a 100 mm stride, 35 mm lift, 8-second reference
cycle and two-cycle run limit. The selected pairs are front-left/rear-right,
then front-right/rear-left. Each pair completes transfer, lift, forward swing,
lower, plant, all-feet push and settle before the next pair starts. The
post-push settle endpoint also waits for encoder tracking before changing
pairs. Nominal
cycle duration excludes encoder waits and startup positioning.

Depth is 288 mm at the ankle pivot; fore/aft stance offsets are 80 mm. There
is no scheduled lateral or fore/aft body transfer in this profile. The nominal
depth stays below the full-stride reach bound, avoiding cyclic height changes
from the old reach adjustment. Quintic trajectories retain smooth endpoints.

Independent two-link calculations over a conservative envelope give front
hip 31.1–54.4 degrees, knee -74.4 to -17.7 degrees, ankle -15.4 to +30.4 degrees,
and maximum reach 315.98 mm. Rear joints use the mirrored branch. These are
geometry calculations, not execution tests or a balance/torque guarantee.

Single-leg crawl keeps its existing `[ankle_ik]` parameters. The dashboard's
diagonal button shows `DIAGONAL 100 MM / 2 CYCLES`. It starts from disarmed,
initializes the BNO085 without arming, then positions and settles the robot
before stepping. STOP can cancel sensor initialization. At two cycles it
holds the completed pose with torque on; STOP during motion finishes the
current placement. DISARM remains immediate torque-off.

## Ankle leveling

During diagonal preparation/motion, the ankle pitch target uses measured hip
and knee angles plus BNO085 gravity in the existing calibrated body frame.
For measured abduction alpha (world +X rotation, mirrored on right legs),
the available-plane pitch target is
`atan2(gx, gy*sin(alpha) - gz*cos(alpha))`. Ankle target is that pitch minus
measured hip and knee. This maximizes sole-normal alignment with world up in
the ankle's available pitch plane. A pitch-only ankle cannot independently
remove sole roll, and there is no ground-contact or force sensor.

The correction relative to nominal IK is restricted to ±8 degrees, smoothed
with a 100 ms filter and 30 degrees/s correction slew. All resulting ankle
targets retain the ±33.25-degree gait limit and individual calibrated travel
checks. The UI reports `LEVELING LIMITED` if the correction or ankle range
clips; the controller does not silently claim a flat sole in that condition.

IMU acquisition uses the existing background BNO085 source at 100 Hz with
120 ms stale-data rejection. Gravity must have valid magnitude and body
tilt must be below 20 degrees. Leveling refreshes alongside joint feedback
at approximately 20 Hz. A delayed correction update over 200 ms faults.
No alternate sensor axis map or fabricated hardware attitude is substituted;
only explicit demo mode uses synthetic upright gravity.

The trajectory retains the 5-degree measured tracking gate, 0.3-degree
delivery tolerance, 18 degrees/s settling speed and 150 ms settling hold.
Its ankle gate compares the compensated target. The hip/knee gates are
unchanged. It still faults if an endpoint cannot settle. This is not a way
to force a stalled real leg into the next phase.

Motor power settings, calibrations and servo directions remain unchanged.
Nominal sagittal/ankle output remains at most 100 degrees/s, abduction at
60 degrees/s, with no packet larger than two degrees and no timing catch-up.
The existing motion-fault path disarms. Initial physical trials need body
support until directions and clearances are established.

## Recording and deployment

On the normal Pi launch configuration, IK trials share the existing download
archive store. They are labeled `IK diagonal 100 mm` and record measured,
desired and delivered joint angles, velocities, cached body gravity, phase,
gate and leveling-limit state. Browser telemetry adds voltage, temperature,
current and stall indications. Diagnostic servo current is not a calibrated
joint torque measurement; gravity in a joint sample is the latest cached
leveling value, not a synchronized sensor acquisition.

The prepared bundle is under
`debug/deployments/ik-diagonal-bigstride-20260920/`. It includes only the
dashboard controller, IK core, browser JS/HTML and dashboard manifest as
runtime changes. Calibration and sensor/runtime dependencies are read-only
hash prerequisites. The installer rejects unexpected Pi file revisions,
backs up replaced files, and restores them if installation fails. It does
not run tests, simulations, motion commands or a walk. Applying the bundle
restarts the existing dashboard service, which owns the hardware adapter.

No upload, Pi connection, hardware check, compilation, test suite, simulation
or training was run while preparing this revision. Full-stride geometry and
source inspection are the only calculations/review performed here.
