# R4 checkpoint 1750: geometry and support-load audit

The URDF skill and its frame-semantics reference were used to inspect the
actual training-input URDF. No URDF, USD, controller training source or
physical limit was changed. The current R4 training continued during this
four-trial deterministic diagnostic at full curriculum.

## Geometry finding

Exact XML forward kinematics, using the saved measured root quaternion and
all measured joint positions, reproduce all 16 terminal ankle-pivot positions
across the four trials to within 0.00045 mm. Results and the input hash are
in `fk-audit.json`. This excludes a front/rear transform or joint-order mistake
in these samples; it does not validate unmeasured hardware calibration.

The body and support legs lower under load. Therefore the planar ankle-pivot
down coordinate can approach its lift target while the rear sole remains on
the ground. The two-link gate remains a body-relative path proxy, not floor
clearance. Do not loosen the gate and treat its progression as successful lift.
Rear soles in the terminal snapshots are largely flat, rather than simply
having an unmodeled toe/heel tilt. The diagnostic records both sole low/high
heights, physical hip/ankle pivots, foot orientations and root pose.

## Offline static load calculation

`simulation/isaac/diagnostics/diagonal_static_load.py` reads the same URDF and
computes the whole-body COM and every actuated joint's gravity moment. A
linear program allocates nonnegative **vertical** forces over the two support
sole patches to minimize the largest joint torque while balancing vertical
force and roll/pitch moments. See `../static-load-screen.json`.

At the existing first-lift nominal pose, the two diagonal cases require at
least approximately 0.821 and 0.824 Nm under this vertical-force allocation,
versus a 0.883 Nm nominal cap and 0.750 Nm cap for the 85% trial motors.
This suggests little load margin, but is not a proof of impossibility: the
calculation excludes horizontal friction forces, transient dynamics, control
lag, collision, and real contact-pressure distribution. It is a restricted
static model, not an independent gait evaluation.

A bounded posture screen found lower static loads with straighter legs,
smaller rear fore/aft offsets and pair-dependent lateral shift. Candidate
minima around 0.46 Nm are theoretical, not achieved simulator results.
Some shifted candidates tilt their rigid soles; the report includes support
corner height range. Allowing forces at all corners is optimistic when those
corners are not coplanar. Earlier symmetric upright dynamic trials fell, so
do not adopt a screened posture on the torque number alone. Any candidate
needs continuous startup/transfer, travel/reach review throughout the long
stride, actual loaded clearance and contact/stop diagnostics before training.

The next correction should address rear unloading and supporting-leg load,
alongside the already diagnosed lift-cancelling residuals. Preserve R4 and its
evidence; keep hardware and actuator limits unchanged.
