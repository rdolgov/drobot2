# Why the rear lift disappears in the saved IK recording

Read-only follow-up: no simulator, controller, training, tests or hardware was
run. These calculations use the saved recording trace and existing URDF.

The rear leg does retract. Most of its upward motion is canceled by body
rotation and body drop, leaving the foot on the floor. This distinguishes
failure of whole-body support from a rear servo that simply never moves.

## Recorded lift decomposed

The baseline is the last saved weight-transfer sample before lift; the end
is the failed lift gate. Values below describe sole-center vertical motion,
not the lowest corner's clearance. All distances are millimeters.

| Trial / selected rear leg | Joint-motion contribution | Body rotation | Body drop | Net sole-center rise |
| --- | ---: | ---: | ---: | ---: |
| Nominal crawl / RR | +22.6 | -17.4 | -4.8 | +0.4 |
| Nominal diagonal FL + RR / RR | +22.7 | -15.7 | -4.9 | +2.1 |
| Nominal diagonal FR + RL / RL | +20.2 | -13.0 | -4.7 | +2.5 |
| 85% diagonal FR + RL / RL | +23.7 | +0.2 | -23.4 | +0.6 |

The nominal diagonal FL + RR foot's lowest corner remains at -0.6 mm
(contact penetration), despite its sole center rising 2.1 mm as the foot
tilts. Its front-left partner rises approximately 29.8 mm at the sole center,
including +12.3 mm from the same body rotation. The resulting tilt therefore
helps lift the front foot while suppressing the rear foot's clearance.

This is an exact algebraic accounting, not isolated causal experiments:

`delta world Z = delta root Z + [R0*(L1-L0)]Z + [(R1-R0)*L1]Z`

Joint and rotation terms depend on this stated ordering. The saved samples
are about 10 Hz. Calculation and full values are in
[`lift-motion-decomposition-20260920.json`](lift-motion-decomposition-20260920.json)
and [`lift_motion_decomposition.py`](lift_motion_decomposition.py).

## Support ankles and floor orientation

At nominal FL + RR timeout, total body tilt is 4.20 degrees. The requested
sole targets are level relative to that tilted chassis, so their resulting
world tilt would also be about 4.20 degrees. The actual supporting rear-left
and front-right soles are only 1.28 and 1.31 degrees from floor level, with
corner height spans of 2.02 and 2.12 mm. The floor constraints and joint
deflection keep them closer to level than their chassis-relative targets.

The rear-left supporting hip, knee and ankle errors are approximately
-6.0, +9.2 and -7.6 degrees; all three are at the nominal 0.883 Nm cap.
Front-right support knee and ankle are also capped. The selected rear-right
ankle itself is at only 0.197 Nm, although its knee is capped. This does not
support treating the new rear ankle alone as the failed actuator.

The ankle target omits measured hip/knee deflection and world/body attitude.
That is a concrete controller limitation, but these traces do not establish
that ankle leveling alone would restore support. Earlier attitude/support
compensation prototypes also failed; do not re-run them unchanged or just
increase swing lift, relax gates, or raise effort caps.

## Reconstruction and limits

The saved trace lacks body quaternion and per-contact wrench. Measured joint
positions were applied to the existing URDF's forward-kinematic transforms.
A proper rigid rotation/translation was fitted from the four body-frame
sole centers to the four recorded world-frame centers. Maximum center-fit
residual was 0.00098 mm. Reconstructed low/high sole corners agree with the
recorded values within 0.00013 mm. This supports the coordinate accounting;
it is not independent evidence that the simulator matches real hardware.

Only net vertical forces are recorded. Actual contact-pressure distribution,
center of pressure and contact moments cannot be recovered, so no claim about
their precise locations is made. The 3.395 kg working mass is accepted; a new
weigh-in is not blocking this work. Loaded actuator response remains a model
assumption, and the available evidence does not prove a particular controller
correction will work.

See [`sole-attitude-audit-20260920.json`](sole-attitude-audit-20260920.json)
for per-sample orientation, force, torque and tracking calculations, input
hashes and source [`sole_attitude_audit.py`](sole_attitude_audit.py).

No training or recording worker is active. The useful scheduled read-only
investigation is complete. Further progress needs a distinct, justified
controller experiment or new loaded-response evidence; the current user
instruction requires specific authorization before executing additional
verification. The idle recurring monitor is paused rather than repeatedly
reporting the same failure. No controller or Pi changes were made here.
