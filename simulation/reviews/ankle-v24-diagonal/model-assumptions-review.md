# Physical assumptions to resolve before another blind training repeat

R6 remains running at the time of this review. Its current source and asset
have not been changed. Failed local diagnostics establish failure of the tested
controllers, not impossibility of diagonal walking on the real robot.

The active V2.3 solid-sole asset has a total mass of 3.444019908818 kg, matching
its manifest and URDF. `simulation/docs/rl-ankle-v23-stride.md` states that added
part masses use CAD volume, 806 kg/m3 effective printed density, 55 g per servo
and hardware/tread allowances, and are unmeasured. Earlier
`simulation/docs/rl-robust-straight-low-stance-v25.md` records a measured 7 lb
(3.17514659 kg) assembled robot before the 16-servo ankle upgrade. That earlier
measurement must not simply replace the current asset mass.

The current actuator model in `simulation/isaac/rl/ankle_v2/env_cfg.py` uses a
0.8825985 Nm conservative cap, 2.6477955 Nm stall parameter, 4.712389 rad/s
no-load speed, 30 Nm/rad position stiffness and 4.5836625 damping. Training
randomizes strength/rate from 85% to 100%. These configured values are not a
direct current measurement of loaded hardware response. The servo profile's
`torque_limit = 900` is a bus setting and is not itself a calibrated torque
measurement in Nm.

The source/checkpoint audit found no frozen actor or normalizer. Support-load
traces show large knee tracking errors near configured torque caps. Multiple
bounded timing, posture and preload corrections still yielded no complete
diagonal steps. This motivates checking current physical mass and loaded
response before selecting a different physical training contract; it does not
justify silently increasing effort limits or weakening the success criteria.

A text question asking for an existing current total-weight measurement,
including battery and all 16 servos, was sent asynchronously during this check.
No answer is assumed and the current R6 run continues. This is missing model
information, not a request for permission to continue authorized training.
No Pi connection or physical motion was requested or performed.

## Completion update

R6 subsequently completed with zero full pairs and zero qualified rear
touchdowns. Its final deterministic diagnostic failed all four trials. A late
Beta-head collapse and adaptive learning-rate increase were also identified;
see `20260919-seed2419-r6/diagnosis.md`. The bounded R7 optimizer pilot uses
unchanged physics while the weight question remains pending. Neither the
older 7 lb measurement nor a guessed new mass has been substituted.

R7 has now completed all 250 iterations without the late uniform-policy
collapse, but its independent four-trial evaluation still produced zero full
pairs and zero rear qualifying touchdowns. Rear clearance peaked at only
0.061 mm in selected swing frames. Supporting-knee saturation and tracking
error persist. See `20260920-optimizer-pilot-r7/diagnosis.md` and its comparison
JSON. Scheduled repetitions are paused pending current physical information,
starting with the weight question already sent. The local training authorization
remains valid; this pause is about the model evidence, not approval.

## User-provided estimation method resolves the weight dependency

On September 20 the user directed use of the repository's measured 12-servo
weight plus four servos from their specification. Feetech confirms 55 ± 1 g
for ST-3215-C018. Thus 7 lb + 220 g gives an approximately 3.395 kg working
estimate. See `mass-estimate-20260920.md` and its JSON calculation. The previous
request for a new weigh-in is no longer a blocker and should not be repeated.
The existing 3.444 kg model already contains the four added motors; the total
difference is only 48.9 g. Diagnosis continues with a bounded mass-sensitivity
comparison while preserving the original per-part ledger and failed evidence.

## Subsequent result and IK alternative

The completed `r7-model249-mass3395` comparison still timed out in all four
trials, with zero full pairs and zero qualifying rear touchdowns. The user
also requested a deterministic IK alternative using the new ankle motors.
That local implementation and its failed simulation trials are recorded in
[`../ankle-ik-walk/diagnosis.md`](../ankle-ik-walk/diagnosis.md). Further
diagnosis should use these traces; the old pending-weight pause is obsolete.
