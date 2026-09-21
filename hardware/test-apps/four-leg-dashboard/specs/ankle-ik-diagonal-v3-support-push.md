# Diagonal IK V3: planted-ankle forward push

Requested September 20, 2026 after the operator reported some backward drift
with V2. This revision changes the commanded support trajectory; actual
forward travel, friction, balance and actuator strength remain unverified.
No automated tests, simulation, compilation or hardware trial were run.

## Motion

The 100 mm stride, 35 mm lift, 8-second reference cycle and two-cycle limit
remain. Front-left/rear-right alternate with front-right/rear-left. Each
placement advances the body reference by 50 mm in total:

- Lift: the planted diagonal drives 5 mm of forward advance.
- Swing: that diagonal drives another 15 mm, reaching 20 mm.
- Lower and plant: retain the 20 mm advance while the moving feet land.
- All-feet push: complete the remaining 30 mm, then settle before swapping.

All segments retain the quintic profile and existing encoder waits. Forward
advance subtracts from every ankle-pivot coordinate, including the swing
pair, so the ideal world-space swing stays 100 mm when the body follows the
reference. The early 20 mm is part of the existing 50 mm advance, never an
extra offset that must be reset backward after landing. `support_push_m`
defaults to zero in the common config and is 0.020 in the diagonal profile;
its accepted bound is the smaller of 25 mm or half the stride. Single-leg
crawl keeps its previous trajectory.

## Loaded ankles and leveling

The scheduled support ankles follow the planned hip/knee IK angles plus
measured IMU pitch compensation during lift, swing, lower and plant. This
retains their propulsive feedforward: delayed measured hip/knee motion can
no longer cancel the planned support-ankle sweep. Both pairs use this
reference during the final all-feet push and settle. Startup and weight
transfer retain measured-angle leveling, as do the swinging ankles.

The intended sole pitch remains level. There is no separate toe-tip impulse
or commanded forward fall. Actual foot pitch may differ while loaded joints
lag the reference. The ±8-degree leveling correction, ±33.25-degree ankle
cap, calibrated joint ranges, motion rate/packet limits, gravity/tilt limits,
tracking timeouts and stop/disarm behavior remain. No torque limit or servo
direction is changed. Scheduled support is not measured ground contact.

Independent two-link endpoint arithmetic (no controller imported or run)
gives the front support ankle 5.30 → 11.32 degrees and the rear support ankle
-5.30 → 1.57 degrees over the 20 mm push: increases of 6.02 and 6.87 degrees.
At the front lift endpoint the pivot is x=25 mm, down=253 mm, with ankle
31.70 degrees. The rear lift endpoint is x=-135 mm, down=253 mm, with ankle
1.81 degrees. The largest endpoint reach is 315.98 mm, below the retained
317 mm reach bound. These calculations do not validate motion or contact.

## UI, recordings and deployment

Client version 15 labels the button
`DIAGONAL 100 MM + ANKLE PUSH / 2 CYCLES`. The status displays planned stance
push in mm and the corners currently assigned ankle drive. It does not
display these as measured body translation or confirmed load-bearing feet.
Start still requires disarmed motors and prepares its own stance.

Trial label: `IK diagonal 100 mm / ankle push`; controller revision:
`ankle_ik_v3_support_push`. The existing `ankle_ik_v2` sample schema is kept
with additive revision, support-push, per-placement body-advance and ankle
feedforward-corner fields. Actual, desired and delivered joint angles remain
in each sample for the next operator trial.

The source-only release is under
`debug/deployments/ik-diagonal-support-push-20260920/`. Installation replaces
five runtime files with a rollback backup and preserves the Pi's calibration
and learned weights. Activation requires a controller restart with the robot
supported and disarmed. Installation does not start a walk.
