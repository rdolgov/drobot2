# Drobot transferred-foot lift audit (V19-V21)

This follow-up audit tested whether the verified isolated 190 mm foot-lift
policy could remain stable after the opposite front foot was placed on the
first 180 mm riser. Every test used the exact 250 mm tread scene, rated actuator
effort, a floating base, gravity, collisions, and the strict 200 mm lateral
corridor. The policy remained camera-blind: IMU, joint state, prior action,
support/contact telemetry, and the fixed analytic stair profile only.

The existing V17 isolated milestone remains the last accepted result. It passed
5/5 deterministic evaluations at 205.55-207.75 mm lift with no fall, 3.27 mm
maximum support slip, 2.33 degrees maximum tilt, and at least 43.31 mm support
margin. Its review video is
[`reviews/ppo-stairs-v17-single-foot-190mm-lift-success.mp4`](../../reviews/ppo-stairs-v17-single-foot-190mm-lift-success.mp4).

## Corrected training contract

The transfer-phase audit found two state-contract bugs and fixes them without
changing the 81-value observation shape:

- Navigation/reward lateral error now uses the mass-weighted 13-link composite
  center of mass relative to the retained support target. Previously the policy
  observed and was rewarded for torso-origin position while termination used
  whole-robot COM.
- Cached phase-training resets now retain the transfer base/COM targets.
  Previously those targets were zeroed, so phase-only PPO trained under a
  different controller state than full-sequence evaluation.

With the corrected frame, the deterministic transferred baseline increased
from 170.7 mm to 181.6 mm, but still left the strict corridor. V19 support-only
PPO then ran 8,192 steps in 327.7 seconds. Its last 20 stochastic episodes had
0 successes, 168.4-183.3 mm lift, 44.9-63.2 mm slip, and the correct retained
target near -88.6 mm. Deterministic evaluation reached 180.6 mm with 59.9 mm
rear-left slip and failed the corridor.

## Traction and adaptive-policy results

Temporary sensitivity configs were evaluated and then removed from the
canonical V16 config because none passed the physical gate:

| Experiment | Lift | Support slip | Stability result |
| --- | ---: | ---: | --- |
| 230 mm commanded apex, normal grip | 210.2 mm | 55.2 mm | Corridor failure |
| High grip, verified swing only | 86.9 mm | 28.4 mm | Stable for 24 s; insufficient lift |
| High grip, full legacy V17 actions | 224.9 mm | 59.1 mm | Corridor failure |
| V20 high-grip full residual, 8,192 steps | 230.9 mm deterministic | 64.5 mm | 0/20 recent successes; corridor failure |
| V21 high-grip curriculum from 100 mm, 8,192 steps | 86.9 mm | 21.8 mm | Stable, but no 100 mm mastery/transition |
| 140 mm transfer unload sensitivity | 213.8 mm | 52.2 mm | 36.9 degree tip failure |
| 120/130 mm transfer unload sensitivities | 169.9/174.8 mm | 51.2/64.0 mm | Corridor failures |

High grip therefore helps stance stability, but it also removes sliding motion
that the current post-transfer pose implicitly relies on for lift. The measured
swing tracking error reached about 0.73 rad under high grip at the rated effort
limit. More vision cannot correct that torque/posture mismatch. Conversely,
allowing legacy full-body actions restores height by reintroducing support-foot
slip, so that is not an acceptable policy.

## Recommended next test

Prioritize a torque-aware quasi-static transfer posture before another long PPO
run:

1. On the real robot, log commanded/measured swing joint angles, motor current,
   supply voltage, servo temperature, IMU roll/pitch, and support-foot motion
   during an unloaded 80-120 mm lift. Stop on rated-current or temperature
   limits.
2. In simulation, optimize a collision-free post-transfer joint pose that keeps
   all three stance feet fixed, composite COM inside the support polygon, and
   swing hip/knee torque below the measured continuous limit while raising the
   foot at least 140 mm.
3. Train a 100 -> 140 -> 190 mm RL curriculum from that feasible pose with
   friction, payload, voltage/torque, and contact-compliance randomization.
4. Use rubber pads/high traction, but randomize around the measured real value;
   do not train only at an optimistic coefficient.
5. Add depth/ToF or a camera later for unknown riser localization and landing.
   Vision is not the current blocker on a known fixed stair.

The accepted claim remains deliberately narrow: V17 proves a camera-blind
single-foot 190 mm lift beside a 250 mm tread. It does not yet prove the
opposite-foot transfer or a complete stair climb.
