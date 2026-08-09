# Distributed-push crawl V5

The filename is retained as a stable documentation link. Its contents now
specify the V5 crawl that replaces the unsuccessful manually proposed
three-joint sequence.

## Decision

V5 is a deterministic, open-loop static crawl based on foot positions rather
than isolated joint commands. One foot is lifted at a time, the other three
remain support candidates, and an all-feet push follows every touchdown. The
same equations run in the hardware dashboard and Isaac; Isaac is not required
at runtime.

The design borrows two established ideas rather than copying a robot-specific
trajectory:

- static quadruped crawl literature divides each step into a center-of-gravity
  transfer plus a swing and recommends the sequence rear right, front right,
  rear left, front left for the studied geometry; and
- the open-source 12-servo SpotMicro implementation similarly combines a
  single-foot swing with body/center-of-mass motion.

Sources:

- [Static Gait Generation for Quadruped Robots with Optimized Walking Speed](https://linqi-ye.github.io/docs/2020A%20Static%20Gait%20Generation%20for%20Quadruped%20Robots%20with%20Optimized%20Walking%20Speed.pdf)
- [SpotMicroESP32-Leika gait implementation](https://github.com/runeharlyk/SpotMicroESP32-Leika/)

## Foot order and phase table

One approximately 6.67-second cycle contains four 1.67-second steps in this
order:

1. rear right (Leg 4);
2. front right (Leg 2);
3. rear left (Leg 3); and
4. front left (Leg 1).

Each step uses the following normalized phase table. Smoothstep interpolation
removes velocity discontinuities at the ends of each phase.

| Phase | Step fraction | Time in a 1.67 s step | Purpose |
| --- | ---: | ---: | --- |
| Weight transfer | 0.16 | 0.267 s | Move the body target fore/aft away from the foot to lift |
| Lift | 0.10 | 0.167 s | Raise only the selected foot |
| Swing | 0.20 | 0.333 s | Advance the selected foot by one stride |
| Lower | 0.10 | 0.167 s | Return the selected foot toward the floor |
| Touchdown | 0.08 | 0.133 s | Hold for contact to develop |
| Weight return | 0.10 | 0.167 s | Remove the temporary body shift |
| All-feet push | 0.24 | 0.400 s | Move all planted foot targets rearward by one quarter stride |
| Step settle | 0.02 | 0.033 s | Hold the completed step |

After four steps, each foot has moved forward by one complete stride while it
was airborne and rearward by four quarter-stride pushes while planted. The
joint targets are therefore periodic without a large whole-body reset.

## Geometry and inverse kinematics

Each sagittal leg uses a two-link analytic inverse-kinematics model with equal
link length `L = 0.159896689 m`. For a downward foot coordinate `d` and forward
coordinate `x`:

```text
r^2 = d^2 + x^2
knee = acos((r^2 - 2 L^2) / (2 L^2))
hip  = atan2(x, d) - atan2(L sin(knee), L + L cos(knee))
```

The IK branch is mirrored so the distal links bend away from the chassis. The
final calibrated motor convention is deliberately explicit:

- front knees command negative values to move downward;
- rear knees command positive values to move downward; and
- the left/right signs for hip abduction continue to come from each tracked
  servo profile.

This fixes the earlier failure where the front and rear legs folded toward one
another. The gait start is the computed phase-zero pose, not the old hardcoded
45-degree stance.

## Selected parameters

| Parameter | Tracked value |
| --- | ---: |
| Cycle count | 2 |
| Period per cycle | 6.666666667 s |
| Total command duration | 13.333333334 s plus stance settling |
| Foot stride | 0.112 m |
| Foot lift | 0.014 m |
| Fore/aft weight transfer | 0.016 m |
| Lateral weight transfer | 0.000 m |
| Nominal downward reach | 0.300 m |
| Nominal front/rear foot separation from each hip | 0.025 m |
| Additional hip-abduction stance angle | 0 degrees |

The current 112 mm raised-foot move is twice the preceding 56 mm physical-trial
setting and four times the 28 mm simulated baseline. Each all-feet phase now
pushes the planted targets rearward by 28 mm. Nominal downward reach was reduced
from 305 mm to 300 mm so the most extended forward target remains inside the
two-link geometric reach. The complete trajectory runs three times faster than
the simulated baseline. Lift and weight-transfer distances are unchanged.
Zero lateral shift is retained from the passing baseline.

## Aggressive actuator settings

The four tracked servo profiles now use the same settings:

| Setting | Value |
| --- | ---: |
| Torque limit | 900 / 1000 (90%) |
| Speed register | 3400 |
| Acceleration register | 254 |
| Dashboard ramp limit | 270 degrees/s |
| Maximum command step | 15 degrees |

`3400` is the maximum ST3215 speed setting exposed by the current software and
`254` is the maximum acceleration byte. The dashboard ramp also equals the
URDF's documented no-load joint-speed limit of `4.712389 rad/s` (270
degrees/s). These are response limits, not a promise that a loaded joint will
reach no-load speed. The tracked torque setting remains the requested 90%; it
is not raised to continuous stall torque.

Vendor references:

- [Waveshare ST3215 documentation](https://www.waveshare.com/wiki/ST3215_Servo)
- [ST3215-C047 datasheet](https://files.seeedstudio.com/products/Feetech/108090003_FEETECH_ST-3215-C047-Datasheet.pdf)

## Isaac evidence

Both retained runs used the floating 4.526 kg robot, 120 Hz physics, 60 Hz
control, the V5 equations with the earlier 28 mm stride and 20-second period,
two cycles, and the same provisional printed-PLA contact model. They do not
validate the 112 mm stride, 300 mm downward reach, or tripled trajectory rate.

| Metric | Sustainable rated cap | Peak/stall cap |
| --- | ---: | ---: |
| Result | PASS | PASS |
| Per-joint torque cap | 0.980665 Nm | 2.941995 Nm |
| Forward displacement | 60.00 mm | 59.73 mm |
| Maximum body tilt | 1.42 degrees | 1.40 degrees |
| Maximum joint error | 0.100 rad | 0.100 rad |
| Maximum joint speed | 1.448 rad/s | 1.842 rad/s |
| Maximum support-tip slip | 3.67 mm | 3.75 mm |
| Completed foot steps | RR, FR, RL, FL | RR, FR, RL, FL |

Reports:

- [`../validation/isaac-distributed-push-v5-selected.json`](../validation/isaac-distributed-push-v5-selected.json)
- [`../validation/isaac-distributed-push-max-power-control.json`](../validation/isaac-distributed-push-max-power-control.json)

The peak/stall run had more torque reserve and greater achieved joint speed but
did not move farther. This is why the physical profiles combine maximum speed
and acceleration settings with a 90% torque cap, rather than using simulated
stall torque as the nominal design point.

## Physical trial

Isaac verifies internal target consistency and screens for obvious collapse;
it does not verify the real foot friction, backlash, supply voltage drop,
wiring, servo heat, or exact mass distribution. For the first V5 trial:

1. Restart the dashboard so it reloads the TOML profiles and Python gait code.
2. With the body supported, click **SET GAIT START STANCE** and verify front
   knees bend forward/down and rear knees bend rearward/down.
3. Keep a tether or support ready, place the feet on a high-friction surface,
   and click **TEST DISTRIBUTED CRAWL** once.
4. Use **STOP + DISARM** for a fall tendency, collision, cable pull, severe
   voltage sag, unexpected heat, noise, or the wrong joint direction.
5. Record which phase and leg first disagrees with the intended motion. Change
   one gait parameter at a time after that observation.

This remains open-loop choreography. It has no IMU balance correction, foot
contact feedback, or closed-loop center-of-pressure control, so maximum servo
speed alone cannot make it actively balance after a disturbance.
