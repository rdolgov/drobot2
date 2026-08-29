# Rectangular-shoe flat-support crawl V9 slow

The filename is retained as a stable documentation link. Its contents now
specify the active V9 rectangular-shoe crawl. Earlier V5/V8/TPU results remain historical
sections in this document.

## Decision

V9 is a deterministic, open-loop static crawl based on rectangular-shoe
contact positions rather than isolated joint commands. One foot is lifted at a
time while the other three remain exact flat-sole support candidates. The
selected foot advances, rejoins the flat branch before load, and a short
all-feet push follows. The same equations run in the hardware dashboard and
Isaac; Isaac is not required at runtime.

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

One 12-second cycle contains four 3-second steps in this
order:

1. rear right (Leg 4);
2. front right (Leg 2);
3. rear left (Leg 3); and
4. front left (Leg 1).

Each step uses the following normalized phase table. Smoothstep interpolation
removes velocity discontinuities at the ends of each phase.

| Phase | Step fraction | Time in a 3.00 s step | Purpose |
| --- | ---: | ---: | --- |
| Weight transfer | 0.15 | 0.450 s | Move the body target fore/aft away from the foot to lift |
| Lift | 0.15 | 0.450 s | Raise only the selected foot to full clearance |
| Swing | 0.20 | 0.600 s | Advance the selected foot while all three supports remain flat |
| Lower | 0.20 | 0.600 s | Return the selected foot toward the flat branch |
| Firm plant | 0.10 | 0.300 s | Hold the new flat contact with the other three soles level |
| Weight return | 0.08 | 0.240 s | Return the open-loop body transfer after contact |
| All-feet push | 0.07 | 0.210 s | Move all flat planted targets rearward by one quarter stride |
| Step settle | 0.05 | 0.150 s | Hold the completed step before another leg can lift |

After four steps, each foot has moved forward by one complete stride while it
was airborne and rearward by four quarter-stride pushes while planted. The
joint targets are therefore periodic without a large whole-body reset.

## Geometry and inverse kinematics

Each sagittal leg uses a two-link analytic inverse-kinematics model. The
proximal servo-axis spacing is `L1 = 0.159896689 m`. The new rectangular PLA
face is `0.030 m` beyond the distal fork axis and its recommended bonded tread
adds `0.001 m`. The effective knee-to-contact length is therefore `L2 =
0.159896689 + 0.031 = 0.190896689 m`.

For a planted flat sole, the shoe normal follows the distal-link axis and must
point down. V9 therefore uses:

```text
hip   = asin(x / L1)
knee  = -hip
d     = sqrt(L1^2 - x^2) + L2
pitch = hip + knee = 0
```

The selected swing leg uses the general two-link solution while airborne:

```text
r^2 = d^2 + x^2
knee = acos((r^2 - L1^2 - L2^2) / (2 L1 L2))
hip  = atan2(x, d) - atan2(L2 sin(knee), L1 + L2 cos(knee))
```

At the start of lift and the end of lower, the general solution meets the
flat-sole branch without a joint-angle jump. The support branch is mirrored so
the distal links bend away from the chassis. The
final calibrated motor convention is deliberately explicit:

- front knees command negative values to move downward;
- rear knees command positive values to move downward; and
- the left/right signs for hip abduction continue to come from each tracked
  servo profile.

This fixes the earlier failure where the front and rear legs folded toward one
another. The gait start is the computed phase-zero pose, not the old hardcoded
45-degree stance. With only two sagittal pitch joints, foot X, foot Z, and sole
pitch cannot all be independently selected. V9 reserves exact zero pitch for
every planted shoe and permits only the unloaded swing shoe to deviate. The
selected 25 mm contact-centre lift and 60 mm stride reduce the unloaded swing
excursion relative to V8. The three support legs remain on the exact flat-sole
branch throughout the move.

The V7 fixed-X support extension is removed. It added downward reach but made
the three loaded soles tip by design. V9 instead recomputes the exact flat
support depth for every current fore/aft target. Zero abduction and zero
lateral transfer keep the rectangular contact planes level in 3D.

## Selected parameters

| Parameter | Tracked value |
| --- | ---: |
| Hardware cycle count | Continuous until STOP + STABLE HOLD |
| Period per cycle | 12.000 s |
| Total command duration | Unbounded until explicitly stopped |
| Finite fallback cycle count | 2 |
| Foot stride | 0.060 m |
| Foot contact-centre lift | 0.025 m |
| Three-support-leg extension | 0.000 m |
| Fore/aft weight transfer | 0.006 m |
| Lateral weight transfer | 0.000 m |
| Nominal downward reach | 0.329341447 m |
| Nominal front/rear foot separation from each hip | 0.080 m |
| Additional hip-abduction stance angle | 0 degrees |

The V8 hardware trial used a 96 mm stride and 4-second cycle. Live telemetry
showed roughly 20-39 degrees of tracking error on moving joints, while voltage,
current, and temperature did not show a matching electrical fault. V9 therefore
reduces stride to 60 mm, lift to 25 mm, and the crawl command ramp to 60
degrees/s. It triples the cycle time, lengthens firm-plant and settle phases,
and advances its trajectory by one fixed 50 ms tick per controller update so a
slow telemetry read cannot cause a catch-up burst. Only one leg is selected as
airborne; the next leg cannot lift until the prior leg completes lowering,
plant, weight-return, push, and settle. The 80 mm stance offset and exact
329.341447 mm stance depth remain unchanged.

## Actuator and controller settings

The four tracked servo profiles now use the same settings:

| Setting | Value |
| --- | ---: |
| Torque limit | 900 / 1000 (90%) |
| Speed register | 3400 |
| Acceleration register | 254 |
| Manual/RL dashboard ramp ceiling | 270 degrees/s |
| Hardcoded crawl ramp ceiling | 60 degrees/s |
| Maximum command step | 15 degrees |

`3400` is the maximum ST3215 speed setting exposed by the current software and
`254` is the maximum acceleration byte. The manual/RL ceiling equals the URDF's
documented no-load joint-speed limit of `4.712389 rad/s` (270 degrees/s), while
the hardcoded crawl applies its lower 60 degrees/s software ramp. These are
response limits, not a promise that a loaded joint will reach no-load speed.
The tracked torque setting remains the requested 90%; it is not raised to
continuous stall torque.

Vendor references:

- [Waveshare ST3215 documentation](https://www.waveshare.com/wiki/ST3215_Servo)
- [ST3215-C047 datasheet](https://files.seeedstudio.com/products/Feetech/108090003_FEETECH_ST-3215-C047-Datasheet.pdf)

## Isaac evidence

The report below is the retained `24 mm` V8 baseline. The active V9 60 mm,
12-second hardware profile has not been simulated or hardware-tested. The
baseline ran two 8-second cycles
with 120 Hz physics, 60 Hz control, and a `2.6477955 Nm` per-joint cap
representing 90% of the documented stall value. Isaac disabled the legacy
fork-tip spheres and attached a `100 x 60 x 6 mm` PLA box plus `94 x 54 x 1 mm`
tread box to each distal link.

| Metric | V8 24 mm baseline |
| --- | ---: |
| Strict result | **FAIL** |
| Forward displacement | 53.3148 mm |
| Lateral drift | 0.3293 mm |
| Minimum base height | 0.373818 m |
| Maximum body tilt | 1.8437 degrees |
| Peak / RMS joint error | 0.223851 / 0.041560 rad |
| Maximum joint speed | 2.26908 rad/s |
| Expected support contact | 87.560% |
| Maximum loaded-foot slip | 5.3432 mm |
| Contact-verified placements | rear right, rear left |

The robot stayed upright and progressed with low tilt, drift, and slip. It is
still a strict failure because the peak transient tracking error exceeds the
`0.15 rad` gate and the rigid coplanar contact model loads only a diagonal pair
strongly enough for both front placements to pass the per-foot force-duration
gate. Broad rigid feet make four-point load distribution numerically
indeterminate; this does not prove that a physical front shoe is loaded.

Report and screenshot:

- [`../validation/isaac-rectangular-flat-crawl-v8.json`](../validation/isaac-rectangular-flat-crawl-v8.json)
- [`../validation/isaac-rectangular-flat-crawl-v8.png`](../validation/isaac-rectangular-flat-crawl-v8.png)

Reproduce the retained 24 mm baseline from the repository root:

```powershell
& C:\isaacsim\python.bat simulation\isaac\run_crawl.py `
  --usd simulation\exports\isaac\quadruped_robot_floating.usdc `
  --headless --gait-mode distributed-push --cycles 2 `
  --period 8 --stride 0.024 --lift 0.035 --support-extension 0 `
  --weight-shift-forward 0.006 --weight-shift-lateral 0 `
  --stance-down 0.329341447 --stance-fore-aft 0.080 `
  --abduction-deg 0 --effort-limit-nm 2.6477955 --start-z 0.455 `
  --min-forward-displacement 0.010 `
  --report hardware\test-apps\four-leg-dashboard\validation\isaac-rectangular-flat-crawl-v8.json `
  --screenshot hardware\test-apps\four-leg-dashboard\validation\isaac-rectangular-flat-crawl-v8.png
```

The proxy omits the CAD-estimated `70.237 g` mass of each shoe, adhesive
compliance, two-millimeter corner radii, structural flex, servo backlash, and
measured tread friction. The older V5/TPU evidence below remains historical.

Two retained V8 tuning comparisons document the older 35 mm lift and 8-second
period; they are not validation of the active V9 profile:

- [`../validation/isaac-rectangular-flat-crawl-v8-12s.json`](../validation/isaac-rectangular-flat-crawl-v8-12s.json)
  used 48 mm lift and a 12-second period; tracking improved only to `0.236925
  rad`, while the same two front placement gates remained incomplete.
- [`../validation/isaac-rectangular-flat-crawl-v8-35mm.json`](../validation/isaac-rectangular-flat-crawl-v8-35mm.json)
  used one 8-second cycle at 35 mm lift; peak error fell to `0.223851 rad`,
  maximum slip to `5.343 mm`, and measured long-edge clearance remained above
  `25.57 mm`.

### TPU-shoe quick check

One explicitly requested quick check used the superseded 112 mm gait for one cycle with
the peak/stall torque cap. Isaac disabled the former 12.5 mm fork-tip spheres
and attached rigid capsule proxies to the distal links. Each capsule preserves
the shoe's `54 x 48 x 48 mm` outer envelope and `60.5 mm` fork-axis-to-nose
reach.

| Metric | Current TPU-shoe quick check |
| --- | ---: |
| Result | **FAIL** |
| Forward displacement | 115.61 mm |
| Maximum body tilt | 3.83 degrees |
| Minimum base height | 0.390 m |
| Maximum joint error | 0.481 rad |
| Maximum joint speed | 3.155 rad/s |
| Expected support contact | 91.3% |
| Maximum support-tip slip | 34.54 mm |
| Completed contact-verified steps | RR, FR, RL |

The robot stayed upright and progressed, but tracking error exceeded `0.15
rad`, slip exceeded `15 mm`, and front-left did not complete the final
contact-duration gate. This is stable preliminary evidence, not a passing
walking result. The proxy omits approximately 53.5 g provisional shoe mass per
leg, TPU deformation, vents, tread/friction measurements, and contact-point
migration over the rocker surface.

Report and screenshot:

- [`../validation/isaac-tpu-shoe-crawl-quick.json`](../validation/isaac-tpu-shoe-crawl-quick.json)
- [`../validation/isaac-tpu-shoe-crawl-quick.png`](../validation/isaac-tpu-shoe-crawl-quick.png)

### Historical flexible-shoe comparison

For the superseded TPU-shoe V5 comparison, Isaac bound a separate compliant
contact material to the rounded shoe proxies.
The selected provisional values are static friction `1.05`, dynamic friction
`0.85`, restitution `0.03`, contact stiffness `8000 N/m`, and contact damping
`45 N s/m`. This approximates energy storage and loss at the sole/ground
interface without attempting to deform the TPU mesh.

| Profile | Outcome | Forward | Max tilt | Max error | Max slip | Verified steps |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 56 mm stride, 60 mm lift, 6.67 s, 16 mm forward shift | Upright, FAIL | 78.95 mm | 5.39 deg | 0.427 rad | 15.75 mm | RR, FR, RL |
| 40 mm stride, 60 mm lift, 12 s, soft material | Collapsed | 540.30 mm | 102.40 deg | 0.642 rad | 39.50 mm | none |
| 40 mm stride, 60 mm lift, 12 s, heavily damped | Collapsed backward | -529.70 mm | 102.80 deg | 0.354 rad | 11.50 mm | none |
| 32 mm stride, 35 mm lift, 8 s, 8 mm lateral shift | Collapsed | 514.90 mm | 100.70 deg | 0.325 rad | 52.70 mm | RL |
| **40 mm stride, 40 mm lift, 8 s, 12 mm forward shift** | **Upright, FAIL; selected** | **50.60 mm** | **7.10 deg** | **0.296 rad** | **11.10 mm** | **RR, FR, FL** |

The selected profile was the least unstable screened TPU candidate. It passed the
upright, displacement, and slip checks but remains a strict failure because
maximum tracking error exceeded the `0.15 rad` limit and rear-left held the
required three-foot support state for only `0.542` of its phase. These
comparisons used Isaac's short-duration peak/stall torque cap; the physical
servo profiles retain their configured 90% torque limit. The physical
controller no longer uses this profile; the rectangular flat-shoe gaits
supersede it. The result remains historical evidence, not evidence of closed-loop
balance.

Selected report and screenshot:

- [`../validation/isaac-tpu-flex-crawl-40mm-8s-moderate.json`](../validation/isaac-tpu-flex-crawl-40mm-8s-moderate.json)
- [`../validation/isaac-tpu-flex-crawl-40mm-8s-moderate.png`](../validation/isaac-tpu-flex-crawl-40mm-8s-moderate.png)

The compliant-contact model is intentionally limited. It omits the shoe's
mass, vent and wall deformation, non-linear hysteresis, lateral buckling,
distributed tread contact, and rocker contact-point migration. Measured TPU
force/deflection and friction data should replace these provisional values
when available.

## Physical trial

Isaac can verify internal target consistency and screen for obvious collapse;
it does not verify the real foot friction, backlash, supply voltage drop,
wiring, servo heat, or exact mass distribution. The 24 mm V8 baseline is
simulated; the active 60 mm, 12-second V9 profile has not been simulated or
hardware-tested.
For its first requested trial:

1. Restart the dashboard so it reloads the TOML profiles and Python gait code.
2. With the body supported, click **SET GAIT START STANCE** and verify front
   knees bend forward/down, rear knees bend rearward/down, and all four shoe
   faces are parallel to the floor.
3. Keep a tether or support ready, place the feet on a high-friction surface,
   and click **TEST DISTRIBUTED CRAWL** once. It repeats until stopped.
4. Use **STOP + STABLE HOLD** for an orderly return to four-foot support. Use
   **DISARM ALL 12** or the physical cutoff immediately for a fall tendency,
   collision, cable pull, severe voltage sag, unexpected heat, noise, or the
   wrong joint direction.
5. Record which phase and leg first disagrees with the intended motion,
   especially whether a support shoe tips or the swing shoe lacks clearance.
   Change one gait parameter at a time after that observation.

This remains open-loop choreography. It has no IMU balance correction, foot
contact feedback, or closed-loop center-of-pressure control, so maximum servo
speed alone cannot make it actively balance after a disturbance.
