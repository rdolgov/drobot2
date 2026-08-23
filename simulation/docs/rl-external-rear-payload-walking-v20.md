# V20 external rear-payload smooth walking

Date: 2026-08-23

V20 corrects the V19 battery placement after review of the training video and
the physical robot. V19 placed the battery enclosure inside the chassis at the
rear-most compatible floor-grid position. The real enclosure is attached to
the outside of the rear plate and centered on that plate. V20 models that
external geometry, retrains from V19, and retains the smoothest acceptable
adaptation checkpoint.

## Corrected hardware transform

The robot coordinate convention is `+X` forward, `+Y` left, and `+Z` up. The
body is approximately 220 x 170 x 100 mm, so its rear face is at `X=-110 mm`
and the rear-plate center is `(X, Y, Z)=(-110, 0, 50) mm`.

The CM5202 box CAD source defines a 144 x 68 x 40 mm main enclosure plus a
3 mm lid. Mounting it on the rear plate requires rotating the holder so:

- its 43 mm box-and-lid depth spans global X and projects rearward;
- its 144 mm length spans global Y, centered left-to-right;
- its 68 mm tab width spans global Z, centered vertically.

The resulting simulation-only collision and inertia proxy is:

| Quantity | Value |
| --- | --- |
| Center | `(-131.5, 0, 50.0) mm` |
| Global size (X, Y, Z) | `(43, 144, 68) mm` |
| X extent | `-153.0` to `-110.0 mm` |
| Y extent | `-72.0` to `+72.0 mm` |
| Z extent | `16.0` to `84.0 mm` |

This puts the box against the outside rear face without embedding it in the
chassis. The exact physical adapter thickness and fastener locations were not
measured, so the robustness model allows additional fore/aft and vertical COM
uncertainty. Measure the installed face-to-box gap and box height before a
future high-fidelity revision.

The V19 mass calculation remains valid:

| Item | Mass |
| --- | ---: |
| Measured CM5202 battery | 416.000 g |
| Box, from STL volume at 1.24 g/cm3 PLA | 80.196 g |
| Lid, from STL volume at 1.24 g/cm3 PLA | 26.984 g |
| Nominal modeled assembly | 523.180 g |

The prior provisional 450 g internal battery is removed before the external
assembly is added. Nominal combined base mass is 2.122299 kg and the estimated
combined base COM becomes `(-32.417, 0, 52.120) mm`. Combined mass remains
randomized by 0.965--1.040, approximately covering a 450--600 g assembly, with
combined-COM jitter of `(+/-4.5, +/-2.0, +/-3.0) mm`.

After V20 training, the fully assembled physical robot was measured at 7 lb,
or approximately **3.18 kg**, including the installed rear battery and its
enclosure. This whole-robot measurement is the preferred mass reference for
the next simulation calibration. It does not retroactively describe the exact
mass distribution used during V20 training; individual link inertias and the
installed center of mass still need measurement or reconciliation.

## Objective and conservative continuation

V20 inherits V19's low-speed 0.04--0.10 m/s commands, 0.8-second diagonal gait
clock, all-four-leg contact metrics, sustained-stall cost, and acceleration,
action-difference, slip, impact, posture, and heading terms. It strengthens the
terms most relevant to the external rear lever arm:

- lateral velocity penalty: 10.0;
- lateral displacement penalty: 20.0;
- yaw-rate penalty: 8.0;
- action second-difference penalty: 0.75;
- joint acceleration penalty: 0.25;
- body linear acceleration penalty: 0.40;
- body angular acceleration penalty: 0.30.

The accepted continuation uses a 5e-5 PPO learning rate, zero entropy bonus,
and a 0.005 desired KL. This deliberately protects the already-smooth V19 gait
while adapting to the corrected rearward inertia.

## Training and checkpoint selection

Both runs used 128 parallel robots and started from V19 `model_899.pt`.

| Run | Seed | Range | Outcome |
| --- | ---: | ---: | --- |
| `drobot_commanded_walk_v20_external_rear_payload` | 2001 | 899--1200 | Rejected pilot: normal 3e-4 learning rate recovered speed but learned a large lateral arc and increasing jerk. |
| `drobot_commanded_walk_v20_external_rear_payload_straight` | 2002 | 899--1048 | Conservative low-rate continuation; checkpoint 900 selected and later checkpoints rejected. |

Deterministic 0.05 m/s trials used the corrected external-payload task. Values
below are means over three uninterrupted 30-second episodes; every row had
zero falls, zero five-second stall windows, and non-zero touchdowns from every
leg.

| Checkpoint | Actual speed | Lateral drift | Joint accel RMS | Body linear accel RMS | Body angular accel RMS | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| V19 899, unadapted baseline | 0.0294 m/s | 0.144 m | 13.973 rad/s2 | 0.427 m/s2 | 4.133 rad/s2 | Stable and smooth, but under-walks. |
| Pilot 950 | 0.0482 m/s | 0.454 m | 15.377 rad/s2 | 0.526 m/s2 | 4.331 rad/s2 | Rejected: lateral arc. |
| Pilot 1050 | 0.0466 m/s | 0.457 m | 15.656 rad/s2 | 0.546 m/s2 | 4.160 rad/s2 | Rejected: drift and jerk. |
| Pilot 1125 | 0.0478 m/s | 0.554 m | 16.660 rad/s2 | 0.559 m/s2 | 4.500 rad/s2 | Rejected: worst drift and jerk. |
| **Conservative 900** | **0.0358 m/s** | **0.132 m** | **14.388 rad/s2** | **0.438 m/s2** | **4.253 rad/s2** | **Selected: improves progress and drift with a small acceleration increase.** |
| Conservative 925 | 0.0557 m/s | 0.336 m | 15.421 rad/s2 | 0.495 m/s2 | 4.340 rad/s2 | Rejected: speed came with drift and jerk. |

At a separate 0.08 m/s command, selected checkpoint 900 completed 30 seconds
without a fall or stall and used all four legs. It averaged 0.0524 m/s actual
speed, 0.103 m lateral drift, 14.872 rad/s2 joint RMS acceleration, 0.450 m/s2
body linear RMS acceleration, and 4.495 rad/s2 body angular RMS acceleration.

Checkpoint 900 is intentionally only a very small policy update. More training
was not automatically considered better: the measured checkpoint comparisons
showed that later updates optimized speed by sacrificing the user's higher
priority of smooth, straight motion.

## Visual review

The 20-second review clip uses the rear-left camera. The dark red battery proxy
is visibly outside the chassis, centered on the rear plate, and travels as a
rigid part of the base. The checked frame at eight seconds confirms the proxy
does not occupy the internal body volume.

## Artifacts

- Isaac checkpoint:
  `simulation/isaac/models/parallel-walking-v20-external-rear-payload/model_900.pt`
- Raspberry Pi model:
  `onboard/models/parallel-walking-v20-external-rear-payload/model_900.onnx`
- Model metadata and hashes:
  `onboard/models/parallel-walking-v20-external-rear-payload/model_900.json`
- 20-second 0.05 m/s review clip:
  `simulation/reviews/parallel-walking-v20-external-rear-payload-model900-20s.mp4`
- Checked review frame:
  `simulation/reviews/v20-external-rear-payload-frame.png`

## Reproduce

Continue conservatively from the bundled V19 checkpoint:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet external-rear-payload `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v19-smooth-rear-payload\model_899.pt `
  -Iterations 150 -NumEnvs 128 -Seed 2002
```

Evaluate the selected policy:

```powershell
& .\simulation\isaac\rl\parallel_walking\evaluate_walking_sustained.ps1 `
  -CommandSet external-rear-payload `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v20-external-rear-payload\model_900.pt `
  -ForwardSpeed 0.05 -Seconds 30 -Episodes 3
```

Record the external-payload review angle:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -CommandSet external-rear-payload `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v20-external-rear-payload\model_900.pt `
  -Command forward -ForwardSpeed 0.05 -NoTimeLimit -RecordSeconds 20
```

This is simulator validation, not a hardware safety guarantee. Begin physical
tests with the robot supported, low current limits, and short durations. Use
the existing stable-pose stop behavior and real-walk recorder before increasing
speed or duration.
