# V19 smooth rear-payload walking

Date: 2026-08-22

V19 continues the selected V18 rectangular-shoe policy and prioritizes smooth,
low-speed walking with the newly installed rear battery assembly. The selected
checkpoint is `model_899.pt`; its deployable export is `model_899.onnx`.

## Hardware model

The flat rectangular shoes remain unchanged from V18: a 100 x 60 x 6 mm sole,
94 x 54 x 1 mm tread, and 70.237 g CAD mass estimate per leg.

The prior robot inertia ledger included a provisional 450 g battery centered
low in the chassis. V19 replaces it instead of adding a second battery:

| Item | Mass |
| --- | ---: |
| Measured CM5202 battery | 416.000 g |
| Box, from STL volume at 1.24 g/cm3 PLA | 80.196 g |
| Lid, from STL volume at 1.24 g/cm3 PLA | 26.984 g |
| Nominal modeled assembly | 523.180 g |

The box source gives a 144 x 68 x 40 mm outer envelope plus a 3 mm lid. The
installed transform was not measured, so the nominal payload center is the
rear-most compatible body-floor grid position, `(-39.104, 0, 24.521) mm` in
the base-link frame (`+X` forward, `+Y` left, `+Z` up). This produces a base
mass of 2.122299 kg and an approximate combined base COM of
`(-9.640, 0, 45.839) mm`.

The robustness phase varies combined base mass by 0.965--1.040, corresponding
approximately to 450--600 g of payload, and jitters combined COM by
`(+/-3, +/-2, +/-2) mm`. This covers enclosure hardware and placement
uncertainty without changing the deployable observation contract. Replace the
estimated transform after measuring the installed box relative to body center.

## Smoothness objective

The policy still receives a modest progress objective and a hard sustained
stall cost; otherwise a perfectly smooth policy could learn to stand still.
Training commands span 0.04--0.10 m/s and require at least 0.025 m/s sustained
progress. Speed is intentionally subordinate to these smoothness terms:

- normalized action first-difference penalty: 0.12;
- normalized action second-difference penalty: 0.65;
- joint velocity penalty: 0.030;
- joint acceleration penalty: 0.20, normalized at 40 rad/s2;
- base linear acceleration penalty: 0.30, normalized at 5 m/s2;
- base angular acceleration penalty: 0.25, normalized at 10 rad/s2;
- planted-foot slip penalty: 0.45;
- touchdown impact penalty: 0.12 above a 14 N soft threshold;
- roll/pitch, tilt, body-height, lateral, and yaw stabilization penalties.

The action-rate and acceleration costs ramp from 25% to 100% over 300 PPO
iterations. The diagonal gait clock, scheduled stance/swing rewards, qualified
touchdowns, and per-leg metrics continue to require coordinated use of all four
legs. The actor contract remains 50 observations, 12 actions, 60 Hz, and a
0.8-second gait period.

## Training record

Both phases used 128 parallel robots and continued V18 `model_299.pt`.

| Run | Seed | Iterations | Purpose |
| --- | ---: | ---: | --- |
| `2026-08-22_21-36-10_manual-headless` | 1901 | 299--609 | nominal 523.18 g payload; ramp smoothness to full |
| `2026-08-22_21-44-12_manual-headless` | 1902 | 600--899 | 450--600 g payload/COM robustness; full smoothness |

At the end of training, vectorized metrics remained at zero fall and zero
sustained-stall rate. All four qualified-touchdown counters were non-zero.
Typical final-window values were 18.2--19.3 rad/s2 joint RMS acceleration,
0.59--0.62 m/s2 body linear RMS acceleration, and 5.5--5.6 rad/s2 body angular
RMS acceleration.

## Deterministic checkpoint selection

Each candidate below ran three uninterrupted 30-second episodes with a
0.05 m/s command and randomized payload estimate. All candidates had zero falls,
zero five-second stall windows, and touchdowns from every leg.

| Checkpoint | Actual speed | Lateral drift | Joint accel RMS | Body linear accel RMS | Body angular accel RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 800 | 0.0524 m/s | 0.703 m | 15.831 rad/s2 | 0.529 m/s2 | 4.496 rad/s2 |
| 850 | 0.0540 m/s | 0.576 m | 15.587 rad/s2 | 0.567 m/s2 | 4.240 rad/s2 |
| **899** | **0.0457 m/s** | **0.204 m** | **14.653 rad/s2** | **0.484 m/s2** | **4.353 rad/s2** |

Checkpoint 899 was selected because it has the lowest joint and body-linear
accelerations by a clear margin and much less drift, while still walking for
the full duration. A separate 30-second 0.08 m/s check also completed without a
fall or stall: 0.0670 m/s actual speed, 0.190 m lateral drift, 14.904 rad/s2
joint RMS acceleration, and all four legs active.

## Artifacts

- Isaac checkpoint:
  `simulation/isaac/models/parallel-walking-v19-smooth-rear-payload/model_899.pt`
- Raspberry Pi model:
  `onboard/models/parallel-walking-v19-smooth-rear-payload/model_899.onnx`
- Model metadata and hashes:
  `onboard/models/parallel-walking-v19-smooth-rear-payload/model_899.json`
- 20-second 0.05 m/s review clip:
  `simulation/reviews/parallel-walking-v19-smooth-rear-payload-model899-20s.mp4`

## Reproduce

Continue or retrain the V19 task:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet smooth-payload -Iterations 300 -NumEnvs 128
```

Evaluate at a chosen low speed:

```powershell
& .\simulation\isaac\rl\parallel_walking\evaluate_walking_sustained.ps1 `
  -CommandSet smooth-payload `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v19-smooth-rear-payload\model_899.pt `
  -ForwardSpeed 0.05 -Seconds 30 -Episodes 3
```

Record a review clip:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -CommandSet smooth-payload `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v19-smooth-rear-payload\model_899.pt `
  -Command forward -ForwardSpeed 0.05 -NoTimeLimit -RecordSeconds 20
```

This is simulator validation, not a hardware safety guarantee. Begin physical
tests with the robot supported, low current limits, short duration, and the
existing stable-pose stop behavior. Use the real-walk recorder to compare IMU,
joint, voltage, temperature, and timing data before increasing duration.
