# VL53L5CX stair perception PPO v7

## Outcome

V7 replaces the v6 policy's eight perfect simulator terrain-height samples
with a hardware-shaped `8 x 8` time-of-flight observation. The policy now uses
IMU/proprioception plus 24 compressed VL53L5CX depth values; RGB camera pixels
are not policy inputs. All treads remain exactly `250 mm` deep, the rise remains
`180 mm`, and the measured one-leg hardware profile remains active, including
the `0.8825985 N m` effort cap.

The sensor pipeline, policy transfer, 512-step PPO smoke training, deterministic
reload, validation heatmap, and H.264 recording passed. The locomotion objective
did not: the recorded four-step attempt advanced `48.97 mm`, lifted the front
left foot `39.03 mm`, reached no stair, and ended at the seven-second
no-forward-progress gate. This is a runnable evaluation package, not a
converged or deployable stair policy.

Review the recording and depth frame on the private Sites page:
https://drobot-stair-sensor-eval.romka.chatgpt.site.

## Sensor and observation contract

The simulated device matches the full-resolution operating mode of an ST
VL53L5CX on Pololu carrier 3417:

| Setting | V7 value |
| --- | --- |
| Grid | `8 x 8` / 64 closest-hit PhysX rays |
| Detection volume | `45 deg` horizontal x `45 deg` vertical (`65 deg` diagonal) |
| Optical origin from base | `[0.1145, 0.0, 0.123] m` |
| Downward pitch | `40 deg` |
| Range | `0.02-4.0 m`; normalized/clipped at `1.5 m` |
| Sample rate | `15 Hz`, held for eight `120 Hz` control frames |
| Latency | one sensor frame / `66.7 ms` |
| Modeled error | bounded `+/-15 mm` through `0.20 m`, then bounded `+/-5%` |
| Modeled dropout | `5%` per zone; explicit simulation assumption |

Rows run from the least-downward to the most-downward ray; columns run left to
right. Each row is median-compressed into left columns `[0,1,2]`, center
columns `[3,4]`, and right columns `[5,6,7]`. The resulting 24 normalized raw
depth values replace the analytic terrain profile. The complete observation is
84 values: walking/IMU/proprioception `48`, ToF depth `24`, goal `1`, navigation
`3`, foot progress `4`, and next-foot target `4`.

The remaining goal, navigation, and foot-progress values are still
simulator-assisted. A hardware deployment must estimate them onboard or train
a later policy without them. V7 removes the largest terrain-geometry shortcut,
but does not yet make the entire observation hardware complete.

## Cheap real prototype

The Pololu carrier was listed at `$19.95` when reviewed on 2026-07-31. It is
approximately `13 x 18 x 3 mm`, `0.5 g` without headers, accepts `2.5-5.5 V`,
and exposes the required `VIN`, `GND`, `SCL`, and `SDA` connections. Pololu
publishes `100 mA` typical and `150 mA` peak current. The board includes
regulation and I2C level shifting, so it is a simpler first Raspberry Pi bench
prototype than a USB stereo depth camera.

Full `8 x 8` mode is limited to `15 Hz`; the `60 Hz` headline applies to
`4 x 4`. Use the ST Linux driver/ULD on the Raspberry Pi, confirm I2C address
and bus behavior with the other devices, and bench-test dark material,
sunlight, invalid zones, actual latency, and front-leg occlusion. The simulated
origin is provisional: design and inspect a real `40 deg` adapter before
drilling or printing a final mount.

References:

- [Pololu carrier product/specifications](https://www.pololu.com/product/3417)
- [ST VL53L5CX product and datasheet](https://www.st.com/en/imaging-and-photonics-solutions/vl53l5cx.html)
- [ST Linux driver](https://www.st.com/en/embedded-software/stsw-img025.html)
- [ST ULD integration manual](https://www.st.com/resource/en/user_manual/um2887-a-software-integration-guide-to-implement-the-ultra-light-driver-of-the-vl53l5cx-timeofflight-8-x-8-multizone-ranging-sensor-with-wide-field-of-view-stmicroelectronics.pdf)

## Source of truth

- `quadruped_stairs_v7_vl53l5cx.yaml` owns the fixed stair geometry, measured
  hardware profile, sensor extrinsics/rate/noise, reward, and PPO settings.
- `_vl53l5cx_contract.py` owns pure ray geometry, config validation, bounded
  noise/dropout, lane compression, and the 24 field names.
- `_vl53l5cx_sensor.py` owns PhysX closest-hit sampling, body-to-world ray
  rotation, cadence, sample-and-hold behavior, and frame latency.
- `_stair_rl_contract.py` supports either the legacy analytic profile or an
  explicitly supplied hardware observation without changing v1-v6 defaults.
- `_quadruped_stairs_env.py` selects the perception mode and exposes static
  sensor/model provenance in checkpoint manifests.
- `train_stairs_v7_vl53l5cx_ppo.py` selects the v7 config and output defaults.
- `validate_vl53l5cx_stairs.py` verifies live Isaac hits, cadence, latency,
  observation shape, and writes the reviewed depth heatmap/report.

Editable YAML and Python are authoritative. Models, JSON reports, heatmaps,
screenshots, and MP4 files are generated evidence.

## Reproduce

Run pure contracts:

```powershell
& .\.venv\Scripts\python.exe -m pytest `
  --basetemp=.pytest-tmp-v7-sensor `
  tests\test_vl53l5cx_stair_perception.py `
  tests\test_quadruped_stairs_rl_contract.py
```

Validate live PhysX sensing:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\validate_vl53l5cx_stairs.py `
  --report reviews\vl53l5cx-stairs-validation.json `
  --heatmap reviews\vl53l5cx-stairs-depth.png
```

Run the same bounded training smoke from the v6 policy:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\train_stairs_v7_vl53l5cx_ppo.py `
  --smoke-test `
  --initialize-from-stairs `
  simulation\isaac\models\ppo-stairs-v6-180mm-25cm-small\drobot_stairs_ppo_initialized.zip `
  --output-dir `
  simulation\isaac\models\ppo-stairs-v7-vl53l5cx-180mm-small
```

Evaluate and record all four `180 x 250 mm` steps:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v7_vl53l5cx.yaml `
  --height-stage 180mm `
  --model simulation\isaac\models\ppo-stairs-v7-vl53l5cx-180mm-small\drobot_stairs_ppo_final.zip `
  --episodes 1 --active-steps 4 `
  --report simulation\isaac\models\ppo-stairs-v7-vl53l5cx-180mm-small\evaluation_report.json

& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\record_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v7_vl53l5cx.yaml `
  --height-stage 180mm `
  --model simulation\isaac\models\ppo-stairs-v7-vl53l5cx-180mm-small\drobot_stairs_ppo_final.zip `
  --seed 147 --active-steps 4 --camera-view external `
  --video reviews\ppo-stairs-v7-vl53l5cx-evaluation.mp4 `
  --thumbnail reviews\ppo-stairs-v7-vl53l5cx-evaluation.png `
  --report simulation\isaac\models\ppo-stairs-v7-vl53l5cx-180mm-small\recording_report.json
```

## Validation actually run

- focused pure contracts: `26 passed, 2 skipped`;
- Ruff on the changed stair/sensor sources and tests: passed;
- Python compilation for every changed stair/sensor runner: passed;
- broader repository sweep (with the collection-only hip preview excluded):
  `191 passed, 2 skipped`; five unrelated CAD assembly tests failed only
  because the optional `cadpy` package is absent from this venv;
- policy expansion: `68 -> 84` inputs, two input matrices expanded, no skipped
  parameters, new columns initialized to zero;
- PPO smoke: `512` steps, report status `PASS`, final sensor count `65`, all
  `64` zones valid in the final training sample;
- live sensor validation: status `PASS`, delivery changes at control frames
  `8`, `16`, and `24`, with `32` rays on stair layer one and `32` on layer two;
- deterministic four-step evaluation: contract status `PASS`, `0/1` success,
  no stair reached;
- recording: status `PASS`, `210` H.264 frames, `960 x 540`, `30 FPS`, `7.0 s`;
- private Sites deployment version 1: succeeded.

Generated evidence:

- `simulation/isaac/models/ppo-stairs-v7-vl53l5cx-180mm-small/`
- `reviews/vl53l5cx-stairs-validation.json`
- `reviews/vl53l5cx-stairs-depth.png`
- `reviews/ppo-stairs-v7-vl53l5cx-evaluation.mp4`
- `reviews/ppo-stairs-v7-vl53l5cx-evaluation.png`

## Limitations and next decision

This run is deliberately too small to establish learning. More importantly,
the earlier rated-torque feasibility gate failed at `180 mm`; adding a sensor
does not change actuator margin, leg reach under load, support transfer, or
collision. The recorded result agrees with that warning.

Do not deploy this checkpoint. First bench-test the real VL53L5CX stream and
finalize its mount. In parallel, resolve the mechanical/actuator feasibility
gate and add foot load/contact sensing. Only then is a longer height curriculum
worth the compute, with sensor extrinsic/friction/mass randomization and a
deterministic unseen-stair acceptance threshold.
