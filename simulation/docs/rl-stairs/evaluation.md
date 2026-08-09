# Evaluating the stair policy

## Current evidence

As of 2026-07-27, deterministic model reload, manifest verification, two
level-1 evaluation episodes, loaded PPO algorithm verification, screenshot
capture, and one H.264 recording have passed as pipeline checks. The corrected
512-step smoke checkpoint achieved `0/2` evaluation success, `0/1` recording
success, and never reached the first stair. The v1 full run was stopped at
964,608 steps after its 950k checkpoint also reached step `0` in all five
level-1 audit episodes. Do not substitute either the successful flat-walking
result or the successful smoke pipeline for stair-climbing evidence. The
replacement experiment is documented under
[`docs/rl-stairs-v2/`](../rl-stairs-v2/README.md).

Evaluate a stair checkpoint only after:

1. generating the stair world;
2. completing the pipeline smoke test;
3. training a separate 57-input stair model;
4. retaining the model's adjacent `.contract.json`.

All commands below run from the repository root.

## Recorded level-1 smoke evaluation — 2026-07-27

The tested command was:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --model simulation\isaac\output\rl\ppo-stairs-v1-smoke\drobot_stairs_ppo_final.zip `
  --episodes 2 `
  --active-steps 1 `
  --screenshot reviews\ppo-stairs-v1-smoke.png `
  --report simulation\isaac\output\rl\ppo-stairs-v1-smoke\evaluation_report.json
```

The evaluator, schema-2 manifest verification, and direct loaded-PPO algorithm
verification all reported PASS. The behavioral result did not:

| Metric | Result |
| --- | ---: |
| Seed | `143` |
| Stair success | `0/2` |
| Mean highest step | `0.0` |
| Episode lengths | `188`, `114` steps |
| Mean forward displacement | `0.322017 m` |
| Mean elevation gain | `-0.137918 m` |
| Mean return | `162.266386` |
| Worst minimum base clearance | `0.219377 m` |
| Worst maximum body tilt | `47.457092 degrees` |
| Screenshot bytes | `449,249` |

Both episodes ended with `body_tipped`, after 188 steps (3.133 seconds) and
114 steps (1.900 seconds). This proves deterministic checkpoint loading,
57-input inference, two physics updates per control action, schema-2
dependency/algorithm checking, episode reporting, and screenshot capture. It
is explicit evidence that the smoke model is not a stair policy.

## Staged deterministic evaluation

The evaluator uses `model.predict(..., deterministic=True)`. `--active-steps`
pins the requested curriculum goal instead of using training progress. Start
with each level separately so a full-stair failure is diagnosable.

### Level 1

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --model simulation\isaac\output\rl\ppo-stairs-v1\drobot_stairs_ppo_final.zip `
  --active-steps 1 `
  --episodes 10 `
  --seed 143 `
  --report simulation\isaac\output\rl\ppo-stairs-v1\evaluation-level-1.json
```

Repeat with levels 2 and 3:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --model simulation\isaac\output\rl\ppo-stairs-v1\drobot_stairs_ppo_final.zip `
  --active-steps 2 `
  --episodes 10 `
  --seed 143 `
  --report simulation\isaac\output\rl\ppo-stairs-v1\evaluation-level-2.json

& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --model simulation\isaac\output\rl\ppo-stairs-v1\drobot_stairs_ppo_final.zip `
  --active-steps 3 `
  --episodes 10 `
  --seed 143 `
  --report simulation\isaac\output\rl\ppo-stairs-v1\evaluation-level-3.json
```

### Full four-step evaluation

Use more episodes for the acceptance run:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --model simulation\isaac\output\rl\ppo-stairs-v1\drobot_stairs_ppo_final.zip `
  --active-steps 4 `
  --episodes 20 `
  --seed 143 `
  --report simulation\isaac\output\rl\ppo-stairs-v1\evaluation-report-20ep.json
```

The default `--active-steps` is all four. The explicit argument above makes
the review record unambiguous. Run additional seed streams rather than
reporting one favorable seed:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --model simulation\isaac\output\rl\ppo-stairs-v1\drobot_stairs_ppo_final.zip `
  --active-steps 4 `
  --episodes 20 `
  --seed 1143 `
  --report simulation\isaac\output\rl\ppo-stairs-v1\evaluation-report-seed-1143.json
```

Evaluation refuses a missing or mismatched manifest. The
`--allow-unverified-model` option should be reserved for documented recovery;
it weakens provenance and is not acceptable for a policy-comparison result.
With a verified manifest, the evaluator also compares the loaded PPO object
against the saved algorithm contract; a renamed ZIP with incompatible
architecture or hyperparameters does not pass.

## Visual inspection and screenshot

Watch one deterministic episode with the fixed external view and save a PNG:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --model simulation\isaac\output\rl\ppo-stairs-v1\drobot_stairs_ppo_final.zip `
  --active-steps 4 `
  --episodes 1 `
  --seed 143 `
  --gui `
  --camera-view external `
  --screenshot reviews\ppo-stairs-v1-evaluation.png `
  --report simulation\isaac\output\rl\ppo-stairs-v1\evaluation-visual.json
```

Use `--camera-view onboard` to inspect the mounted camera. Camera images are
for review only; the version-1 policy does not consume pixels.

During visual review, look for:

- body or legs colliding with riser faces;
- feet slipping sideways or walking near the stair edge;
- a ballistic hop that satisfies the goal but is not repeatable climbing;
- oscillation while holding the top-platform goal;
- foot tunneling, interpenetration, or visibly unstable contact;
- the base clearing each local tread with useful margin;
- cables, covers, or printed geometry that the current collision model omits.

## Record one deterministic run

The recorder captures a single episode as H.264 MP4 using an offscreen RTX
camera. At the defaults, it writes `960 × 540` at 30 FPS. Because control runs
at 60 Hz, it records every second policy step. A timeout-length episode can be
up to 22 seconds and 660 frames.

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\record_stairs_ppo.py `
  --model simulation\isaac\output\rl\ppo-stairs-v1\drobot_stairs_ppo_final.zip `
  --active-steps 4 `
  --seed 143 `
  --camera-view external `
  --video reviews\ppo-stairs-v1-evaluation.mp4 `
  --thumbnail reviews\ppo-stairs-v1-evaluation.png `
  --report simulation\isaac\output\rl\ppo-stairs-v1\recording_report.json
```

Expected outputs:

- `reviews/ppo-stairs-v1-evaluation.mp4`;
- `reviews/ppo-stairs-v1-evaluation.png`;
- `simulation/isaac/output/rl/ppo-stairs-v1/recording_report.json`.

The report must say `"status": "PASS"` and record nonzero
`recorded_frames`, `video_bytes`, and `thumbnail_bytes`. It also stores the
episode metrics and model-contract verification. A playable video is review
evidence for one seeded episode, not a substitute for multi-episode metrics.

The requested FPS must divide the configured 60 Hz control rate exactly.

### Recorded level-1 smoke recording — 2026-07-27

The smoke checkpoint was recorded with this command:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\record_stairs_ppo.py `
  --model simulation\isaac\output\rl\ppo-stairs-v1-smoke\drobot_stairs_ppo_final.zip `
  --active-steps 1 `
  --video reviews\ppo-stairs-v1-smoke.mp4 `
  --thumbnail reviews\ppo-stairs-v1-smoke-recording.png `
  --report simulation\isaac\output\rl\ppo-stairs-v1-smoke\recording_report.json
```

The recorder, schema-2 model-manifest verification, and direct loaded-PPO
algorithm verification all reported PASS:

- seed `143`, active level `1`;
- 660 frames at `960 × 540`, 30 FPS, H.264 MP4;
- `reviews/ppo-stairs-v1-smoke.mp4`: 14,730,825 bytes;
- `reviews/ppo-stairs-v1-smoke-recording.png`: 174,296 bytes;
- `simulation/isaac/output/rl/ppo-stairs-v1-smoke/recording_report.json`
  preserves the runtime contract and episode metrics.

The episode timed out after 22.0 seconds, reached highest step `0`, and did not
complete the level. It moved `0.542857 m` forward while remaining on the
approach. The PASS status validates deterministic inference, dependency and
algorithm contract checking, offscreen RTX capture, thumbnail output, and
H.264 encoding. It does not indicate stair competence.

## Evaluation report fields

Each completed episode records:

| Field | Interpretation |
| --- | --- |
| `stairs_completed` | Held the configured goal for 0.5 seconds without failure |
| `active_step_count` | Requested curriculum level |
| `highest_step_reached` | Highest analytic tread index reached, 0–4 |
| `return` | Sum of shaped rewards; useful only with physical metrics |
| `length_steps`, `duration_s` | Time to success/failure or the 22-second truncation |
| `forward_displacement_m` | Final X minus reset X |
| `lateral_displacement_m` | Final Y minus reset Y |
| `elevation_gain_m` | Final base Z minus reset base Z |
| `final_terrain_height_m` | Analytic surface under the final base X |
| `minimum_base_clearance_m` | Worst local clearance during the episode |
| `maximum_body_tilt_deg` | Worst tilt from projected gravity |
| `goal_hold_duration_s` | Consecutive time held beyond the goal |
| `failure_reasons` | Exact low-clearance, tip, corridor, or backward causes |
| `terminated`, `truncated` | Whether an outcome ended the episode or time expired |

The aggregate report includes `success_count`, `success_rate`, failure reason
counts, mean return, mean highest step, mean forward displacement, mean
elevation gain, worst minimum clearance, and worst maximum tilt.

## How to interpret results

- `success_rate` is the primary completion metric. Report its numerator,
  denominator, seed, active-step level, and model hash.
- Mean `highest_step_reached` shows partial progress when success is still
  zero. Break it down by episode; a mean can hide bimodal behavior.
- Full-stair success should end on terrain height `0.16 m` and normally show
  positive elevation gain. A strange combination can reveal a geometry or
  reward exploit.
- Forward displacement alone is insufficient because the approach is over a
  meter long.
- High return with low success means shaping terms are masking task failure.
- Many 22-second truncations with no failure can mean stable stalling at a
  riser.
- `base_clearance_too_low` points to body/riser contact or collapse.
- `body_tipped` points to unstable posture or aggressive ascent.
- `left_stair_corridor` means `|world Y| > 0.48 m`; review whether inherited
  flat-policy drift is being transferred.
- `moved_too_far_backward` means world X fell below `-1.20 m`.
- Minimum clearance just above `0.20 m` and tilt just below `45.6 degrees`
  technically pass but provide little robustness margin.

Compare checkpoints using identical config, world, composed-dependency, and
PPO algorithm contracts, plus the same active level, episode count, and seed
streams. If any of those differ, treat the runs as different experiments.

## Suggested simulation acceptance gate

Before describing a checkpoint as a useful **simulated stair policy**, record:

- successful contract verification;
- multiple independent evaluation seed streams;
- high completion at levels 1, 2, 3, and 4;
- no non-finite state;
- low failure and timeout rates;
- consistent positive elevation onto the `0.16 m` platform;
- acceptable lateral displacement and no corridor exits;
- clearance and tilt comfortably inside termination thresholds;
- a visually reviewed external recording with plausible contacts;
- no reliance on a single unusually favorable checkpoint or seed.

No numeric success threshold has yet been approved for this experiment.
Choose and document one before final model selection rather than after seeing
the results.

## Sim-to-real assumptions and limitations

Passing the simulation gate does not authorize hardware stair tests.

- The policy receives an exact analytic terrain profile from simulator world
  X. Hardware has no equivalent sensor pipeline yet.
- The staircase is one fixed profile: four `40 mm` risers and `230 mm`
  treads. Stair dimensions, edge shape, approach angle, and landing are not
  randomized.
- Friction, foot compliance, link mass/inertia, backlash, battery voltage,
  torque/thermal derating, control latency, packet loss, and IMU/encoder noise
  are not randomized here.
- The reset always begins on flat ground in a narrow pose range.
- The observation has no foot-contact sensors and no image/depth features.
- The task teaches ascent only. It does not test descent, turning on stairs,
  damaged feet, payload changes, or interrupted recovery.
- The collision proxies and contact material approximate, but do not fully
  reproduce, printed hardware and household stairs.
- One direct Isaac environment is used rather than many randomized Isaac Lab
  clones, increasing wall-clock time and overfitting risk.

Before hardware, replace or reproduce the terrain inputs with a documented
perception/state-estimation contract, add relevant randomization, test
latency and actuator limits, and verify inference timing. First physical
motion should be suspended or guarded with a human-operated emergency stop.
Only after stable guarded flat-floor tests should the robot approach a low
single step with fall protection; a four-step staircase is a later safety
stage.
