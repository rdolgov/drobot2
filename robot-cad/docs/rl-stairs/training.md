# Training the stair policy

## Validation status

As of 2026-07-27, pure-contract tests, static Isaac world generation, and a
corrected 512-step PPO smoke run initialized from the flat checkpoint have
passed. Schema-2 manifest verification and direct inspection of the loaded PPO
algorithm also passed. The smoke run validates the integration path but did
not learn stair climbing. The v1 full run was stopped at 964,608 steps after
its 950k checkpoint completed `0/5` first-step evaluation episodes and never
reached step index 1. Continue with the separate
[v2 training guide](../rl-stairs-v2/training.md), not this v1 run.

An earlier full launch was intentionally stopped before meaningful training
after review found that its `+50` success bonus could undervalue completion
relative to discounted stationary loitering. No result from that aborted
launch is treated as a current checkpoint. The corrected contract uses `+400`
at `gamma = 0.995` and was re-smoked before full training.

All commands assume PowerShell in the `robot-cad/` directory.

## 1. Install the Isaac RL dependencies

The repository setup script installs the pinned Gymnasium,
Stable-Baselines3, and TensorBoard versions into Isaac Sim's Python:

```powershell
.\scripts\setup_isaac_rl.ps1
```

Use an ordinary activated development environment for pure tests:

```powershell
pytest tests\test_quadruped_stairs_rl_contract.py
```

These tests exercise the stair math without launching Isaac. They do not
validate PhysX contacts, USD composition, PPO rollout, convergence, rendering,
or hardware.

### Recorded pure-contract validation — 2026-07-27

The focused test produced `......s...`: nine tests passed and the
Torch-dependent policy-transfer unit was skipped in the ordinary project
environment. The subsequent Isaac smoke run exercised that transfer path with
real Stable-Baselines3 tensors, copying 11 tensors exactly and expanding both
57-input first layers successfully. The skip therefore remains transparent
without being misreported as a pure-unit-test pass.

## 2. Generate the stair world

The generated stair world sublayers the validated manual world and adds four
static collision layers:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\create_stairs_world.py `
  --base-world exports\isaac\quadruped_robot_manual_world.usda `
  --output exports\isaac\quadruped_robot_stairs_world.usda `
  --report simulation\isaac\output\rl\ppo-stairs-v1\world_report.json
```

Do not proceed merely because the process exits. Open
`world_report.json` and require `"status": "PASS"`. A passing static report
should record four stair layers, four collision layers, zero stair rigid
bodies, one articulation root, 13 robot rigid bodies, 12 revolute joints, one
camera, one IMU, hashes for both worlds and the YAML, and the inherited
`/World/Materials/PrintedPlaContact` material.

This is static stage validation only. It does not prove the robot can reset,
contact a tread correctly, or climb.

Regenerate the world whenever the base world, stair geometry, physics rate, or
task configuration changes. That change intentionally invalidates manifests
from older checkpoints.

### Recorded world validation — 2026-07-27

World generation passed under Isaac Sim 6.0.1:

- four stair layers and four collision layers were present;
- stair layers had zero rigid-body APIs, so they remained static;
- the composed stage retained one articulation root, 13 robot rigid bodies,
  12 revolute joints, one camera, and one IMU;
- the world was 3,075 bytes and its SHA-256 was
  `15c82eee755d00c734cc65819c4be7c8f8520fe46df4943e1b5567f044a2cf8a`;
- the corrected YAML SHA-256 was
  `ad6c75e684a4a28d0baa9930bd266132bbb3f8525c0957a8753d46ef4c7d6e6b`;
- the report hashed both composed dependencies:
  `quadruped_robot_manual_world.usda` as
  `30c773a88564c21b87bb3da8fb0762f0d63f7350e2a31fc6d7ab7fae1542e3d9`
  and `quadruped_robot_floating.usdc` as
  `71b639bd877913bffeac47a1cfcb6f3dcabbbd1e25c6fba90b8b87e7ea96c6b8`;
- the inherited contact material was
  `/World/Materials/PrintedPlaContact`;
- `simulation/isaac/output/rl/ppo-stairs-v1/world_report.json` recorded
  `"status": "PASS"`.

This validates composition and static stage facts only, not contacts under
motion or stair completion.

## 3. Run the 512-step smoke test

Use an isolated output directory. The recorded smoke validation also exercised
the optional flat-policy initialization:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --smoke-test `
  --initialize-from-flat simulation\isaac\output\rl\ppo-walk-v1-2m\drobot_walk_ppo_final.zip `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v1-smoke
```

Smoke mode uses 512 total steps by default, 128-step rollouts, batch size 64,
and two epochs. It compresses all curriculum levels into those 512 steps and
therefore checks level transitions as well as reset, observation, action,
reward, PPO update, checkpoint, final save, and manifest paths.

Expected output:

```text
simulation/isaac/output/rl/ppo-stairs-v1-smoke/
├── training_report.json
├── monitor.csv
├── tensorboard/
├── checkpoints/
│   ├── drobot_stairs_ppo_<steps>_steps.zip
│   └── drobot_stairs_ppo_<steps>_steps.zip.contract.json
├── drobot_stairs_ppo_final.zip
└── drobot_stairs_ppo_final.zip.contract.json
```

Require `training_report.json` to say `"status": "PASS"` and
`"smoke_test": true`. A PASS proves pipeline execution only. A 512-step model
is not expected to climb and must not be promoted as a trained stair model.

If the smoke test fails, preserve the report before rerunning. Its
`error`, `traceback`, hashes, device details, and elapsed time are the first
debugging record.

### Recorded smoke validation — 2026-07-27

The corrected command above passed:

- 512 environment steps completed in 13.950 seconds;
- the policy had 164,633 parameters;
- the runtime confirmed 57 observations, 12 actions, 120 Hz physics, 60 Hz
  control, and two physics updates per action;
- the loaded model exactly matched the requested Stable-Baselines3 PPO
  algorithm/training-mode contract;
- transfer copied 11 same-shaped tensors and expanded the actor and critic
  input weights from 48 to 57 columns;
- all nine new input columns were zero-initialized and no optimizer state was
  transferred, with zero skipped tensors;
- curriculum callbacks transitioned at progress fractions `0.12109375`,
  `0.28125`, and `0.48046875` to 2, 3, and 4 active steps;
- the final model and matching `.contract.json` were saved under
  `simulation/isaac/output/rl/ppo-stairs-v1-smoke/`.

No episode ended within the 512-step smoke rollout because the episode limit is
1,320 steps. This is a successful pipeline test, not evidence of an effective
stair policy.

## 4. Choose initialization strategy

Use one of these, not both.

### From scratch

This is the clean experimental baseline:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v1-scratch
```

### Initialize from the evaluated flat policy

This creates a new stair model, copies compatible policy parameters, zeros the
nine new input columns, and starts with a fresh optimizer:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --initialize-from-flat simulation\isaac\output\rl\ppo-walk-v1-2m\drobot_walk_ppo_final.zip `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v1-from-flat
```

Transfer is accepted only when the source has 48 observations, 12 actions, and
ELU activation. The training report should include `flat_policy_transfer`,
the flat model hash, exactly 11 exact tensor copies, exactly two expanded input
layers, zero skipped tensors, and `"optimizer_transferred": false`.

Transfer may shorten the time needed to discover stable locomotion, but it is
not automatically better. The evaluated flat checkpoint has repeatable
negative-Y drift, so a fair study should keep separate from-scratch and
transferred directories and compare both with the same stair evaluation
seeds.

## 5. Run the full default job

The default YAML requests 3,000,000 control steps. The stopped v1 run was
initialized from the evaluated flat model:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --initialize-from-flat simulation\isaac\output\rl\ppo-walk-v1-2m\drobot_walk_ppo_final.zip `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v1
```

Training is headless by default. Add `--gui` only for short diagnosis because
rendering can reduce rollout throughput. The default PPO device is CPU; test
`--device cuda` only after comparing real steps per second on this
single-environment workload.

The curriculum uses the configured 3,000,000-step denominator. On a fresh
run, goals change at approximately 360,000, 840,000, and 1,440,000 steps.

### Active corrected run — started 2026-07-27 13:52:08 local

The command above is running under
`simulation/isaac/output/rl/ppo-stairs-v1/`; stdout and stderr logs are kept in
that directory. The first full PPO rollout completed:

- `2,048 / 3,000,000` steps;
- about 37 environment steps/s and roughly 54 seconds for the rollout;
- no stderr output;
- three Monitor rows: return/length `1167.607 / 1320`,
  `168.783 / 175`, and `1116.110 / 1320`.

The two 1,320-step rows are timeouts; the 175-step row is an early
termination. Monitor CSV does not contain stair height, success, or failure
reason, so these values do not show improvement or stair completion. The first
checkpoint is scheduled at 50,000 steps, approximately 20–25 minutes at the
current throughput. A rough full-run estimate is 22–25 hours, subject to
rollout/update speed changes.

The operating-system process ID is intentionally not recorded here because it
can change after a restart. Use the current process and log files for live
monitoring, then use checkpoint evaluation—not Monitor return alone—for
learning claims.

To run a deliberately shorter experiment:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --total-timesteps 1000000 `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v1-1m
```

That non-smoke example does **not** reach the four-step curriculum because
1,000,000 is less than the configured 1,440,000-step transition. Change the
versioned YAML intentionally if a different curriculum schedule is desired;
do not infer that `--total-timesteps` rescales it.

## 6. Resume a stair checkpoint

Resume only a 57-input stair checkpoint with its adjacent schema-2
`.contract.json`. The config, generated world, both composed world
dependencies, model hash, runtime semantics, and PPO algorithm/training mode
must still match:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --resume simulation\isaac\output\rl\ppo-stairs-v1\checkpoints\drobot_stairs_ppo_1500000_steps.zip `
  --total-timesteps 500000 `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v1-resumed
```

With resume, Stable-Baselines3 retains the checkpoint timestep and optimizer
state; the requested total is additional learning work. The report records
`resume_checkpoint` and `resume_contract_verification`.

A smoke checkpoint cannot be resumed as a full job: its saved
`training_mode: smoke`, rollout 128, batch 64, and two epochs do not match the
full contract's rollout 2,048, batch 256, and ten epochs. Resume a full
checkpoint with full settings, or continue a smoke checkpoint only with
`--smoke-test`.

The next manifest retains the original `transferred_from` flat-model lineage
and adds `resumed_from` hashes for the direct parent model and its manifest.

`--allow-unverified-resume` permits deliberate recovery when the manifest is
missing. It does not make an incompatible checkpoint safe and should never be
used to bypass an actual hash or semantic mismatch. Preserve the source model
and document why the override was necessary.

## 7. Watch TensorBoard

In a second PowerShell window:

```powershell
& C:\isaacsim\python.bat -m tensorboard.main `
  --logdir simulation\isaac\output\rl\ppo-stairs-v1\tensorboard
```

Open the local URL printed by TensorBoard. Useful plots include:

- `rollout/ep_rew_mean`: broad reward trend, not stair completion by itself;
- `rollout/ep_len_mean`: sudden shortening can mean quick success or quick
  failure, so compare it with episode reports;
- `train/approx_kl` and `train/clip_fraction`: whether PPO updates are becoming
  too large;
- `train/entropy_loss`: whether exploration collapses early;
- `train/explained_variance`: whether the value function predicts returns;
- `train/value_loss` and `train/policy_gradient_loss`: optimization health,
  not physical acceptance metrics;
- `time/fps`: actual end-to-end rollout/update throughput.

The environment also puts recent episode metrics and curriculum transitions
in the final `training_report.json`. Check `highest_step_reached`,
`stairs_completed`, clearance, tilt, and failure reasons instead of judging
only the PPO loss.

## Output and manifest guide

| Output | Meaning |
| --- | --- |
| `world_report.json` | Static USD composition, geometry, prim counts, config/world hashes, and both composed dependency hashes |
| `training_report.json` | Run status, software/device versions, PPO values, elapsed time, final model, contract, curriculum transitions, and recent episodes |
| `monitor.csv` | Stable-Baselines3 episode return/length/time records |
| `tensorboard/` | Scalar event files |
| `checkpoints/*.zip` | Scheduled resumable PPO checkpoints |
| `*.zip.contract.json` | Schema-2 model/config/world/dependency hashes, environment semantics, PPO/training mode, seed, and transfer/resume provenance |
| `drobot_stairs_ppo_final.zip` | Final SB3 model from that invocation |

`simulation/isaac/output/` is ignored by Git. Keep important reports and
models in backed-up experiment storage with the same relative manifest
pairing. Never copy a `.zip` without its `.zip.contract.json`.

## Reading progress and avoiding false positives

- Rising return with flat `highest_step_reached` usually means reward shaping
  is being optimized without climbing.
- High success at levels 1–3 and low success at level 4 isolates difficulty in
  the final riser/platform goal or 0.5-second hold.
- Positive forward displacement with little elevation gain suggests walking
  on the approach, stalling at a riser, or exploiting the edge of the
  corridor.
- Repeated terrain-height gains followed by `body_tipped` suggest aggressive
  hopping rather than stable stair locomotion.
- Minimum clearance approaching `0.20 m` or tilt approaching `45.6 degrees`
  indicates little safety margin even if the episode succeeds.
- A short `ep_len_mean` can improve because of fast success or worsen because
  of fast failure. Inspect the ending reason.
- The curriculum advances by time, not mastery. If the policy is not reliable
  at a level before the next transition, first compare checkpoints and metrics
  before changing reward weights.
- Evaluate intermediate checkpoints. The final optimization step is not
  guaranteed to be the best policy.

For a clean learning study, hold configuration and evaluation seeds constant,
run at least one scratch and one transferred job, and compare success rate and
physical metrics rather than selecting whichever TensorBoard reward looks
largest.
