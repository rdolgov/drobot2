# Training and monitoring stair v2

Run all commands from `robot-cad/`.

## Generate the separate v2 world

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\create_stairs_world.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v2.yaml `
  --report simulation\isaac\output\rl\ppo-stairs-v2\world_report.json
```

Require `"status": "PASS"` in `world_report.json`.

## Pipeline smoke

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v2.yaml `
  --smoke-test `
  --initialize-from-flat simulation\isaac\output\rl\ppo-walk-v1-2m\drobot_walk_ppo_final.zip `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v2-smoke
```

Smoke validates startup, 60-value observations, physical-height reward
calculation, transfer, PPO updates, manifests, and watchdog reporting. It does
not run long enough to reach the 100k behavioral gate.

## Full guarded training

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v2.yaml `
  --initialize-from-flat simulation\isaac\output\rl\ppo-walk-v1-2m\drobot_walk_ppo_final.zip `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v2
```

Checkpoints and manifests are written every 25,000 steps. Do not resume a v1
57-input checkpoint into this 60-input experiment.

## Read live progress

```powershell
Get-Content -Raw `
  simulation\isaac\output\rl\ppo-stairs-v2\progress_watchdog.json
```

Important fields:

| Field | Meaning |
| --- | --- |
| `completed_episodes` | Training episodes seen by the watchdog |
| `first_step_climb_episodes` | Episodes that reached step 1 and gained at least 20 mm actual base height |
| `first_step_climb_rate` | Physical first-step reaches divided by completed episodes |
| `maximum_step_reached` | Highest analytic stair index reached |
| `maximum_base_elevation_gain_m` | Best physical base rise above reset height |
| `initial_gate_passed` | Whether the 100k evidence threshold passed |
| `abort_reason` | Exact failed threshold or stagnation reason |

TensorBoard:

```powershell
& C:\isaacsim\python.bat -m tensorboard.main `
  --logdir simulation\isaac\output\rl\ppo-stairs-v2\tensorboard
```

Read `stair/first_step_climb_rate`, `stair/maximum_step_reached`,
`stair/maximum_base_elevation_gain_m`, and `stair/success_rate` before PPO
return. A rising shaped return without these physical metrics is not progress.

The v2 transfer uses a `0.00005` fine-tuning learning rate and a `0.03`
target-KL limit. The latter stops the remaining optimization epochs whenever
one PPO update moves too far from the walking policy. This matters because the
stair-height and success rewards are substantially sharper than the flat-task
reward. Treat repeated `approx_kl` values above the limit as an unstable run,
even when return is increasing.

## Abort outcomes

Successful completion writes:

```text
training_report.json: "status": "PASS"
drobot_stairs_ppo_final.zip
```

Automatic no-progress termination writes:

```text
training_report.json: "status": "ABORTED_NO_PROGRESS"
drobot_stairs_ppo_aborted.zip
progress_watchdog.json: "status": "ABORTED"
```

The aborted model is retained for diagnosis. Do not resume it unchanged; fix
the failed observation, reward, reset, or curriculum assumption first.

## Observed 50k review stop

The stabilized run was stopped by user request at 50,001 reported steps,
before the automatic 100k gate. It completed 107 episodes with no physically
elevated first-step episode and no success. Maximum base elevation was
0.018115 m, below the 0.02 m qualifying threshold. The complete 50k checkpoint
is retained; `training_report.json` records `ABORTED_BY_USER`. Do not resume
this checkpoint unchanged. The next experiment is the staged foot-lift and
depth-perception design in [perception-plan.md](perception-plan.md).

## Evaluate a checkpoint

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v2.yaml `
  --model simulation\isaac\output\rl\ppo-stairs-v2\checkpoints\drobot_stairs_ppo_100000_steps.zip `
  --active-steps 1 `
  --episodes 10 `
  --gui `
  --camera-view external `
  --report simulation\isaac\output\rl\ppo-stairs-v2\evaluation-100k-level1.json
```

The checkpoint evaluator verifies the model/config/world/dependency/PPO
contract before applying actions.
