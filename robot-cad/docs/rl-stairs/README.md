# Stair-climbing reinforcement learning

This folder documents the separate `Drobot-Quadruped-Stairs-v1` experiment.
It does not replace the flat-ground task or its model. The stair task has its
own configuration, 57-value observation contract, environment, training
entry point, checkpoints, reports, evaluation, and recordings.

## Current status — 2026-07-27

The corrected stair task is implemented. Its pure contracts, generated Isaac
world, exact flat-policy transfer, 512-step PPO pipeline, schema-2 checkpoint
manifest, loaded-algorithm checks, deterministic reload, screenshot, and
recording paths have now been exercised. The learned behavior has not
converged: the smoke checkpoint completed zero of two level-1 evaluation
episodes and reached no stair in its recorded episode.

| Scope | Current claim |
| --- | --- |
| Python/NumPy contract behavior | `9 passed, 1 skipped`; the skipped Torch transfer unit was exercised by the Isaac smoke run |
| Isaac stair-world generation and static stage inspection | PASS: four collision layers and the expected robot/sensor prim counts |
| 512-step Isaac/PPO smoke test | PASS from flat-policy initialization; pipeline validation only |
| Deterministic level-1 smoke evaluation | PASS manifest + loaded-algorithm checks; `0/2` stair success |
| Full 3,000,000-step training | Stopped at 964,608 steps after the 950k audit proved `0/5` first-step reaches |
| H.264 smoke recording | PASS manifest + loaded-algorithm + encoding checks; `0/1` stair success |
| Physical robot | Not tested |

The corrected YAML SHA-256 is
`ad6c75e684a4a28d0baa9930bd266132bbb3f8525c0957a8753d46ef4c7d6e6b`;
the generated world SHA-256 remains
`15c82eee755d00c734cc65819c4be7c8f8520fe46df4943e1b5567f044a2cf8a`.
The smoke model has 164,633 parameters and trained for 512 steps in
13.950 seconds. Transfer copied 11 tensors exactly, expanded exactly the actor
and critic input matrices from 48 to 57 inputs, zero-initialized the new
columns, and skipped no tensors. No episode ended during those 512 steps,
although the compressed curriculum emitted all three scheduled transitions.
The recorder then wrote 660 H.264 frames at `960 × 540` and 30 FPS after both
manifest and loaded PPO algorithm verification passed.

The full v1 run was stopped at 964,608 steps. A five-episode audit of its 950k
checkpoint recorded `0/5` success and step index `0` in every episode, with
mean forward displacement `0.735 m` and lateral displacement
`0.410-0.504 m`. Three episodes left the corridor. Its high mean return was
therefore approach-walking reward, not stair learning. See the separate
[v2 correction guide](../rl-stairs-v2/README.md) for the close-start,
navigation-aware, physical-height-reward experiment and its automatic
100k-step progress gate.

## Reading order

1. [Architecture](architecture.md) explains the world, observations, actions,
   policy network, curriculum, reward, termination, and model contract.
2. [Training](training.md) gives exact PowerShell commands for world
   generation, smoke/full/transfer/resume jobs, TensorBoard, and output review.
3. [Evaluation](evaluation.md) gives staged deterministic evaluation and
   recording commands, metric interpretation, acceptance criteria, and
   sim-to-real limitations.
4. The existing [flat-walking guide](../rl-training.md) records the source
   policy that may optionally initialize the stair network.

Run every command from the `robot-cad/` directory. Scripts importing Isaac Sim
must use `C:\isaacsim\python.bat`, not the ordinary project Python.

## Experiment at a glance

| Item | Stair-task contract |
| --- | --- |
| Task ID | `Drobot-Quadruped-Stairs-v1` |
| World | `exports/isaac/quadruped_robot_stairs_world.usda` |
| Stair profile | Four `40 mm` rises, `230 mm` treads, `1.00 m` width, `0.50 m` top platform |
| Policy observation | 57 floats: 48 walking values + 8 analytic terrain samples + 1 goal distance |
| Policy action | 12 normalized joint-position offsets |
| Network | Separate actor and critic MLPs with two `256`-unit ELU layers |
| Algorithm | Stable-Baselines3 PPO |
| Timing | `120 Hz` physics, `60 Hz` control, two physics steps per action |
| Default training budget | 3,000,000 environment steps |
| Curriculum | Goal after 1, 2, 3, then 4 steps |
| Success reward | `+400`, larger than discounted stationary loitering at `gamma = 0.995` |
| Episode limit | 22 seconds / 1,320 control steps |
| Optional initialization | Policy weights from the 48-input flat-walking PPO model |

## Source map

All stair-specific implementation lives separately under
`simulation/isaac/rl/stairs/`.

| Source | Responsibility |
| --- | --- |
| [`__init__.py`](../../simulation/isaac/rl/stairs/__init__.py) | Marks the stair experiment as a separate Python source package |
| [`quadruped_stairs_v1.yaml`](../../simulation/isaac/rl/stairs/quadruped_stairs_v1.yaml) | Versioned task, staircase, curriculum, reward, termination, action, reset, and PPO values |
| [`create_stairs_world.py`](../../simulation/isaac/rl/stairs/create_stairs_world.py) | Composes collision stair layers over the validated flat Isaac world and writes a static-stage report |
| [`_stair_geometry.py`](../../simulation/isaac/rl/stairs/_stair_geometry.py) | Pure geometry definitions used by the generator and tests |
| [`_stair_rl_contract.py`](../../simulation/isaac/rl/stairs/_stair_rl_contract.py) | Pure observation, curriculum, height-query, reward, and failure functions |
| [`_quadruped_stairs_env.py`](../../simulation/isaac/rl/stairs/_quadruped_stairs_env.py) | Gymnasium/Isaac environment and episode metrics |
| [`_policy_transfer.py`](../../simulation/isaac/rl/stairs/_policy_transfer.py) | Controlled 48-input to 57-input policy initialization |
| [`_run_support.py`](../../simulation/isaac/rl/stairs/_run_support.py) | Schema-2 dependency, environment, PPO algorithm, transfer, and resume manifests |
| [`train_stairs_ppo.py`](../../simulation/isaac/rl/stairs/train_stairs_ppo.py) | PPO creation/resume, curriculum callback, checkpoints, TensorBoard, and training report |
| [`evaluate_stairs_ppo.py`](../../simulation/isaac/rl/stairs/evaluate_stairs_ppo.py) | Deterministic multi-episode evaluation and optional screenshot |
| [`record_stairs_ppo.py`](../../simulation/isaac/rl/stairs/record_stairs_ppo.py) | Deterministic external/onboard H.264 recording |
| [`test_quadruped_stairs_rl_contract.py`](../../tests/test_quadruped_stairs_rl_contract.py) | Isaac-free focused contract tests |

The stair environment intentionally reuses the reviewed robot, IMU adapter,
standing pose, joint limits, rated torque, and base 48-value observation from
the flat-walking implementation. It rejects a changed articulation DOF order
instead of silently applying actions to the wrong joints.

## Recommended workflow

```text
pure-contract tests
        ↓
generate and inspect stair world
        ↓
512-step pipeline smoke test
        ↓
full training (from scratch or flat-policy initialization)
        ↓
evaluate checkpoints at active levels 1, 2, 3, and 4
        ↓
multi-seed full-stair evaluation
        ↓
record a representative run
        ↓
only then plan guarded hardware experiments
```

Keep every output directory unique. In particular, do not write stair
checkpoints into a flat-walking run folder and do not overwrite a from-scratch
run with a transferred run. The schema-2 manifest beside each stair model
binds it to the exact configuration, world and its two composed dependencies,
runtime interface, PPO algorithm/training mode, and transfer/resume lineage.

## What this experiment is intended to teach

This first stair version answers a narrow question: can the existing 12-joint
robot learn to move forward over one fixed, low staircase when given an
explicit local terrain profile? It is designed to make the learning contract
easy to inspect, not to solve general terrain perception.

It does **not** yet teach the robot to infer stairs from its camera. The eight
terrain values come from the known simulated stair geometry and simulator
world position. Replacing those values with a depth/camera estimator, adding
terrain and actuator randomization, and proving a safe sim-to-real path are
separate future tasks.
