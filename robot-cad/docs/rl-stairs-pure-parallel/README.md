# Pure parallel stair PPO

## Scope and result status

This task is a separate, pure-reward PPO experiment for the full-size stair:
`180 mm` rise and `250 mm` tread. It intentionally removes the scripted gait
phase, prescribed leg order, inverse-kinematics reference, and action replay
used by earlier stair experiments. PPO controls all 12 joints directly in 128
GPU-parallel Isaac Lab environments.

The first 80-iteration run is a pipeline and exploration result, not a
converged stair policy. It processed 245,760 transitions in 109.09 seconds and
did not produce a repeatable climb. One training sample reached 0.4588 m of
forward progress, just beyond the first riser, while the largest observed base
height gain was 0.1133 m. Those maxima occurred at different iterations and do
not establish a successful step. The deterministic video likewise shows the
short-run policy leaning into the first riser without climbing it.

Follow-up training first extended the original reward chain by 1,228,800
transitions; it converged mainly on surviving in front of the riser. The reward
was then corrected to measure the physical fork-tip contact point rather than
the distal-link origin. A 308-iteration run processed 946,176 transitions, and
a 400-iteration lift-hold continuation processed another 1,228,800. Across all
pure-parallel runs in this experiment, PPO processed 3,649,536 transitions.

The lift-hold continuation produced two logged events with two simultaneous
force/height-verified tread contacts (iterations 399 and 606). The full-climb
success rate remained 0%, and a fresh deterministic playback of iteration 600
approached and lifted at the first riser but did not climb. These are useful
exploration and reward-shaping results, not a stair-placement pass.

## Editable sources

- `simulation/isaac/rl/parallel_stairs/pure_stairs_env.py`: vectorized
  observation, direct action, reward, failure gates, and reset behavior.
- `simulation/isaac/rl/parallel_stairs/pure_stairs_env_cfg.py`: robot,
  actuator, sensor, physics, and scene configuration.
- `simulation/isaac/rl/parallel_stairs/exact_stairs_terrain.py`: exact stair
  mesh generator.
- `simulation/isaac/rl/parallel_stairs/agents/rsl_rl_ppo_cfg.py`: PPO and
  256-by-256 actor/critic configuration.
- `exports/isaac/quadruped_robot_floating.usdc`: robot asset loaded by the
  task.

Isaac Lab was installed separately at
`C:\Users\roman\Documents\dev\IsaacLab`, `develop` commit
`90ee100616d9b77eb8e28f171252dc58e39181d7` (version 3.0.0), and linked to the
local Isaac Sim installation.

## Policy contract

The actor receives 70 values at 30 Hz:

| Input | Values | Real source |
| --- | ---: | --- |
| Body angular velocity | 3 | IMU gyroscope |
| Projected gravity | 3 | IMU orientation/gravity estimate |
| Joint position error | 12 | Servo feedback |
| Joint velocity | 12 | Servo feedback/estimate |
| Previous action | 12 | Controller state |
| Foot load/contact | 4 | Four foot load/contact channels |
| Compressed depth | 24 | VL53L5CX 8-by-8 depth grid |

The depth model runs at 15 Hz with one sensor-frame of latency, near-field
`+/-15 mm` error, 5% far-field proportional error, and 5% dropout. The 8-by-8
grid is compressed into three lateral lanes for each of eight rows.

RGB is not a policy input. It is enabled only for the review recording. Stair
coordinates, simulator body pose, terrain height, and other privileged ground
truth are excluded from the observation. Ground truth is used only to compute
reward, failure termination, and evaluation metrics.

The action is a normalized 12-vector mapped to joint-position targets around
the nominal stance. The actuator configuration retains the real-test
`0.8825985 N*m` effort cap on every joint.

## Reward and reset

The reward combines forward base displacement, upward base displacement,
incremental maximum-foot clearance, a persistent hold reward that saturates at
`190 mm` clearance, contacts supported on higher treads, an alive term,
uprightness, action-rate, effort, and body-rate penalties. There is
no gait clock, commanded foot, gait phase, reference trajectory, or scripted
action. Episodes terminate for insufficient base height, excessive tilt,
excessive lateral displacement, walking backward out of the approach, or the
12-second time limit.

The terrain uses four solid steps, a 0.45 m approach measured from the robot
spawn origin, and a 0.75 m top platform. Static/dynamic friction are 1.10/0.90.

## Reproduction

From the repository root in PowerShell:

```powershell
& C:\Users\roman\Documents\dev\IsaacLab\isaaclab.bat -p simulation/isaac/rl/parallel_stairs/zero_agent_pure_parallel_stairs.py --task Drobot-Pure-Stairs-Direct --num_envs 128 --device cuda

& C:\Users\roman\Documents\dev\IsaacLab\isaaclab.bat -p simulation/isaac/rl/parallel_stairs/train_pure_parallel_stairs.py --rl_library rsl_rl --task Drobot-Pure-Stairs-Direct --num_envs 128 --seed 1055 --device cuda --max_iterations 80 --run_name pure128-seed1055

& C:\Users\roman\Documents\dev\IsaacLab\isaaclab.bat -p simulation/isaac/rl/parallel_stairs/train_pure_parallel_stairs.py --rl_library rsl_rl --task Drobot-Pure-Stairs-Direct --num_envs 128 --seed 1055 --device cuda --resume --load_run 2026-08-02_21-25-21_pure128-seed1055 --checkpoint model_79.pt --max_iterations 400 --run_name pure128-resume400-seed1055
```

To record a deterministic 12-second review run, supply the selected checkpoint
as an absolute path:

```powershell
& C:\Users\roman\Documents\dev\IsaacLab\isaaclab.bat -p simulation/isaac/rl/parallel_stairs/play_pure_parallel_stairs.py --rl_library rsl_rl --task Drobot-Pure-Stairs-Direct --num_envs 1 --seed 1056 --device cuda --checkpoint C:\absolute\path\to\model.pt --video --video_length 360
```

## Validation and limitations

- A 128-environment robot smoke ran for 300 steps at an aggregate 761.2
  environment-steps/second including simulator startup.
- A two-iteration PPO smoke survived beyond the earlier one-frame termination
  defect and reached about 1,985 environment-steps/second.
- The first bounded training run averaged roughly 2,200-2,500
  environment-steps/second after startup.
- The follow-up chain ran at roughly 2,001-2,635 environment-steps/second and
  reached two valid simultaneous tread contacts during stochastic exploration.
- Deterministic playback loaded the saved 70-input, 256-by-256 actor and wrote
  an 8-second, 240-frame MP4. It did not climb the first riser.
- No simulation result here proves hardware transfer, robust first-step
  acquisition, or a full climb. Continue training and evaluate several unseen
  seeds before considering mechanical changes or a larger neural network.
