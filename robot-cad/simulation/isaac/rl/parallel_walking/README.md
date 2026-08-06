# Parallel commanded walking

This task trains a pure PPO policy across many Isaac Lab environments. It starts every
robot in the same symmetric, four-foot neutral stance used by the corrected stair task.
There is no scripted gait, phase clock, foot order, or motion trajectory.

The policy observation has 48 hardware-reproducible values:

- commanded forward velocity, lateral velocity, and yaw rate (3)
- body IMU angular velocity, projected gravity, and linear acceleration (9)
- joint position error, normalized joint velocity, and previous action (36)

The flat-ground walker deliberately does not consume the depth sensor. Depth is useful for
stairs and obstacles, but not for learning basic velocity tracking; keeping it out makes this
first policy faster to train and easier to transfer. The command input shape already supports
forward, backward, lateral, and yaw commands without changing the network later.

## 1. Visible five-robot training

Start a new forward-only policy:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_visible.ps1 -Fresh
```

The defaults are five visible robots and 20 PPO iterations. The V2 task starts with commands
between `0.05` and `0.10 m/s`, then reaches `0.10` to `0.18 m/s` over roughly the first 1,000
PPO iterations. Omit `-Fresh` on later runs to automatically continue the newest V2 checkpoint.
Override the defaults with `-Iterations` and `-NumEnvs`.

## 2. Headless parallel training

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -Iterations 500 -NumEnvs 128
```

This also resumes the newest forward checkpoint automatically. Use `-Fresh` only when an
intentional new random policy is wanted.

The main improvement signals are:

- `Metrics/mean_velocity_error_m_s`: should decrease
- `Metrics/mean_commanded_speed_m_s`: should approach the commanded speed range
- `Metrics/commanded_distance_m`: should increase for the forward curriculum
- `Metrics/distance_success_rate`: should rise toward one
- `Metrics/action_saturation_rate`: should stay low rather than approach one
- `Metrics/fall_rate`: should approach zero
- `Mean reward`: should rise, but is secondary to the physical metrics above

## 3. Preview one robot

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 -Command forward
```

To record a 30-second review clip:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -Command forward -RecordSeconds 30
```

The preview script selects the newest matching V2 checkpoint unless `-Checkpoint` is supplied.
V1 checkpoints are retained in `drobot_commanded_walk_forward_direct`, but should not seed V2:
their scalar action standard deviation grew far beyond the action range.

## Later: backward and turns

Once forward walking is stable, initialize the directional curriculum from a forward model:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet directional `
  -Checkpoint .\logs\rsl_rl\drobot_commanded_walk_forward_v2_direct\RUN\model_N.pt `
  -Iterations 500 -NumEnvs 128
```

After that first transfer, repeat the same command without `-Checkpoint`; it resumes the newest
directional run. Preview the learned commands with:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -CommandSet directional -Command left
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -CommandSet directional -Command right
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -CommandSet directional -Command backward
```

Left and right currently mean yaw turns with a small forward velocity. The reserved lateral
command remains zero until a later strafing curriculum is intentionally added.
