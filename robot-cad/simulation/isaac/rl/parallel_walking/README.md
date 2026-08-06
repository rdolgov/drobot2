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

The defaults are five visible robots and 20 PPO iterations. The V3 task starts with commands
between `0.04` and `0.08 m/s`, then reaches `0.10` to `0.18 m/s` over roughly the first 1,000
PPO iterations. Omit `-Fresh` on later runs to automatically continue the newest V3 checkpoint.
Override the defaults with `-Iterations` and `-NumEnvs`.

V3 uses a native bounded Beta policy, rational (non-vanishing) velocity tracking, progressive
distance milestones, and symmetric foot air-time/touchdown rewards gated by commanded-direction
progress. These rewards prescribe neither a gait phase nor a leg order.

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
- `Metrics/swing_step_rate`: should become non-zero without dominating the episode
- `Metrics/touchdowns_per_episode`: should become non-zero as stepping emerges
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

The preview script selects the newest matching V3 checkpoint unless `-Checkpoint` is supplied.
V1 and V2 checkpoints are retained, but should not seed V3. V1 had exploding Gaussian noise;
V2 remained bounded but learned saturated action means and zero distance successes. V3's Beta
policy has a different output head and must start fresh.

## Later: backward and turns

Once forward walking is stable, initialize the directional curriculum from a forward model:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet directional `
  -Checkpoint .\logs\rsl_rl\drobot_commanded_walk_forward_v3_direct\RUN\model_N.pt `
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
