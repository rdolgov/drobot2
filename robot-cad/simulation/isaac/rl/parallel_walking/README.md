# Parallel commanded walking

This task trains a pure PPO policy across many Isaac Lab environments. It starts every
robot in the validated symmetric, four-foot neutral stance. There is no scripted gait,
phase clock, foot order, or motion trajectory.

The policy observation has 48 hardware-reproducible values:

- commanded forward velocity, lateral velocity, and yaw rate (3)
- body IMU angular velocity, projected gravity, and linear acceleration (9)
- joint position error, normalized joint velocity, and previous action (36)

The flat-ground actor deliberately does not consume the depth sensor. Depth is useful for
stairs and obstacles, but a featureless flat plane contributes no depth information about
body speed. The actor is the same two-layer 256x256 MLP shape as the independently validated
single-environment walking policy and uses deployable IMU and joint state. During training
only, the critic also sees simulator base velocity, base height, and foot contacts. Those
privileged values are never required by the deployed actor.

## 1. Visible five-robot training

Start a new forward-only policy:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_visible.ps1 -Fresh
```

The defaults are five visible robots and 20 PPO iterations. V15 trains at the validated
`0.15 m/s` target with a 60 Hz control rate. Omit `-Fresh` on later runs to automatically
continue the newest V15 checkpoint. Override the defaults with `-Iterations` and `-NumEnvs`.

V15 is based on the reward shape, reset stance, termination limits, and 256x256 MLP capacity
of the independently validated `ppo-walk-v1-2m` policy. The main correction is the implicit
PhysX servo drive used by that validated world: the prior explicit PD implementation drove
neutral joints continuously near 4.6 rad/s even with zero policy action. The corrected neutral
pose settles near 0.05 rad/s with four steady ground contacts. V15 initializes its matching actor
layers from that earlier pure-RL policy, then uses PPO across 128 updated-torque environments to
adapt it; no scripted gait, phase, or trajectory is added. The transfer helper is
`bootstrap_walking_from_sb3.py`.

The policy uses a 0.1 initial action standard deviation instead of 1.0, less entropy pressure,
and a survival/failure balance that makes a full stable episode worth more than a fast early
fall. `gamma=0.995` covers the 480-step episode better, and a 0.1 global reward scale keeps
critic targets conditioned without changing reward ratios. At 0.15 m/s, perfect tracking earns
about 2.5 times the per-step reward of standing, so the reward still strongly favors motion.
Net displacement, base height, falls, touchdown quality, and action saturation are measured
independently so a high return cannot be mistaken for walking.

## 2. Headless parallel training

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -Iterations 500 -NumEnvs 128
```

This resumes the newest forward checkpoint automatically. Use `-Fresh` only when an intentional
new random policy is wanted; it creates another run and does not erase earlier checkpoints.

The repository includes the selected stable V15 checkpoint at
`simulation/isaac/models/parallel-walking-v15/model_125.pt`. On a clean checkout, preview and
training fall back to it automatically. If a later training run regresses, explicitly restart
from this checkpoint instead of continuing the newest file:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v15\model_125.pt `
  -Iterations 250 -NumEnvs 128
```

The main improvement signals are:

- `Metrics/mean_velocity_error_m_s`: should decrease
- `Metrics/mean_commanded_speed_m_s`: should approach `0.15 m/s`
- `Metrics/net_forward_displacement_m`: should approach `1.2 m` over eight seconds
- `Metrics/net_lateral_displacement_m`: should stay near zero
- `Metrics/mean_base_height_m`: should remain near the `0.373 m` target
- `Metrics/distance_success_rate`: should rise toward one
- `Metrics/action_saturation_rate`: should stay low rather than approach one
- `Metrics/qualified_touchdowns_per_episode`: should become non-zero as real steps emerge
- `Metrics/fall_rate`: should approach zero
- `Reward/forward_velocity_tracking`: should approach `2.0`
- `Mean reward`: should rise, but is secondary to physical displacement

## 3. Preview one robot

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 -Command forward
```

To record a 30-second review clip:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -Command forward -RecordSeconds 30
```

The preview script selects the newest matching V15 checkpoint unless `-Checkpoint` is supplied.
On a clean checkout it uses the bundled selected model. Underscore-prefixed calibration and
workflow directories are deliberately ignored during automatic checkpoint selection.

To let a healthy robot continue beyond the eight-second training horizon, while still resetting
it if it falls, use:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -Command forward -NoTimeLimit
```

Close Isaac Sim to stop an unlimited preview. This option affects preview only; PPO training
retains fixed eight-second episodes.

## Later: backward and turns

Once forward walking is stable, initialize the directional curriculum from a forward model:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet directional `
  -Checkpoint .\logs\rsl_rl\drobot_commanded_walk_forward_v15_rl_transfer_direct\RUN\model_N.pt `
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
