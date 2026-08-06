# Parallel commanded walking

This task trains a pure PPO policy across many Isaac Lab environments. It starts every
robot in the validated symmetric, four-foot neutral stance. There is no scripted gait,
phase clock, foot order, or motion trajectory.

The policy observation has 48 hardware-reproducible values:

- commanded forward velocity, lateral velocity, and yaw rate (3)
- body IMU angular velocity, projected gravity, and linear acceleration (9)
- joint position error, normalized joint velocity, and previous action (36)

The flat-ground actor deliberately does not consume the depth sensor. Depth is useful for
stairs and obstacles, but a featureless plane contributes no useful terrain information.
The actor is a two-layer 256x256 MLP using deployable IMU and joint state. During training
only, the critic also sees simulator base velocity, base height, and foot contacts. Those
privileged values are never required by the deployed actor.

## V16 sustained-walking correction

V15 could move during an eight-second episode but its Gaussian action mean became heavily
saturated and it settled into a fixed pose after roughly ten seconds without a reset. V16
addresses that failure directly:

- the actor uses a native bounded Beta distribution, so every sampled and deployed action
  stays in `[-1, 1]`
- the timeout curriculum ramps from 8 to 32 seconds over 1,000 PPO iterations
- a two-second rolling-speed reward preserves forward motion and penalizes sustained stalls
- checkpoint selection uses uninterrupted 30-second evaluation, not training return alone
- preview uses an asset-following third-person camera so the complete robot stays visible

The selected `model_250.pt` was chosen from a 1,000-iteration, 128-environment run. In three
uninterrupted 30-second evaluations it had `0/3` falls, no stalled five-second windows, mean
forward displacement `4.443 m`, mean final five-second speed `0.142 m/s`, and mean lateral
drift `1.604 m`. The drift is the next locomotion issue to improve, but the previous
post-ten-second stop is gone.

## 1. Visible five-robot training

Start a new forward-only policy:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_visible.ps1 -Fresh
```

The defaults are five visible robots and 20 PPO iterations. Omit `-Fresh` on later runs to
continue the newest V16 checkpoint. `-Fresh` creates a separate run; it does not erase an
existing model. Override the defaults with `-Iterations` and `-NumEnvs`.

## 2. Headless parallel training

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -Iterations 500 -NumEnvs 128
```

This resumes the newest accepted forward checkpoint automatically. The repository bundles
the selected V16 policy at
`simulation/isaac/models/parallel-walking-v16/model_250.pt`, which is the clean-checkout
fallback. To continue that exact policy explicitly:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v16\model_250.pt `
  -Iterations 500 -NumEnvs 128
```

The main improvement signals are:

- `Metrics/min_rolling_forward_speed_m_s`: should remain above `0.04 m/s`
- `Metrics/sustained_stall_rate`: should approach zero
- `Metrics/mean_velocity_error_m_s`: should decrease
- `Metrics/mean_commanded_speed_m_s`: should approach `0.15 m/s`
- `Metrics/net_forward_displacement_m`: should keep increasing with the horizon
- `Metrics/net_lateral_displacement_m`: should stay close to zero
- `Metrics/current_episode_horizon_s`: ramps from 8 to 32 seconds
- `Metrics/action_saturation_rate`: should stay low rather than approach one
- `Metrics/qualified_touchdowns_per_episode`: should be non-zero as steps emerge
- `Metrics/fall_rate`: should approach zero
- `Mean reward`: should rise, but is secondary to physical displacement and stalls

Evaluate a checkpoint without reset-dependent transients:

```powershell
& .\simulation\isaac\rl\parallel_walking\evaluate_walking_sustained.ps1 `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v16\model_250.pt `
  -Seconds 30 -Episodes 3
```

## 3. Preview one robot

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 -Command forward
```

For a continuous preview that resets only on a fall, not on a training timeout:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -Command forward -NoTimeLimit
```

Close Isaac Sim to stop an unlimited preview. To record exactly 30 seconds with the
asset-following camera:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -Command forward -NoTimeLimit -RecordSeconds 30
```

The tracked review clip is
`reviews/parallel-walking-v16-model250-sustained-30s.mp4` (1,800 frames at 60 fps).
Underscore-prefixed calibration and workflow directories are ignored during automatic
checkpoint selection.

## Later: backward and turns

Once forward walking and heading retention are stable, initialize the directional curriculum:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet directional `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v16\model_250.pt `
  -Iterations 500 -NumEnvs 128
```

After that first transfer, repeat the command without `-Checkpoint` to resume the newest
directional run. Preview learned commands with `-Command left`, `right`, or `backward` and
`-CommandSet directional`. Left and right mean yaw turns with a small forward velocity; the
reserved lateral command remains zero until a strafing curriculum is intentionally added.
