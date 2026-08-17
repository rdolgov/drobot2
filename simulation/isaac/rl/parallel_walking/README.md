# Parallel commanded walking

This task trains a pure PPO policy across many Isaac Lab environments. It starts every
robot in the validated symmetric, four-foot neutral stance. V18 provides an analytic
diagonal-pair gait reference and clock, while PPO remains responsible for balance and
the final joint commands.

The policy observation has 50 hardware-reproducible values:

- commanded forward velocity, lateral velocity, and yaw rate (3)
- sine and cosine of the gait clock (2)
- body IMU angular velocity, projected gravity, and linear acceleration (9)
- joint position error, normalized joint velocity, and previous action (36)

The flat-ground actor deliberately does not consume the depth sensor. Depth is useful for
stairs and obstacles, but a featureless plane contributes no useful terrain information.
The actor is a two-layer 256x256 MLP using deployable IMU and joint state. During training
only, the critic also sees simulator base velocity, base height, and foot contacts. Those
privileged values are never required by the deployed actor.

## V18 coordinated walking

V18 fixes the stationary bent-knee behavior seen in the V17 review. Its reward makes
commanded forward speed and sustained displacement the dominant objective, explicitly
penalizes two-second stalls, and adds action-rate plus action-acceleration costs for
smoothness. A 0.8-second gait clock schedules diagonal pairs: front-left with rear-right,
then front-right with rear-left. Per-leg touchdown and scheduled-contact metrics ensure
that a policy cannot score well by dragging or ignoring a leg.

The selected `model_299.pt` completed three deterministic, uninterrupted 30-second
trials with zero falls and zero stalled five-second windows. It averaged 4.789 m forward
at 0.160 m/s. Every leg remained active (38--51 touchdowns per trial), with 80--95%
scheduled-contact agreement. Mean absolute lateral displacement was 0.570 m, which is
the main remaining weakness.

The selected checkpoint is
`simulation/isaac/models/parallel-walking-v18-coordinated/model_299.pt`. Later
checkpoints were rejected because additional training increased drift and action
saturation; a hard joint-action pairing experiment was also rejected because it caused
the moving policy to freeze.

## V17 rectangular-shoe smooth-walking training

V17 retrains the sustained forward policy for the 2026-08-13 flat rectangular
shoe: a 100 x 60 x 6 mm PLA sole, a 94 x 54 x 1 mm bonded tread, and a 70.237 g
CAD mass estimate per shoe. The old spherical fork-tip collisions are disabled.
The nominal pose uses the hardware controller's 80 mm fore/aft flat-sole stance.

The actor input and output contracts remain unchanged from V16. The objective
adds second-difference action smoothing, planted-foot slip, touchdown impact,
qualified touchdown, and stronger lateral/yaw regulation. This makes the new
policy deployable through the same 48-value IMU/joint interface while asking it
to use the wider contact patch smoothly instead of learning around point feet.

The superseded V17 workflow initialized from its selected rectangular-shoe
checkpoint unless `-Fresh` was supplied. Its continuations are stored under
`logs/rsl_rl/drobot_commanded_walk_forward_v17d_rectangular_smooth_direct/`;
the earlier V17a/V17b/V17c experiment directories are retained as training
evidence but are no longer searched automatically.

## V16 sustained-walking baseline

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
continue the newest selected V18 checkpoint, falling back to the bundled coordinated
policy for the first continuation. `-Fresh` creates a separate run; it does not erase an
existing model. Override the defaults with `-Iterations` and `-NumEnvs`.

## 2. Headless parallel training

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -Iterations 500 -NumEnvs 128
```

This resumes the newest accepted forward checkpoint automatically. The repository bundles
the selected coordinated rectangular-shoe policy at
`simulation/isaac/models/parallel-walking-v18-coordinated/model_299.pt`, which is the clean-checkout
fallback. To continue that exact policy explicitly:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v18-coordinated\model_299.pt `
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
- `Metrics/qualified_touchdowns_<leg>`: all four values should remain non-zero
- `Metrics/fall_rate`: should approach zero
- `Mean reward`: should rise, but is secondary to physical displacement and stalls

Evaluate a checkpoint without reset-dependent transients:

```powershell
& .\simulation\isaac\rl\parallel_walking\evaluate_walking_sustained.ps1 `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v18-coordinated\model_299.pt `
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

The selected coordinated review clip is
`simulation/reviews/parallel-walking-v18-coordinated-model299-30s.mp4`
(1,800 frames at 60 fps). The older V16 clip remains available for comparison.
Underscore-prefixed calibration and workflow directories are ignored during automatic
checkpoint selection.

## Later: backward and turns

Once forward walking and heading retention are stable, initialize the directional curriculum:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet directional `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v18-coordinated\model_299.pt `
  -Iterations 500 -NumEnvs 128
```

After that first transfer, repeat the command without `-Checkpoint` to resume the newest
directional run. Preview learned commands with `-Command left`, `right`, or `backward` and
`-CommandSet directional`. Left and right mean yaw turns with a small forward velocity; the
reserved lateral command remains zero until a strafing curriculum is intentionally added.
