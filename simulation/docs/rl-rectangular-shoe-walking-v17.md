# Rectangular-shoe smooth-walking PPO V17

## Objective

Retrain the 60 Hz Isaac Lab forward-walking policy for the flat rectangular
shoe designed on 2026-08-13, while reducing the sustained V16 policy's lateral
drift and visibly abrupt joint motion. This is a flat-ground learning task; it
does not claim hardware transfer or stair capability.

## Robot and contact model

- source articulation: `simulation/exports/isaac/quadruped_robot_floating.usdc`
- structural sole: 100 x 60 x 6 mm
- bonded tread proxy: 94 x 54 x 1 mm
- fork-axis to tread face: 31 mm
- CAD PLA mass estimate: 70.237 g per shoe
- tread friction: 1.05 static, 0.85 dynamic
- compliant contact: 12,000 N/m stiffness and 45 N s/m damping
- nominal stance: 80 mm fore/aft with equal and opposite hip/knee angles so
  every planted sole is flat
- joint torque cap: 0.8825985 N m
- control/physics rates: 60/120 Hz

The shoe is represented by nested sole and tread collision boxes on each
distal rigid body. The legacy spherical fork-tip collision is disabled. The
estimated shoe mass and approximate combined distal-link inertia are included;
rounded corners, adhesive shear, printed-part flex, tread compression, and
manufacturing variation are not.

## Policy and rewards

The actor remains a bounded Beta policy with a 48-value hardware-reproducible
observation: command, IMU, joint position error, normalized joint velocity, and
previous action. No simulator-only shoe, force, or body-velocity value reaches
the actor. The privileged critic retains base velocity, height, and four foot
contacts.

V17 keeps V16's forward-speed and sustained-progress objective and adds:

- action second-difference penalty for smoother target acceleration;
- stronger action-rate and joint-speed penalties;
- planted-foot tangential slip penalty;
- touchdown overload penalty above 18 N;
- a small reward for touchdowns after at least 100 ms of swing;
- stronger lateral-velocity and yaw-rate penalties to address V16 drift.

### V17a to V17b reward revision

V17a continued for 500 PPO iterations (4,096,000 new simulator steps) from the
25-iteration shoe-transfer checkpoint. It stayed smooth and usually fall-free,
but stable batches plateaued around 0.01 to 0.02 m/s and continued to report
stall windows. Faster exploratory episodes reached roughly 0.04 m/s but tipped.

V17b therefore adds a normalized instantaneous forward-progress reward of 1.0,
raises the two-second sustained-progress weight from 0.75 to 2.25, raises the
sustained-stall penalty from 0.50 to 1.25, and doubles the termination penalty
from 100 to 200. The change supplies a useful reward gradient away from the
stable standing solution while making a fast fall more expensive. Shoe physics,
observations, action limits, and smoothness terms are unchanged.

### V17c straightness refinement

Deterministic V17b evaluation showed that more progress did not automatically
mean a better gait: late checkpoints either fell or accumulated roughly 0.4 to
0.6 m of lateral displacement in 30 seconds. V17c resumed from the stable V17b
`model_800.pt` and added squared command-frame lateral-displacement and body-tilt
costs. Its termination penalty increased from 200 to 350. This reduced lateral
excursions during training, but later checkpoints either became slower or
reintroduced falls, so none displaced `model_800.pt` as the selected policy.

## Training record

The initial 25-iteration launch found a stale pre-refactor USD path and stopped
before physics or PPO began; it produced no checkpoint. The asset was corrected
to `simulation/exports/isaac/quadruped_robot_floating.usdc`, then the following
runs completed on one RTX 5090 with 128 environments:

| Stage | Run directory | New PPO iterations | New control steps | Result |
| --- | --- | ---: | ---: | --- |
| shoe transfer | `drobot_commanded_walk_forward_v17_rectangular_smooth_direct/2026-08-15_16-25-39_manual-headless` | 25 | 204,800 | `model_274.pt` |
| V17a smooth continuation | `drobot_commanded_walk_forward_v17_rectangular_smooth_direct/2026-08-15_16-28-36_manual-headless` | 500 | 4,096,000 | `model_773.pt`; stable but slow |
| V17b progress continuation | `drobot_commanded_walk_forward_v17b_rectangular_progress_direct/2026-08-15_16-39-30_manual-headless` | 300 | 2,457,600 | `model_1072.pt`; faster candidates, some falls/drift |
| V17c straightness refinement | `drobot_commanded_walk_forward_v17c_rectangular_smooth_direct/2026-08-15_16-54-46_manual-headless` | 300 | 2,457,600 | `model_1099.pt`; endpoint rejected |

Total exploration after V16 was 9,216,000 new control steps. V17a, V17b, and
V17c required 592.84, 357.52, and 346.84 seconds respectively; the short
transfer run is excluded from those timing figures.

The first transfer used:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -Iterations 25 -NumEnvs 128 -Seed 4817
```

The V17b and V17c continuations used explicit checkpoints to prevent an
unreviewed endpoint from being selected automatically. For example, V17c used:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -Iterations 300 -NumEnvs 128 -Seed 4831 `
  -Checkpoint .\logs\rsl_rl\drobot_commanded_walk_forward_v17b_rectangular_progress_direct\2026-08-15_16-39-30_manual-headless\model_800.pt
```

## Checkpoint selection

Selected V17b `model_800.pt` completed all three deterministic 30-second
episodes. Its measured means were:

- 0/3 falls;
- 0.3486 m forward displacement;
- 0.0249 m absolute lateral displacement;
- 0.0059 m/s final five-second speed;
- 0.9615 stalled-window fraction;
- approximately 0.0009 action saturation.

It is packaged as
`simulation/isaac/models/parallel-walking-v17-rectangular/model_800.pt` with
SHA-256
`EED2DBE61A28E7632A4CDD5F2B097C729CE9FD334FFEB2BD9EECB0F08ACB6B6A`.

The alternatives were rejected for concrete reasons:

| Checkpoint | Falls | Mean forward | Mean absolute lateral | Decision |
| --- | ---: | ---: | ---: | --- |
| V17b `model_800.pt` | 0/3 | 0.3486 m | 0.0249 m | selected |
| V17b `model_1000.pt` | 0/3 | 0.5904 m | 0.5677 m | excessive diagonal drift |
| V17b `model_1072.pt` | 1/3 | 0.6418 m | 0.4200 m | fall and drift |
| V17c `model_825.pt` | 0/3 | 0.4010 m | 0.1075 m | worse heading retention |
| V17c `model_975.pt` | 1/3 | 0.3036 m | 0.0502 m | fall and slower progress |

The final 30-second review was recorded with:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v17-rectangular\model_800.pt `
  -Command forward -NoTimeLimit -RecordSeconds 30 -Seed 4822
```

The tracked clip is
`simulation/reviews/parallel-walking-v17-rectangular-model800-30s.mp4`; its
SHA-256 is
`A6FE6AA682DADF79522D6EC93A95779C2311CF79159C2C69EAC23A09B7517B4D`.

## Acceptance and limitations

A checkpoint should be selected from uninterrupted 30-second trials, not PPO
return alone. Report falls, forward distance, final five-second speed, lateral
displacement, stall windows, and action saturation, and visually inspect the
recording for sole drag, impacts, and body oscillation.

The selected policy prioritizes balance, low drift, and smooth bounded action,
but it does not meet the 0.15 m/s command and spends most five-second windows
below the sustained-speed threshold. It is a useful safe simulation baseline,
not a finished fast walking policy. The faster optimization branch demonstrated
that progress is possible with the new shoes, while also showing that the next
iteration needs a stronger gait/symmetry prior or heading-observable policy
rather than more weight on forward displacement alone.

The CAD shoe has not yet been physically measured with its final tread, so its
mass, inertia, friction, and compliance remain estimates. Hardware trials must
start suspended or guarded with an emergency stop and current/temperature
monitoring. A successful simulator run is not approval for autonomous floor
testing.
