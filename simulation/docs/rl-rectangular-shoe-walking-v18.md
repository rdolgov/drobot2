# V18 coordinated rectangular-shoe walking

Date: 2026-08-15

## Goal and result

V18 replaces the V17 stationary bent-knee behavior with sustained forward
walking on the latest 100 x 60 mm flat rectangular shoes. The selected policy,
`model_299.pt`, walks for the full 30-second evaluation without a reset, fall,
or stalled five-second window. It uses every leg in a diagonal-pair gait and
keeps action smoothing in the training objective.

The accepted artifact is
`simulation/isaac/models/parallel-walking-v18-coordinated/model_299.pt`.
Its SHA-256 is
`E17371E8E64D92A2F6FB638112000BC68BDF2B61D509945186CEB812F6BECDAB`.

## Policy contract

The actor is a bounded Beta policy with two 256-unit ELU layers and 12 actions.
Its 50 deployable observations are:

- commanded forward/lateral/yaw velocity: 3;
- gait-clock sine and cosine: 2;
- IMU angular velocity, projected gravity, and linear acceleration: 9;
- joint-position error, normalized joint velocity, and previous action: 36.

The clock is deterministic and can be reproduced on hardware. The actor does
not consume privileged simulator velocity, height, or contact data.

## Coordinated gait

The reference gait has a 0.80 s period, 65% stance duty, 80 mm stride, 25 mm
foot lift, and a 0.60 s startup ramp. Front-left and rear-right share a phase;
front-right and rear-left share the opposite phase. The overlap around pair
transitions provides short four-foot support intervals.

The reference is deliberately a reward target rather than a hard controller
constraint. A trial that averaged joint actions within each diagonal pair was
rejected: it immediately reduced speed to approximately zero and produced no
qualified touchdowns. Mirrored joint frames and unequal instantaneous loading
require small leg-to-leg corrections. Coordination is therefore enforced by
the gait phase, reference pose, scheduled stance/swing contacts, and per-leg
touchdown monitoring.

## Reward configuration

All values below are applied before the global reward scale of 0.10.

| Objective | Weight |
| --- | ---: |
| commanded forward-velocity tracking | +4.00 |
| instantaneous normalized progress | +2.00 |
| two-second sustained progress | +4.00 |
| sustained speed below 0.08 m/s | -4.00 |
| analytic gait-reference tracking | +2.00 |
| scheduled stance / scheduled swing | +0.50 / +1.50 |
| upright / alive | +1.00 / +0.10 |
| lateral velocity / displacement | -5.00 / -8.00 |
| yaw rate / body tilt | -5.00 / -1.50 |
| action rate / second difference | -0.03 / -0.12 |
| action saturation / light diagonal action similarity | -0.50 / -0.10 |
| support-foot slip / touchdown impact | -0.30 / -0.05 |
| fall termination | -350.00 |

The forward terms dominate standing. The two-second rolling term and stall
penalty prevent a reset-dependent burst from being scored as walking. Action
rate, second-difference, velocity, slip, impact, and servo target-rate limiting
discourage jerks.

## Training history

All main runs used 128 environments and 64 policy steps per PPO iteration.

| Run | Seed | Checkpoint range | Outcome |
| --- | ---: | --- | --- |
| `2026-08-15_17-16-49_manual-headless` | 5801 | fresh to `model_299.pt` | selected; stable motion near 0.15 m/s |
| `2026-08-15_17-26-01_manual-headless` | 5802 | 299 to 375 | continued curriculum; not selected |
| `2026-08-15_17-28-34_manual-headless` | 5803 | 375 to 450 | drift and saturation increased |
| `2026-08-15_17-30-59_manual-headless` | 5804 | 450 to 625 | all legs active, but drift remained worse |
| `2026-08-15_17-35-29_manual-headless` | 5805 | 625 onward | rejected hard-pairing diagnostic; robot froze |

The fresh selected run performed 300 iterations, or 2,457,600 simulated policy
steps, in 435.39 seconds. Later checkpoints were not accepted merely because
they were newer: `model_450.pt`, for example, covered only 4.291 m in 30 seconds,
slowed to 0.127 m/s in the final window, drifted 1.596 m laterally, and saturated
17.7% of actions.

Representative training command:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -Fresh -Iterations 300 -NumEnvs 128 -Seed 5801
```

## Deterministic selection evaluation

The selected checkpoint was run three times for 30 uninterrupted seconds with
zero reset noise and a 0.15 m/s forward command:

- mean forward distance: 4.7890 m;
- mean speed: approximately 0.160 m/s;
- mean final five-second speed: 0.1600 m/s;
- falls: 0/3;
- stalled five-second windows: 0%;
- mean absolute lateral displacement: 0.5705 m;
- action saturation: 6.28--6.38%;
- mean absolute normalized action rate: 0.110--0.111 per step;
- mean absolute normalized action second difference: 0.135--0.137 per step;
- scheduled contact agreement: 79.6--95.1%, depending on leg;
- touchdowns per leg per trial: front-left 38, front-right 38--39,
  rear-left 48--51, rear-right 38--39.

The nonzero, repeated touchdown counts for every leg are the acceptance check
for four-leg operation. The unequal counts reflect brief rear-left contact
chatter, not an inactive leg. Lateral drift remains the next improvement target.

Evaluation command:

```powershell
& .\simulation\isaac\rl\parallel_walking\evaluate_walking_sustained.ps1 `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v18-coordinated\model_299.pt `
  -Seconds 30 -Episodes 3
```

## Review recording

The 1,800-frame, 60 fps recording is
`simulation/reviews/parallel-walking-v18-coordinated-model299-30s.mp4`.
Its SHA-256 is
`CB3F11AFF54C2C2B1FFE0FD15C793C7E44F3CCC992C860CF3A0F586F3F8B07C4`.

Recording command:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -Command forward -CommandSet forward `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v18-coordinated\model_299.pt `
  -RecordSeconds 30 -NoTimeLimit
```

## Deployment note

Hardware inference must supply the two gait-clock values in the documented
observation order and reset the phase when a walking command begins. Begin with
the robot unloaded or on a safety tether and validate joint directions and
limits before any untethered hardware trial; simulation acceptance does not
replace that physical check.
