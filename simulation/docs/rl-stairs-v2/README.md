# Stair-climbing reinforcement learning v2

`Drobot-Quadruped-Stairs-v2` corrects the failed long-approach assumptions in
the original stair experiment. It is a separate configuration, world, output
directory, model contract, and documentation set. The v1 checkpoints are kept
as diagnostic evidence and are not resumed into v2.

## Why v2 exists

The v1 trainer was stopped at 964,608 steps. A deterministic five-episode
level-1 audit of its 950k checkpoint found:

| Metric | V1 result |
| --- | ---: |
| First-stair completion | `0 / 5` |
| Highest stair reached | `0` in every episode |
| Mean forward displacement | `0.735 m` |
| Lateral displacement | `0.410-0.504 m` |
| Corridor exits | `3 / 5` |
| Mean shaped return | `1,225` |

The base began approximately `1.15 m` before the first riser and never reached
it. High return came from walking and staying upright, not climbing. The v1
policy also lacked lateral-position and absolute-heading observations, while
its timestep curriculum advanced to harder goals without mastering step one.

The audit is stored locally at
`simulation/isaac/output/rl/ppo-stairs-v1/evaluation-950k-level1-5ep.json`.

## V2 corrections

| Contract | V2 behavior |
| --- | --- |
| Reset X | `0.18-0.24 m`, only `0.31-0.37 m` before the riser |
| Reset Y/yaw | `+/-0.01 m`, `+/-0.75 degrees` |
| Observation | 60 values: walking 48 + terrain 8 + goal 1 + lateral 1 + heading sine/cosine 2 |
| Physical climb reward | `150 x` actual base `delta-z` |
| Terrain reward | `40 x` analytic terrain-height change |
| Path control | stronger centerline, lateral-velocity, and heading penalties |
| Corridor | failure at `|Y| > 0.30 m` |
| Exploration | transferred policy log standard deviation is reset to `-0.70` (`sigma ~= 0.50`) |
| Curriculum | advances only after at least 70% success in a recent mastery window |
| Early gate | at 100k steps, require at least three physically elevated first-step reaches and a 2% reach rate |
| Stagnation gate | abort after 300k steps without a new stair level or successful episode |

“Physically elevated first-step reach” means the episode reached analytic step
index 1 and the base rose at least `0.02 m` above its settled reset height.
Crossing a world-X threshold without gaining height is not counted.

## Source and output separation

- Configuration:
  `simulation/isaac/rl/stairs/quadruped_stairs_v2.yaml`
- Generated world:
  `simulation/exports/isaac/quadruped_robot_stairs_v2_world.usda`
- Smoke output:
  `simulation/isaac/output/rl/ppo-stairs-v2-smoke/`
- Full output:
  `simulation/isaac/output/rl/ppo-stairs-v2/`
- Live watchdog:
  `simulation/isaac/output/rl/ppo-stairs-v2/progress_watchdog.json`

See [Architecture](architecture.md) for observation/reward/curriculum details
and [Training and monitoring](training.md) for exact commands and abort
interpretation. See [Stair perception and staged learning](perception-plan.md)
for the sensor decision, current camera blind region, policy interface, and
next training curriculum.

## Validation recorded on 2026-07-27

The v2 world passed static Isaac validation with four collision layers, no
stair rigid bodies, and the expected robot/sensor counts. Its SHA-256 is
`c17fc5b8ab4d636a29f9bbb99da2bc5bdb843d0a8ea8a9a75745403e206d989a`;
the stabilized v2 YAML SHA-256 is
`2631c570f2e991537da739d53f5ac00b8057eae94c66f8af00a046c922936e23`.

The 512-step transfer smoke passed in 16.481 seconds:

- 60-value observation and 12-value action contracts verified;
- 166,169 policy parameters;
- 11 tensors copied exactly and two input matrices expanded from 48 to 60;
- no transfer tensors skipped;
- policy log standard deviation reset to `-0.70`;
- schema-2 model, world, dependency, and PPO contracts passed;
- progress-watchdog report path created.

A deterministic smoke reload also passed manifest and algorithm verification.
It started visibly beside the first riser but did not climb; a 512-step smoke
model is not behavioral evidence.

The first full v2 attempt was preserved at 16k after PPO KL reached 4.32. The
configuration was corrected to use a `0.00005` learning rate and `0.03`
target-KL cap. The replacement run kept KL controlled and was stopped by user
request at the 50k review point:

| Metric | Stabilized v2 result at stop |
| --- | ---: |
| Completed episodes | `107` |
| Physically elevated first-step episodes | `0` |
| First-step successes | `0` |
| Maximum actual base elevation | `0.018115 m` |
| Highest analytic terrain index | `2` |

The analytic index only means the base crossed an `X` boundary; it is not
accepted as a climb without physical elevation. The 50k checkpoint and manual
stop report are retained under `simulation/isaac/output/rl/ppo-stairs-v2/`.
This result shows obstacle engagement but no learned climbing behavior.

## Real-stair feasibility block

Stair PPO remains paused. A separate scripted inverse-kinematics experiment
tested one `280 mm` tread at `100`, `140`, `180`, and `196 mm` riser heights
under the rated servo torque. Although every ideal foot target fit the URDF
joint limits, none of the floating-base trials cleared the riser or contacted
the tread. The `100 mm` case achieved only `16.0 mm` of foot lift, and the
taller cases lost support and tipped into the block.

The report therefore sets `curriculum_authorized` to `false`. Revise the
physical foot/contact model, scripted weight transfer, and actuator/load
margin before returning to the staged perception curriculum. See
[`../stair-feasibility/README.md`](../stair-feasibility/README.md) for the
complete experiment, measurements, commands, and next gates.
