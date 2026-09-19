# V2.3 active ankle stair-climbing RL with D405 depth

September 13, 2026 request: train stair climbing on the new movable ankle/foot
robot using reinforcement learning and the previously discussed RealSense D405.
Training is authorized for this task. No independent evaluation, test suite,
UI validation, recording, hardware connection or motion check is included.

Status: **first stage completed; stair climbing was not demonstrated.** The
1,000-iteration, 128-environment PPO run with seed 2501 finished all 8,192,000
transitions and saved `model_999.pt`. PPO reported 2,069.63 seconds (34 min
30 sec) of training. The checkpoint was written September 13 at 07:58 PDT.
The curriculum stopped at 20 mm because its progression gate was not met.
No 40-180 mm stage was run.

## Completed training result

There were **0 full climbs in 3,089 completed training episodes**. Two early
episodes achieved simultaneous front-pair contact on the first or a higher
tread. No episode achieved the paired rear-foot placement milestone or
simultaneous all-feet support on the first or higher treads. These metrics do
not establish that an individual rear foot never moved or never touched a step.

| Last 512 completed training episodes | Result |
| --- | ---: |
| Full three-step climbs | 0% |
| Front-pair placement milestone | 0% |
| Rear-pair placement milestone | 0% |
| Falls | 21.68% |
| Mean forward speed | 0.00449 m/s |
| Mean contact-foot slip speed | 0.00372 m/s |
| Mean absolute ankle angular velocity | 0.336 rad/s |
| Valid D405-proxy depth samples | 48.85% |
| Rate-limited joint targets | 34.37% |

The policy became slower and fell less often, without learning the required
stair transfer. These are nonstationary stochastic training episodes, not an
independent evaluation of the final deterministic policy. Active episodes at
the end of training were not extended or treated as successes. This result
does not prove the mechanism cannot climb, or isolate the camera versus
control contribution. The next useful training change should address staged
rear-foot unloading and support transfer; it was not run in this experiment.

- [Concise results](../reviews/ankle-v23-stairs-d405/20260913-seed2501/RESULTS.md)
- [Detailed results, checkpoint and source hashes](../reviews/ankle-v23-stairs-d405/20260913-seed2501/training_summary.json)
- [Archived final checkpoint](../isaac/models/ankle-v23-stairs-d405/20260913-seed2501/rise-020mm-model_999.pt)
- [Archived configuration and contract](../isaac/models/ankle-v23-stairs-d405/20260913-seed2501/README.md)

The Isaac worker exited with Windows exit code 0. Its shared Omniverse Hub
child inherited stdout, leaving the outer supervisor waiting for pipe EOF
after training had finished. The worker's actual exit code was read before
stopping only the blocked supervisor. Completion bookkeeping was recovered
from that exit status, the final-iteration log, the training-return line and
the saved checkpoint. Original bookkeeping, the observed exit record and
recovery script remain alongside the logs. The shared Hub was left running;
no training was repeated. Future supervision now drains output through a
daemon reader and stops waiting after worker exit. That post-run supervisor
change received source inspection only; no separate verification was run.

## Robot and control

This task reuses the existing V2.3 R3 retained lower leg and solid-sole imported
asset at `simulation/exports/isaac/ankle_v23_stride/active/robot.usdc` (estimated
3.444 kg). The 16 actions control hip abduction, hip flexion, knee and ankle
pitch on each leg. It does not revert to the earlier rigid-foot stair model.
The ankle moves the foot as an articulated link; there is no invented toe joint.

The shared mechanics retain the 0.8826 Nm conservative effort cap, coupled
torque/speed envelope, joint travel bounds, 60/100/100/100 deg/s target-rate
limits by joint type, 120 Hz physics and 60 Hz commands. Motor strength and
target speed vary independently between 85% and 100%. A slow crawl reference
is corrected by learned filtered residuals, including independently learned
ankle corrections. Reference ankle leveling uses measured hip/knee and body
pitch. No analytic terrain heights enter the actor or the reference controller.

The earlier V2.3 walking study did not produce qualifying rear-foot steps.
This is a fresh recurrent actor and critic, not a claim that that policy could
already climb. Rewards include unloading/clearance, tread landing after a
qualified lift, sole contact, slip, forward progress, posture and motor work.

## D405 observation

Manufacturer data: [RealSense D405 specifications](https://www.realsenseai.com/products/stereo-depth-camera-d405/)
give an ideal depth range of 0.07-0.50 m, 87 x 58 degree depth FOV, no onboard
IMU and 60 g camera mass. The robot retains its separate body IMU.

This first feasibility model uses 16 x 12 sparse pinhole depth samples as a
low-dimensional proxy for a reduced depth image. Optical-axis depth and a
separate validity mask are sampled at 30 Hz, delayed by one 33.3 ms frame and
held between samples. Out-of-range samples are invalid, never filled from an
oracle height map. The GRU actor can retain past front-camera observations as
rear feet approach a previously observed edge. Its 467 inputs contain 82
proprioceptive/controller values, 192 depths, 192 validity values and frame age.
The privileged critic has 483 inputs, adding body velocity/height, foot load,
terrain under feet and achieved foot progress for training only.

The provisional low front mount is `(0.14, 0, 0.02)` m relative to base,
pitched down 40 degrees. The earlier high 25-degree mount would leave much
of the close ground outside the chosen 0.50 m observation limit. At nominal
body height, the center-column bottom ray reaches the floor at approximately
0.35 m optical depth and 0.29 m ahead of the base. This is a geometric estimate,
not a rendered frustum or a verified mount design.

Depth uncertainty assumptions: +/- (2% of depth + 1 mm) uniform sample noise,
up to +/- 2 mm per-episode bias, 5% pixel dropout, 2% whole-frame dropout,
and +/- 3 degree / 5 mm mounting variation. These are experiment settings,
not a calibrated sensor error model. Ray intersections honor occlusion by
the same solid stair boxes used in physics. This is not a stereo renderer:
robot self-occlusion, texture/lighting failures, motion blur and real camera
calibration are not modeled. It does not connect to a physical camera.

The existing robot asset's camera mass is retained. Extra D405, adapter and
cable mass/inertia are not added in this pilot, so even training success would
require a payload-corrected model before a hardware feasibility claim.

## Bounded curriculum and outcome criteria

The first stage uses three 20 mm risers, 250 mm treads and a 1.2 m top
platform. The first riser is 340 mm ahead of the nominal reset base. Every
episode starts before the stairs with +/-25 mm fore/aft and +/-10 mm lateral
jitter. Episodes last up to 60 seconds. Commands are 0.025-0.035 m/s.

Each stage gets 1000 PPO iterations, 128 environments and 64 transitions per
environment per iteration: 8,192,000 transitions. Seed 2501 starts fresh.
Rise stages are 20, 40, 60, 100, 140 and 180 mm. A later stage starts only if
the previous completed stage has at least 512 recent completed episodes with
at least 70% full-stair successes and at most 10% falls. Otherwise the
experiment stops and reports that learning has not established feasibility.
Later stages initialize the actor from the preceding stage with a fresh
critic, optimizer and iteration count. These are training curriculum gates,
not held-out reliability estimates.

A full success requires all four soles to be on the highest platform, with
upward contact load, all four traction-patch corners within 8 mm of the same
horizontal surface, upright/non-fallen body, and body speed below 0.08 m/s,
continuously for one second. A side collision, dragged toe, front-foot-only
placement or a historical combination of different contacts cannot count as
full-stair success. The report separately records both front feet on the first
or higher tread, both rear feet, and simultaneous all-feet first-tread support.
Fall and timeout episodes remain in the denominator.

## Run and artifacts

Implementation: [`ankle_v23_stairs/`](../isaac/rl/ankle_v23_stairs/).
Persistent curriculum state, logs, exact commands and source/asset snapshots
are under `simulation/reviews/ankle-v23-stairs-d405/20260913-seed2501/`.
Checkpoints and effective configurations go to
`logs/rsl_rl/drobot_ankle_v23_stairs_d405/`.

```powershell
& ./cad/.venv/Scripts/python.exe simulation/isaac/rl/ankle_v23_stairs/run_curriculum.py --output C:/Users/roman/Documents/dev/drobot2/simulation/reviews/ankle-v23-stairs-d405/20260913-seed2501 --iterations 1000 --num-envs 128 --seed 2501
```

Do not duplicate a running workflow. The launcher rejects an existing output
directory with workflow state. Individual job supervision uses a file lock
and records final checkpoint hashes after the training process exits.
No policy export or deployment is part of this run.
