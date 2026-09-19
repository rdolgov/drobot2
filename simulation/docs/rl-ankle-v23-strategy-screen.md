# V2.3 stair strategy and reward screen

September 13 request: experiment with turning, sideways movement, and lifting
and planting one leg first, to identify more promising reward combinations.
Status: all fourteen PPO jobs completed. Seven strategies received two seeds
each, totaling 14,680,064 training transitions and 11,861 completed episodes.
No separate tests, evaluation rollouts, videos, UI checks or hardware
operations were run.

## Results and promising directions

The gradual-turning reward is the strongest lead for a first tread placement:
it produced the experiment's only qualifying landing, with the front-right
foot on the staircase with 20 mm risers. This occurred once in seed 2601's 563 completed
episodes and zero times in seed 2602's 561 episodes. It did not repeat and is
not a reliable stepping policy. Across all fourteen jobs there were no
front-pair, rear-pair or all-feet-first-tread milestones, and no full climbs.

The table gives equal-weight means of the two seeds' final 256 completed
training episodes. The 0.2% placement figure represents that single event in
the combined final windows; it must not be read as evidence of repeatability.

| Strategy | Held placement | Supported lift | Falls | Forward speed | Mean peak supported yaw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Free exploration | 0.0% | 93.2% | 67.2% | 18.5 mm/s | 6.2 degrees |
| 45-degree objective | 0.0% | 58.8% | 49.8% | 10.4 mm/s | 6.2 degrees |
| 90-degree sideways objective | 0.0% | 39.6% | 62.1% | 13.7 mm/s | 6.7 degrees |
| One-leg placement sequence | 0.0% | 20.1% | 16.6% | 1.4 mm/s | 4.9 degrees |
| Sequence + weight transfer | 0.0% | 16.8% | 9.2% | -1.4 mm/s | 4.9 degrees |
| 45 degrees + sequence + weight transfer | 0.0% | 38.5% | 21.3% | 4.6 mm/s | 9.6 degrees |
| 90 degrees with gradual turning reward | 0.2% | 10.4% | 5.3% | 1.3 mm/s | 13.4 degrees |

The gradual reward improved mean peak supported heading change from 6.7 to
13.4 degrees compared with the original 90-degree objective. Its individual
seed values were 18.8 and 7.9 degrees; these do not establish a learned
90-degree sideways gait. Its low speed and low lift rate also limit what can
be inferred from the isolated landing.

The 45-degree objective combined with placement sequencing and weight transfer
is another candidate for developing a supported first-foot lift. Seed 2602
finished with 55.1% supported-lift episodes and 5.5% falls in its final window.
Across that whole run, front-left lifts occurred in 169 episodes, versus 17
front-right, 26 rear-left and 27 rear-right. The result was seed-dependent:
seed 2601 finished at 21.9% supported lifts and 37.1% falls. Neither seed
planted on a tread.

The placement sequence and weight transfer consistently improved survival,
but often led to very slow or backward motion. Conversely, free exploration's
high lift rate mostly came from the rear-right foot: 1,228 and 969 episodes
in the two runs. High lift frequency alone is not evidence of a useful gait.

The next training candidate is gradual turning for landing discovery, with
the combined turn/placement/weight-transfer policy as a possible starting
point for the supported-lift subskill. Those are tentative follow-up
directions, not demonstrated stair-climbing solutions. No additional training
was launched after the fourteen jobs.

[Full per-seed comparison](../reviews/ankle-v23-strategy-screen/20260913-two-seed/COMBINED_RESULTS.md)
and [machine-readable metrics](../reviews/ankle-v23-strategy-screen/20260913-two-seed/combined_comparison.json)
retain all counts and seed ranges. Candidate checkpoints:
[gradual turning, seed 2601](../isaac/models/ankle-v23-dense-turn/20260913-two-seed/sideways_dense-seed2601/model_255.pt)
and [turn + sequence + transfer, seed 2602](../isaac/models/ankle-v23-strategy-screen/20260913-two-seed/oblique_place-seed2602/model_255.pt).
Each archive includes effective configuration and a source/asset contract.
The saved hash records agree for all shared source and asset paths across
the fourteen jobs.

### Source-based follow-up hypotheses

The common progress reward is `30 * supported_forward_delta`, capped per
control step, while a fall costs 2. For a hypothetical supported 0.30 m advance
without backward motion or clipping, those two terms contribute +9 and -2
before all other rewards and costs. This could reward risky advance followed
by a fall; the aggregate telemetry does not prove that the policy exploits it.
Rewarding retained progress or canceling progress credit at failure is a
candidate follow-up. It is not changed halfway through this matched screen.

The waypoint/load-transfer arms can earn dense shaping rewards without a
completed placement. Their low speed and improved survival are consistent
with learning balance before placement, or settling into a stationary local
optimum. These are hypotheses, not demonstrated causal explanations.

## Focused turning follow-up

A seventh strategy completed separately in
`simulation/isaac/rl/ankle_v23_dense_turn/`, after the original
twelve jobs, with the same two seeds, 256 iterations, 64 environments and
64 transitions per environment per iteration. This adds 2,097,152 transitions
for a completed total of 14,680,064 across fourteen jobs. The original
screen's source and six strategy definitions are not modified.

At a 90-degree error, the original heading score
`exp(-(heading_error / 0.35)^2)` is approximately 1.8e-9. That narrow reward
provides very little change for small exploratory turns far from the goal.
The follow-up replaces it with `cos(heading_error)` at the same weight 2 and
adds weight 0.5 for tracking commanded angular speed with a Gaussian width of
0.25 rad/s. Both terms require supported, upright motion and retain the same
five-second heading-target ramp and final translational stopping period.

This changes the turning reward only; initial heading, joint controller,
camera, stairs, physical limits, PPO, observation channels and milestone
metrics stay the same. It still requires learning the turn from a forward
start. It does not assume a pre-rotated robot or a swiveling camera.

Run artifacts are under
`simulation/reviews/ankle-v23-dense-turn/20260913-two-seed/`, with checkpoints
under `simulation/isaac/models/ankle-v23-dense-turn/20260913-two-seed/`.

## Common experimental setup

The V2.3 R3 robot, 16 active actuators, existing torque/speed/travel caps,
85-100% motor randomization, three 20 mm risers with 250 mm treads, reset
position variation and sparse D405 observation remain as in the
[preceding run](rl-ankle-v23-stairs-d405.md). Camera noise, latency, range
limits and missing-pixel masks are unchanged. It remains a geometric depth
proxy without robot self-occlusion or extra D405/mount mass.

All seven new arms share a neutral hip/knee reference with measured-joint ankle
leveling and identical larger residual ranges. This removes the former
forward-only crawl trajectory that would resist sideways movement. Heading
and lateral deviation near the center of the stairs are no longer universally
penalized. Physical joint/torque/target-rate bounds remain enforced. All arms
get the same four-second period with zero commanded translational velocity
at the end of a 36-second episode. The heading command still points toward
its goal; this period is not an angular-stop test.

The 469 actor inputs are the prior proprioception/depth channels plus sine
and cosine of relative reset heading (an IMU/heading-estimator signal). The
485 critic inputs also contain privileged simulator quantities. Terrain
geometry, foot contacts and whole-robot center of mass are reward/critic
information, never direct actor observations or target-joint commands.
The actor is a fresh GRU for every run. No old failed policy is resumed.

Velocity commands are expressed in the body frame for progress toward the
stairs' world +X direction. Turning targets ramp over five seconds. The
one-foot sequence is front-left, front-right, rear-right, rear-left, with eight
seconds per requested foot. Its timing depends only on elapsed time; it does
not use privileged contacts to switch actor commands. A lift/advance/lower
waypoint shapes reward but never moves a foot or writes joint positions.

| Arm | Heading objective | Placement shaping | Load-transfer shaping |
| --- | --- | ---: | ---: |
| free | Unrestricted | 0 | 0 |
| oblique | Turn toward 45 degrees | 0 | 0 |
| sideways | Turn toward 90 degrees, then move laterally toward stairs | 0 | 0 |
| place_first | Unrestricted | 3 | 0 |
| place_transfer | Unrestricted | 3 | 2 |
| oblique_place | 45 degrees | 3 | 2 |

Turning has weight 2 in its three arms. The camera stays forward-facing on
the body during turns, so sideways motion can reduce its view of the stairs.
The recurrent actor must retain earlier camera evidence; an invisible terrain
map or swiveling camera is not silently added.

Common rewards include supported forward advance, velocity tracking,
three-foot-supported lift, genuinely held tread placements, upright posture,
contact slip, power, action changes and falls. There is an inactivity penalty.
The load-transfer terms use the mass-weighted whole-robot COM projection in
the other three feet's support triangle, actual upward foot forces, unloading
the selected foot, and adequate support load. This prevents airborne or
unsupported "unloading" from earning the intended load-transfer reward.

This compares strategy/reward combinations within a common new controller.
It is not a pure reward-only ablation against the previous crawl controller.

## Budget and comparison criteria

Each of six strategies gets two independent seeds (2601, 2602), 64 parallel
environments, 256 PPO iterations, and 64 transitions per environment per
iteration. That is 1,048,576 transitions per job and 12,582,912 total. Two jobs
run concurrently on the local GPU. All use the same GRU/PPO configuration,
including gamma 0.999, GAE lambda 0.97 and entropy coefficient 0.005.

Comparison uses shared physical milestones, not each arm's differently shaped
reward total. Final comparison values are equal-weight means of the two seeds'
last 256 completed training episodes, with individual counts/ranges retained.

- A supported lift requires >12 mm clearance, >80 ms air time, three other
  support feet and an upright body. Landing back on the floor consumes its
  qualification; an old lift cannot qualify a later dragged placement.
- A tread placement then requires 250 ms contact, all traction corners within
  the common contact tolerance, slip <15 mm/s, upright posture and at least
  two other support feet. Metrics identify each of the four feet separately.
- Front/rear pairs and all-feet-first-tread milestones require simultaneous
  qualified contact; historical separate contacts cannot count as a pair.
- Full climbs retain the common three-step/top-platform one-second hold gate.
- Supported base advance, heading change and sideways motion are diagnostic
  outcomes; none substitutes for a climb. All timeout/fall episodes stay in
  the denominator. Unfinished final episodes are not extended.

Reported peak yaw is the mean of each completed episode's greatest absolute
heading change while upright with at least two foot contacts. The sideways
fraction is accumulated absolute body-Y speed divided by accumulated absolute
body-X plus body-Y speed under that same support gate. It can include sway or
small motions and does not establish a sideways gait. A heading objective is
an attempted strategy, not evidence that its requested angle was achieved.

Rank first by full climbs, all-feet-first-tread support, rear-pair placement,
any held placement, number of unique placed feet, supported lift and then
supported base advance. Any lead is preliminary training evidence with only
two seeds, not a generalization or hardware feasibility result.

## Artifacts and launch

Code: `simulation/isaac/rl/ankle_v23_experiments/`.
Run state, logs, source/asset snapshots and comparison reports:
`simulation/reviews/ankle-v23-strategy-screen/20260913-two-seed/`.
Archived checkpoints, configs and contracts:
`simulation/isaac/models/ankle-v23-strategy-screen/20260913-two-seed/`.

```powershell
& ./cad/.venv/Scripts/python.exe simulation/isaac/rl/ankle_v23_experiments/run_screen.py --output C:/Users/roman/Documents/dev/drobot2/simulation/reviews/ankle-v23-strategy-screen/20260913-two-seed --iterations 256 --num-envs 64 --workers 2
```

The launcher reserves workflow state exclusively and will not duplicate an
existing run. It waits for actual worker completion, archives each model, and
writes `comparison.json` and `RESULTS.md` once all twelve jobs finish. It does
not start a separate evaluator, renderer, hardware session or hosted review.

The completed focused follow-up used:

```powershell
& ./cad/.venv/Scripts/python.exe simulation/isaac/rl/ankle_v23_dense_turn/run_screen.py --output C:/Users/roman/Documents/dev/drobot2/simulation/reviews/ankle-v23-dense-turn/20260913-two-seed --iterations 256 --num-envs 64 --workers 2
```

The combined report was generated from saved telemetry only, without loading
or exercising a policy:

```powershell
& ./cad/.venv/Scripts/python.exe simulation/isaac/rl/ankle_v23_dense_turn/combine_results.py --screen simulation/reviews/ankle-v23-strategy-screen/20260913-two-seed --followup simulation/reviews/ankle-v23-dense-turn/20260913-two-seed
```

A later [research-informed continuation study](rl-ankle-v23-research-followup.md)
compared progress shaping and pre-contact terrain penalties against an unchanged
reward control. Its six additional runs completed 9,437,184 transitions and
4,444 episodes without a held landing or climb. See the
[follow-up results](../reviews/ankle-v23-research/20260913-potential-edges/SUMMARY.md)
for the matched comparison; its warm starts and budget differ from this original screen.
