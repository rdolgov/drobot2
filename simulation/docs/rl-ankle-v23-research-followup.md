# Research follow-up: first footholds while turning

Completed: all six runs, 9,437,184 additional transitions and 4,444 completed
episodes, with no held landings or climbs. See the
[results and interpretation](../reviews/ankle-v23-research/20260913-potential-edges/SUMMARY.md).

The preceding 14-job training screen produced one held tread landing, no paired
placements and no stair climbs. The gradual turning reward reduced falls but
did not establish a sideways gait. This follow-up asks whether rewarding progress
toward a foothold, and feedback before unsafe contact, improve that bottleneck.

## Research and selected adaptations

- [Jeon et al., Benchmarking Potential Based Rewards for Learning Humanoid Locomotion (ICRA 2023)](https://arxiv.org/abs/2307.10142)
  compared conventional task rewards with potential differences. Their reported
  gains were mainly robustness to reward scaling, with only marginal convergence
  improvement. Their implementation used a shaping discount of 1. This experiment
  instead uses the PPO discount, 0.999, and a custom first-foothold potential. It is
  an adaptation, not a reproduction or a guarantee of improved learning.
- [Jiang et al., Omnidirectional Humanoid Locomotion on Stairs via Unsafe Stepping Penalty and Sparse LiDAR Elevation Mapping (2026 preprint)](https://arxiv.org/abs/2603.07928)
  provides dense feedback for approaching risers and unsafe edge placement before
  bad contacts. Here, straight stair geometry and the quadruped's actual sole
  corners supply reward-only distances. Their humanoid and LiDAR perception system
  are not reproduced; the forward-facing D405 proxy remains unchanged.
- [Rudin et al., Learning to Walk in Minutes Using Massively Parallel Deep RL](https://arxiv.org/abs/2109.11978)
  supports terrain curricula as another useful direction. Difficulty progression
  is deferred until first-tread landings are repeatable: changing stair height or
  reset distance in this comparison would obscure the reward comparison.

Another candidate is a privileged terrain teacher followed by a depth-based
student. [Miki et al., Learning robust perceptive locomotion for quadrupedal robots
in the wild](https://arxiv.org/abs/2201.08117) trained a teacher with ground-truth
environment information, then a recurrent student by imitation using partial,
noisy observations. Here that would provide a diagnostic separation between
discovering a workable movement and learning to infer terrain from the camera.
That diagnostic interpretation is our proposal; it has not been demonstrated on
this robot. Other later experiments are separating forward stepping from turning
and a curriculum of supported weight transfer followed by single-foot landing.
These are candidate next steps, not completed experiments or established solutions.

## Matched training design

Three arms, each with continuation seeds 2701 and 2702:

| Arm | Change from the previous gradual-turning reward |
| --- | --- |
| continue_turn | None; controls for more training from the same parent actor |
| potential | Replaces forward displacement and sustained lift rewards with a potential difference |
| potential_edges | Same potential change, plus riser-approach and edge-overhang costs |

Every job loads the actor and its observation normalizer from
`simulation/isaac/models/ankle-v23-dense-turn/20260913-two-seed/sideways_dense-seed2601/model_255.pt`.
Critic, optimizer and iteration counter start fresh in every arm. These are two
random continuations of one pretrained actor, not independent pretraining seeds.
Selection of that parent followed the previous screen's single landing, so the
parent itself is a selected result, not evidence of reliable stepping.

Budget: 384 PPO iterations × 64 environments × 64 transitions per environment =
1,572,864 transitions per job; six jobs = 9,437,184 additional transitions.
Two jobs run concurrently. The parent had 1,048,576 training transitions before
this follow-up. No separate policy evaluation, recording, UI check or hardware
operation is part of this experiment.

Robot: V2.3 R3, 16 actions including active ankles, 3.444 kg estimated mass,
0.8826 Nm effort cap with the inherited torque-speed envelope. Controller,
residual range, rate limits, domain randomization, actor/critic architectures,
469/485 observations, 36-second episodes, initial position distribution,
90-degree heading objective and three 20 mm risers are shared across arms.
The final four seconds stop the translational command; this is not a test of
stopping angular motion.

## Exact reward changes

The potential arm removes the old `30 * clipped_forward_delta * supported`
reward and the continuous `0.5 * lift_score * dt` reward. Other terms, including
the supported held-placement bonus, remain unchanged.

It adds `0.999 * Phi(next_state) - Phi(current_state)` each control transition.
Current-state potential is measured before physics, including the first step
after reset. Next potential is zero at a true terminal fall or success, while
time-limit truncations retain it for bootstrapping. This does not justify a claim
that the total objective is policy invariant: two base rewards were also removed.

`Phi` consists of 15 times forward base advance (clipped to 0–1.2 m), plus foot
proximity scores weighted FL/RL/FR/RR = 6/2/6/2. A proximity score is
`clip(1 - remaining_path / 0.4, 0, 1)`. The path goes above the first riser with
35 mm clearance, forward to 70 mm beyond the first edge, then down to its tread.
Once the whole sole is at least 10 mm past the edge, the remaining vertical path
is distance to the tread. The lowest sole corner sets clearance. This first-tread
potential deliberately targets the missing first-landing milestone; it is not a
complete higher-stair objective and can cease to help after the first tread.
The region switch can create a potential jump for a foot crossing below safe
clearance, another reason to compare the extra contact penalties empirically.

The contact-cost arm adds:

- Riser approach: for each foot, the strongest proximity to a riser within 60 mm,
  multiplied by insufficient clearance and positive forward foot speed relative
  to 30 mm/s (clipped at 3). Sum over feet; weight 1.5 per second.
- Edge placement: for each foot, the largest fraction of its X span lying before
  an edge it straddles, gated to within 20 mm vertically of the upper tread.
  This projected-span proxy is not an exact unsupported sole-area calculation.
  Sum over feet; weight 1.0 per second. A high swing across an edge has zero cost.

No reward term writes targets, moves joints or teleports the robot. Oracle stair
geometry stays in the reward/critic. All three arms compute the same diagnostic
values without drawing additional random samples.

## Evidence and limits

Compare physical milestones, not differently scaled reward totals. A held landing
requires a supported airborne lift above 12 mm, followed by 0.25 s planted tread
contact, low slip and at least two other supporting feet. Shared telemetry reports
falls, supported lifts, paired landings, all-feet placement and full climbs. Final
tables average each seed's last 256 completed stochastic training episodes.
Whole-job counts and per-seed values are also retained.

New placement-event JSONL records the leg, stair level, body yaw, sole position,
corner gaps and hold duration. Episode-end records show whether each placement
episode later fell or completed a climb. Events in unfinished final episodes are
not included in completed-episode counts.

The D405 remains a 16×12 geometric depth proxy with range, latency, noise and
dropout. It omits self-occlusion and camera/mount mass; sideways viewing coverage
is not assured. The policy also assumes a separate heading estimator. Results
apply to this simulation and this short training budget, not physical feasibility
or reliable real-camera stair climbing.

Artifacts: `simulation/reviews/ankle-v23-research/20260913-potential-edges/` stores
commands, workflow status, training logs, metrics, source/asset/parent hashes,
completion records and results. Final models are archived under
`simulation/isaac/models/ankle-v23-research/20260913-potential-edges/`.

Launch (a fresh output folder is required for another screen):

```powershell
.venv/Scripts/python.exe simulation/isaac/rl/ankle_v23_research/run_screen.py --output simulation/reviews/ankle-v23-research/20260913-potential-edges --iterations 384 --num-envs 64 --workers 2
```

Verification consists only of the requested training jobs and read-only analysis
of their saved outputs. No separate automated verification is requested or run.
