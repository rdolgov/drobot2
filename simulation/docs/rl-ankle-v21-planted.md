# Printable ankle V2.1: flat-stance walking training

User request, September 12: run more thorough RL with the new printable design,
emphasize the ankle motor for stable planted soles, and record an example after
training. Asset import, PPO training, checkpoint selection and walk recording
are within this request. No standalone tests, URDF validator, mesh-closure
check, slicer or hardware operation is being run.

Status: both 3,000-iteration training jobs, all 12 selection trials, and both
36-second recordings completed on September 12, 2026. Orchestration status is
[`workflow.json`](../reviews/ankle-v21-planted/workflow.json).
Results are below. The first selection attempt finished one trial, then
failed while resetting an environment buffer created inside PyTorch inference
mode. The evaluator now uses `torch.no_grad()` so those buffers remain writable
between trials. The resumed selection and recordings completed from the saved
training checkpoints; neither PPO job was restarted. The failed report/log are retained in `selection/` with
the `inference-reset-failure` suffix for provenance.
The user's later [V2.3 retained-leg foot](../../cad/docs/ankle-v23-existing-leg-foot.md)
is a different mechanical arrangement. These saved policies remain V2.1; their
recordings/results must not be labeled as V2.3 or as a validation of that print.
The local task follow-up, `finish-ankle-v2-1-rl-recordings`, is paused at final
delivery. No training or recording work remains in this V2.1 run.

## Completed results and recordings

The full planned 98,304,000 training transitions completed across the two
seeds. Both returned at iteration 2999 with exit code 0 and saved final
checkpoints. Selection compared six checkpoints at two commands in 16
environments per trial (192 episodes total, including the nominal cases).

The best selection score belonged to **seed 2101, iteration 1500**. More
training did not improve this seed's selection score. All three seed-2101
checkpoints passed the configured gate; only iteration 2999 passed for seed
2102. Passing that limited gate is not hardware qualification or proof of a
clean gait: it does not impose a direct slip or lateral-drift threshold.

The two nominal recordings command 0.03 m/s, stop the command after 30 seconds,
and show the full 36 seconds at 30 fps (1080 frames, 1280 x 720).

| Recorded measurement | Seed 2101 / iteration 1500 | Seed 2102 / iteration 2999 |
| --- | ---: | ---: |
| Forward speed during moving command, m/s | 0.03430 | 0.01832 |
| Forward displacement over clip, m | 1.00515 | 0.52314 |
| Lateral displacement over clip, m | +0.16300 | -0.05425 |
| Planted fraction of contacting stance feet | 99.49% | 89.12% |
| Contact fraction of expected stance feet | 91.75% | 97.48% |
| Mean contacting-stance tilt, degrees | 1.41 | 1.48 |
| Mean contacting-stance slip, m/s | 0.01809 | 0.00843 |
| Mean ankle absolute angular velocity, rad/s | 0.3560 | 0.4756 |
| Individual ankle excursions, degrees | 8.71-11.30 | 17.10-25.85 |
| Mean speed after stopping, m/s | 0.00583 | 0.00587 |
| Falls / recording duration | 0 / 36 s | 0 / 36 s |

Planted fraction is conditional on scheduled stance feet actually contacting
the ground, using the force/corner-height proxy described below. It does not
mean all feet stay planted continuously or that pressure is uniform over each
sole. The best policy still exhibits substantial sliding and lateral drift.
The second policy uses larger ankle excursions but achieves less forward
speed; this is an observation across seeds, not a controlled causal comparison.

- [Best V2.1 recording, seed 2101](../reviews/ankle-v21-planted/walk-seed2101/model_1500-0.03mps-learned.mp4)
- [Second V2.1 recording, seed 2102](../reviews/ankle-v21-planted/walk-seed2102/model_2999-0.03mps-learned.mp4)
- [Seed 2101 recording report](../reviews/ankle-v21-planted/walk-seed2101/report.json)
- [Seed 2102 recording report](../reviews/ankle-v21-planted/walk-seed2102/report.json)
- [All selection trials](../reviews/ankle-v21-planted/selection/report.json)
- [Checkpoint ranking](../reviews/ankle-v21-planted/checkpoint-ranking.json)
- [Archived best weights](../isaac/models/ankle-v21-planted/seed2101-model_1500.pt)
- [Archived second-seed weights](../isaac/models/ankle-v21-planted/seed2102-model_2999.pt)

For the selected seed-2101 checkpoint, selection across nominal plus weaker
motors achieved mean 0.03214 / 0.05028 m/s at commands 0.03 / 0.05, with 0/16
falls at each command. Planted fractions were 98.67% / 97.98%, and slip was
0.01650 / 0.01918 m/s. The selected seed-2102 checkpoint achieved 0.01626 /
0.02967 m/s, with 1/16 falls at each command. These cases were used for model
selection and are not an independent final generalization test.

Review performed at completion: read both training completion records and
final log tails, all selected-policy reports, encoder completion logs, and
saved video frames at 10 and 30 seconds for each seed. Both encoders reported
1080 frames. No additional simulation, training, benchmark, mesh verification,
slicing, or hardware check was launched during this final review. This effort
does not establish faster running, an ankle-versus-fixed-foot improvement,
hardware transfer, or compatibility with the later V2.3 R3 solid-sole model.

## Model and controller

Canonical input is
[`quadruped_robot_v21_printable.urdf`](../../cad/exports/urdf/quadruped_robot_v21_printable.urdf).
The green replacement shank and assembled sole/cheeks come directly from the
printable V2.1 CAD. The old fork and gold placeholder envelopes are replaced.
Knee-to-ankle distance remains 159.896689 mm; sole remains 100 x 60 x 6 mm with
a 1 mm, 94 x 54 mm traction patch. Each leg has an extra ST3215 and 16 joints
are actuated in total. The old V2 pilot and V43/V44 tasks are unchanged.

Estimated whole mass is 3.350937565 kg. New printed parts use CAD-native volume,
COM and inertia at 1240 kg/m3 times an assumed 0.65 effective solid fraction.
This approximates walls plus infill and is not a slicer or scale measurement.
Each shank owns two 55 g servos and a 4 g fixing allowance; each foot adds 6 g
tread and 10 g fasteners. Other links retain their previous calibration.
No whole-robot renormalization hides the added motor mass. Calculations and
assumptions are in the URDF ledger and
[`training-mass-properties.json`](../../cad/reviews/ankle-v2-printable/training-mass-properties.json).

Detailed CAD is used for visuals; documented external box approximations
provide collisions. Adjacent bodies are collision filtered, general
self-collision remains on, and foot bodies remain separate for sensing.
The lower-leg STL triangulation warning remains unresolved. Physics collision
does not consume that mesh; this training does not clear the print for use.

PPO learns corrections to a slow crawl reference: 0.60 Hz, 78% stance duty,
18 mm swing lift, staggered phases (0, .5, .75, .25), 70 mm fore/aft stance
offset and 285 mm hip-to-ankle down reach. A command curriculum increases
from 55% to 100% of the sampled 0.025-0.065 m/s over 1,500 iterations.
Episodes include a startup ramp and a final three-second stopping period.
Ten percent of training commands are stands.

The ankle reference uses measured hip/knee angles and body pitch to compensate
their actual rotation, rather than merely cancelling their requested targets.
RL retains approximately +/-0.22 rad ankle correction authority. All residual
actions pass through a 0.12 s filter, then the existing target rate/packet
limits. Actor input has 82 values, including filter state and expected stance;
the critic has 90. The initial sole-velocity history artifact is suppressed
with an explicit per-environment valid-history flag, including randomized
first episode ages.

All motors retain the conservative 0.8826 N m sustained cap, torque-speed
envelope, target rates of 60 deg/s abduction and 100 deg/s other joints, and
2-degree packet limit. Strength and target rate vary independently from
85% to 100%. Ankle +/-35 degrees is a simulation assumption; physical travel
has not been qualified. Latency/backlash, mass/COM/friction uncertainty,
thermal limits and correlated electrical supply sag remain unmodeled.

## Rewards and measured contact

The goal is full-sole support **during stance**. Swing feet must leave the
floor. There is no reward simply for spinning the added motor.

Rewards combine speed/heading tracking, scheduled stance/swing participation,
stance flatness, swing clearance, low sole slip, body stability, target
smoothness and estimated mechanical work. All four corners are transformed
from the actual foot bodies. A planted contact proxy requires foot net force
above 1 N, corner-height spread below 5 mm, and highest tread corner below
4 mm above the floor. This is not a measurement of contact pressure across
the whole sole. One pitch ankle cannot independently remove foot roll.

Training logs include actual speed, falls, stance contact fraction, planted
fraction among contacting stance feet, stance tilt/slip, and ankle angular
speed. These are stochastic training summaries, not held-out results.

## Budget and completion pipeline

Seeds 2101 and 2102 each use 256 environments, 64 rollout steps and 3,000 PPO
iterations: 49,152,000 transitions each, 98,304,000 combined. Each receives
12 times the per-design transition budget of the original 500-iteration pilot.
The actor and critic use 256 x 256 x 128 networks and bounded Beta actions.
This is a training effort target, not a convergence guarantee.

The supervisor waited for both final checkpoints, then evaluated
iterations 1500, 2250 and 2999 from both seeds. Each candidate gets 36 seconds
at 0.03 and 0.05 m/s, with a stop after 30 seconds, across one nominal and
15 independently weaker motor configurations seeded with 2201. Failed
episodes are retained and masked after their first failure; resets are not
counted as successful continued travel.

Selection gate: at most 10% falls, at least half commanded speed, at least
70% planted stance contacts, and mean stopped speed at most 0.015 m/s at
both commands. Ranking favors candidates passing all gates, then the
documented score in `evaluate_record.py`. If none passes, it still records
the best available attempt and labels the result unaccepted.

After selection it recorded each training seed's best checkpoint at 0.03 m/s
under nominal conditions. Each MP4 includes a time/command/speed overlay,
real elapsed simulation time, and a following camera. A fall ends the clip;
failures are not hidden by resets. JSON reports and per-step traces accompany
the videos. Selected weights are archived under
`simulation/isaac/models/ankle-v21-planted/`.

This effort does not train matched rigid/locked controls and cannot establish
a causal advantage over those morphologies. The older pilot does not provide
a fair baseline for this changed model, reward, prior or training budget.

## Reproduction

From the repository root, after producing the CAD link-frame meshes and URDF:

```powershell
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v21/import_assets.py
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v21/train.py --rl_library rsl_rl --task Drobot-Ankle-V21-Planted --num_envs 256 --seed 2101 --max_iterations 3000 --run_name printable-2101 --visualizer none env.metrics_path=C:/Users/roman/Documents/dev/drobot2/simulation/reviews/ankle-v21-planted/20260912-seed2101/train.metrics.jsonl
# The second job uses seed/run name/path 2102. Do not duplicate running jobs.
& ./cad/.venv/Scripts/python.exe simulation/isaac/rl/ankle_v21/finish_training.py
```

Exact launch arguments, source hashes, completion status, logs and source
snapshots are retained per seed. The completion worker records evaluation
and recording command arrays. A complete status requires successful workers
and completed reports; checkpoint existence alone is insufficient.

Link-frame CAD sources reuse `ankle_v2_printable.py` without changing printed
geometry. They are `ankle_v21_shank_link.step.py` and
`ankle_v21_foot_link.step.py`, generated with CAD `scripts/gen --write` and
`scripts/export --stl`. The shank is translated from ankle origin to knee
origin; the foot is assembled in ankle coordinates. They are simulation
visuals, not replacements for the separate print-oriented files.
