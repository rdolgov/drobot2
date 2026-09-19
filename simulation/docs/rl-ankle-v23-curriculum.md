# V2.3 stair curriculum — 2026-09-15

**Scope correction:** these 20 mm results do not meet the previously discussed
four-by-180 mm stair target. [Full-height training continuation](rl-ankle-v23-fullheight.md)
tracks that active objective separately.

The privileged teacher has recorded three full floor-start climbs of three
20 mm risers, with all four traction patches planted on the top and stable
for one second. All three recorded zero episode joint-range excess. The
depth/proprioception student has also recorded one qualifying full climb in
309.70 seconds, with zero episode joint-range excess.
These are training trajectories under candidate servo/contact assumptions;
frozen-policy reliability and hardware transfer have not been established.

The winning teacher checkpoint is
[`full_climb_behavior_3.pt`](../isaac/models/ankle-v23-clear-resolved/clear-resolved-teacher-full-seed2922/full_climb_behavior_3.pt).
Its [recorded trajectory plot](../reviews/ankle-v23-curriculum/20260915/clear-resolved-teacher-full-seed2922/training_evidence.png)
and [source manifest](../reviews/ankle-v23-curriculum/20260915/clear-resolved-teacher-full-seed2922/training_evidence.json)
are generated from existing telemetry, without replaying the controller.
The student checkpoint is
[`full_climb_behavior_1.pt`](../isaac/models/ankle-v23-depth-student/depth-student-full-seed2951/full_climb_behavior_1.pt),
with its [trajectory plot](../reviews/ankle-v23-curriculum/20260915/depth-student-full-seed2951/training_evidence.png)
and [source manifest](../reviews/ankle-v23-curriculum/20260915/depth-student-full-seed2951/training_evidence.json).
Both teacher and student use known-stair references. The D405 input remains a
geometric depth proxy. This result does not establish camera-derived foothold
planning, a causal benefit from the camera, or physical-camera performance.

| Retained result | Training evidence | Scope |
| --- | --- | --- |
| Corrected first-tread teacher | 14 qualified floor-start stage completions by update 114 | All four feet on the first tread; not a full climb |
| Corrected full-climb teacher | Three floor-start climbs, 309.38–310.63 s each, zero joint-range excess | Three 20 mm risers, 250 mm treads, one-second planted top hold |
| Depth/proprioception student | One floor-start full climb, 309.70 s, zero joint-range excess | 485 inputs, 65-degree downward sparse camera proxy; no privileged-state tail |

The successful reference moves the front pair ahead, then advances the rear
feet individually with planned body shifts. Collision rejection keeps
nonadjacent links apart along the sampled reference. Dynamics use 240 Hz
physics, 60 Hz control, and requested 24/1 position/velocity solver iterations.
The candidate servo damping is 0.30 N m s/rad with stiffness 30 N m/rad;
damping has not been identified from the physical servo. Motor effort/rate
caps, joint-angle limits, self-collision, and the success guard remain enabled.
This sequence of adaptive experiments does not isolate each change's causal
effect, and a training-event checkpoint does not establish a frozen-policy
success rate.

The user authorized continued research and RL training attempts. No standalone
evaluation, test suite, build, browser validation, hardware connection or motion
check is part of this work. Results below are stochastic training telemetry.

## Geometry correction

The read-only calculation in
`simulation/reviews/ankle-v23-curriculum/20260915/geometry_calc.py` traverses the
current URDF transforms (including joint origins and axes) and finds the sole
centers and inertial COM. The 94 × 54 mm tread bottom is foot-local
`x = 0.0432117 m`, consistent with the collision box.

For the prior neutral stance (`forward = 70 mm`, `down = 285 mm`), the sole
bottom is **373.908389 mm below the base**. The prior reset used base height
356.2117 mm, placing the tread bottoms **17.696689 mm below the floor**. Its
342.2117 mm height target was 31.696689 mm below the matching neutral geometry.
This is a source/kinematics finding, not a separate simulator check. Earlier
experiments have been preserved and must not be relabeled as corrected runs.

The new isolated task uses `forward = 25 mm`, `down = 300 mm`, giving a neutral
base height of 388.908389 mm and initial base height 390.908389 mm (2 mm sole
clearance). This straighter stance reduces the moment arms of vertical loads.
The robot mass, joints, collision geometry and motor torque/rate limits are
unchanged. The new starting position is close to the first riser, with every
foot initially over the floor.

## First curriculum task

`simulation/isaac/rl/ankle_v23_foothold/` trains residual PPO around a preplanned
front-left foot trajectory: settle, lift 45 mm, move forward 114 mm, lower to
the 20 mm tread, and hold. Other feet have stationary world-position reward
targets. The residual controls all 16 joints. Reference ankle targets use
measured hip/knee angles and body pitch. Terrain and contact truth are used
for rewards and telemetry only; the reference uses a known fixed step plan,
elapsed time, IMU orientation and joint measurements.

Actor input includes the existing noisy/delayed D405 geometric depth proxy.
Because this first task uses a supplied fixed footstep plan, it does **not**
establish camera-driven foothold selection. Camera mass, self-occlusion and a
real D405 connection remain absent. No hardware deployment claim is made.

Job: `simulation/reviews/ankle-v23-curriculum/20260915/guided-fl-seed2801/`.
Fresh actor and critic, seed 2801, 128 environments, 256 PPO iterations,
64 transitions per environment per iteration (2,097,152 planned transitions).
Actual completion and checkpoint are recorded by the supervisor; a running
job or partial telemetry is not a completed result.

The first held placements appeared early in training. The current source
records the common airborne-placement criteria, plus a one-second stable
hold near the supplied foothold. The full-climb detector is unchanged and
reported separately.

## Research informing the next attempts

- [Reverse Curriculum Generation for Reinforcement Learning](https://arxiv.org/abs/1707.05300): begin with easier starts near a goal, then expand the starting region. Our manually staged foothold task is an adaptation, not a reproduction of that algorithm.
- [BaRC](https://arxiv.org/abs/1806.06161): use approximate dynamics and physical priors to organize a reachable curriculum. Here, URDF reach and stance geometry inform target selection.
- [Learning Bipedal Walking On Planned Footsteps](https://arxiv.org/abs/2207.12644): supply upcoming footholds and train phase-dependent contact/placement behavior. This paper concerns different robots; its performance does not transfer as evidence for this model.
- [RLOC](https://arxiv.org/abs/2012.03094): combines learned footstep planning with model-based tracking and learned corrective control. This supports trying a hybrid controller when unrestricted joint exploration fails.
- [Egocentric Vision Locomotion](https://proceedings.mlr.press/v205/agarwal23a.html): recurrent memory and a staged depth training procedure address the front-camera blind region around hind feet. This informs a later perception stage, after contact mechanics work.
- [StairMaster](https://arxiv.org/abs/2606.25765): a recent preprint uses privileged teacher training, depth/proprioception distillation, and PPO refinement. A privileged teacher would be a separately labeled intermediate result here.

Next decisions are based on held placements and falls, not just total reward.
The intended progression is one front foot, front pair with weight transfer,
rear feet and body advancement, then the entire staircase and perception
variation. Failed approaches should prompt changes to the motion/learning
curriculum while preserving honest success criteria.

## First completed result and ensuing changes

The initial one-foot job completed all 256 iterations and 2,097,152 transitions.
It recorded 599 one-second foothold holds across 3,410 completed episodes,
635 falls, and zero full climbs. An early rolling success rate reached about
66%, but the final 512 episodes contained only 4 foothold successes (0.78%)
and no falls. The final checkpoint is therefore **not a reliable landing
policy**, despite the intermediate learning breakthrough.

The original curriculum ended successful episodes early while paying a large
positive per-second tracking reward. That makes postponing completion more
valuable in some states. Edge-overhanging feet can also earn substantial
distance reward. This is a reward-design diagnosis supported by training
traces, not a controlled causal ablation.

`ankle_v23_footplan` addresses this by retaining the fixed episode horizon,
paying each placement milestone once, and rewarding retention of qualified
planted feet. It trains a front-left then front-right sequence with a 35 mm
body shift. It starts from iteration 64 of the first job, with a fresh critic
and optimizer. The historical field `foothold_success` in this package means
**both front feet held simultaneously for one second**; the event records
explicitly identify it as an intermediate front-pair result.

`ankle_v23_crawlplan` extends the sequence to rear feet. Its pure arithmetic
planner keeps a 2.8 mm reserve from the two-link reach, lowers the body when a
supporting leg would otherwise overextend, and clips ankle references to the
unchanged 33.25-degree soft limit. Swing feet may tilt at that limit. The
four-phase stage puts both front feet on the first tread and advances the
rear feet along the floor; it is not yet an all-feet first-tread result.

The recurrent actor gets 16 additional desired-joint-error inputs. Transfer
copies the existing 469 input columns and normalizer, initializes the new
GRU input columns to zero, and gives the new normalization dimensions zero
mean/unit variance. All other actor weights are copied; critic/optimizer are
fresh. This transfer happens as part of training, not a standalone check.

The planned full six-cycle path ends with all four feet on the third tread.
That geometric plan alone is **not evidence that the robot can execute it**.
Actual training must establish the full-climb event.

## Camera follow-up

[RealSense's D405 specification](https://www.realsenseai.com/product-family/d405-series/)
lists a 7–50 cm ideal depth range and nominal 87 × 58 degree field of view.
The existing 40-degree downward proxy orientation leaves a near-foot blind
region at the corrected roughly 39 cm torso height. Simple pinhole geometry
suggests a steeper downward mount may expose the next landing region better.
This is a mounting hypothesis, not a sensor or hardware check.

## Front-pair outcome and rear-foot attempts

The front-pair job completed 384 iterations (3,145,728 transitions). It recorded
998 one-second front-pair holds across 2,556 completed training episodes, with
356 falls. The final 512 episodes had **80.664% front-pair success** and 2.734%
falls. Full climbs and simultaneous all-four-feet first-tread placements were
both zero. The checkpoint is
`simulation/isaac/models/ankle-v23-footplan/front-pair-seed2811/model_383.pt`.
This is an intermediate training result, not a held-out performance estimate.

The first four-phase crawl was intentionally stopped after logged iteration
218 of 512, with no completed four-foot sequence. `early_stop.json` records
that decision; its supervisor correctly reports an incomplete/failed job,
not completion. The last periodic checkpoint is iteration 192.

A read-only URDF contact-force calculation found a roughly 1.0 Nm minimum
peak torque for one planned rear-left swing pose, even with favorable contact
assumptions. This exceeds the 0.883 Nm nominal cap. This is a limitation of
that pose, not proof that the robot cannot climb. The calculation assumes a
friction pyramid and permits forces at all stance tread corners; it is
optimistic when a foot is tilted and is not a dynamic feasibility proof.

`ankle_v23_balanced` moves the torso farther forward before the last rear lift,
adds a small planned body pitch based on the support heights, doubles phase
duration to 18 seconds, and adds static load compensation as joint-position
target offsets (`estimated torque / 30 Nm/rad`). All compensation still goes
through the original joint-target limits, rate limits, torque-speed envelope,
motor strength randomization, contacts and physics. No external support forces
or body-state corrections are used during an episode.

The compensation table is calculated from the URDF at 0.2-second intervals.
Sixteen of 2,181 samples exceed the weakest randomized motor cap or have no
static solution; compensation is clamped and residual control must handle
the discrepancy. This table is a controller prior, not a success claim.

The revised task mixes floor starts with reference-phase curriculum starts.
Only floor-start episodes are eligible for the full-climb counter. Curriculum
arrivals, planned-stage completion, and floor-start stage completion are
reported separately; the policy is never credited with a full climb merely
for being initialized on stairs. Subsequent resets use a 30% floor-start
probability. The first batch may start from the asset's floor state.

An additional fresh-policy arm (`ankle_v23_balanced_small`, seed 2832) halves
the residual ranges to reduce exploratory disturbance near the torque cap.
The warm arm (seed 2831) starts from front-pair iteration 320. These are
adaptive attempts with different initialization and sample budgets, not a
controlled one-variable comparison.

## Shorter steps, state information, and stance ankle control

The four-phase velocity-compensation attempt was stopped after more than 200
iterations with no sequence successes and about 95% falls. The balanced arms
were also stopped without a completed sequence. Their `early_stop.json` and
supervisor records retain the incomplete status and actual iteration counts.

`ankle_v23_shortstep` uses an alternating sequence: front-left 125 mm,
rear-right 85 mm, front-right 125 mm, rear-left 85 mm, then another 85 mm for
each rear foot. Static planning optimizes torso translation and supplies
gravity/contact torque offsets. A velocity lead of `Kd/Kp * planned_qdot`
compensates damping against a moving target. It preserves the original drive
gains and all motor limits. Its 1,561 static load samples are within the weakest
randomized cap; this is only a geometric/static calculation.

That run produced a few six-phase completions from curriculum resets, but no
floor-start completion. It was stopped after more than 180 iterations. The
last 512-episode breakdown showed floor starts usually failing in phase zero,
with forward body drift. Small foot-position errors alone concealed this
failure: root-position error was much larger.

`ankle_v23_guided` adds body velocity, reference-relative torso/foot errors,
contact flags, and plan progress to the critic, and optionally to a privileged
teacher actor. It saves successful trajectories from existing training episodes
without launching replay or evaluation rollouts. A teacher-first approach is
consistent with [StairMaster](https://arxiv.org/abs/2606.25765); any result using
those extra actor inputs is explicitly labeled privileged-state control.

`ankle_v23_supported` adds swing-foot tracking and desired-contact-pattern
rewards, following the contact-phase and footstep-reward ideas in
[Singh et al.](https://arxiv.org/abs/2207.12644). Its static planner penalizes
center-of-mass positions outside a 5 mm interior margin of the foot-center
support polygon. Not every later phase achieves that margin; the per-phase
metadata records the actual value. First front steps are shortened to 115 mm.
The physical sole shape and contact model do not change.

During stance, this variant holds the planned ankle reference, retaining
attitude stiffness instead of continually following the measured torso tilt.
During swing it still levels the foot from measured joint/IMU angles. It
restores the original front-pair residual ranges and uses 70% floor starts
after resets. The first run trains only two phases (front-left placement,
rear-right advance). These bundled adaptive changes are not a causal ablation.

All reported outcomes remain stochastic training evidence. No separate tests,
evaluation runs, hardware connections, or motion checks have been performed.
The D405 input remains a sparse geometric proxy; neither the fixed footstep
planner nor a privileged teacher proves camera-derived stair planning.

The supported two-phase run recorded 33 floor-start successes (49 including
curriculum starts) before an unexpected process exit near logged iteration 60.
Eight successful floor-start trajectories are retained. Only iteration zero
had been checkpointed. The guided six-phase run also ended unexpectedly near
iteration 128, with no floor-start sequence successes. Both jobs are explicitly
marked `interrupted`; their cause is unknown, and neither is a completed job.

`ankle_v23_compact` saves a periodic checkpoint every 16 iterations and captures
the exact behavior actor whenever a full climb occurs during training. Its
first six phases match the supported plan. It then puts the rear feet onto
the first tread before advancing either front foot again. Later cycles move
front-left, front-right, rear-right, rear-left. This reduces the large leg
extension and unfavorable support triangle in the earlier alternating plan.
Its full path has 24 phases. Static calculation found no above-cap samples
among 1,441 load references, but several phases still rely on some of the
finite sole support area. This does not establish dynamic feasibility.

The restart trains the two-phase stage from the interrupted run's available
iteration-zero checkpoint. An eight-phase run starts from its iteration-16
checkpoint, targeting all four feet on the first tread. These are intermediate
training tasks; full success still requires all three risers from the floor.

## September 16 continuation: range failures and reference collisions

The compact two-phase continuation was stopped after more than 118 iterations,
with 60 floor-start stage successes among 775 completed episodes. Its iteration
112 checkpoint is retained under `simulation/isaac/models/ankle-v23-compact/`.
The eight-phase continuation was stopped with no sequence completion. Neither
is a full stair-climbing result. Later geometry findings below also limit how
the two-phase successes should be interpreted.

`ankle_v23_precise` reduced residual amplitudes to 0.06/0.10/0.12/0.10 rad and
initialized a trainable Beta(20,20) actor with zero action mean. It introduced
an episode-wide joint-range guard: any overshoot beyond the configured hard
limits by more than 0.02 rad disqualifies success. It retains detailed failure
poses and captures an exact behavior checkpoint on a full training climb.
The run was stopped after 3,794 episodes, all ending in falls, with no success.
Failures included ankle deflections far beyond the 35-degree limits.

`ankle_v23_response` changed the assumed servo damping from 4.5836625 to 0.30
N m s/rad while preserving Kp=30, torque/speed caps, target rates, and motion
ranges. This is a servo-response hypothesis, not a hardware calibration.
Its velocity feedforward uses the same revised Kd/Kp ratio. Early floor-start
survival improved, but there were no completed climbs. Runtime metadata
confirmed that backend and policy joint orders match, requested joint limits
are present, and body masses sum to the URDF's 3.44402 kg. Reported implicit PD
effort remains an estimate, not a measured resolved motor torque.

`ankle_v23_resolved` used the same response hypothesis with 240 Hz physics,
60 Hz control, and requested 24 position / 1 velocity iterations. NVIDIA
documents interactions between articulation drives, contact constraints, and
joint limits in its [PhysX articulation documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/Articulations.html).
Both response arms were stopped when a more direct geometry defect was found;
their short runs do not isolate a reliable solver effect.

The read-only `collision_geometry_calculation.py` applies a 15-axis separating
axis calculation to the actual URDF collision boxes, excluding same-link and
directly adjacent link pairs. The old compact reference had overlaps greater
than 0.5 mm in 85 of 144 poses sampled every two seconds. Its first rear-right
advance approached the still-unmoved front-right foot; subsequent rear advances
eventually put rear feet ahead of the front feet. Some later poses overlapped
foot housings and lower legs substantially. Static load feasibility alone had
missed this. The old two-phase training successes therefore do not establish a
collision-clear step sequence. Self-collision was enabled throughout.

`ankle_v23_clear` changes the foot order: move both front feet before the rear
feet, use two short rear advances on the floor, advance the front pair again,
then bring the rear pair onto the first tread. Later front-pair / rear-pair
cycles maintain separation. The complete reference has 26 twelve-second
phases and a four-second terminal hold. Landings require the full 100 mm sole
to fit on one horizontal surface. Candidate torso poses reject nonadjacent
box penetration above 0.1 mm. The final 0.2-second static sampling found zero
overlaps and zero above-cap load cases among 1,560 samples. These are finite
geometry/load calculations, not dynamics or continuous collision guarantees.
Some phases still use the finite sole support area rather than a positive
foot-center support margin.

Two fresh privileged-teacher training jobs use this corrected reference:
`clear-teacher-full-seed2921` at 120 Hz / 8-position / 4-velocity iterations,
and `clear-resolved-teacher-full-seed2922` at 240 Hz / 24 / 1. Both have 128
environments, 512 planned PPO updates, small residuals, Beta(20,20)
initialization, damping 0.30, and 50% floor-start probability after resets.
Only actual floor starts are eligible for full-climb success. A success also
requires all four traction patches on the third tread, low body speed, a
one-second hold, and the episode-wide joint-range guard.

No standalone evaluation, test suite, hardware connection, or motion check
was run in this continuation. All dynamic evidence comes from the explicitly
requested RL training. D405 input is still a geometric depth proxy; these
teacher policies also use privileged state and known-stair planning.

The coarse clear-reference arm was stopped after repeated large joint-range
violations. The finer arm produced qualified near-top curriculum arrivals and
kept range overshoot small in its captured falls. This is training evidence
for the combined resolution change, not an independent solver-convergence
study. A second finer-physics job, `clear-resolved-first-tread-seed2931`, trains
the ten-phase first-tread stage (128 environments, 320 planned iterations).
Curriculum and floor-start completions continue to be counted separately.

The full run's first floor-start failures clustered around phase five. One
retained record shows the rear support foot advanced into the first stair edge
and lifted while the other rear foot was swinging. The old final floor
foothold offered only approximately 3 mm clearance for the 100 mm sole. The
new `ankle_v23_margin` reference keeps 20 mm there and 15 mm at stair landings.
An initial 25 mm floor-clearance proposal had one sampled load at 1.0047 times
the weakest motor cap; it was not trained. Moving that floor foothold forward
5 mm shortened the subsequent swing and removed the sampled load exception.
The resulting reference has zero sampled overlap/load exceptions in 1,560
poses. `margin-teacher-full-seed2941` trains this variant with 64 environments,
512 planned iterations, and the same finer physics and servo assumptions.

[Ye et al.'s optimized static-gait paper](https://linqi-ye.github.io/docs/2021Optimized%20Static%20Gait%20for%20Quadruped%20Robots%20Walking%20on%20Stairs.pdf)
also motivates checking knee configuration and coordinated body/foot motion.
The read-only `knee_posture_calculation.py` examined both uniform knee-bend
directions on the clear reference. Only 248/312 or 242/312 sampled poses,
respectively, respected all joint limits; maximum ankle excess was about
16.8 or 18.3 degrees. Neither is a drop-in viable reference, and neither was
trained. Changing the body/foot path would be needed before trying those
configurations dynamically.

The finer first-tread run recorded **14 qualified floor-start stage successes**
by iteration 114. A qualification includes all four selected feet on the first
tread, the required lift history, a one-second planted hold, and the episode
joint-range guard. Eight floor-start trajectories are retained in that job's
`training_trajectories/` directory. Iteration 112 is archived under
`simulation/isaac/models/ankle-v23-clear-resolved/clear-resolved-first-tread-seed2931/`.
Training was deliberately stopped early to prioritize the full-climb arms;
the checkpoint contract explicitly records incomplete training. These are
stochastic training episodes, not independent evaluation of a frozen actor.

`prepare_depth_student.py` prepares a subsequent online distillation package
from either full-climb reference family. It has not been executed or trained
at this point. The intended student has the 485 proprioception/reference/depth
inputs and excludes the teacher's 23 privileged state features. It uses a
65-degree downward camera proxy, while the frozen teacher retains its original
40-degree proxy. The installed RSL-RL distillation mechanism acts with the
student and supplies labels from the teacher. A local subclass preserves the
teacher's recurrent state across optimization chunks. Known-stair footstep
references remain part of this controller; this would not establish autonomous
camera-derived stair geometry or foothold selection.

The student package has now been generated from `ankle_v23_clear_resolved`;
its launch still awaits a qualified full-climb teacher event. The first-tread
job also contains `training_evidence.png` and `training_evidence.json`, rendered
from its eight retained recordings without replaying a policy. The plotted
example has zero recorded episode joint-range excess. Long training episodes
can span optimizer updates: an event checkpoint captures weights at that
event, not necessarily one fixed actor used for the entire recorded climb.

## First full floor-start climbs

`clear-resolved-teacher-full-seed2922` recorded **three qualified full climbs**
by training step 18,655 (around PPO iteration 288). The retained successful
episodes started on the floor, crossed all three 20 mm risers, and ended with
all four traction patches on the third tread and the required one-second
hold. Their recorded durations were 309.38, 309.55, and 310.63 seconds; all
three had zero episode joint-range excess. Each recording begins with four
floor-level feet and ends with four level-three feet and four contact flags.

The event checkpoint `full_climb_behavior_3.pt` is retained under
`simulation/isaac/models/ankle-v23-clear-resolved/clear-resolved-teacher-full-seed2922/`.
The job contains three full trajectories and a rendered `training_evidence.png`
plus its source manifest. The teacher training was deliberately stopped after
this feasibility result, and its archive remains marked incomplete relative
to the original 512-update budget. This establishes a simulated training
trajectory under the stated servo/contact assumptions, not a robust frozen
policy, a normal-height staircase capability, or a hardware result.

The margin arm was also stopped early and iteration 160 retained; no full climb
had been observed in it. Compute is now assigned to
`depth-student-full-seed2951`: 128 environments, 512 planned distillation
updates, the successful teacher event checkpoint as frozen supervisor, a
485-input student with no privileged-state tail, and the 65-degree D405 proxy.
Student actions drive the training dynamics. The known-stair reference and
all motor/range/success limits remain in place.

## First full climb by the depth/proprioception student

`depth-student-full-seed2951` recorded a qualifying floor-start full climb at
step **18,582**, environment 34, during online distillation. The retained
trajectory starts with all four feet on floor level zero, lasts **309.70 s**,
and ends with all four feet on level three and all four contact flags set.
The full-climb detector recorded a **1.0167 s** top hold. The episode-wide
joint-range excess was **zero**. Student actions, sampled from the 485-input
recurrent actor, drove the dynamics; the frozen privileged teacher supplied
training labels only.

The successful student is stored as `student_state_dict` in the archived
`full_climb_behavior_1.pt`; the same training checkpoint also contains the
frozen teacher and optimizer state. Do not load `teacher_state_dict` as the
student. The camera proxy is pitched 65 degrees downward; the student receives
IMU/joint/action history, phase/reference inputs, and sparse noisy/delayed
depth. It does not receive the teacher's 23 privileged state features.

The run was deliberately stopped after this event, with update 307 last
reported, rather than consuming all 512 planned updates. The supervisor's
nonzero termination and incomplete archive contract are retained, with the
reason in `early_stop.json`. Final telemetry contains one full climb among
131 completed floor-start episodes, plus six separate curriculum goal
completions. These changing-policy, censored training counts are **not a
frozen-policy success-rate estimate**. The full climb did not meet the extra
exact-reference-position stage criterion, but it met the independently
defined physical top-contact/velocity/hold/range criteria.

The student's successful checkpoint was captured at the event; the long
episode may span many preceding optimizer updates. No standalone policy
evaluation, simulation replay, tests, build, hardware connection or hardware
motion check was run. The plot only renders the recorded training data.

An isolated `ankle_v23_depth_refine` PPO continuation was prepared while
distillation ran. It would warm-start the student actor and matching
privileged critic with a fresh, fixed-1e-4 PPO optimizer. It **was not launched**
because the student achieved the requested first simulated feasibility
milestone. Further work remains on reliability, camera-derived terrain
planning, actual servo identification, and larger risers.
