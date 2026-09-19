# V2.3 full-height stairs — 2026-09-17

The active objective is **four 180 mm risers with 250 mm treads**, matching the
earlier project target. The preceding three-by-20 mm result is a small-step
curriculum milestone and does not meet this objective. The 180 mm riser is
nine times the height used for that result. Dimensions are approximately
7.09 inches rise and 9.84 inches tread; the original 250 mm tread should not
be described as meeting a 10-inch minimum.

The user authorized continued training on full-size stairs and requested a
video. Training and the requested recording are permitted; no hardware motion
or connection is authorized here. All work remains in simulation.

The user also clarified that small-stair videos are not needed. Future
recordings must use the full-height 180 mm stairs; do not schedule or run new
small-stair recordings. Existing small-stair videos are historical artifacts.

Latest continuation, September 18 at 11:41 PDT: the corrected
`stairs-floor-load-consistent-s3281` trial is training. It fixes a mismatch
between the intended rear-left unloading forces and the servo targets.
The former 11.734 N rear-left load allocation is replaced by the exact
constrained 2 N allocation; motor limits and action gains are unchanged.
The original command limits admit instantaneous takeoff with these
corrected preloads, though a finite climb remains unproven. Three training
workers are active. The requested full-height video is complete: it achieves
the whole-first-tread hold at 175.1 s from a three-feet-on-first reset,
after a long delay involving right-foot self-contact. It reaches the 180 s
clip limit before a second-tread plant. The corrected floor trial has
25 measured rear-left unloading events by step 1,400. No floor-start rear plant
or complete climb has succeeded. See the
[current analysis](rl-ankle-v23-reward-analysis.md).

Earlier continuation, September18 at10:58 PDT: the tilted-transfer trial
logged one strict flat front-left plant on tread2, with2.275 N load and
5.087 mm maximum sole gap. This is grounded-curriculum, changing-policy
progress; subsequent right-foot self-contact and support loss prevent a
stable continuation. No floor-start rear plant or complete climb has
succeeded. Its actor is preserved and a new requested frozen full-height
video is recording. Four RL runs continue. See the
[current evidence](rl-ankle-v23-reward-analysis.md).

Earlier continuation, September18 at05:00 PDT: no full climb. The requested
frozen video finished but failed to reproduce the training touchdown,
stalling with the rear foot pressing sideways into the first riser.
[Completed video](../reviews/ankle-v23-curriculum/20260915/video-second-touchdown1-s3291/frozen-second-tread-180mm.mp4).
Four full-height RL jobs remain active, including a new33-pose lateral
detour. Its selected leaders reach the second-tread approach after an
actual curriculum first-tread hold, then contact the second riser. Offline
floor-start mechanics favors trying rear-left first and permits an
instantaneous dynamic takeoff, but no connected floor-start transfer has
been executed. See the [current evidence and limitations](rl-ankle-v23-reward-analysis.md).

Earlier continuation, September18 at04:20 PDT: one actual loaded tilted
front-left touchdown on tread2, followed by an8.49mm supported lift of
the other front foot. This starts from the grounded three-on-first
curriculum and later loses useful support. There is still no qualified
flat second-tread plant, complete floor-start transfer or full climb.
The saved touchdown actor is being recorded on the full-height stairs;
frozen reproduction is pending. A new RL branch supplies the missing
load-transfer signal for valid tilted contacts while preserving strict
plant/success and physical constraints. Four workers are active. See the
[current analysis and reconstructed training figure](rl-ankle-v23-reward-analysis.md).

Earlier continuation, September 18 at 03:41 PDT: no complete floor-start
transfer, qualified second-tread plant or four-riser climb. The prospective
load reward has 24 changing-policy curriculum holds. A new full-height
training branch continues guidance through 23 accepted static second-tread
poses after the measured first-tread hold; large connecting pose changes
remain unproven. Tilted touchdown is recorded separately from flat planting
and full success. The landing-corridor run was archived to free its slot.
Four workers remain live. New contact-force calculations and a low-load
support-reward mismatch are documented in the
[current analysis](rl-ankle-v23-reward-analysis.md).

Earlier continuation, September 18 at 03:20 PDT: no complete floor-start
transfer, second-tread plant or four-riser climb. Four full-height RL jobs
remain active, including a new reward-only trial for a lightly loaded toe
contact on the next tread before flat-foot credit. Offline mechanics admits
tilted touchdown carrying 2–4 N, but demanding 8 N fails. Further flattening
runs into right-leg collisions or local joint/load limits. An alternate
sequence can reduce the other front foot's load to 2 N but has not fully
unloaded it. The oldest parent was archived before its slot was reassigned;
all physical limits and success checks remain unchanged. See the
[current analysis and figure](rl-ankle-v23-reward-analysis.md).

Earlier continuation, September 18 at 02:52 PDT: no complete floor-start
transfer or four-riser ascent. The landing-reward trial has 82 curriculum
holds; the residual-handoff trial has 30, with later falls still occurring.
A new full-height frozen video reaches rear guide97 but fails to plant.
Its safe foot placement is about 33 mm from the reference point, revealing
another guidance gate to isolate. A fourth job now permits that bounded
offset only when the whole foot fits and clears the tread; it has three
curriculum holds at step720. Static math also
finds a full FL lift and crossing above the second tread by allowing all three supports
to rock. Flattening while the foot remains unloaded fails; the next hypothesis
is tilted touchdown, loading the new contact, then flattening. None of these static
endpoints is installed as a new reset or reference. See the
[current evidence and video](rl-ankle-v23-reward-analysis.md).

Earlier continuation, September18 at02:27 PDT: the landing-reward trial has44
curriculum first-tread holds; the parent has20. No complete floor-start
first-tread transfer or four-riser ascent has succeeded. A new256-environment
trial preserves the learned residual action meaning after guidance fades;
it has three curriculum holds at step920, with no second-tread foot credit.
Two stalled branches were archived and stopped. The next-step mechanics
calculation reaches a120mm FL lift above tread1 with FR support rocking,
but the higher pose fails collision/joint constraints. No new reset or
reference has been installed from those calculations. A full-height video
of the actual saved stage-event4 actor is recording. Details and limitations
are in the [reward analysis](rl-ankle-v23-reward-analysis.md).

Earlier continuation, September 18 at 02:03 PDT: the original pair-transfer
run has achieved sixteen actual all-four first-tread holds from the grounded
three-on-first curriculum. The stage actors and trajectories are saved.
There is still no complete floor-start first-tread transfer or four-riser
climb. A new alignment reward trial is training from the rear-stage event4
actor; startup confirms exact transfer of all 14 actor tensors. At step1,000
it has eight curriculum stage holds and 53 floor-start front-pair holds,
with zero full climbs. It
addresses an overly narrow clearance prerequisite in the old landing
shaping, while preserving the physical and success constraints.

The frozen model56 rear-transfer recording uses the existing three-on-first
start and finishes 90 seconds stalled at guide8, with no new rear credit,
fall or range excess. It is explicitly a
curriculum clip, not a floor-start climb or frozen reproduction of the
new stage actors. The best completed floor-start video still demonstrates
the front-pair hold. See the latest [reward and mechanical analysis](rl-ankle-v23-reward-analysis.md)
for milestone evidence, contact calculations and the pending floor connection.

Earlier continuation, September 18 at 01:12 PDT: four existing full-height
RL runs continue, still without a whole-robot first-tread hold or full
climb. The stance-anchor reward has some rear-lift leaders above 160 mm,
but supporting-foot contact remains unresolved. New static analysis finds
grounded hip-recovery and front-foot repositioning samples, with no proven
next rear lift. A compliant-contact modeling gap was identified for the
next planning refinement. See the [current report](rl-ankle-v23-ik-progress.md).

Earlier continuation, September 18 at 00:48 PDT: rear-support now lifts the
whole rear foot about 140–150 mm, then stalls as a supporting front foot
drifts into the next riser. A matched 1,024-environment stance-anchor reward
comparison is training to discourage this drift. Rear-retreat was archived
and stopped; four jobs remain. There is still no whole-robot first-tread hold
or full climb. See the [current report](rl-ankle-v23-ik-progress.md).

Earlier continuation, September 18 at 00:42 PDT: the new 1,024-environment
rear-support comparison reduces corrections on the three supporting legs
while retaining full swing-foot authority. Early partial rear lift reaches
136.7 mm on the unchanged 180 mm stairs, starting with three feet on the
first tread. There is still no whole-robot first-tread hold or full climb.
Three older arms were archived and stopped; four training runs remain.
See the [current report](rl-ankle-v23-ik-progress.md).

Earlier continuation, September 18 at 00:15 PDT: two new rear-transfer
comparisons are training on the same full stairs. One retreats along the
floor before lifting; the other overlaps retreat with a 0.5–5 mm unloading
lift. Both start from the original grounded reset and share the same
bounded clearance reward and actual geometry gates. The older aggregate
arm was archived and stopped for the second comparison. Initial startup
encoding/path errors were corrected; both failed-start logs are retained.
Six runs remain active, with no whole-robot first-tread hold or complete
climb. See the [current report](rl-ankle-v23-ik-progress.md).

Earlier continuation, September 18 at 00:03 PDT: the rear-retreat RL trial
is running with a revised 122-pose static rear route and a bounded
clearance reward. Recorded forces and reconstructed plastic geometry
show rear-foot contact with the vertical riser. The revised guide asks
the foot to retreat before lifting, then cross and plant with a small
tilt; its final static settling pose meets the existing corner-gap
tolerance. Connecting dynamics remain unproved. The original grounded
reset and full-height stairs remain. The older large exact-IK baseline
was archived and stopped to allocate this trial. Six runs continue;
there is still no whole-robot first-tread hold or complete climb.
See the [current report](rl-ankle-v23-ik-progress.md).

Earlier continuation, September 17 at 23:35 PDT: a new comparison uses
one-tenth residual corrections during the guided rear segment, retaining
the transferred pair actor and full-height task. Recorded target variation
is large relative to the rear reference's narrow static torque margin;
the new run tests that hypothesis without relaxing motor, joint or contact
limits. The older rear-bearing arm was archived at update 40 and stopped.
Six RL runs continue, with no whole-robot first-tread hold or complete
climb yet. See the [current report](rl-ankle-v23-ik-progress.md).

Earlier continuation, September 17 at 23:18 PDT: the new pair-transfer RL
run preserves the complete actor from the successful frozen front-pair
recording and continues with aggregate load sharing and gradual handoff.
Training confirmed exact equality of all actor tensors at initialization.
The side-impact run was archived at update 40 and stopped to keep six
concurrent runs. Eight additional static searches using actual outer-sole
contact corners found no valid rear takeoff; no candidate pose was installed.
The complete four-riser climb remains unsolved. See the
[current report](rl-ankle-v23-ik-progress.md).

Earlier continuation, September 17 at 23:04 PDT: the frozen clearance10
checkpoint reproduced both front feet planted on the first 180 mm tread
at 39.85 seconds from the floor. The completed
[video](../reviews/ankle-v23-curriculum/20260915/video-clearance10-pair-s3291/frozen-floor-start-180mm.mp4)
ends one second after that hold. Neither rear transfer, whole-robot
first-tread hold nor complete climb has succeeded. A sixth training arm
started at 22:58, using a gradual observable release of reference assistance
after the pair hold. Offline analysis of the recorded pose identifies rear
hip target headroom as an additional takeoff constraint. All runs remain
reference-assisted simulation teachers, not camera-only policies.
See the [current report](rl-ankle-v23-ik-progress.md) and
[reward analysis](rl-ankle-v23-reward-analysis.md).

Earlier continuation, September 17 at 22:39 PDT: no full climb or whole-robot
first-tread hold has succeeded. Static force calculations show the old
minimum-per-raised-foot load reward favors a distribution above the motor
cap at three rear-lift reference poses. A new total-load reward trial
permits feasible unequal sharing while retaining individual contact and
landing requirements. It replaces the archived larger post-pair edge-only
run. Five trials remain active: exact IK, rear-bearing, side-impact,
clearance10 and aggregate load. See the
[current report](rl-ankle-v23-ik-progress.md) and
[reward analysis](rl-ankle-v23-reward-analysis.md).

Earlier continuation, September 17 at 22:25 PDT: no complete climb or
whole-robot first-tread hold has succeeded. Direct force logs now confirm
front-left contact against the riser at the recurring crossing stall.
Offline geometry/force calculations found a revised 10 mm whole-foot
clearance sequence through a flat landing. A fifth isolated RL trial uses
that sequence; its connecting motion is unproven. The other four runs are
exact IK, loaded-edge support, side-impact penalty, and bearing support
extended to the grounded rear curriculum. The older foot-tracking arm was
archived and intentionally stopped. The latest frozen full-height video
still shows one front-foot plant. See the
[current report](rl-ankle-v23-ik-progress.md) and
[reward analysis](rl-ankle-v23-reward-analysis.md).

Earlier continuation, September 17 at 22:00 PDT: the frozen pair-checkpoint
[video](../reviews/ankle-v23-curriculum/20260915/video-posture-damping-pair-s3291/frozen-floor-start-180mm.mp4)
is finalized. It plants one front foot, then stalls; the training front-pair
hold did not reproduce in this 90-second clip. No whole-robot ascent has
succeeded. Four RL jobs continue: sharper foot tracking, the larger exact-IK
baseline, loaded-edge support, and a new side-impact penalty. The original
posture-damping baseline was archived at update 32 and stopped to allocate
the new trial. A recorded sole/riser contact motivated the collision cue;
offline work is also considering a deeper first-tread landing before rear
transfer. See the [current report](rl-ankle-v23-ik-progress.md).

Earlier continuation, September 17 at 21:38 PDT: three reference-assisted RL
runs now achieve front-pair holds on the 180 mm first tread. Rear transfer,
whole-robot first-tread hold and the complete four-riser climb remain
unsolved. Four full-height training arms are active, including a new trial
that recognises loaded edge contact during weight transfer after the front
pair has held. Its landing and full-climb gates retain the full-foot rule.
Offline rear-right airborne poses are feasible but their floor takeoff is
unresolved; a diagonal foot-order calculation is also running.

The original [full-height video](../reviews/ankle-v23-curriculum/20260915/video-reference-loaded-s3291/frozen-floor-start-180mm.mp4)
is finalized and shows one front foot reaching the first tread, with no
fall or whole-robot ascent. The later pair checkpoint is being recorded
from the floor in a separate progress clip. See the
[current IK/RL report](rl-ankle-v23-ik-progress.md) for evidence and limits.

Earlier continuation, September 17 at 21:16 PDT: the posture-damping RL trial
has achieved simultaneous front-foot holds on the first 180 mm tread, with
its first at step 847 and 36 cumulative events by step 1760. The first-event
checkpoint and provenance are preserved. Rear transfer and all-four-riser
success remain unresolved, and this pair milestone is not yet reproduced
in a frozen recording. Three training jobs continue: posture damping,
sharper swing-foot reward, and a larger 1024-environment version using a
batched solution of the same bounded IK objective. The original one-foot
recording remains active. See [latest IK progress](rl-ankle-v23-ik-progress.md).

Earlier continuation, September 17 at 20:52 PDT: one front foot now plants
on the full-height first tread in reference-assisted RL training and in
the requested frozen-policy floor-start recording. The recording repeated
that partial milestone at about 16 simulated seconds and is still running.
There is still no all-four-foot first-tread hold or full staircase climb.
The first-event checkpoint and source provenance are preserved. A new
reward-only comparison encourages transferring weight onto the planted
foot while unloading a foot still on the lower level. Offline IK has
found a compact first-plant endpoint that can unload the next front leg
within 78% of the weak motor cap, versus 125% for the earlier landing.
A new RL trial uses the compact first-front reference and has 34 cumulative
actual first-foot plants at step 1480, with its remaining interpolation
limitations recorded. Static analysis also found a second-front plant by
rocking the first planted foot. A separate 267-waypoint two-front trial
started at 20:27 PDT; it has crossed its first actual-plant gate by step 620.
At step 1300, the two-front trial has 25 first-foot plants and has reached
the second-front crossing, but no simultaneous front-pair hold. Rear-first
alternative-order poses now have feasible static allocations, although
connected takeoff and ascent remain unresolved. A further isolated trial
started at 20:48 PDT to preserve nominal joint posture motion through the
damped IK controller; it retains the two-front route, rewards and limits.
This is not a demonstrated two-front or whole-robot climb.
See [current IK progress](rl-ankle-v23-ik-progress.md),
[reward analysis](rl-ankle-v23-reward-analysis.md), and the
[first plant recording frame](../reviews/ankle-v23-curriculum/20260915/video-reference-loaded-s3291/first-foot-plant.png).
Four RL workers and the requested recorder are active. All remain in
simulation with a privileged teacher and D405 proxy.

Earlier continuation, September 17 at 19:00 PDT: no full climb yet. Two new
256-environment RL trials use state-driven pose targets on the same stairs:
`stairs-waypoints-256-s3281` for the rear transfer and
`stairs-front-rear-256-s3281` for grounded front/rear transfers with pitch
commands centered on the desired foot angle. Both allow 320-second episodes.
They retain physical limits and floor-only full-climb eligibility. The old
milestone run was stopped and update 128 archived after 4,870 ended episodes
without stage/full success. Four other comparisons continue. The
[IK continuation](rl-ankle-v23-ik-progress.md) records the front floor-to-tread
static candidate, rear landing limitations and exact reward/action changes.
No new video or hardware operation was run.

The user's latest instruction is to train only on full-height stairs. Do not
launch reduced-height stairs or near-goal hover/landing drills that practice
only a small foot drop. `realstairs-landing-seed3221` was deliberately stopped
after update 93 because its dominant reset practiced a 15 mm landing. Its
update-80 checkpoint is preserved under
`simulation/isaac/models/ankle-v23-realstairs-landing-explore/realstairs-landing-seed3221`.
The job's `early_stop.json` records the reason; its supervisor reports a failed
exit because the worker was intentionally stopped.

The force and sideways runs subsequently completed 448 updates each with
zero full climbs. The following reward comparison also finished without a
first-tread completion or full climb. The subsequent work used differential IK foot
actions and free foot order in the supported lift-milestone reward experiment
`stairs-milestones-512-s3281`, plus the
separate linear sole-pitch trial `stairs-linear-256-s3281`. A fourth trial,
`stairs-contact-margin-256-s3281`, started September 17 at 16:39 PDT. It
replaces the foot-center stability margin with a hull of near-surface contact
corners; static analysis found that the center triangle excluded mechanically
supported poses. It otherwise matches the linear trial's settings. At 16:55,
`stairs-knee-branch-256-s3281` replaced the oldest bounded-IK baseline after
its checkpoint was preserved. The new trial changes the floor RR knee bend
and corresponding static PD preload at unchanged floor footholds; static
unloading torque falls below the weakest motor cap. The earlier
Cartesian, readiness, and wider-tilt jobs were intentionally stopped and their
checkpoints archived after identifying reward, action-range, and independent
joint-clipping problems. All current runs include motor target/rate guards
inside the IK solve. The new reward pays each supported clearance milestone
once per foot and tread, with larger rewards for actual planting. The linear
pitch trial increases rotation explored by small actions while retaining the
same physical joint and motor limits. Offline pose-and-force
searches address rear-foot transfer, folded-foot clearance and staggered
support across different treads.
See the [IK continuation](rl-ankle-v23-ik-progress.md) for the current approach
and the [earlier reward analysis](rl-ankle-v23-reward-analysis.md) for the
completed comparison. No full staircase climb has succeeded. Historical
small-step artifacts below are retained only as the record of earlier work.

At about 17:29 PDT a fifth trial, `stairs-landing-align-256-s3281`, began.
It adds airborne corner-height alignment shaping after a foot crosses the
next tread edge, with all existing physical limits and grounded resets.
Offline calculations now connect a recorded 154 mm front-foot lift to a
loaded first-tread landing; a refit satisfies the sampled static force and
collision constraints. Rear-foot calculations have also reached actual floor
contact using the loaded support edges, but the full ascent remains unsolved.
These calculations are not executed gait demonstrations or new reset poses.

The later rear calculation reaches a slightly tilted, loaded first-tread
plant with the other three feet already above. It remains a static candidate:
intermediate poses reveal collisions, including the rear foot passing through
the front foot during final lowering. The detailed
[18:16 continuation](rl-ankle-v23-ik-progress.md) records the contact loads,
motor margins and remaining path defects. None of this establishes a climb.

## Current result

The [180 mm recording](../reviews/ankle-v23-curriculum/20260915/video-fullheight-seed3031/frozen-teacher-180mm.mp4)
shows one frozen deterministic teacher policy placing and holding **both front
feet on the first tread from a floor start**. The rear feet stay on the floor.
There is **no complete four-riser climb yet**. The policy uses a known-stair
reference and privileged state in addition to the D405 depth proxy; this is
not a camera-only policy or hardware result.

The recording uses loaded-run checkpoint `model_80.pt`, seed 3031, and a
76-second horizon. Its report records a qualified stage event at 71.58 seconds,
no fall, and peak joint-range excess 0.00001574 rad. The 77-second MP4 includes
the terminal pose. The checkpoint, SHA256, and report are preserved under
`simulation/isaac/models/ankle-v23-realstairs-recorded/realstairs-loaded-seed3021`.
This is one requested recording, not a reliability estimate.

The loaded baseline completed all 320 updates and archived `model_319.pt`.
It recorded 481 qualified stage holds in 1,022 ended episodes: 64 from floor
starts and 417 from curriculum starts. These training episodes can span policy
updates and are not fixed-checkpoint reliability measurements.

## New references and exploration arm

The supported arm transfers body weight after each front-foot placement.
Source inspection also found that the older second-foot planning calculation
did not explicitly constrain the tilted swing sole against the floor. The
simulation floor was present throughout, so the recorded dynamic result is
still valid. New reference calculations constrain the sole corners, start
phase two at the exact preceding endpoint, and move the clearance bump earlier
to avoid the first riser. A phase-boundary joint discontinuity was caught in
the reference calculations and corrected before launching this arm.

`realstairs-quiet-seed3051` copies that supported environment into an isolated
package and multiplies both Beta action parameters by four. This preserves
the deterministic action mean and roughly halves initial sampling standard
deviation. Sampling, log probability, entropy, and KL all use the scaled
parameters. It starts from the same loaded-run update-112 actor as the
supported arm. By update 75 it had 25 qualified floor-start holds, whereas
the original exploration arm had none at update 75. Seeds differ, so this
is a promising lead rather than a controlled causal estimate. The quiet arm
was stopped after update 184 to train a longer sequence. It had 95 floor-start
holds, 312 curriculum holds, and zero full climbs; update 176 is archived.
The original supported arm stopped after update 168; update 160 is archived.

The rear-left connecting path became admissible when forward movement and
descent were synchronized, instead of advancing at the full lift height.
`realstairs-rear-diagonal-descent-reference` satisfies all 91 sampled geometry
and load conditions. `realstairs-diagonal-seed3071` trains front-right onto
the first 180 mm tread, then rear-left from x=113 to 175 mm on the floor.
At update 146, it had 40 qualified floor-start holds and 201 curriculum holds.
The rear-left floor move is a preparatory step, not an additional stair climbed.
This diagonal arm has now completed all 320 updates: 1,011 ended episodes,
126 floor-start stage holds, 448 curriculum holds, and zero full climbs.
`model_319.pt` is archived. The new
[77-second frozen recording](../reviews/ankle-v23-curriculum/20260915/video-diagonal-seed3161/frozen-teacher-180mm.mp4)
reproduces this two-phase milestone from the floor with seed 3161. It reaches
the 76-second episode horizon without a fall and reports peak joint-range
excess of 0.00035429 rad. It still shows only one foot on the first tread;
rear-left advances on the floor. The recording report is copied beside the
archived checkpoint. It is one frozen recording, not a reliability estimate.

`realstairs-sequence-seed3101` adds front-left onto the same tread and then
shifts COM x toward 230 mm. The final static calculation supplies at least
4 N per foot. Its three 36-second reference phases use 112-second episodes,
384 planned updates, and a diagonal-run update-80 actor warm start. At update
45, it had 38 curriculum stage holds and no completed floor-start holds yet.
It subsequently produced one qualified floor-start three-phase hold, at
training step 11,543 in environment 60. The front feet were on the first
tread, rear-left had advanced on the floor, and rear-right remained on the
floor. The run was stopped after update 246; update 240 is archived.
Its actor continues in the new four-phase arm. The full-climb count remains zero.

The 45-degree first-foot reference also became admissible after moving its
landing inward, correcting initial sole clearance, and recomputing the full
path. Its maximum reference joint speed is 36.1 degrees/s. The initial
`realstairs-oblique45-seed3061` arm tipped backward at the first lift and was
stopped after update 48; update 32 is archived. A second arm added four
privileged normal-force observations, a weakest-support-foot reward, and a
4 N planned support-force minimum. That arm also fell in all 512 ended floor
episodes and stopped after update 62; update 48 is archived.

`realstairs-oblique-curriculum-seed3091` therefore mixes floor starts with
starts 15-72% through the first-foot motion. Only reset step zero is eligible
for a floor-start result. Its teacher has 512 actor inputs and 528 critic
inputs; the added forces are simulator state, not D405 measurements. At update
61, five curriculum starts reached the stage goal and no floor starts did.

That oblique curriculum arm was stopped after update 88 with no floor-start
stage holds; update 80 is archived. `realstairs-mean-policy-seed3121` then
transferred its recurrent encoder and normalization into a fresh Beta head
that parameterizes mean and concentration separately. It started at the same
Beta(80,80) distribution, retained action amplitudes and physical limits, and
stopped after update 76 with no floor-start holds. Update 64 is archived. This
short experiment does not establish that this policy representation cannot work.

`realstairs-balance-seed3141` adds bounded body pose feedback to the 45-degree
reference. Damped foot Jacobians map root position/orientation error into
stance-leg joint corrections, limited to 0.12 rad and filtered over 0.12 s.
The original motor effort, joint, and target-rate limits still apply. This is
a privileged teacher controller using simulator root state. It is not a
camera-only deployment controller. An earlier launch with seed 3131 failed
before training because its Windows provenance path was too long; the new
run uses the archived warm-start checkpoint and a shorter job name.

Further pure calculations found compact four-foot endpoints on the first
tread, with front centers at x=530 mm and rear centers at 395-415 mm. They fit
the sampled collision and joint constraints and use about 26-27% of the
weakest modeled torque cap. These endpoints do not establish a connecting
motion. A rear-right floor shuffle after both front feet land is also being
planned. The force calculation explicitly applies opposing kinetic friction
`Fx=-0.9 Fz` to the sliding foot, rather than treating it as freely movable.
The initial 91-pose path meets geometry/load conditions, but its unloading
transition was too abrupt for the intended slow motion. The revised 91-pose
path ramps the planned tangential force while the foot is still stationary,
then uses the full kinetic friction constraint during sliding. It satisfies
the geometry/load conditions and the combined four-phase reference remains
below the weakest modeled target-rate limits. `realstairs-shuffle-seed3151`
trains that four-phase sequence in 128 environments, for 448 planned updates,
warm-started from sequence update 192. This phase advances a foot on the floor
and is not counted as another stair climbed. The stage's lift-history rule
explicitly excludes the intentional rear-right shuffle; the four-riser goal
is unchanged.

The initial shuffle arm stopped after update 86 with no qualified stage holds;
update 80 is archived. Its last-phase traces show rear-right still carrying
roughly 11-15 N instead of the planned 2-3 N, and its foot remains about 140 mm
behind the target. `realstairs-force-seed3181` keeps the same four-phase path
and adds bounded normal-force feedback only during that shuffle. Foot-load
errors produce filtered joint-position corrections through damped Jacobians;
the controller applies no external forces. Its revised load plan keeps at
least 1.2 N per foot during transfer and 4 N at the endpoint, with peak static
torque ratio 0.962 at the weakest motor strength. Physical effort, rate, and
joint limits remain unchanged. It warm-starts from the completed diagonal
update-319 actor, so this is a combined controller/initialization experiment,
not a controlled estimate of the feedback's effect alone. The four additional
force observations are privileged simulator data.

An early force-feedback trace at training step 4,440 reaches rear-right
x=210.9 mm, compared with roughly 132 mm in the earlier shuffle arm. It still
misses the 275 mm target by about 65 mm and loses a planted front contact.
That trace starts partway through the sequence. It is evidence of movement,
not a qualified stage hold or a floor-start result.

The same arm subsequently produced its first qualified curriculum stage hold
at step 6,707 in environment 51, after starting at reference phase 1. The
rear-right sole reached x=250.5 mm, within the stage's 35 mm tolerance around
275 mm; both front feet were planted on tread one. It held for 1.0167 seconds.
The first ended batch of 55 floor starts had no qualified stage completions;
52 reached the 148-second episode horizon without falling. Full climbs remain
zero. These are training trajectories, not frozen-policy reliability results.

The balance-feedback arm reached its first qualified floor-start first-foot
hold at training step 1,851. By update 101 it had 17 ended floor-start holds
and 27 curriculum holds. These are still changing-policy training results.
It completed all 320 updates with 1,643 ended episodes, 186 floor-start stage
holds, 178 curriculum holds, 484 falls, and zero full climbs. Its final
`model_319.pt` is archived under `ankle-v23-realstairs-body-feedback`.
The [41-second angled recording](../reviews/ankle-v23-curriculum/20260915/video-oblique-seed3201/frozen-teacher-180mm.mp4)
uses that frozen deterministic policy with seed 3201. It reproduces the
first-foot placement from a 45-degree floor start: front-right holds tread
one for over one second at 30.75 seconds. The recording reaches its 40-second
episode horizon without falling, with peak joint-range excess 0.00037384 rad.
The other three feet remain on the floor; no full climb occurs. It is one
recording of a privileged simulator-state teacher, not D405-only control or
a reliability estimate. Its report is archived beside the checkpoint.

Allowing foothold locations to vary found a collision-free three-foot
first-tread endpoint at yaw -36.95 degrees, roll 23.43 degrees, and pitch
-18.46 degrees, all within the original 95% planning joint envelope. Its
calculated peak static torque is about 25% of the weakest motor cap with all
four feet loaded. Rear-right remains on the floor. Attempts to connect the
pose by holding the other footholds fixed have not produced admissible paths.
This is an endpoint calculation, not a reachable-pose or stair-climb claim.

`realstairs-explore-seed3171` is a separate, unscripted curriculum experiment
from that endpoint, using 256 environments and 512 planned PPO updates.
Instead of small corrections around an intermediate motion path, the actor
selects joint targets across 99.5% of the unchanged physical joint ranges.
Motor torque, target rates, collision settings, and fall limits are unchanged.
The 16-second episodes reward net body/loaded-foothold progress and a local
goal of all four traction patches planted on tread 1 or higher for one second.
The actor uses a direct mean/concentration Beta head initialized at the loaded
reset pose. Every episode starts partway up the stairs and is explicitly
ineligible for a floor-start full-climb result. No intermediate trajectory is
prescribed, and no success has been established for this experiment yet.
By update 64, all 1,024 ended curriculum episodes had lasted the 16-second
horizon without a fall, but none reached the fourth-foot stage goal. This
shows that the reset stance can persist in the dynamic model; it does not
show that the stance can be reached from the floor or that the last foot can
be transferred.

This direct-position arm was stopped after update 128 had been saved, with
more than 2,000 ended curriculum episodes, no falls, and no stage goals.
Its checkpoint and incomplete-run contract are archived. The foot remained
about 255 mm from its goal: the policy mostly preserved the initial pose.

`realstairs-velocity-seed3191` is a new isolated training arm with 128
environments and 448 planned updates. The actor now selects joint-target
velocities, integrated at 60 Hz and filtered over the same 0.12 seconds.
The initial policy has zero mean velocity and Beta(16,16) actions, with the
same gravity-compensated initial targets. Original physical position, rate,
and torque limits still apply. The actor additionally observes servo target
errors and rear-right goal error (531 actor / 547 critic inputs). Its reward
adds progress in unloading and raising rear-right, gated on the other three
feet remaining planted on actual treads. These changes are a combined
exploration strategy, not a controlled single-factor ablation. Every episode
remains a curriculum start and cannot qualify as a floor-start full climb.

This first velocity-action arm stopped after update 100; update 96 is archived.
It had 854 ended episodes, 150 falls, and zero local goals. Recent goal distance
stalled near 247 mm despite fewer falls. `realstairs-side-seed3211` separately
trains the same action/reward/controller configuration from a wider -57.19-degree
reset pose. That pose has three feet on tread one and rear-right on the floor.
All-four-contact static torque is 16.1% of the weakest cap; removing rear-right
requires about 100.2% in the ideal static solver, or 102.9% when bounded PD
setpoint headroom is also imposed. The pose is therefore a valid loaded reset,
not an accepted weakest-strength unloading path. Physical limits and drive
gains remain unchanged. It uses 128 environments and 448 planned updates.

The now-stopped `realstairs-landing-seed3221` mixed easier last-foot landing resets with the two
harder floor-foot resets. Reset probabilities are 60% with rear-right 15 mm
above tread one, 15% at 50 mm above it, 10% hovering before the riser at
x=285 mm / z=215 mm, and 7.5% at each 37/57-degree floor-foot pose. The other
three feet already start on the first tread in every case. The three hover
poses satisfy sampled collision, joint, static torque, and bounded PD target
conditions; calculated torque ratios are 0.574, 0.433, and 0.841 respectively.
Attempts to extend the same fixed-foothold hover family down to 100 mm or floor
height fail geometry/load conditions and are excluded. This is a mixed reset
curriculum, not a connected reference or a floor-start climb. Reset variant
indices are recorded in episode, event, trace, and exported trajectory data.
It retains the same 531/547 observations, velocity actions, rewards, physical
limits, and gains, with 128 environments and 448 planned updates.

The first two mixed-curriculum goal events occurred at steps 200 and 323, both
in the 15 mm hover variant. Saved training trajectories begin with rear-right
at z=196.9/198.1 mm (including reset clearance) and end at z=181.7/181.0 mm,
with all four patches at level one for over one second and zero recorded
joint-range excess. A third event at step 1,533 uses the same easy variant.
These establish some successful landing behavior during training. They do
not establish transfer from a floor foot, a floor-start ascent, a connected
climb, or fixed-policy reliability. The early run also has many falls.

[Smooth Exploration for Robotic Reinforcement Learning](https://arxiv.org/abs/2005.05719)
and [Colored Noise in PPO](https://arxiv.org/abs/2312.11091) support investigating
temporally coherent exploration. The present velocity-action arm is a simpler
experiment with integrated targets; it does not implement either paper's
noise algorithm. No improvement is assumed before its training evidence.

Free-foot calculations also found a two-left-feet-on-first-tread endpoint at
about -49 degrees yaw, with the right feet on the floor and peak static torque
about 28% of the weakest modeled cap. A rear-left hover at 205 mm can be fitted
near that endpoint, but its predecessor with rear-left on the floor could not
be fitted while keeping the other three footholds fixed. No connected motion
has been established. Free three-foot unloading poses have also been found
geometrically, but their initial torque estimates exceed the weakest modeled
cap by 13-17%; those are not accepted training references.

The dynamic alternatives discussed in
[Scout II step-climbing research](https://nereus.mech.ntua.gr/pdf_ps/clawar00.pdf)
and [quadruped jumping with RL](https://arxiv.org/abs/2011.07089) motivate
considering momentum and paired-foot motion when static support is limiting.
Those robots and actuators differ from V2.3; neither paper establishes that
this robot can jump these stairs. No jumping reference or higher-power model
has been launched here.

Rear-right transfer remains unresolved. Further calculations examined different
knee branches, low and diagonal swings, steeper body poses, 90-degree approaches,
and heel/toe/corner support. Edge/corner calculations include only the selected
contact points in their force optimization. None of those candidate rear-right
paths has been accepted for training. Local optimization failures do not prove
that no feasible gait exists.

## Servo load assumption

The [manufacturer's ST-3215-C018 specification](https://www.feetechrc.com/525603.html)
lists **10 kg·cm rated torque at 12 V** and **30 kg·cm stall torque**. The current
continuous simulation cap, 0.8825985 N·m, is 90% of the rated value. Episode
strength randomization further scales it by 0.85-1.0; the static reference
calculations use the weakest value. The torque-speed envelope is also retained.
No higher-torque or assumed thermal-burst training has been run. Several sampled
rear-right lifts require roughly 1.1-1.3 N·m in the static optimization, which
exceeds this cap. These are pose-specific load estimates, not a global minimum
for every possible climbing strategy.

## Motion planning before the new training stage

The 20 mm reference explicitly rejects changed geometry. Replacing the terrain
height alone would leave an invalid foot path, so this continuation computes
new references from the unchanged V2.3 URDF. The geometry calculations use
actual joint origins/axes, joint limits, nonadjacent collision boxes, stair
boxes, and contact-force/torque feasibility. They do not establish dynamic
success.

The approach follows the emphasis on coordinated body posture and foot motion
in [Optimized Static Gait for Quadruped Robots Walking on Stairs](https://linqi-ye.github.io/docs/2021Optimized%20Static%20Gait%20for%20Quadruped%20Robots%20Walking%20on%20Stairs.pdf).
Forward and 30/60/90-degree approach poses were considered. A forward first
landing could be reached with lower foot-position and stance-flatness errors
than the sampled angled configurations. This is a local pose calculation,
not a general rejection of sideways stair climbing.

`calculate_fullheight_reach.py` found a forward front-right lift to 205 mm and
landing at 180 mm using body roll/pitch. After self-collision correction,
the isolated lift used a calculated peak torque of about 95% of the weakest
modeled motor cap. The initially generated connecting motion failed geometry
conditions at 30 of 91 samples: the old rear-knee branch reached the rear-hip
limit, tilted stance feet, and briefly intersected the riser. That reference
was rejected before training.

`calculate_realstairs_reverse.py` instead starts with the rear-left knee bent
the other way. Its floor pose has unchanged foot positions, no detected
nonadjacent collision, and about 29% of the weakest motor cap in the static
calculation. The revised path passed all 91 sampled geometry/load calculations.
It was stretched from 18 to 36 seconds to reduce its maximum reference joint
rate from 123.8 to 61.9 degrees/s. The initial learning stage is one front-right
foot planted on the actual 180 mm first tread; stage completion is reported
separately from the four-riser goal.

The separately prepared `ankle_v23_realstairs` environment uses four 180 mm
collision stairs, a full 3D roll/pitch/yaw reference, and planned ankle angles
through the swing. It retains the motor effort/rate limits, self-collision,
240 Hz physics and 60 Hz control. The candidate damping remains 0.30 N m s/rad
and has not been identified from the physical servo.

`realstairs-first-fr-seed2971` launched 128 environments for 256 planned PPO
updates, warm-started from the earlier teacher actor. By update 83 it had
recorded **11 qualified first-foot holds**, all from floor starts, and no full
climbs. The hold requires a planted 180 mm tread contact for one second,
support contacts, reference posture, lift history, and the episode joint-range
guard. Common older placement telemetry assumes a nearly level body; this
stage detector instead follows the explicitly planned body inclination.

A second candidate path for the front-left foot initially intersected the
first riser with its tilted sole at six sampled poses. Adding 25 mm of local
clearance during transfer removed those sampled collisions and load failures.
`realstairs-front-pair-seed2981` now trains the combined two-phase path in 128
environments for 384 planned updates, warm-started from first-foot update 64.
It mixes floor starts and starts with the first foot already placed. Only
floor starts qualify for a full climb. Both-front-feet holds remain a stage
milestone; the full target still requires the fourth tread.

## Continued full-height work (September 16 evening)

The deeper first-foot arm `realstairs-first-margin-seed2991` moved its landing
target from world x=405 mm to 425 mm. It was stopped after more than 128 PPO
updates with no qualified stage holds; its update-128 checkpoint and incomplete
job contract are preserved. Trace samples include correctly positioned feet
that carry less than the required 1 N, so proximity alone is not success.

The original front-pair baseline was stopped after at least 192 updates and
605 ended episodes with zero qualified front-pair holds. Some surviving
episodes approach the target closely, but both front feet are often unloaded.

The contact-reward inspection found that a legacy retained-foot bonus depends
on a nearly level body and does not recognize the planned nose-up posture.
`realstairs-contact-seed3001` changes that bonus, adds stance corner-height and
support-load rewards, and tries bounded 3D ankle leveling. It was stopped early
around update 16 to address a separate servo setpoint issue. This short arm is
inconclusive; its intermediate checkpoint is retained.

`realstairs-headroom-seed3011` uses the corrected contact shaping and the
original 3D ankle reference. The planned pose stays inside 95% of the URDF
range, but servo setpoints may use 99.5% to supply PD load compensation near
the planned envelope. The URDF stops, motor effort and rate caps, actual joint
range success guard, and foot support criterion remain unchanged. Previously,
clamping the compensated target to 95% could remove the requested holding
torque at a reference boundary. This run uses 64 environments.

`realstairs-loaded-seed3021` adds a forward body weight transfer after the
second foot touches down. Its calculated endpoint puts COM x near 190 mm and
supports at least 4 N per foot, with a calculated peak joint torque below 40%
of the weakest modeled cap. All 91 sampled poses of that second phase satisfy
the geometry/load conditions. It uses 128 environments. These calculations
are not dynamic success evidence. Both new arms have 320 planned updates.

At the earlier loaded-run update 38, **30 qualified front-pair holds** had finished in
curriculum episodes that started with the front-right foot on the first tread.
The event stream subsequently reached 40 holds. At that early snapshot none starts
from the floor, and there are no full climbs. The first event is step 2068,
environment 127, with a 1.0167-second hold and both front feet at level 1.
This is training evidence; no fixed-checkpoint reliability claim follows.

Additional calculations found collision-free forward advances of the two
front feet on the first tread. Rear-foot sequencing remains unresolved:
several three-foot support poses exceed the motor caps. A rear-left-first
short advance has feasible isolated poses. Its first connecting path was
rejected for collisions; source inspection then found that the path used the
front-right goal x coordinate for the rear-left foot. That indexing error is
corrected in a new reference calculation, and the erroneous outputs are kept
as rejected artifacts rather than evidence against the intended gait.
No rejected reference is sent to training. The static force solver now retries a numerical HiGHS
failure with the interior-point method and checks returned equation residuals;
a numerical failure is not treated as proof of mechanical impossibility.

No separate automated tests, evaluation batch, hardware connection, or robot
motion was run. Authorized RL training, the requested video rollout, source
inspection, and pure geometry/static force calculations are the work performed.

## Requested video

`video-student-seed2961` records one frozen deterministic run of the saved
20 mm student, with the dimensions labeled in the frames. This is separate
from the earlier stochastic training success, which spanned policy updates.
The [recording](../reviews/ankle-v23-curriculum/20260915/video-student-seed2961/frozen-student-20mm.mp4)
completed and **failed at 306.22 seconds during the final rear-foot move**.
It did not reproduce the earlier changing-policy training success. The
[report](../reviews/ankle-v23-curriculum/20260915/video-student-seed2961/report.json)
records zero full climbs, a fall, and 0.00164 rad peak joint-range excess.
The terminal fall is preserved instead of showing a reset pose. This is one
requested recording, not a reliability benchmark.
