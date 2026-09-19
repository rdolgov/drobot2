# Full-height stairs: IK and RL continuation, September 17

The goal remains a complete ascent of four 180 mm risers and 250 mm treads.
No full climb has succeeded. All work here is simulation training or offline
geometry/static-force arithmetic; no hardware connection or motion was run.

## Latest continuation: September 18, 11:41 PDT

No complete climb or floor-start rear plant. A new floor-load-consistent
trial is training with corrected servo preloads for the 14 appended rear
unloading poses. The prior trial accidentally used an independent force
redistribution that loaded rear-left to 11.734 N instead of the intended
2 N. Both that trial and its action-authority comparison were stopped and
preserved. With the exact constrained-force targets, instantaneous
takeoff is compatible with the original policy command limits, including
the first filtered tick; no command gain increase is needed for that
calculation. This remains instantaneous arithmetic, not a proven motion.

The requested frozen second-foot actor video is complete. It achieves the
whole-first-tread hold at 175.1 s from the three-on-first grounded reset;
the 180 s clip limit arrives before a second-tread plant. The long delay
includes right-foot self-contact, and the nominal hold reference has only
0.474 mm separation between those foot boxes. The corrected floor trial
has recorded 25 measured rear-left unloading events from actual floor
starts at step 1,400, but measured front-right support is inconsistent
and no rear plant or complete floor transfer has been established.
An offline alternate RR landing location gives about 5 mm spacing with
69.57% of the weak torque cap; it is not installed and still needs an
approach from the preceding swing.
See the [current analysis](rl-ankle-v23-reward-analysis.md)
for the correction, active jobs and reference-clearance investigation.

## Earlier continuation: September 18, 10:58 PDT

One strict flat front-left plant on the second180mm tread has now occurred
in changing-policy training:2.275 N load, maximum sole gap5.087 mm, existing
0.25s hold. It starts from grounded reset2. The actor and trajectory are
preserved, and a requested frozen clip is recording. Other support degrades
and the right feet collide during the continuation; this is not a stable
whole-robot second-stage transfer or a complete floor-start climb.

Four RL workers remain active after the mature free-swing control was
archived to free a recording slot. Tilted-transfer has386 curriculum holds,
two tilted second-tread touchdowns and one flat second plant at the10:49
snapshot. Detour has one tilted touchdown; swing-clearance shaping has no
second-tread event yet. No floor-start rear plant was found in any of these
five runs. Offline arithmetic finds a wider30mm-buffer swing/crossing route,
but it has not been installed or executed. See the
[current analysis](rl-ankle-v23-reward-analysis.md).

## Earlier continuation: September 18, 05:00 PDT

No complete climb. The saved second-tread actor's requested frozen video
has completed: it failed the rear transfer, stalling at guide11 with a
28.24 mm sole error and1.54 N sideways contact against the first riser.
It did not reproduce the changing-policy training touchdown and did not
fall. See the [full-height video](../reviews/ankle-v23-curriculum/20260915/video-second-touchdown1-s3291/frozen-second-tread-180mm.mp4).

A separate33-pose detour reference reduces the first appended joint change
from52.72 to5.15 degrees, frees the airborne foot's XY position, and provides
a lateral path around the second riser. Four workers remain live. Detour
leaders have achieved the curriculum first-tread hold and reached guide115,
but now press into the second riser with2–3 N while other-three support is
good. The contact-norm advancement gate detects the problem; explicit
clearance/contact reward shaping is the next hypothesis.

From the actual floor-start front-pair footprint, a14-endpoint small posture
sequence unloads rear-left to2 N. Static complete unloading fails locally,
but a floating-base instantaneous inverse-dynamics calculation permits
takeoff with sideways/backward body acceleration at the original weak
motor cap. These are offline endpoints and instantaneous equations, not
an integrated or executed climb. See [current analysis](rl-ankle-v23-reward-analysis.md).

## Earlier continuation: September 18, 04:20 PDT

One actual tilted FL touchdown on the second180mm tread has occurred in
the grounded curriculum. It carries about2N with a22.6mm maximum traction
gap, so it receives no flat-foot credit. After the guidance fade, FR
actually lifts its whole foot8.49mm with measured support from the other
three feet; later support deteriorates. No floor-start transfer, flat
second-tread plant or complete climb has been achieved. The actor and
trajectory are archived, and one requested frozen full-height video is
recording; reproduction is still pending.

The new `tilted_transfer` RL branch treats a valid loaded tilted contact
as a higher support only for load-transfer shaping. The original term is
zero while every flat-foot credit remains1, missing the observed next-foot
takeoff. Temporary load levels do not alter plant or success criteria.
It inherits the separate `free_swing` correction that uses full contact
force magnitude, preventing downward self-contact from being rewarded
as unloading. An independent `light_bearing` trial uses1.5N bearing-force
saturation, with original contact and geometry checks.

Offline alternatives fully unload FL from a grounded pose, then reach
330mm absolute foot height using earlier support-foot rocking;360mm fails.
After tilted FL touchdown, rear-right-first unloading reaches1N but not
zero, and rear-left-first fails at8N. Local fixed-contact failure does not
rule out different postures or dynamic motion. Four training workers are
live. See the [current analysis and figure](rl-ankle-v23-reward-analysis.md)
and [new reward contract](../reviews/ankle-v23-curriculum/20260915/tilted-transfer-contract.json).

## Earlier continuation: September 18, 03:41 PDT

No full floor-start transfer, qualified second-tread plant or full climb.
The prospective-load reward now has 24 changing-policy curriculum holds;
its early actors are archived. A separate 256-environment branch continues
guidance through 23 static second-tread poses after the actual first-tread
hold. It retains original grounded resets and physical/success constraints.
The first appended pose has a 52.7-degree joint difference, so connecting
motion is a major unresolved limitation. A loaded tilted touchdown is
logged separately from a qualified flat plant or full climb.

Actual-vertex force redistribution does not solve the next FR unload on
tread2. It improves the floor-connection RR unload to 2 N. Four short
forward floor-reposition endpoints lower peak motor effort, but further
unloading still fails and sliding dynamics are unproven. The accepted
floor endpoint also exposes a reward mismatch: one supporting foot carries
1.23 N, producing only 0.409 support quality under the current 3 N scaling.
A hypothetical 1.5 N scaling gives 0.818; it has not been installed.

The landing-corridor run was preserved and stopped at 26 curriculum holds
to free the new branch's slot. Four workers remain live. See the
[current evidence](rl-ankle-v23-reward-analysis.md) and
[second-tread training contract](../reviews/ankle-v23-curriculum/20260915/second-tread-guidance-contract.json).

## Earlier continuation: September 18, 03:20 PDT

Full floor-start transfer and the four-riser climb remain unsolved. The
landing-reward, residual-handoff and landing-corridor trials have 134, 40
and 19 changing-policy curriculum holds at different training ages. The
oldest parent was preserved and stopped with 31 such holds. Its slot now
runs a reward-only prospective-load trial from the same event4 actor.

Offline mechanics finds a tilted FL touchdown on tread2 carrying 2–4 N;
an 8 N requirement fails. Modest loading permits a 16.6 mm heel gap at
61.2% of the weak continuous motor cap. Rolling the physical contact from
the sole edge to the tread edge reduces this to 14 mm in accepted static
samples. Larger flattening changes collide the right legs; a finer guarded
13 mm attempt instead reaches local joint/load constraints. Keeping FL
tilted and unloading FR next reaches 2 N, but not zero load. These results
do not establish a continuous or dynamically stable path.

The new RL reward adds a bounded potential for a lightly loaded next-tread
toe or corner before full flat-foot credit. It preserves the existing
flat landing reward and all strict success and physical constraints. It
is too early to judge the new trial, which has no placements at step280.
See the [mechanical figure](../reviews/ankle-v23-curriculum/20260915/second-tread-touchdown.png)
and [current analysis](rl-ankle-v23-reward-analysis.md) for force limits,
contact assumptions, research and the live experiment contract.

## Earlier continuation: September 18, 02:52 PDT

No floor-start first-tread transfer or full climb yet. The landing-reward
job has 82 curriculum holds; the residual-handoff job has 30, with at least 15 of
those episodes later falling. A full-height frozen recording of the saved
stage-event4 actor reaches rear guide97 but stalls without planting. FK
shows the full foot is safely over the tread while a 33 mm reference-point
offset exceeds the guide's 25 mm gate. A new isolated landing-corridor job
allows a bounded safe offset using the entire collision geometry; actual
landing and success criteria remain strict. It has admitted 42 otherwise
blocked guide advances and achieved three actual curriculum holds by step720.

Allowing all three supports to rock yields a static FL pose 10 mm above
the second tread at 83.9% of the weakest motor cap. Nine accepted crossing
samples put the whole foot inside tread2 at 70.9% of that cap. Direct
unloaded flattening fails; a tilted touchdown followed by load transfer
before flattening is the next mechanical hypothesis. No connected or
executed path is claimed. The latest floor rear-unloading search also fails.
All four training jobs, video, snapshots and
mechanical limitations are described in the
[current analysis](rl-ankle-v23-reward-analysis.md).

## Earlier continuation: September 18, 02:27 PDT

The landing-reward trial now has44 grounded-curriculum first-tread holds;
the original parent has20. Full floor-start transfer and four-riser ascent
remain unsolved. Post-stage logs reveal both a2.8x XYZ action-gain increase
and a change from ankle-rate residuals to absolute-angle commands during
handoff. An isolated256-environment trial preserves the residual action map
while the same nominal guidance fades. At step920 it has three curriculum
holds and zero second-tread foot credits. The two stalled1024-environment
rear-support branches were archived and stopped; all original constraints
remain. A new full-height clip of the saved stage-event4 actor is recording.

Offline second-tread calculations allow FL to lift120mm above tread1 when
FR rocks on its grounded corner; static peak torque is39.6% of the weak cap.
The150mm lift candidate fails joint/collision constraints. The accepted
endpoint needs a substantial body shift, so this is not a connected path or
an installed training reference. Alternate knee/foot orientation calculations
continue. See the [current analysis](rl-ankle-v23-reward-analysis.md) and
[geometry figure](../reviews/ankle-v23-curriculum/20260915/second-tread-mechanics.png).

## Earlier continuation: September 18, 02:03 PDT

The original pair-transfer run has now produced sixteen all-four-feet holds on
the first tread from the grounded three-on-first curriculum. Checkpoints
and trajectories are archived. None is a floor-start first-tread hold or
four-riser climb. The first four holds take 32.45–52.30 simulated seconds;
the later two take 70.85 and 56.55 seconds. The trajectory reconstruction
matches logged sole geometry within 9.1 micrometres, without replaying a
simulator. Frozen reproduction of these event actors remains unproven.
[Milestones](../reviews/ankle-v23-curriculum/20260915/pair-rear-stage-archive.json).

Only one of the 105 nominal rear poses satisfies the old clearance latch
that enables landing shaping. A new `rear_landing` trial removes that
prerequisite from an additional bounded alignment potential. It starts
from the archived rear-stage event4 actor, preserving all actor tensors
and using fresh critic/optimizer. `stairs-rear-land-s3281` is running at
step1,000 with eight curriculum stage holds, 53 floor-start front-pair holds
and zero full climbs; startup confirms all 14 actor tensors transferred exactly. The
full-height model56 recording finished at 90 seconds without rear credit,
fall or range excess. The controller, reference, resets, physical limits
and actual landing/stage/full-climb checks are unchanged.

The fixed-command spring/contact calculation admits a near-flat final
equilibrium within weak motor caps. The new retreat route's lift pose has
much less margin: the strong case saturates three motors, and the weak
search fails. These are approximate static calculations, not executed
paths. The FL-replant/four-foot load-shift extension still cannot release
RR below 1 N. A separate backward-planning screen from the grounded rear
reset identifies FL as a possible unloading choice. Two accepted static FL
lift endpoints use at most 47.3% of the weak cap with RR heel height <=40 mm.
Ten above-riser retraction samples are accepted, reaching 58.3%; the next
fails before crossing back to the floor. Extra foot rotation fails to
avoid the other leg; a lateral corridor avoids collision but still fails
the pose/equilibrium constraints. No connected floor route is
established. Detailed evidence and limitations are in the
[reward analysis](rl-ankle-v23-reward-analysis.md).

## Earlier continuation: September 18, 01:12 PDT

No full climb or whole-robot first-tread hold. Four training workers remain
active. Matched step-600 leader samples reach 160.9 mm rear-foot clearance
with the stance-anchor reward versus 144.3 mm in its baseline, but both
retain the support-foot collision and neither completes rear placement.
The selected records do not establish a success-rate improvement.

New offline work addresses the connection from the recorded front-pair
hold. Allowing supporting feet to rock admits two grounded hip-recovery
endpoints. Rear-right load can be reduced to 4 N in an accepted static
sample; searches for 1 N and effectively zero load fail. Unloading a front
foot is feasible at one recovered endpoint. Accepted static samples now
describe right-front lift/retraction/landing and left-front lift/advance.
The right-front landing still cannot support the next rear lift within
the torque cap. None of these endpoints or sampled paths was installed.

Read-only source inspection and primary PhysX documentation identify a
contact-model gap: ideal rigid-corner force allocation omits the finite
compression required by the existing compliant contacts. Recorded-gap
spring arithmetic has the correct force scale but is only approximate.
The next planning refinement should enforce contact/load consistency and
check whether same-tread repositioning enables a rear transfer.
See [evidence and reward implications](rl-ankle-v23-reward-analysis.md).

No active training source, reference, reset, physical limit or success gate
was changed. Work consisted of existing authorized RL training, offline
mechanics calculations and source/log research; no standalone tests,
new video rollout, evaluation batch or hardware check ran.

The left-front landing continuation also admits a static endpoint at
X=0.4500 m and at most four degrees of tilt, with three-support utilization
57.6%. Its changed footprint reduces torso pitch to about -39.3 degrees,
but neither rear leg can yet unload at that fixed pose. A further body/load
shift and compliant-contact refinement remain necessary hypotheses. This
endpoint is not installed in any runtime or reset.

At 01:12, live worker IDs were 15572 (pair-transfer), 33248 (rear-unload),
14500 (rear-support), and 30508 (stance-anchor). Actual plant-event logs
showed zero newly credited rear-foot landings in each run. All offline
calculation sessions from this continuation have finished.

## Earlier continuation: September 18, 00:48 PDT

The new rear-support arm reached roughly 140–150 mm whole-foot lift, then
lost front-left support while that foot drifted into the next riser.
Recorded-state geometry finds 0.06–0.10 mm plastic overlap in selected
step-300/400 states, versus about 12.5 mm nominal clearance. At step 620,
maximum rear height was 144 mm and five rear samples were still unloaded
above 30 mm. Neither rear landing nor a whole-robot stage has succeeded.

The next comparison, `stairs-stance-anchor-s3281`, is now training with the
same seed, actor, 1,024 environments and controller as rear-support. It adds
only a bounded potential for supporting soles remaining close to their
intended XY footholds during the guided rear transfer. Existing support
quality gates it, and existing handoff fades it. No physics, reference,
reset or success threshold was relaxed. Its outcome is still unknown.
See [reward and mechanics details](rl-ankle-v23-reward-analysis.md).

The older rear-retreat arm was preserved at saved update 16 and stopped
at iteration 18. Four jobs remain: pair-transfer, rear-unload, rear-support,
and stance-anchor. The early rear-support checkpoint is preserved under
`simulation/isaac/models/ankle-v23-realstairs-rear-support/stairs-rear-support-s3281/early-rear-lift`.
No new video rollout or standalone verification was launched.

## Earlier continuation: September 18, 00:42 PDT

No full climb or whole-robot first-tread hold yet. The new
`stairs-rear-support-s3281` run (1,024 environments, worker 14500) reduces
guided corrections on the three supporting legs to one tenth while keeping
full swing-foot authority. It retains the exact transferred actor, rear
unloading reference, rewards, grounded resets and physical limits.

At step 200 its rear-foot minimum height reached 136.7 mm; 15 of 514 current
rear-curriculum samples were above 30 mm with less than 1 N upward load.
This is further partial lift from a three-on-first-tread start, not a
floor-start climb, completed rear transfer, or frozen policy reproduction.
The larger population and changing policy preclude attributing the gain
solely to the support correction scale.

Static searches found only a modest motor-margin improvement, from 96.48%
to about 94.55% of the weakest cap, without proving a connected motion.
Recorded-state force calculations distinguish commanded saturation from
possible alternative load allocations. Read-only asset inspection found
no material USD/URDF inertial mismatch. Details and reward implications
are in the [reward analysis](rl-ankle-v23-reward-analysis.md).

The mature clearance10, smooth-handoff and rear-residual arms were archived
and stopped to concentrate compute. Their saved updates are 48, 32 and 16;
later unsaved updates are not claimed preserved. Four runs now remain:
pair-transfer, rear-retreat, rear-unload and rear-support. No active runtime
package was edited. No standalone tests, evaluation batch, new recording,
or hardware check was launched in this continuation.

## Earlier continuation: September 18, 00:15 PDT

**No complete climb or whole-robot first-tread hold yet.** Two new rear
comparisons are running: `stairs-rear-retreat-256-s3281` (worker 34344)
and `stairs-rear-unload-s3281` (256 environments, worker 33248). Both
retain the same exact transferred front-pair actor, grounded reset,
physical bounds, reward and measured clearance gates. The second changes
only seven moving rear targets to lift 0.5–5 mm while retreating, because
the first remains loaded on the floor at its retreat gate. In the first
run's step-340 snapshot, all 126 rear-curriculum environments were at that
gate, with median rear load 7.18 N and plastic clearance 1.02 mm.

The overlapping lift/retreat prefix has seven accepted static samples;
its last sample uses 96.48% of the weak continuous torque allowance.
No lifted reset was introduced. The first two launches failed before any
policy execution (source encoding, then Windows provenance path length);
both logs are retained. The corrected shorter-name run confirms exact
actor transfer and is training. Its result remains unknown.

Early recorded evidence is encouraging but transient. At step 60 (three
simulated seconds), the unloading run reached rear guide index 12 and a
maximum whole-foot clearance of 10.9 mm. At the matched pre-update step
100, two sampled rear loads were 2.52 and 4.58 N, compared with 7.00 and
10.06 N in the floor-retreat run. Ten of 125 current rear environments
had passed the retreat gate, but none still had both rear force below
1 N and whole-foot clearance above 5 mm; maximum clearance had fallen
to 4.71 mm. This supports the unloading hypothesis while showing that
the lift has not yet been sustained. These are changing-state,
reference-assisted training samples, not frozen success trials.
[Matched early analysis](../reviews/ankle-v23-curriculum/20260915/rear-unload-initial-analysis-step100.json).

The aggregate-load arm was preserved at update 40 and intentionally stopped
for this comparison. Checkpoint SHA256:
`0a9b9c1f0c183753b006776580dbecef6a01c07dede938f521a3e04086223aad`.
At its archive snapshot it had 54 cumulative pair events at step 5260,
with no stage/full success. Administrative status is `stopped_early`;
the raw interrupted-worker report is retained. Six live runs now remain:
clearance10, smooth handoff, pair transfer, rear residual, rear retreat
and rear unload. The rear-residual arm's first pair checkpoint (step 794)
was also preserved; it has no new frozen recording.

[Unloading comparison](../reviews/ankle-v23-curriculum/20260915/rear-unload-contract.json)
· [Retreat hold snapshot](../reviews/ankle-v23-curriculum/20260915/rear-retreat-initial-hold.json).
Next analysis should compare early rear foot force and actual whole-foot
clearance at matching logged steps in these two runs, then inspect whether
either reaches a real rear plant. Their event counts are not independent
success rates. No separate test suite, evaluation batch or hardware check
was run.

## Earlier continuation: September 18, 00:03 PDT

**No complete climb or whole-robot first-tread hold yet.** The new
`stairs-rear-retreat-256-s3281` training run is active, worker PID 34344.
Its startup confirms exact transfer of all 14 tensors from the reproduced
clearance10 front-pair actor. The original grounded resets, floor-front
route, full residual action authority, physical bounds and success gates
remain. The new rear guide has 122 static samples, a measured plastic-sole
retreat gate and a bounded support-gated clearance potential.

The route pulls the foot back, lifts it to 195 mm above the floor, crosses
the 180 mm riser, plants at no more than four degrees of tilt and lowers
the supporting heels. The first fully flat landing failed; that sample
was excluded. The accepted settling endpoint keeps every traction corner
within 5.6 mm of the tread and has a bounded four-foot static force solution.
All connecting dynamics remain unproved. The early training snapshot at
step 40 shows the new guide waiting at the retreat gate (index 7), with
sampled plastic clearances of 0.4–1.8 mm rather than advancing into lift
despite insufficient clearance. This confirms the gate is active, not
that rear transfer has succeeded.

The older 1024-environment exact-IK baseline was archived and intentionally
stopped to allocate this trial. Preserved checkpoint `model_72.pt` SHA256:
`5ce00c2cdf9e75b8eb60cb3411ed9e0a50f615f7e5229ef83dac531dadf7f2f7`.
At the archive snapshot it had 232 cumulative pair events at step 10360
and no stage/full success. Later unsaved updates are not claimed preserved.
Its raw stopped-worker report is retained, and administrative status is
`stopped_early`. Six runs remain: clearance10, aggregate load, smooth
handoff, pair transfer, rear residual and rear retreat.

[Reward and geometry contract](../reviews/ankle-v23-curriculum/20260915/rear-retreat-contract.json)
· [Near-flat landing](../reviews/ankle-v23-curriculum/20260915/rear-retreat-landing.json)
· [Support settling](../reviews/ankle-v23-curriculum/20260915/rear-retreat-settle.json).
Only requested RL training and offline analysis were run; no separate
test suite, evaluation batch or hardware check was launched.

## Earlier continuation: rear riser obstruction

No whole-robot first-tread hold or complete climb has succeeded. The exact
actor-transfer arm recorded its first front-pair event at step 794 and that
checkpoint was archived. The previously completed clearance10 frozen video
remains the reproduced floor-start result: both front feet planted at 39.85 s.

Full collision-box reconstruction shows why a rubber-pad-only clearance
measure can be misleading. At rear-residual step 200, two sampled rear-right
plastic soles overlap the first vertical riser by 0.295 and 0.471 mm while
their inset rubber pads remain about 6 mm behind it. The earlier side-impact
logs independently contain rear-foot horizontal normal forces of 1.02–2.64 N
with zero upward component at rear guide indices 4–12. These are selected
diagnostic samples, not an incidence rate. They support a riser-contact
bottleneck, without proving it is the only cause of failure.

Seven offline grounded retreat samples increase plastic-sole clearance from
the original 0.2 mm to 10 mm with the three supporting pivots and edges fixed.
All pass the existing static geometry and weak-motor force constraints;
the last sample still uses 98.27% of the weak continuous torque limit.
A subsequent full-foot lift calculation is in progress. No new reset or
reference has been installed from these samples.

The one-tenth residual trial confirms reduced target drift but has not yet
improved rear transfer. At the same pre-update step 100, the parent had one
ended fall and the smaller-residual arm five. At step 480 the latter had
144 ended falls, 129 from rear curriculum starts; no first-tread success.
These early results do not establish a training-success rate or causation.

Evidence: [riser geometry](../reviews/ankle-v23-curriculum/20260915/rear-collision-geometry-analysis.json),
[recorded contact forces](../reviews/ankle-v23-curriculum/20260915/rear-riser-force-analysis.json),
[grounded retreat](../reviews/ankle-v23-curriculum/20260915/rear-riser-retreat.json),
[same-step residual comparison](../reviews/ankle-v23-curriculum/20260915/initial-rear-residual-comparison.json).

## Earlier continuation: 23:35 PDT

**The complete climb and whole-robot first-tread hold remain unsolved.**
The new `stairs-rear-residual-256-s3281` trial started at 23:34 with the
same exact transferred pair actor. It changes only the guided rear
curriculum's residual action scale to one-tenth of the pair-transfer
comparison. Nominal reference motion, the front-floor route, unguided
actions, rewards, resets, motor/joint limits and success gates are retained.
Worker PID 16404 is live and its startup confirms all actor tensors match.

Recorded stable-index rear traces have 0.50–0.79 degrees RMS temporal
servo-target variation over selected 10-second windows. At fixed joints,
this corresponds to 0.26–0.42 Nm with Kp=30; it is not measured dynamic
torque variation. The initial rear reference has only about 2.05% static
torque margin at the weakest motor setting. The comparison asks whether
lower residual variation allows that load transfer to occur. Source data
also rule out a reset/reference mismatch, and lowering static minimum
support force did not improve these reference torque optima. See the
[reward analysis](rl-ankle-v23-reward-analysis.md).

The rear-bearing arm was archived at update 40 and intentionally stopped
for this allocation. Its checkpoint SHA256 is
`626a20839c7b8a8e71988e27c689fd60a5eb0c1308dc701bb090b8a325ee6554`.
At the archive read it had 43 cumulative pair events at step 5280 and no
stage/full success; this is not a convergence verdict. Six runs remain:
exact IK, clearance10, aggregate load, smooth handoff, pair transfer and
rear residual. No additional evaluation batch, automated test or hardware
check was run.

## Earlier continuation: 23:18 PDT

**No whole-robot first-tread hold or complete climb yet.** The new
`stairs-pair-transfer-256-s3281` run continues the actor from the completed
clearance10 recording. The training process confirms that all 14 actor
tensors, including the recurrent network, action head and normalizer, are
identical immediately after loading. The critic and optimizer are fresh.
This combines clearance10's route with aggregate raised-foot loads,
bearing-contact shaping and the observable two-second handoff. It is a
combined performance experiment, not an isolated reward comparison.
[Transfer contract](../reviews/ankle-v23-curriculum/20260915/pair-transfer-contract.json)
· [Actual training transfer record](../reviews/ankle-v23-curriculum/20260915/stairs-pair-transfer-256-s3281/train.metrics.actor_transfer.json).

The side-impact arm was preserved at `model_40.pt` and intentionally stopped
to allocate this run. Its checkpoint SHA256 is
`9114e2618bef7aafd5fa070722e9722d6b30a368c0459847e6c0c0b560028b57`.
It had 51 cumulative pair events at step 5260 and no stage/full success.
This stop is compute reallocation, not a claim of convergence failure.
Six runs remain: exact IK, rear-bearing, clearance10, aggregate load,
smooth handoff, and pair transfer. The new worker is PID 15572, started
at 23:16:15; its first read at step 100 confirms training is advancing.
The separate fresh smooth-handoff run achieved its first front-pair event
at step 870 in environment 70; five events were logged by step 900, and
the first-event checkpoint and source provenance are archived. This is
changing-policy training evidence, not a frozen reproduction or rear lift.

An offline extension now permits contact forces at actual plastic-sole
corners. The rubber pad is inset 3 mm and protrudes 1 mm, so sufficiently
large rocking angles change which geometry touches first. Eight grounded
rear-takeoff endpoint searches, retaining an original swing-floor pivot,
failed the existing acceptance constraints. None was installed as a
reference or reset; no running training geometry was changed.
[Outer-sole endpoint calculations](../reviews/ankle-v23-curriculum/20260915/outer-sole-takeoff.json).

## Earlier continuation: 23:04 PDT

**The frozen clearance10 policy reproduced both front feet on the first
180 mm tread at 39.85 seconds.** The recording starts with all four feet on
the floor and finishes at 40.85 seconds, one second after the qualified
pair hold. There is no fall, whole-robot first-tread hold or complete climb.
This is one deterministic seed, not a success-rate estimate.
[Video](../reviews/ankle-v23-curriculum/20260915/video-clearance10-pair-s3291/frozen-floor-start-180mm.mp4)
· [Recording report](../reviews/ankle-v23-curriculum/20260915/video-clearance10-pair-s3291/report.json).

The frozen checkpoint SHA256 is
`3b9a6f3fa9d254c78ac14eb9a3b09774ab7537d9626cb594b8adf6000cf238eb`.
At training step 1160, clearance10 had 67 pair events versus 24–31 in the
earlier 256-environment comparisons. Those comparisons also differ in
controller or reward, so this is not an isolated causal estimate. By step
1720, clearance10 had 79 cumulative pairs. All policies remain privileged,
reference-assisted teachers with a simulated D405 proxy.

The new smooth-handoff job started at **22:58:01** after the recorder exited;
its worker is PID 34660. It retains the clearance10 reference and reward,
blends reference assistance away over two seconds after measured completion,
and exposes that transition in the phase observation. This is now the sixth
active training run; no result from its handoff is established yet. The
aggregate-load trial has also achieved a first pair at step 916 and its
checkpoint and provenance are archived. Rear transfer remains unresolved.

Offline reconstruction of the completed recording identifies another
constraint: both rear hip joints are near their −60-degree stops. With no
foot contact, static gravity compensation at those exact joint positions
would need about −60.405-degree PD targets, outside the permitted range.
The force model supports the four-foot pose but cannot unload either rear
foot there. A body shift and joint repositioning must precede a static lift;
the calculation does not rule out other poses or dynamic motion.
[Recorded-pose force analysis](../reviews/ankle-v23-curriculum/20260915/frozen-front-pair-force-analysis.json).

Only requested training/recording and offline calculations were performed;
no additional evaluation batch, test suite or hardware check was run.

## Earlier continuation: 22:39 PDT

**The full climb and whole-robot first-tread hold remain unsolved.** The
clearance10 trial has begun first-front planting; its revised second-front
crossing has not yet produced a demonstrated improvement.

A bounded-force calculation identified a reward conflict at rear-guide
poses 6, 8 and 10. Saturating the old minimum raised-foot load fraction
requires every support foot to carry at least 6.757 N. At those poses this
needs 13–15% more torque than the allowed weak-motor cap, although uneven
loads can unload RR within the cap. A new aggregate-load trial started at
22:32:58. It rewards valid total raised-foot load before saturation, while
keeping each candidate's three-contact support requirement and strict
landing gates. It is an isolated reward change from the rear-bearing arm.
See [reward analysis](rl-ankle-v23-reward-analysis.md).

The five active runs are exact IK (1024 environments), rear-bearing,
side-impact penalty, clearance10, and aggregate load (256 each). The
post-pair edge-only run was intentionally stopped after preserving
`model_32.pt`, its source provenance and milestone checkpoint. This was
compute reallocation, not a convergence result. Its final checkpoint SHA256
is `45d394dd1e95c6ec582f8ce0f4fc224a4a76b830362f42f7bf0266a7eec7d9b0`.

The grounded rear-pose search initially failed with a tighter pre-edge
margin and infeasible force seed. Retaining the original 0.2 mm pre-edge
margin and starting from an independent feasible force allocation gives
two accepted static endpoints near 0.9704 of the weak-motor cap. Their
original pose is about 0.9830 under the same 3 N support minimum. This is
only a small margin improvement, and the connecting body motion remains
unproven; no new rear reference or reset was installed.
[Offline endpoint calculation](../reviews/ankle-v23-curriculum/20260915/rear-curriculum-torque-margin-r2.json).

Read-only arithmetic also found a sizable command discontinuity when
front-pair guidance ends. A gradual observable handoff is a later candidate;
it has not been changed in any active run. No additional evaluation batch,
test suite, benchmark or hardware check was run.

## Earlier continuation: 22:25 PDT

**No whole-robot first-tread hold or complete climb has succeeded.** Five
reference-assisted RL jobs are now active on four 180 mm risers and 250 mm
treads. The latest frozen video still establishes only one front-foot plant.

The new side-impact logs directly identify the front-left stall. At step
1500, the eight most affected examples are at guide indices 239/240; their
FL world-X contact forces are approximately −2.93 to −3.38 N and Z forces
are zero. Isaac Lab defines this field as **net normal contact force**, with
tangential friction separate. This supports contact against the vertical
riser. These severity-selected examples are not a population success rate.
See [contact evidence](../reviews/ankle-v23-curriculum/20260915/second-front-horizontal-contact-evidence.json).

Offline constrained pose/force calculations found a revised FL sequence
with **10 mm whole-foot clearance**, crossing, flattening, and landing.
All 53 second-front samples have independent bounded static allocations;
the largest peak torque ratio is 0.663753 against the 85%-strength motor
cap. A 15 mm clearance attempt failed its constraints. Neither discrete
sequence proves feasible connecting motion. A new isolated
`stairs-clearance10-256-s3281` job started at **22:25**, using the 10 mm
sequence with the exact-IK baseline's reward and controller. The existing
first-front route, floor reset, grounded rear curriculum, robot limits,
actual-contact gates and full-climb criteria are retained.
See [reference arithmetic](../reviews/ankle-v23-curriculum/20260915/clearance10-reference-arithmetic.json).

The rear-bearing trial started at **22:07**. It extends the existing
edge-support shaping to the grounded rear curriculum. At step 960, only
18 of 132 rear examples fail the support gate, but median guide index is
still 7 and swing error is 31.4 mm. The higher support score follows the
changed definition; it is **not evidence of better balance or climbing**.
The foot-tracking arm was archived at update 32 and intentionally stopped;
the larger exact-IK baseline continues. No active package was edited.

| Run | Last read step | Cumulative front-pair events |
| --- | ---: | ---: |
| `stairs-exact-ik-1024-s3281` | 4980 | 169 |
| `stairs-edge-support-1024-s3281` | 3520 | 146 |
| `stairs-stumble-256-s3281` | 2000 | 29 |
| `stairs-edge-curriculum-256-s3281` | 960 | 13 |
| `stairs-clearance10-256-s3281` | Starting | — |

These are unequal-exposure changing-policy event counts. All strict stage
and full-climb counters are zero. The stumble and rear-bearing first-pair
checkpoints are archived with provenance. Their first events occurred at
steps 895 and 852 respectively; neither has frozen-policy reproduction.

The deeper-FL placement search found a strict static sample at X = 400 mm,
Z = 183 mm. Farther samples retained small normal residuals after polishing
and were not accepted under the original criteria. Screening three nearly
flat X = 520 mm candidates found no rear-unloading force allocation under
the same motor and setpoint limits. This does not establish global
infeasibility, and no deeper-replant route was installed.

Work remains authorized training, requested full-height recording, source
inspection and offline arithmetic. No extra test suite, build, evaluation
batch, benchmark or hardware check was run. These are privileged simulation
teachers with a D405 proxy, not camera-only policies.

## Earlier continuation: 22:00 PDT

**The full climb is unsolved. The frozen pair-checkpoint recording finished
without reproducing the front-pair milestone: it plants FR, then stalls at
the second-front crossing.** The 90-second clip has no fall or stage success
and contains one extra second of final-frame hold. Training pair events must
remain separate from frozen-policy evidence.

[Latest full-height video](../reviews/ankle-v23-curriculum/20260915/video-posture-damping-pair-s3291/frozen-floor-start-180mm.mp4)
· [Recording report](../reviews/ankle-v23-curriculum/20260915/video-posture-damping-pair-s3291/report.json)

The frozen stall gives a specific mechanical clue. Reconstructing body pose
from the logged four sole centres and joint angles reproduces those centres
within 0.14 micrometres. At step 1320, the wider plastic FL sole reaches
X = 340.178 mm at the first riser, while the traction patch reaches only
335.272 mm. The sole overlaps the stair box by about 0.178 mm in the rigid
geometry calculation. Upward foot force is zero and the hip is heavily
loaded. This supports a riser-contact explanation, rather than establishing
it from direct horizontal-force logs. See
[reconstructed recorded pose](../reviews/ankle-v23-curriculum/20260915/frozen-second-front-stall-pose.json).

An isolated side-impact reward trial started at **21:54**. It clones exact
IK and adds a bounded cost based on horizontal foot force exceeding five
times absolute vertical force, with a 1 N deadband and 4 N ramp. Maximum
cost is 0.5 per second. It also records full foot-force vectors so the next
training evidence can check this mechanism directly. The exact-IK baseline,
sharper-foot arm and edge-support arm continue. No landing gate, physical
limit or stair size was changed. The earlier posture-damping arm was stopped
after preserving `model_32.pt`, full provenance, and its front-pair milestone;
this was a compute reallocation, not a convergence claim.

| Active run | Last read step | Cumulative front-pair events |
| --- | ---: | ---: |
| `stairs-foot-tracking-256-s3281` | 3680 | 41 |
| `stairs-exact-ik-1024-s3281` | 3020 | 140 |
| `stairs-edge-support-1024-s3281` | 1580 | 75 |
| `stairs-stumble-256-s3281` | 80 | 0 |

Counts have unequal exposure and are not success rates. All whole-robot stage
and full-climb counters remain zero. The edge-support first pair occurred at
step 860, and its checkpoint/provenance are archived.

The rear-right pivot continuation now has accepted static samples through
185 mm clearance, but original-floor takeoff remains unresolved. Diagonal
rear-left-second continuation improved from 100 to 130 mm clearance; the
140 mm attempt failed. Both front feet separately have static lift samples
above the second tread at 365 mm clearance, but the two front-right crossing
attempts failed even near their starting position. These samples are not
connected executable trajectories. A new calculation instead tries moving
FL deeper onto the same first tread, then checking rear unloading. No new
rear/second-tread reference has been installed from these incomplete routes.

Sources and reward calculations are in the
[reward analysis](rl-ankle-v23-reward-analysis.md). Work remains authorized
simulation training, requested video, and offline calculations; no extra
test suite, build, evaluation batch, benchmark or hardware check was run.

## Earlier continuation: 21:38 PDT

**Three reference-assisted RL runs now have actual front-pair holds from floor
starts. No whole-robot first-tread hold or complete ascent has succeeded.**

| Active run | Logged step | Cumulative front-pair events |
| --- | ---: | ---: |
| `stairs-posture-damping-256-s3281` | 3620 | 55 |
| `stairs-foot-tracking-256-s3281` | 2460 | 38 |
| `stairs-exact-ik-1024-s3281` | 1880 | 103 |
| `stairs-edge-support-1024-s3281` | 620 | 0 |

These are changing-policy event counts at unequal training exposure, not
success rates or an equal-budget reward comparison. Each of the first three
arms has an archived first-event checkpoint and source provenance. The edge
support arm started at 21:26 and has begun planting its first front foot.
Its reward change applies only after an actual front-pair hold.

The static rear-right candidate permits support-foot rocking up to 32 mm of
corner-height spread. One pose satisfies the existing geometry, contact,
upright, joint, servo-headroom and weak-motor constraints: independent force
allocation gives peak torque ratio **0.951936**. The body turns about 26.9°,
rolls −18.2° and pitches −29.8°. However, the rear foot is already about
97 mm above the floor. It is an airborne endpoint, not a takeoff solution.
Continuation has accepted discrete samples through 150 mm clearance; the
80 mm lowering attempt failed the constraint tolerance. All eight searches
that fixed a rear-right corner to its original floor contact also failed.
None of these rear samples has been installed as an executable reference.
See [pivot search](../reviews/ankle-v23-curriculum/20260915/compact-rear-extended-pivots.json),
[continuation](../reviews/ankle-v23-curriculum/20260915/compact-rear-pivot-continuation.json),
and [floor takeoff search](../reviews/ankle-v23-curriculum/20260915/compact-rear-ground-pivot.json).

A separate calculation is trying diagonal order: front-right, then rear-left.
It continues the previously accepted 100 mm rear-left lift while allowing
the other supporting feet to rock about fixed contact points. This addresses
the geometry before committing to both fronts on the tread. Connecting the
body shift from the preceding first plant remains unresolved.

At the accepted rear-right endpoint, the previous support-quality formula
returns a minimum of **0.2716**, because it penalizes every raised corner.
The new trial recognises a loaded near-plane corner for temporary support,
after the measured front-pair hold. It requires upward measured load, one
horizontal surface, no large penetration, and at most 40 mm heel rise.
Support-position shaping uses near-plane corner locations. Full-foot landing
credit and stage/full-climb gates are unchanged. See
[reward arithmetic](../reviews/ankle-v23-curriculum/20260915/edge-support-reward-arithmetic.json).

Only two stored leader states captured the common second-front stall at
waypoints 239–240. Their median foot error was approximately −21.9 mm X,
−16.5 mm Y and −2.9 mm Z. This limited sample points toward horizontal
tracking error; it does not establish the cause across all stalled replicas.
The planned poses have no detected nearby nonadjacent front-left self
intersections and about 5 mm separation from the first stair. The USD has
no authored contact/rest-offset attributes in the inspected front-left
subtree; no simulator setting was inferred or changed from that absence.

The requested original frozen-policy video is **finalized**: 321 seconds at
real-time playback, one front-right tread credit, no fall, no whole-robot
stage or full climb. It uses 180 mm risers and 250 mm treads. A new frozen
progress recording uses the archived posture-damping pair checkpoint and
the same floor seed. It stops one second after a front-pair hold, at the
first environment terminal, or at 90 seconds, whichever comes first. At
27.05 seconds it had one front-foot credit and no fall; the pair had not
yet reproduced. Both use reference assistance, privileged RL corrections,
and the simulated D405 proxy.

[Completed full-height video](../reviews/ankle-v23-curriculum/20260915/video-reference-loaded-s3291/frozen-floor-start-180mm.mp4)
· [Video report](../reviews/ankle-v23-curriculum/20260915/video-reference-loaded-s3291/report.json)
· [New recording report](../reviews/ankle-v23-curriculum/20260915/video-posture-damping-pair-s3291/report.json)

No additional test suite, build, separate benchmark, hardware check or
policy-evaluation batch was run. Work consists of the authorized RL jobs,
requested recording, source/log inspection and offline geometry/force math.

## Earlier continuation: 21:16 PDT

**Both front feet now reach and hold the first 180 mm tread in ordinary
floor-start RL training. Rear transfer, whole-robot first-tread hold and
the four-riser climb remain incomplete.**

The posture-damping trial's first simultaneous front-pair hold occurred at
step 847 (42.35 simulated seconds), environment 200. FL/FR sole heights were
180.640/181.183 mm, forces 2.645/6.424 N, and maximum corner gaps
1.481/2.756 mm. The runtime gate also requires both tread credits, low slip,
0.25 seconds of contact, root speed below 80 mm/s and retained range/upright
criteria. This is changing-policy evidence, not a frozen-policy reproduction.
At step 1760 it had 36 cumulative pair events and 20 current guide completions.
The first event checkpoint and complete training provenance are archived in
`simulation/isaac/models/ankle-v23-realstairs-posture-damping/`
`stairs-posture-damping-256-s3281/front-pair-milestone/`.
See [pair-hold evidence](../reviews/ankle-v23-curriculum/20260915/front-pair-milestone-evidence.json).

| Current job | Purpose | Started PDT |
| --- | --- | --- |
| `stairs-posture-damping-256-s3281` | Reference-centred damping; first demonstrated front-pair hold | 20:48 |
| `stairs-foot-tracking-256-s3281` | Only tighten guided swing-foot similarity scale from 50 to 20 mm | 21:05 |
| `stairs-exact-ik-1024-s3281` | Same sharper-foot objective with batched bounded IK and 1024 environments | 21:09 |

The sharper-foot experiment keeps the potential bounded, gamma-matched,
and terminal-zeroed. It changes no advancement, landing or success gate.
At step 760 the posture-damping leader had 25.56 mm swing error but only
3.07 mm root error, joint RMS 0.02353 rad and support quality 0.9913, which
motivated the targeted reward comparison. See [reward arithmetic](../reviews/ankle-v23-curriculum/20260915/foot-tracking-potential-arithmetic.json).

The larger run enumerates all 81 lower/free/upper assignments only for legs
whose unconstrained four-joint velocity exceeds a position/rate guard. The
quadratic objective, damping centre and physical guards match the smaller
run. At 60 static reference/request combinations, eight needed box bounds;
the largest difference from 32 coordinate sweeps was 0.000481 rad/s. Thus
this is mainly a compute allocation, not a proposed explanation for the
large foot-position errors. No standalone benchmark or simulator evaluation
was run. Its actual PPO training saved checkpoints and reached step 400.
The latest logged iteration took 105.89 seconds for 131,072 transitions;
this is a live training observation under shared load, not an isolated speed
comparison. See [bounded IK arithmetic](../reviews/ankle-v23-curriculum/20260915/bounded-ik-stationarity-arithmetic.json).

The compact first-front baseline was archived at update 24, the older
load-transfer arm at 40, and the original two-front arm at 16. Their live
process identities were checked before stopping and their source/checkpoint
evidence preserved. These are compute reallocations, not convergence claims.
All remaining training uses four 180 mm risers with 250 mm treads, unchanged
motors and a privileged teacher. The original frozen one-foot recording is
still running; it had reached 282.05 seconds with one credited front foot
and no fall. No whole-robot success has been reported by any arm.

## Earlier continuation: 20:52 PDT

**A front foot now plants on tread one in training and in the requested
frozen-policy recording. No whole-robot first-tread hold or full climb has
succeeded.** This is a hybrid reference-assisted controller with learned RL
corrections, not evidence that an unaided policy discovered the motion.

At step 800, `stairs-reference-velocity-256-s3281` reported 78 current guided
completions. Source inspection confirms that front completion requires real
FR tread credit: upward force above 1 N, every traction corner within 8 mm
of one horizontal tread, slip below 25 mm/s, and a 0.25-second hold. Its
whole-robot stage/full counts remain zero. These are current changing-policy
episodes, not a success rate or a frozen evaluation batch.

The independently launched `stairs-reference-loaded-256-s3281` also reached
75 logged new FR plants by step 536. Its first event occurred at step 304
from floor variant 0: FR force 2.724 N, sole height 182.055 mm, and maximum
corner gap 4.317 mm. The first-event checkpoint and source provenance are
preserved under `simulation/isaac/models/ankle-v23-realstairs-reference-loaded/`
`stairs-reference-loaded-256-s3281/first-foot-milestone/`.

The requested one-episode recording began at 19:39 PDT using that frozen
checkpoint, a floor reset, seed 3291, and the same runtime/asset hashes. It
repeated the FR plant at approximately 16 seconds. At this inspection the
recording is still running, so the MP4 is not yet a finalized deliverable.
See the [first plant frame](../reviews/ankle-v23-curriculum/20260915/video-reference-loaded-s3291/first-foot-plant.png)
and [recording report](../reviews/ankle-v23-curriculum/20260915/video-reference-loaded-s3291/report.json).
The video uses real-time 10 FPS playback for the 20 Hz controller.

### Active comparisons

| Job | Change | Started PDT |
| --- | --- | --- |
| `stairs-load-transfer-256-s3281` | Additional post-plant weight-transfer potential | 19:45 |
| `stairs-compact-reference-256-s3281` | Compact floor stance and 213 first-front reference poses, retaining load-transfer reward | 20:12 |
| `stairs-two-front-256-s3281` | Append 54 second-front poses, require actual first plant before advancing and simultaneous front-foot hold to finish | 20:27 |
| `stairs-posture-damping-256-s3281` | Centre IK damping on nominal reference velocity during guided motion | 20:48 |

All use 256 environments, fresh Beta(8,8) actors, seed 3281, 256 planned
updates, 320-second episodes, unchanged physical limits and four 180 mm
risers. The reference-velocity assistance is capped at 0.35 rad/s before
the current Jacobian maps it to bounded Cartesian commands. RL adds up to
25 mm/s XYZ and 0.35 rad/s angular corrections. Assistance stops after
the actual guided contact goal. No root force or pose teleport drives motion.

The loaded comparison changes only the final front servo preload, from
unloaded to a four-foot allocation with FR 3.852 N and peak motor ratio
0.61743. The load-transfer comparison adds a bounded, discounted potential
for loading planted higher feet while unloading a freely selected foot on
the lowest credited level. It changes no contact or success threshold.
See [reward analysis](rl-ankle-v23-reward-analysis.md).

Eight older arms have been stopped with checkpoints preserved: rear-only
waypoints (update 8), contact margin (72), knee branch (64), linear pitch
(120), landing alignment (56), and front/rear pitch-centered guidance without
reference velocity (24), the original reference-velocity arm (24), and loaded
landing without the later transfer reward (32). The frozen loaded-policy
recorder continues separately. These were compute reallocations after
no stage/full completion, not converged-policy or equal-budget verdicts.
Their `early_stop.json` files retain the worker identity, reason and metrics.

### Why the next foot is difficult

At the existing first-foot landing pose, independently unloading FL needs
peak torque 1.25482 times the weak motor cap. Unloading either rear foot at
that exact pose has no feasible bounded static allocation. Local pose-change
searches also failed their constraints. This does not prove impossibility;
it identifies a poor starting stance for the next transfer. See
[next-foot calculation](../reviews/ankle-v23-curriculum/20260915/after-first-plant-unloading.json).

A separate offline route retains the original compact floor support locations.
It now has 54 accepted static poses from floor contact through 185 mm full-foot
clearance, crossing and a flat first-tread plant. The new FR plant is deeper
on the tread, with sole center approximately (474, -199, 180) mm. Its four-foot
loaded allocation has peak ratio 0.66848. At this same pose, independently
unloading FL has peak ratio **0.78115**, compared with **1.25482** at the earlier
landing. Rear-foot unloading is still infeasible at this exact pose. These
calculations suggest front-left next; they do not prove a connecting motion.
See [compact landing loads](../reviews/ankle-v23-curriculum/20260915/compact-front-landing-loads.json).

Earlier coupled-solver failures near 140/150 mm were numerical contact
residuals. Separating geometry from force allocation resolved them without
widening the physical limits. At 180 mm, the remaining residual was a
1.53-micrometre undershoot of an arbitrary exact flight plane. Requiring at
least that height instead allowed 180 and 185 mm solutions; the 185 mm pose
has peak ratio 0.74148. The 190 mm local solve failed, so crossing uses the
accepted 185 mm clearance over the unchanged 180 mm riser.

The connected route still needs refinement. All 54 discrete force LPs pass,
but 159 straight quarter-segment interpolations include 77 LP failures and
20 ground-intersection samples. Maximum support drift is 1.952 mm; there
were no self or stair intersections in these samples. Intermediate poses
were refitted with exact support constraints and bounded forces. See
[route arithmetic](../reviews/ankle-v23-curriculum/20260915/compact-front-route-arithmetic.json)
and [intermediate refits](../reviews/ankle-v23-curriculum/20260915/compact-front-interior-refit.json).

All 159 intermediate refits were accepted, producing 213 discrete reference
poses. Further quarter-segment arithmetic over this denser sequence still
finds 60 of 636 LP failures and four ground intersections, with maximum depth
0.12531 mm and support drift 0.12575 mm. No self or stair intersections were
found. Thus these are useful soft targets, not a certified continuous path.
See [refined route arithmetic](../reviews/ankle-v23-curriculum/20260915/compact-front-route-refined-arithmetic.json).

At 20:12:23 PDT the new isolated compact-reference RL trial launched with
those 213 poses, a physically grounded compact floor reset, and a loaded
final FR waypoint. Every installed discrete preload passes an independent
force allocation at friction 0.85 and 0.04-degree target reserve; maximum
peak ratio is 0.85391. The static interior limitations are included in its
training contract. It uses actual constrained simulation, never interpolated
state resets or root forces. The rear reference, free floor branch, reward,
assistance, actor dimensions and full-climb rules match the load-transfer
trial. This is a reset/reference comparison, not a reward-only ablation.
The compact trial earned its first actual FR plant at step 537, with sole
height 180.563 mm, upward force 1.806 N and maximum corner gap 1.281 mm.
The event checkpoint and provenance are archived in its model family's
`first-foot-milestone` directory. At step 1480 it has 34 cumulative new-foot
events and 18 currently completed guided episodes, with zero stage/full
completions. These are changing-policy training counts, not a success rate.

At 20:27 PDT a separate `stairs-two-front-256-s3281` trial launched with
267 poses: the 213 first-front poses followed by 54 independently allocated
second-front samples. The swing leg changes from FR to FL after the first
plant. Waypoint 212 and later require actual FR tread credit before advancing.
Guide completion requires both front feet actually planted on tread one,
slip below 25 mm/s, a 0.25-second hold and root speed below 80 mm/s. The
whole-robot and full-climb criteria remain unchanged. Dedicated front-pair
events preserve up to four behavior checkpoints. Update 0 was saved at
20:30 PDT; at step 580 it had no new-foot, pair, stage or full-climb event.

The second front leg also has accepted static lift poses through 160 mm,
with FR supported on the first tread and the rear feet on the floor. At
160 mm the peak ratio is 0.38505, but the next 180 mm solve fails geometry
constraints near knee, abduction and support-ankle limits. An alternative
positive-knee branch only reached 80 mm before its 100 mm solve failed. See
[second-front lift](../reviews/ankle-v23-curriculum/20260915/compact-second-front-lift.json)
and [opposite knee branch](../reviews/ankle-v23-curriculum/20260915/compact-second-front-positive.json).

Allowing the planted FR foot to rock about its toe edge unlocked accepted
second-front clearances through 185 mm, with peak ratio 0.36962 at that
height. The planner's Euler search box was widened to +/-48 degrees while
retaining the existing upright inequality and all physical limits. This
relaxes an arbitrary search restriction, not the simulator's fall rule.
The subsequent crossing and lowering also produced an accepted static
FL plant. Both front contacts now have a bounded four-foot allocation:
FL 3.895 N, FR 3.401 N, maximum FR corner gap 3.924 mm, and peak ratio
0.36601. The base is pitched -46.20 degrees, so dynamic upright margin
is limited. This sequence is now installed in the separate two-front trial;
its dynamic execution has not yet been demonstrated.
See [rocking support calculation](../reviews/ankle-v23-curriculum/20260915/compact-second-front-rocking.json),
[second-front crossing](../reviews/ankle-v23-curriculum/20260915/compact-second-front-crossing.json),
and [two-front contact loads](../reviews/ankle-v23-curriculum/20260915/compact-two-front-loads.json).

The next issue is rear unloading: at that two-front endpoint, RL unloading
requires 2.23476 times the weak cap and RR has no bounded target allocation.
Local posture searches for either rear foot failed their constraints. A
further weight shift, changed footholds or a different contact sequence is
needed; this is not proof that climbing is impossible. See
[rear unloading attempts](../reviews/ankle-v23-curriculum/20260915/compact-rear-unloading.json).
Three additional fixed-corner posture searches per rear leg also failed
their constraints. These bounded local searches do not establish physical
impossibility. Alternative foot order and support-position endpoint searches
are being calculated separately; no failed candidate is used as guidance.
See [point-pivot attempts](../reviews/ankle-v23-curriculum/20260915/compact-rear-point-pivots.json).

### Alternate order and the IK controller

With only FR planted on tread one, alternative-order static searches found
rear-right and rear-left swing poses at independently allocated peak ratios
0.74319 and 0.45046. Their other three contacts retain their original
locations, but the selected rear foot is already off the floor; these poses
alone do not demonstrate takeoff. Keeping the rear sole flat at its original
floor XY failed. Allowing the rear-left foot to rock at that XY produced a
zero-full-foot-clearance unloading endpoint at ratio 0.65580. The preceding
body shift is not yet demonstrated. See [alternate-order endpoints](../reviews/ankle-v23-curriculum/20260915/rear-second-rear-transfer.json)
and [rocked takeoff](../reviews/ankle-v23-curriculum/20260915/rear-second-rocking-takeoff.json).
Separate free-foothold searches also found rear-unloaded poses after two front
plants, but they move support positions by up to 100 mm and are not executable
support slides. No such endpoint has been installed as a reset or reference.

The two-front trial had 25 actual first-foot plants at step 1300. Eighteen
current environments had reached waypoints 239/240 in second-front crossing,
but no second-front contact hold or whole-robot success occurred. Many other
environments stall at 191/192 with small swing-position errors but joint RMS
above 0.1 rad. Static URDF Jacobian arithmetic found that zero-centred damping
retains only 2.96% of the weakest FL direction near waypoint 191, and 10.33%
near 220. This supports a controller hypothesis, not a measured causal
diagnosis of every stalled environment.

The new isolated posture-damping trial changes the guided IK objective from
`||J v-u||^2 + 0.015^2 ||v||^2` to
`||J v-u||^2 + 0.015^2 ||v-v_nominal||^2`.
Nominal velocity remains bounded at 0.35 rad/s, with identical Cartesian
requests, target/rate guards, torque limits, references, rewards and contact
success criteria. Unguided damping remains centred at zero. Additional
read-only diagnostics expose support, root-position and orientation errors.
Worker 32944 was launched at 20:48:34 PDT, saved its first PPO checkpoint at
20:51:09 and reached step 220. All four training workers and the separate
recorder were confirmed live. See [Jacobian arithmetic](../reviews/ankle-v23-curriculum/20260915/guide-ik-damping-arithmetic.json).

Further static continuation reached 185 mm full-foot clearance for the
rear-right-second pose, but its downward continuation failed below 40 mm.
The rear-left route from the rocked takeoff reached 100 mm and failed its
120 mm solve. Neither is a complete connected floor-to-tread transfer, and
neither is installed in training. See [rear height continuation](../reviews/ankle-v23-curriculum/20260915/rear-second-height-continuation.json)
and [rear-left takeoff continuation](../reviews/ankle-v23-curriculum/20260915/rear-left-second-grounded-lift.json).

At the final inspection, compact training had 65 cumulative plants at step
2460; the two-front trial had 27 at step 1480, with no pair, stage or full
success. The two-front first-event checkpoint and provenance are archived.
The frozen recording had reached 220.05 simulated seconds without a fall,
still with only its original FR first-tread credit. Its MP4 remains open
until the one-episode recording finishes.

At 20:02 PDT all four RL workers and the recorder were confirmed live by
process command line. The new load-transfer trial had 80 cumulative new
foot/tread plant events and 79 current guided completions, with zero stage
or full-climb completions. These differing counters reflect episode resets;
they are not success rates. The added potential appears in actual post-plant
training records, but no performance advantage is established yet.

Training, the requested full-height recording and offline arithmetic are
the executed work. No standalone test suite, evaluation batch, small-stair
run or hardware check was performed. The D405 input remains a simulated
proxy with a privileged teacher; camera-only control remains future work.

## Earlier continuation: 19:00 PDT

No first-tread stage or full climb has succeeded. Two new isolated RL jobs
are now launched on the same four 180 mm risers, with 250 mm treads:

| Job | New hypothesis | Start |
| --- | --- | --- |
| `stairs-waypoints-256-s3281` | State-driven rear-transfer waypoint observations and potential | 18:46 PDT |
| `stairs-front-rear-256-s3281` | Add a grounded front-transfer reference and center the pitch action on the desired foot tilt | 19:00 PDT |

Both use 256 environments, 256 planned updates, fresh Beta(8,8) actors,
320-second episodes and gamma 0.999687548824 in PPO and shaping. They remain
privileged teachers. No joint, motor, rate, collision or success limits were
relaxed. Only floor starts qualify for full climb. The rear reference begins
with three feet on tread one and the RR collision box on the floor; the new
front reference begins with all four feet on the floor. The existing 2 mm
reset settling offset is retained. These are combined experiments, not
single-factor ablations. Neither commands root motion or an XYZ trajectory.

At 19:03 all six workers were confirmed live by their process command lines,
with about 26.5/32.6 GB GPU memory used. The new front/rear run saved
`model_0.pt` at 19:03:30, confirming its first PPO update completed. During
the first 100 control steps (five simulated seconds, before the first PPO
update), the rear-only guided run logged 43 rear-curriculum falls, while the
front/rear centered-pitch run logged none. This is an early stability sign,
not climbing evidence or an isolated causal comparison: the mixed front
reset/guidance also differs. See
[initial rollout counts](../reviews/ankle-v23-curriculum/20260915/initial-guidance-stability.json).
The new worker PIDs are 28136 (rear-only) and 30440 (front/rear).

The older milestone run was intentionally stopped at step 17,518, with
4,870 ended episodes, 922 falls, and no stage/full success. Its update-128
checkpoint, parameters, source hashes and stop reason are preserved under
`simulation/isaac/models/ankle-v23-realstairs-lift-milestones/stairs-milestones-512-s3281`.
The linear-pitch, contact-margin, knee-branch and landing-alignment runs
continue. At 18:54 their logged updates were 95, 65, 53 and 35 respectively,
also without stage/full success. The rear-guided run had reached step 362;
early ended episodes were all falls, while surviving episodes had not reached
the 320-second timeout. Do not interpret that censored fraction as its eventual
failure rate or compare it directly to the mature jobs.

### Grounded front transfer

Working backward from the previous front-foot lift found 31 accepted static
poses down to actual FR collision-box floor contact. Joined to the existing
crossing and landing samples, this gives an 82-pose candidate for moving one
front foot from floor to tread one, with the other three feet still on the
floor. At 243 quarter-segment samples there were no detected self, stair or
floor intersections, and all physical-bound static force LPs succeeded at
friction 0.85. Maximum motor ratio was 0.86413 against the 0.75021 Nm weak
motor cap; maximum support-point drift was 65.66 micrometres. This is sampled
static arithmetic, not a continuous-path proof or executed gait.

The extra target reserve is sensitive: 120 interior LPs failed at exactly
0.05 degrees reserve, 18 failed at 0.0499 degrees, and none failed at the
physical target bounds. These results remain visible in the report. The new
front waypoint preloads use 0.04 degrees reserve at each discrete pose; all
82 endpoint allocations succeeded. At the final loaded plant, FR supports
3.85 N and the peak ratio is 0.61743 with the stricter 0.05-degree reserve.
This is one-foot progress, not a whole-robot first-tread completion.
See [candidate](../reviews/ankle-v23-curriculum/20260915/front-floor-to-tread-candidate.json),
[interior arithmetic](../reviews/ankle-v23-curriculum/20260915/front-floor-to-tread-arithmetic.json)
and [ground prefix](../reviews/ankle-v23-curriculum/20260915/front-ground-prefix.json).

### Rear reference and remaining room problem

Sixteen riser-adjacent poses were refitted with 1 mm clearance for the rear
shank, moving motor and sole. Three contact-consistent poses were inserted
across the largest body shift. Four accepted poses repair only the first half
of the final lowering segment. Adding the previously feasible flat supported
first-tread endpoint produces 105 soft targets. The last lowering connection
and support-flattening motion remain unproven; this is not installed as a
forced joint trajectory. The flat four-foot endpoint has static peak ratio
0.44597, but reaching it still requires learning the intervening dynamics.

A lateral detour of the rear foot alone reached the weak motor cap before
making enough space. Changing the front-right support location gave two
accepted touchdown endpoints: 20 mm outward, and 20 mm forward plus 20 mm
outward. Their loaded rear-foot forces were 9.53 N and 6.96 N, with peak
ratios 0.7195 and 0.6521. Purely forward offsets of 20/40 mm and an outward
40 mm offset were not found feasible in this local search. Neither the front
foot relocation nor the connecting rear motion has been demonstrated, so
these alternatives are not active reset/reference changes.
See [foothold room search](../reviews/ankle-v23-curriculum/20260915/front-foothold-room.json)
and [rear-only detour](../reviews/ankle-v23-curriculum/20260915/rear-inboard-touchdown.json).

### Reward and action analysis

The new potential uses 16 times waypoint fraction plus two times pose/foot/
body similarity gated by real support. Advancement requires measured joint,
foot, body and orientation proximity with other-foot support for 0.1 seconds.
The reward uses gamma times the next potential minus the previous one, with
terminal potential zero. Existing actual-plant, stage and full bonuses remain.
Reference attainment alone never counts as a stair-climbing success.

The original absolute pitch action has a significant neutral-command issue:
zero attempts to flatten a foot. The new grounded FR reference requires
about -30.40 degrees; the grounded RR reference requires -44.86 degrees.
The front/rear experiment centers the pitch action on these reference angles,
while mapping residual -1/+1 to the same absolute -110/+110 degree endpoints.
RL still chooses XYZ commands and pitch corrections through the retained
bounded IK and servo model. The rear-only experiment keeps the original
absolute pitch map for comparison.

Local command-rate arithmetic estimates about 12.03 seconds for the earlier
rear candidate at maximum commands, before filtering or dynamic corrections.
This motivated longer episodes; it is not a global traversal-time lower
bound. [Timing calculation](../reviews/ankle-v23-curriculum/20260915/rear-action-timing-arithmetic.json).
The [sampled training comparison](../reviews/ankle-v23-curriculum/20260915/reward-comparison-latest.json)
shows front lifts reaching milestone tiers without new-tread plant credits
in the sampled histories. Thus lift height alone is inadequate evidence of
useful progress. Samples cover the first eight environments and changing
policies; they are not equal-budget or frozen-policy comparisons.

Only requested training and offline calculations were run. No separate tests,
policy evaluation, new video or hardware check was run. A floor-start complete
four-riser ascent and its frozen-policy full-height recording remain pending.

## Earlier continuation: 18:16 PDT

All five RL workers were confirmed live at 18:09, using about 26.4/32.6 GB
of GPU memory. Milestone, linear, contact-margin, knee-branch and landing
alignment had logged updates 117/512, 74/256, 41/256, 32/256 and 14/256.
There is still no first-tread completion or full climb in these runs.

The stricter rear route now reaches a **loaded, slightly tilted plant**:
97 accepted static samples connect the listed endpoint poses from rear-foot
floor contact to the 180 mm tread. The final RR sole tilt is 1.792 degrees,
with traction heights 180.000–183.175 mm. Only RR corner 3 carries load in
the independent loaded-landing calculation: 6.44 N, with peak motor ratio
0.6949 (0.5214 Nm). The unloaded touchdown pose needs ratio 0.9875 under the
stricter 0.85 friction pyramid and 0.05-degree target reserve. The three
other feet are already on the first tread throughout this calculation.

The endpoint poses **do not yet form an accepted motion**. Among 288
quarter-segment interpolations, all static force allocations remain within
the physical motor cap (maximum ratio 0.9861), but 27 samples intersect
stairs and three have self-intersections. The largest stair intersection
is 0.639 mm; the largest self-intersection is 4.029 mm between the FR and RR
feet on the final lowering segment. Maximum contact-point drift is 0.367 mm.
This identifies a concrete next planning change: route the rear foot around
or above the front foot before lowering, and add clearance and contact
projection between the crossing poses. No new reference or reset was installed.
See [fixed-contact tilted plant](../reviews/ankle-v23-curriculum/20260915/released-support-fixed-tilted-plant.json)
and [interpolation arithmetic](../reviews/ankle-v23-curriculum/20260915/released-support-fixed-tilted-arithmetic.json).

The earlier numerical stall near 26 degrees came almost entirely from a
15.3-micrometre miss of an arbitrary exact 185 mm flight-height target.
Replacing it with minimum clearance allowed the sequence to flatten while
retaining the support contacts and motor limits. Requiring a fully flat
unloaded foot at touchdown still failed; permitting a small tilt allowed
actual tread contact followed by load allocation. The contact geometry,
forces and later whole-robot hold still need dynamic training evidence.
The equivalent FL knee branch preserved the footholds and static support,
but did not resolve that exact-height numerical failure. Removing both
extra support-edge forces also failed the weakest-motor screen in all 97
sampled poses, so those force-bearing edge contacts must not simply be ignored.

Only the requested RL training and offline calculations were run. No
standalone tests, policy evaluation, new video or hardware check was run.
The full floor-start ascent and its frozen-policy recording remain outstanding.

## Earlier continuation: 18:03 PDT

The five full-height RL workers continue unchanged. At 18:00, milestone,
linear, contact-margin, knee-branch and landing-alignment had logged updates
111/512, 68/256, 38/256, 29/256 and 11/256. All report zero first-tread
completions and zero full climbs. The newest job has not yet run for one
maximum-length episode per environment, so its ended-episode fall fraction
is especially censored by surviving episodes. These are training logs, not
a fixed-policy evaluation.

The rear calculation now has an 82-pose static candidate from actual rear
foot collision-box contact with the floor to a flat 180 mm tread landing,
with the other three feet already on that tread. At its loaded landing, the
RR foot carries 8.81 N and the peak static motor ratio is 0.5661. This is a
single-leg transfer candidate, not a four-foot floor-start ascent.

The candidate is **not accepted as a connecting motion**. Among 243 sampled
interior interpolations, force allocation stays within the weakest motor
cap, but 20 samples intersect stair geometry. The largest detected
intersection is 0.636 mm. The two support edges also rotate in XY by enough
to move one contact point 10.30 mm. Neither issue is hidden by the static
endpoint acceptance. See [static route and landing](../reviews/ankle-v23-curriculum/20260915/released-support-edge-landing-warm.json)
and [interpolation arithmetic](../reviews/ankle-v23-curriculum/20260915/released-support-edge-route-arithmetic.json).

A stricter refit holds the second endpoint of each loaded support edge
within 50 micrometres per XY axis. It preserves the floor contact, lift and
crossing samples; continuation has reached 30 degrees of rear sole tilt
while flattening. At that snapshot the refit was still running. Its samples require a final
connecting-path calculation before use as a reference. No reset or training
package has been changed from these rear calculations. See
[refitted prefix](../reviews/ankle-v23-curriculum/20260915/released-support-fixed-edges-partial.json)
and [warm continuation](../reviews/ankle-v23-curriculum/20260915/released-support-fixed-edges.json).

Numerical changes are confined to the offline helper: flat-foot height uses
the sole center when flat-normal equalities are present; fixed zero forces
have no redundant friction rows; and a rigid rectangular contact edge uses
only its independent height bounds. These preserve the physical feasible
set. A separate geometry projection and independent static force LP also
finds a flat landing, recorded in
[projected landing](../reviews/ankle-v23-curriculum/20260915/released-support-projected-landing.json).
No standalone tests, policy evaluations, recording or hardware checks were run.

## Earlier continuation: 17:40 PDT

Five isolated full-height RL jobs are running. The new
`stairs-landing-align-256-s3281` started at about 17:29 with 256 environments,
256 planned PPO updates and seed 3281. It clones the linear-pitch arm and
adds a potential for aligning the airborne foot with its next tread after
the entire traction patch crosses the edge. The same floor and grounded
three-on-first resets remain; this is not the retired hover/drop experiment.
The actor and critic still receive privileged state. See the
[preparation script](../reviews/ankle-v23-curriculum/20260915/prepare_realstairs_landing_alignment.py).

At 17:37, the milestone, linear, contact-margin, knee-branch and landing
alignment jobs had logged updates 102/512, 59/256, 29/256, 20/256 and 2/256,
respectively. All still report zero first-tread completions and zero full
climbs. Training placement counters can include replanting on the same tread;
they are not ascent counts or frozen-policy reliability measurements.

### Front-foot transfer and the landing reward

A recorded linear-arm floor-start state had lifted front-right to 153.9 mm
minimum traction clearance, with the other three feet loaded. Constraining
all three supporting soles flat failed. Preserving their actual lowest
corners instead allowed a continuation to 195 mm, then across the first
riser and down to a flat 180 mm landing. The calculation retains the original
joint limits, weakest motor cap, rear payload and collision geometry.

The initial sequence's interpolated poses encountered PD target limits.
Refitting 51 poses with 0.05 degrees of target reserve and a stricter 0.85
friction pyramid resolved the sampled issue. All 150 quarter-segment interior
samples admit static contact forces under the physical 0.9 coefficient and
existing target limits; their maximum motor ratio is 0.8628. No sampled
self, stair or floor intersections were detected. Maximum support-point
drift is 0.0657 mm. A separate loaded-landing calculation assigns 3.77 N to
front-right with peak motor ratio 0.6171.

This is a partial transfer from an already-lifted recorded pose, not a
ground-start motion, continuous collision proof, dynamic rollout or climb.
It was not installed as a reset or reference. Artifacts:
[recorded geometry](../reviews/ankle-v23-curriculum/20260915/recorded-swing-geometry-1705.json),
[refitted route](../reviews/ankle-v23-curriculum/20260915/front-route-headroom.json),
[interpolation arithmetic](../reviews/ankle-v23-curriculum/20260915/front-route-headroom-arithmetic.json).

The existing direct post-edge swing score is flat while the foot is unloaded:
its landing component requires contact force. The new shaping term uses a
Gaussian of all four corner height errors, with 35 mm width, only when the
patch is wholly over the next tread. Its potential rises by 1.049 along the
computed airborne alignment and lowering sequence. A credited-level term
retains this potential when an actual landing advances the target. It uses
the same discounted potential difference and terminal zeroing, not a
per-step hover bonus. Actual planted-foot and full-climb bonuses are unchanged.
See [reward arithmetic](../reviews/ankle-v23-curriculum/20260915/front-landing-reward-arithmetic.json).

### Rear-foot contact model

Removing a duplicated fixed-corner height inequality improved the offline
solver. The older rear crossing still fails its next 1 mm advance; the
alternate FL foothold search finished with all four offsets (-5, -10, -15,
-20 mm) rejected. Single-corner support on all three feet also cannot support
the older released-foot route's 25 mm pose within the weakest motor cap.

However, the original force distribution uses loaded edges on rear-left
and front-right. An edge screen finds two arrangements within the cap.
Keeping FL corner 2, RL edge 0–2 and FR edge 1–3 then admits ten static samples
from 23.3 mm actual whole-foot clearance down to rear-foot collision-box
contact with the floor. The floor endpoint's peak ratio is 0.9803, with
zero reported collisions or floor penetration. The rear foot is unloaded
in that calculation; traction corners remain 1.86 mm above the floor
because the angled sole's collision edge touches first. This is not yet a
connected ascent or a new training reset. See
[edge screen](../reviews/ankle-v23-curriculum/20260915/released-support-edge-screen.json)
and [floor continuation](../reviews/ankle-v23-curriculum/20260915/released-support-ground-edges.json).

Only requested RL training and offline calculations were run. No standalone
test suite, additional policy evaluation, new recording or hardware check
was performed. The complete ascent and its recording remain outstanding.

## Current continuation: 16:57 PDT

The four active jobs are now the quadratic lift-milestone, linear-pitch,
contact-margin, and new knee-branch trials. All retain four 180 mm risers,
250 mm treads, the same physical motors, and the full floor-start success
criterion. None has completed a first-tread stage or a full climb in these
current trials. The newest jobs are too young for a performance conclusion.

### A different floor knee bend reduces unloading torque

The original floor reset has RR knee +39.438 degrees, unlike the negative
knee branch used in the high-clearance IK samples. Changing its hip/knee/ankle
angles to approximately +14.956/-39.438/+24.483 degrees preserves the same
body pose, flat sole orientation, and floor foothold to numerical precision.
It has no reported self, stair, or ground penetration, including at the
nominal reset's +0.2 m world translation.

At these fixed floor poses, a static force allocation with RR unloaded and
at least 2 N normal force on each remaining foot gives:

| Floor heading | Original RR bend, peak/weakest cap | Alternative bend, peak/weakest cap |
| --- | ---: | ---: |
| 0 degrees | 1.016 | 0.937 |
| 45 degrees | 1.041 | 0.953 |

Both alternatives are within the 0.7502 Nm weakest modeled torque cap and
bounded PD setpoint headroom. Standing on all four feet needs ratios 0.285
and 0.324 for the original and alternative poses respectively. This is a
fixed-pose, quasi-static comparison with a conservative friction pyramid;
it does not prove the original bend could not unload after changing posture,
or that the alternative can perform a complete swing. See
[knee-branch force arithmetic](../reviews/ankle-v23-curriculum/20260915/floor-knee-branches.json)
and [translated reset geometry](../reviews/ankle-v23-curriculum/20260915/floor-knee-reset-world-geometry.json).

`stairs-knee-branch-256-s3281` started at 16:55 PDT with 256 environments,
256 planned updates, seed 3281, and a fresh Beta(8,8) actor. It clones the
linear-pitch arm's controller, observations and rewards. Only the two floor
reset joint poses and the static PD preloads needed to hold them change; the
footholds, root poses, reset probabilities, and three-on-first reset remain
the same. This supplies an alternative initial posture, not a demonstrated
transition between knee branches. Worker PID 29700 was confirmed live and
had reached control step 52 at 16:57. See the
[preparation script](../reviews/ankle-v23-curriculum/20260915/prepare_realstairs_knee_branch.py).
It subsequently reached step 307. The first recorded floor-start states
contain negative RR knee angles, confirming that training is using the
intended alternative branch; that is not evidence of climbing performance.

To free a training slot, `stairs-bounded-1024-s3261` was intentionally stopped
at 16:55 after logged update 75/512. It had 5,223 ended episodes at the
pre-stop snapshot, with no stage/full-climb completion. Its update-72
checkpoint, parameters, hashes, terminal status and stop reason are preserved
under `simulation/isaac/models/ankle-v23-realstairs-bounded-ik/stairs-bounded-1024-s3261/`.
This is an incomplete budget, not evidence that the baseline could never
learn. The other three workers continue unchanged. At 16:57 the milestone,
linear and contact-margin jobs had logged updates 78/512, 35/256 and 5/256.

### Rocking supports remain a candidate, not a solved transfer

A fixed-pose screen considered adding a second supporting corner pivot.
Most choices fail force allocation; the best FR corner requires ratio 1.044
before changing body posture. Joint pose/force optimization then finds a
two-pivot pose at ratio 1.000 with exact support contacts, but moving the
rear leading corner another 4 mm clips the rear sole against the first
riser. The FL second-riser collision is absent in that failed candidate.

The helper can now allocate force to additional corners only when they are
constrained to tread height. Using the grounded FL and FR edges produces an
accepted initial sample, but the following 2 mm advance still fails the full
constraint set. These are static contact models; no connecting motion,
load transfer, or no-slip edge rotation has been demonstrated. No reference
or reset was installed from them. See
[two-pivot force screen](../reviews/ankle-v23-curriculum/20260915/two-pivot-force-screen.json),
[two-pivot samples](../reviews/ankle-v23-curriculum/20260915/two-pivot-crossing.json),
and [edge-contact samples](../reviews/ankle-v23-curriculum/20260915/edge-support-crossing.json).
The alternate-FL-foothold search is still running; its -5, -10 and -15 mm choices
failed the numerical constraints. A local optimizer failure is not a proof
of mechanical impossibility.

The older released-support arrangement already admits a static upper-tread
landing. A new attempt to extend its last 25 mm clearance pose to angled
whole-foot floor contact still fails: its lowest collision point remains
11.71 mm above the floor. This is rejected, and is not used as a hover reset.
See [released-support floor attempt](../reviews/ankle-v23-curriculum/20260915/released-support-ground-contact.json).

Only requested RL training and offline geometry/force/reward arithmetic
were executed. There was no standalone test, policy evaluation, new video,
or hardware check. The complete ascent and subsequent recording remain
unproven; the goal is still active.

## Earlier continuation: 16:40 PDT

Four isolated RL jobs are training. The new `stairs-contact-margin-256-s3281`
trial started at 16:39 PDT with 256 environments and 256 planned PPO updates
(8,388,608 transitions). It uses the same seed, budget, environment count,
linear pitch controller, rewards other than the support-margin feature,
resets, and physical limits as `stairs-linear-256-s3281`. Both start from a
fresh Beta(8,8) actor. Their support-margin observation has the same dimensions
but a different meaning. A single seed remains an exploratory comparison.

At 16:40 the bounded and quadratic milestone jobs had logged update 66/512,
and the linear-pitch job 26/256. Their ended episode counts were 4,503, 2,426,
and 448 respectively, with zero first-tread completions or full climbs. The
new contact-margin job had begun collecting training transitions. These are
ongoing training counts, not frozen-policy success estimates.
Its worker PID 20212 was confirmed by command line and had advanced to control
step 213 at 16:43 PDT. GPU use with all four jobs was about 28.4/32.6 GB.

### A reward feature was underestimating the support area

The old stability margin used the triangle joining the supporting foot
centers. Across 85 accepted discrete static IK samples, this reported the COM
projection **18.6–27.4 mm outside** that triangle. The actual contact hull in
the static force model had **20.7–29.5 mm of positive margin**. These poses
satisfy the modeled weakest motor torque cap. The 100 × 60 mm feet cannot be
treated as point contacts for this particular shaping feature.

The new trial forms the XY convex hull of traction corners within 3 mm of
their horizontal terrain surface, on supporting feet with normal force above
1 N. It excludes the candidate swing foot and raised corners. An insufficient
or collinear contact set receives negative margin. This changes the existing
readiness term and its observation, retaining the force/support gates and
strict landing/full-climb criteria. It remains a geometric heuristic: mixed
contact heights, friction, momentum, torque, and force distributions still
matter. It is not a dynamic stability guarantee.

For these 85 static samples, the near-surface estimate agrees with the margin
from the explicitly modeled contacts: one fixed FL corner and two flat feet.
That agreement is limited to these samples. See
[support-footprint arithmetic](../reviews/ankle-v23-curriculum/20260915/contact-footprint-arithmetic.json)
and the [isolated preparation script](../reviews/ankle-v23-curriculum/20260915/prepare_realstairs_contact_margin.py).
The three earlier training packages remain unchanged.

### Ground contact and the next collision bottleneck

The rear foot can reach an angled floor-contact endpoint using its entire
URDF collision geometry, with no reported penetration and a weakest-cap
static torque ratio of 0.885. Its sole-normal tilt is 37.1 degrees. The lowest
actual foot collision point touches the floor; the smaller traction patch is
1.53 mm above it. The RR foot is unloaded in this calculation. It does not
establish loaded takeoff or a connection from the initial floor stance. See
[floor-contact endpoint](../reviews/ankle-v23-curriculum/20260915/corner-ground-contact.json).

With the same support arrangement and FL corner pivot, additional discrete
samples raise the rear traction patch to 185 mm before the first edge,
using about 99.4 degrees of sole-normal tilt and torque ratio 0.869. During
the subsequent forward search, the leading corner reaches x=341.24 mm.
Trying x=345.24 mm fails: the FL supporting leg's proximal arm and knee
collide with the **second** riser. This is not merely rear-toe clearance.
A 1 mm retry with signed collision constraints also fails to satisfy the
full pose/force constraints. An alternate-FL-placement search is underway;
its candidates are different footholds, not motion with a sliding support.
See [crossing samples](../reviews/ankle-v23-curriculum/20260915/corner-riser-crossing.json)
and [refined attempt](../reviews/ankle-v23-curriculum/20260915/corner-riser-crossing-refined.json).
The offline helper now checks the ground plane as well as self/stair
collisions. No new static pose or path has been installed as a reset.

### Clearance shaping can oppose the forward phase

Counterfactual reward arithmetic on 86 discrete candidate samples shows an
approximately **0.852 potential drawdown** during forward swing after folding.
The rear leading edge advances from x=253.24 to 337.24 mm while its lowest
traction corner lowers from 184.07 to 155.38 mm. Its six lift bonuses have
already paid, so they supply no new reward through this part. The reduction
comes mostly from clearance/approach shaping, with some support-quality
change. This is a possible credit-assignment barrier, not evidence of a
validated motion or a measured PPO failure mechanism. It does not include
timing, dynamic costs, or the eventual planting reward. The contact-margin
trial leaves this term unchanged so the stability-feature comparison remains
isolated. A subsequent crossing reward needs to preserve actual edge
clearance and landing requirements. See
[reward arithmetic](../reviews/ankle-v23-curriculum/20260915/corner-route-reward-arithmetic.json).

Reading existing training traces at 16:35 found a sampled bounded-run front
foot lift of 159.7 mm from the floor, with 26.1 degrees of sole-normal tilt.
This uses four logged environments per run and is not a landing or a success
rate. The linear-pitch run is younger and has not shown better qualified
clearance in those samples. See
[recorded geometry snapshot](../reviews/ankle-v23-curriculum/20260915/recorded-swing-geometry-1635.json).

No standalone test, policy evaluation, new recording, or hardware check was
run in this continuation. The executed work is authorized RL training and
offline IK/reward analysis. Camera-only D405 control remains unestablished.

## Completed reward comparison

Both earlier jobs finished their full 256 updates:

| Run | Ended episodes | Falls | First-tread completions | Full climbs |
| --- | ---: | ---: | ---: | ---: |
| `stairs-progress-r0-s3231` | 2,647 | 1,985 | 0 | 0 |
| `stairs-support-r2-s3231` | 2,463 | 1,760 | 0 | 0 |

Their final 512 episodes had low average support quality, approximately
0.023 and 0.031. The logged per-foot histories for environments 0–7 contained
no newly credited landings. These settings are not being repeated unchanged.
The checkpoints and completed contracts are archived in the corresponding
model families. Training histories are not frozen-policy reliability estimates.

## What IK changed in the diagnosis

The earlier reward selected the lowest foot and kept that selection until it
landed. This imposes a stair-foot sequence and encourages every foot to gather
on one tread before any advances further. That sequence is not established
as mechanically suitable for this robot.

[A stair gait study](https://linqi-ye.github.io/docs/2021Optimized%20Static%20Gait%20for%20Quadruped%20Robots%20Walking%20on%20Stairs.pdf)
separates pose selection from motion sequencing and emphasizes coordinating
body height/pitch with leg motion. It motivates examining the connecting
motion, not treating a reachable landing endpoint as a solved step. Its
robot has substantially different geometry and joint ranges, so its results
are not evidence that this V2.3 robot can climb the target stairs.

New calculations examined three changes:

1. Permit the unloaded swing sole to tilt. Only the supporting patches need
   to remain flat. Individual poses at 30, 90, and 150 mm lowest-corner
   clearance were found within the weakest torque cap. They use different
   support footholds, so they do not establish a connecting motion.
2. Fit several swing poses together with three shared support footholds.
   Subsequent refinement found poses at 30, 90, 150, and 185 mm clearance
   satisfying the numerical support-plane and collision constraints. Their
   bounded-PD static torque ratios are approximately 0.904, 0.945, 0.964,
   and 0.682 against the weakest modeled cap. Some optimizations reached an
   iteration limit; acceptance of these individual arithmetic results rests
   on their constraint residuals, not the optimizer's success flag. The floor
   takeoff and flat landing poses failed. Interpolation is not established.
3. Move the front-right foothold forward within tread one to leave room for
   the rear foot. The search advances that foothold by 40 mm in x and
   -40 mm in y, along its sagittal direction. This removes the geometric
   landing obstruction, but the initial fit exceeded torque/headroom limits
   in several phases. A joint pose-and-force optimization subsequently found
   exact individual samples at 30, 90, 150 and 185 mm clearance, and a flat
   landing, with weakest-strength torque ratios approximately 0.909, 0.798,
   0.988, 0.682 and 0.838. The 150 mm solve reached an iteration limit but its
   numerical constraints pass. The flat floor takeoff still failed.
   These solutions also use different knee branches and, in some cases,
   swing-foot tilts far outside the controller's 30-degree pitch request.
   They are not accepted as a connected path or an executable reference.

The mixed-level search also considered a front foot on tread two while a rear
foot remained on the floor. Some approximate endpoints exist, but no accepted
connecting motion was found. Rotating a known stance by 180 degrees and
swapping the legs reproduced its geometry, but did not materially reduce its
unloading torque demand. These are investigated hypotheses, not solved gaits.

Arithmetic artifacts:

- [Contact arrangement search](../reviews/ankle-v23-curriculum/20260915/contact-search-spread.json)
- [Tilted swing search](../reviews/ankle-v23-curriculum/20260915/contact-search-tilted_swing.json)
- [Shared-support samples](../reviews/ankle-v23-curriculum/20260915/rear-common-support-search.json)
- [Refined support constraints](../reviews/ankle-v23-curriculum/20260915/rear-common-support-refined.json)
- [Forward-foothold search](../reviews/ankle-v23-curriculum/20260915/rear-foothold-room-search.json)
- [Simultaneous pose and force optimization](../reviews/ankle-v23-curriculum/20260915/rear-pose-forces.json)
- [Swing tilt constraints](../reviews/ankle-v23-curriculum/20260915/rear-tilt-constraints.json)

A 30-degree world-normal tilt constraint retained a feasible 90 mm clearance
sample with torque ratio 0.818. The initial higher-clearance searches failed;
these are local optimization failures, not proofs of infeasibility. Following
the feasible negative-knee branch in 15 mm increments found further static
samples at 105, 120 and 135 mm, with torque ratios 0.810, 0.803 and 0.880.
The 150 mm sample failed its geometry constraints. This improves the search
over simply jumping between unrelated knee configurations, but still does
not establish a crossing. See the [branch calculation](../reviews/ankle-v23-curriculum/20260915/rear-negative-branch.json).
These are intermediate foot heights above the same 180 mm stair, not smaller
training stairs. The next search examines staggered support with one front
foot on tread two before lifting the last rear foot from the floor.
The first fixed-foothold and free-foothold staggered solves did not satisfy
their constraints. Four further restarts bent the nearly straight rear knees
and reduced body roll; all also failed their constraints. None of these
staggered candidates has been accepted as a training reset or reference.

The original pre-edge search required the whole sole to clear the riser before
crossing. That is conservative: a tilted foot could potentially put its
leading edge over the tread while its trailing edge remains behind the riser.
The toe-crossing calculations below use actual collision clearance for that
transition. Final planting must still satisfy the full-patch criterion.

None of the rejected or incomplete paths has been installed as a motion
reference in training.

## Active RL approach

The initial isolated `ankle_v23_realstairs_cartesian` package changed the action space
from 16 joint velocities to 12 foot-velocity components and four desired sole
pitch offsets. A damped differential IK calculation maps these requests into
servo position-target increments using the existing robot's joint axes and
current link transforms. It applies no forces to the body and does not set
the simulated body pose after reset.

The policy runs at 20 Hz, with physics still at 240 Hz. Foot velocity requests
are bounded to 70 mm/s per axis, sole tilt initially to 30 degrees, and the original
motor torque/speed, joint-position, randomized strength, and target-rate limits
remain. The existing two-degree target increment guard is retained; at 20 Hz
that is a stricter command-rate bound than at 60 Hz. This is a combined action,
reward, and decision-frequency experiment, not a single-variable ablation.

The baseline reward permits any foot order. It uses each foot's next tread, graded
unloading/support/clearance, movement over the riser, and contact-based landing
credit. A foot/tread bonus can be earned once, after 0.25 seconds of qualified
contact. The all-four-first-tread milestone does not end the episode. Only a
floor-start episode with all four patches on tread four for one second can
be a full climb; the existing force, slip, speed, fall, and range checks remain.

This version allows a front foot to advance further while rear feet catch up.
It does not yet explicitly reward the newly identified same-tread preparatory
move; that depends on the foothold-room calculation producing a useful route.

The reset distribution remains 25% straight floor, 25% oblique floor, and
50% three-on-first-tread with the remaining foot on the floor. No tiny stairs
or hovering last-foot resets are used. These are privileged teachers with a
D405 depth proxy, not camera-only policies.

| Job | Environments | Planned updates | State |
| --- | ---: | ---: | --- |
| `stairs-ik-s3251` | 128 | 256 | Stopped at 15:26 PDT; checkpoint 72 preserved |
| `stairs-ik-1024-s3261` | 1,024 | 512 | Intentionally stopped after logged update 27 |
| `stairs-ready-s3251` | 128 | 256 | Stopped at 15:26 PDT; checkpoint 32 preserved |
| `stairs-tilt-1024-s3261` | 1,024 | 512 | Stopped after update 9 for the IK correction below |
| `stairs-bounded-1024-s3261` | 1,024 | 512 | Active constrained-IK successor |
| `stairs-milestones-512-s3281` | 512 | 512 | Active supported-lift reward experiment |
| `stairs-linear-256-s3281` | 256 | 256 | Active linear sole-pitch exploration experiment |

These jobs use 128-step collection chunks (6.4 seconds), gamma 0.999100269973
(matching the prior discount per physical second), lambda 0.995, and fresh
Beta(8,8) actors. The larger job uses additional available GPU capacity;
hardware and terrain difficulty are unchanged.

At the later inspection the baseline small/large jobs had reached updates
30/15, with no first-tread or full-climb success. The new readiness worker was
live and recording trajectories. Reading the ended-episode credit histories
(330 small, 1,125 large at that snapshot) found no credited stair-height gain.
The generic placement counter can include re-landings on an already occupied
tread and must not be interpreted as ascent. A sharp change in recent fall
rate when long episodes time out is also not proof of policy improvement.

## Reward cancellation found and corrected in a separate arm

The old per-foot unloading feature is `clip(1 - Fz/16, 0, 1)`. Summing it
across four feet gives a constant while every force remains in that linear
range and total weight stays fixed. Above 3 N on each foot, all support gates
remain one, so shifting weight away from one foot earns no unloading progress.
Below 3 N, the unloaded foot reduces the support gates of the other candidate
feet, making the old total preparation potential decrease.

The [force-gated arithmetic](../reviews/ankle-v23-curriculum/20260915/reward-load-shift-with-support.json)
isolates this effect with zero sole gaps and center-triangle margins held at
zero. It is a counterfactual calculation of reward terms, not a dynamically
realized trajectory. With total weight fixed at 33.786 N:

| Rear foot load | Old preparation potential | New candidate readiness |
| --- | ---: | ---: |
| 8.45 N, equal sharing | 2.888 | 1.194 |
| 4 N | 2.888 | 1.750 |
| 2 N | 2.334 | 2.000 |
| 0 N | 1.339 | 2.250 |

The isolated `ankle_v23_realstairs_readiness` arm replaces summed preparation
with the maximum readiness of a freely chosen candidate foot, gated by the
other three actual supports. It retains summed clearance/forward progress.
It also removes the previous downward jump at the over-edge latch: the
pre-crossing clearance/approach score tends to 6, the crossed score starts at
6, and actual landing adds up to 2. Landing bonuses, terminal conditions and
discounted potential differences remain.

This was a reward-only comparison against the 128-environment Cartesian run:
same seed 3251, control, observations, PPO settings, reset distribution,
terrain, and motor constraints. The active baseline package was not edited.
Source/asset copies and hashes are recorded per job. Neither arm demonstrated
a first-tread completion before its deliberate retirement for the controller
correction. These unfinished runs do not establish a final performance comparison.

## Wider swing rotation and the toe-crossing calculation

A toe-first search relaxes the unnecessary requirement that every corner be
above 180 mm while the foot is still entirely behind the riser. It constrains
the leading corner to 185 mm, checks the full collision boxes, and retains
the same three support footholds and bounded-PD torque limits.

The initial 30-degree world-normal tilt search failed. With a 110-degree
limit, it found two accepted static samples:

| Leading corner x | Riser edge x | Weakest-strength torque ratio | Required sagittal sole pitch |
| --- | --- | ---: | ---: |
| 334 mm | 340 mm | 0.710 | -90.21 degrees |
| 349 mm | 340 mm | 0.695 | -86.84 degrees |

The foot is almost vertical. Its ankle remains within the existing
35-degree joint stop; the hip and knee account for the rest of the sole's
orientation. The next sample, with leading corner x = 369 mm, failed its
constraints and collided with the next stair. These are not a complete
crossing, a connected floor takeoff, or a verified landing.

The [toe poses](../reviews/ankle-v23-curriculum/20260915/rear-toe-clearance-110.json)
and [controller-request calculation](../reviews/ankle-v23-curriculum/20260915/folded-foot-controller-requests.json)
show that the old 30-degree request range could not hold these configurations.
The isolated `ankle_v23_realstairs_swing_tilt` package uses
`110 degrees * action * abs(action)` for sole-pitch requests. The quadratic
mapping keeps small commands gentle while permitting large folds. It retains
the readiness reward, 20 Hz control, angular-rate clamp, all joint/motor
limits, and the original full-height resets. No calculated pose is installed
as a reference or a hovering reset.

The large baseline was stopped deliberately at 14:58 PDT to reallocate its
compute. Its `model_24.pt` is preserved under
`simulation/isaac/models/ankle-v23-realstairs-cartesian-superseded/stairs-ik-1024-s3261/`
with an incomplete-training contract. `early_stop.json` records the reason;
the supervisor correctly records an interrupted run. This is not evidence
that the unfinished baseline could never learn. The smaller baseline remained
active until the later retirement described below.

The replacement `stairs-tilt-1024-s3261` started with a fresh actor. It was
later superseded by the bounded-IK correction below. Old worker PID 5060 was
subsequently reused by a Python driver, so process command lines must be
checked when identifying live jobs.

The offline kinematic model was also compared algebraically with nine saved
training failure states. Reconstructed sole positions differed from the logs
by at most 0.012 mm, and COM positions by at most 0.005 mm. This supports frame
and joint-order consistency; it does not validate the physical model or a
trajectory. See [recorded-state calculations](../reviews/ankle-v23-curriculum/20260915/recorded-state-fk-comparison.json).

An endpoint search on the positive-knee branch failed both with fixed and
free support footholds. Four true-sideways takeoff starts, with yaw allowed
between -100 and -80 degrees, converged to unacceptable sole-normal errors
of about 0.228. These failures do not prove that sideways or staggered motion
is impossible; they do rule out using these samples as valid references.
See [positive-knee endpoints](../reviews/ankle-v23-curriculum/20260915/rear-positive-landing.json)
and [sideways takeoff](../reviews/ankle-v23-curriculum/20260915/sideways-takeoff.json).

## Joint limits must be inside the Cartesian IK solve

The original controller solved unconstrained differential IK and then clipped
each joint independently. That changes the Cartesian motion direction when
one joint needs to move faster than the others. Saved-reset calculations for
a pure 70 mm/s upward request found:

| State and leg | Independently clipped x velocity | Constrained x velocity | Constrained z velocity |
| --- | ---: | ---: | ---: |
| Straight floor, front-left | +91.19 mm/s | +2.54 mm/s | +38.38 mm/s |
| Straight floor, rear-left | +86.92 mm/s | -2.61 mm/s | +38.19 mm/s |
| Three-on-first, front-left | +77.16 mm/s | -0.56 mm/s | +27.01 mm/s |

These are fixed-root, linearized kinematic calculations, not measured foot
velocities under contact. They expose a command-mapping problem that rewards
alone should not have to compensate for. The near-straight rear knee in the
three-on-first reset also has low instantaneous upward authority; its
calculated knee command already reaches the rate bound, so no extra automatic
knee-bending bias has been added.

The isolated `ankle_v23_realstairs_bounded_ik` package solves the same damped
least-squares objective with target-position headroom and target-rate bounds
included. It uses 32 coordinate-minimization sweeps on each four-joint leg,
starting from the clipped unconstrained solution. For the 12 saved-reset
calculations, 32 sweeps matched the independent bounded least-squares
objective to numerical precision. This arithmetic comparison does not
establish convergence for every possible pose or physical tracking accuracy.

The rate, torque, position, and two-degree command guards are retained. So
are the readiness reward and wider sole-pitch requests. See
[reset Jacobians](../reviews/ankle-v23-curriculum/20260915/reset-ik-authority.json)
and [bounded-IK arithmetic](../reviews/ankle-v23-curriculum/20260915/bounded-ik-arithmetic.json).

The short wider-tilt run was stopped deliberately at 15:15 PDT to apply this
controller correction. Its `model_8.pt` and incomplete contract are archived
under `simulation/isaac/models/ankle-v23-realstairs-swing-tilt-superseded/stairs-tilt-1024-s3261/`.
This was not a performance conclusion from nine updates.

`stairs-bounded-1024-s3261` starts fresh and plans 512 updates, or 67,108,864
transitions. Worker PID 7236 was confirmed by its command line at 15:32 PDT.
The first six seconds of saved motion for four environments were similar to
the earlier wider-tilt run under small initial random commands. This is before
the first PPO update and does not establish a learning improvement. See the
[initial motion comparison](../reviews/ankle-v23-curriculum/20260915/initial-ik-rollout-comparison.json).

The two smaller, independently clipped IK runs were stopped deliberately at
15:26 PDT to reallocate compute. The Cartesian baseline had 713 ended episodes
and 339 falls; readiness had 372 and 208. Both had zero first-tread completions
and full climbs. Checkpoints 72 and 32, respectively, are preserved under
`simulation/isaac/models/ankle-v23-realstairs-cartesian-baseline/stairs-ik-s3251/`
and `simulation/isaac/models/ankle-v23-realstairs-readiness-baseline/stairs-ready-s3251/`.
Their incomplete contracts and `early_stop.json` files identify the reason;
these are not completed-budget performance conclusions.

## Once-only supported lift milestones

The isolated `ankle_v23_realstairs_lift_milestones` package adds an exploration
reward to the corrected IK controller. It retains the readiness potential,
wide sole-pitch action, physical limits, resets, and four full-height stairs.
For each foot and next tread, the lowest traction corner can earn six 30 mm
height milestones, each worth 0.25. Eligibility requires the other three
support qualities to be at least 0.65, the lifted foot's normal load below
1 N, proximity to its next riser, and valid range/fall conditions. The policy
observes the current per-foot milestone history.

The maximum lift bonus is 1.5 per foot per tread, versus 8 for a qualified
actual landing. Each tier pays once per foot/tread in an episode: hovering,
lowering and re-lifting, or revisiting the same tread cannot repay it. This
changes the learning objective deliberately to help exploration before the
first landing. It does not change completion criteria or introduce smaller
stairs. Intermediate foot heights remain over the same 180 mm riser.

`stairs-milestones-512-s3281` uses 512 environments and plans 512 updates,
or 33,554,432 transitions. Its worker PID 3448 was confirmed by command line
at 15:32 PDT. The corrected 1,024-environment run continues unchanged beside
it. Seeds and environment counts differ, so this is an exploratory comparison,
not an isolated statistical estimate of the reward effect.

At 15:34 PDT, 127 of its first 211 ended episodes earned at least one lift
milestone. All were floor starts. There were 210 total earned tiers, at most
four across all feet in an episode, and zero stage/full completions. All
these early ended episodes were falls; surviving episodes had not yet reached
their time limit, so this is not a final fall rate. In the sampled first eight
environments, the highest qualified corner-clearance tier was 30 mm. A tier
is neither a landing nor evidence of sustained support. The
[log analysis](../reviews/ankle-v23-curriculum/20260915/lift-milestone-log-analysis.json)
can be refreshed from existing logs without running a policy.

## Support-foot repositioning opens more toe clearance

Moving the front-left support 10 mm backward and 10 mm outward on its existing
tread allowed a static sample with the rear toe at x = 369 mm, past the
x = 340 mm riser. Its bounded-PD peak torque was 0.650 Nm, or 0.866 of the
weakest modeled cap, with the modeled collision constraints satisfied. See
[support-foot room](../reviews/ankle-v23-curriculum/20260915/folded-foot-room.json).
This provides a concrete reason to allow same-tread repositioning; it does
not establish a connected floor takeoff, transfer, or landing.

A direct next sample at x = 379 mm failed and moved far in joint space.
The subsequent calculation warm-starts both the accepted pose and forces,
limits changes in body/joint coordinates, and advances the toe in 2 mm
increments. Discrete feasible poses still do not prove an executable path.
This found accepted samples at x = 371, 373, and 375 mm with weakest-strength
torque ratios 0.881, 0.898, and 0.919. Maximum joint changes between adjacent
samples were 3.7–4.0 degrees. The x = 377 mm solve failed its equality
constraints and was rejected despite having no reported collision. See the
[local continuation](../reviews/ankle-v23-curriculum/20260915/folded-foot-local-route.json).
At the accepted 375 mm sample, the rear knee and both right ankles reach
the search's 99% joint bounds. The rejected 377 mm sample also reaches the
front-right hip-flexion bound. This identifies a local geometric bottleneck
for these fixed support footholds, not a proof that no climbing gait exists.
No calculated samples have been installed as training resets or references.

## A revised support arrangement reaches a static landing

Allowing the front-right support to move 18.6 mm backward and 10.3 mm inward
from the preceding arrangement released the local crossing bottleneck. The
new fixed support centers are approximately FL (0.52184, 0.12510, 0.18),
RL (0.39656, 0.25218, 0.18), and FR (0.46810, -0.37546, 0.18) metres.
This is a choice of different planted locations, not permission to slide
the supports while executing a motion.

With those locations fixed, warm-started static samples advance the rear
toe from x = 399 to 431 mm, put every traction corner beyond the riser, and
flatten the foot for a landing at z = 180 mm. The final three-support
equilibrium has peak torque ratio 0.746 against the weakest cap. A separate
four-foot equilibrium at the same landing, with at least 2 N on each foot,
has ratio 0.422. The near-flat intermediate solve reached an iteration limit
but met the numerical residual thresholds; the final landing solve converged.
See [foothold selection](../reviews/ankle-v23-curriculum/20260915/front-support-release.json)
and [crossing/landing samples](../reviews/ankle-v23-curriculum/20260915/support-release-route.json).

Tracing backward reaches a lowest-corner height of 25 mm, with torque ratio
0.961. The next requested height of 10 mm fails the geometric constraints
and intersects the first riser with the rear sole. There is still no
grounded takeoff with these support footholds. A further search chose nearby
support footholds for the missing floor endpoint. The rear-left-only
relocation failed. Allowing two or three supports to relocate found exact,
collision-free endpoint geometries, but with weak-cap torque ratios 1.057
and 1.076. Those candidates were rejected.
See [takeoff samples](../reviews/ankle-v23-curriculum/20260915/support-release-takeoff.json)
and [grounded arrangement search](../reviews/ankle-v23-curriculum/20260915/grounded-support-arrangement.json).

That objective traded lower effort against small foothold displacement, so
the rejected results did not establish a mechanical lower bound. A corrected
search made the weak motor cap a hard constraint and reduced the displacement
bias. It found a flat, grounded rear-foot unloading endpoint with ratio
0.99876 and no modeled collisions. The support moves from the prior
arrangement are FL (+10.6, -29.6), RL (-5.2, +39.0), and FR (-40.0, -14.7) mm
in world x/y. The roughly 0.1% torque margin is very small. This endpoint
has a different support arrangement from the crossing/landing sequence and
cannot be spliced into it without further work. A new full-height static
lift search starts from this actual floor endpoint with the hard cap retained.
See [hard-cap endpoint](../reviews/ankle-v23-curriculum/20260915/grounded-support-hard-cap.json)
and [grounded lift search](../reviews/ankle-v23-curriculum/20260915/grounded-rear-route.json).
The first requested 15 mm lift failed: the solve reached its local 10-degree
joint-change bound and did not satisfy the target height. The finer search
starts in 2.5 mm increments while retaining the full-riser destination and
hard motor cap. This is IK sampling resolution, not smaller training stairs.
The first finer solve also failed to converge. Replacing the exact minimum
corner-height equality with separate lower bounds on every corner, and
penalizing large changes from the previous pose, found accepted samples from
2.5 through 15 mm. The next 30 mm request failed. An additional guard for all
eight corners of the full sole box, with finer increments, extended the
accepted samples to 17.5 mm. The 20 mm request failed with its lowest corner
0.44 mm short of the target. The rear abduction joint was approaching its
99% search bound. These local optimizer failures are not impossibility proofs.
See [smooth clearance](../reviews/ankle-v23-curriculum/20260915/grounded-rear-route-smooth.json)
and [whole-sole guard](../reviews/ankle-v23-curriculum/20260915/grounded-rear-route-guarded.json).

A separate static search now allows one front support to pivot about a fixed
traction corner. Only that corner carries force on the pivoting foot; its
other corners must stay above the tread and within 10 mm of it. The other
two supporting feet stay flat, and the weak torque cap and friction pyramid
remain. This examines a contact mode excluded by the earlier all-flat support
assumption. It does not change simulator physics. The candidates and actual
contact-force residuals are saved in
[corner-support takeoff](../reviews/ankle-v23-curriculum/20260915/corner-support-takeoff.json).
All six initial corner candidates failed their residual thresholds. The
closest front-left candidate missed its fixed pivot by 0.257 mm and reached
the iteration limit. A longer refinement is examining that candidate. A
second allowed contact envelope, if needed, uses at most 12 mm corner gap
and at least 2.5 N normal force; its lower-bound support quality is 0.662,
still above the reward's 0.65 support gate. The physical motor cap remains
hard. No failed candidate is accepted as a training reference.
The longer refinement converged for the original 10 mm / 3 N envelope,
without needing the second envelope. It found a 30 mm rear-foot clearance
sample with torque ratio 0.99907, equality residual below 2e-10, and no
modeled collision. The front-left support uses its fixed traction corner 2;
the other two supports remain flat. Its very small torque margin and the
unproven transition to that corner-contact distribution remain limitations.
Separate static continuations now trace this sample upward and back toward
the floor. See [refined corner contact](../reviews/ankle-v23-curriculum/20260915/corner-support-takeoff-refined.json).
The upward continuation accepted all 42 samples through its requested 135 mm
lower bound; the last actual lowest-corner height is 146.0 mm, with torque
ratio 0.803. The downward continuation accepted 11 samples to 2.5 mm, with
ratio 0.886, then failed when flat sole orientation was imposed at the floor.
Thus the rocking-support arrangement extends the candidate clearance route,
but a grounded takeoff and riser crossing are still unproven. The small
intermediate heights are static sampling points on the 180 mm task, not
training stairs or installed hovering resets. See
[upward samples](../reviews/ankle-v23-curriculum/20260915/corner-support-up.json)
and [downward samples](../reviews/ankle-v23-curriculum/20260915/corner-support-down.json).

The combined 53 accepted endpoint samples are not an executable reference.
Arithmetic at 156 interior points of joint/body interpolation found 28
samples with terrain intersections, including rear sole/shank penetration
up to 3.11 mm. Support centers drift by up to 0.312 mm; interpolated PD
setpoints exceed their position guard by up to 0.0000685 rad. This explicitly
rejects simple interpolation despite the feasible endpoint samples. No
part of this path is installed in training. See the
[interpolation residuals](../reviews/ankle-v23-curriculum/20260915/support-release-interpolation.json).

The useful design implication is that supporting-foot placement and the
whole swing path matter together. A reward for endpoint height alone does
not identify a collision-free route. Adding a global penalty for nearing
joint limits would also be questionable: several feasible clearance poses
need those ranges. Any posture guidance should preserve the necessary swing
range while making room through support-foot placement.

## Initial exploration under-covers large sole rotations

The freshly zeroed actor head is Beta(8,8). For independent initial commands,
the 20 Hz filter reduces action standard deviation from 0.243 to 0.101.
The quadratic 110-degree pitch mapping therefore has a stationary mean
absolute request of only 1.12 degrees. Even before filtering, an action
large enough to request an 86.84-degree fold has two-sided probability about
8.4e-7. Filtering makes sustained large folds harder to sample. These are
initial-distribution calculations, not measurements of a learned policy.

The integrated free vertical-command standard deviation over 96 seconds is
37.2 mm, before IK/contact dynamics. It is not predicted actual foot motion.
This motivates the staged clearance bonuses already running and considering
more coherent or less attenuated pitch exploration if learning stalls.
The live action mapping has not been changed on this evidence alone. See
[exploration arithmetic](../reviews/ankle-v23-curriculum/20260915/initial-exploration-arithmetic.json).

## Separate linear-pitch training experiment

Geometry reconstructed from saved joint/sole positions supports investigating
rotation. In four logged environments of the milestone run, the largest
qualified front-left lift from the first tread was 126.9 mm, with only 3.75
degrees of actual sole-normal tilt and a knee at approximately -90 degrees.
One floor-start front-right lift reached 113.4 mm with 2.41 degrees of tilt.
The main run's sampled floor-start maximum was 156.3 mm with 25.41 degrees
of tilt. The reconstruction error was below 0.017 mm. These are sparse
training samples, not a full evaluation or evidence that the reward arm is
superior. See [recorded swing geometry](../reviews/ankle-v23-curriculum/20260915/recorded-swing-geometry.json).

The isolated `ankle_v23_realstairs_linear_pitch` package changes the sole-pitch
mapping from `110 * action * abs(action)` to `110 * action` degrees. A filtered
command of 0.1 now requests 11 degrees instead of 1.1. This is a desired whole
sole orientation, not an enlarged ankle joint range. The pitch request range,
0.7 rad/s angular-rate clamp, action filter, bounded IK, 2-degree target guard,
motor limits, lift rewards, observations, and full-height resets are retained.
Its traces additionally record raw/filtered actions and requested pitch.

`stairs-linear-256-s3281` started at 16:04 PDT with 256 environments, a fresh
Beta(8,8) actor and 256 planned updates (8,388,608 transitions). Worker PID
16444 was confirmed by command line and was recording trajectories at 16:07
PDT. The two earlier jobs continue unchanged. Environment counts and budgets
differ, so this remains an exploratory comparison. The archive family is
`simulation/isaac/models/ankle-v23-realstairs-linear-pitch/`.

The first 96 logged pitch requests, sampled before the first PPO update,
had mean absolute value 9.21 degrees and RMS 11.05 degrees. The calculated
stationary initial RMS is 11.08 degrees for the linear mapping versus 1.90
degrees for the quadratic mapping. This confirms the intended exploration
change, not climbing performance. See
[initial pitch observations](../reviews/ankle-v23-curriculum/20260915/linear-pitch-initial-observations.json).
At 16:14 PDT the new run had logged update 5/256. Main and quadratic
milestone runs had logged 48/512 and 45/512. None had a stage or full-climb
completion. The new run's early ended episodes were falls; episodes that
survive to the 96-second time limit are not yet represented in those counts.

All active jobs retain four 180 mm risers and 250 mm treads. No full climb
has succeeded, and no independent evaluation or new video has been run.
At the 15:37 PDT inspection, the corrected main run had logged update 15/512
and 1,056 ended episodes (377 falls); the milestone run had logged update
9/512 and 282 ended episodes. Both still had zero first-tread completions
and full climbs. Both worker command lines were confirmed live. Generic
placement counters are not treated as ascent evidence because they can
include re-landings on an already occupied tread.

At 15:50 PDT, the main and milestone runs had logged updates 27/512 and
24/512. Reading 1,488 and 874 ended episodes, respectively, found no newly
credited tread-height gains in either floor or three-on-first starts. The
milestone run had 475 episodes earning lift tiers. Its first-eight-environment
samples included 90 mm qualified clearance above the floor and 120 mm above
an initially occupied first tread, both on front feet. No rear-foot lift tier
appeared in that small sample. This is exploration progress, not landing
success or a controlled estimate of reward superiority. Both runs continue
unchanged; no additional simulator evaluation was launched.
At 15:57 PDT both workers were confirmed live by command line. The latest
published totals were 2,254 ended episodes for the main run and 1,174 for the
milestone run, with zero stage or full-climb completions in both.

## Additional research considered

[StairMaster](https://arxiv.org/html/2606.25765v1) uses look-ahead trunk-pitch
guidance, terrain-specific foothold penalties, and depth-memory training for
a Go2. It supports looking at body posture and edge clearance together. Its
robot and hollow-stair task differ from this robot, and its pitch targets
cannot be copied without checking our narrower joint limits. The current
experiment does not add that pitch reward or claim camera-only control.

[Versatile Locomotion Skills for Hexapod Robots](https://www-video.eecs.berkeley.edu/papers/IROS_2024___march_final.pdf)
uses privileged teachers and depth-based students on a low-cost six-legged
robot. Its stair reward removes several restrictions used for other skills.
That supports reconsidering unnecessary gait restrictions, but the extra
legs and different geometry prevent direct capability comparisons. Its
small-to-large stair curriculum is not being adopted: every active stair
training job here retains 180 mm risers.

## Completion remains unproven

Keep the full goal active. A useful intermediate pose, partial training
milestone, or completed training budget does not establish a solution.
Required evidence includes a connected floor-start ascent of all four real
height risers, retained physical limits and contact criteria, and a frozen
policy recording on the same full-height stairs. Camera-only control remains
separate work after a successful teacher is found.

No standalone tests, evaluation batch, small-stair recording, or hardware
check was run. Training and the user's requested IK/static analysis are the
work performed in this continuation.
