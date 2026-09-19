# Full-height stair reward analysis — 2026-09-18

**No complete four-riser climb has succeeded.** The target remains four
180 mm risers and 250 mm treads in simulation. No small stairs, near-goal
hover resets, hardware motion, or camera-only deployment is part of this
experiment. The best completed floor-start recording reproduces a
qualified front-pair hold at 39.85 seconds from the floor. The earlier two
recordings each achieved one front-foot credit. The original pair-transfer
run has now achieved rear placement and whole-robot first-tread holds from
the grounded three-on-first curriculum. A complete floor-start transfer
and the four-riser climb remain unresolved.
[Floor-start front-pair video](../reviews/ankle-v23-curriculum/20260915/video-clearance10-pair-s3291/frozen-floor-start-180mm.mp4).

## Current results, September 18 at 11:41 PDT

**No floor-start rear plant or complete climb yet.** Three RL workers are
active on four 180 mm risers and 250 mm treads: tilted-transfer,
swing-clearance, and the new `stairs-floor-load-consistent-s3281` trial.
The latter starts from the archived first flat second-foot event actor,
with an exact 14-tensor actor transfer and fresh critic/optimizer. The
recording of that actor has completed: it achieved the actual whole-first-tread
hold at **175.1 simulated seconds**, then reached guide 109 before the
180-second clip limit. It did not reproduce the second-tread plant. It starts
with three feet on tread 1 and rear-right on the floor, so this is a frozen
rear-transfer result, not a floor-start ascent. The video is 181 seconds at
10 fps, including the final held frame; the recorder exited successfully.
[Full-height video](../reviews/ankle-v23-curriculum/20260915/video-flat-second1-s3291/frozen-flat-second-attempt-180mm.mp4),
[current snapshot](../reviews/ankle-v23-curriculum/20260915/continuation-20260918-1140.json).

At step 1,400 the corrected floor trial has **25 measured rear-left
unloading events from actual floor starts**, 42 front-pair holds and no
floor-start rear plant. The first unload is at step 957: rear-left contact
magnitude 2.931 N, front-right upward load 1.282 N, other-three support
score 0.427, and credits `[1,0,1,0]`. This passes the original new guide's
3 N unload/0.35 support gate; it is not airborne, a planted rear foot,
a whole-tread hold or a full climb. The same run separately has 25 first-tread
holds and one tilted second-tread contact from the grounded curriculum.
[Preserved measured events](../reviews/ankle-v23-curriculum/20260915/floor-unload-first-events.json).

The corrected floor trial has reached its final guide 279. Earlier, at step 960,
16 floor environments had passed the front-pair phase. Sampled rear-left
upward loads were about 2.5–2.8 N in three leaders, but front-right was
unloaded in two of them. Supporting front-right while unloading rear-left
remains inconsistent. These are early changing-policy observations, not rear
takeoff or a result of a controlled comparison.

### Corrected floor unloading preloads

The first new floor-transfer trial had an implementation error. Its 14
appended postures came from a constrained contact-force optimization, but
their servo targets came from a separate minimum-torque LP. That LP omits
the selected rear-leg maximum load and redistributes the forces. At the
final posture it loads rear-left to **11.734 N**, instead of the intended
**2 N**. The original pose and force solution was valid; the commands did
not preserve that force allocation. These are calculated equilibrium
loads, not observed training contact forces.

The corrected trial computes `tau = -(gravity + J^T f)` from the exact
saved constrained vertex forces, then `q_target = q + tau / 30`. The final
normal loads are `[12.041, 2.000, 2.362, 17.383] N`; rear-left force magnitude
is 2.625 N, below the existing 3 N guide-completion gate. Peak static torque
is 91.93% of the weak continuous cap at the endpoint and at most 94.58%
across the 14 poses. The largest target correction is 1.809 degrees. The
physical poses, all previous references, rewards, actuator limits, action
mapping, filter and success criteria are unchanged.
[Force/target arithmetic](../reviews/ankle-v23-curriculum/20260915/floor-preload-consistency.json),
[corrected trial contract](../reviews/ankle-v23-curriculum/20260915/floor-load-consistent-contract.json).
At the calculated endpoints, all 14 guide support gates pass; their minimum
predicted score is 0.446, and the final rear-left other-three score is 0.787.
The recorded contact losses therefore remain a dynamic tracking/control
problem, rather than a reason to weaken the actual contact gates.
[Offline guide arithmetic](../reviews/ankle-v23-curriculum/20260915/floor-load-consistent-guide-arithmetic.json).

![Calculated unloading force correction](../reviews/ankle-v23-curriculum/20260915/floor-preload-correction.png)

**The earlier inference about insufficient action authority is withdrawn.**
It used the same mismatched preload. With the corrected starting targets,
the original 25 mm/s XYZ and 0.35 rad/s pitch residual limits admit
instantaneous rear-left takeoff solutions, including the first filtered
control tick. The solutions reach the original weak torque cap and assume
zero velocity and ideal sticking contacts; they do not prove a finite or
robust maneuver. No action gain increase is used in the corrected run.
[Corrected inverse dynamics](../reviews/ankle-v23-curriculum/20260915/floor-controlled-takeoff-load-consistent.json).
The old `floor-controlled-takeoff.json` and
`floor-dynamic-command-authority.json` describe the mismatched starting
targets and must not be applied to the corrected unloading reference.

Both erroneous trials were stopped and retained with their source
provenance: floor-rear-transfer at step 820 (two front-pair holds, no rear
unload) and takeoff-authority at step 100. The latter had not saved a PPO
checkpoint. Their results are not evidence against corrected unloading.
The mature light-bearing and detour trials were also archived and stopped
to reallocate compute; their latest stable saved models are preserved.

### Long first-tread delay and foot spacing

At 138.05 seconds, the frozen actor is at guide 104, exactly the first-tread
hold gate. All ordinary guide errors pass, but rear-right's maximum sole
gap is 8.091 mm, above the unchanged 8 mm plant condition. Offline geometry
also finds 0.425 mm contact overlap between the right feet. None of the
last eleven one-second samples has all four feet within the geometric
plant gate simultaneously. Sparse samples do not reconstruct the full
20 Hz hold/slip conditions.

The policy eventually satisfies the actual first-tread gate at 175.1 s.
At the clip limit, FL is lifting toward tread 2 but touches its vertical
riser, RL has lost contact, and the right feet overlap by up to 1.32 mm.
There are only 4.9 s of recording after the first-stage hold; the clip limit
does not establish what a longer second-tread attempt would do.

The nominal hold reference itself has only 0.474 mm separating-axis space
between the right foot collision boxes; its preceding poses also have
submillimetre clearance. This explains why modest tracking errors can
produce contact between those feet. No active reference has been changed
for this finding.
[Recorded geometry](../reviews/ankle-v23-curriculum/20260915/flat-second-frozen-analysis.json),
[reference spacing](../reviews/ankle-v23-curriculum/20260915/first-tread-foot-spacing.json).

Two fixed-anchor attempts to gain 2 mm spacing failed numerically; neither
is a proof that such a posture cannot exist. Letting the rear-right
**landing location** shift within 15 mm produced accepted 2 mm and 5 mm
spacing targets. The final rear-right contact shifts -2.45 mm in X and
+7.66 mm in Y, with all four heel gaps below 3.14 mm and peak torque 69.57%
of the weak continuous cap. Each of the two endpoint changes is under
0.76 degrees per joint. These are alternate static landing endpoints;
an approach from the preceding rear-foot swing is still needed. No foot
slide or state teleport has been executed or added to training.
[Alternate landing endpoint](../reviews/ankle-v23-curriculum/20260915/first-tread-relocated-hold.json).

A read-only audit of the original second-tread touchdown/load/roll
allocations found their saved and independent normal forces agree to
numerical precision. The newly discovered floor preload error should not
be generalized to those original landing references.
[Allocation audit](../reviews/ankle-v23-curriculum/20260915/second-tread-preload-allocation-audit.json).

Only the requested RL training/recording and offline source, geometry,
force and dynamics calculations were run. No additional test suite,
evaluation batch or hardware check was run.

## Earlier results, September 18 at 10:58 PDT

**The tilted-transfer branch has produced one strict flat FL plant on tread2.
There is still no floor-start rear plant, complete floor-start transfer or
four-riser climb.** At step15,731, environment134 (grounded reset2) held a
flat second-tread contact for the existing0.25s plant gate:2.275 N upward
load and maximum absolute traction-corner gap5.087 mm. This occurred at
episode129.75s, after an actual whole-first-tread hold and a tilted tread2
touchdown. Credits became `[2,1,1,1]`. The exact behavior actor is archived
as `second_tread_foothold_behavior_1.pt`, SHA256
`0fdb436af1236cb51628de32aad33e67a7a15ed069ec52308b859a5b31bfe705`.
[Archived milestone and trajectory analysis](../reviews/ankle-v23-curriculum/20260915/first-flat-second-plant-analysis.json).

This is **per-foot progress, not a stable whole-robot maneuver**. At the
nearest logged state (episode130.2s), FL still has2.62 N load and3.82 mm
maximum sole gap, but FR is touching RR instead of the tread. Offline FK
finds right-foot collision-box overlap up to1.08 mm, with opposing lateral
contact forces. RL also overhangs the first riser's front edge, reducing its
conservative support score. Later FL contact is lost; persistent credit
records history, not continuous support. FK matches logged corners within
2.77 micrometres. The policy changed during the episode.

One requested frozen video of this newer actor is now recording on the same
180 mm stairs and grounded reset, seed3291, with a180s maximum and no seed
sweep. It allows time after tilted touchdown to attempt flattening. The
earlier frozen touchdown actor's130s video failed at rear guide11.
The new clip has already passed that early guide, but its final outcome is
pending. Its report is
[here](../reviews/ankle-v23-curriculum/20260915/video-flat-second1-s3291/report.json).

At10:49 the five changing-policy runs had the following results. The
free-swing control was then archived and stopped atstep27,040 to make room
for recording; its stable `model_208.pt` and51 behavior actors are retained.
Four other RL runs continue. These are unequal training ages and event
counts, not controlled success-rate estimates.

| Branch | Step | Curriculum first-tread holds | Tilted tread2 touchdowns | Flat tread2 plants | Full climbs |
| --- | ---: | ---: | ---: | ---: | ---: |
| light_bearing | 27,900 | 60 | Not instrumented | 0 | 0 |
| free_swing | 26,760 | 39 | 0 | 0 | 0 |
| tilted_transfer | 24,960 | 386 | 2 | 1 | 0 |
| detour | 23,940 | 307 | 1 | 0 | 0 |
| swing_clearance | 21,800 | 104 | 0 | 0 | 0 |

[Observed snapshot](../reviews/ankle-v23-curriculum/20260915/continuation-20260918-1049.json).
The newer swing-clearance reward therefore has **not yet demonstrated an
improvement**. It multiplies only the planned-swing pose bonus by
`0.25 + 0.75 * exp(-(contact_norm/2N)^2) * clamp(clearance/10mm,0,1)`.
The clearance is computed from all physical foot collision vertices; a
foot may remain behind the riser or lift above its top. Loaded phases,
actions, observations, guides and physical/success constraints are unchanged.
At the earlier measured wall contacts this reduces the tracking potential
bonus to25%; it does not award a repeated airtime bonus.
[Contract](../reviews/ankle-v23-curriculum/20260915/swing-clearance-contract.json),
[recorded-state arithmetic](../reviews/ankle-v23-curriculum/20260915/swing-clearance-reward-arithmetic.json).

![Contact geometry and reward hypothesis](../reviews/ankle-v23-curriculum/20260915/swing-clearance-reward-analysis.png)

The primary [legged_gym reward source](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)
distinguishes foot impacts on vertical surfaces through horizontal versus
vertical force. This project already tried a broad version earlier without
solving the climb. The new term is specific to expected swing phases. The
[Solo12 study](https://www.nature.com/articles/s41598-023-38259-7) also warns
that excessive penalties can favor standing still; this motivates retaining
part of the tracking signal and checking actual progress, not assuming that
adding a penalty will help.

Offline continuation also finds a wider route: from the accepted250mm lift,
four sideways/posture adjustments establish30mm horizontal riser clearance;
the whole foot then lifts to approximately370mm while keeping that buffer.
Eight further accepted static endpoints bring the whole foot inside tread2.
The largest change among these new lift/crossing endpoints is15.52 degrees,
and peak static torque is at most82.7% of the weak continuous cap.
Tilted landing and heel lowering reach20mm; the15mm attempt fails locally.
This route is **not installed in any running reference**, and its connecting
dynamics remain unproved.
[Wider lift](../reviews/ankle-v23-curriculum/20260915/second-tread-buffer-refinement.json),
[crossing](../reviews/ankle-v23-curriculum/20260915/second-tread-buffer-crossing.json),
[landing](../reviews/ankle-v23-curriculum/20260915/buffer-second-tread-landing.json).

The earlier floor-start instantaneous takeoff solutions require at most
1.924 degrees of servo target change from the accepted2 N rear-left pose,
within the existing2-degree one-policy-step target guard. This is only
additional actuator arithmetic; the Cartesian controller and a finite
contact-consistent takeoff are still unproved.
[Target arithmetic](../reviews/ankle-v23-curriculum/20260915/small-shift-dynamic-target-arithmetic.json).

Only authorized training, requested recording, source/log inspection and
offline calculations were used. No extra test suite, evaluation batch,
build or hardware check was run.

## Earlier results, September 18 at 05:00 PDT

**The requested frozen video is complete and failed to reproduce the training
touchdown. No full climb or complete floor-start first-tread transfer has
succeeded.** The saved touchdown actor ran for 130 simulated seconds on the
four 180 mm risers, with the original grounded three-on-first reset and
matching runtime provenance. It stopped at rear guide11, with credits
`[1,1,1,0]`, no first-tread hold, no second-tread touchdown and no fall.
The video includes one additional held second (1,310 frames at 10 fps).
[Completed full-height video](../reviews/ankle-v23-curriculum/20260915/video-second-touchdown1-s3291/frozen-second-tread-180mm.mp4),
[recording report](../reviews/ankle-v23-curriculum/20260915/video-second-touchdown1-s3291/report.json).

Recorded-state arithmetic initially isolates the sole tracking gate:
28.24 mm error versus its 25 mm limit; joint, root, orientation and support
checks pass. More detailed FK also finds the RR sole touching the vertical
first riser: 0.135 mm collision-box overlap and 1.54 N measured contact
force, despite **zero upward force**. The foot is not freely suspended.
Relaxing the sole gate alone would hide this contact. The completed video
therefore does not support a reliable frozen rear transfer.
[Frozen gate calculation](../reviews/ankle-v23-curriculum/20260915/second-touchdown-frozen-analysis.json),
[contact and controller reconstruction](../reviews/ankle-v23-curriculum/20260915/current-stair-stall-analysis.json).

The new `detour` training branch replaces the 23 appended second-tread
poses with 33 accepted static poses: grounded FL unloading, smaller lifts,
earlier rocking of existing support corners, a lateral swing-foot path,
crossing and tilted landing. The first joint change is 5.15 degrees instead
of 52.72; the largest appended change is 18.85 degrees. The final lateral
lift keeps the whole physical foot 10 mm behind the riser, and crossing
poses clear its top by approximately 10 mm. Earlier rocking poses have
only about 8–9 mm horizontal buffer. This is a sequence of static endpoints,
not a proved continuous trajectory. No new reset, motor strength, injected
force or success relaxation is used.
[Reference contract](../reviews/ankle-v23-curriculum/20260915/detour-contract.json).

![Offline second-tread detour, not an executed climb](../reviews/ankle-v23-curriculum/20260915/second-tread-detour-plan.png)

At step1,200 the selected detour leaders reached guide115 after actual
curriculum first-tread holds. Their whole FL foot was about 250–253 mm
above the floor (70–73 mm above tread1), while the foot pressed into the
second riser with 2.08–3.06 N contact force. Other-three support was
0.956–0.975. The contact-norm gate correctly stops advancement. Older
`free_swing` and `tilted_transfer` leaders similarly press into this riser
at guide109. Current guidance similarity does not itself include the
selected foot's contact magnitude; the generic readiness term can select
a different foot. This identifies an explicit free-swing/obstacle reward
hypothesis, rather than evidence that extra training alone will resolve it.

Four workers were confirmed live at04:51. Counts are repeated events during
changing-policy training at different ages, **not success rates**.

| Branch | Step | Curriculum first-tread holds | Tilted tread2 touchdowns | Flat tread2 plants / full climbs |
| --- | ---: | ---: | ---: | ---: |
| light_bearing | 4,460 | 24 | Not instrumented | 0 / 0 |
| free_swing | 3,660 | 25 | 0 | 0 / 0 |
| tilted_transfer | 2,160 | 18 | 0 | 0 / 0 |
| detour | 1,020 | 10 | 0 | 0 / 0 |

The original second-tread branch was preserved and stopped atstep3,860
with49 curriculum holds and its one tilted touchdown, freeing the detour
slot. Its stable `model_24.pt`, 62 other checkpoint/behavior files and source
provenance are archived. This resource choice does not establish convergence
or infeasibility. No floor-start rear plant was found in the four current
branches at the preceding04:42 read.
[Process/log snapshot](../reviews/ankle-v23-curriculum/20260915/continuation-20260918-0451.json).

Separate **floor-start** mechanics starts from the actual recorded front-pair
pose. With its existing footprint, rear-left-first static unloading reaches
2 N; rear-right-first reaches4 N. A refined14-endpoint posture/load sequence
reaches the2 N rear-left pose with each joint transition no more than6 degrees.
Further static unloading fails locally; this is not a proof that the task is
impossible or that rear-left must always move first.
[Rear-order calculation](../reviews/ankle-v23-curriculum/20260915/recorded-pair-rear-orders.json),
[small shifts](../reviews/ankle-v23-curriculum/20260915/recorded-pair-small-load-shift.json),
[relaxed load shifts](../reviews/ankle-v23-curriculum/20260915/recorded-pair-relaxed-load-shift.json).

An instantaneous floating-base dynamics calculation at that refined pose
admits rear-left takeoff with zero rear-left contact, original motor caps,
and modest sideways/backward body acceleration. For a0.5 m/s² minimum
upward foot acceleration, body acceleration is approximately
`[-0.372, +0.635, -0.043] m/s²`. Static three-contact equilibrium fails at
the same pose. The calculation follows the mass-matrix/contact-Jacobian
formulation described in [MIT's multibody dynamics notes](https://underactuated.mit.edu/multibody.html).
It uses URDF inertias, ideal sticking contacts and zero initial velocity;
peak torque reaches the weak continuous cap. **No time integration,
finite takeoff trajectory, robustness margin or hardware motion is proved.**
This supports allowing dynamic weight shifts in RL rather than requiring
a nearly motionless transfer. The current free-order reward already has no
hard fixed-speed or level-body requirement; no such restriction was removed.
[Instantaneous takeoff arithmetic](../reviews/ankle-v23-curriculum/20260915/small-shift-dynamic-takeoff.json).

Only requested RL training/recording and offline arithmetic were run. No
additional evaluation sweep, test suite, build or hardware check was run.

## Earlier results, September 18 at 04:20 PDT

**One actual loaded, tilted front-left touchdown on tread2 has occurred in
training. There is still no qualified flat second-tread plant, complete
floor-start first-tread transfer, or four-riser climb.** The milestone starts
from the grounded curriculum with three feet on tread1 and rear-right on
the floor. It first executes the existing whole-first-tread hold, then
continues to the second tread under the original constrained physics.
The policy changes during training; this is not frozen-policy reliability.

At step1579, environment163 held the tilted touchdown for 0.25 seconds.
The front-left foot carried 2.011 N, all traction corners were inside
tread2, and its largest traction-corner gap was 22.552 mm. Flat-foot credit
correctly remained `[1,1,1,1]`. The exported actor is preserved as
`second_tread_touchdown_behavior_1.pt`, SHA256
`7c1e36f307be507c2e499def921389845554e7bf08d4e330ab68234d176ee104`.
Selected logged joints reconstruct the simulator's traction corners within
9.4 micrometres. Earlier in this trajectory the whole FL collision foot
cleared the first tread by 183.23 mm with zero measured contact force.
[Milestone archive](../reviews/ankle-v23-curriculum/20260915/second-tread-milestone-archive.json),
[actual lift measurements](../reviews/ankle-v23-curriculum/20260915/second-tread-training-lift-analysis.json).

After nominal guidance faded, the robot actually unloaded FR and lifted
its whole collision foot **8.49 mm** above tread1 at episode87s. Its measured
force was zero and other-three support was0.970. The initial decrease in
the original FL candidate's other-three support is expected as FR unloads;
it is not itself failure. Later useful support deteriorates and FL lifts
off its new contact. This provides an observed partial next-foot maneuver
to reinforce, but no second plant or stable continuation.
[Handoff trajectory analysis](../reviews/ankle-v23-curriculum/20260915/second-touchdown-handoff-analysis.json).

![Actual training geometry and reward arithmetic](../reviews/ankle-v23-curriculum/20260915/actual-second-touchdown-and-reward.png)

The new `tilted_transfer` trial addresses a specific missing signal. The
old transfer potential determines higher and lower supports entirely from
flat-foot credits. Consequently it remains exactly **zero** throughout
this tilted touchdown and subsequent FR takeoff, because all credits are1.
The new branch recognizes a currently loaded, fully contained next-tread
corner as a higher support for **transfer shaping only**, after the actual
curriculum first-tread hold. It retains the existing40mm bearing geometry,
measured contact>1N, full contact-force norm for unloading, and all strict
flat-plant/success conditions. Temporary load levels never become foot credit.

For an uncredited tilted support, load shaping saturates at2N, consistent
with the feasible touchdown calculation. Credited supports retain the
original aggregate force scale. On the recorded trajectory the proposed
transfer potential progresses from3.019 at79s to7.013 at86s, then falls to
0.183 when support deteriorates at90s. These are **potential values**, not
per-step bonuses: the reward still uses `gamma*Phi(next)-Phi(previous)`.
The separate flat landing, lift, readiness and task rewards remain intact.
This is a hypothesis about useful shaping, not evidence the new policy
has learned it.
[Arithmetic](../reviews/ankle-v23-curriculum/20260915/tilted-transfer-reward-arithmetic.json),
[training contract](../reviews/ankle-v23-curriculum/20260915/tilted-transfer-contract.json).

Four 256-environment workers were confirmed live at04:20. Every run uses
the same archived stage-event4 actor, seed3281, and fresh critic/optimizer;
startup confirms all14 actor tensors transferred in the new run. Counts
below are repeated changing-policy events at different training ages.

| Training branch | Step | Curriculum first-tread holds | Tilted tread2 touchdowns | Flat tread2 plants / full climbs |
| --- | ---: | ---: | ---: | ---: |
| second_tread | 2,900 | 43 | 1 | 0 / 0 |
| light_bearing | 2,280 | 20 | Not instrumented | 0 / 0 |
| free_swing | 1,460 | 18 | 0 | 0 / 0 |
| tilted_transfer | 120 | 0 | 0 | 0 / 0 |

`free_swing` fixes another observed reward error: a foot pushing downward
against another link was counted as unloaded when vertical force was
clamped tozero. A recorded FL force of `[-2.21,3.89,-12.87] N` accompanied
a1.785mm collision-box overlap with the rear-left proximal link. Candidate
unloading now uses the magnitude of the measured force, and the20 appended
unloaded guide poses require magnitude<1N before advancing. This does not
detect every possible opposing-contact cancellation. `tilted_transfer`
inherits this correction. The independent `light_bearing` branch changes
support-force saturation3N→1.5N while retaining contact>1N and geometry.
That earlier hypothetical normalization is now installed **in that branch**.
[Free-swing contract](../reviews/ankle-v23-curriculum/20260915/free-swing-contract.json),
[light-bearing contract](../reviews/ankle-v23-curriculum/20260915/light-bearing-contract.json),
[snapshot](../reviews/ankle-v23-curriculum/20260915/continuation-20260918-0420.json).

The older prospective-load branch was archived and stopped atstep4,620
with33 curriculum holds and no second plant/full climb to free the new
trial's slot. Its stable `model_32.pt` and45 available behavior actors,
source provenance and logs are preserved. Earlier this continuation,
residual-handoff stopped atstep7,460 with41 holds, and rear-landing stopped
atstep10,420 with197 hold events (194 saved stage actors). These are
resource decisions, not convergence or impossibility claims.

Additional offline mechanical calculations explore alternatives without
changing the running references. A grounded four-contact posture sequence
fully unloads FL through small joint changes; subsequent lifting reaches
250mm absolute foot height. Switching the support feet to existing contact
corners extends this to330mm before the360mm attempt fails. After the tilted
FL tread2 contact, a rear-right-first sequence reaches1N RR load at94.6% of
the weak continuous cap, but the zero-load attempt fails. Releasing extra
edge force points does not solve it. The rear-left-first attempt fails at
8N. These are local static results under fixed contact assumptions, not
proof that another posture or dynamic sequence is impossible. The actual
FR takeoff above uses a different pose and randomized actuator strengths.
[Grounded unloading](../reviews/ankle-v23-curriculum/20260915/second-tread-grounded-unload.json),
[early rocking](../reviews/ankle-v23-curriculum/20260915/second-tread-early-rocking.json),
[rear-foot alternatives](../reviews/ankle-v23-curriculum/20260915/second-tread-rear-alternatives.json).

Research supports preserving useful motion freedom while enforcing the
robot's constraints. [Blind Stair Climbing](https://arxiv.org/html/2402.06143v1)
uses a position goal to allow variable approach and stepping speeds, rather
than imposing a constant velocity. [CaT](https://constraints-as-terminations.github.io/)
explicitly incorporates constraint violations into learning and reports
Solo-12 climbing a24cm platform. These are different robots and do not
establish feasibility here. My inference is to reward actual contact
progress and load transfer while retaining the original motor limits;
the precise foot order and timing should remain adaptable. CaT itself has
not been implemented in this experiment. [StairMaster](https://sivan666666.github.io/StairMaster/)
also highlights depth artifacts, temporal memory and sensing blind spots;
those remain future camera-policy work, not properties of the current
privileged teacher.

One requested frozen-policy full-height recording is running from the
saved touchdown actor, seed3291, maximum130 simulated seconds. At this
snapshot it has reached41.05s without a first-tread hold. Its result is
pending; the training touchdown is not claimed reproduced. No separate
test suite, evaluation batch, hardware check, or small-stair run was done.
The plot above reconstructs existing logged states without launching a
policy. All training still uses privileged simulation inputs with a D405
proxy; camera-only ascent remains unsolved.

## Earlier results, September 18 at 03:41 PDT

**Full floor-start transfer, a qualified second-tread plant, and the
four-riser climb remain unsolved.** All four current training workers were
confirmed live. At this snapshot the landing-reward trial has 169 curriculum
holds at step 9,120; residual-handoff has 41 at step 7,000; prospective-load
has 24 at step 1,860. Their logged plant levels still peak at tread1. The
new second-tread-guidance branch has started and is at step100. These are
changing-policy events from different training ages, not frozen success
rates. The prospective-load trial's first 13 stage actors and early front/
rear plant actors are archived.
[Snapshot](../reviews/ankle-v23-curriculum/20260915/continuation-20260918-0341.json),
[prospective-load milestones](../reviews/ankle-v23-curriculum/20260915/prospective-load-milestone-archive.json).

`stairs-second-tread-s3281` began at 03:39 with 256 environments, seed3281,
the same frozen event4 actor, and fresh critic/optimizer. Startup confirms
all 14 actor tensors transferred exactly. It clones the prospective-load
reward/controller and appends **23 accepted static FL poses**, extending
rear-curriculum guidance from 105 to 128 poses. The new sequence includes
lifting above tread2, crossing it, tilted touchdown, modest loading and
partial heel lowering. It cannot start until the existing actual all-four
first-tread one-second hold has happened. Floor/free variants and grounded
resets are retained; there is no all-four-on-first reset or state injection.

The bounded nominal servo guidance continues through these poses instead
of ending at the first tread. It completes only after a separately measured
loaded FL touchdown on tread2, held 0.25 s with other-three support, bounded
heel gap, low slip/speed and the original fall/range guards. That intermediate
event has its own counter and actor export. **It does not award a flat foot
credit, a floor-start transfer, or full-climb success.** All original plant,
stage, full-climb, physical motor, joint and rate checks remain unchanged.

The main limitation is substantial: the first appended pose differs by up
to **52.7 degrees at one joint**, 55.3 mm in root position and 15.5 degrees
in body orientation. The controller applies its normal bounded commands;
the robot does not jump between states. These discrete endpoint poses do
not prove a connected feasible path or stable execution. The experiment
asks whether RL can resolve the connections under actual constrained
physics. The longer route also changes the rear guidance fraction observed
by the warm-started actor, so this is not a reward-only comparison.
[Guidance contract and source hashes](../reviews/ankle-v23-curriculum/20260915/second-tread-guidance-contract.json).

The landing-corridor trial was archived and stopped to free this slot. Its
last snapshot is step4,420 with 26 curriculum holds and no second-tread
plant/full climb. `model_32.pt`, all 26 saved stage actors, provenance and
metrics are preserved. Many replicas remained at guide98 after the corridor
allowed progress past guide97. Stopping is a resource decision, not proof
of convergence or impossibility.

New offline contact calculations distribute forces over actual collision
vertices near the horizontal contact plane. They retain the original
physical constraints. On tread2 this does not resolve FR unloading: the
2 N endpoint is feasible, while the 1 N attempt stalls around 1.11 N at
the weak motor cap. Allowing FR to roll to a single corner also fails.
[Edge-force calculation](../reviews/ankle-v23-curriculum/20260915/second-tread-edge-unloading.json).

For the unresolved floor connection, the same method improves the earlier
FL-replant endpoint to a feasible **2 N RR load**. Releasing additional
contact edges still does not reach 1 N. Moving the lightly loaded RR
contact forward along the floor in four 10 mm increments gives accepted
static endpoints and reduces peak normalized torque from 0.941 to 0.866.
The subsequent 1 N unload fails. These calculations explicitly move a
contact location: they do not establish sliding dynamics, no-slip motion,
takeoff, or a connected path from the recorded floor-start policy.
[Floor repositioning](../reviews/ankle-v23-curriculum/20260915/replanted-pair-floor-reposition.json).

The 40 mm floor-reposition endpoint carries approximately
`[1.23, 18.96, 11.60, 2.00] N` on FL/RL/FR/RR. Its actual contact geometry
is accepted statically, but the current bearing reward normalizes each
supporting foot by 3 N. The weakest-other-three score is therefore **0.409**,
below the lift threshold 0.65. A hypothetical 1.5 N bearing normalization
raises that score to **0.818**. This is a concrete shaping mismatch to
investigate; the new normalization is **not installed**, and the rear foot
still carries 2 N, so this endpoint earns no supported-lift milestone.
[Reward arithmetic](../reviews/ankle-v23-curriculum/20260915/light-bearing-reward-arithmetic.json).

This continuation ran authorized training and offline calculations, with
read-only log/source inspection and checkpoint preservation. No separate
test suite, policy evaluation batch, new video rollout or hardware check
was run. The existing full-height videos remain available below. The
policies still use privileged simulation inputs rather than camera-only
control.

## Earlier results, September 18 at 03:20 PDT

**No complete floor-start transfer, second-tread plant, or four-riser climb
has been established.** Four 256-environment RL jobs remain active. The
landing-reward trial has 134 grounded-curriculum first-tread holds at step
7,500, residual-handoff has 40 at step 5,340, and landing-corridor has 19 at
step 2,940. Their different ages and repeated changing-policy events are
not comparable frozen success rates. Later falls remain common. The new
prospective-load trial is at step 280 with no placements yet.
[Training snapshot](../reviews/ankle-v23-curriculum/20260915/continuation-20260918-0320.json).

The oldest pair-transfer parent was archived and stopped at step 17,560
with 31 curriculum holds and no complete floor-start transfer. Its stable
PPO checkpoint `model_136.pt`, all 31 saved stage actors, source provenance
and metrics are preserved. This is resource reallocation, not a claim of
convergence or impossibility. Its event4 actor remains the common starting
actor of the four active comparisons.

The new mechanics calculations isolate two distinct landing difficulties:

| Offline calculation | Result under the original physical limits |
| --- | --- |
| Lower a tilted FL foot from above tread2 to contact | Accepted static samples at 365 and 360 mm; not an executed trajectory |
| Load the new contact | 2 N and 4 N accepted; demanding 8 N failed |
| Lower the heel with a modest load | A 16.6 mm heel gap at 2 N uses 61.2% of the weakest continuous motor cap |
| Roll from the outer sole edge to the tread edge | A nearly coplanar contact transition admits a 15 mm heel gap; a guarded continuation reaches 14 mm |
| Demand 10 mm heel gap in the larger search | Rejected: right-leg parts overlap by up to 28.2 mm, even though the static force calculation fits the torque cap |
| Continue with smaller guarded posture changes | The 13 mm target fails the local feasibility constraints; widening the search and reducing minimum FL load to 1.25 N does not resolve it |
| Keep FL tilted and unload FR next | 4 N and 2 N FR targets are feasible; the 2 N sample uses 92.2% of the weak cap. Complete FR unloading fails |

These are local static results, not proofs of global infeasibility or a
connected dynamic path. A supplementary load calculation distributes
forces over actual foot vertices within 31 micrometres of the tread plane;
it improves the four-contact torque allocation but still does not admit
complete FR unloading at that fixed pose. None of these poses was installed
as a reset or as a new training reference.
[Geometry figure](../reviews/ankle-v23-curriculum/20260915/second-tread-touchdown.png),
[touchdown/load calculations](../reviews/ankle-v23-curriculum/20260915/second-tread-tilted-touchdown.json),
[heel lowering](../reviews/ankle-v23-curriculum/20260915/second-tread-contact-roll.json),
[alternate foot order](../reviews/ankle-v23-curriculum/20260915/second-tread-next-leg.json).

`stairs-prospective-load-s3281` started at 03:16 with the exact same archived
event4 actor, seed 3281, 256 environments, and fresh critic/optimizer as
the residual-handoff comparison. Startup confirms equality of all 14
actor tensors. Its only behavioral change is an extra potential:

`Phi_toe = 2 * sum(credited_levels + supported_next_tread_load)`.

The load term saturates at **2 N**, before flat-foot credit is required.
It requires actual upward contact, support from the other three feet,
all traction corners inside the next tread, the correct horizontal surface,
a nearest-corner gap below 8 mm with a 4 mm Gaussian, penetration no more
than 8 mm, and a candidate heel gap no more than 100 mm. The 100 mm allowance
applies only to this candidate reward; it does not relax the existing
40 mm bearing-support gate or the strict flat plant/stage/full-climb rules.
The existing flat landing potential remains, so the policy still benefits
from flattening after establishing contact. As elsewhere, the added reward
is `gamma*Phi(next)-Phi(previous)` with terminal handling, not a repeated
positive payment for holding or tapping the toe. At step 280 this term has
not yet observed a qualifying next-tread contact. By step 400 it registers
nonzero prospective-load scores (population maxima 0.815 for FR and 0.549
for RR), while actual plant, stage, second-tread and full-climb counts
remain zero. The new shaping signal is being exercised; improved climbing
is not established.
[Experiment contract](../reviews/ankle-v23-curriculum/20260915/prospective-load-contract.json).

The hypothesis follows the distinction between whole-body motion and
landing-force control in the primary
[IROS stair-climbing paper](https://scholarx.skku.edu/item/2d5232e2-d47f-404f-a85e-0989f3f0d99e).
That paper motivates separating the problems; its robot and controller do
not establish feasibility here. [CAJun](https://proceedings.mlr.press/v229/yang23b.html)
likewise separates learned gait/body/foot decisions from a lower-level
motor-command optimizer. The present system has not reproduced that
centroidal controller. A better contact/whole-body controller remains a
possible direction if reward changes do not produce useful continuation.

Work this continuation: authorized RL training, read-only source/log
inspection, offline IK/static-force arithmetic, and a local geometry figure.
No automated test suite, separate policy evaluation batch, new video
rollout, hardware connection or motion was run. The completed full-height
videos remain linked below. The active policies still use privileged
simulation inputs; they are not D405-only controllers.

## Earlier results, September 18 at 02:52 PDT

Full floor-start transfer and the four-riser climb remain unsolved. The
landing-reward job has 82 curriculum first-tread holds at step 5,140; the
residual-handoff job has 30 at step 2,920. At least fifteen completed-stage episodes
in the latter have subsequently ended in falls. Its early longer survival
did not resolve continued climbing. Neither has a second-tread placement.
The original parent has 21 stage holds. These are changing-policy,
repeated curriculum events, not frozen-policy success rates.
[Later saved snapshot](../reviews/ankle-v23-curriculum/20260915/continuation-20260918-0252.json).

The newer frozen stage-event4 actor's requested recording has completed:
90 simulated seconds, 910 video frames including the final hold frame,
with no fall but also no new rear-foot credit. Unlike model56's guide8
stall, this actor reaches guide97 with the entire rear foot over tread1.
At the recorded 81.05 s sample, its sole is 32.85 mm from the reference,
mostly a 26.12 mm X / 19.90 mm Y offset. Vertical error is only 0.85 mm.
The entire physical foot lies at X 405–519 mm, safely inside the first
tread's 340–590 mm bounds, and its lowest point is 184.27 mm. Body, joint,
orientation and other-three support errors satisfy their guide gates in
the inspected late recording; the 25 mm sole-position gate blocks progress.
[Completed video](../reviews/ankle-v23-curriculum/20260915/video-pair-stage4-s3291/frozen-grounded-rear-180mm.mp4),
[geometry arithmetic](../reviews/ankle-v23-curriculum/20260915/landing-corridor-arithmetic.json).

`stairs-landing-corridor-s3281` has started with the same source actor,
256 environments, seed 3281 and fresh critic/optimizer. The new gate
applies only to grounded rear guidance at index 97 or later. It permits
25–40 mm sole error only if the complete URDF foot collision geometry
is inside tread1 with 4 mm horizontal margins, at least 3 mm above its
surface, and within 15 mm vertical reference error. All other guide
conditions and actual force/slip/dwell/fall/full-climb criteria stay intact.
The controller and rewards match the residual-handoff branch. Startup
confirms all 14 actor tensors transferred exactly. At step 720, the new
corridor has admitted 42 otherwise-blocked guide advances and the job has
three actual curriculum first-tread holds, with no second-tread credit or
full climb. This targets an unnecessary exact-position
requirement, without declaring an unplanted foot successful.
[New contract](../reviews/ankle-v23-curriculum/20260915/landing-corridor-contract.json).

The mechanical search now has an accepted static full-height lift. Allowing
all three supporting feet to rock on fixed grounded corners permits the
entire FL foot to reach 370 mm, 10 mm above the second tread. The independent
force allocation uses 83.9% of the weakest continuous motor cap, with
support heel gaps about 21, 15 and 5 mm. Joint, servo-target, friction,
upright and collision limits are retained. The body shifts rearward and
rotates; this is why a flat-foot constraint on the support legs was too
restrictive for that search. Nine forward-crossing samples have also passed
the static constraints. The final entire-foot X span is 606–719 mm, inside
the second tread's 590–840 mm bounds, at 70.9% of the weakest motor cap.
The whole foot retains 10 mm vertical clearance. Neither these endpoints nor their
connecting motion has been installed in training. The positive-knee
alternative refines successfully to 240 mm, then fails at 270 mm.
[Three-support rocking calculation](../reviews/ankle-v23-curriculum/20260915/second-tread-three-pivots.json),
[forward crossing](../reviews/ankle-v23-curriculum/20260915/second-tread-crossing-three-pivots.json),
[updated geometry figure](../reviews/ankle-v23-curriculum/20260915/second-tread-mechanics.png).

The direct next calculation, flattening the FL foot in the air at 400 mm
while only the original three supports carry weight, fails the pose/force
constraints. It used the preceding accepted crossing pose with a 3.76 mm
whole-foot rear-edge margin; downstream landing constraints retained their
4 mm traction inset and collision checks. A failed search is not an
impossibility proof. The next physically motivated route is to lower the
tilted foot onto a valid corner on tread2, transfer weight to that new
contact, and then flatten it. The current load-transfer shaping recognizes
raised feet by completed flat-foot credits, so it may not encourage that
intermediate toe-loading sequence. This is a new hypothesis to calculate
before changing the reward, not a completed toe-contact path.
[Failed unloaded flattening](../reviews/ankle-v23-curriculum/20260915/second-tread-flat-landing.json).

For the missing floor connection, a separate single-corner support search
starts from the prior accepted FL replant / RR-load-at-most-4 N pose. Even
with 40 mm heel allowance, neither the 1 N nor near-zero RR load target
passes the pose/equilibrium constraints. No floor reference was installed
from those failed endpoints.
[Floor unloading calculation](../reviews/ankle-v23-curriculum/20260915/replanted-pair-corner-unload.json).

This supports the current reward direction: real support on the other
three feet, rocking rather than mandatory flat support, full-foot clearance,
and alignment inside a valid landing area. It also shows why guidance
should accept a safe placement area instead of insisting on one exact
point. The remaining floor-start connection, dynamically feasible next-step
motion, and camera-only policy are still open. Four full-height training
jobs are active; no unrelated tests or hardware checks were run.

## Earlier action-handoff analysis, September 18 at 02:27 PDT

There is still no complete floor-start first-tread transfer or four-riser
climb. At control step3,120, the landing-reward continuation has 44 actual
first-tread holds from the grounded rear curriculum. The original parent
has 20 at step13,580. These counts include repeated, changing-policy
episodes and are not frozen-policy success rates or a controlled comparison
of the landing reward alone.

The rich states show a repeatable next failure worth isolating. After the
stage event at step743, environment122 develops front-left backward contact
force within0.85s, consistent with hitting the second riser. By the last
selected state5.85s later, its body has drifted62.6mm forward. The earlier
analysis showed support hips reaching their limits. The command mapping
changes during the two-second handoff: XYZ residual gain goes from0.025 to
0.07m/s per action, and pitch changes from0.35rad/s per action to an absolute
angle request with gain2 and a0.7rad/s clamp. At one recorded post-handoff
state, the old formula requests ±0.7rad/s on supporting ankles; preserving
the residual map would request approximately +0.144 and -0.110rad/s.
Same-state arithmetic over13 selected completed-stage traces confirms this
is a substantial mapping change. It does not establish that it causes every
fall. [Command analysis](../reviews/ankle-v23-curriculum/20260915/rear-stage-handoff-analysis.json).

`stairs-residual-handoff-s3281` is now training in an isolated package.
It preserves the0.025m/s XYZ and0.35rad/s pitch-rate residual map as nominal
reference feedback fades. The original free-floor variant keeps its
mapping. Rewards, grounded reset distribution, source actor,256 environments,
seed3281, fresh critic/optimizer, physical caps and actual success gates
match the landing trial. Startup records exact equality of all14 actor
tensors. At step920 it has three curriculum first-tread holds,51 floor-start
front-pair holds, no second-tread foot credit and no full climb. This is too
early to judge whether the changed handoff improves continued climbing.
[Trial contract](../reviews/ankle-v23-curriculum/20260915/residual-handoff-contract.json).

The1024-environment rear-support and stance-anchor branches were archived
and intentionally stopped at steps6,820 and5,520, each with zero first-tread
holds. Their latest saved checkpoints, provenance and logs remain preserved.
This reallocates compute; it does not prove that more training could never
help those branches. The parent, landing trial and new handoff trial remain
active. One additional full-height video is recording the archived parent
stage-event4 actor with the same seed3291 as the earlier model56 video.
It starts with three feet on tread1 and RR on the floor; no outcome is yet
claimed. No evaluation batch, hardware check or unrelated tests were run.

The next-step load calculations give useful constraints on reward design.
Keeping all three supports flat failed the first10mm FL-lift search.
Allowing FR to rock about its original grounded corner produced accepted,
collision-free static FL endpoints from190 to300mm absolute height, using
39.6–43.8% of the weakest continuous motor cap. The FR heel rises about25mm,
within the existing40mm bearing-support reward gate. The first endpoint
requires a29mm rearward /46mm lateral body shift and as much as53degrees of
joint change; a connecting path has not been established. At330mm, the
search hits joint bounds and self-collisions, before clearing the360mm second
tread. A tilted leading-edge candidate also fails. A positive FL knee-branch
search reaches210mm; its240mm candidate misses the equality tolerance and is
being refined without relaxing physical constraints. None of these endpoints
has been installed in a reset or training reference.
[Static geometry figure](../reviews/ankle-v23-curriculum/20260915/second-tread-mechanics.png),
[rocking calculation](../reviews/ankle-v23-curriculum/20260915/second-tread-rocking-front-lift.json).

The immediate reward priorities remain: transfer weight onto three real
contacts, allow a load-bearing support foot to rock, reward collision-free
clearance of the whole swing foot, and reward flattening/placement inside the
next tread before requiring contact. Actual plant bonuses still require force,
low slip and dwell. A high foot alone is insufficient, and a new reference
should not demand a pose that the motor, joint and collision constraints reject.

The research supports separating stance support from swing placement.
[CAJun](https://proceedings.mlr.press/v229/yang23b.html) trains a high-level
policy for gait timing, base velocity and swing-foot position while a lower
controller handles stance and swing motor commands. That architecture is a
useful direction for this robot, not evidence that its results transfer to
our much smaller servo-limited mechanism. The current branch isolates the
observed action-map change first; it does not implement CAJun's force optimizer.

## Rear-stage milestones and a landing reward mismatch, September 18 at 02:03 PDT

The ongoing original pair-transfer run has produced sixteen first-tread holds
from the grounded curriculum. These satisfy the actual all-four contact,
low-slip, low-speed, one-second hold and retained joint/fall guards. All
sixteen started with three feet already on the first tread and RR on the floor;
none is a floor-start first-tread hold or a complete climb. The first four
holds occurred at episode ages 32.45, 39.55, 52.30 and 47.65 seconds. Their
checkpoints and recorded training trajectories are preserved, along with
events five and six. Earlier portions of an episode can use earlier PPO
weights; saving the event actor does not prove frozen-policy reproduction.
[Archived milestone evidence](../reviews/ankle-v23-curriculum/20260915/pair-rear-stage-archive.json).
The later checkpoints through event16 are also
[archived](../isaac/models/ankle-v23-realstairs-pair-transfer/stairs-pair-transfer-256-s3281/rear-stage-milestones/later-stage-archive.json);
the original event4 warm-start manifest remains unchanged.

Offline FK reconstruction of the six saved trajectories agrees with their
logged sole centers to within 9.1 micrometres. The sampled maximum height
of the entire rear foot is 184.3–190.2 mm. These are 1 Hz samples, so they
cannot rule out a higher transient between samples. The successful motion
uses an oblique body orientation near -46 degrees yaw, with supporting
foot rocking. It establishes a useful rear-transfer subtask; the original
floor-start front-pair recording ends in a different pose and has not been
connected to it.
[Trajectory geometry](../reviews/ankle-v23-curriculum/20260915/rear-stage-trajectory-analysis.json).

A source-level reward mismatch was identified. The old rear trajectory
has 105 poses, but only index 98 meets the nominal crossing latch:
all traction corners above 188 mm and beyond the first edge. Most crossing
poses put the lowest point at 185 mm. The existing force-dependent landing
score is enabled only after that latch, so small tracking errors can leave
landing shaping inactive. Actual plant credit does not require the latch,
which is consistent with the new training milestones.

The isolated `rear_landing` package adds this potential, keeping the
controller, trajectory, physics, resets and actual success checks:

`Phi_land = 4 sum(credited_level + support_other_three * alignment)`

Here alignment is `exp(-mean((corner_z-next_tread_z)^2)/0.020^2)`, requires
the full traction patch inside the target tread's XY bounds, and becomes
zero after the fourth credit. It does not require force on the swing foot
or the old clearance latch. The other three contacts still gate shaping;
real landing still requires the existing force/slip/hold checks. Credited
levels retain the potential across a plant, and the existing gamma
difference and terminal zeroing remain. A hypothetical 13 mm heel lift
with two corners on the plane has alignment potential 3.238; flattening
raises it toward 4. Holding a fixed value gives a small negative gamma
difference, not repeated positive reward.
[Reward arithmetic](../reviews/ankle-v23-curriculum/20260915/rear-landing-reward-analysis.json).

The next 256-environment trial, `stairs-rear-land-s3281`, is now training
and has reached control step 1,000, with eight curriculum first-tread holds,
53 floor-start front-pair holds and zero full climbs. Before launch, its warm start was advanced from
pair-transfer model56 to the archived rear-stage event4 actor, SHA256
`3ec13e3044cf3494a7bb9c2e45a53e7832a36cafe3e980806c7514b1a93fe125`.
Startup confirms all fourteen actor tensors transferred exactly, with fresh
critic and optimizer. This is a continuation comparison, not a matched fresh-optimizer
ablation. It also records selected rear states and saves the first four
rear-plant event actors separately from front-foot events.
[Trial contract](../reviews/ankle-v23-curriculum/20260915/rear-landing-contract.json).
The first rear landing occurs at step713. In the selected leading/highest
rear-state diagnostics through step820, 107 uncredited samples receive
nonzero new alignment shaping while the old clearance latch is false.
This shows the new term is active where intended; it does not isolate its
effect from the more trained warm actor. Early stage, rear-plant and
front-pair actor checkpoints are saved.
[Early observations](../reviews/ankle-v23-curriculum/20260915/rear-landing-early-training.json)
and [new milestone archive](../reviews/ankle-v23-curriculum/20260915/rear-landing-milestone-archive.json).

The requested frozen video uses the previously archived model56, seed3291
and the original grounded rear reset. It completed 90 simulated seconds
plus a one-second final-frame hold, producing 910 frames at 10 fps. It remained
at guide8, pressing the rear foot into the riser without a new rear credit,
fall or joint-range excess. This
recording predates the newly saved stage actors; it is not evidence that
those actors succeed or fail when frozen. The clip labels its three-foot
start explicitly. [Recorded attempt](../reviews/ankle-v23-curriculum/20260915/video-rear-landing-s3291/frozen-grounded-rear-180mm.mp4).
The new trial initially hit a Windows path-length error while copying
provenance, before a policy worker started. Extended-length copy paths
fixed the issue; the failed folder is retained and the replacement run is
active. The unproductive rear-unload worker was archived at
model32 and stopped to free this GPU slot; the original pair, rear-support
and stance-anchor jobs continue.

### Contact consistency and the missing floor connection

A new static calculation keeps the final reference servo commands fixed
and relates each traction-corner normal force to 12,000 N/m compression.
It finds near-flat algebraic equilibria under both 85% and 100% strength:
body displacement under 0.8 mm per axis, maximum joint adjustment 0.132
degrees, and peak torque 50.9% of the weak cap. Tread compression is
0.10–0.37 mm; no other collision is reported. The weak case reached its
iteration limit but meets the stated algebraic residuals. This is a
four-corner approximation, not PhysX manifold reproduction or a new
accepted runtime reference. It suggests that compression alone does not
explain the 11–13 mm landing gaps observed in the live policy.
[Final-contact calculation](../reviews/ankle-v23-curriculum/20260915/rear-compliant-equilibrium.json).

The same calculation at the newer retreat/lift route's waypoint7 finds a
100%-strength algebraic equilibrium with three motors saturated, while
the 85% attempt fails its residual and collision constraints. Both reach
the optimizer's iteration limit. Failure to find a weak equilibrium is
not proof that none exists, but this supports the concern about little
actuator reserve in that route.
[Lift-contact calculation](../reviews/ankle-v23-curriculum/20260915/rear-lift-compliant-equilibrium.json).

Further four-foot load-shift calculations after the earlier FL replant
again reduce RR load to 4 N, but fail at 1 N and effectively zero load,
even with a larger unexecuted yaw allowance. No rejected pose was installed.
At the now-observed rear-reset configuration, an ideal-contact screen
instead permits FL unloading at 78.7% of the weak cap; RL would require
145.4%, and FR has no feasible allocation. This motivates investigating
the floor connection backwards through FL movement. A connected reverse
path is not established. A subsequent FL-lift calculation admits 185 and
190 mm whole-foot minimum heights with peak static torque 47.3% of the
weak cap, while RL/FR remain flat on the first tread and RR bears on its
original ground corner with heel gap at most 40 mm. The connecting body
shift from the reset is unproven. Ten subsequent retraction samples above
the riser are accepted, with peak torque rising to 58.3%. The next sample
fails; the foot remains over the tread, so no descent to the floor has
been established. No new pose or path was installed in a runtime package.
[FL lift](../reviews/ankle-v23-curriculum/20260915/rear-reset-front-predecessor.json)
and [retraction samples](../reviews/ankle-v23-curriculum/20260915/front-predecessor-retraction.json).
The failed retraction intersects the RL shank/foot. Allowing swing tilt up
to60degrees does not resolve it. Keeping the swing sole center between
Y=0.09 and0.16m avoids the collision in the next search, but the sixth
sample still fails pose/equilibrium constraints before reaching the edge.
Neither failed candidate was used as a motion reference.
[Rotation attempt](../reviews/ankle-v23-curriculum/20260915/front-predecessor-tilt-retraction.json)
and [lateral-corridor attempt](../reviews/ankle-v23-curriculum/20260915/front-predecessor-clear-retraction.json).
[Replant/load shift](../reviews/ankle-v23-curriculum/20260915/replanted-pair-load-shift.json)
and [predecessor choices](../reviews/ankle-v23-curriculum/20260915/grounded-rear-predecessor-loads.json).

Primary research supports treating terrain-safe placement separately from
lift height ([Shi et al.](https://arxiv.org/html/2310.04675v2)) and separating
stance and swing control objectives ([CAJun](https://proceedings.mlr.press/v229/yang23b.html)).
The new formula and controller conclusions here are our hypotheses, not
reproductions of those systems. Only requested training, recording and
offline arithmetic ran; no standalone test suite or hardware checks ran.

## Floor-to-rear connection and reward comparison, September 18 at 01:12 PDT

The previous continuation made progress by launching the two rear-transfer
comparisons and identifying support-foot drift. Both workers remain live.
The two 1,024-environment arms have identical recorded first-four joint
states at steps 20 and 100, before the first PPO update. After learning
begins, their behavior diverges. At matched step 600, selected rear-foot
heights reach 160.9 mm with the stance-anchor reward and 144.3 mm in its
baseline. Both still show the supporting foot against the next riser, and
neither has a qualified rear landing or stage. These selected leaders are
not a population success-rate comparison. At step 1,160, the reward arm
still has five current rear feet unloaded above 30 mm, but this is an early
training observation, not a reliable ascent result.
[Aligned training observations](../reviews/ankle-v23-curriculum/20260915/stance-anchor-training-comparison.json).

The missing floor-start connection was investigated separately. Earlier
calculations kept every foot flat after the recorded front-pair hold; the
new four-contact search permits rocking on fixed supporting edges. Two
grounded endpoints recover rear-hip travel with bounded static torques.
Neither permits an immediate rear-leg unload at its fixed pose. A further
load-shift search admits an 8 N and then 4 N rear-right load, but its 1 N
and effectively zero-load attempts fail the existing constraints. No
recovery endpoint or failed load-shift pose was installed as a reset.
[Grounded recovery](../reviews/ankle-v23-curriculum/20260915/frozen-pair-edge-recovery.json)
and [load transfer](../reviews/ankle-v23-curriculum/20260915/frozen-pair-edge-load-shift.json).

The recovered pose does permit unloading either front leg: independent
static force allocation uses 59.3% of the weak cap for front-left or 74.5%
for front-right. This motivates an extra same-tread repositioning step.
Eight accepted right-front static samples lift the full foot 5–10 mm,
move its center from X=0.4772 m to 0.4200 m, and lower it to the first
tread with at most four degrees of tilt. Required three-support peak
utilization is at most 68.2%. The right-front final pose still requires
218.9% of the weak cap to unload the rear-right even under an optimistic
four-corner landing-force assumption; that is a rejected next move, not
a stair-climbing solution. The left-front alternative has six accepted
static lift/advance samples reaching X=0.4500 m. These samples do not
prove a continuous or dynamic path from the frozen recording.
[Unloading choices](../reviews/ankle-v23-curriculum/20260915/pair-reposition-unloading.json),
[right-front reposition](../reviews/ankle-v23-curriculum/20260915/pair-front-right-reposition.json),
[right-front landing](../reviews/ankle-v23-curriculum/20260915/pair-front-right-landing.json),
and [left-front reposition](../reviews/ankle-v23-curriculum/20260915/pair-front-reposition-refined.json).

The left-front lowering and planting calculation subsequently also succeeds
at X=0.4500 m, with three-support peak utilization 57.6% and at most four
degrees of foot tilt. This sampled endpoint makes the torso pitch about
-39.3 degrees instead of -47.1 degrees at the recovery seed. Neither rear
leg can unload at that fixed pose under the static setpoint constraints,
even with optimistic contact corners on the newly planted foot. The next
useful geometric question is a further body/load shift at this changed
footprint, rather than assuming a replant alone solves the transfer.
[Left-front landing](../reviews/ankle-v23-curriculum/20260915/pair-front-left-landing.json).

This changes the reward-design emphasis. Low torque or recovered joint
range alone can favor a stationary pose that still cannot release the next
leg. A useful preparatory reward should favor a *supported next transfer*,
and must allow intermediate same-tread repositioning rather than demanding
that every step immediately gains a stair. Repeated bonuses for replanting
the same tread would invite cycles; any such shaping should remain bounded
and tied to improved transfer readiness. The existing active rewards were
not changed during this analysis.

### Contact-model limitation identified from source and primary documentation

The retained runtime binds compliant foot contacts with stiffness 12,000
N/m and damping 120 Ns/m. [PhysX's contact documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#compliant-contacts)
describes normal forces from an implicit spring-damper at each contact
point, and friction computed per contact patch. The current static planner
instead allocates forces at ideal, zero-penetration corners. Its feasible
forces are therefore not a self-consistent compliant-contact equilibrium.

For scale, 3 N on one ideal spring requires 0.25 mm compression; 17 N
requires 1.42 mm. A zero-velocity four-corner estimate on 71 loaded recorded
foot samples is 0.75–1.47 times the recorded upward force (median 1.18).
Ten supporting-foot side-contact samples imply 12.2–16.0 kN/m when opposing
normal force is divided by reconstructed overlap. This is approximate
arithmetic: actual manifold points, damping, other faces and substep timing
were not recovered, so it is not calibration or a validation of the model.
It motivates refining the planner's contact-load consistency, not changing
simulation stiffness to make a trajectory succeed.
[Contact arithmetic](../reviews/ankle-v23-curriculum/20260915/contact-compliance-arithmetic.json).

At 01:12 PDT, all four workers were confirmed live. Their current snapshots
were pair-transfer step 7,160; rear-unload 4,040; rear-support 2,140; and
stance-anchor 1,500. Reading the actual plant-event logs found zero new
rear-foot tread credits in all four, and zero stage/full-climb events.
All four existing full-height RL jobs continue. Their new front-pair event
checkpoints were archived. No new runtime package, reset, hardware check,
standalone test, evaluation batch or video rollout was introduced here.

## Supporting foothold drift, September 18 at 00:48 PDT

The reduced-support-residual run progressed beyond its original rear lift,
then stalled around 140–150 mm. Full-foot FK/SAT reconstruction now finds
its supporting front-left plastic sole against the *second* riser. At step
100 the selected samples had 8.9–10.0 mm clearance; by steps 300 and 400,
all five selected samples had 0.06–0.10 mm overlap. Their nominal references
retain about 12.5 mm clearance, excluding the original 2 mm XY reset jitter.
Recorded negative-X normal contact and lost upward support are consistent
with this geometry. [Support collision analysis](../reviews/ankle-v23-curriculum/20260915/support-foot-riser-contacts.json).

At step 400, the selected supporting setpoints were within only 0.04–0.08
degrees RMS of the nominal reference, yet estimated PD demand reached 2.68
times a motor cap. At those deformed recorded poses, the static calculation
either failed the setpoint reserve or required 117–120% of the actual caps.
Thus some early poses allow different feasible load allocations, while the
later drift can carry the robot outside that feasible region. Reducing
residuals alone has not solved the transfer.

A matched 1,024-environment reward comparison, `stairs-stance-anchor-s3281`,
has now started with the same actor, seed, controller, rear reference,
resets and all physical/success constraints as rear-support. Its only
behavioral design change is an additional bounded potential during guided
rear transfer:

`Phi_anchor = 2 * other_three_support * exp(-0.5 * mean_support_legs(||sole_xy - reference_sole_xy||^2 / 0.010^2)) * (1 - handoff)`.

The reference includes each environment's original reset offset. At full
support, one supporting foot drifting 12 mm lowers this potential from 2
to 1.573. It uses the existing discount-consistent potential difference and
terminal zeroing, so holding a pose cannot repeatedly earn a positive bonus.
The existing handoff fades this encouragement. This is a soft reward for
retaining usable footholds, not a locked foot or a changed actuator command.
[Reward contract and arithmetic](../reviews/ankle-v23-curriculum/20260915/stance-anchor-contract.json).

The original rear-retreat arm was archived at saved update 16 and stopped
at training iteration 18 to release compute. The current four arms are
pair-transfer (256), rear-unload (256), rear-support (1,024) and stance-anchor
(1,024). The reward comparison has not yet established an outcome. At the
latest rear-support step 620, five current rear samples remained unloaded
above 30 mm; maximum whole-foot height was 144 mm. Whole-robot first-tread
and full-climb counts remain zero. The first saved rear-support actor and
partial-progress logs are archived separately from the older front-pair
video. No active package was edited after launch.

## Supporting motor margin, September 18 at 00:42 PDT

The next comparison, `stairs-rear-support-s3281`, is training with 1,024
environments. It keeps the rear-unload reference, reward, grounded resets,
actor transfer and all physical limits. Only guided rear supporting-leg
residuals are scaled to 0.10; the selected swing leg retains full residual
authority. Free control and the floor/front-foot controller are unchanged.
This tests whether independently moving the support feet disrupts their
required load distribution. Its larger population prevents a clean matched
efficacy comparison with the 256-environment unloading baseline.

The first evidence is further rear lift: at step 200, the maximum lowest
point of the entire rear foot was 136.7 mm above the floor, with 15 of 514
current rear-curriculum samples both above 30 mm and below 1 N upward load.
No rear landing, whole-robot first-tread hold or full climb was established.
These are changing-policy, reference-assisted samples starting with three
feet already on the first tread; the 180 mm staircase is unchanged.

The preceding static unloading endpoint requires 96.48% of the weakest
motor cap. A wider pose search reduced this to 95.44% with fixed supporting
edges, or 94.55% when allowing a 10 mm change in their second corner. Those
are unconnected endpoints, not demonstrated transitions or new resets.
[Pose search](../reviews/ankle-v23-curriculum/20260915/rear-robust-support-pose.json).

Thirty recorded rear-unload samples at guide index 7 all had at least one
supporting joint whose estimated PD demand exceeded even the strongest
continuous cap. Applied torque remained capped. Replacing the target with
the nominal reference at the same measured state also exceeded a cap in all
30 samples; target drift alone therefore does not explain the problem.
The estimate uses post-step positions and velocities, not synchronized
physics substeps. [Saturation analysis](../reviews/ankle-v23-curriculum/20260915/rear-servo-saturation-analysis.json).

Additional force calculations use the new run's recorded root quaternion,
joint positions and individual motor strengths. At step 20, eight selected
rear states admitted a hypothetical unloaded static equilibrium within
their torque limits (89.7–96.9% peak utilization), although their current
servo requests reached 155.6%. This suggests room for better support-force
distribution; it does not show that a stable connecting motion exists.
At step 100, two of four selected states failed the strict 0.04-degree
setpoint-reserve constraint. A diagnostic relaxed-envelope calculation
identified only about 0.04 degrees of rear-ankle reserve shortfall; the
relaxed envelope is never installed or counted as feasible. Contact corners
in these calculations are inferred from geometry, not measured loads.
[Recorded-state calculations](../reviews/ankle-v23-curriculum/20260915/rear-measured-equilibrium.json).

A read-only USD/URDF/runtime-metadata comparison found maximum COM and
mass discrepancies of 1.6e-9 m and 1.4e-8 kg, respectively. This rules out a
large inertial transcription mismatch in the inspected asset; it does not
validate contact dynamics. [Inertial comparison](../reviews/ankle-v23-curriculum/20260915/stair-asset-inertial-comparison.json).

The reward sequence remains: transfer load to the supporting feet; free
the plastic sole from the riser; raise the whole foot; move it over the
edge; then reward a loaded, low-slip landing. A generic height bonus or a
large penalty on all motor effort would oppose this near-limit maneuver.
The next candidate to size from recorded data is a bounded support-gated
potential for reducing *requested torque beyond the physical cap*, while
retaining full credit for high but feasible effort. No such extra torque
reward has yet been added to the active runs.

[CAJun](https://proceedings.mlr.press/v229/yang23b.html) separates high-level
RL commands from low-level control of stance and swing legs. That motivates
the present separation, but the one-tenth residual gain is our hypothesis,
not a reproduced result from that paper. [CaT](https://arxiv.org/abs/2403.18765)
uses stochastic terminations for constraints; it was considered but is not
implemented here, and no new termination penalty was introduced.

Clearance10, smooth-handoff and rear-residual were intentionally stopped
after archiving saved updates 48, 32 and 16. Pair-transfer, rear-retreat,
rear-unload and rear-support remain active. Their historical front-pair
checkpoints and full-height video remain preserved. No standalone tests,
evaluation batch, new video rollout or hardware check was run in this
continuation; authorized training and offline calculations were performed.

## Overlapping unloading and retreat, September 18 at 00:11 PDT

At rear-retreat step 340, all 126 current rear-curriculum environments
were still at index 7. Median plastic-riser clearance was 1.02 mm and
median rear upward load was 7.18 N; the maximum clearance was 3.10 mm,
below the 5 mm advance gate. Median lowest-foot penetration was 0.46 mm.
These observations suggest that asking a still-loaded foot to move backward
along the floor may be a poor sequence. They do not prove a friction or
contact-stiffness cause. [Recorded snapshot](../reviews/ankle-v23-curriculum/20260915/rear-retreat-initial-hold.json).

A matched comparison, `stairs-rear-unload-s3281` (256 environments), now changes only the
first seven moving rear targets: the foot gains 0.5–5 mm of vertical
clearance as it retreats 0.5–10 mm. The reset itself remains the original
grounded pose; no airborne state is inserted. Each static sample retains
the three original supporting contacts and bounded force allocation.
The last unloading sample uses 96.48% of the weak continuous torque cap,
versus 98.27% for the floor-contact retreat endpoint. This modest static
margin improvement is not a dynamic success result.

Reward terms, measured advance gates, actor checkpoint, full residual
authority, subsequent reference, physics and success conditions match
the retreat comparison. Additional traces record actual servo targets,
joint velocities and all normal-force components for four environments.
The older aggregate-load arm was archived at update 40 and intentionally
stopped for this comparison; the original rear-retreat run continues.
The initial launch failed before training because a generated contract
string used Windows encoding. A second launch reached Windows' 260-character
limit while archiving provenance. The corrected launch uses an ASCII string
and shorter run folder, with the same experiment and both failed-start logs
retained. Neither failed launch exercised a policy.
[Unloading contract](../reviews/ankle-v23-curriculum/20260915/rear-unload-contract.json).

Before the first PPO update, matching step-100 traces for two rear
environments show loads falling from 7.00/10.06 N in floor retreat to
2.52/4.58 N with overlapping lift. In the latter run 10 of 125 rear
environments had passed the retreat gate. A transient 10.9 mm whole-foot
lift at step 60 was not maintained: by step 100 maximum clearance was
4.71 mm and no current rear sample combined force below 1 N with more
than 5 mm of clearance. This is promising mechanical progress from the
changed reference, not evidence of a learned reliable ascent.
[Early comparison](../reviews/ankle-v23-curriculum/20260915/rear-unload-initial-analysis-step100.json).

## Rear-foot collision clearance, September 18

The static reference left only 0.2 mm between the plastic sole and the first
riser. New reconstruction of recorded poses finds the plastic sole against
the riser while the rubber pad remains about 6 mm behind it. The retired
stumble arm independently logged negative-X rear contact forces up to
2.64 N with zero upward force. Selection was biased toward large side
impacts, so these samples establish occurrence, not frequency.
[Geometry](../reviews/ankle-v23-curriculum/20260915/rear-collision-geometry-analysis.json)
and [force evidence](../reviews/ankle-v23-curriculum/20260915/rear-riser-force-analysis.json).

The new rear-retreat experiment adds a 122-pose static reference for the
grounded rear curriculum: pull the plastic sole 10 mm behind the riser,
lift the whole foot to 195 mm, cross, land with at most four degrees of
tilt, then lower supporting heels. A fully flat landing attempt failed and
was excluded. The accepted final pose has every traction corner within
5.6 mm of the first tread, with an independent bounded four-foot static
load allocation. These are static samples, not an executed transfer.
The original reset and its initial servo target remain exact copies.
[Experiment contract](../reviews/ankle-v23-curriculum/20260915/rear-retreat-contract.json).

The added potential is
`2 * other_three_support * max(clip(plastic_riser_gap / 0.010), clip((whole_foot_min_z - 0.180) / 0.010))`,
with each clip bounded to [0,1]. Its clearance factor remains one after
the existing edge-crossing latch or actual first-tread rear credit. It is
restricted to the grounded rear curriculum and uses the existing
gamma-matched potential difference and terminal zeroing. At full support,
0, 5 and 10 mm of retreat correspond to potentials 0, 1 and 2. Holding
the saturated potential contributes -0.000625 per step, so merely staying
clear cannot earn a repeated positive reward. This leaves the existing
load transfer, once-per-height lift credits, planted-foot, stage and full
climb rewards in place.

Advancement additionally waits for at least 5 mm of measured plastic-sole
clearance during the retreat/lift segment and at least 185 mm of whole-foot
height during crossing. Actual rear plant credit is required before the
support-settling segment. These guide gates address the old 25 mm sole-center
tolerance, which could advance a reference despite outer-sole contact.
Physical success criteria, actuator limits and stair dimensions are unchanged.
The route and its shaping change together; this is a combined experiment.

The run was launched from the same reproduced front-pair actor with full
residual authority. The smaller-residual comparison remains separate; its
early traces reduce target variation without demonstrating rear progress.
No standalone automated checks, evaluation batch or hardware check was run.

## Rear-reference tracking and residual scale, 23:31 PDT

The serialized resets match the first guide pose and servo targets to
floating-point precision. There is no reset/reference mismatch to fix.
The rear reset's desired static drive reaches **0.734835 Nm**, leaving
only **0.015373 Nm**, or about **2.05%**, below the 0.750209 Nm weak-motor
cap. [Reset comparison](../reviews/ankle-v23-curriculum/20260915/reset-reference-match.json).

In selected stable-index traces, actual joints differ from the reference
by about 1.6–2.8 degrees RMS, while reconstructed pre-control servo targets
drift roughly 0.4–1.4 degrees RMS from their reference. The nominal target
rate is unclipped in these samples, allowing that target reconstruction.
One degree of target change corresponds to 0.524 Nm of proportional torque
at a fixed joint with Kp=30. This is a sensitivity calculation, not a
measurement of torque change in the moving robot.
[Tracking decomposition](../reviews/ankle-v23-curriculum/20260915/rear-reference-tracking-analysis.json).

Nine selected 10-second windows show 0.50–0.79 degrees RMS temporal target
fluctuation, equivalent to 0.26–0.42 Nm at a fixed joint. Learned action
means, exploration and changing geometry are not separated by these logs.
The magnitude motivates an isolated lower residual-scale comparison; it
does not establish that action noise causes the entire stall.
[Window arithmetic](../reviews/ankle-v23-curriculum/20260915/rear-target-variation-analysis.json).

Lowering the static minimum support load from 2.5 N to 1.1 N does **not**
improve the optimum torque ratio at rear waypoints 0, 4, 8 or 12. The ratios
remain 0.97951, 0.94646, 0.92505 and 0.88689 respectively. That constraint is
not the limiting factor in these calculations, so contact requirements have
not been weakened. [Force-margin calculation](../reviews/ankle-v23-curriculum/20260915/rear-reference-force-margin.json).

The `stairs-rear-residual-256-s3281` run started at 23:34 PDT. It clones
pair transfer and reduces only
the existing guided rear variant's residual corrections to one-tenth:
XYZ 2.5 mm/s and pitch 0.035 rad/s. Nominal reference motion remains; the
front-floor route, unguided action scale, rewards, resets and physical
limits are unchanged. The same actor checkpoint is retained. Lower gain
may also limit useful corrections, so success must come from actual
training progress. Additional target and joint-velocity logs will separate
future servo errors more directly. Its startup record confirms exact actor
tensor transfer. The old rear-bearing arm was archived at update 40 and
stopped to allocate the comparison; the pair-transfer control remains live.
[Comparison contract](../reviews/ankle-v23-curriculum/20260915/rear-residual-contract.json).

## Retain the demonstrated actor while changing transfer shaping, 23:18 PDT

`stairs-pair-transfer-256-s3281` is now training with the exact actor that
reproduced the frozen front-pair hold. The startup record confirms equality
of all 14 actor tensors, including its GRU, complete action head and
observation normalization. The previous dormant transfer implementation
would have reset the action head; this new isolated package explicitly
preserves it and requires the archived checkpoint's known SHA256.

The new run combines clearance10, aggregate raised-foot load, bearing
support and gradual handoff. Critic and optimizer start fresh for the changed
reward. Observation dimensions and ordering remain 567/583, while support
feature values and post-completion phase values change with those components.
The reset distribution, physical limits and strict success gates are
unchanged. No new rear-headroom bonus is included. This combination seeks
whole-task progress and cannot attribute any improvement to one component.
[Transfer contract](../reviews/ankle-v23-curriculum/20260915/pair-transfer-contract.json)
· [Training startup evidence](../reviews/ankle-v23-curriculum/20260915/stairs-pair-transfer-256-s3281/train.metrics.actor_transfer.json).

The [two-stage stair-learning paper](https://arxiv.org/abs/2602.14473)
reports reusing a learned stair policy for a later U-shaped stair task on
a Go2. Its abstract supports considering policy reuse; it does not validate
our robot, reward combination or training outcome. Our task keeps the same
full-height stairs throughout.

The offline force model also now accepts actual outer plastic-sole contact
corners. The 1 mm rubber protrusion and 3 mm inset mean the plastic corner
can become the first contact beyond approximately 18.4 degrees of simple
edge rocking. That is geometry arithmetic, not a required gait angle.
Eight searches with front plastic pivots and an original rear-floor pivot
found no accepted takeoff endpoint. Physical collision, motor and joint
constraints were retained. These calculations did not alter any live
training package or install an unsupported trajectory.
[Failed outer-sole searches](../reviews/ankle-v23-curriculum/20260915/outer-sole-takeoff.json).

## Recorded rear-joint headroom, 23:04 PDT

The frozen clearance10 pose at 40 seconds reconstructs from its recorded
joints, root and soles. Both rear hips are approximately −60.001 degrees.
An unloaded rear leg needs about −0.211 Nm at its hip just to balance its
own gravity in this pose. With Kp=30, that requires a −60.405-degree target;
the runtime target bound is −59.7 degrees, and the conservative static
screen reserves another 0.04 degree. The static target shortfall is therefore
approximately 0.745 degree. It is a control-headroom constraint even though
the requested gravity torque is below the motor's nominal torque cap.

The four-foot force screen succeeds at 0.356–0.534 of the weak motor cap,
depending on the assumed contact corners. Neither fixed-pose rear-unload
screen succeeds. The normal-force-only XY support hull still puts the COM
18.9 mm inside when RL is omitted, showing why that geometric score alone
does not establish a feasible lift. Contact compliance, hard-stop reactions
and dynamics are omitted; no global impossibility result is claimed.
[Calculation and recorded geometry](../reviews/ankle-v23-curriculum/20260915/frozen-front-pair-force-analysis.json).

Reward implication: reward progress toward an **actuatable supported
takeoff**, then unloading, whole-foot clearance and planted contact.
Supporting-foot force, swing-joint target headroom and motor demand must
agree. A candidate future headroom potential should be bounded, gated by
the other three measured supports, and paid through the existing potential
difference; it should not reward repeatedly moving away from and back to
a joint limit. No headroom term has been installed yet. The aggregate-load
and smooth-handoff trials already running address separate diagnosed
constraints without changing physical limits.

Four subsequent grounded IK attempts kept the recorded sole XY positions
and tread levels while seeking rear hips inside −58.5 or −55 degrees.
None passed the original position/normal/contact acceptance criteria;
their relaxed force screens also did not establish either rear unload.
This reinforces that increasing hip headroom alone is insufficient evidence
of a usable route. The search is local and does not rule out rocking,
replanting a foot, or other body poses.
[Rejected local endpoints](../reviews/ankle-v23-curriculum/20260915/frozen-pair-hip-recovery.json).

At the latest read, rear-bearing step 3640 and aggregate-load step 1580
both have median rear-guide index 7 and swing-position errors near 29–30 mm.
Their median remaining RR loads are 8.0 and 9.1 N respectively. Exposure
differs, so these are descriptive snapshots, but there is no demonstrated
rear-transfer improvement yet. A higher bearing-quality score alone is not
progress. [Saved comparison](../reviews/ankle-v23-curriculum/20260915/rear-transfer-latest-comparison.json).

## Rear-load conflict and reward correction, 22:33 PDT

Reconstruction of three existing rear-curriculum traces finds the body
28–87 mm behind its current guide target, with the rear-right foot still
carrying 5.2–14.5 N. These are three recorded environments, not a population
estimate. The desired poses can unload RR under the bounded static model;
the actual-pose force screens cannot. Contact-plane errors in the latter
prevent treating those screens as a global infeasibility result.
[Trace analysis](../reviews/ankle-v23-curriculum/20260915/rear-curriculum-stall-analysis.json).

There is a concrete conflict in the transfer potential. Its old raised-load
factor is the **minimum** of each raised foot's force divided by 20% of body
weight, saturated per foot. Reaching its maximum therefore favors at least
6.757 N on every raised foot. At guide pose 8, a feasible allocation is
approximately FL 3.00 N, RL 16.99 N, FR 13.79 N, RR 0 N. That uses 0.9382
of the 85%-strength motor cap. Forcing every supporting foot to at least
6.757 N instead needs 1.1484 of that cap. Poses 6 and 10 show the same
conflict: 1.1512 and 1.1342. These are static calculations at the particular
reference poses, not proof that no other pose can carry those loads.
[Bounded-force calculation](../reviews/ankle-v23-curriculum/20260915/rear-raised-load-reward-conflict.json).

The new `stairs-aggregate-load-256-s3281` arm started at **22:32:58**. It
clones the rear-bearing arm and changes only this load aggregation:

`raised_load = clip(sum(valid raised-foot Fz) / (0.2*m*g*raised_count), 0, 1)`

Summation precedes saturation, so the feasible uneven distribution can
receive full group-load credit. The candidate's readiness still uses the
weakest of the other three individual support contacts. Full-foot landing,
actual upward contact, low slip, joint limits and full-climb success gates
are unchanged. The term remains inside the same gamma-matched potential
difference with terminal zeroing. No extra recurring bonus is added.
[Reward arithmetic](../reviews/ankle-v23-curriculum/20260915/aggregate-load-reward-arithmetic.json).

Early logs show the new group factor reaches 1 with uneven loads; the old
minimum would give approximately 0.30–0.49 for those same examples. This
checks what the reward is measuring, not whether the policy climbs better.
The larger post-pair edge-only run was archived at update 32 and stopped
to allocate this comparison. Exact IK, rear-bearing, stumble and clearance10
remain active alongside it. No active package was edited.

## Command handoff remains a separate hypothesis

Two recorded front-pair events in one earlier training environment were
followed by resets within 1.8 and 37.3 seconds. The current completion switch
changes the action mapping abruptly: XYZ scale rises from 0.025 to 0.07 m/s,
reference-centred pitch-rate residuals switch to absolute-pitch control,
and nominal reference velocity ends. The final guide fraction does not
explicitly expose completion in the actor's phase feature.

At two recorded pre-handoff poses, applying the two existing command
formulas to the same filtered actions yields maximum per-foot XYZ changes
of 23.3 and 34.7 mm/s, and pitch-rate changes of 0.572 and 0.381 rad/s.
This quantifies a control discontinuity; it does not establish the cause
of either reset. An observable, gradual handoff is now installed in the
isolated `stairs-smooth-handoff-256-s3281` trial, started at **22:58:01**.
It uses clearance10's reference and unchanged rewards. The two existing
bounded command formulas and nominal IK damping centre are blended over
two seconds, while the actor phase feature exposes blend progress. Physical
joint, speed and effort limits remain unchanged. Its effect is not yet known.
[Existing trace observations](../reviews/ankle-v23-curriculum/20260915/front-pair-handoff-observations.json)
· [Command arithmetic](../reviews/ankle-v23-curriculum/20260915/front-pair-handoff-command-arithmetic.json).

The [PMTG paper](https://proceedings.mlr.press/v87/iscen18a.html) studies
policies coupled to trajectory generators to supply prior motion structure
and memory. Making our generator transition observable is an application
of that idea, not a result established by the paper for this robot; the
two-second duration is an experiment-specific choice.

## Measured bottlenecks and next trial, 22:25 PDT

The side-impact cue is now supported by direct training logs. At step 1500,
the eight highest-severity examples have FL contact force −2.93 to −3.38 N
along world X and zero along Z, at the second-front crossing. The local
Isaac Lab sensor source defines this as net **normal** force, with friction
separate. It is consistent with a foot pushing into the vertical riser.
The sample is selected by severity and does not estimate a success rate.
[Saved evidence](../reviews/ankle-v23-curriculum/20260915/second-front-horizontal-contact-evidence.json).

The reward implication is to measure **the lowest point of the whole foot**,
including the wider plastic sole, against the riser. Sole-centre height or
traction-pad height alone can miss the contact. Offline analysis found a
10 mm whole-foot-clearance sequence through crossing and flat landing.
The new `stairs-clearance10-256-s3281` trial changes that reference only,
retaining exact-IK rewards and actual-contact success gates. Its 53 static
second-front samples stay within 0.663753 of the weak motor cap; connecting
motion is still unproven. This comparison addresses whether the current
reward is guiding the policy along a route with too little clearance.

Collision cost also needs scale discipline. At 3 N horizontal and zero
vertical load, the current term costs 0.25 per second: 30 seconds costs
7.5 before discounting, almost the +8 plant reward. A long stall can cost
more than the −12 fall penalty. That creates a possible incentive to end
an episode early; it is not evidence that the current policy has learned
this behavior. Keep the no-collision-cost baseline, avoid increasing this
weight blindly, and consider a bounded per-episode collision budget or
clearance potential if the termination tradeoff appears in the logs.
Neither alternative has been installed.

The separate `stairs-edge-curriculum-256-s3281` arm extends bearing-contact
and load-transfer shaping to the existing grounded rear-leg curriculum.
The original edge-support arm only enabled it after a floor-start front
pair hold. At step 960, 18/132 rear examples fail the support gate under
the new definition, yet median guide index remains 7 and swing error is
31.4 mm. Changing the support score removes a shaping conflict without
demonstrating physical improvement. Full-foot landing credit is unchanged.
[Training snapshots](../reviews/ankle-v23-curriculum/20260915/rear-bearing-training-snapshot.json).

The remaining objectives are actual supported clearance, a stable planted
foot, and load transfer that permits another foot to lift. Preserve the
gamma-matched potential difference and actual-contact milestone rewards;
do not award ascent for elevated body pose, hovering feet or wall contact.
All five running arms remain simulation teachers with privileged features.

## Side-impact shaping, 22:00 PDT

The frozen second-front stall has a reconstructed plastic-sole/riser overlap
of about 0.178 mm while the narrower traction pad is still behind the riser.
Upward foot force alone reports zero at this foot. This suggests that a
horizontal-contact cue could help the policy clear the edge. It is a
geometry-and-torque diagnosis; direct horizontal forces were not recorded
in that clip.

The primary legged_gym implementation includes a foot-stumble cue comparing
horizontal contact force to five times absolute vertical force.
[Source implementation](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py).
The new isolated `stairs-stumble-256-s3281` trial uses a 1 N deadband, a
bounded severity ramp over the next 4 N, and maximum cost 0.5 per second:

`severity = clamp((norm(Fxy) - 5*abs(Fz) - 1 N) / 4 N, 0, 1)`

The worst foot determines the cost, multiplied by the control timestep.
This term is a collision penalty, not potential-based shaping. Other rewards,
controller, references, robot limits and success gates match the exact-IK
baseline. The separate edge-support modification is not included. Population
is 256, so comparisons to the 1024 baseline require accounting for exposure.
It started at 21:54 and logs full force vectors for the most affected feet.
See [arithmetic](../reviews/ankle-v23-curriculum/20260915/stumble-reward-arithmetic.json).

## Loaded edge support after the front-pair hold, 21:38 PDT

The new `stairs-edge-support-1024-s3281` arm started at 21:26. It clones the
1024-environment exact-IK arm with a change only after the actual floor-start
front-pair hold. Temporary bearing contact and full-foot landing have separate
roles in the reward. Landing credit still requires the original flat patch,
load, low slip and contact duration.

Offline analysis found one rear-right airborne pose with a feasible bounded
force allocation and unchanged collision constraints. Its supporting feet
carry approximately 3.00, 17.93 and 12.85 N, with heel gaps up to 28.54 mm.
The old minimum support quality is 0.2716 despite a peak motor ratio 0.951936.
This is evidence of a shaping conflict, not a connected climbing solution:
floor takeoff searches remain unsuccessful.

After front-pair completion, the new bearing term uses the nearest corner
gap with a 4 mm Gaussian scale. It requires measured upward force above 1 N,
one surface, minimum gap at least −8 mm, maximum gap at most 40 mm, and a
nearest corner within 8 mm. Force saturation remains at 3 N. The better of
flat and bearing quality gates readiness and supported lifting. Raised-foot
load transfer also recognises this bearing contact, while the landing term
keeps its original full-foot quality. Support-location estimates weight
near-plane corners rather than using the elevated sole centre; individual
corner forces are not measured by this feature.

On the static sample the new minimum bearing quality is approximately 1.
This is a formula calculation, not demonstrated RL improvement. The reward
remains a bounded, gamma-matched potential difference with terminal zeroing.
The robot, controller, resets, references and landing/full-climb gates are
unchanged. See [arithmetic](../reviews/ankle-v23-curriculum/20260915/edge-support-reward-arithmetic.json)
and [current results](rl-ankle-v23-ik-progress.md).

## Additional research and local implications, 21:47 PDT

Shi et al. combine horizontal foot motion with terrain-conditioned swing
height and discourage unnecessarily high lifting. Their reported experiments
also cover multiple travel headings. My inference for this robot is to keep
clearance tied to the actual obstacle and reward progress toward a landing;
additional foot height alone does not address the horizontal errors observed
at the second-front stall. Their results do not establish feasibility for
this robot's joint limits or 180 mm stairs.
[Terrain-Aware Quadrupedal Locomotion](https://arxiv.org/html/2310.04675v2).

BeamDojo uses polygonal-foot placement rewards and separate critics for
dense locomotion and sparse foothold rewards. Separating value estimation
is a possible later experiment if the contact milestones remain overwhelmed
by ordinary tracking signals. Its terrain-relaxation stage is not part of
these full-height-only experiments. No second critic has been installed.
[BeamDojo, RSS 2025](https://www.roboticsproceedings.org/rss21/p068.html).

The local force calculations also suggest trying a front foot on tread two
before rear takeoff. At the existing two-front stance, unloading FL/FR has
minimum peak torque ratios approximately 0.518/0.692, while the previous rear
unloading attempts were much harder. A front-right lift has accepted static
samples through 365 mm full-foot clearance, but second-tread crossing and
landing remain unresolved. The first lift sample moves backward substantially;
the samples are not a connected route. This also exposes a possible tradeoff
in the lagging-foot transfer term: rewarding load on every raised foot may
discourage temporarily advancing a front foot farther. The current term has
not been changed on this new hypothesis.

Source inspection found a further control transition to assess: after guide
completion, the same normalized XYZ action changes scale from 25 to 70 mm/s
and nominal joint assistance ends. The current guide phase feature does not
explicitly encode that completion flag. A gradual, observable handoff is a
possible separate controller experiment. At the time of this historical
inspection it was not implemented; the 22:58 trial described above now
addresses it. Inspection alone does not prove it causes post-pair falls.

## First-foot result and the next reward, 19:46 PDT

Latest result, 21:16 PDT: the posture-damping controller has produced actual
simultaneous front-foot holds on tread one. The first occurred at step 847;
36 were logged by step 1760. No rear transfer or whole-robot climb is solved.
This supports continuing the physical route rather than counting a stronger
tracking score as success.

The separate 21:05 reward comparison changes only the swing-foot error scale
inside guided similarity from 50 mm to 20 mm. At 25 mm error with unit support
and zero other errors, the foot contribution changes from approximately
1.879 to 1.353, and its local slope magnitude increases from 9.39 to 42.29
per metre. The potential remains bounded by its original maximum. These are
function calculations, not measured policy improvements. See
[foot-tracking arithmetic](../reviews/ankle-v23-curriculum/20260915/foot-tracking-potential-arithmetic.json).

The earlier hypothesis concerned temporary rocking contacts. Original support
quality uses the largest corner gap, so at a 50 mm gap its geometric factor
is only `exp(-4) = 0.0183` even when a toe is loaded. Raised-foot transfer
also requires the strict flat-plant flag. That can conflict with a physically
useful toe pivot. The later static candidate and resulting isolated reward
trial are described above. No contact-success rule has been loosened.

Reference-velocity assistance plus RL corrections has produced real front
foot plants on the 180 mm first tread. A frozen first-event checkpoint from
the loaded-landing trial repeated the plant approximately 16 seconds into
the requested floor-start recording. This is one foot, with no whole-robot
first-tread hold or full climb. It also does not isolate the learned policy's
contribution from the reference assistance. The [IK progress report](rl-ankle-v23-ik-progress.md)
records provenance and the video state.

The original readiness potential takes the best candidate among all four
feet. After a light new plant, that can favor unloading the newly raised
foot again. Synthetic fixed-total-load arithmetic illustrates the issue:

| Force example | Existing readiness | Extra transfer potential |
| --- | ---: | ---: |
| New FR foot carries 3.85 N | 1.769, favors FR | 1.723 |
| Shift load to FR at 6.85 N | 1.394, favors FR | 4.125 |
| Unload the other front foot to 1 N | 2.125, favors FL | 6.375 |
| Unload FR again to zero | 2.250, favors FR | 0 |

These force distributions illustrate reward arithmetic only; they do not
establish mechanically feasible poses or trajectories. In particular, the
current landing pose cannot statically unload FL within the weak motor cap.

The isolated `stairs-load-transfer-256-s3281` trial started at 19:45 PDT.
It retains the loaded trial's reset, assistance, reference, PPO settings,
observation layout, physical limits and success rules. Its sole behavior
change is this additional potential after actual floor-start progress:

`3 * minimum raised-foot load fraction * best lowest-level-foot readiness`

A raised foot must currently satisfy the planted geometry and contact test.
Its load fraction saturates at 20% of total robot weight (about 6.76 N);
the existing readiness combines unloading, other-foot support quality and
stability. The added potential is bounded by 7.5, and contributes through
the existing gamma-matched potential difference with terminal zeroing. It
is not a recurring bonus for holding the same pose. The 20% target is an
experimental shaping choice, not a new landing-success threshold. The
candidate foot remains free to be either front or rear, with any heading.
See [force/reward arithmetic](../reviews/ankle-v23-curriculum/20260915/post-plant-load-transfer-arithmetic.json).

The practical progression is now: keep three supports, clear the full
riser, plant and hold, transfer meaningful load to that new contact, then
unload another foot. Static IK supplies reachable candidate configurations;
RL learns corrections through the existing constrained actuator controller.
The next route calculation also moves the supporting feet closer together
to avoid ending the first plant at a stance with inadequate motor headroom.

By 20:03 PDT that compact endpoint has an independently feasible FL-unloaded
allocation at 0.78115 of the weakest motor cap, versus 1.25482 previously.
The static second-front lift initially reached 160 mm; allowing a toe-edge
pivot of the planted front foot subsequently gave full clearance and a
second-front plant endpoint. Connecting motion remains unproven. This supports including next-step
feasibility when choosing the first foothold instead of maximizing immediate
forward progress alone. A compact first-front reference trial launched at
20:12 PDT, with the load-transfer reward retained. It uses 213 individually
bounded static poses; remaining interpolation limitations are recorded in
the training contract and the [IK report](rl-ankle-v23-ik-progress.md).
By step 1480 that compact trial recorded 34 actual new-foot plants, with no
whole-robot stage completion. A separate two-front trial began at 20:27 PDT
with 54 additional second-front pose targets. Actual first-front credit gates
the transition into the second motion. Its final guide condition requires
a simultaneous, low-slip two-front hold; merely visiting an IK target earns
no contact success. It recorded its first front-foot plant by step 620.

The current rear-transfer calculations show why an unloading reward alone
is insufficient. At the proposed two-front endpoint, rear-left unloading
needs 2.23476 times the weak motor cap and rear-right allocation fails.
Local fixed-foot and point-pivot posture searches have not resolved this.
Foot order and foothold placement must be considered before prescribing a
rear-lift target. Force feasibility here uses full three-dimensional wrench
balance, including contact heights, not only a horizontal support triangle.

## Earlier guidance changes

### Controller attenuation identified at 20:48 PDT

The contact-gated two-front trial passes its first plant and reaches the
second crossing, but currently has no pair hold. Static Jacobian arithmetic
shows the damped Cartesian IK strongly attenuates some reference posture
directions near straight-leg configurations. An isolated trial now centres
the damping penalty on the bounded nominal joint velocity during guidance,
keeping all rewards, references and physical limits fixed. This matters for
reward design: increasing a joint-tracking reward cannot directly remove a
controller's suppression of the required motion. The calculation establishes
the attenuation mechanism; training must establish its practical effect.

Additional primary research, [Optimized Static Gait for Quadruped Robots
Walking on Stairs](https://linqi-ye.github.io/docs/2021Optimized%20Static%20Gait%20for%20Quadruped%20Robots%20Walking%20on%20Stairs.pdf),
separates body-pose selection from contact sequence and foothold planning.
Its analysis shows why body pitch affects leg workspace and knee torque,
and its simulated gait alternates body shifts and leg swings. That supports
analysing reachable weight shifts and foot order before strengthening ascent
rewards here. Its much larger joint ranges and different robot prevent
copying its postures or treating its results as evidence for our robot.

Additional primary work supports the hybrid control choice.
[Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201)
combines conventional feedback with learned corrections for contact-rich
assembly. [Residual Policy Learning](https://arxiv.org/abs/1812.06298)
studies corrections to hand-designed and model-predictive controllers in
manipulation tasks. Our inference is to let a feasible stair reference reduce
the exploration burden while RL handles errors in tracking and balance.
Neither paper establishes stair capability for this robot, and the present
first-foot result has not isolated the contribution of RL from its reference.

At 18:46 and 19:00 PDT two isolated guided trials began. The first adds
105 state-driven rear-transfer waypoints to the grounded three-on-first
reset. The second adds an 82-pose floor-to-first-tread front-foot reference
and centers pitch commands on the reference foot angle. It retains the
full absolute pitch range and all physical motor/IK limits. Both allow
320-second episodes, with PPO and shaping gamma 0.999687548824. These are
combined hypotheses, not single-factor ablations; neither has demonstrated
a stage completion or full climb.

The guidance potential is `16 * waypoint_fraction + 2 * similarity * support`.
Similarity combines joint, swing-foot, root-position and orientation errors;
support comes from measured contact force and corner gaps on the other feet.
Targets advance after 0.1 seconds near the pose with sufficient support,
rather than by a clock. Guidance contributes a discounted potential
difference, with terminal zeroing. Actual foot planting remains +8, first
whole-robot tread hold +20 and full ascent +120. Front guide completion
requires actual first-tread FR credit; rear guide completion requires the
strict whole-robot first-tread hold. Neither is reported as a full climb.

The action analysis matters as much as reward weights: the existing absolute
pitch command maps zero to a flat foot, whereas the new front and rear
grounded poses require about -30.40 and -44.86 degrees. Centering the command
on the planned tilt lets the policy explore around a useful configuration.
The new mapping keeps both +/-110-degree endpoints and existing rate guards.

The new front candidate passes sampled collision checks and physical-bound
static force allocations through a full one-foot floor-to-tread transfer.
It does not establish dynamic execution or a whole-robot climb. The rear
reference still has unproven connections near landing. See the
[19:00 IK continuation](rl-ankle-v23-ik-progress.md) for calculations,
reserve sensitivity and training provenance. No active package was edited.

At 17:29 PDT, `stairs-landing-align-256-s3281` began a separate reward
comparison against the linear-pitch arm. Existing post-edge shaping depends
on contact force, leaving its direct swing score flat during airborne
flattening and lowering. The added potential uses a Gaussian of all four
traction-corner height errors (35 mm width), gated by the whole patch being
over the next tread and weighted by other-foot support quality. It increases
by 1.049 along a calculated front-foot crossing-to-landing sequence. A
credited-level term preserves the potential when landing advances the target;
the reward remains a discounted potential difference with terminal zeroing.
Landing and full-climb bonuses are unchanged. This is an exploratory reward
hypothesis, not a measured improvement. The floor and grounded three-on-first
resets remain unchanged. See the
[reward calculation](../reviews/ankle-v23-curriculum/20260915/front-landing-reward-arithmetic.json)
and [current IK continuation](rl-ankle-v23-ik-progress.md).

Further primary research supports testing planned foothold tracking once a
usable reference exists. [DTC: Deep Tracking Control](https://pubmed.ncbi.nlm.nih.gov/38232148/)
combines a model-based reference planner with a learned tracking policy.
[Learning Bipedal Walking On Planned Footsteps](https://arxiv.org/abs/2207.12644)
conditions a policy on upcoming footsteps, including stair sequences in
simulation. The inference for this robot is to use feasible contact and
foothold targets to reduce exploration, while letting RL learn balance and
tracking corrections. These papers do not establish that this robot's motors
can execute the computed poses. Current rear-transfer samples are used only
as soft targets; they still need contact-path refinement before they could
serve as a prescribed connecting motion.

[Blind Stair Climbing](https://arxiv.org/html/2402.06143v1) reports that a
position-goal formulation lets robots slow down and choose different stepping
motions, unlike strict continuous velocity tracking. Our inference is to
retain free timing and direction, including sideways motion, and emphasize
actual supported footholds. Its lower-height curriculum is not adopted here.

The rear-path calculation also exposed an unnecessary planner restriction:
holding the lowest swing corner at exactly 185 mm during flattening. One
rejected candidate satisfied support contact to about 0.0002 micrometres and
static force balance, but its swing corner was 15.3 micrometres below that
arbitrary flight plane. Using **at least 185 mm** clearance lets the same
fixed-support sequence continue to a flat airborne foot, within the existing
motor cap. The landing still targets actual 180 mm tread contact. This
supports a reward design that saturates clearance once it is sufficient and
leaves freedom to adjust flight height while rotating; it does not justify
weakening contact or success criteria. The existing clearance score already
saturates, so this finding changes the offline planner, not the live jobs.

The next endpoint calculation reaches tread contact at 1.792 degrees of sole
tilt and can assign 6.44 N to its actual grounded corner within the motor
cap. Requiring full sole flattening before contact was unnecessarily
restrictive for that candidate. This motivates rewarding a supported plant
before demanding final settling. It does not prove a gait: sampled
interpolation still collides with the other foot, and the FL supporting sole
is not yet within the strict whole-robot planted gate. A useful motion
reference must guide clearance of both the stair and the other legs/feet.

Update: both reward runs described below completed all 256 updates, with
zero first-tread completions and zero full climbs. Their final episode counts
were 2,647 and 2,463. The active successor and new calculations are documented
in the [IK continuation](rl-ankle-v23-ik-progress.md). Earlier active-run and
startup statements below describe the historical launch snapshot.

At 16:39 PDT, a new `stairs-contact-margin-256-s3281` trial began. Offline
analysis found positive actual support-hull margins of 20.7–29.5 mm in 85
static samples that the old foot-center triangle labeled unstable. The new
reward uses near-surface foot corners for this feature. Separate arithmetic
also identified a 0.852 potential reduction during a candidate forward swing
after folding; that term is unchanged in the new trial to isolate the support
margin comparison. The [current IK continuation](rl-ankle-v23-ik-progress.md)
contains the limitations, artifacts, and four active training jobs. No full
climb has succeeded.

## What the completed runs actually learned

The sideways run `realstairs-side-seed3211` completed 448 updates and 3,790
ended episodes. It recorded zero first-tread completions and zero full climbs.
Its last 512 ended episodes averaged 11.15 N on rear-right and a minimum
goal distance of 268 mm. The policy mostly preserved its starting pose.

The force-feedback sequence `realstairs-force-seed3181` completed 448 updates
and 535 ended episodes. It recorded one curriculum stage completion, zero
floor-start stage completions in 221 ended floor episodes, and zero full
climbs. A stage here is a front-foot sequence and a rear-foot floor shuffle,
not all four feet ascending a tread.

[The offline diagnosis](../reviews/ankle-v23-curriculum/20260915/realstairs-reward-diagnosis.json)
uses existing logs, not additional policy rollouts. In the last 120 one-second
trace records for environments 0–3 of the sideways run:

| Measurement | Result | Implication |
| --- | ---: | --- |
| Three other feet pass the strict planted gate | 44.4% of samples | Unloading/clearance shaping is switched off in most of this sample |
| Rear-right load below 3 N | 0.6% | The foot rarely unloads sufficiently |
| Rear-right sole center above 30 mm | 2.5% | It rarely begins a substantial ascent |
| Rear-right load, median | 11.52 N | A heavily loaded foot cannot simply be swept forward |
| Front-left largest corner gap, median | 7.45 mm | This lies close to the old 8 mm binary gate |

These are sampled training states, not population estimates or fixed-policy
reliability measurements. At the episode-average 11.15 N load, `mu * Fz` is
about 10.04 N for the model's dynamic friction coefficient 0.9. This is a
friction scale, not evidence of measured horizontal sliding force.

The old unloading score `exp(-Fz/3)` is only 0.0243 at that load. Its slope
is 0.00809 per N before applying the binary support gate. A linear bounded
unloading score `clip(1 - Fz/16, 0, 1)` has slope 0.0625 per N while active.
That supplies a more consistent early signal as load starts to decrease.

The old PPO configuration used gamma 0.999, lambda 0.97, and 64 control
steps per rollout at 60 Hz. The raw GAE coefficients decay by a factor of e
in about 0.53 seconds, while collection chunks span 1.07 seconds. This is
not a hard limit on what the critic can learn, but it makes slow unloading
and stepping harder to connect directly to later rewards.

Finally, the old unscripted exploration environment deliberately terminated
at the all-four-on-first-tread milestone. It was a local transfer experiment;
even success there could not continue into a four-riser climb in that episode.

## Mechanical calculation

The static solver uses URDF masses, gravity, contact corner forces, a friction
pyramid with coefficient 0.9, the weakest modeled motor strength, and the
original bounded PD setpoints at stiffness 30. It does not simulate dynamics.

- The original wider sideways reset uses about 16.1% of the weakest motor
  cap with all four feet loaded. Removing rear-right support at that same
  pose requires about **102.9%**, including PD setpoint headroom.
- A posture search within 99% of the existing physical joint limits found
  an unloading endpoint at **98.4%** with no detected self or stair collision.
  Its load margin is only about 1.6%; it changes support footholds and is not
  a demonstrated route from the original reset.
- Holding those new support footholds fixed, a sampled rear-foot lift to
  30 mm requires **106.3%** of the weakest cap; the 90 mm sample requires
  **101.8%**. Higher samples encounter contact-flatness, joint, or collision
  failures. No valid complete connecting path was found.

See [unloading endpoint](../reviews/ankle-v23-curriculum/20260915/realstairs-unload-clear-calculation.json)
and [rear-foot samples](../reviews/ankle-v23-curriculum/20260915/realstairs-rear-transfer-calculation.json).
The earlier lower-torque candidate that intersected the next riser is retained
as a rejected calculation. None of these new endpoints is used as a training
reset or reference.

The working hypothesis is that this route needs support-foot repositioning,
different body rotation, or dynamic load transfer. A reward for raising the
rear foot while insisting on unchanged support geometry is unlikely to be
enough. These local calculations do not prove that stair climbing is impossible.

## New training experiment

Two runs compare rewards with the same seed, 128 environments, 256 planned
updates, 256-step collection chunks, gamma 0.9997, lambda 0.995, and fresh
Beta(8,8) policies. Each run plans 8,388,608 training transitions. Collection
chunks now span 4.27 seconds; raw GAE decay is about 3.14 seconds. This pair
isolates the additional support potential, but comparison against the older
run also changes observations, exploration, resets, horizon, and learning
settings, so it is not a reward-only comparison against that older baseline.

Resets are 25% straight floor, 25% oblique floor, and 50% the previous sideways
pose with three feet on the first tread and rear-right on the floor. Every
foot ascent is on the actual 180 mm stair geometry. There are no hovering
or reduced-height resets. Episodes can run for 96 seconds and continue after
the first-tread milestone. Only floor starts can count as a full climb.

The policy observes which foot is being encouraged, its next tread, the
clearance/landing phase, and progress already credited. It selects all 16
joint-target velocities, with the same physical joint, target-rate, torque,
and collision limits as before. It can move any support foot. No intermediate
joint trajectory is prescribed.

| Reward component | Purpose |
| --- | --- |
| Smooth support quality | Give graded feedback as corner gaps and contact loads improve; retain strict success checks |
| Bounded linear unloading score | Encourage shifting weight off the selected foot before lifting |
| Clearance from the lowest sole corner | Encourage the entire traction patch to rise above the next riser |
| Forward approach after clearance | Encourage movement over the riser without dragging into it |
| Descent and loaded patch contact | Turn a lifted foot into a usable foothold |
| +8 once per selected foot/new tread | Reward qualified landings held 0.25 s, without paying repeatedly for the same tread |
| +20 first-tread milestone; +120 fourth-tread completion | Distinguish local progress from the staircase objective |
| Small time, energy, action-change and soft-range costs; −12 fall | Discourage indefinite hovering, wasteful motion, and falling |

The shaping part is `gamma * Phi(next) - Phi(previous)`, with zero potential
at absorbing terminal states and time-limit bootstrapping retained. Landing
and task bonuses are separate intentional task rewards. Potential differences
avoid a repeated positive reward simply for holding a raised foot. This does
not eliminate every possible reward-design failure; contact, force, clearance,
and successful trajectories are logged to inspect what the agent actually does.

- `stairs-progress-r0-s3231`: clearance, unloading, landing, and forward-progress
  potential; no extra center-of-mass term.
- `stairs-support-r2-s3231`: the same experiment plus a **nonnegative**,
  support-quality-weighted center-of-mass margin potential. This softly
  encourages weight transfer toward the other three feet. A triangle through
  foot centers is a conservative heuristic, not a full contact-wrench solver.

The brief predecessor `stairs-progress-r2-s3231` was intentionally stopped.
Review found that multiplying a negative margin by contact quality could make
contact loss look like immediate improvement. Its update-0 checkpoint is
archived. The successor uses a nonnegative support score in a separate package;
the active baseline's source and provenance were not changed.

Both active runs are privileged simulation teachers with a D405 depth proxy.
They are not D405-only controllers. No full climb is claimed from startup or
early training, and their changing-policy episodes are not frozen evaluation.

At the early inspection, the baseline had reached control step 3,313 with
214 ended episodes and the revised support run step 743 with 15 ended
episodes. Neither had a first-tread completion or a full climb. All episodes
that had ended at that point ended in falls; surviving episodes were still
running, so this is not a final fall-rate estimate. The runs remain active;
these early results do not establish an improvement over the previous policy.

## Research informing these choices

[Ng, Harada and Russell's reward-shaping paper](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
provides the discount-consistent potential-difference construction. Here it
motivates the shaping formula; it does not guarantee learning or physical
feasibility.

[Terrain-Aware Quadrupedal Locomotion](https://arxiv.org/html/2310.04675v2)
uses terrain-related foot rewards and terrain-dependent lifting-height terms.
It supports tying rewards to foot geometry and terrain, rather than a generic
body-height bonus. The present reward and robot differ from that work.

[Learning Transferability for U-Shaped Stair Climbing](https://arxiv.org/abs/2602.14473)
reports a stall penalty as part of its Go2 stair-climbing method. That motivates
watching for stationary reward optima, but its robot and training setup do not
establish that V2.3 can climb this staircase.

Work performed: authorized RL training, read-only log/source inspection, and
offline geometry/static-force calculations. No separate automated tests,
evaluation batch, video rollout, or hardware check was run.
