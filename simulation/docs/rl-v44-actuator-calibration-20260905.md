# V44 continuation: actuator/load hypotheses and contact curriculum

## Latest update — September 6, 19:12 PDT: study sequence stopped

42 completed matched100-update training/evaluation: tilt measure improved5.4%,
but speed fell1.6% and FR clearance remained3.981mm;26/30 gates, mechanism
screen failed. Both allocations had zero failure terminations; evaluations
remained upright and stopped stably. No further experiment is queued.
V44 study35 remains best nominal all-round candidate:1.541cm/s, all-leg
clearance/release and stable stop,28/30 gates, not qualified running. Its exact
experimental target mapping must be integrated before copying weights could
reproduce the simulated controller. No Pi access/deployment or hardware checks.
[Selection, video and mechanical feasibility](../reviews/v44-actuator-20260905/CANDIDATE_SELECTION_AND_MECHANICAL_FEASIBILITY.md),
[42 final evidence](../reviews/v44-actuator-20260905/BODY_TILT_42_RESULTS.md).

## Latest update — September 6, 18:16 PDT

41's longer/slower steps reduced acceleration but lost speed and foot release;
the comparison failed and its video is complete.42 now compares body-leveling
reward8/512 with matched100-update continuations. Both initial replays matched
exactly; control training started. Per the user's latest boundary, finish42,
then stop experimenting and rank the best candidate for discussion of a real
trial and mechanical feasibility.35 remains provisional best, not qualified
running. No Pi access/deployment or controller export is authorized here.
[Current summary](../reviews/v44-actuator-20260905/PROGRESS_SUMMARY.md),
[42 progress](../reviews/v44-actuator-20260905/BODY_TILT_42_RESULTS.md).

## Latest update — September 6, 17:44 PDT

40 completed all200 PPO updates and50/100/200 evaluations.4,915,200 transitions,
zero failure terminations; complete provenance/allocation audit. Final25/30
gates, speed0.013854m/s, FL/FR clearance4.756/2.095mm and heading12.68deg.
Rear clearance improved to8.488/10.462mm and stopping remained stable, but
this is not a qualified model. All three registered screens failed; no extension
or deployment. Final video is being recorded. The next separately registered
41 probe compares longer/slower steps at the same command and unchanged caps.
[40 final results](../reviews/v44-actuator-20260905/CONSTANT_STANCE_40_RESULTS.md),
[41 protocol](../reviews/v44-actuator-20260905/LONG_STROKE_41_PROTOCOL.md).

## Latest update — September 6, 17:29 PDT

40's initial replay matched exactly; actual 200-update training is running.
The completed 50/100-update evaluations both pass 26/30 gates and fail the
registered adaptation screen. After100, rear clearance improves to8.618/6.933mm,
but front-right clearance falls to3.716mm and speed to0.01343m/s (17.40% below
unadapted39). All-leg release passes, both episodes remain upright and stops
settle. Angular/joint acceleration meet retention; linear acceleration does not.
Training continues to200 with frozen settings. No Pi deployment. See
[40 results and video](../reviews/v44-actuator-20260905/CONSTANT_STANCE_40_RESULTS.md).

## Latest update — September6,16:44 PDT

39 constant-stance timing is12.82% faster,18.67% lower body linear acceleration
and11.53% lower angular acceleration, but rear core clearance regresses to
2.840/3.331mm and RR release64.02%. Both episodes nonfallen, stops settle,
all mapping audits pass.26/30 gates; mechanism screen rejected, separately
registered adaptation-feasibility screen passed. No physical limit changed.

40 initial replay launched16:43:56 before200-update PPO adaptation on39's
reference, full34 optimizer/checkpoint, same rewards/native Beta/controller.
Evaluations after50/100/200 updates. No40 PPO result yet; no deployment.
[40 progress](../reviews/v44-actuator-20260905/CONSTANT_STANCE_40_RESULTS.md),
[39 metrics/video](../reviews/v44-actuator-20260905/CONSTANT_STANCE_39_RESULTS.md).

## Latest update — September6,16:33 PDT

38 completed27/30 gates; RR clearance+0.551mm and RL+0.357mm. RL misses the
registered0.5mm mechanism gain; no training extension or deployment. Filtered
target tracking RMS slightly worsens despite better raw-request knee error.
Completed local video and the preidentified label-only reporting recovery are
preserved in [38 results](../reviews/v44-actuator-20260905/REAR_FEEDBACK_38_RESULTS.md).

39 now compares reference fore-aft timing. Original planted-foot velocity is
not constant; replacement has constant stance velocity and C2-matched swing
boundaries. Same geometry/lift/cadence/caps, explicitly increased swing x
overshoot. A large cadence increase is not feasible analytically under the
existing90deg/s knee limit. [39 protocol/status](../reviews/v44-actuator-20260905/CONSTANT_STANCE_39_RESULTS.md).
No40 training has started; a potential adaptation observer is only drafted.

## Latest update — September 6,16:14 PDT

37 completed matched50-update allocations with no failure terminations, but
was rejected: concentration2 gives0.01484m/s, RR core3.528mm and26/30 gates.
Its control is byte-identical to36 control; numerical rollout identity and
distribution audits passed. Structured-config and schema-only recoveries are
preserved, not hidden. Lower sampling noise reduced actual sampled joint FD
acceleration RMS3.97%, not enough to improve deterministic walking.
[37 results](../reviews/v44-actuator-20260905/BETA_CONCENTRATION_37_RESULTS.md).

Read-only URDF-frame reconstruction reproduces recorded shoe heights within
0.000288mm. Weak rear low-clearance samples lose roughly3–4.4mm between
filtered-target geometry and actual measured geometry. This is an ordered
kinematic association, not proof of a dynamic cause. No geometry changed.
Study38 now compares gain0.25 vs0.5 on fixed34, same2-degree correction cap,
same limits/rewards/controller arithmetic. Control launched16:14; no result yet.
[38 findings/protocol](../reviews/v44-actuator-20260905/RECENT_CLEARANCE_38_FINDINGS.md).
No deployment, Pi access, unrelated tests or production enablement.

## Latest update — September 6,15:11 PDT

36 completed both50-update allocations (1,228,800 transitions each,zero failure
terminations), exact initial replays and final evaluations. Both26/30 gates;
the added normalized slip penalty leaves speed effectively unchanged and
increases worst slip2.47%. Right-foot clearance remains below5mm. Mechanism
rejected; source/config/observer/full-resume audits passed. A namespace-only
prelaunch recovery is preserved separately from the original failed attempt.
[36 results and local20s video](../reviews/v44-actuator-20260905/NORMALIZED_SLIP_36_RESULTS.md).

37 started initial replays15:11:07 PDT, PID15204. It compares native-equivalent
factor1 with factor2 Beta concentration, both from full34 with the same rewards
and controller. Saved-state CPU sizing predicts26.58% lower mean exploration
standard deviation with identical same-state mean; no gait benefit claimed.
Manifest SHA `85412d8c947a42d932c476e4d61d0e5da9f2e1cc3ab92f26d384fb3c2db2f206`.
[37 protocol](../reviews/v44-actuator-20260905/BETA_CONCENTRATION_37_PROTOCOL.md),
[37 progress](../reviews/v44-actuator-20260905/BETA_CONCENTRATION_37_RESULTS.md).
No production changes/deployment, hardware access, commits/pushes, hosted
reviews or unrelated verification. Earlier source snapshots remain immutable.

## Earlier update — September 6,14:26 PDT

35 completed at14:08:41:28/30 gates and0.015411 m/s, stable stopping and all
absolute clearance/release gates. Its knee-tracking gains4.6/7.3% miss the
predeclared10%, so preserve the negative mechanism decision. Independent
envelope/causality/bound audits passed. Local20 s video shared.
[35 results](../reviews/v44-actuator-20260905/FRONT_STANCE_35_RESULTS.md).

36 initial control replay launched14:23:49, PID57040, before a matched50-update
per-arm normalized-slip reward comparison. Original34 controller, front
feedback off; full34 checkpoint396/optimizer, lambda0.99. No new performance
result yet. [36 protocol](../reviews/v44-actuator-20260905/NORMALIZED_SLIP_36_PROTOCOL.md).
Manifest SHA `34439a727d3d6c39c1919c0e710f69fbb3338d4e325b1d3627105423cba42a70`.

## Earlier update — September 6, 14:02 PDT

34 completed50 updates and both fixed-command simulations. High speed0.014413
m/s is18.7% above the matched33 control;27/30 gates pass. Low-command RR core
clearance falls to3.855 mm; release regresses, so the combined screen is rejected.
No deployment. [34 results and video](../reviews/v44-actuator-20260905/CREDIT_34_RESULTS.md).

35 is now a separate fixed-policy front-stance feedback comparison, not more
training on the failed branch. Zero-gain replay launched14:01:45 PDT, PID39608;
gain0.25 treatment requires exact replay first. Limits unchanged, requested
pair pitch preserved, smooth stance/start/stop gates, no integral. Both front
knees need >=10% loaded tracking improvement with all clearance/no-regression
checks retained. [35 protocol](../reviews/v44-actuator-20260905/FRONT_STANCE_35_PROTOCOL.md).
Manifest SHA `fa184f94e519452a96cf9432cdc310024dcb6e725d5111a4a32c08b7bee4c9f7`.
Prior source snapshots, failures and experiment artifacts remain immutable.

Started September 5, 2026, approximately 15:25 PDT. **Active work; no qualified
new policy or deployment.** The user requested continued autonomous work toward
the locomotion goal. Actual training and simulation evaluation are authorized;
new physical robot motion, deployment, commits/pushes and unrelated verification
remain outside scope. The Pi is not being accessed.

Baseline: [first training session](rl-v44-training-20260905.md), especially final
common-floor authority checkpoint 199 and its diagnostic traces. That pilot
completed 200 updates but failed speed, clearance, slip, effort and all-leg
release acceptance. Existing artifacts remain unchanged.

## Latest checkpoint and diagnostic summary

**Active — September6,13:12:34PDT:**34 testslambda.99 versus33'slambda.95,
samefull31model347,optimizer,controller/rewards,50updates.33completed:
speed+6.32/+6.16%low/high,but lostlowRRclearance andFLrelease atboth; rejected.
34 starts baseline replays; no performance result/deployment yet.
[33 results/video](../reviews/v44-actuator-20260905/CONTINUATION_33_RESULTS.md),
[34 protocol/progress](../reviews/v44-actuator-20260905/CREDIT_34_RESULTS.md).

**Active — September6,12:39:49PDT:**33 starts baseline replays before50additional
PPOupdates from full31model347.32upper command improves release/all4core,
27/30gates,but actualspeed−1.393%withcommand+16.667%. Same rewards/controller
for33; complete optimizer/critic/normalizer resume,no reset or deployment.
[32 results](../reviews/v44-actuator-20260905/SPEED_RESPONSE_32_RESULTS.md),
[33 protocol/progress](../reviews/v44-actuator-20260905/CONTINUATION_33_RESULTS.md).

**Latest — September6,12:27PDT:**31 completed50updates and final evaluation:
26/30 gates,0.011602m/s,RR5.0565mm,all4core>=5mm,no falls,stop/hold pass.
Mechanism screen passes,not hardware/run qualification.32 now compares the
unchanged model at its trained upper command after exact low replay. Its first
launch was rejected before simulation by the existing exact-command guard;
v2 uses an explicit source-checked research route with unchanged numeric gates.
[31 result/video](../reviews/v44-actuator-20260905/SWING_TILT_DOSE_RESULTS.md),
[32 protocol](../reviews/v44-actuator-20260905/SPEED_RESPONSE_32_V2_PROTOCOL.md).

**Historical launch — September6,11:50:41PDT:** dose31 changes only
swing-tilt weight2→4,same original bootstrap/settings.30 completed: RR4.633mm
(+.549mm versus29),speed.011622m/s (+1.492%),25/30 gates. FL/FR clearance
decreased; retain their5mm requirements. No extension of failed30 or deployment.
[Protocol](../reviews/v44-actuator-20260905/SWING_TILT_DOSE_PROTOCOL.md),
[30 results/video](../reviews/v44-actuator-20260905/SWING_TILT_TRAINING_RESULTS.md).
Manifest `755d0a23b42982b613792034c83d6f61503e5957b0093873b4067f1f53ee71f0`.

**Historical launch — September6,11:23:34PDT:**30 adds scheduled-swing
sole-tilt weight2 to the retained28 feedback.29 completed50 updates with no
failure terminations,but speed+12.84% traded RR clearance5.173→4.085mm and
was rejected. Saved traces show learned RR requested pitch+1.57°,not worse
joint following.30 is a same-bootstrap/same-settings reward-only comparison.
[Protocol](../reviews/v44-actuator-20260905/SWING_TILT_TRAINING_PROTOCOL.md),
[29 results/video](../reviews/v44-actuator-20260905/PITCH_FEEDBACK_TRAINING_RESULTS.md).
Manifest `2bdc1040e088ffd6d956fa2e590b7a167c93b40ef5f90519512ecc85142f30ea`.

**Historical launch — September6,10:54:25PDT:**29 begins with exact
fixed-actor replay,then50 PPO updates298..347 and canonical evaluation.28 passed
all9 mechanism conditions,with RL5.451mm/RR5.173mm corep05 and26/30 gates.
Speed remains.01015m/s.29 retains original rewards and all physical limits;
aim is faster motion while retaining all-foot5mm clearance. No deployment.
[Protocol](../reviews/v44-actuator-20260905/PITCH_FEEDBACK_TRAINING_PROTOCOL.md),
[progress](../reviews/v44-actuator-20260905/PITCH_FEEDBACK_TRAINING_RESULTS.md).
Manifest `ddb4c4fd928481c1ab093c717c85548c001d586b4233c0c49aec7c99e9cac41c`.

**Historical launch — September6,10:42:54PDT:** both-pitch28 applies
the27 equal/opposite correction to both rear legs.27 completed10:40:19 with
RR corep05=5.050734mm (+1.570597mm original),RL4.741728mm (strict recovery
miss by.001848mm),25/30 gates,stable stopping. No PPO extension selected yet.
[28 protocol](../reviews/v44-actuator-20260905/REAR_FEEDBACK_BOTH_PITCH_PROTOCOL.md),
[27 results](../reviews/v44-actuator-20260905/REAR_FEEDBACK_PITCH_RESULTS.md).
Manifest `0dcbfd57fcbaa18785dca076206aa92c0d3bcde160a1372ad3909eb92349831c`.

**Historical launch — September6,10:33:07PDT:** pitch27 projects the
already capped RR-only correction onto equal/opposite hip/knee increments,
preserving added requested pitch and nonincreasing correction norm. No other
controller or reward change.26 failed RL recovery despite RR+1.019mm.
See [27 protocol](../reviews/v44-actuator-20260905/REAR_FEEDBACK_PITCH_PROTOCOL.md)
and [26 results/video](../reviews/v44-actuator-20260905/REAR_TRACKING_MASK_RESULTS.md).
Manifest `eac32129eea9d7099096b7b7ec0eb436f0653256c123096e2ac42def5cd94b0b`.

**Historical launch — September6,10:21:16PDT:** mask26 isolates
RR-only feedback from both-rear feedback at identical gain.25 and2° cap.
Both-rear replay must exactly match25,including all feedback trace fields.
Prospective screen additionally requires recovering original RL clearance.
See [protocol](../reviews/v44-actuator-20260905/REAR_TRACKING_MASK_PROTOCOL.md).
Manifest `7d0a032eeb08faeef9cb1b86ad1f987463abec99d20bfb099087e81c6dd4d746`.

**Completed10:10:48PDT:** feedback25 improved RR clearance.958234mm and
release1.38675pp, but fell short of its exact1mm OR2pp mechanism threshold.
RL clearance fell.318289mm; all other conditions passed,25/30 gates unchanged.
No training extension selected. [Results](../reviews/v44-actuator-20260905/REAR_TRACKING_FEEDBACK_RESULTS.md).

**Historical launch — September6,10:03:41PDT:** rear tracking25
compares original298 with zero versus bounded gain.25 rear-swing correction.
It preserves original rewards and every downstream motor/geometry limit.
No training/deployment is automatic. See
[protocol](../reviews/v44-actuator-20260905/REAR_TRACKING_FEEDBACK_PROTOCOL.md).
Manifest `4302f7a6d574d75202b28c38b048e0d59a31d3ffe38d1db3a01075cd82e91863`.

**Completed09:52:07PDT:** raw-smoothness24 was rejected despite small gains:
speed+1.487%,RR clearance+0.111mm vs control20,25/30 gates. No meaningful
rear improvement and no extension. Stable stopping retained. Its local20s
video completed09:55:22PDT and5/12/19s frames were inspected. See
[results](../reviews/v44-actuator-20260905/RAW_SMOOTHNESS_RESULTS.md).

**Historical launch — September6,09:32:44PDT:** candidate24 begins with an exact
fixed-policy replay, then50 PPO updates and canonical evaluation. Only the
raw dimensionless action-second-difference coefficient changes3→1. Original
contact-gated sustained reward is restored; control20 is the comparator.
All physical smoothness terms and motor caps remain. See
[protocol](../reviews/v44-actuator-20260905/RAW_SMOOTHNESS_TRAINING_PROTOCOL.md).
Manifest `6050be00b2096f0a5deadea6ea43e2c8719cf6ca43e2022fedf50903f1fb26bc`.
The preceding passive4-update measurement exactly reproduced actions/motion
and final checkpoint;99.01% of run squared raw curvature was its algebraic
innovation component,0.166% conditional-mean component. This motivates an
experiment, not a claim that weaker regularization is better. See
[measurement results](../reviews/v44-actuator-20260905/ACTION_STATISTICS_RESULTS.md).

**Latest completed result — September 6,09:11:39PDT:** candidate22's50-update
sustained-progress reward experiment failed its physical improvement screen.
Speed is1.258% above control20, but RR core clearance is2.381mm versus3.480mm
at the original baseline; the required rear improvement did not occur.
It passes25/30 gates and retains stable stopping. See the
[complete table and receipts](../reviews/v44-actuator-20260905/SUSTAINED_PROGRESS_RESULTS.md).
This reward variant is closed. A local illustrative video completed09:14:01PDT
and was inspected. Subsequent support evidence shows unloaded but shallow RR
core swing, not missing scheduled stance contacts. No Pi deployment, hardware
motion, commit or push.

**Historical launch — September 6, 08:52:10 PDT:** the sustained-progress reward
comparison has launched its initial fixed-policy identity replay. If all saved
physical rows and canonical metrics reproduce exactly, candidate22 will train
for 50 PPO updates from the original298 optimizer-reset bootstrap, followed by
the canonical evaluation. Only the instantaneous pair-release factor on the
signed three-second sustained-progress term changes. The other four progress
terms, all foot/contact/smoothness costs, original .5 descent, physical limits
and all30 gates remain. Control20 is reused from its verified saved records.
No replacement model or deployment is claimed. See the
[selected protocol](../reviews/v44-actuator-20260905/SUSTAINED_PROGRESS_TRAINING_PROTOCOL.md).
Manifest `0bd6a4cec6515e06243032dbe51ba9b24d54af4197ad90a5cd8a7b7ac96b94d5`
contains 297 input receipts and 123 snapshots. This is a new reward experiment,
not an extension of the rejected .3-descent or lambda99 branches.
At **08:55:40 PDT**, the initial replay completed with exact identity and actual
candidate22 PPO training launched. See the evolving
[progress/results record](../reviews/v44-actuator-20260905/SUSTAINED_PROGRESS_RESULTS.md).

**Current result:** the lambda-only pair completed September 6 at **05:48:49 PDT**.
The treatment is13.84% faster than the final control and newly passes the speed
gate (26/30 versus25/30), but RR clearance/release regressed beyond the registered
own-initial limits. Both branches are closed without extension; neither is a
qualified replacement. See the [compact progress table and full result](../reviews/v44-actuator-20260905/CREDIT_HORIZON_TRAINING_RESULTS.md).
The completed saved-data analysis is `76a412745da9440c443cdd118c5119d8593616e57f1cd7caf27601f5079bd0f5`.
The research-only 20-second treatment347 clip completed at 08:05:10 PDT and
was shared locally. It does not alter the failed decision.

**Latest physics result, completed at 08:35:33 PDT: rejected.** Later rear-foot
descent gave 2.34% more speed but reduced rear-right clearance from 3.480 to
2.824 mm and release from 67.334% to 66.179%. Both arms passed 25/30 gates;
neither fell, and stopping remained stable. The control reproduced all 3,528
historical physical rows and canonical metrics exactly. See the
[timing result and phase analysis](../reviews/v44-actuator-20260905/REAR_DESCENT_TIMING_RESULTS.md).
No active job remains from this comparison. The narrow sustained-progress
reward timing proposal is the next retained hypothesis, still unapplied;
no new PPO allocation, model promotion or deployment has occurred.

Historical launch at 08:28:32 PDT: original298 with unchanged
5 mm peak rear lift, comparing descent fractions .5 and .3. The .5 control must
first reproduce all saved baseline physical rows and canonical metrics exactly.
The offline timing calculation passed: about 1.586 mm additional late-swing
commanded lift, no additional peak lift or limiter events on the saved trace.
This does not establish improved physical clearance. See the
[prospective timing protocol](../reviews/v44-actuator-20260905/REAR_DESCENT_TIMING_PROTOCOL.md).
Prepared manifest `ffd4cd9dc31b231526a4b2474eb6fec84a25c6fccdb05fbc4c3d9b23112a2c21`
captures 116 input files and 113 snapshots. Only four package files change;
the other 75 package members, original actor, references, physical asset,
rewards and numeric gates are unchanged. No PPO allocation or Pi deployment
is selected for this mechanism comparison.

Historical launch record: the matched lambda-only credit-horizon
training pair launched September 6 at **05:21:37 PDT**. Both branches used
original Control09 model298, fixed 5 mm rear lift, identical preserved
actor50/critic58/normalizer values and reset Adam state at LR5e-5. Only
GAE lambda differs (.95 / .99); gamma .995, physics, rewards, references,
stop behavior and all gait gates remain unchanged. The allocation is 50
updates per arm, followed by both canonical evaluations; no deployment.

Preparation manifest: `f31328078f1a0248c47eed95d845c63f9c5f12f4c44fd5593f0690b62dc0fb71`.
Bootstrap receipt: `05558668ce312a979b9f1d1a1e02c004e47bc545578c2340aa9c1b58a34f0f4a`.
Run-start receipt: `dcd6ddeee3d61711183d21bab327f877ed8b434d011d41ea050400090c4acf70`.
These are actual selected artifacts under `credit-horizon-pair-v1`, not a
prospective launch. Root reviewed the complete source, peer review, the
three-file training/reporting-only byte bridge, and tensor-preservation
results before launch. All other 76 package members remain exact relative
to the initial02 evidence; no new initial physics replay was necessary.
Passive credit diagnostics capture existing PPO computations without extra
policy/value forwards. At approximately 05:43 PDT, the outer supervisor observed
both complete 50-update runs and final model347 checkpoints. Both canonical
walking/stopping evaluations subsequently completed, as recorded above.
The [saved first-rollout comparison](../reviews/v44-actuator-20260905/CREDIT_HORIZON_FIRST_ROLLOUT_COMPARISON.md)
found exact common initial scalar and physical/action evidence between the two
arms, and exact common physical/action evidence against prior lift19. This is
observed first-rollout identity, not a whole-training passivity guarantee.

The preceding projected-action pair
completed September 6 at 04:54:32 PDT, including both 50-update runs and both
canonical evaluations. Both retain 25 passing / 5 failing original gates,
no falls and stable stopping. Speed improved to 0.0120864 / 0.0119247 m/s
(zero / 5 mm lift), but zero lost 3.929 percentage points of rear-right
release and lift lost 1.026228 mm of rear-right core clearance from their
respective initial baselines. Both fail the predeclared no-regression screen;
neither final347 checkpoint is selected for further training or deployment.
The lift still has a rear-clearance advantage over the final zero control,
which is distinct from acceptable learning. A [20-second local illustration of lift347](../reviews/v44-actuator-20260905/rear-pitch-projected-five-mm-347-video-v2/README.md)
completed at 04:59:37 PDT; it does not change that decision. Completed-data diagnostics
are recorded in `rear-pitch-training-diagnostics-attempt02-summary-v1.json`
(SHA `c9d137d292d15a2e820e32e47731993c1ba36b74264c833c8f3fef1b539a675e`).
See [final measured results and branch closure](../reviews/v44-actuator-20260905/REAR_PITCH_PROJECTED_TRAINING_ATTEMPT02_RESULTS.md)
and [the separate next comparison design](../reviews/v44-actuator-20260905/LAMBDA_ONLY_TRAINING_PROTOCOL.md).
The pair launched at 04:30:19 PDT after both new initial comparisons passed exactly.
The earlier 50-update pair launched at
03:53:34 PDT but stopped at 04:05:27 PDT before completing its allocation.
Both complete two-episode trajectories were retained exactly; the rear
pitch-preserving lift still passes its mechanism screen. Neither result is
a qualified gait. No final checkpoint evaluation or learning decision exists
for the interrupted pair, and its partial checkpoints will not be promoted.
The fixed-X5mm construction failed its offline prerequisite on61rear-left
shoe-tilt samples. The pitch-preserving alternative then passed its complete
offline screen:1,357requests solved, no new rate-limited ticks, and retained
core target lift approximately100% on both rear feet. The associated backward
swing-X change peaks at6.45mm. The two-arm nominal physics comparison completed
at 02:35:41 PDT on September 6. Zero lift reproduced all 1,764 original baseline
samples exactly. The 5 mm treatment improved both rear clearances by 2.85/3.61 mm,
speed by 1.96%, and rear release by 5.38/2.62 percentage points, with no new failed
gate, no fall, and stable stopping. Front clearance fell by 2.44/1.34 mm but
remained above 5 mm. All five pre-existing numeric failures remain; this is a
useful mechanism, not success at the locomotion goal.
See the [complete measured results](../reviews/v44-actuator-20260905/REAR_PITCH_LIFT_PHYSICS_RESULTS.md)
and
[prospective physics protocol](../reviews/v44-actuator-20260905/REAR_PITCH_LIFT_PHYSICS_PROTOCOL.md).
The matched critic90 comparison is complete and closed without extension:
both100-update Isaac Lab runs finished at01:39PDT September6, and the final
matched evaluations finished by01:43:22PDT. Both candidates still fail the same
five original gates. Added causal critic context improved final normalized
V-to-GAE-target error by5.06%, but absolute error by only0.46%; this is not an
independent value-accuracy result or a practical gait fix. Final speed is
0.01020/0.01087m/s (zero/causal), both below65% of the0.01909m/s command.
Rear-right core clearance remains approximately zero. No falls and stable
stopping are retained. See the [complete final decision and all30 gates](../reviews/v44-actuator-20260905/CRITIC_CONTEXT_FINAL_RESULTS.md).

The [fixed-X rear-lift offline prerequisite](../reviews/v44-actuator-20260905/REAR_FIXED_X_LIFT_OFFLINE_PROTOCOL.md)
is complete and negative. Its failed orientation constraint motivated the
separate pitch-preserving calculation and current physics comparison.
Original source snapshots, references and qualification gates remain unchanged.
The preceding effort-observable pair completed its100-update allocation and all
evaluations; neither candidate qualifies and neither final checkpoint is extended.
The friction-zero and10×velocity-cap hold diagnostics are complete and do not
explain the force gap; their altered parameters are not used for training.
Native13's100-update training and both evaluations completed; its final model
is22.94% slower than Native12 and retains the same five failed gates. That
branch is closed without extension. Previous source snapshots remain immutable;
the next study uses a separately versioned source record. No candidate qualifies.
Detailed chronological evidence follows below.

### Projected learning pair selected and launched

The complete source workflow was reviewed by root and both independent source
reviewers before preparation. Source manifest
`50b5e67f3de95a2907ff4262a05f375054055a1fea91d63014257fda1db5ef52`
records 236 input files, 118 new snapshots and 79 package members. The physical
environment, configuration, projection and both contracts remain identical to
the completed initial pair; the new trainer observer and evaluation ancestry
binding are explicitly recorded.

Saved-state preparation completed at 03:53:16 PDT with no failure or cleanup.
Comparison `dc8f47f1a6662a18a45368790a80a04ef7df0b009338acb5cb66f40fcdedb741`
confirmed that both bootstraps preserve the original `f624e142...` actor,
critic, normalizer and iteration exactly. Only Adam moments and the selected
learning rate were reset. Bootstrap receipt is
`de9c56504ba85361f7bf0ad61a358ebc5860eb3b534697890c9a917d3e0fc02f`.

Root explicitly selected the reviewed pair and launched it once at 03:53:34 PDT.
The outer PID is 12488; central supervisors are 30104 (zero lift) and 34064
(5 mm lift). Experiment namespaces are `drobot_v44_ablation_pitch-zero-16` and
`drobot_v44_ablation_pitch-lift-17`. Each allocation is 50 additional updates,
labels 298–347, 128 environments, 192 transitions per rollout, seed 1701 and
fixed learning rate 0.00005. Both use the common feasible-action projection;
post-projection lift amplitude is their only environment difference.

The owned supervisor enforces 60 minutes per training arm and 90 minutes for
the whole pair, then exactly two bound nominal evaluations, without retries or
automatic extension. Published observer failures or mapping faults stop the
owned pair and retain evidence. Existing hardware limits and all 30 evaluation
gates remain unchanged. Launch receipt is
[run-start.json](../reviews/v44-actuator-20260905/rear-pitch-projected-training-pair-v1/run-start.json).
That attempt subsequently stopped on the numerical range rejection below;
it is not a completed or deployable model.
No unrelated tests, builds, browser or physical hardware checks were run.

### Projected learning interruption: captured quintic roundoff

The owned pair stopped at 04:05:27 PDT after 712.89 seconds. Completion SHA256 is
`e965441f384b3c04257dbfe0910b79a9c1af8eb2d6428e749c0c9536ef5927b2`.
Zero16 rejected a request in rollout index 33 after 53 captured transitions;
the physics counter remained 12778, before the rejected step. Environment 11,
rear-right, age 104 and phase 0.7750005125999451 requested common lift
`0.0050000000000000044 m`, exceeding the exact 5 mm upper bound by only
`4.336808689942018e-18 m`. The smooth-step helper bounds its input but its
expanded quintic output can round above one. This is not an identified
joint-domain infeasibility or physical fall.

The original strict range check detected the invalid numerical result. Both
owned training process trees were stopped; a subsequent read-only exact-PID
inventory found none of the outer, central, Kit or reported child PIDs alive.
The zero-side cleanup reported an already-exited Kit child; this is preserved
in the receipt, not relabeled as a clean completion. Both incomplete runs,
their observer files, checkpoints, failure data and source snapshots remain.
The complete-only analyzer and candidate-video path were not executed.

Next is a minimal correction bounding each quintic output to its mathematical
[0,1] range in both common and physical lift paths. The 0..5 mm guard, target
authority, physical limits and all original gates remain strict. New source
review and deterministic initial evidence precede a separately selected new
50-update pair from the original checkpoint; neither partial checkpoint is
resumed or treated as a favorable learning result. No deployment is authorized.

The correction is now implemented in environment source
`bb30c3ab12832b611e912f644c0757ff5eab59c0d4e357312da9a2f53da3d1b3`;
independent source comparison found only the two Q-output clamps and comments.
The new observer `rear_pitch_training_diagnostics_attempt02.py` changes only
its environment hash to that version (`72401a65840add5d24d2a17a649033c577892631fe1f26078ff13748408597ad`).
Trainer source now selects that observer; evaluator version dispatch remains
before simulator construction, hash-checks the binding and allows only the
two explicit study validators. Defaults, original-actor guard and gates are
unchanged. The corrected initial simulation subsequently launched as recorded below;
the new observer/trained-candidate route has not run yet.
See [captured failure and exact partial-update accounting](../reviews/v44-actuator-20260905/REAR_PITCH_PROJECTED_TRAINING_ATTEMPT01_FAILURE.md).

### Attempt02 initial comparison: actual launch

After complete root and peer source review, preparation verified the exact
two-block byte transformation from the preserved c26 environment snapshot to
bb30. The new initial manifest is
`1931b0c546cc60d492a34784cf7e72e73ce29cf3c89d75c82aebe78f4cf7f1fc`
(232 input files, 114 new source snapshots, 79 package members).
Root inspected its real source bridge and both commands, then launched once
at 04:21:51 PDT: outer PID 54184, zero-arm PowerShell PID 53448.
The selected [protocol](../reviews/v44-actuator-20260905/REAR_PITCH_Q_CLAMP_ATTEMPT02_PROTOCOL.md)
requires unchanged physical targets, trajectories, geometry and numeric gates;
only four named intermediate Q/lift leaves have predeclared tiny arithmetic
bounds, with every changed value retained. The initial comparison does not
automatically authorize or launch training; the separate selection is recorded below.

Both initial02 arms completed normally at 04:28:38 PDT, total 407.59 seconds,
with no cleanup or failure. Completion SHA256 is
`7a110de3d0bb61aab7f4724f1533cbdee0245f07a7ba6717bccea86612b57b54`.
Each compared all 3,528 rows and the full numeric report exactly to its prior
initial arm: zero changed intermediate leaves, zero projection changes and
no falls. Thus none of the four predeclared numerical allowances was used.
The same five mechanism criteria and front margins pass, and the same five
gait gates still fail. Identity receipts are `e55c9f70...` (zero) and
`fbcc9013...` (5 mm); mechanism receipt is `90eaf760...`.

### Attempt02 fresh learning: selected and launched

Root separately selected the reviewed 50-update comparison after reading
the successful initial02 results. Training manifest
`a192466ad823991796e1b8ea0dd6ce67d80eaab982a093ae1d5eb8a83aa97ed3`
contains 244 input files, 124 new snapshots and 79 package members, with the
exact initial02 source bridge. The unchanged saved-state helper/comparator
confirmed original f624 actor/critic/normalizer/iteration preservation in both
new bootstraps. Only Adam moments and the selected learning rate were reset.
Bootstrap receipt is
`9be11fa999d844f89e375145380e4d02481e8aab3d989b8c93917d21664a6e0b`;
comparison is `9e08e78bc714390d9159cc6adc6f09c691e19f53020de6f5cb55b19c56202c7c`.

The pair launched once at 04:30:19 PDT, outer PID 21336. Zero18 uses central
PID 30260 / Kit 46564 and run name `pz18_22cd6eae700a41e88278b97efda0f703`;
lift19 uses central PID 20676 / Kit 28508 and
`pl19_5f1fe2b956d7420aaac7a10c49f8d22b`. No partial16/17 checkpoint was used.
Run-start SHA is `9ba9941618de9aa057e8440543dca2f1a24d01a37cdea2924377d766c8e14208`.
The matched 128x192, seed1701, fixed LR0.00005, 50-update labels298–347,
per-arm/pair deadlines, two final canonical evaluations and no-regression
rules are unchanged. Launch does not establish completion or improvement.
The new saved-data analyzer and isolated-video workflow are source-reviewed
but unexecuted; both require actual completed-pair evidence before use.

### Common-F5 action projection: finite calculation complete, integration next

The [saved-data calculation](../reviews/v44-actuator-20260905/REAR_PITCH_PROJECTION_FEASIBILITY_RESULTS.md)
completed at approximately03:00 PDT in12.67 seconds. All5,428 active saved
mean proposals were retained exactly;44,270 sampled contexts had no empty
domain, and398,430 artificial float32 proposals passed the arithmetic screen.
Result SHA256 is `490aa8998278e3a32b8f19d843c81f76329fd66dc2ff4db7af3a83a4813e4fae`.
The artificial nine-point stencil changed8/9 proposals, sometimes by0.323rad;
that is not an observed policy intervention frequency. The finite grid is
explicitly not a nonemptiness proof over all continuous contexts. No PPO
selection follows automatically from it.

Root selected a vectorized Torch integration and then a bounded initial
zero-update simulator pair under the existing evaluation authority. Both
arms project into the same feasible region; only the subsequent0/5mm lift
differs. Raw policy actions and all existing limits/rewards/gates remain
unchanged. The projection is default-off and not exportable to hardware.
The [prospective initial protocol](../reviews/v44-actuator-20260905/REAR_PITCH_PROJECTED_INITIAL_PROTOCOL.md)
requires both complete new trajectories to match their respective old pair
exactly, zero changed deterministic mean targets, and individual front-foot
core clearance>=5mm. Source is being reviewed; this entry asserts no launch,
runtime parity or new training result. Invalid domains reject before physics
and retain partial evidence. No Pi command or deployment is included.

The initial pair subsequently passed source review by root and two peers.
Preparation completed at03:21:31 PDT, capturing229 hash-bound inputs,
112 new snapshots and79 package members, plus historical snapshots bound in
place. Prepared manifest SHA256:
`d11b6ec755fc842da7e87efbbf67ee412aa6dcc714e9e4a187f503a4f0f7527a`.
Supervisor source SHA256:
`131c42625c5296d8cf53e24d50f78b0dd7e6fda0199765b411c40e445c721371`.
The one selected invocation started at10:21:52 UTC /03:21:52 PDT, supervisor
PID46784 and zero-arm PowerShell PID33752. It retains the15-minute per-arm,
30-minute total and20-second owned-cleanup bounds. This launch entry is not
completion; both previous trajectories and all numeric report fields still
must compare exactly before selecting training. The
[50-update training design](../reviews/v44-actuator-20260905/REAR_PITCH_PROJECTED_TRAINING_DESIGN.md)
is only source preparation, with pre-reset projection telemetry and explicit
regression/continuation criteria. No new PPO invocation is asserted here.

The initial pair subsequently completed at10:28:37 UTC /03:28:37 PDT in
405.1289 seconds; both children exited0 and no cleanup intervention was needed.
[Completed results](../reviews/v44-actuator-20260905/REAR_PITCH_PROJECTED_INITIAL_RESULTS.md)
record all3,528 rows of each arm exactly equal to their previous trajectories,
zero projection changes, unchanged numeric report values and all30 gate
outcomes. Both front feet retain>=5mm; all five mechanism criteria pass.
Completion SHA256 is `4b15f5185913d2c703dd880b960308bf850fead938ebb984a619f7ccd179fc28`.
The same five gait failures remain. After this observation, root separately
chose the bounded50-update design, subject to final source/bootstrap review.
No continuous-context certificate or stochastic-learning result is claimed.

The new study observer is source-reviewed by root and two peers, frozen at
`0df408c19e7c54529c14575005cc4ef6d1c2e2a33508468eac8c2b6fc12c1b4f`.
It records before asynchronous reset, delegates original reward/process/update
calls once, and preserves ordinary termination causes and explicit failure
evidence. Root subsequently added opt-in trainer activation and exact trained-
checkpoint ancestry plumbing to the evaluator/wrapper. Those source changes
occurred AFTER the initial evaluation: they must be captured separately and
must not be represented as the original evaluated package bytes. The physical
environment, action mapper, config, loader and production contract remain
the exact initial-evaluation versions. No unrelated test/build or hardware
check was run; the new source paths await their selected training/evaluation.

### Rear pitch-preserving lift: actual preparation and launch

The source-only preparation completed at 02:29:58 PDT, binding 116 files,
111 native snapshots and all 78 package source/data members. The reviewed
[prepared manifest](../reviews/v44-actuator-20260905/rear-pitch-lift-pair-v1/prepared.json)
has SHA256 `20a510c7e5de62931b30f9653919353a1713fd800570dc8fa2260a37570cdc53`.
The supervisor source is `4a190916bbd9418ac88e0bebf2b920b9aa22d9b4a6da2dddafd5a1c0f0032af5`.
The original checkpoint, canonical URDF frames, simulator asset, references,
runtime dependencies and all five changed package sources are bound explicitly.

The single selected invocation started at 09:30:26 UTC / 02:30:26 PDT,
supervisor PID 28528, first-arm PowerShell PID 15004. Each arm has two nominal
episodes at seed 3101, command 0.01909090909090909 m/s, 25 s run plus the original
phase-alignment wait, 0.9 s cubic stop ramp and 3 s hold. The bounds are 15
minutes per arm, 30 minutes total and 20 seconds of owned-process cleanup.
No retry, extra training or deployment is part of this invocation.

This launch entry does not claim completion. Exact zero-branch trajectory
identity is required before treatment; requested-geometry rejection retains
partial evidence without invented completed steps, falls or numeric gates.
All 30 original gates and physical limits remain unchanged. The URDF skill
informed source-level joint-frame reasoning; no robot-description edits or
unrelated tests, builds, UI checks or hardware checks were run.

The comparison subsequently completed normally at 09:35:41 UTC / 02:35:41 PDT
in 315.14 seconds. Both arms exited 0 and retained 3,528 nonterminal samples,
two complete nonfallen episodes each. No process termination was needed.
The external completion SHA256 is
`a7a0e58c7116603dc6ce2847b50620e5d26a210ecb550a99593d716eb929399d`.
One saved-data-only analysis then exited 0 in 1.85 seconds, without reloading
the actor or advancing physics. Its [result JSON](../reviews/v44-actuator-20260905/rear-pitch-lift-pair-analysis-v1.json)
has SHA256 `46c58db88d4bf1397707bf0bca69b2d3f52ce554a4a0193b33f822879cb2d1e5`;
all five predeclared mechanism criteria pass. All 30 original gate outcomes
are retained, including the same five failures in both arms.

The next source-design step must make exploratory actions feasible without
increasing absolute joint/rate limits or catching an invalid mapping with a
silent fallback. Training with `O(reference + residual)` is not equivalent to
changing only the reference and then adding the old residual. The beneficial
fixed-actor witness is therefore not automatically a training-ready controller.
Front-foot clearance margins and coordinated support remain explicit concerns.
No new training invocation or deployment is claimed by this result.

The subsequent [20-second local progress video](../reviews/v44-actuator-20260905/rear-pitch-lift-video-v1/review/videos/play/rl-video-step-0.mp4)
completed at 02:52:51 PDT, observed exit 0, 117.18 seconds wall time, with no
owned-process termination. It uses the same original actor plus the 5 mm
simulation mapping, but the established illustrative playback protocol:
seed 1701, one nominal environment, 15 s run, 0.9 s stop ramp and remaining hold.
It is not the seed-3101 two-episode gate evaluation. Saved media is H.264,
1280 x 720, 60 FPS, 1,200 frames and exactly 20 seconds; SHA256
`ccb1202abd5da82f16979ca26d39b2fc658fdcd14c9dd16db12f4622665dc6a5`.
Source manifest `ef13ffbfe47b35d2d92676b3352e8a9875fe9a642630f00e176e78afd88b896b`
binds the reviewed recorder and completed pair. Frames at 5, 12 and 19 seconds
were decoded and inspected: upright robot, small forward advance, final upright
hold. These views do not establish contact forces, full-sole contact or running.
Playback-generated actor exports remain isolated, incomplete review artifacts
without the lift mapping; they are not deployment packages.

### Rear fixed-X lift: selected saved-data prerequisite

After the critic pair closed, one offline calculation was selected and started
around01:55PDT September6. It reads the original1,764-row Control09 episode;
it does not load the actor, advance physics or access hardware. The fixed
request is5mm times the recorded swing envelope on rear feet, preserving
requested tread-center body-X and abduction, with original residual/joint/rate
bounds. No additional amplitude or solver restart is selected after failure.

Reviewed calculator SHA256 is
`eece229952a85b56077540f3d9af78002e7a4516a9c617d7d0006cfb705875b1`;
prospective protocol SHA256 is
`e38cbf8bb8cf5adb71e9fc2be5f7c465877962cff263cdc03eef781ca2bc0249`.
Output is a new `rear-fixed-X-lift-feasibility-v1.json` in the study directory,
with a600-second finite budget. At selection, no result or subsequent physics
comparison is claimed. Failed local solves are retained, including bound,
branch and line-search evidence; they are not proof of universal infeasibility.
The URDF skill guided joint-frame/corner provenance. Simulator-captured shoe
corners, not the URDF legacy fork-tip sphere, define this calculation.

The calculation subsequently completed once, exit0, in21.51seconds. It processed
all1,764rows and reproduced identity delivery exactly, but failed61rear-left
tilt-guard rows out of1,357positive-lift opportunities. The candidate/fallback
sequence also introduces56new limited joint ticks. It is not selected for
physics. See [the complete negative result and next source-only direction](../reviews/v44-actuator-20260905/REAR_FIXED_X_LIFT_RESULTS.md).

### Critic-context pair: actual preparation and launch

Both reviewed bootstrap preparations exited0 at01:02:31PDT September6. Their
704,435-byte checkpoint files are identical (SHA256
`2e34aaf66730b902e0f9fc85400ea234c2c12b7ccfb5f9c5c12e5652b45b99cc`).
Original actor tensors, critic columns0–57 and normalizer history are retained;
the inherited normalizer count is7,372,800. Added critic columns are zero,
new-feature means are zero and variances one, and Adam moments are reset.
These are checkpoint transformations, not extra policy-forward evaluations.

Both trainers started at01:08:49PDT and logged loading their exact bootstraps
before actual learning label298. Observed run directories are:

- Zero-history control: `logs/rsl_rl/drobot_v44_ablation_context-zero-14/2026-09-06_01-08-59_czc14_aaedd907ee5c4dd1a0d488b4de6903e0`
- Causal-history treatment: `logs/rsl_rl/drobot_v44_ablation_context-causal-15/2026-09-06_01-08-59_ccc15_8d0de59272a34445936e386fc03a16dc`

Each receives100 additional updates,128 environments and192 steps per update.
The immutable [selection](../reviews/v44-actuator-20260905/critic-context-pair-selection-v1.json)
and [source manifest](../reviews/v44-actuator-20260905/critic-context-pair-source-pins-v1.json)
precede execution. Separate supervisor records retain actual process identity,
logs and eventual completion. The planned50/100-update evaluations and local
20-second causal347 progress clip have not completed at this launch entry.
Existing training computations supply the new critic diagnostics; bootstrapped
target agreement is not an independent measurement of true value accuracy.

The actual [training bindings](../reviews/v44-actuator-20260905/critic-context-pair-training-bindings-v1.json)
were captured after both exact load lines, saved configurations, observer
metadata and initial labels298–304 were visible. Their SHA256 is
`30d251d23dcd5b4913693aac93358b3648c79682dc47749a679517178224e7eb`.
The bounded watcher started at01:12:47PDT (worker33108), compared the full saved
agent/environment/command settings and accepted only the prescribed common
changes plus the treatment's context mode. It waits for matched347/397
publications, starts four evaluations in total and has no training-extension
or deployment path. [Actual watcher evidence](../reviews/v44-actuator-20260905/critic-context-pair-milestone-watcher-v1/prepared.json).

The [initial saved-data coverage note](../reviews/v44-actuator-20260905/CRITIC_CONTEXT_INITIAL_COVERAGE.md)
confirms that the added causal weights are learning: their Frobenius norm grows
from.0210 after the first update to.1637 after update7, while zero-control added
weights/inputs remain exactly zero.31/32 features vary once stop/hold is seen;
packet-hold history is correctly inactive in this nominal-only experiment.
The small early target-agreement difference is not evidence of better gait.
No extra model forward or simulation was used for this saved-data calculation.

### Critic-context halfway result: insufficient motion improvement

The matched50-update evaluations ran concurrently with training from01:23:01
and both completed successfully by01:30:00PDT. Each contains two full nominal
episodes with no falls and stable four-foot hold. Causal speed is.012270882m/s
versus.012165375 control (+.867%); these are64.28% versus63.72% of command.
Both still fail the same five original gates: speed, every-leg release,
core-corner clearance, contact-patch slip and approximate-PD soft-effort occupancy.
Causal worst-episode rear core p05 is2.245/−.214mm (RL/RR), versus1.960/−.210mm
control; the unchanged clearance requirement is5mm. Scheduled rear release is
64.43/58.94% causal versus64.43/58.78% control. Neither is a qualified gait.

The source-reviewed offline analyzer exited0 after checking all50 recorded
rollouts per arm and the continuous deterministic sample streams. For the
latest20 rollouts it pools337,330 active-run transitions per arm. Normalized
V/GAE-target RMSE changes.615740→.610283 (only.886% reduction), while absolute
error RMS slightly increases26.0254→26.2626 and target standard deviation also
increases42.2668→43.0334. This is not independent value accuracy or proof of
critic bias. Causal added-column norm reaches1.10082 versus zero-control0:
the inputs are used, but the motion/value improvement is not substantial.
[Complete m50 findings](../reviews/v44-actuator-20260905/CRITIC_CONTEXT_M50_RESULTS.md)
and [saved pooled comparison](../reviews/v44-actuator-20260905/critic-context-pair-m50-analysis-v1.json)
retain coverage, phase-specific results and all unchanged gates.

The already-selected100-update allocations continue unchanged. Source-only
fallback proposals address [rear-lift feasibility](../reviews/v44-actuator-20260905/REAR_CORNER_LIFT_WITNESS_PROPOSAL.md)
and [cycle-scale credit assignment](../reviews/v44-actuator-20260905/CYCLE_CREDIT_ASSIGNMENT_HYPOTHESIS.md);
neither subsequent simulation/training experiment is selected at this entry.

### Critic-context training complete; final evaluation in progress

Both supervised training processes exited0 without interruption or forced
termination at01:39:21(control)/01:39:23(causal)PDT. All100 labels298–397,
2,457,600 additional transitions per arm, exact loaded bootstrap and saved
configuration criteria are observed. Each observer published100 hashed gzip
records, restored its owned hooks, and reported no error, pending transition
or unpublished current rollout. This is a complete recording, not independent
evidence that the learned behavior is correct.

Final checkpoint397 hashes are
`d13aec2972081f2f71f03382f9010635412bbcb76f1c75ad1ee74cb3543f191d`
(control) and
`9361972de17d56c3309878f6a5fb494b3e63a1a900d431d696075bdf193ca509`
(causal). The watcher confirmed both final completion records and checkpoint
quiescence, then launched the two final nominal evaluations at01:39:41–42PDT.
No further training is started or allocation extended at this point.

Newest local progress clip: [causal critic90 model347,20seconds](../reviews/v44-actuator-20260905/critic-context-causal-m50-video-v1/review/videos/play/rl-video-step-0.mp4).
Recording ran01:23:30–01:26:59PDT September6 and exited0. It is20.000seconds,
H.264,1280×720,60FPS and1,200frames (1,006,167bytes; SHA256
`ead1817becf42260092af2f800eed9d763a651b0a9527a857a9cc8fd11f3886c`).
The exact copied causal347 checkpoint hash is
`98ca8413d73332ee06efa628c2e356a6a58fa4a24290feb7906ac22c5accd05c`.
Frames at5/12/19seconds were decoded and inspected: small forward advance and
an upright final stance are visible, but these still images do not establish
smoothness, adequate foot clearance or successful running. Playback is separate
from the numerical25-second/two-episode evaluations. Isolated playback exports
remain unqualified review files only. No training settings changed for filming.

The earlier [native-effort347 clip](../reviews/v44-actuator-20260905/effort-native-m50-video-v1/review/videos/play/rl-video-step-0.mp4)
is preserved. That unqualified50-update checkpoint moves slowly and drags the rear feet;
[matched findings](../reviews/v44-actuator-20260905/EFFORT_PAIR_M50_CONTROLLER_FINDINGS.md)
retain the five failed gates. Older clips below are historical evidence.

Two actual Isaac Lab PPO runs completed their fixed200-additional-update
allocation: fresh07 from scratch and warm08 from the earlier common-floor199.
All matched50/100/200-update evaluations completed successfully at19:47:33.
Neither final policy qualifies. Fresh final199 moves .009222m/s against
.019091m/s commanded, remains straight and holds stably after stopping, but
still fails speed, rear release/clearance, slip and approximate effort gates.
Warm final398 regresses to .005736m/s, about80degrees final heading error and
unstable contact during hold; reject that continuation.

- [Final200-update numerical comparison](../reviews/v44-actuator-20260905/freshwarm-m200-comparison-v1.md)
- [20-second final fresh-policy progress video](../reviews/v44-actuator-20260905/fresh07-final200-progress-video-v1/videos/play/rl-video-step-0.mp4)
- [Matched evaluation records](../reviews/v44-actuator-20260905/freshwarm-milestone-watcher-v1/)

The immutable initialization pair is complete; source/config records and all
rejected outcomes are retained. The subsequent height-only versus unchanged
reward pair trained from fresh final199, after one documented pre-Kit
launch-path correction. Its completed comparison is described below. Only the
height reward differs; no history-policy, actuator-gain or smoothness change is
mixed in. This is still a low-speed contact-control investigation, not a
demonstrated running policy.

At20:23:58, the source-reviewed bounded milestone watcher started (launcher
PID35044; actual Python worker40408, as recorded by its own preparation).
It compared only the exact248/298/398 checkpoint pairs, two nominal
episodes per arm, using the frozen arm-specific overrides and unchanged gates.
It launched six evaluations, did not terminate either trainer, and did not
export or deploy. Intermediate reports are not training completion or
qualification. [Watcher records](../reviews/v44-actuator-20260905/height-pair-milestone-watcher-v1/)
and [protocol](../reviews/v44-actuator-20260905/HEIGHT_PAIR_MILESTONE_WATCHER.md)
preserve commands, identities, checkpoint readiness and source/config hashes.
The50-update evaluations actually launched at20:24:10 (Kit57324/27712), after
later training label302 and more than10seconds of unchanged checkpoint bytes.
Watcher source SHA256 is
`80ce052a99df609ca4c3d655ce7c402b7e0ee2e72164271f609f70b1931dc42c`.
The50/100-update pairs have now completed; neither shows a useful height-reward
advantage. Both still fail the same five motion/load gates and retain passing
stop/hold behavior. [Milestone results](../reviews/v44-actuator-20260905/HEIGHT_PAIR_MILESTONE_RESULTS.md).
Both fixed200-update training allocations completed successfully at20:48:43;
all three milestone pairs/six evaluations completed at20:52:34. The height
treatment is rejected as unsupported by this bounded experiment: both final
models move only about.009m/s and fail the same five gates, with worse rear
clearance than earlier checkpoints. No automatic extension or weight retuning.
The separate raw-contact diagnostic exited0 with600 expected samples; its
offline coverage/contact-motion audit and independent math review are complete.
A fresh20-second local control398 video also completed. No model is qualified
or deployed, and neither fixed training allocation is being extended blindly.

The recording completed successfully at20:56:13:
[latest20-second simulation clip](../reviews/v44-actuator-20260905/control09-final200-progress-video-v1/videos/play/rl-video-step-0.mp4).
It illustrates the remaining failure, not a qualified running gait. FFprobe
reports20.000s, H.264,1280x720,60fps,978256bytes. Video SHA256
`0a9e5149c665e3ed46e2a18f33aa5f04e482caf5ebd13712e6648b737d7978e2`;
the isolated playback checkpoint matches final control398 exactly. Five local
frames were extracted; motion/hold frames02/05 were visually inspected. No
hosted review or Pi update occurred.

### Completed contact evidence and next priority

[Raw-contact findings](../reviews/v44-actuator-20260905/raw-normal-control298-v1/CONTACT_FINDINGS.md)
and an [independent frame/timing review](../reviews/v44-actuator-20260905/RAW_CONTACT_MATH_PEER_REVIEW.md)
confirm600 complete120Hz samples:480 late-run and120 final-hold. All7800
sensor/frame normal-force reconstructions agree with the independent net forces
within2.057e-6N. The diagnostic reproduces the previous control298 episode0
actor inputs/outputs, joint positions, delivered targets and root poses exactly
across1764 rows. That observed agreement is specific to this run, not a general
guarantee that reporting changes cannot affect a rollout.

Both rear feet load mainly near their **trailing** tread edges during the late
run (loaded mean COP approximately47.3/44.3mm behind tread center). The right
rear also has small grazing forces during core swing, below the existing1N
release threshold. Force release therefore does not establish adequate clearance.
These observations concern this simulated V44 checkpoint, not a reinterpretation
of the user's earlier real front-foot video. Sparse solver contact points do not
measure physical contact area.

On matched loaded material points, reported planar velocity RMS is18.7–27.4mm/s
in the late run, while consecutive120Hz pose-derived motion is1.15–3.21mm/s.
The same distinction persists in hold. COM frames, mirrored foot coordinates,
first-frame weighting and sampling timing were independently checked. Instantaneous
solver velocities and interval-average pose differences are different quantities;
this is not proof of a solver bug, zero actual sliding or grounds to loosen gates.

The [saved-data target-smoothness scale audit](../reviews/v44-actuator-20260905/TARGET_SPACE_SMOOTHNESS_SCALE_AUDIT.md)
also rejects treating a proposed delivered-target acceleration cost as a simple
rescaling of raw action second differences. Existing floor-table lookup already
contributes substantial target curvature and is shared by simulator and Pi.
Interpolation is a distinct, unimplemented control-contract hypothesis. A new
smoothness penalty could encourage further cancellation of useful rear lift;
its provisional40rad/s²/weight3 values are not calibrated or selected.

**Next bounded priority:** prepare a separate reporting-only filtered-friction
capture, preserving the original unfiltered sensor, controller, physics and
checkpoint. Actual normal plus tangential contact wrenches during final hold can
support a qualified quasi-static required-torque comparison with the approximate
implicit-PD estimate. It cannot by itself measure motor current, drive-only
torque, hardware gains or dynamic motor capacity. This measurement is prioritized
because current sampled-training effort costs (about−6.29 RMS and−3.74 soft-limit)
are much larger than the slip costs (about−.003 combined); scalar means do not
establish gradient importance, but they caution against assuming a slip-only
reward change resolves the main learning pressure. No new training arm is
selected yet.

The [normal+friction capture source](../reviews/v44-actuator-20260905/NORMAL_FRICTION_CAPTURE_IMPLEMENTATION.md)
was read by root and independently reviewed before launch at21:21:41 PDT
(2026-09-06T04:21:41.7151399Z). Retained external cmd process21844/session47016
has a900-second bound and may terminate only its owned process tree on timeout.
Output: `normal-friction-control298-v1`; expected480 late-run plus120 hold rows.
Helper SHA256 `68fdcffbca0e614c4aef16c557e88cdc124ccb10ae9ec723c682f48a203fabc9`;
launcher SHA256 `2b45049c3aec96a384cf6160a031f4fb756c95df3785251b42e26bb63bfcc40f`.
It preserves the original unfiltered sensor and adds four one-foot/ground
reporting views, separately indexed normal and friction arrays, actual body
mass/inertia readbacks and configured disturbance buffers. Launch/startup is
not completion or qualification. Static torque analysis follows the documented
[conditional method](../reviews/v44-actuator-20260905/STATIC_CONTACT_WRENCH_TORQUE_METHOD.md),
including unresolved friction-sign and inertial uncertainty, rather than choosing
a sign that matches PD telemetry.

The external process subsequently exited0 at21:22:48 PDT
(2026-09-06T04:22:48.3400181Z), without timeout or forced termination. Its
[external exit record](../reviews/v44-actuator-20260905/normal-friction-control298-v1/external-process-exit.json)
is retained beside23MB of contact data, the report, full trace and source snapshots.
Initial header inspection reports patch friction,600 captured intervals, no
filtered getter errors, nonempty friction anchors on all four feet throughout
hold, successful body mass/inertia/gravity readbacks, restored observer and no
lifecycle errors. Independent per-row provenance, coverage and trajectory audits
are still required before interpreting joint loads; these header fields alone
are not that audit. No simulation/training job remains running at this point.

The independent [retained-data audit v2](../reviews/v44-actuator-20260905/normal-friction-control298-v1/contact-integrity-v2.json)
completed at04:30:54 UTC, before the package was released for further source work.
All snapshot/current launch-input hashes matched. All1764 selected trace rows
(actor inputs/actions, q/qdot, requested/delivered/filtered targets, approximate
torque and root pose/velocities) match the prior normal capture exactly. All600
raw body poses/COM velocities and resolved-partner normal-contact multisets also
match; native actor IDs changed but their resolved path did not. All2400 foot-
frame normal multisets match, with at most12 normal/6 friction records per128-slot
view and no positive-load empty friction case at the stated1e-6N audit threshold.
That availability threshold is not a changed1N support/qualification threshold.

One auxiliary measurement is incomplete: the recorder expected a3-vector but
the configured disturbance buffers have shape[1,3] after selecting an environment.
All600 frames preserve null values and explicit read errors for those two fields.
Pinned nominal-source logic implies zero configured disturbances, but this is
**not a successful external-force readback**. V1's stricter all-fields failure is
retained; v2 explicitly scopes its pass to contact/trajectory/provenance integrity.
Static arithmetic must retain a no-other-external-wrench assumption, not silently
substitute measured zeros. The recorded contact forces and rollout remain usable.

While that conditional torque calculation is prepared, an opt-in phase-table
interpolation implementation is now source-authorized, with existing floor lookup
remaining the default and production export guarding the experimental mode.
[Prospective plan v2](../reviews/v44-actuator-20260905/PHASE_INTERPOLATION_IMPLEMENTATION_PLAN_V2.md)
preserves the original plan and states mechanism-screen tolerances before any
interpolated rollout. This is not a selected training arm, Pi change, or claim
that interpolation fixes motor loading. No new probe has launched yet.

### Conditional static torque result and measurement follow-up

The [final-hold wrench calculation](../reviews/v44-actuator-20260905/STATIC_CONTACT_WRENCH_RESULTS.md)
completed on the120 saved hold intervals. Canonical JSON SHA256 is
`1101b1d1f994e1890180ea98ec44188507c847af17fb4d609b277b16307bcf23`.
After a preserved failed parser attempt, the source explicitly accounts for the
three fixed sensor frames fused into the13 captured bodies, without double-
counting their mass. Parent/child joint anchors agree within0.238micrometres.
All conditional contact calculations retain unknown external/passive terms and
both native/reversed friction interpretations; neither is selected by PD fit.

Under the native friction orientation, mean knee requirements FL/RL/FR/RR are
approximately+.0324/−.0005/+.0242/−.1307Nm, versus retained PD estimates
+1.0244/−1.1541/+.8654/−1.9050Nm. This is a material discrepancy, not established
spare motor capacity. Native whole-body moment residual RMS is.000655Nm versus
.020697Nm for reversed friction, but external wrench and sign assumptions
remain unresolved. The hold has nearly constant positions/targets; pose-derived
inertia proxies are small but are not certified physical uncertainty bounds.
PD is computed before the second physics solve and captured afterward, whereas
contact poses are post-solve. It is not a synchronized measured drive-torque pair.

The [drive-mode artifact audit](../reviews/v44-actuator-20260905/DRIVE_MODE_AUDIT.md)
rules against one specific hypothesis: all12 revolute joints in the exact
referenced USD explicitly author force drives, not acceleration drives. Stored
angular gains use per-degree units and convert consistently to the configured
per-radian30/4.58 values. The USD matches its previously retained snapshot; this
is read-only artifact/source evidence, not a retroactively captured runtime
drive-type measurement. No gains, force limits or effort gates changed.

Consequently, no new effort-reward interpretation or longer PPO run is selected.
A separate [solver-iteration sensitivity protocol](../reviews/v44-actuator-20260905/SOLVER_VELOCITY_ITERATION_PAIR_PROTOCOL.md)
has been source-reviewed and selected for isolated launcher preparation to
investigate constrained velocity/effort semantics. It compares only requested
8/4 versus 8/1 position/velocity iterations, not presumed better physics; actual
executed island counts are not available through the inspected getter API.
The future capture must distinguish composed configuration from execution,
record live force-drive types, and repair the auxiliary shape readback only in
new diagnostic source. No solver experiment has launched yet.
Independently, root and a peer read
the five-file opt-in interpolation change and selected the four bounded
floor/linear × zero/fixed-policy mechanism probes from the prospective v2 plan.
Their launcher/configs are being prepared; they are not a substitute for resolving
the torque discrepancy and have not launched at this point. Production/Pi remains
floor lookup, and nondefault export is blocked. No unrelated tests/builds ran.

The four-probe supervisor subsequently launched at **21:56:15 PDT**
(2026-09-06T04:56:15.758894Z), external Python PID 46236/session 19103.
Its first paired zero-residual wrappers are 29672 and 20584. The reviewed
supervisor is `230c24d20d5a0fdc89c52b0c7f076b0eca0538d4147a5cbd4f8e69e83908a7dc`;
the [prepared manifest](../reviews/v44-actuator-20260905/phase-lookup-probes-v1/prepared.json)
is `3aa7a08219f5ebbea2d6a077bb3965ddedf9ed453854c90ef565ba14a0f63c33`.
It retains 53 source/input snapshots and runs two zero cases, then two fixed
control298 cases, with a 900-second total limit and at most 20 seconds for
owned-process cleanup. No retries or further training are implicit. The
prospective 30% acceleration / 95% speed / rear release and clearance / no newly
failed gate screen remains unchanged. Launch is not a completed result.

**Completed at 22:02:09 PDT:** all four probes exited zero in 353.26 seconds,
without timeout or forced cleanup. The [full result](../reviews/v44-actuator-20260905/PHASE_LOOKUP_PROBE_RESULTS.md)
is mixed and fails the prospective overall screen. Target acceleration RMS
falls 70.38% without corrections and 32.03% with the fixed policy, but that
policy's right-rear release falls 2.3883 percentage points, exceeding the
unchanged 2-point tolerance. Speed is slightly higher and no previously passing
official gate newly fails; neither overrides the failed screen. Do not promote
interpolation into the proposed next training stage or widen the criterion.
The new floor episode exactly reproduces all common old episode-zero trajectory
fields, apart from reset IDs. No qualified gait or physical-load identification
results. The independent solver diagnostic is the next selected priority.

For context, NVIDIA's [PhysX 5.7 solver documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/Simulation.html#constraint-solver)
recommends one velocity iteration as a default and explains why constrained
reported velocities need not equal pose finite differences. This supports the
bounded comparison, not a prediction that its result will improve this robot.
The installed binary's exact SDK minor is still unconfirmed; source documentation
does not replace the forthcoming composed-setting and numerical readbacks.

### Solver-pair launch transport attempt 1

Root selected and launched the reviewed solver-pair supervisor at 22:14:55 PDT
(05:14:55 UTC). The first cmd wrapper, PID 9620, exited 1 after about two seconds,
before the Python launcher created an arm directory or Isaac emitted a startup
banner. Stderr contains only `The syntax of the command is incorrect.` The
[failed supervisor record](../reviews/v44-actuator-20260905/solver-iteration-pair-v1/completion.json),
logs, command and external exit are preserved; neither solver evaluation ran.
No process remained for forced cleanup.

The source-level transport diagnosis points to a quoted `PYTHONEXE` environment
override being inserted into the batch file's own quoted conditional. This is
not a reproduced shell test. Root selected a new, separately recorded transport
attempt that removes the override and uses the already-existing pinned `kit.exe`
default, matching earlier successful launches. Original sources and manifest
`3c8b1b30c529541a83041fabf43dab44bb6bc2b24018e0765c4ffa1b0c494bd4`
remain unchanged. The new attempt does not change either arm's physics,
checkpoint, observer, gates, finite allocation or deployment status.

Transport attempt 2 also exited 1 before Isaac/launcher startup at
22:18:56–22:18:58 PDT (PID 20824), with the same cmd syntax error. Its
[separate record](../reviews/v44-actuator-20260905/solver-iteration-pair-attempt02-v1/completion.json)
and original input manifest
`8f00a9d621e69f92d0a2237ac852b60ea2f61032f29661bd3d4f9ecf7623b6eb`
are preserved. Removing the quoted environment override was **not sufficient**;
do not present that earlier source hypothesis as a proven root cause.
Root selected transport attempt 3 using a PowerShell `-File` wrapper matching
the successfully completed phase-probe launch path, instead of the failing
direct cmd invocation. The two physical captures still have not run, and their
finite protocol/observer/checkpoint remain unchanged. No standalone shell
smoke test or unrelated verification was introduced.

Attempt 3 launched successfully at **22:23:03 PDT** (05:23:03.928868 UTC),
PowerShell wrapper PID 6584, external supervisor session 29549. Isaac Lab/Kit
startup is observed, unlike the two failed cmd transports. This still is only
startup evidence, not completed capture or model qualification. The
[attempt-3 pin manifest](../reviews/v44-actuator-20260905/solver-pair-source-pins-attempt03-v1.json)
is `2176831b7bab9a25015e230305409504883a4224a5ee7bb0a732a830736ee22e`;
observer remains `cfad3106bf502ad3b46d7750a7b8b8f815c07c1bfce8ec63da31da2017d9dfde`.
Both original failed transports and all source versions are retained. The
supervisor will run 8/1 only after the new 8/4 capture passes its unchanged
baseline/readback integrity requirements.

### Solver pair completed; effort-channel isolation is the next priority

The attempt-3 external supervisor completed both captures at **22:25:24 PDT**
(2026-09-06T05:25:24.317576 UTC), exit0, with no timeout or forced cleanup.
Both saved-data audits pass: 480 late-run and120 final-hold120Hz intervals,
zero integrity/readback errors, and600 measured zero auxiliary force/torque
buffers. The new8/4 observer reproduces all1,764 selected historical trajectory
rows and all600 common raw pose/normal/friction rows exactly. The8/1 trajectory
differs as expected for changed closed-loop physics. Neither candidate qualifies;
changing velocity-iteration requests does not remove the five main failures.
These are composed requests, not measured internal solver-island counts.

[8/4 readback analysis](../reviews/v44-actuator-20260905/solver84-control298-v1/readback-analysis-v1.json)
has SHA `ae7a7645d516e9f914ec7e7b44ab413880e009071c1cd32e7737850cf11d8080`;
[8/1 analysis](../reviews/v44-actuator-20260905/solver81-control298-v1/readback-analysis-v1.json)
has SHA `474e882f478bdfd64c50107e55285375af4e91a274b5476d4d1d9e0def0135da`.
Native force-drive types, K30/B4.58366,2.647795Nm limits, zero user effort and
zero **new three-parameter** joint-friction properties are observed in both arms.
These do not establish zero legacy joint friction; see the subsequent finding.
Post-solve incoming
joint-force projection is retained separately from the pre-solve PD estimate.

For the8/4 right-rear knee hold, projected incoming mean is−.130680070Nm;
independent conditional native-friction inverse statics gives−.130680145Nm,
while retained pre-solve PD is−1.905011739Nm. Native and cached position targets
agree exactly and the offline PD reconstruction agrees within float rounding.
The other knees show the same substantial distinction. Solver8/1 retains it.
This is not a target-delivery error or evidence that motor current is measured.

The next bounded priority is to finish free-axis wrench/frame and limit/clamp
isolation using these retained records, then decide whether an **effort-signal-only
reward comparison** is justified. The present reward materially penalizes the
approximate PD observable; the new evidence cautions against treating it as
actual drive load. No reward/gate/physics change is made by this finding, and no
additional PPO job has started. Safety caps, production behavior and Pi remain
unchanged. Historical results and both failed transport attempts are preserved.

### Specific legacy-friction gap found before changing the effort reward

The independent frame audit reproduces each native projected torque from the
recorded incoming child-joint-frame wrench and exact USD joint mapping within
1.19e−7Nm. All12 USD revolute axes use local+Z; mirrored signs reside in their
frame rotations. Reported endpoint joint positions/velocities are comfortably
inside their limits, but this alone cannot exclude constraint impulses inside
TGS subiterations.

A more immediate source gap was then found: the pinned USD authors legacy
`physxJoint:jointFriction` approximately.02 on all12 joints, while our observed
getter and current Isaac Lab setter concern the **new** static/dynamic/viscous
triplet. [PhysX5.8 joint-friction documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Articulations.html#articulation-joint-friction)
describes a legacy fallback when those new parameters are zero. The installed
native SDK minor is still unconfirmed, and authored USD is not live readback.
Therefore earlier shorthand saying all joint friction was zero is too broad;
only the measured new triplet was zero. Preserve old artifacts rather than
retroactively replacing their missing legacy measurement.

Root selected **one additional passive8/4 capture**, same checkpoint/protocol,
adding the existing `get_dof_friction_coefficients()` readback at construction,
late run and final hold. It changes no physical parameter and must reproduce
the current8/4 baseline. Launcher/observer source is being prepared; no capture
has started at this point. A velocity-cap sensitivity experiment and the
conditional effort-reward pair are deferred until this concrete question is
resolved. No hardware, training limit or qualification gate is relaxed.

**Passive readback completed at22:44:22 PDT:** root launched the source-reviewed
observer at22:43:11 (owned PowerShell53264, session57113), and its external
supervisor observed exit0 without cleanup. All12 live legacy coefficients equal
`.019999999552965164` at construction, late run and final hold, while every new
friction triplet remains zero. The current8/4 actor/target/state trace and raw
poses/COM velocities/net normals reproduce exactly. This confirms nonzero legacy
friction is present; it does not yet identify the force discrepancy's cause.

[Passive completion](../reviews/v44-actuator-20260905/legacy-friction-capture-supervisor-v1/completion.json)
and [capture](../reviews/v44-actuator-20260905/legacy-friction-control298-v1/normal-friction-contacts120.jsonl)
are preserved. Raw SHA is
`f4a88666ed980890608ba8dea107c8e017030c8bfaea33ef1c50156609903267`;
trace SHA is `981f90c69567dfab3d085b603eeba5408214f4732e5f748631b22b8598999a20`.
The150-snapshot manifest (149 files plus itself) is
`ae252210ccd12e622063f64c563b5a7e94b30615b4614f44d46f52d3b1b12d77`.

Root next selected a **single hold-only legacy-zero sensitivity capture** with
the entire original motion/prefix unchanged. Only the legacy coefficient will
switch to zero at the first final-hold capture boundary; force/velocity/position
limits, solver, targets, other friction fields and rewards stay fixed. The
prospective protocol/source is being prepared and must retain matching prefixes,
initial hold state, before/after coefficient readbacks and unchanged parked
targets. This tests one mechanism; it neither promotes a frictionless hardware
model nor selects a new training reward. No PPO, Pi change or deployment occurs
in this comparison.

Read-only configuration tracing also confirms that `ImplicitActuatorCfg` in the
frozen environment explicitly requests `friction=0.0` (cfg line245). Installed
Isaac Lab applies that through its new-friction-property writers (backend
lines4614–4627), not the deprecated scalar setter. Thus the observed legacy.02
is a concrete mismatch with this nominal actuator-friction intent, not a newly
requested physical friction value. Its effect still requires the isolated
comparison; removing a simulator fallback does not measure real gear friction.

The hold-only intervention actually launched at **22:54:55 PDT**, owned
PowerShell53176/session89686, after root source review and independent peer
review. Its [162-file source pin manifest](../reviews/v44-actuator-20260905/legacy-hold-zero-source-pins-v1.json)
is `a6cd452eaf277755bf178d86e5953f0ea3ab2a9cc3be10cb8f7ed63d34a8fd08`;
observer SHA is `676440e3d0dcc43ca1378143e6bfeb147cbb48f242d9a428d0d6251ad440bd6c`.
The [prospective protocol](../reviews/v44-actuator-20260905/LEGACY_ZERO_HOLD_SENSITIVITY_PROTOCOL.md)
is `62725a1b68c93fdef6f9f6514cb70730b1f3435de43fcc4dfe3b4150c7a22bb8`.
It separately reports the first.1s transient and final.5s tail. A75% reduction
in each pooled PD-minus-projected and reported-qdot-minus-pose-FD RMS gap is a
predeclared descriptive screen, not a walking/safety qualification. Completion
and retained-data interpretation remain pending at this entry.

**Completed at22:56:07.967 PDT, exit0 without forced cleanup.** All1,704
preceding control rows,480 late raw rows, first hold pre-solve state and parked
targets match the passive baseline. The sole switch is recorded atcontrol1704,
physics counter3409; all12 legacy coefficients read back zero immediately and
at output. [Independent saved-data analysis](../reviews/v44-actuator-20260905/LEGACY_HOLD_ZERO_INDEPENDENT_CAUSAL_NOTE.md)
rejects the proposed friction explanation for this hold: final.5s pooled force
gap is0.786537245→0.786537919Nm RMS and velocity gap is
0.0319056104→0.0319058249rad/s RMS. Neither75% reduction screen passes.
Maximum paired root displacement is1.897e-7m, joint difference1.669e-6rad and
mean foot normal-load change3.409e-5N; a large support redistribution does not
conceal improvement. Live parameter presence alone did not establish a
material active contribution. Raw capture SHA is
`fa1e526f5c99f5ce99ba257ebf2a1b43a91c5cf2bcbfb0009fa031b83011ea26`.

Root selected one remaining finite force-mechanism exclusion: raise only the
native simulation velocity cap10× during the final captured hold, preserving
the complete passive prefix and all other parameters. Source preparation and
review are underway; it has not launched at this entry. Endpoint velocities
alone cannot exclude internal solver clamp impulses. This diagnostic cap will
not enter training, production or hardware. After its result, select a finite
training intervention rather than extending the attribution search indefinitely.

The cap-only capture has now launched through the reviewed owned supervisor
(PowerShell52964, execution19047). Its168-file manifest SHA is
`733eee7ca766c00cd0d96c9495aeda088c21f33748d16d3e2a0c07c10f267cb4`;
[protocol](../reviews/v44-actuator-20260905/VELOCITY_CAP_HOLD_SENSITIVITY_PROTOCOL.md)
SHA is `13ce1912e54652b3b65cb99ac45971a724dab3b121f29d549ff4fa5b1c101e36`.
It uses the completed passive legacy baseline, not the zero-friction arm.
Only one native max-velocity setter is permitted at the original final-hold
boundary. Root and an independent reviewer read the full new source before
launch. The fixed process bound is900s; completion/results pending here.

**Cap diagnostic completed23:11:10.847PDT, exit0/unforced.** The actual new
native cap readback is45.83662414550781rad/s; legacy friction remains.02.
The [independent result](../reviews/v44-actuator-20260905/VELOCITY_CAP_HOLD_INDEPENDENT_RESULT.md)
finds both final-tail gaps **exactly unchanged**,0% reduction, and all1,764
complete common trace rows/600 raw rows identical after only UUID/contact-ID
normalization and removal of the new intervention marker. Thus this bounded
clamp explanation is closed for the captured hold; no additional attribution
sweep is selected. The high diagnostic cap is discarded with that process.

### Selected next actual training: effort-observable pair

Root selected a100-additional-update matched PPO experiment from the original
control298 checkpoint, with the original physical plant/caps/legacy friction.
Control keeps retained approximate-PD reward inputs; treatment substitutes
reported active generalized joint effort **only** into the existing effort
RMS, excess, torque-speed and inter-leg operating-point costs. Both arms collect
the same fresh native readout and retain both channels. This is an objective-
sensitivity experiment, not a claim of isolated drive/current measurement.

The source implementation is underway with defaults off and explicit V44-only
ablation markers/export rejection. All old PD-based evaluation gates, target
rates, packet/force/velocity limits, observations, references, other rewards
and stopping logic remain unchanged. Allocation is128×192 transitions/update,
seed1701, LR5e-5, entropy.002, gamma.995/lambda.95, identically reset Adam.
Matched zero/50/100-update evaluations use checkpoints298/347/397 and unchanged
nominal motion/stop criteria. No automatic extension or policy promotion follows
lower reward cost alone. Bootstrap preparation/training have not launched at
this entry; all historical source versions and failed outcomes are preserved.

At23:28:18.649PDT the actual zero-additional-update evaluation pair launched
(owned PowerShell31376/24492, execution48704). The reviewed100-file/32-role
manifest is [effort-pair-source-pins-v1.json](../reviews/v44-actuator-20260905/effort-pair-source-pins-v1.json),
SHA `4a4d68f5deb98cbf0cf63fe7158a3a11895b371adc07fb53bc8d1053f15afbf7`.
All78 package members and the actual relevant backend/API/runtime sources are
bound;95 byte-preserving source/config/reference/asset snapshots are retained.
Both baseline processes use the original deterministic actor, not zero actions.
The supervisor requires complete no-fall two-episode run/ramp/hold coverage and
equal physical trace fields/gates before either training wrapper can proceed.

Both new bootstrap preparations completed successfully at23:28:45PDT. They
preserve source actor/critic/normalization and iteration298, reset Adam only,
and set LR5e-5. Control bootstrap SHA is
`8766ef2f47c3f98dfa01721f66372bd221dfc228383fd16a230b7f869d8e0534`;
native bootstrap SHA is
`cca79f49bef21090ec0b9caa04e1ede4eee97b6974f8598bccf1aca1dab1dc92`.
Timestamped provenance differs, so the complete checkpoint files are not
claimed byte-identical. No actor forward pass occurred during conversion.

**Baseline completed23:31:40.528PDT, both exits0.** All3,528 recorded control
steps (two complete29.4-second episodes) match exactly between arms except
reset UUID and the intentional reward-observable label. All30 original gates
also match, including the five existing failures. This establishes comparable
starting behavior, not a successful gait. An independent saved-data audit also
matches all1,764 historical common rows from the prior passive capture and
finds exactly0Nm native-channel difference at300 aligned final-substep samples.
See [force-channel findings](../reviews/v44-actuator-20260905/EFFORT_BASELINE_FORCE_CHANNEL_FINDINGS.md).

**Actual training launched23:32:32PDT.** Both runs logged timestamp
`2026-09-05_23-32-39` and loaded their reviewed iteration298 bootstrap:

- Control: `drobot_v44_ablation_effort-control-11`, run
  `ectl11_13823a4069044a349f651da9ed28a811`, Kit36572, execution80109.
- Native-effort treatment: `drobot_v44_ablation_effort-native-12`, run
  `enet12_3261c6de50b8465890780d0d7aedd146`, Kit5928, execution59157.

At23:40:06.734PDT, the reviewed finite watcher started (Python31840,
execution52120). Its90-minute deadline and four evaluations cover exact
checkpoints347/397 only. Intermediate checkpoint347 requires a later iteration
and stable bytes; final397 requires successful external trainer completion.
It never terminates either trainer and does not extend either100-update budget.
Actual run bindings are in
[effort-pair-training-bindings-v1.json](../reviews/v44-actuator-20260905/effort-pair-training-bindings-v1.json)
(SHA `6b3383be9e76a0d75b5105828266fa6927f04f5c1b8ddb6764299209db5e2346`);
watcher source SHA is
`25f623aec793695d63daa0efa0b4c17ea4f5c043bac3dad9ab05ba77f6e2e0d9`.
The frozen100 inputs remain unchanged. No model is qualified or deployed.

The50-update pair launched at23:45:40/41PDT (PowerShell47176/28564), after
both trainers logged label349 and checkpoint347 was stable for at least10s.
Control347 SHA is
`8898a015bc0ef4160de30e51fff66bafed974e857c8e110aaa00f9eaf64df259`;
native347 SHA is
`af21ae8c1c9577713722eefe029e223c78c8315ebdb1d8e5aca2cde96cc979c4`.
The separate20-second native347 recording is also running through the reviewed
isolated-copy route (PowerShell42992, execution65007), with a12-minute bound
and20-second owned-cleanup allowance. Its15-second run/seed1701 playback is
illustrative, not the25-second/seed3101 numerical protocol. Any automatic
playback exports remain unqualified artifacts inside the new local review
directory; they are not deployment packages. Recording source SHA is
`dea968aad9f45d641d6d7d3ce0c1370a224d974b83e0b22969f8e48b5cfb9346`;
separate video source-pins SHA is
`330c0f85df82531f80ea0aac7a7b939ca8a1deb567f3aab30de57aab58962ab8`.

**Native347 progress recording completed23:50:08.383PDT**, exit0, with no
forced cleanup. The [new local20-second clip](../reviews/v44-actuator-20260905/effort-native-m50-video-v1/review/videos/play/rl-video-step-0.mp4)
is H.264,1280×720,60fps/1200frames,998490bytes; SHA
`ed2b2f62eda4f2e3d8d7d6b3a82c975d7c5fcb31c4b042b835ff429766906f27`.
Its isolated checkpoint matches native347 exactly. Root extracted and viewed
5s/12s walking frames and19s hold: upright body and a planted terminal stance
are visible, but the small displacement is not a running demonstration. The
video was shared locally in chat; no hosting or deployment occurred. Numerical
two-episode comparisons remain the gate authority and were still running when
the clip completed.

**Matched50-update evaluations completed23:52:07.508PDT**, both exit0 and
3,528 recorded steps each. The alternative has mixed effects, not a useful
passing gait: control/native speed is0.01204807/0.01141770m/s (63.11/59.81% of
command). Native improves rear-left/right scheduled release to65.52/63.10%
versus control63.49/60.09%, still below70%. Official rear core p05 remains
native1.562/−0.140mm versus control1.707/−0.167mm, against a5mm requirement.
Both retain the same five failed gates and passing no-fall/stop/hold criteria.
Neither cost reduction nor the greater training return is a qualification.
The reviewed stdlib saved-trace analysis completed once into
`effort-pair-learning-m50-v1.json`, SHA
`bf5e3bdd0847fcfdfd22210f3d2066b9522f9e97d801098bb5021a625f9e0b8e`.
The fixed100-update allocations continue unchanged; there is no extension.

**Both100-update training allocations completed September6,00:00:29/35PDT**
(control/native), with exit0 and all external completion criteria true. Both
logged exactly298..397 inclusive; no extension occurred. Final control397 SHA
is `1f34ec60585cd746c8a8520fc191e0c6d54f6011368e4c77f0acd6932b645f42`;
native397 SHA is
`b605bb2d676a9ca3fbd0b22ec857674ceb14b0b8b70d007c03b33b05e8b16254`.
The final pair of evaluations launched at00:00:55/56PDT (PowerShell46892/55268),
after external completion and stable final checkpoint hashes were observed.
Evaluation results are pending here; training completion is not qualification.

**Final pair completed00:04:25.273PDT**, with both exits0, complete traces,
unchanged inputs and all four planned milestone evaluations finished. Watcher
completion was observed at00:04:25.401, with no error or forced cleanup.
[Final independent analysis](../reviews/v44-actuator-20260905/EFFORT_PAIR_FINAL_CONTROLLER_DECISION.md)
rejects this branch as a sufficient fix: final native speed0.00857120m/s is
16.81% below control0.01030268m/s, and rear core p05 is0.683/−0.055mm rather
than the required5mm. Both still fail the same five gates, despite no falls
and passing all stop/hold criteria. The final saved-data analysis SHA is
`c1656ca08a189ee57ba47c51e24c2e6f4efcb01a4bef0eabff03203b6fb48b05`.
Neither final397 is extended or promoted. Useful native-force telemetry and
the failed learning outcome are separate conclusions.

The [prospective next learning hypothesis](../reviews/v44-actuator-20260905/EFFORT_PAIR_DECISION_AND_NEXT_HYPOTHESIS.md)
identifies absolute critic value-loss clipping as a possible adaptation obstacle
after the reward-scale shift. This is not a measured critic-error diagnosis:
current traces lack exact pre-step critic observations and full rewards/GAE
targets, and deterministic evaluation uses a different reward-curriculum clock
and run duration from training. A source-only unclipped-value comparison is
being prepared, conditional on the completed100-update decision. It would keep
actor ratio clipping and all physical/reward/gate settings unchanged; no new
training has been selected or launched at this entry. No return or critic bias
is reconstructed from partial effort costs.

### Selected and launched Native13: critic value-loss clipping only

After the completed final comparison, root selected the independently reviewed
[Native13 protocol](../reviews/v44-actuator-20260905/NATIVE_VALUE_UNCLIPPED_13_PROTOCOL.md)
and authored an immutable selection record, SHA
`f463ea2e3fad7bbd13ad4d65c6f674d74b48ceaa0cab26d3ab22719e48e3e4d8`.
The only algorithm difference from the completed Native12 comparator is
`agent.algorithm.use_clipped_value_loss=false`. Actor ratio clip0.2, separate
actor/critic gradient limits, all reward/physical settings and existing gates
remain unchanged. This is a bounded motion comparison, not proof of critic bias.
Original control298 is reused with the same Adam reset; no397 continuation,
critic reset, return rescaling or additional context features are combined.

The new bootstrap was actually prepared at00:07:08.335PDT, SHA
`d369576f6f6313327e5ca5cf77bb67378823b5113b724b69781a490e8f4c27eb`.
Actual training started00:07:37.014PDT (Kit17896, execution93774), invocation
`20260906T070736995411Z-191547169ac248e2bbe20278909817b6`, experiment
`drobot_v44_ablation_native-vf-unclipped-13`, run label
`nvf13_191547169ac248e2bbe20278909817b6`. The finite allocation is100 updates,
128×192 transitions, seed1701, the same LR5e-5/gamma.995/lambda.95/entropy.002.
Its50/100 checkpoints will be compared to the exact completed Native12 records.
Protocol SHA is
`92efa951e5744d3cd3e242ffc26ef09ed748182bff4ee93b98155b79ce46667b`;
launcher SHA is
`33adf1f41558f35d884f5b153e1d6659829ce742ed35263e1c35a30c5ccc50a9`.
The actual logged run timestamp is`2026-09-06_00-07-43`. Root read and compared
the saved YAML: agent differences are exactly experiment name, run name and
`use_clipped_value_loss`; environment differences are exactly the run-specific
`log_dir`. Thus the new environment file hash differs for a known metadata
reason (SHA`e7e84a7d191ceae2050b84a21d295dde6a51a59ff0f04eaa7889f97ffdfc1de6`),
not a physical-setting change. The milestone watcher must allow only that exact
old-to-new log-path substitution rather than broadly ignoring environment data.
No package edits, deployment, Pi access, physical motion, unrelated verification,
commit or push accompanies this selected training change.

**Native13 training completed00:20:03.778PDT**, exit0, all100 labels298..397 and
all external completion criteria true. Final397 SHA is
`88b28546476df416f98e960b5ba9e4ae2e83e8648592bf41b26fe8f71c58fa58`.
The independently reviewed two-evaluation watcher started00:20:17.912PDT
(Python56360, execution45240,45-minute bound), after training had completed.
Thus this run's retrospective50-update review is not claimed to have occurred
concurrently with its trainer. Checkpoint347 still meets the exact later-label
and stable-bytes rule; its SHA is
`1c14a43b4bd6adb4a5985136e75b0d9f833a14229e2546059d2a35fc814e615b`.
The50-update evaluation actually launched00:20:31.097PDT (PowerShell12488).
Watcher SHA is
`0613fb134cb552a40849a653e9f371b14b31562e6b3c81194a02b221a9b42b45`;
actual launch-binding SHA is
`f5fbbc6319a7fe2b8f03470c9f64706278cd89396e71272057a0bb02457cf080`.
The watcher source-imports only the pinned ordinary evidence helper, launches
two new evaluations and never reruns Native12 or terminates a trainer.

The completed Native13 m50 report retains the same five failed gates. Speed
is0.00859019m/s versus clipped Native12's0.01141770m/s, and official rear core
p05 is−0.160/−0.143mm. Stable stopping and no falls remain, but this is not a
successful value-clipping fix. Final397 evaluation subsequently completed as
recorded below.

### Native13 closed; next critic-information experiment selected

The watcher completed both evaluations at00:25:16.759PDT September6, exit0,
failure:null, no forced cleanup. Final397 speed is0.00660460m/s versus Native12's
0.00857120m/s (22.94% slower); rear core-swing corner p05 is−0.070/−0.119mm,
versus the required5mm. Release is63.10/63.64% at the rear. No falls and stable
stopping remain, but heading error rises to6.19degrees. The same five gates
fail: PD effort occupancy, all-leg release, speed, slip and core clearance.
Lower native effort accompanies less forward motion, not useful efficiency.
Do not extend or deploy Native13. See the
[final independent decision](../reviews/v44-actuator-20260905/NATIVE_VALUE_PAIR_FINAL_CONTROLLER_DECISION.md)
and saved arithmetic SHA`22c04770a386fb742d32736fdeab322e6b1cef675becff276d88a6ad1203a698`.
These traces do not identify critic bias or true physical value targets.

Root then selected the [critic-context proposal](../reviews/v44-actuator-20260905/CRITIC_CONTEXT_SCHEMA_PROPOSAL.md),
SHA`af524801d72619c756361546b1e67034bafe92b4215a10efc5425766ccf0d528`.
Its specific hypothesis is critic state aliasing: the existing58 instantaneous
inputs omit rolling-speed history, episode/cycle release counters, path error
and the already-sampled stopping schedule used by rewards. Both new arms use
critic90 with zero-extended initial weights and preserved old normalization
statistics/count, actor50 unchanged, and a common optimizer reset. Causal32
versus zero32 is the only information treatment. These compact summaries are
not a full Markov state, and value-target agreement is not independently measured
value accuracy. Actual training rewards/values/GAE targets must be recorded;
no reconstruction from partial historical traces will substitute for them.

Planned finite screen:100 updates per arm,128×192 transitions, seed1701,
original PD effort cost, clipped value loss, unchanged LR5e-5/gamma.995/lambda.95
and physical/reference/stop/gate settings. Separate study bootstraps start from
original control298, not any rejected397. Exact347/397 nominal evaluations use
the existing two-episode protocol; all-feet and motion gates still determine
usefulness. Source implementation and independent review precede launch. No
training, simulation, checkpoint migration, deployment or hardware action has
yet occurred for this selected information comparison at this entry.

Source review before launch found one required common compatibility change:
the saved PPO configuration retains a non-null symmetry extension even though
both symmetry learning flags are false. Installed RSL-RL still calls its detached
logging-only symmetry loss, and this repository's legacy mirror function accepts
only58 critic features. Both new90 arms therefore explicitly set
`agent.algorithm.symmetry_cfg=null`. No enabled augmentation, mirror-learning
loss, reward, or stochastic sample is removed; the extra deterministic symmetry
logging forward is omitted in both arms. This is a common architecture-support
change, not the zero-versus-causal treatment. The mandatory new observer keeps
its no-symmetry guard, and the saved-config audit must allow only this exact
common difference. This incompatibility was found by source inspection before
any new run, not by a failed startup or an unrelated test.

Meanwhile the [offline effort-observable calculation](../reviews/v44-actuator-20260905/SOLVER_EFFORT_REWARD_COUNTERFACTUAL.md)
completed: on matched recorded samples, using net active joint force instead
of retained PD reduces the four effort-penalty magnitudes about94% in late run
and98% in hold. This calculation preceded selection of the finite reward-input
experiment above; it is **not** identified drive torque, a motor-current
estimate or a learning result. Its result SHA is
`32a260a9500e4d5efb514187489bf24c7b754d91139898232baaebac0bd64b05`.
Both old force channels and all original limits remain preserved.

The [final height-pair geometry audit](../reviews/v44-actuator-20260905/HEIGHT_PAIR_M200_GEOMETRY.md)
records the unsuccessful intervention: body height rises over training while
requested rear lift and speed decline. All original release, clearance, effort,
speed and slip qualification failures remain visible. No Pi access, deployment,
physical motion, unrelated tests/builds, commit or push occurred.

## Approach

1. Constrain observable response with the existing real recording, without
   pretending that unknown support/load identifies an exact servo model.
2. Run controlled simulation hypotheses for stiffness and damping, preserving
   mass, COM, shoe contacts, nominal torque, packet cap and target-rate limits.
3. Improve mechanically coherent support/clearance before longer training.
4. Train separate candidates and evaluate nominal and randomized performance,
   including stop/hold behavior. Share local videos and retain rejection data.

## Drive-gain hypothesis experiment

The inherited implicit drive uses stiffness 30 Nm/rad and damping approximately
4.58366 Nm·s/rad. Existing real data shows small near-neutral errors after the
last completed target writes, but does not establish comparable static loading.
These settings are therefore explicitly **hypotheses, not calibrated hardware
gains**. A candidate does not become physically credible merely by moving better
under a more responsive simulated drive.

All evaluations use the exact same 20 mm common-floor reference at 0.03 m/s /
0.30 Hz, nominal domain, seed 3101, phase-zero start, 25-second run plus the same
stop/hold protocol. Actor outputs are forced to zero: this isolates the prior
and plant, not learned policy performance. The loaded checkpoint is not driving
the robot. Full step traces are saved.

| Hypothesis | Stiffness | Damping | Purpose |
| --- | --- | --- | --- |
| `k30-b4p58` | 30 | 4.58366 | Same-implementation baseline |
| `k60-b4p58` | 60 | 4.58366 | Moderate stiffness sensitivity |
| `k120-b4p58` | 120 | 4.58366 | Higher stiffness, unchanged damping |
| `k120-b18p33` | 120 | 18.33465 | Higher stiffness while retaining the approximate damping/stiffness ratio |

New explicit nominal gain multipliers apply independently of physical-domain
masking; their default is one for all existing tasks. V44's approximate implicit
PD torque telemetry is now synchronized with the gains written to PhysX. Local
installed-source inspection showed that the physics gain writers update physics
buffers but do not update the actuator object's independent torque calculation.
Leaving that object at its original gains would make the new experiment's
effort comparisons misleading. This does not make implicit torque a measured
hardware torque or an exact resolved PhysX joint torque.

The gain metadata records actual simulation gain buffers. Effective torque
limits stay at the existing nominal 2.6478 Nm; target limits remain 60 degrees/s
abduction, 90 degrees/s sagittal and at most 2 degrees per delivered packet.
No hardware limit was relaxed and no production profile was promoted.

Artifacts: [gain-study folder](../reviews/v44-actuator-20260905/), including
[reproduction wrapper](../reviews/v44-actuator-20260905/evaluate_gain.ps1).
At 15:30:14 PDT, the baseline/K60/K120 fixed-damping evaluations launched as
hidden processes 39080 / 50084 / 36348. All completed around 15:32 PDT.
The fourth hypothesis launched at 15:32:42 (wrapper PID 8884) and also completed.

### Observed gain sensitivity (not hardware identification)

| Hypothesis | Mean forward m/s | Tracking error p95, degrees | Core lowest-corner p05 FL / FR / RL / RR, mm | Rear release RL / RR |
| --- | ---: | ---: | --- | --- |
| K30 / B4.58 | 0.00599 | 8.07 | 4.16 / 2.90 / -0.18 / -0.22 | 26.4% / 19.0% |
| K60 / B4.58 | 0.01148 | 4.92 | 5.26 / 4.18 / 3.18 / 0.80 | 39.3% / 36.5% |
| K120 / B4.58 | 0.01770 | 2.97 | 7.09 / 6.80 / 10.86 / 10.46 | 44.9% / 44.4% |
| K120 / B18.33 | 0.00870 | 7.66 | 4.87 / 3.28 / -0.16 / -0.19 | 30.0% / 25.0% |

All four finish without a recorded fall and settle into hold. None qualifies:
even the fastest is only 59% of the command and does not meet all-leg release,
slip and effort criteria. Increasing both stiffness and damping restores rear
dragging despite higher static stiffness. Dynamic response therefore matters;
the result does **not** establish that the physical servos have K120/B4.58.
The same-implementation K30 baseline reproduces the earlier full-cadence
zero-actor forward-speed result exactly. Original reports/traces and new offline
`*-analysis.json` reports are retained rather than overwritten.

Synchronizing implicit-actuator gains also corrects torque-derived **rewards**
(soft effort, RMS effort and torque-speed terms), not only displayed diagnostics.
Older nominal unit-gain results are unaffected by that synchronization alone;
randomized or non-unit-gain candidates must retain this source difference in
their provenance.

### Real command transport findings

The four current leg-config file hashes match the September 5 recording. Source
and pinned SDK inspection reconstruct each position-write payload at address 41
as `[254, position_low, position_high, 0, 0, 72, 13]`: raw acceleration 254,
time 0 and speed 3400. There is no configured nonzero move duration. This is
source reconstruction, not a captured wire packet or firmware-register readback.
The recording does not include the firmware revision or internal profile state.

The previously linked [STS3215 datasheet](https://files.seeedstudio.com/products/Feetech/108090023_STS3215-C001_Datasheet.pdf)
describes a 7.4 V / 19.5 kg-cm variant, so it cannot establish the limits of the
variant used with the recorded 12.2 V supply. The manufacturer's
[protocol manual](https://files.seeedstudio.com/wiki/robotics/Actuator/feetech/Communication_Protocol_Manual.pdf)
directs users to the model-specific memory table. No universal conversion of
3400 or 254 into physical speed/acceleration is assumed here. No hardware setting
has been changed.

### Separate contact-curriculum probe

At 15:40:17 PDT, three zero-actor crawl evaluations launched: K30/B4.58,
K120/B4.58 and K120/B18.33 (wrapper PIDs 39552 / 18232 / 29000).
New, separate references use 80% stance duty and sequential RR/FR/RL/FL swings,
constant backward stance-center velocity, 20 mm nominal lift and 0.12 Hz cadence
at 0.00825 m/s. The run lasts 34 seconds plus ramp/hold. This is intentionally
a slow contact experiment, not a running-policy acceptance test.

The new traces include every body mass and world COM plus the mass-weighted
whole-robot COM. Earlier traces contain only root pose and base COM information,
which is insufficient to reconstruct total COM because limbs contribute more
than half of the model's mass. Force-based contact masks and geometric corner
positions still do not identify the true pressure-center locations. Neither a
three-foot schedule nor a projected support margin alone proves dynamic balance.

The unshifted crawl probes completed and are **rejected as training foundations**:

| Drive hypothesis | Forward speed, m/s | Rear release RL / RR | Rear core corner p05 RL / RR, mm |
| --- | ---: | --- | --- |
| K30 / B4.58 | 0.00072 | 0.25% / 0.29% | -0.61 / -0.51 |
| K120 / B4.58 | 0.00652 | 12.75% / 10.60% | -0.28 / -0.28 |
| K120 / B18.33 | 0.00293 | 1.00% / 0.00% | -0.42 / -0.51 |

No falls occurred, but no probe provides genuine all-leg clearance. The stock
trot evaluator's paired-swing score is inapplicable to sequential crawl; it was
left visible rather than converted into a claimed passing running test. The
independent clearance and participation failures alone reject the crawl.

The [offline COM/support comparison](../reviews/v44-actuator-20260905/crawl-support-COM-comparison.json)
shows the simulated whole-body COM outside the **intended** three-foot support
triangle in every rear core-swing sample across all three gains. For K120/B4.58,
mean rear-swing margins are -29.1 / -28.8 mm using shoe centers and -9.6 / -9.1 mm
using lowest-edge approximations. Front-swing center margins are positive,
approximately 19 mm. The feet actually loaded often retain the rear foot that
was supposed to lift and unload a front foot instead. Recorded total COM agrees
with recomputation from all body masses/COMs within 0.000000082 m; this checks
telemetry arithmetic, not hardware model accuracy.

**Next experiment:** add smooth pre-swing weight transfer, retaining the
unshifted references as rejection evidence. Start with a separate lateral-only
30 mm body-shift hypothesis during the four-support intervals. Recompute all
shoe-corner depths under abduction and preserve reach, joint, velocity and
packet limits. Lateral shift alone may not satisfy the conservative center
triangle because the rear support boundary also has a forward component. Only
the next dynamic probe can show whether the intended rear foot actually unloads.
Do not spend a longer PPO run reinforcing the known incorrect contact schedule.

## Real-data response report and replay plan

The reproducible [real actuator constraints](../../debug/walk-trials/20260905T193441588801Z-2a2a9832/actuator-constraints/README.md)
now include chronological fit/holdout windows and neutral recovery. At the
recorded 1.375 Hz gait, delayed affine target-waveform models reduce front-joint
held-out prediction errors from roughly 12–13 degrees to 1.5–3.1 degrees. That
does not identify a pure firmware delay, physical stiffness/damping or static
gain. Front knees subsequently converge below one degree near neutral; the
same recording supports substantial dynamic lag and accurate final position.

A separate simulation replay applied the reconstructed
completed-write target sequence, including its actual neutral return, without
reapplying the policy, host packet limiter or stop logic. Plant response,
gravity and contact remain. Unknown initial root pose/support prevent this
from becoming an exact hardware reconstruction or a calibration certificate.
All four replays completed; see the [recorded-target comparison](../reviews/v44-actuator-20260905/RECORDED_REPLAY_COMPARISON.md).
None matches real amplitude, phase and neutral hold together. Higher stiffness
improves some amplitudes and final neutral error but makes the simulated response
much earlier than the real joints. A single convenient gain or universal delay
is not accepted as identified hardware dynamics.

### Smoother shifted-crawl candidate and first executions

Preferred new references:
`references/v44_common_floor_crawl_shift30_duty085_v1/{low,high}.json`.
They retain 55 mm stroke, 20 mm nominal corner lift and 30 mm lateral transfer,
with 85% stance duty and 0.09–0.10 Hz cadence (command 0.00582–0.00647 m/s).
Four scheduled supports occupy 40% of the cycle. At 0.10 Hz, each full lateral
transfer takes one second: peak planned lateral speed 0.1125 m/s and acceleration
0.3464 m/s², versus 0.270 m/s and 1.995 m/s² for the earlier duty-0.80 proposal.
No higher target or effort limits are needed. Analytical peak knee rate is
85.28 degrees/s, peak abduction rate 15.30 degrees/s, and nominal packet step
1.422 degrees. Full 3D stance sole tilt remains below 11.22 degrees.

The [frame and support report](../reviews/v44-actuator-20260905/SUPPORT_COM_AND_SHIFTED_CRAWL.md)
records the derivation. The `cad:urdf` frame guidance prompted a read-only check
against the canonical exported URDF: its mirrored joint-frame rotations map
abduction to body +X on the left and -X on the right. Hip offsets and all shoe
corners are transformed in their proper frames. No CAD/URDF file changed, and
no separate URDF validator/viewer or hardware sweep was run.

Opt-in `gait_phase_table_lateral_reference_enabled` consumes explicit planned
lateral displacement/phase-derivative tables. It retains straightness scoring
on **unplanned** drift/velocity instead of penalizing prescribed weight transfer.
The loader requires bounded finite, zero-mean 1024-row vectors, matching low/high
tables, declared sign/units, unit reference scaling, and compatible phase/path
settings. Existing tasks default to false. Cycle history stores residual sway;
the stop target removes the temporary planned lateral offset without erasing
actual accumulated drift. Startup joint blending and rate-limited homing are
not exact Cartesian motion; remaining tracking transients are still penalized.

At 15:59:41 PDT, two shift085 zero-actor probes launched (K30/B4.58 and
K120/B4.58; wrapper PIDs 45556 / 40284), plus the real-target replay under
K30/B4.58 (wrapper PID 47816). The shifted-crawl comparison changes duty/cadence
as well as sway relative to the first crawl; it is not a single-variable causal
ablation. If promising, a matched-duty zero-shift control will isolate the
transfer effect. Reports, traces and output logs stay local in the study folder.

### Lateral-only result and mass-ledger correction

Both shift085 probes completed and failed rear unloading: rear scheduled and
core release were **zero** under both drive hypotheses. K120 released the front
feet, with approximately 9 mm mean core clearance, while the rear lowest corners
remained near ground. Whole-body COM actually swayed 41–45 mm peak-to-peak, so
the transfer was not missing or sign-reversed. Its projected position remained
outside the intended rear-swing support triangle. See
[shift085 results](../reviews/v44-actuator-20260905/SHIFT085_COM_RESULTS.md).
The next reference combines lateral and **forward** pre-swing transfer and is
being screened against full 3D corner reach, rate and sole-angle constraints.

The recorded-target replay exposed a separate authored mass discrepancy:
3.307747841 kg versus the documented seven-pound target, **3.175146590 kg**.
Source accounting explains the difference: four newly authored shoes add mass
absent from the legacy ledger, while base reconstruction omits the legacy merged
camera/IMU contributions. This is not physical-domain randomization. The linked
replay comparison preserves the exact ledger, source hashes and limitations.

An ablation-only dry-mass scale **0.6196256484037118** replaces the experimental
value 0.65060760808206, preserving the 0.523179545 kg pack. Its 16:10:41 PDT
zero-actor probe completed and reports **3.175146818 kg** (rounding-level match).
That corrects total mass for this authored asset; it does not identify the real
mass distribution, restore fixed-sensor COM/inertia, or alter production defaults.
The probe otherwise retains K30/B4.58 and the original diagonal reference.
Forward speed changes only from 0.0059867 to **0.0059923 m/s**; rear release remains
26.1%/19.3%. Correcting mass alone does not fix the gait. Both old and corrected
reports/traces remain in the study folder as distinct evidence.

### Combined-transfer executions and next learning pilot

The bounded geometry search produced two new, distinct references, documented in
[combined-transfer preparation](../reviews/v44-actuator-20260905/COMBINED_TRANSFER_PREPARATION.md).
The preferred X10/Y30 candidate has 10 mm forward and 30 mm lateral transfer,
0.325 m common corner depth, 0.090/0.080 m front/rear stance offsets, maximum
72.405 degrees/s joint target rate and 11.684-degree stance sole tilt. The exact
level-root moving-height target is 0.370696689 m. The X20 alternative retains
0.330 m depth with different stance offsets; this is a compound geometry comparison.

Offline mass-weighted FK was reconstructed from canonical joint frames and
recorded body COMs, agreeing across sampled poses to below 0.23 micrometres in
local COM coordinates. Under corrected mass, X10's intended rear-swing COM is
inside center-point triangles by about 3 mm but outside lowest-edge point
triangles by about 11 mm. These different approximate pressure-point choices
do **not** prove balance; they prioritize a dynamic probe, not certify a gait.

At **16:22:52 PDT**, actual zero-actor simulations launched for X10/K30,
X10/K120 and X20/K120 (wrapper PIDs 21012 / 49972 / 55280), all with corrected
total mass, unchanged torque/rate/packet caps, and the 34-second run/stop/hold
protocol. Results are named `xy10-k30-massfit`, `xy10-k120-massfit` and
`xy20-k120-massfit` in the study folder. Launch is not completion evidence.

The new opt-in forward-reference setting validates bounded displacement and
derivative tables and uses the same applied phase/cadence/ramp as the reference.
Instantaneous velocity errors distinguish planned transfer from tracking error;
net-speed/stall objectives remain actual recorded movement. Positive progress
waits for the full rolling window. Stop anchoring removes temporary planned X/Y
offsets while retaining real drift. All existing task defaults remain disabled.

`--schedule-aware-crawl` changes evaluation only when explicitly selected with
the V44 ablation. It replaces the inapplicable paired-swing gates with complete
per-leg core-bout coverage and simultaneous released/clear swing foot plus three
vertically loaded support feet. Insufficient coverage fails. Original speed,
straightness, clearance, slip, effort and neutral-hold gates remain; reports
explicitly say **crawl foundation, not running qualification**. Default trot
evaluation is unchanged. No separate test suite was executed.

The fresh learning configuration, intended hyperparameters and decision criteria
are recorded in [the pilot hypothesis](../reviews/v44-actuator-20260905/XY_TRAINING_HYPOTHESIS.md).
It is preparation only until an external supervisor record confirms actual
training. The next meaningful learned checkpoint will receive a local video.

The three XY probes completed at 16:25:33. None releases either rear foot.
X10/K120 gives 100% coordinated front core swings, but both rear core scores
remain zero; X20 worsens front clearance too. No fixed crawl is qualified.
The next experiment is therefore explicitly **contact-discovery learning**,
not fine-tuning an already functioning crawl. Wider bounded residuals may adjust
load sharing; checkpoint 50 is an early decision point, not a commitment to
extend a configuration with no rear-release improvement.

A source review found an important inherited reward conflict: the running task
had disabled three-foot support/excess-airborne objectives and reduced missing
stance cost. The new crawl pilot explicitly restores schedule-identity support,
soft progress gating and support/airborne penalties while retaining a small
exploration floor and dense per-foot discovery signals. Gamma 0.999/lambda 0.995
provide a longer, but still bootstrapped, credit horizon for the ten-second gait.

**16:27:29 launch attempt failed before simulation/training began.** The external
supervisor correctly reports zero iterations and `failed`: the prelaunch source
snapshot hit Windows' path-length limit. Its prepared/process/completion and
error logs are preserved in `xy10-contact-training-supervisor-v1`. The retry will
use shorter evidence/isolated-experiment paths with full identities retained in
manifests, rather than skipping reproducibility evidence. No trained checkpoint
was produced by this failed attempt.

The second attempt published the compact source snapshot but failed before
learning because the absolute Kit script invocation lacked the repository on
Python's import path. Both the supervisor environment and trainer now explicitly
include the repository root. Failure evidence remains in supervisor-v2.

**Actual learning began on the third attempt**, launched at 16:32:16 PDT
(wrapper PID 35192). Effective configs confirm 128 environments, 60 Hz control,
192-step rollouts, 150 fresh updates, seed 1701, LR 0.0002 fixed, entropy 0.002,
gamma 0.999, lambda 0.995 and the reviewed crawl-support overrides. Iteration
output is present at approximately five seconds/update. Run:

`logs/rsl_rl/drobot_v44_ablation_xy10-contact-massfit-03/2026-09-05_16-32-21_xy10-con_07462843e9eb4f0e9eb40055504f6a70`.

The external records are in `xy10-contact-training-supervisor-v3`; full source,
reference and asset snapshots use flat numbered files with original logical
paths/hashes retained in the manifest. Full UUID run identity is retained while
the human label and isolated experiment path are shortened. The production
experiment is not renamed. Completion and qualification remain separate.

### First learned checkpoint: rejected

Model 50 was saved at 16:37:33. Its deterministic 34-second nominal evaluation
launched at 16:37:57 (wrapper PID 33372) and completed at 16:40:36. All four legs
still have **zero scheduled/core release**. Mean forward speed is
0.000336518 m/s, approximately 5.2% of the command, versus 0.000341763 for its
zero-actor baseline. Mean root height improves only about 0.53 mm; that does
not overcome the support/posture deficit. No fall occurred and neutral hold is
stable, but standing without falling is not walking. No qualification or
deployment follows from those two passing safety observations.

An actual 20-second, 60 fps/1280x720 local recording is available:
[model50 video](../reviews/v44-actuator-20260905/xy10-model50-video/videos/play/rl-video-step-0.mp4).
It uses a separate copied checkpoint and nominal ablation inputs, runs for
15 seconds then ramps/holds; it is an illustrative review clip, not the
34-second quantitative evaluation. No hosted site was created. Review-only
playback exports remain isolated beside the copied checkpoint, not in onboard
deployment packages.

The [ordered geometry decomposition](../reviews/v44-actuator-20260905/XY_PROBE_RESULTS.md)
reproduces recorded sole corners using measured simulation joint/root poses.
At a level reference-height root, rear joint poses provide clearance. Applying
the recorded root attitude removes roughly 11–18 mm, and recorded root-height
loss another 5–11 mm. These are ordered geometric substitutions, not independent
causal effects: dynamics and contact couple the quantities. A preserved 1 mm
generator/simulator shoe-depth discrepancy was also found, too small to explain
the lost lift. The active pilot's references are not silently regenerated.

The 150-update pilot remains bounded; it will not be extended unchanged on the
basis of rising reward. The next hypothesis is being prioritized using the
stronger earlier diagonal-gait baseline and the newly measured posture/support
deficit. Crawl was a contact diagnostic, not an assumption that eventual running
must satisfy quasi-static three-foot support.

The crawl pilot **completed all 150 updates (0–149)** at 16:49:04 PDT: 3,686,400
transitions, 999.12 seconds reported training time, child exit zero. The external
supervisor records `completed_observed` with every completion criterion true.
Final checkpoint SHA-256:
`136f7f1a32a7130c1e9a49c3e71f536168eaa9b59b3adce6d99f66a0e90f41f7`.
Completion is not gait qualification; final-checkpoint evaluation remains
separate. This pilot is not extended unchanged.

At 16:49:14, a matched mass-corrected evaluation of the earlier diagonal
model199 launched (wrapper PID 28564), keeping its exact original common-floor
reference, residual scales, K30 and 0.03 m/s command. Two complete episodes also
capture uncached PhysX max-force readbacks across construction, resets and first
steps. The earlier 0.883 Nm startup table is printed before constructor domain
settings; source inspection shows normal episode resets do not restore that
authored cap. The new getter evidence will check configured physics properties,
not equate approximate actuator torque with measured joint force.

A new isolated diagonal-posture refinement will preserve the baseline actor
and critic but reset optimizer moments and LR to 0.00005. The training-preparation
helper created a distinct bootstrap at
`logs/rsl_rl/drobot_v44_ablation_diagonal-posture-04/bootstrap/model_199.pt`, with
source and transformed-checkpoint hashes in its continuation sidecar. The
original checkpoint is untouched. This is input preparation, not yet a trained
refinement or deployment package.

### Diagonal baseline results and revised priority (16:58 PDT)

The two-episode mass-corrected diagonal evaluation completed at 16:51:00.
All five uncached PhysX readbacks match the expected and cached **2.647795 Nm**
joint force limits exactly. The earlier printed initialization table is not the
live episode limit. This establishes the configured cap, not motor calibration
or a measured applied torque.

At a 0.03 m/s command, model199 produces 0.009914 m/s. Scheduled release is
77.12/75.04/60.30/63.17% (FL/FR/RL/RR); rear core-swing release is already
99.38/99.69%. Rear minimum-corner height is nevertheless only 1.79/3.27 mm
mean and 0.45/0.92 mm p05. It does not qualify. The reference is the original
**10 mm** common-floor table, not the separate 20 mm probe reference.

The ordered offline decomposition changes the next hypothesis. For RL/RR,
nominal level-root clearance 10.70/10.70 mm becomes 4.10/11.02 mm at policy
targets, 3.83/6.26 mm at actual joint positions, 7.85/9.60 mm after measured
root attitude, then 1.79/3.27 mm after measured root height. In this gait,
attitude **adds** about 4.0/3.3 mm of rear clearance. These are conditional
geometric substitutions, not independent causal effects. Blanket flattening
could remove useful lift; a height/upright-only objective is not justified.
Estimated stance hard-cap occupancy is also substantial (front knees 58–69%,
rear hips 34–66%). The normalized-posture opt-in implementation remains
disabled and has not been exercised; preparation is not evidence of benefit.

Next priority is a bounded continuation emphasizing existing **actual sole
corner clearance**, while retaining phase-compatible attitude and unchanged
effort, smoothness, slip, straightness and stopping costs. No actuator limit is
raised. A separate offline 15 mm reference calculation fits 0.30 Hz but requires
95.27 deg/s knee motion at 0.40 Hz, above the unchanged 90 deg/s cap. Therefore
the active 10 mm reference is not silently replaced by that higher lift.

Final crawl model149 evaluation launched at 16:58:57 PDT (wrapper PID 57152),
using its original nominal JSON and schedule-aware crawl gate, to close that
bounded pilot with final-checkpoint evidence rather than extrapolate model50.

Final crawl evaluation confirms rejection: 0.00041445 m/s (6.41% of command),
zero scheduled release at all four feet, p95 joint tracking error 7.106 degrees.
No fall and stable four-foot neutral hold do not establish locomotion. This
configuration is not extended unchanged.

At 17:02 PDT the next actual training launch uses
[diagonal-clearance-learning-nominal-v1.json](../reviews/v44-actuator-20260905/diagonal-clearance-learning-nominal-v1.json)
and the [explicit hypothesis](../reviews/v44-actuator-20260905/DIAGONAL_CLEARANCE_HYPOTHESIS.md).
Only the clearance objective weight changes, **6 to 18**; its actual-corner
measurement, 10 mm C2 target and phase definition stay unchanged. Body posture
normalization remains disabled. The compatible seed has a new separately named
clearance bootstrap, SHA256
`f71b9dbd4ad1814829053588e2282ef2654f0a89db46f5ecb844ce3f3c316cb0`.
Optimizer reset and LR reduction to 5e-5 are documented continuation changes.
The earlier posture bootstrap remains unused, not renamed or overwritten.
The pilot requests 100 additional updates, labels 199–298, 128 environments and
192-step rollouts. Gamma/lambda and 16-second episode/10–12-second moving
durations remain seed-matched. Intermediate checkpoint250 and final298 receive
actual evaluations; launch and training completion are not qualification.

The supervisor child PID is 42660 (wrapper 45112), and actual learning began in
`2026-09-05_17-02-15_clearanc_f907fa0a921d40f99b0f46ce3773c758`.
An additional early checkpoint225 evaluation launched at 17:05:36 (wrapper
1832), after 26 new updates, without changing the prepared launch hypothesis.
It uses the same two nominal phase-zero, 25-second episodes as model199.

Checkpoint225 completed evaluation at17:08:32. It is **not improved as a gait**:
speed rises0.009914→0.010425m/s and front-left worst-core p05 rises2.61→5.28mm,
but rear-left falls0.445→0.195mm and rear-right0.888→−0.302mm. Rear scheduled
release is60.0/60.4%. All six prior failed gates remain failed. No falls and
100% four-foot/stable neutral hold are retained; estimated overall soft-effort
occupancy drops slightly32.79→31.92%, while joint acceleration rises6.58→6.94
rad/s². The two-episode evaluation takes about176seconds concurrently with
training; stdout is buffered and the trace is only published at completion.
Missing intermediate files therefore did not establish an initialization hang.

The [offline scorecard](../reviews/v44-actuator-20260905/diagonal-baseline199-vs-clearance225.md)
conservatively flags missing/explicit-override metadata differences: the new JSON
explicitly pins sole target0.010 and transition0.5, whereas the baseline inherited
those same values from its saved/current source config. This is not a changed
reference or physical plant. The interpretation relies on that source audit,
not on treating an absent override as a measured zero. Baseline and candidate
share command, nominal-domain draws and initial/stop phase coverage.

Seed training updates175–199 show clearance's weighted episode-mean term was
only−0.5751 before global reward_scale0.1, versus effort RMS−6.2684, soft-effort
cost−3.6121 and action-acceleration−2.2987. Tripling its weight is an objective
change, not itself evidence of better motion. The continuation also uses128
environments versus the seed's256, so one new update has half the transitions.

Checkpoint250 evaluation (launched17:10:08, wrapper48664) repeats a weak-foot
trade, not a qualified improvement. Actual speed is0.009061m/s; worst core p05
is1.859/2.760/−0.202/2.550mm in FL/FR/RL/RR order. Rear-left scheduled release
falls to55.07%, rear-right improves to65.49%. All six prior gates still fail.
Joint/body accelerations and several slip values improve, but estimated soft
effort occupancy rises to34.70%. No falls; stable four-foot hold remains100%.
See the [saved comparison](../reviews/v44-actuator-20260905/diagonal-baseline199-vs-clearance250.md).

The225 trace independently reconstructs the unchanged mean clearance-deficit
cost0.16439→0.16451 while the worst swinging-foot deficit grows0.26718→0.28691.
That supports a subsequent bounded mean/worst-foot aggregation experiment,
not a larger torque cap or stronger blanket upright target. It remains a
hypothesis, to be compared against the preserved seed and completed pilot.
A20second local model250 recording launched at17:14 PDT; it illustrates an
unqualified intermediate checkpoint, not a successful new running model.

The first model250 recording attempt failed before rollout: the environment
module saw the new optional worst-foot field while its configuration had
already been imported before that multi-file edit completed. No video resulted.
The traceback and isolated copied checkpoint are retained. Source inspection
shows `--train_env_cfg` uses the current task configuration, not an old pickled
environment. Further simulator launches wait for the complete source edit to
be frozen; no fallback silently substitutes a missing configuration field.

The weight18 pilot completed at17:15:31, all100 labels199–298 observed,2,457,600
new transitions,792.57seconds reported training. Supervisor exit0 and every
completion criterion pass. Final model298 SHA256:
`253fde3af1cb7d01174bfdf8de87bae2512123bf53c66ff20333d62afa5cfee6`.
Final evaluation is still separate; there is no successful-gait claim from
completed training.

Final298 evaluation and a fresh model250 recording retry launched together at
17:19:34 (wrappers45712/50924) after the four-file source change was complete and
frozen. The video retry passed the prior initialization failure and loaded the
policy normally. Existing failed-attempt artifacts are preserved in their own
directory.

### Worst-swing-foot continuation

The next bounded pilot uses
[diagonal-worstfoot-learning-nominal-v1.json](../reviews/v44-actuator-20260905/diagonal-worstfoot-learning-nominal-v1.json)
and [its launch hypothesis](../reviews/v44-actuator-20260905/DIAGONAL_WORSTFOOT_HYPOTHESIS.md).
The new opt-in fraction0.5 averages the original mean squared deficit and the
worst scheduled, active swing-foot squared deficit. In a diagonal pair this
weights the worse foot75% and the other25%. Equal deficits retain the same
magnitude. Stance feet never enter the maximum; zero/one swing foot retains
the original calculation. All defaults/production contracts pin the fraction
to0; validation rejects booleans, nonfinite values and fractions outside[0,1].
No pose, reference, actuator limit or other reward is changed.

Training starts from the same original199 seed with the same optimizer-reset,
LR5e-5,128-environment/192-rollout and100-update budget as the coefficient-only
pilot. A distinct bootstrap SHA256 is
`ef6adb514027c574628a420877cfec2a5181a59f16cafd3cdc24cd917d8c0ffd`.
The source was reviewed without separate automated tests; the opted-in path
will be exercised by the actual authorized training/evaluations. No deployment
or Git publication follows from this preparation or launch.

The worst-foot pilot launched17:21:18 (wrapper32372), with effective run
`2026-09-05_17-21-23_worstfoo_6b0cf4ec88b343c3be779c56cf0ff32e`.
Its actual training exercises the new fraction0.5 path; completion/evaluation
remain pending. The fraction0 path was also exercised by the completed final298
evaluation and successful model250 recording retry after the source freeze.

The coefficient-only **final298 is rejected**: speed0.00755683m/s (25.19% of
command), worst core p05 FL/FR/RL/RR2.051/3.068/−0.270/2.373mm, scheduled
release71.80/70.90/58.58/65.10%, overall approximate soft-effort fraction33.65%.
No falls and four-foot stable hold are retained, but all six prior failed gates
remain failed. See [final comparison](../reviews/v44-actuator-20260905/diagonal-baseline199-vs-clearance298.md).
No unchanged extension of that objective is justified by its completed updates.

The successful [model250 local clip](../reviews/v44-actuator-20260905/diagonal-clearance-model250-video-v2/videos/play/rl-video-step-0.mp4)
is20.000seconds, H.264,1280×720,60fps,973314bytes. It commands0.03m/s for15seconds
then ramps/holds. Extracted review frames are beside it. It was shared in chat
as an **unqualified intermediate**, not a newly deployed or successful model.

### Effort measurement caveat under investigation

The baseline's final-hold estimate matches the configured clipped PD formula
within0.000965Nm, with correctly ordered K30/B4.58366 and2.647795Nm caps.
All eight sagittal joints exceed the configured **0.980665Nm soft threshold**;
this is not the hard cap or proof of excessive actual motor current. The four
normal loads sum31.14804N, near the simulated weight31.14819N, but installed
contact-sensor `net_forces_w` contains normal forces only, not tangential
friction or a full joint moment balance.

At60Hz the final-second positions are almost stationary while reported joint
velocities remain nonzero (up to0.0617rad/s). That can reflect substep aliasing,
cache timing or solver semantics; current evidence does not identify which.
A passive120Hz pre/post-physics diagnostic is being prepared to distinguish
them, without changing the plant or any limit/gate. Approximate PD telemetry
must not be silently relabeled as independent solver or real motor torque.
The unchanged3second hold alone does not mathematically force the29.4second
evaluation to fail the10% soft-effort gate: it contributes6.72percentage points.
Do not lengthen trials just to dilute that metric.

The worst-foot pilot completed17:33:49: all100 labels199–298,2,457,600 new
transitions,741.8seconds reported training and every external completion check
passing. Final SHA256:
`1ddf887552509447d67a5619085ccaefa04b35b7aa9dd894b62a6853bb370df5`.
Its225 evaluation shows only a mixed early result: speed0.010160m/s, rear-left
worst p050.705mm but rear-right−0.197mm. No qualification follows from that.
Model250 evaluation with the new passive120Hz capture launched17:33:59
(wrapper14932); final298 evaluation follows separately.

### Correct target-ceiling measurement and passive substep observation

Source review found the aggregate target-ceiling metric/gate still used a
scalar90deg/s and fixed control dt, although V44's detailed diagnostics already
used per-joint60/90deg/s, randomized scale, actual packet elapsed time and the
2degree packet cap. This could miss abduction saturation and misclassify
post-hold delivery. V44 now shares the corrected calculation across detailed
and aggregate metrics. The existing15% threshold and all-joint/all-selected-tick
denominator remain unchanged; held packets count as zero. Historical non-V44
arithmetic remains an explicit fallback. This changes measurement, not physics.
New reports/traces include the exact nominal/effective bounds, allowed packet
deltas, flags and scaled sustainable-effort thresholds. Prior reports are not
silently rewritten or relabeled as using this corrected definition.

The passive helper installs only around existing selected env.step calls during
the scheduled last second of hold. It snapshots cached state and independent
PhysX state immediately before/after each120Hz step, restores the step method,
and writes buffered exclusive output after the episode loop. It adds no steps,
forces or target updates. Getter errors remain explicit unavailable channels;
partial step failures attempt evidence preservation then re-raise the original
error. Host-copy overhead means this is not a timing benchmark. Main reviewed
the helper, integration and metadata source before the actual authorized run;
no separate tests or smoke jobs were launched.

### Completed worst-foot pilot and 120 Hz capture

The final worst-foot298 evaluation completed successfully but remains rejected.
At the unchanged 0.03 m/s command it achieves **0.007104 m/s**, versus the original
199 seed's 0.009914 m/s. Worst episode core-corner p05 values are FL2.582,
FR2.251, RL0.555 and RR2.912 mm; all remain below the 5 mm target. All six
previous failure gates remain failed, including rear release, speed, slip and
approximate soft effort (34.38%). There are no falls and the final four-foot
hold remains stable. These are two nominal development episodes, not randomized
qualification. See [preserved final comparison](../reviews/v44-actuator-20260905/diagonal-baseline199-vs-worstfoot298.md).
The comparison labels incomplete historical metadata conservatively; the old
implicit 10 mm clearance/0.5 transition match the new explicit values, while
the corrected target-ceiling measurement is a separately documented change.

Model250's passive physics capture completed17:35:54 with **240/240** expected
120 Hz records, 120 per episode, zero snapshot/step/lifecycle errors and no
truncation. Its gait still fails: speed0.008599 m/s, worst rear-left core p05
-0.223 mm and approximate soft occupancy35.42%. Raw backend velocity versus
position-difference discrepancy persists at120Hz; it is not explained just by
60Hz sampling or stale pre-step Lab caches. For example, episode0 FL hip raw
velocity RMS is0.02464 rad/s, versus0.00003595 rad/s from each actual physics
interval's position difference. Pre-step cache and raw state agree exactly.

This discrepancy alone is **not proof of a simulator bug**. PhysX documents
different velocity/position integration semantics under constrained motion,
and a particular TGS steady-state discrepancy when gravity is applied only
once per step. That is a hypothesis to check against the installed configuration,
not permission to change reported torque or bypass gates. The installed Lab
default appears to enable external forces every iteration already; saved/runtime
configuration is being checked. See [official solver explanation](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/Simulation.html#constraint-solver).
Projected joint forces and approximate PD are also distinct quantities and
must not be interchanged as if either were measured physical servo current.

Next geometry hypothesis: reduce the fixed stroke from55 to35mm and increase
body-relative corner lift from10 to15mm, at0.30–0.35Hz. Offline geometry predicts
smaller stance sole tilt, with the high-cadence knee target below90deg/s;
0.40Hz is explicitly excluded because it would exceed that limit. A new
isolated reference and nominal zero-residual/original-seed probes are being
prepared before another training commitment. This slow foundation is not a
claim that the user's running-speed goal is achieved.

### Short-stroke probes launched; solver flag not changed

At17:45:40 two nominal evaluations launched using original model199 and a
zero-residual control, both with the new35mm/15mm reference, command
0.019090909 m/s,25 seconds motion,0.9 second ramp and3 second hold, two
development episodes each. Wrapper PIDs46108/55348; outputs are
`shortstroke-zero-*` and `shortstroke-model199-*` in the study directory.
The command/cadence differ from the original55mm baseline; cross-reference
comparisons are mechanism evidence, not matched-speed improvement claims.

The new low command lies below the inherited `active_translation_threshold_m_s`
of0.020. Because the environment uses a strict `command_speed > threshold`,
leaving it there would incorrectly turn off translation rewards. This isolated
reference config therefore sets it to the existing gait deadband0.004. The
sustained speed floor stays0.020: its existing command-scaled definition is
`min(0.020, 0.65*command)`, or0.012409 m/s at the low command. No acceptance
gate, hardware rate, torque cap, residual envelope or production default is
changed. Reference generation/analytical checks alone did not qualify motion.
See [hypothesis and exact inputs](../reviews/v44-actuator-20260905/SHORT_STROKE_HYPOTHESIS.md).

The separate installed-source/saved-YAML audit confirms TGS with external
forces every iteration **already enabled**,8 position/4 velocity iterations,
120Hz physics and60Hz control. Actual earlier startup tables show zero static,
dynamic and viscous joint friction on all12 DOFs (shoe contact friction is
separate). No solver flag was toggled. Installed Isaac Sim identifies itself
as6.0.1-rc.7, PhysX integration110.1.13; the exact embedded SDK minor has not
been established, so the cited5.7 docs are not a local-version claim.

### Short-stroke transfer results and observation-contract finding

Both probes completed17:48:28. With zero learned correction, actual speed is
only0.002114m/s (11.07% of command), and rear scheduled release is20.36/7.86%.
With the original199 actor, speed is0.008710m/s (45.62% of command), release
FL79.89/FR82.06/RL68.10/RR70.88%, and paired release quality now passes.
Both retain zero falls and stable final four-foot holds. The actor variant
still fails every-leg release, 5mm core clearance, speed, slip and soft effort
(32.40%). Thus the residual actor helps this reference, but the new reference
alone is not a successful gait and absolute speed has not improved over the
older55mm seed. A bounded continuation, rather than deployment, is being prepared.

The short-stroke actor's worst-episode core p05 is FL9.409/FR9.349 mm but only
RL0.923/RR1.323 mm. The front feet now have genuine clearance headroom in this
simulation; the rear-foot deficiency remains. The zero-residual reference has
rear p05 below ground by approximately0.18/0.22mm. See the preserved
[same-reference actor comparison](../reviews/v44-actuator-20260905/shortstroke-zero-vs-seed199.md)
and [different-reference contextual comparison](../reviews/v44-actuator-20260905/diagonal-baseline199-vs-shortstroke199.md).
The generic comparison utility conservatively flags an absent crawl-metadata
field; these were explicitly diagonal evaluations without the crawl option.

A separate source/record audit establishes a real observation-definition
mismatch: Pi joint velocity is a clipped difference between calibrated encoder
samples using completed-feedback timestamps; training consumes raw PhysX joint
velocity. In the September5 recording,145 repeated feedback pairs retain
identical position and velocity, and217 retained9–12ms pairs match the encoder
finite-difference calculation to less than1e-4rad/s. Runtime normalization
matches within2.45e-8. This establishes the calculation/cadence mismatch,
**not that it causes dragging**. No Pi or policy contract is silently changed.

Existing simulated traces lack the exact complete50-dimensional actor input.
Reconstructing its IMU/noise/history channels would invent evidence, so the
next evaluator-only diagnostic will capture the actual pre-inference input,
raw output and submitted action before an offline paired-input sensitivity
evaluation. A60Hz finite-difference counterfactual is only a definition probe,
not a replica of the Pi's100Hz, quantized, asynchronous feedback path. Physical
solver states, rewards, effort measurements and gates remain unchanged.

### Continuation06 and exact actor capture launched

At17:55:44 the short-stroke continuation launched (wrapper21128), supervised
under `shortstroke-clearance-supervisor-v1`, namespace
`drobot_v44_ablation_shortstroke-clearance-06`. It starts from the separately
prepared original199 actor/critic, not a rejected reward-refinement checkpoint.
Bootstrap SHA256 `5cb261edae696b78f6a887573dfb1556ef5689ccf453417a10711d4047884d5a`;
Adam moments cleared, LR5e-5. The bounded request is100 further updates199–298,
128 environments,192-step rollouts,16second episodes with10–12seconds moving,
baseline clearance weight6 and mean aggregation. All physical limits and
acceptance criteria remain unchanged. See the immutable launch
[hypothesis](../reviews/v44-actuator-20260905/SHORT_STROKE_CONTINUATION_HYPOTHESIS.md).

The original55mm model199 actor-capture evaluation launched alongside it
(wrapper26524), two nominal episodes at0.03m/s with the existing25+.9+3second
protocol. New output prefix `diagonal-model199-actorcapture`. Main reviewed
the exact input/output extension before launch; evaluator SHA256
`090033fa5fa8fa3c81926cf6f63b201039b7adcd0f694340ba00e978965a3df3`.
It captures actual50-channel input before the sole policy call, raw12-action
output, submitted/clipped action and pre/post physics/reset identities. There
are no new observation/noise draws or counterfactual actions in this simulation.
The possible cached first observation is preserved for diagnosis, not silently
regenerated. See [trace contract](../reviews/v44-actuator-20260905/EXACT_ACTOR_TRACE_CONTRACT.md).

Offline short-stroke analysis finds100% rear force release in both rear core
swing windows (322samples per leg), despite inadequate corner height. Requested
abduction maxima are only0.091/0.057/0.060/0.103degrees in FL/RL/FR/RR order,
well below the0.413degree residual limit, with zero near-limit occupancy.
The current evidence therefore points to shallow rear swing, not a binding
abduction clip; it does not justify enlarging that authority. A20second local
short-stroke seed recording also launched17:52:40 (wrapper27776). It is an
intermediate reference-transfer video, not a newly trained or deployed model.

### Exact capture/replay result and confirmed evaluator initialization bug

The actual capture completed17:58:42 with3528 rows and reproduces the old
baseline's speed0.0099143665m/s and soft occupancy0.32787699 exactly. Its
38,616,515byte trace SHA256 is
`194fd6846e2799ef78463ff140a40ad77ea46586880d14a9be187981efacec1d`.
The ordinary-Torch saved-input replay reproduces all42,336 original action
components: maximum error3.5763e-7, RMS6.8614e-8, no failures against the
predeclared2e-6 absolute/2e-5 relative tolerances. This is actual authorized
actor evaluation, not fabricated observations or a simulator smoke test.

On3526 causally attributable inputs, replacing only qdot with the previous
completed60Hz position difference gives steady-run action-delta RMS0.005227,
p95 absolute0.008843, maximum0.090527, versus original action RMS0.452980.
Final-hold delta RMS is0.001483. Typical sensitivity is small, with larger
transients; this does not prove closed-loop equivalence or test100Hz encoder
quantization/timestamp effects. No action was executed in a counterfactual
plant, and no hardware target delta is claimed. See the preserved
[offline replay result](../reviews/v44-actuator-20260905/diagonal-model199-actor-velocity-counterfactual.json).

It also confirms a separate **evaluator initialization bug**: episode1's first
actor input still contains the previous stopped command [0,0,0.28131619] in
normalized feature units and differs from the reset joint state by up to
0.071497rad (4.09648degrees). All3526 noninitial inputs pass state-alignment
checks. Both first inputs were excluded from the causal velocity sensitivity;
original-action replay still includes them. The wrapper's get_observations
returns cached obs_buf after private _reset_idx rather than generating a fresh
input. A separate versioned evaluator-only correction is being prepared after
the final reset/phase/command setup. It will intentionally generate one fresh
observation and may change later RNG draws; historical evidence remains intact.
Training environment, Pi controller, physics limits and gates are not changed.

The verified [short-stroke seed clip](../reviews/v44-actuator-20260905/shortstroke-seed199-video-v1/videos/play/rl-video-step-0.mp4)
is20.000seconds,1280x720,H.264,60fps,997743bytes. Five extracted review frames
are beside it; main inspected frame02 and shared the local clip as an
unqualified intermediate. It is not the continuation06 training result.

### Corrected-initialization comparisons underway

Main reviewed the evaluator-only correction against the installed DirectRLEnv
reset sequence. It finalizes the already-selected reset/phase/schedule through
scene write, forward kinematics and one new observation, without another reset
or physics step. Source SHA256
`63aa16f980303c064aa7377edaa3b20fdd0708bcd762e936a2643366b5d2e961`;
new initialization version `v44.finalized-reset-fresh-observation.v1`.
Per-episode counter and feature/state alignment evidence is required in the
actual evaluation. No separate tests, compilation or smoke run was launched.
See [root cause and correction](../reviews/v44-actuator-20260905/EVALUATOR_RESET_OBSERVATION_FIX.md).

Three two-episode comparisons launched18:06:36, retaining25seconds motion,
0.9second ramp and3second hold, seed3101/nominal/fixed phase0:

| Output prefix | Checkpoint/reference | Command m/s | Wrapper PID |
| --- | --- | ---: | ---: |
| diagonal-model199-freshreset | Original199 / old55mm10mm | .030000 | 7080 |
| shortstroke-model199-freshreset | Original199 /35mm15mm | .019090909 | 23020 |
| shortstroke-model250-freshreset | Continuation06 intermediate250 /35mm15mm | .019090909 | 56684 |

All use the corrected evaluator and exact actor trace. The old/new reset
comparisons deliberately preserve the intentional fresh-observation RNG
consumption difference rather than pretending to hold random streams fixed.
No model is qualified just because its initialization is now correct.

Recorded-data reward inspection also finds the current legacy height cost is
very small: around-0.0001145 before global scaling at6.10mm sag, versus the
clearance cost around-0.7706. A potential height-only normalized objective is
being evaluated on paper; it is not enabled. The existing all-posture opt-in
would also replace tilt terms, which is not justified because attitude helps
rear clearance in these traces. Preserve helpful tilt, physical caps, effort
uncertainty and neutral-hold semantics in any separate future hypothesis.

Continuation06 completed18:10:42: all100 requested labels199–298,2,457,600
additional transitions,889.29seconds reported training, successful exit and
every external completion criterion true. Final checkpoint SHA256
`9935d3dd4bfd43e3eb3cc41b4d35077822ddfe8c95595ea02a49337d652942a9`.
Its corrected-reset final evaluation was launched separately after this
completion, with prefix `shortstroke-model298-freshreset`. Completion is
training evidence only, not a gait-quality or deployment pass.

### Corrected evaluation results: reset fixed, continuation not accepted

The three initial comparisons completed18:11:43–44; final298 completed18:13:22.
Corrected initialization has exactly zero command/q/qdot feature gaps and no
counter advance in all reported episodes. Baseline speeds barely change:
old-reference1990.00992169m/s, short-reference1990.00871062m/s. Both retain
zero falls and stable holds. Fixing evaluation correctness did not fix dragging.

Intermediate250 increases actual speed to0.01074339m/s (56.27% command) and
reduces approximate soft occupancy to25.41%, but rear scheduled release falls
to61.70/66.95%, and heading and12degree sole-tilt gates fail. Final298 achieves
0.00953288m/s (49.93% command), soft occupancy26.75%, rear release47.43/62.94%,
and additionally fails lateral drift and paired release. Both have no falls;
hold stable fractions98.89/94.44% still pass but are worse than the seed.
The completed continuation is **not accepted**. See [intermediate comparison](../reviews/v44-actuator-20260905/shortstroke-freshreset199-vs250.md)
and [final comparison](../reviews/v44-actuator-20260905/shortstroke-freshreset199-vs298.md).

The proposed height-only reward is not selected: corrected250's rear core root
heights are.37250/.37259m, already inside its proposed3mm deadband about.375m.
Compared with the corrected seed, body height improves roughly3mm while
requested rear level-root clearance drops RL9.208→3.026mm and RR15.885→8.866mm.
Rear actual core means worsen to1.633/2.235mm despite100% core force release.
Thus the stronger current signal is the learned residual erasing swing lift,
not continued failure to raise the body. Additional height weight would be
nearly inactive precisely in those failing windows.

The next bounded hypothesis is smoothly attenuating sagittal residual
correction near scheduled peak swing. Offline projection shows a reason to
restrict it to the rear legs: front-right currently needs its learned lift
correction. A C2 envelope could preserve rear reference lift without reverting
to the unsuccessful globally zero-residual policy. This preserves scheduled
stance authority, not identical physical support or delivered trajectories.

### Rear-only swing correction: paired simulation launched

At 18:30:51, main launched the same continuation06 model250 with all-one control
scales versus `[1, .25, 1, .25]` in FL, RL, FR, RR order. Both retain the full
short-stroke learning configuration, corrected reset, nominal seed3101, two
25-second moving episodes, .9-second ramp and 3-second hold. Only the scale
vector differs in the supplied JSONs. No extra training is launched before
this discriminating comparison. See the immutable
[hypothesis](../reviews/v44-actuator-20260905/REAR_SWING_ATTENUATION_HYPOTHESIS.md)
and [launch/source record](../reviews/v44-actuator-20260905/rear-swing-pair-v1/launch.json).

The simulation-only implementation was read by main and independently reviewed.
It uses named hip/knee indices, unchanged abduction, exact all-one arithmetic
bypass, preserved parked authority, and unchanged joint/rate/packet/effort
limits. Nonidentity values require a strict exact-V44 ablation marker and are
explicitly rejected by the production export contract. No Pi implementation
or runtime parity is claimed. This is still an unqualified hypothesis.

Source inspection also found historical `_applied_gait_phase` aliases the phase
accumulator and therefore reports the next phase after target creation. The
pair deliberately preserves that existing reward/evaluator convention. Its
trace now independently copies the true target phase and per-joint mask;
the expected nominal offset is one control step, .005–.0058333 cycles.
A phase-consistency change must be a separate experiment, not an undisclosed
second intervention. Saved-data recalculation moves at most .625 mm of the
clearance target and both rear feet still fail; it cannot explain the observed
6–7 mm loss. See the [phase supplement](../reviews/v44-actuator-20260905/PHASE_ALIAS_SUPPLEMENT.md).

### Rear-only pair completed: reject the transfer, do not train it unchanged

Both arms completed at 18:33:48 with exit zero, reports and exact traces, and
no source changes during execution. The identity arm exactly reproduces prior
fresh-reset250 observations, actions, targets, physical q/qdot and estimated
effort. The candidate's per-joint mask agrees with its actual target phase to
5.2e-7; all front, abduction, scheduled-stance and parked multipliers are one.
No packet-cap violation or meaningful elapsed-rate excess occurred.

| Metric | Identity | Rear swing quarter |
| --- | ---: | ---: |
| Actual forward speed, m/s | .0107434 | .0071104 |
| Command fraction | 56.27% | 37.24% |
| RL worst-episode core corner p05, mm | .325 | 1.507 |
| RR worst-episode core corner p05, mm | 1.024 | .397 |
| FL scheduled release | 79.12% | 64.71% |
| Approximate effort soft-limit occupancy | 25.41% | 28.22% |
| Final heading error, mean absolute | 11.34 deg | 21.17 deg |
| Falls | 0/2 | 0/2 |
| Hold stable fraction | 98.89% | 100% |

The candidate reduces joint/body acceleration and improves the loaded-sole
tilt gate, but reduces speed by 33.8%, worsens rear-right clearance and front-left
release, increases estimated effort occupancy and adds a lateral-drift failure.
It is not an overall improvement. No continuation with this mapping is justified
by these data. All final-hold targets reach exact neutral; the rejection is
walking quality, not collapse at the end.

The [full comparison](../reviews/v44-actuator-20260905/rear-swing-identity-vs-quarter-model250-v1.md)
preserves every failed gate and per-leg result. Its generic comparator labels
different override vectors context-only and notes the missing crawl-enabled
field. The deliberately controlled intervention is documented by the paired
launch and source snapshots; neither arm enables sequential crawl.

A new [20-second progress recording](../reviews/v44-actuator-20260905/shortstroke250-progress-video-v1/videos/play/rl-video-step-0.mp4)
of unchanged model250 started at 18:37:44 and completed at 18:39:09. The clip is
H.264, 1280x720, 60 fps, 20.000 seconds, 1,015,729 bytes. Five frames were
extracted; main inspected frame02 and shared the local clip. It is illustrative,
not qualification and not the rejected mask. It uses the playback wrapper's
seed1701 and 15-second moving segment, distinct from the two-episode evaluator.

The next bounded discriminator is actor-only causal position-FD velocity input.
Earlier source/real-log analysis proves a velocity-definition mismatch and
frozen-state actor replay showed modest typical sensitivity, with larger
transients. Closed-loop impact is not established. The proposed evaluator-only
option changes only the 12 velocity input channels, with independent reset-aware
PRE-position history and actual physics-counter elapsed time. Raw physics,
reward/effort calculations, critic input and all gait gates remain unchanged.
This is a 60 Hz sensitivity proxy, not the Pi's quantized asynchronous 100 Hz
feedback process, and will not be called a hardware-matched model.

The evaluator-only implementation passed main and independent source review;
the controlled pair launched at 18:49:38 (supervisor wrapper44368). Both use
unchanged model250 and identity short-stroke configuration, with the same
two-episode nominal protocol. The actual consumed input and original full
observation are saved separately. Every first FD input is zero; subsequent
history must span exactly two physics steps and agree with cache timestamps.
No extra actor call, observation draw or physics step is introduced.

See [FD hypothesis](../reviews/v44-actuator-20260905/ACTOR_FD_PAIR_HYPOTHESIS.md),
[implementation](../reviews/v44-actuator-20260905/ACTOR_FD_CLOSED_LOOP_DIAGNOSTIC.md)
and [launch/source snapshot](../reviews/v44-actuator-20260905/actor-fd-pair-v1/launch.json).
Evaluator SHA256 at launch:
`8f8e7921ce8e279daaf564a0d3d1f0d808119106d50b5bcee8e46f7364eff497`.
The FD report is explicitly diagnostic-only even if individual numerical gates
pass. This is not a new checkpoint or training observation contract.

### FD pair completed: input mismatch is not the principal gait correction

Both arms completed at 18:52:41, exit zero, with all sources unchanged. Raw
input reproduces the preceding identity trajectory exactly. The FD arm changes
only indices26:38; all 134,064 other policy-input components remain exact.
Its 3,526 noninitial intervals each span two physics steps, and both first
inputs after reset use zero. Reconstructed FD error is below 3.8e-9 rad/s;
there are no clipped components, meaningful delivered-rate excesses or packet
cap violations. Final neutral targets and no-fall stops are retained. See
[recorded integrity results](../reviews/v44-actuator-20260905/ACTOR_FD_CONTROLLER_RESULTS.md).

Actual speed changes .0107434 to .0108801 m/s (+1.27%). Raw action second
difference decreases .0145203 to .0090681, but physical joint acceleration RMS
only decreases 7.6070 to 7.5102 rad/s2 (about1.27%). Rear worst-episode corner
p05 remains .330/.968 mm versus .325/1.024 mm; scheduled rear release slightly
decreases. The same seven gait gates fail. This proxy is not a substantive
walking correction and is not promoted to a training or Pi input change.
[Full numerical comparison](../reviews/v44-actuator-20260905/actor-raw-vs-fd60-model250-v1.md).
The generic comparison does not itself model the new input-source field;
interpret it together with the explicit FD treatment and launch record.

### Fresh07 versus warm08: actual bounded training started

Based on that result, main selected the bounded initialization comparison,
keeping rewards, geometry, raw training observations and all caps unchanged.
See the [selected hypothesis](../reviews/v44-actuator-20260905/FRESH_WARM_PAIR_HYPOTHESIS.md)
and [background rationale](../reviews/v44-actuator-20260905/FRESH_VS_WARM_TRAINING_RATIONALE.md).

The new warm bootstrap was prepared at 18:55:28 from original common-floor199.
Actor/critic and stored iteration199 were preserved, Adam cleared, and LR set
to5e-5. New bootstrap SHA256:
`66ad9d5ad7f33a461476419733fa035c408078615a4fbef17ea30fac58731990`.
The source optimizer LR had been .0002; this paired allocation deliberately
uses a shared conservative .00005, not an independently optimized fresh learner.
This limits what a finite fresh failure can establish.

Both actual training processes started at 18:56:46 and reached their first
updates successfully. Initial ETA was about35 minutes per arm. Shared settings
are 200 additional updates,128 environments,192 rollout steps,seed1701,fixed
LR5e-5,entropy.002,gamma.995,lambda.95,save interval1:4,915,200 new transitions
per branch. They differ in the full inherited initialization bundle, not just
actor weights; lifetime experience and realized exploration are not matched.

| Arm | Exact run directory under logs/rsl_rl | Supervisor wrapper PID | Milestones at50/100/200 new updates |
| --- | --- | ---: | --- |
| Fresh07 | drobot_v44_ablation_shortstroke-fresh-07/2026-09-05_18-56-52_fresh07_5cca5dec5ab641fb991087756e556652 |35176|49/99/199|
| Warm08 | drobot_v44_ablation_shortstroke-warm-08/2026-09-05_18-56-52_warm08_32068fb189f04b9fa580201b8c1ed29a |46264|248/298/398|

Each external supervisor saves prepared/process/completion records, immutable
source/config snapshots and exact checkpoint provenance. At 18:59:47 both had
completed13 new updates. Reward and action-distribution differences at this
stage describe initialization/exploration, not an accepted physical gait.
The experiment stops at200 per arm; checkpoint evaluation must not use a file
still being saved or confuse a numeric label with additional-update count.

### Automatic matched checkpoint evaluation: first results

At 19:12:39 PDT, the bounded local milestone watcher started (wrapper PID59276).
It evaluates exactly50/100/200 additional-update checkpoints from both branches,
with two nominal episodes each, fixed phase zero, seed3101, .019090909 m/s,
25 seconds moving, .9-second stop ramp and3-second hold. It checks immutable
source/config/run identity and checkpoint stability; final evaluation additionally
requires successful supervisor completion and the final checkpoint hash. It
cannot extend training, promote or deploy a model. Sources remain frozen through
both training completions and all six evaluations.

The first pair completed at19:18:29 with both exits zero and preserved reports,
traces and provenance in
[freshwarm-m50](../reviews/v44-actuator-20260905/freshwarm-milestone-watcher-v1/freshwarm-m50/completion.json).
The100-update pair started at19:21:03 while training continued.

| Nominal50-update result | Fresh07 model49 | Warm08 model248 |
| --- | ---: | ---: |
| Actual forward speed, m/s | .00636526 | .0106404 |
| Fraction of commanded speed |33.34%|55.74%|
| Rear-left/right scheduled release |25.98% /25.73%|62.32% /67.57%|
| Rear-left/right worst episode core-corner p05, mm |-.161 /-.202|.445 /1.017|
| Approximate soft-effort occupancy |44.05%|25.17%|
| Physical joint acceleration RMS, rad/s2 |7.016|7.406|
| Falls / episodes |0 /2|0 /2|
| Settled episodes |2 /2|2 /2|

Neither is qualified. Both fail speed, rear release/clearance, contact-slip and
soft-effort criteria; fresh also fails paired release quality, warm support-sole
tilt. Early warm-start advantage is not proof of eventual convergence or a
fair lifetime-data comparison. These are intermediate results, not a reason to
extend the finite allocation. The generic
[offline comparison](../reviews/v44-actuator-20260905/freshwarm-m50-comparison-v1.md)
conservatively flags missing sequential-crawl metadata; the explicit frozen
invocation/config records establish that both used the same diagonal setup.

The100-update pair completed at19:26:42, again with successful preserved
completion records. Fresh speed increased to .00881493m/s (46.17% of command),
but rear-left core p05 stayed below ground (-.092mm) and rear-right reached only
1.461mm. Rear release was38.77% /34.75%. Warm speed decreased to .00953288m/s
(49.93%), with rear p05 -.256 /-.134mm and final absolute heading error28.77deg.
Both still stopped without falling. Warm's checkpoint hash exactly matches
the preceding100-update continuation06 final298; this is reproducibility of
the same initialized learning trajectory through that point, not independent
evidence from a new seed. The next100 updates remain part of the bounded plan.
[Full100-update comparison](../reviews/v44-actuator-20260905/freshwarm-m100-comparison-v1.md).

The [m50 geometry audit](../reviews/v44-actuator-20260905/FRESHWARM_M50_GEOMETRY.md)
distinguishes sustained low posture from startup settling: fresh root height
stays about364.1-364.4mm across consecutive5-second steady windows, whereas
warm stays about371.5-371.7mm. Fresh retains requested rear lift but much of it
is consumed by low body height; warm raises the body but cancels much of the
rear-left reference lift. These ordered kinematic substitutions are descriptive,
not proof that independently raising the body is physically feasible. The active
height reward is the legacy metre-squared expression, not the previously proposed
3mm-deadband normalized-height treatment.

The [m100 geometry follow-up](../reviews/v44-actuator-20260905/FRESHWARM_M100_GEOMETRY.md)
shows fresh's root rising to366.952mm, but rear-left requested level-root lift
shrinking from15.899 to11.929mm. The net rear-left clearance barely changes;
rear-right improves. Warm rises further to373.522mm while both rear clearances
regress. Higher body position alone is therefore not a universal correction.

The [reward-unit audit](../reviews/v44-actuator-20260905/FRESH_WARM_REWARD_CONFLICT_AUDIT.md)
confirms that active stalling is penalized strongly, rather than explicitly
preferred over walking. It also distinguishes stochastic raw-action regularizers
from delivered and physical motion: the former are much larger in these logs.
Those aggregate magnitudes are not policy gradients or proof of a cause. Two
conditional follow-ups remain separate: motion-only height shaping for a
persistently low fresh policy, or causal training-action/target capture to
disentangle exploration costs. Neither is mixed into the current pair, and
neither warrants weakening physical limits or acceptance gates.

At19:28:34 a separate local20-second playback was launched from fresh07 model99,
SHA256 `50ea1deb9c54ac4f8e35ad336fca6b647b8e4f67e94e9d541956b84ce1246788`,
with unchanged identity settings. It is a progress recording, not an acceptance
episode, and its isolated review exports cannot become a deployment package.

That playback finished and its wrapper was no longer running at19:31:02.
The [local progress clip](../reviews/v44-actuator-20260905/fresh07-m100-progress-video-v1/videos/play/rl-video-step-0.mp4)
is20.000seconds,1280x720,H.264,60fps,967,689bytes, SHA256
`b823bdb0f3aebced2bfbe1980d4dddadac871fdf3b93a95a34e319041f97d7ba`.
The copied checkpoint hash matches the evaluated model99. Five frames were
extracted for local review; frame02 was inspected. No fall-free or gate claim
is inferred from that single frame or substituted for the evaluator results.

### Existing real recordings cannot supply a missing crawl control arm

The [archive-only coverage audit](../reviews/v44-actuator-20260905/REAL_TRIAL_CALIBRATION_COVERAGE.md)
finds all four available trials are RL, not hardcoded crawl. Only September5
records host-completed motor-output events, so older requested targets cannot
be treated as actual delivered actuator-input histories or pooled into the
same dynamics fit. August26 additionally has a59-tick (5.19-degree) different
ID4 calibration center; the three later calibration snapshots agree. Per-motor
voltage is sparse and does not measure fast supply droop or actual torque.
This is a limit on inference, not evidence of a newly diagnosed hardware fault.

The [policy-history research note](../reviews/v44-actuator-20260905/HISTORY_POLICY_RESEARCH.md)
records a possible future causal-history experiment with primary sources. The
current actor has current proprioception and one delivered-target vector, not
a temporal state estimator. History may help distinguish delayed response, but
cannot repair an incorrect plant. No observation/architecture change is mixed
into this active initialization comparison.

### Bounded fresh/warm training completion observed

Fresh07 completed at19:43:51 and warm08 at19:44:03 PDT. Each supervisor records
exit0, all200 requested labels in order, correct isolated run identity, saved
effective configs, a successful learn-return marker and the final checkpoint.
There were no forced terminations or supervisor/inspection errors. Both remain
`UNQUALIFIED_ABLATION`; neither completion automatically starts more training.

| Arm | Final label | Final checkpoint SHA256 | Reported training seconds |
| --- | ---: | --- | ---: |
| Fresh07 |199|`55dcba3c408613e6e6cf7a58ba7fb6782caf9b95c501001008cc047dfcd36114`|2814.92|
| Warm08 |398|`e06f5a7e11eb66ad8477f1c9e572559fb048c395049b39a28099b29c09923bc4`|2826.05|

Observed completion records:
[fresh07](../reviews/v44-actuator-20260905/shortstroke-fresh07-supervisor-v1/completion.json),
[warm08](../reviews/v44-actuator-20260905/shortstroke-warm08-supervisor-v1/completion.json).
The pair collected9,830,400 new transitions in total; equal additional data does
not erase warm's preceding lifetime training. Final evaluation is a separate
required step and uses the final hashes after observed completion.

### Final matched evaluations completed: prefer fresh direction, qualify neither

The watcher completed all three milestones and all six evaluator processes at
19:47:33 with failure=null, no cleanup/termination required and no source drift.
Its process has exited. The fixed200-update training allocation is not extended
automatically. The [final comparison](../reviews/v44-actuator-20260905/freshwarm-m200-comparison-v1.md)
and [complete watcher evidence](../reviews/v44-actuator-20260905/freshwarm-milestone-watcher-v1/completion.json)
retain every gate, not only favorable averages.

| Nominal final result | Fresh07 model199 | Warm08 model398 |
| --- | ---: | ---: |
| Forward speed, m/s |.00922201|.00573555|
| Fraction of command |48.31%|30.04%|
| Rear-left/right scheduled release |66.38% /64.18%|75.35% /66.64%|
| Rear-left/right worst episode core-corner p05, mm |3.219 /1.311|1.461 /.742|
| Approximate soft-effort occupancy |32.32%|31.83%|
| Physical joint acceleration RMS, rad/s2 |6.242|9.156|
| Final absolute heading error |2.26deg|80.00deg|
| Settled episodes |2 /2|0 /2|
| Stable/four-foot hold fraction |100%|4.72%|
| Falls / episodes |0 /2|0 /2|
| Failed numerical gates |5|11|

Warm's final-speed-near-zero reading does not mean a successful stop: it fails
the settle and stable/four-foot hold criteria. Its logged mean_settle_time_s=0
is a no-success summary value, not instantaneous settling. Fresh preserves
front clearance, straightness and stopping and improves both rear feet versus
its early checkpoint. Nevertheless it does not pass the physical gait gates
and is not exported, deployed or relabeled as ready. Wider randomized
qualification is not substituted for correcting these nominal failures.

### Final geometry, load calculation and selected next comparison

The [m200 geometry audit](../reviews/v44-actuator-20260905/FRESHWARM_M200_GEOMETRY.md)
finds fresh's sustained moving root height370.477mm and rear-core heights
371.261/371.197mm versus375mm target. Its legacy height cost averages only
-.0000632313raw before the common.1scale. Direct-target-phase rear worst p05 is
3.004/.760mm; those are supplementary values, not rewritten official gate
results. Frozen-pose translation to exactly375mm gives6.819/4.819mm, still short
for rear-right. This counterfactual is not a feasible independent body command;
the proposed3mm deadband also does not force exact375mm height.

The [conditional gravity/torque study](../reviews/v44-actuator-20260905/QUASISTATIC_TORQUE_BUDGET.md)
uses recorded m50 body masses/COM and joint transforms, including both swing
core centers. Under optimistic whole-footprint pressure support, planned
diagonal poses admit below-soft vertical-only loads; near-lowest edge points
are more restrictive and sometimes statically infeasible. Tangential optimized
solutions frequently use the friction boundary. This neither identifies actual
drive torque nor establishes a dynamic gait. It does not support increasing
the.980665Nm soft or2.6478Nm hard threshold. Actual rear feet lie behind their
hips; a tentative all-feet-forward reading of a local gait parameter was rejected
using the recorded transforms before any geometry change.

On that bounded evidence, main selected the
[height-only paired hypothesis](../reviews/v44-actuator-20260905/HEIGHT_ONLY_PAIR_HYPOTHESIS.md):
two200-additional-update branches from exact fresh07 final199, unchanged reward
control versus motion-only normalized height. Keep existing3mm/10mm/weight4
parameters, use commanded-motion quintic blending, preserve exact legacy hold
and tilt, other rewards, inputs, references and all physical/acceptance limits.
Both branches reset Adam and curriculum consistently; this is not an
uninterrupted continuation or a treatment of warm's different failure.
Implementation/source review and explicit launch records are required before
calling the next training active. The3mm deadband is intentionally limited, not
a promise that height alone will fix the remaining rear-right deficit.

Final fresh07 playback launched at19:49:36 (wrapper35100) and had finished by
19:51:25. The [latest local clip](../reviews/v44-actuator-20260905/fresh07-final200-progress-video-v1/videos/play/rl-video-step-0.mp4)
is20.000s,1280x720,H.264,60fps,994,445bytes; SHA256
`aea81e34ca7780d91057fc11e362ddcf153b487240e3aafac335daf5f42959ed`.
The copied checkpoint hash matches fresh final199; frame02 was inspected from
five extracted frames. Original intermediate clips remain preserved.

### Height-control09 / height-only10: source review and actual launch

The opt-in [implementation](../reviews/v44-actuator-20260905/HEIGHT_ONLY_REWARD_IMPLEMENTATION.md)
was reviewed independently by main and the controller agent. Default false and
parked height retain exact legacy arithmetic, tilt and combined-normalization
branches are unchanged, active run speeds must exceed the standstill deadband,
conflicting modes are rejected, and production requires the new flag false.
No tests, compiler/build or unrelated verification was run; the authorized
actual training is the subsequent execution described here.

The first wrappers at20:01:00 failed before Kit: the central trainer hashes
suffixes longer than24 characters, while the study wrapper had used literal
long bootstrap directories. No training updates or simulator starts occurred in
those attempts. Old wrapper logs, bootstrap directories and exact failed source
are preserved. The corrected wrapper uses `height-control-09` / `height-only-10`
and rejects suffixes longer than24 before creating a bootstrap. Nothing about
the scientific configuration changed.

New optimizer-only reset bootstraps were prepared at20:02:19/21 from fresh199;
their hashes are control
`948b356afadbec939c1ef59ccd9e0f7f74aada1277c09b8377f4173df6eb15e5`
and height
`6b39ebe28ea107263170c175feb33b1d3f460953e69b2189a46d206579b6c601`.
Both preserve actor/critic/stored199, clear Adam and use LR5e-5 without input
rescaling or a curriculum sidecar. Different serialized hashes include distinct
preparation metadata; their common source and helper behavior are recorded.

Actual Kit children started at20:02:43. First updates were confirmed by20:03:57,
with initial remaining-time estimates near32minutes per arm. New exact runs:

| Arm | Experiment / exact run | Wrapper / Kit PID |
| --- | --- | --- |
| Control09 | `drobot_v44_ablation_height-control-09/2026-09-05_20-02-49_hctl09_5bcb5eb66b0147eaafb96f94e2d79004` |22916 /41272|
| Height10 | `drobot_v44_ablation_height-only-10/2026-09-05_20-02-49_honly10_189df7f591bb4c3aac5cc1fedf99af36` |15592 /38756|

These live in `logs/rsl_rl`; new supervisor records are
[control09](../reviews/v44-actuator-20260905/height-control-09-supervisor-v1/)
and [height10](../reviews/v44-actuator-20260905/height-only-10-supervisor-v1/).
Each runs200 additional updates199..398 with matched248/298/398 milestones.
Source/config/hypothesis are frozen through training and milestone evaluation.

Frozen new source hashes: env
`936798ea78d557bfbcb2e8d89596699268368b523723375e1c02ad610213ecb0`, cfg
`0d7809ec48cf0a334aa8f0e2ab383a85970edd29fd92334423f434483a60d64e`, loader
`926b066431cec5f49fcae15e2846e8fdfa6d1601be75741bf22859e43f9113de`, production
guard `f333f00278ab4f84c0a15cc7f18de7a33cd2af953fd53b86b1042be8acd46719`.
The selected hypothesis hash is
`98745c9101d227348fe730451d52a9ae4e13f2566b6c8c52db0f73d828f67913`;
the corrected study launcher hash is
`2b3e0848204179beda8e3e6ad3d72a6a1f11dcc2b1757f7b58b0aab2751e7be3`.
All previous source snapshots remain intact rather than being backfilled.

### Passive contact reporting: next diagnostic prepared, not executed

Installed-source research found a raw normal-contact getter on the existing
unfiltered contact view. It may provide individual solver contact points and
force-weighted support centers without changing the sensor setup. This is not
a hardware pressure measurement, and its runtime availability/coverage remain
unverified. Point-buffer capacity, sensor mapping, partner actor identity,
normal-force sum agreement and120Hz timing must be checked before interpreting
the result. Tangential friction would require separate filtered reporting and
is not part of the first proposed capture. The current training package remains
frozen. [Installed API and diagnostic design](../reviews/v44-actuator-20260905/CONTACT_MANIFOLD_CAPTURE_PROPOSAL.md).

The selected one-episode diagnostic input is control09 model298 (checkpoint
`f624e14294f83947d46a433c0eb27747921313ed6c9cac489c47e57fe45e9a89`),
not an automatically selected newest file. New study-only launcher/recorder
sources reuse the existing raw normal-contact view and wrap each existing
physics call once, without changing targets or adding physics/actor calls.
Planned capture is480 late-run physics steps plus120 final-hold steps at120Hz,
including raw normal forces/points, independent aggregate forces, body poses,
COM velocities and local COM poses. The source-reviewed
[implementation contract](../reviews/v44-actuator-20260905/RAW_NORMAL_CONTACT_IMPLEMENTATION.md)
requires restoration before writing and separate external process-exit/artifact
completion evidence because Kit fast shutdown can bypass Python `finally`.
Source review also caught an unsigned-ID host-copy portability concern before
launch. The final helper copies unsigned Warp values directly through NumPy;
floating snapshots are unchanged. Its frozen hash is
`c98e25cb598b5ba4925f46f73cadb82961f3a61d7924bd6835a6713f024ad214`;
launcher hash is
`1ebb05cbcc33246d346a7cc20ce32eec620711321cd92298f96947a36e2496a5`.
The actual diagnostic launched at20:49:03.6803562PDT, retained external cmd
PID15964 / execution session86369, with a15-minute deadline applying only to
its owned process tree. New outputs are
`raw-normal-control298-v1`, with adjacent unique stdout/stderr logs. Its
initial IsaacLab banner is observed; capture/completion are still pending.

A [target-space smoothness alternative](../reviews/v44-actuator-20260905/TARGET_SPACE_SMOOTHNESS_HYPOTHESIS.md)
has been documented, not selected or implemented. It distinguishes delivered
setpoint acceleration from raw dimensionless policy differences, without
claiming either equals physical acceleration. Final/contact evidence and an
offline scale audit must precede choosing any further training comparison.

### Height-pair50-update evaluation: no treatment benefit observed

Both exact248 evaluations completed successfully at20:29:49, with unchanged
checkpoint/source/config/trace links and exit0. Both fail the same five gates:
speed, every-leg swing release, rear corner clearance, contact slip and estimated
soft-effort occupancy. Control speed is.0111866m/s; height-only is.0111452m/s,
against.0190909m/s commanded. Both improve over original fresh199's.009222m/s,
so that improvement cannot be credited to the height treatment.

Control versus treatment rear official worst-episode core p05 clearance is
3.739/1.563mm versus3.469/1.164mm (RL/RR), while rear scheduled release is
66.38/65.10% versus65.52/64.33%. Approximate soft-effort occupancy is32.230%
versus32.185%; both have zero falls, stable four-foot hold and.9s settling.
Direct-phase saved geometry finds mean steady body heights370.8432 versus
370.8491mm, only.0058mm apart. These tiny differences from one matched seed do
not establish statistical superiority; there is no useful treatment response yet.
The later100/200-update results remain pending, rather than being inferred from
these first two episodes. The100-update pair began at20:30 using exact298 files.

An [offline completed-log calculation](../reviews/v44-actuator-20260905/heightpair-first100-reward-log-v1.json)
pins the clean first100-update log prefixes. During new indices75..99, mean
logged height terms are rounded0.0000 control and-.008544 treatment, while
action-second-difference terms are-2.121476/-2.138828 and joint-acceleration
terms-.024088/-.025092. Both have identical mean smoothness multiplier.909688.
These are rounded episode aggregates, not instantaneous paired costs, gradients,
or a causal decomposition. The treatment is active, but its aggregate magnitude
does not by itself show that a larger weight would improve physical behavior.

### Height-pair100-update evaluation: the same failures persist

Both exact298 evaluations completed at20:35:39, exit0. Empty live log files
were buffered output, not failed simulations: actual owned Kit processes were
active, then published full reports/traces and version-banner-only stderr.
No restart or duplicate evaluation was launched.

Control versus height speed is.0101209/.0100562m/s (53.01/52.68% of command).
Both fail the same five gates as50. Height's rear official core p05 is
2.067/-.125mm versus control1.895/-.128mm; RR is effectively at ground level.
Approximate overall soft-effort occupancy32.065/31.973% is nearly unchanged;
run-only occupancy32.2985/32.3039% does not improve. All legs still fail the
15mm/s contact-patch slip gate. Both remain no-fall with successful settling and
passing stable/four-foot hold.

[Direct-phase geometry](../reviews/v44-actuator-20260905/HEIGHT_PAIR_M100_GEOMETRY.md)
finds371.8266/371.8303mm mean steady root height, a treatment difference of only
.0037mm. Both rise about.98mm versus50 but suppress more rear target lift and
lose actual clearance. Height alone is not explaining the current bottleneck.
This is not a final200-update conclusion or causal proof about the shared
smoothness curriculum. Final results and a separately instrumented contact
diagnostic are the next decision inputs; no larger weight or new architecture
has been silently introduced.

## Completion evidence and export guards

### Height pair: both actual training completions observed

At20:48:43PDT, both external supervisors recorded `completed_observed`, exit0,
all11 completion criteria true, all200 labels199..398, successful learn-return
markers and final398 checkpoint hashes. No interruption/forced termination was
reported. Training wall-time observations are2750.20s control and2750.13s height.
The final control checkpoint SHA is
`c06a08bfa0c9bc0fea5ee7f20cd49061d17068c1ac02558e42019ef3aacc49bc`;
height is
`e3bfc495584e6428e5d196af088184c2d84fb4bf2ec463af300b8ff3bb4d6b39`.
The milestone watcher still requires their final paired evaluation before
closing this experiment. Source/configuration stay frozen through that and
the new contact diagnostic; no automatic training extension is authorized by
these completion records.

### Height pair closed: reject the proposed treatment

At20:52:34 the watcher completed all three milestone pairs, six evaluations,
with `completed_observed`, no failure and no owned-child cleanup. Both final398
reports preserve the same five failures as50/100. Control/height speed is
.00900393/.00900611m/s (47.16/47.17% of command); approximate soft-effort
occupancy29.819/30.119%; rear scheduled release60.14/63.48% versus58.89/61.63%.
Official rear core p05 is control.259/.781mm versus height-.092/-.079mm, far
short of5mm. Every leg still fails the15mm/s slip gate. Both remain no-fall,
successfully settled and four-foot holding; joint acceleration is6.186/6.318rad/s².

The treatment produces no useful height, speed or clearance advantage. Both
arms request progressively less rear lift while the body rises. This is a
rejected height-only pilot, not successful running, not a reason to relax a
gate, and not a guarantee that an even larger height weight would help. Keep
the stronger earlier fresh199/selected intermediate evidence rather than
assuming the last checkpoint is best. Final geometry is recorded separately
from preserved official phase/gates.

### Raw-contact capture exited successfully; independent audit underway

The retained external cmd process15964 exited0 at20:51:15.8243502PDT without
forced termination. Its [external exit record](../reviews/v44-actuator-20260905/raw-normal-control298-v1/external-process-exit.json)
is deliberately process-only pending full artifact audit. The contact header
reports exactly480 late-run steps (controls1290..1529) and120 final-hold steps
(1704..1763),13 sensors,52 contact-buffer capacity, no truncation, lifecycle or
physics errors, zero unknown force-coverage steps, and observer restoration
before writing. The sole resolved other actor is
`/World/ground/terrain/GroundPlane/CollisionPlane`. These header observations
are not a substitute for independently checking raw ranges/counters/force sums,
source snapshots and motion equivalence. Capture/report/trace are retained.

To show the current limitation, a new20-second control398 recording launched
at20:54:44 (external wrapper53124, execution71903), using the existing isolated
review-copy wrapper, nominal.019090909m/s,15s motion then stop/hold. It is not a
qualified candidate video or a deployment export. Output is
`control09-final200-progress-video-v1`; completion/video inspection pending.

The new ordinary-Python supervisor captures the child process outside Kit so
fast shutdown cannot skip completion evidence. It requires complete iteration
labels, matching source/config/run identity, the expected final checkpoint and
successful exit/learn-return markers. UUID-suffixed run names avoid timestamp
directory reuse. The third crawl launch exercised these guards through all
150 updates and observed successful completion. The two earlier launch failures
remain documented separately. No separate test suite or smoke run was launched.

Production V44 contract checks now pin nominal stiffness/damping scales to 1/1
and synchronized gain telemetry to true. Gain ablations remain in explicitly
separate experiment names; a faster hypothesis cannot silently become a
production-qualified model. Nothing has been exported to or deployed on the Pi.

## Current status after the bounded initialization pair

- Completed and preserved: actuator/mass/geometry audits, recorded-target
  replays, crawl/reference physics probes, and the unsuccessful150-update crawl
  pilot with final evaluation and local video.
- Completed but unqualified: coefficient-only, worst-foot-aware and short-stroke
  diagonal continuations, each100 additional updates, intermediate/final
  evaluations and preserved checkpoint/config/source provenance.
- Completed: passive120Hz PhysX capture, exact actor-input replay, simulator/Pi
  velocity-definition audit, and correction of stale initial evaluator inputs.
  Corrected nominal baselines remain unsuccessful; the bug was real but was not
  the main cause of dragging.
- Completed and rejected: rear-only swing attenuation transfer. Exact traces
  establish correct mask delivery and unchanged safety caps, but closed-loop
  rear-right clearance, speed and front-left release regress. No masked
  continuation is launched. Recorded FK shows that root-height, attitude and
  loaded tracking losses consume the extra requested lift.
- Completed: actor-only60Hz FD comparison. Typical raw-action jitter improves,
  but physical acceleration barely changes and all seven failed gates remain.
- Completed: fresh07 and warm08, each200 additional updates, all three exact
  matched milestones and their six nominal evaluations. Fresh is the stronger
  direction but remains unqualified; warm final is rejected including hold
  regression. Their fixed allocation is closed, not automatically extended.
- Training complete: height-control09 / height-only10 from fresh final199,
  200 additional updates each, both observed successful20:48:43. All three
  milestone pairs completed20:52:34; no useful height-treatment benefit, same
  five failed gates and declining rear clearance. This allocation is closed.
- Contact capture exited0 with all600 expected samples; independent offline
  contact/support/motion audit is active. A local final-control progress video
  is recording. Target-space smoothness remains a proposal plus offline scale
  audit, not a selected new training job or changed controller.
- No qualified running/walking replacement, Pi deployment, physical motion,
  commits/pushes or hosted review. Only the expressly authorized training,
  simulation/actor evaluations and recorded-data/source analysis have run;
  unrelated tests/builds/hardware checks have not.
