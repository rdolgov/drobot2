# IK-guided diagonal walking RL

Requested September 20, 2026: train and evaluate a flat-ground policy based
on the diagonal IK walk, with longer paired strides, immediate support-ankle
push, a level body and smooth low-sway movement; record a simulation video.
This request explicitly authorizes training, evaluation and recording for
this task. It does not authorize physical robot motion or policy deployment.

## Reference and learning contract

The separate `ankle_v25_ik` package inherits the existing 16-motor physics,
foot geometry and strict physical step accounting. It replaces the old V24
reference and output controller, which repeatedly failed rear-foot clearance.
It starts a fresh zero-mean residual actor instead of using a stair policy or
the unsuccessful V24 checkpoints. No older experiment source is overwritten.

The trajectory follows the current IK stance (80 mm fore/aft, 288 mm down),
35 mm lift, alternating FL+RR / FR+RL, and 80–100 mm final world strides.
Stride grows from 60% to 100% over the first 200 iterations. Nominal commands
are 0.020–0.025 m/s. Each placement is two nominal seconds, before termination
or stopping; a full cycle is four seconds. Startup lasts 2.5 seconds.

Support ankles follow planned hip/knee angles plus bounded, filtered body-pitch
correction. The early 20 mm body advance occurs during lift, with 30% in the
first 10% of lift, following the V6 ankle-lead proposal. Lifted feet move
forward with the body before finishing the world stride. Ankle pitch also
compensates learned hip/knee changes before applying its own learned residual.
All 16 joint commands share the most restrictive interpolation fraction,
with 60 Hz output, 4-degree packet ceiling and 200 degrees/s sagittal limit;
abduction keeps 60 degrees/s. Missed output frames do not allow catch-up bursts.

The training clock is timed, without the hardware encoder-onset or endpoint
gates. Earlier training repeatedly ended before useful exploration at swing.
Timed phase advance is **not** counted as a successful step. This difference
is explicit: the learned model is not a drop-in replacement for V23 or the
hardcoded Pi controller. A hardware handoff needs a matching reference,
observation, action mapping and guarded controller implementation.

The fresh actor has 82 proprioceptive inputs and 16 bounded residual actions;
the critic has 101 inputs including simulation contact/COM information.
PPO uses a Gaussian with log standard deviation clamped to 0.08–0.60, initial
0.30, fixed 0.0001 learning rate and a zero-initialized mean output layer.
This avoids relying on the Beta policy that collapsed in an earlier run.
Residuals are filtered at 100 ms and bounded by joint and calibrated ankle
limits. No contact forces or simulator sole poses enter the actor.

## Reward and independent measurements

Existing strict full paired-landing counts require real flight of at least
100 ms, at least 8 mm clearance, at least 65% of the intended world stride,
limited sideways foot travel, continued opposite-pair support, and a quiet
four-foot plant. Each foot and each pair is counted separately. Standing,
scraping and a single front-foot step do not count as full paired walking.

Rewards combine:

- Measured minimum clearance of the selected pair and actual completed long
  strides, with extra credit for the least-used pair/leg.
- Early measured support-ankle movement during lift and forward body motion
  during the push, conditional on opposite-pair contact.
- Small differences between selected feet's heights to encourage synchronization.
- Level projected gravity and low roll, pitch and yaw rates, angular/target
  acceleration, action changes, lateral/vertical velocity, yaw and foot slip.
- Flat supporting soles, retained support and limited ankle saturation.

The additional dense clearance term uses actual opposite-pair contact without
requiring a static COM hull as a hard gate. Strict qualification metrics retain
their original criteria; a reward improvement alone does not qualify walking.

## Physical assumptions

The 3.39514659 kg estimate uses the earlier 7 lb twelve-servo robot plus four
55 g servos. Mass and inertia are uniformly scaled from the retained USD;
the COM distribution has not been measured on the updated robot. The original
0.8825985 Nm working torque cap, torque-speed envelope, 120 Hz physics,
collision geometry and joint bounds remain. Training randomizes motor
strength/rate from 85% to 100% and adds occasional single-frame output holds.
These are existing modeling assumptions, not measured loaded servo dynamics.

## Artifacts

Each run has source/calibration provenance, hashes, exact arguments, status,
optimizer records and checkpoints. Independent first-episode evaluation covers
both starting diagonals with nominal and 85% motors; post-reset episodes are
excluded. Videos show actual rendered simulation states with phase, selected
pair, displacement, tilt, foot clearance/contact and successful-step counts.
Reset frames are excluded if a trial terminates.

Read `simulation/reviews/ankle-v25-ik/` for actual progress and results.
Neither writing this document nor launching a process means training succeeded.
Pi V6 remains staged from the earlier task; this RL task does not activate it.

## Second candidate: physical paired clearance

Independent checkpoint-100 evaluation of the first candidate moved 0.22–0.51 m
in 32 seconds, with 1.87–2.35 degree RMS tilt, but no complete paired steps.
The rear feet scraped instead of performing qualifying flight and long strides.
The first run was therefore stopped after iteration 313; its saved iteration-300
checkpoint and all earlier evidence are retained. It did not finish its originally
configured 500 iterations, and its status explicitly records the early stop.

The separate `ankle_v25_pair` package starts a fresh actor with a second seed.
It removes positive forward-motion credit without actual minimum pair clearance
during lift or a completed physical pair during the final push. Backward motion
still incurs its original cost. It increases minimum-pair clearance and height
synchronization rewards and adds a stronger level-body cost during swing.
It also expands the bounded abduction/hip/knee/ankle residual ranges to
0.14/0.22/0.35/0.15 radians, with up to 25 mm extra swing lift and a 12 mm
fore/aft window. The existing joint limits, output rates, mass and torque caps
remain unchanged. These reward and control-range changes form one candidate;
they are not an isolated ablation of a single term.

This candidate uses a 100-iteration stride curriculum and a 300-iteration run.
Its rollout must use `ankle_v25_pair/run.py`, while first-candidate weights use
`ankle_v25_ik/run.py`. Both have separate frozen source snapshots and contracts.
Checkpoint selection is based on actual clearance, strides, displacement and
body motion in independent evaluation, not training reward alone.

## Completed results

The second run completed all 300 iterations and saved `model_299.pt` with
SHA-256 `8c0b253c853b4654038833d91e04677a85c97c08a38054cf115ee3db2ab2b4b7`.
The two runs together performed approximately 10.06 million environment
transitions; the first run's final unsaved iterations are not a released model.

The final candidate survived all four 32-second first-episode trials, advanced
5.2–26.0 cm, and had 1.05–1.56 degrees RMS body tilt over the entire trial.
It completed **zero strict paired landings in all four trials**. Rear-right
clearance improved in some trials, but rear-left lift and long placements did
not become reliable; performance also depended on the starting diagonal and
motor-strength condition. Lower tilt and some forward progress do not certify
the requested walking pattern. No policy is selected for physical deployment.

The 36-second review compares the same 2.5–14.5 second interval at real speed
for the IK reference, initial RL checkpoint 100 (more forward translation),
and final pair-focused checkpoint 299 (flatter body). Full recordings and
their own traces/reports are retained. All use the same close camera and show
measured phase, contacts, clearance, displacement and complete-step counts.

See [the results and videos](../reviews/ankle-v25-ik/README.md). No physical
motor check, Pi activation, UI change, Git commit or push was performed as
part of this training task.
