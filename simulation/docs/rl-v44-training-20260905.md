# V44 actual training and evaluation — September 5, 2026

Subsequent autonomous work is recorded in the
[actuator/contact continuation](rl-v44-actuator-calibration-20260905.md).
The completed-session status below is retained as historical evidence.

Status: **actual PPO training and checkpoint evaluations completed. The final
200-update wider-authority pilot improved, but still fails nominal walking
acceptance. No qualified new model, no active training job and no deployment**.

The user explicitly authorized actual training, then checkpoint evaluation while
training. Simulation videos stay local. No Pi connection, torque/motion command,
service change, model selection, Git commit/push, or unrelated test suite is part
of this run.

## Inputs and reproducibility

- Implementation: [V44 design and acceptance criteria](rl-flat-contact-hardware-matched-v44.md).
- Baseline: [September 5 real trial](../../debug/walk-trials/20260905T193441588801Z-2a2a9832/ANALYSIS.md).
- Repository HEAD at launch: `4b54328423188c1f9208c5fe73cf95026678c743`, plus
  existing uncommitted source changes. HEAD alone does not reproduce this run.
- Python/runtime: `C:\isaacsim\python.bat`, installed Isaac Lab, PhysX backend.
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB reported memory.
- Profile: `flat-contact-hardware-matched-run-stop-external-rear-payload`.
- Fresh weights (no V43 bootstrap), seed 1701, 128 environments, initial budget
  1,500 PPO iterations. Fixed learning rate `3e-5`, entropy coefficient zero.
- Saved effective configuration confirms a 10% nominal / 90% randomized
  environment mix, 64 rollout steps per PPO iteration, parked-contact override
  enabled and the 3-second speed window.
- Commands 0.03–0.05 m/s, cadence 0.35–0.45 Hz. This is a contact-correction
  stage, **not a high-speed running qualification**.
- Generated `references/v44_flat_contact_low.json` and
  `references/v44_flat_contact_high.json` before training. Both passed their
  generator's built-in analytical reachability/soft-limit/rate rejection checks.

Pre-launch source review found two reward mismatches with the slower gait:

1. Added V44-only `gait_phase_table_park_contact_override`: at zero command all
   four feet are scheduled support, avoiding an inherited swing penalty during
   the desired planted neutral stop. Older tasks keep their original behavior.
2. Increased V44 sustained-speed averaging from the inherited 0.75 seconds to
   3.0 seconds, covering a full slow gait cycle rather than scoring individual
   swing phases as stalls. Both settings are included in the provenance contract.

## Run timeline

- 13:26 PDT: launched hidden PowerShell process 44136 with the command below.
- Isaac Lab log name: `2026-09-05_13-26-33` (wrapper adds `manual-headless`).
- Runtime reports physics 120 Hz and policy 60 Hz on `cuda:0`.

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet flat-contact-hardware-matched-run-stop-external-rear-payload `
  -Fresh -Iterations 1500 -NumEnvs 128 -Seed 1701
```

Console logs: [stdout](../reviews/v44-training-20260905/train.stdout.log),
[stderr](../reviews/v44-training-20260905/train.stderr.log).
Checkpoints/configs are under
`logs/rsl_rl/drobot_commanded_run_v44_flat_contact_hardware_matched_stop_external_rear_payload/`.
Completion must be established from the actual final iteration and checkpoint,
not just the process exiting. The training entrypoint attempts to write
checkpoint/config/reference-bound `.training.json` files after it returns;
installed Kit fast shutdown can bypass that hook even after completed training.
If the hook does run, an interrupted partial run can also receive provenance
and is not thereby a completed 1,500-iteration run.

Source/reference SHA-256 at launch (before subsequent training revisions):

| File | SHA-256 |
| --- | --- |
| `references/v44_flat_contact_low.json` | `5f3d5024efbdaff6bb98137a4b99383724ed59866a47d56b12b913e35d6428c2` |
| `references/v44_flat_contact_high.json` | `373db80961b2fb77e99e55cb6d3ec3023edc59bc4384d3cb9f00368fd296c6d4` |
| `commanded_walking_env.py` | `e92971f6beea005355afaf657c943c93a4b347f199bb96e6e9063c5888083332` |
| `commanded_walking_env_cfg.py` | `190be48c3af9c803ba9ab2b14b3e5d26592c23710337408d806e451aca472ff1` |
| `agents/rsl_rl_ppo_cfg.py` | `185d43940316904c36fdcd5138dedaf35fd7b2936690347d3c6ab3b4db32ec42` |

## Evaluation plan and acceptance

Evaluate deterministic checkpoints with nominal and randomized physics, including
actual progress at requested speed, lateral/yaw drift, all-leg participation,
world-frame sole tilt, minimum swing-corner clearance, slip, and graceful stop.
Read clearance together with sample counts: no swing opportunities is not a
clearance success. Use 20-second videos for visual review; do not infer physical
hardware performance from simulation or reward alone.

## First completed evaluations

| Checkpoint / evaluation | Actual progress | Contact result | Stop result |
| --- | --- | --- | --- |
| 50, nominal seed 4401, 20 s continuous at 0.04 m/s | 0.352 m; 0.0176 m/s | Rear feet contact 100%; zero qualified rear releases | Not part of this continuous trial |
| 100, nominal seed 3101, 15 s + 0.9 s ramp + 3 s hold | 0.01785 m/s after 1.5 s startup exclusion | Rear core-swing corners approximately -0.34 mm; rear slip RMS 0.050–0.052 m/s | No fall; settles; four-foot hold; maximum hold drift 2.75 mm |

Checkpoint 100 **fails qualification**: insufficient speed, rear-foot release,
whole-foot clearance and slip, plus the inherited conservative effort gate.
Support sole p95 tilt is under 4.6 degrees, but a level foot that drags is not an
acceptable gait. No joint reached the corrected target-rate/packet ceiling in
this nominal evaluation. Joint tracking p95 is 9.03 degrees pooled; the front
sagittal target/position correlation lag is approximately 0.20–0.25 seconds.
That lag is a cyclic correlation estimate, not identified transport latency.

- [Checkpoint 50 sustained metrics](../reviews/v44-training-20260905/model50-nominal-sustained.json)
- [Checkpoint 100 run/stop metrics](../reviews/v44-training-20260905/model100-nominal-runstop.json)
- [Checkpoint 50 diagnostic video](../reviews/v44-training-20260905/model50-video/videos/play/rl-video-step-0.mp4)
- [Reproducible headless video script](../reviews/v44-training-20260905/record_candidate.ps1)

The video is 1,200 frames, 1280x720 at 60 fps, exactly 20 seconds. It uses nominal
physics, seed 1701, 15-second motion and a ramp/hold. It is **not the same seed or
checkpoint as the checkpoint-100 numeric evaluation**. Playback's automatic
exports are isolated under the review directory and are not deployment packages.

### Training-log interpretation

The active process's legacy `commanded_distance_m`/`mean_commanded_speed_m_s`
can become zero at the stop phase because their reward-side accumulator is
gated off at zero command; that is not proof of zero movement. Its early
`sustained_stall_rate` can exceed one because the age-derived denominator is
one sample short. A logging-only patch is prepared for the next source revision,
and was applied after stopping the first process (the running process held the
source file open). It changes diagnostic counters/labels only. Selection uses the
separate evaluator's run/contact/stop metrics. `distance_success_rate` also
excludes stopped episodes and is not an acceptance measure for this task.

## First run stopped; next controlled experiment

The 1,500-iteration initial budget was **not completed**. After the printed
iteration 335, the learner was interrupted through its own private console.
The latest saved checkpoint is `model_325.pt`; all earlier checkpoints and
evaluations are preserved. The idle batch prompt and its own PowerShell host
were then terminated. No robot/server process was involved. The production
provenance hook did not complete, so these baseline weights have no
`.training.json` and are not an export/deployment candidate.

Checkpoint 200 repeated the poor rear release in two nominal episodes. The
problem is not remedied by selecting a later early checkpoint. Analytical review
of the original reference found up to approximately 32 mm of diagonal support
depth mismatch at the middle command. Body roll/pitch can absorb some mismatch,
so this is not proof that every 3D solution is impossible; it is a conflict
between a near-level body, flat world-frame soles and the nominal trajectory.
The maximum residual authority cannot be assumed to repair the whole-cycle
geometry plus loaded tracking error.

A **separate unqualified common-floor ablation** will preserve the old reference,
plant, packet limits and original task defaults. The candidate solves for a
shared lowest-foot-edge support depth, permits modest stance sole angle instead
of demanding incompatible perfectly flat support, and includes the lower tread
when calculating clearance. Its training/evaluation overrides and experiment
name will be isolated and hash-recorded. This is an experimental correction, not
a relabeled or qualified running model.

### Common-floor training pilots

Both are fresh, isolated, **unqualified simulation experiments**, not production
V44 packages. Original task defaults, references, physical plant, packet cap,
per-joint rates and torque limits are unchanged. Common-floor geometry uses the
actual lower tread, 55 mm stroke, 0.30–0.40 Hz cadence and 0.03–0.04 m/s commands.
Analytical maximum stance sole angle is 11.05 degrees, maximum 0.40 Hz joint rate
75.80 degrees/s, support depth mismatch under 0.20 mm. Dynamic tests remain
necessary; this intentionally trades exact body-relative flatness for coherent
support height.

| Experiment suffix | Started PDT / launcher PID | Environment changes | PPO pilot |
| --- | --- | --- | --- |
| `common-floor-nominal-01` | 13:46:40 / 43088 | Common-floor prior, fully nominal physical domain | 200 iterations, 256 envs, 192-step rollouts, seed 1701, fixed LR 2e-4, entropy .002 |
| `common-floor-authority-02` | 13:51:00 / 32480 | Same as 01, sagittal residual scale .20 → .50 | Same fresh PPO settings |

The 192-step (3.2 s) rollout is longer than the earlier 64-step (1.07 s) rollout.
The higher learning rate and modest entropy bonus are explicit fresh-learning
settings, not an unrecorded continuation. Pilot 02 changes only correction
authority relative to 01: hip/knee maximum corrections become 8.59/11.46 degrees;
abduction authority and all delivered-target safety constraints remain unchanged.

- [Nominal experiment overrides](../reviews/v44-training-20260905/common-floor-nominal.json)
- [Authority comparison overrides](../reviews/v44-training-20260905/common-floor-authority.json)
- [Training command script](../reviews/v44-training-20260905/train_common_floor.ps1)
- [Evaluation command script](../reviews/v44-training-20260905/evaluate_candidate.ps1)

The initial separate ablation launcher attempted to capture checkpoint/config/
reference/override hashes in `ablation-manifest.json` on return and refused
production qualification. Installed Kit fast shutdown bypassed this return
hook even on successful completion; see the completion-evidence section below.
Evaluator/playback use the same strict override format and record its hashes.
For randomized evaluation, use a JSON with the desired broad physical ranges:
`--domain-mode randomized` activates per-environment randomization but does not
magically unpin explicit common strength/payload ranges from the nominal JSON.

Reference-only common-floor evaluation at 0.04 m/s (seed 3101) yielded only
0.00183 m/s, zero falls, 3.93 mm maximum hold drift and poor foot release/clearance.
The geometric correction alone is therefore **not a working walking policy**.
All actor outputs were forced to zero; the checkpoint loaded for the evaluator
did not control the robot. [Reference-only metrics](../reviews/v44-training-20260905/common-floor-prior-only.json).

Pilot 01 was deliberately interrupted after printed iteration 104, preserving
the last scheduled checkpoint `model_100.pt`. Its planned 200 iterations were
**not completed**. The private Isaac console was interrupted, then its own idle
batch/PowerShell tree was terminated. This frees GPU time for pilot 02; no
hardware/service process was touched. The interrupted pilot has no completed
ablation manifest and must not be mistaken for a production export.

No change to the Pi or deployment selection. Further results will be appended.

### Matched lift-only diagnostic

Generated a separate 20 mm-lift reference with unchanged 55 mm stroke and shared
floor geometry. Cadence is limited to 0.25–0.30 Hz (0.025–0.030 m/s commands):
20 mm at 0.40 Hz would exceed the unchanged 90 degrees/s target envelope, while
0.30 Hz has an analytical peak of 86.23 degrees/s. The generator now accepts
explicit lift/cadence arguments and never overwrites previous reference files.

Matched nominal, seed-3101, **zero-actor-output** evaluations at 0.03 m/s and
0.30 Hz:

| Prior | Actual speed | Front core-corner p05 FL/FR | Rear result |
| --- | --- | --- | --- |
| 10 mm lift | 0.00130 m/s | -0.14 / -0.18 mm | Poor release, corners remain on floor |
| 20 mm lift | 0.00604 m/s | 3.96 / 2.82 mm | Still poor release; corner p05 near -0.2 mm |

Neither fell; neither passed locomotion/contact gates. More lift helps the front
but is not sufficient for a working gait or clean rear-foot motion. This is not
a learned-policy comparison and does not establish a unique hardware cause.

- [10 mm metrics](../reviews/v44-training-20260905/common-floor-lift10-prior-0p03.json)
- [20 mm metrics](../reviews/v44-training-20260905/common-floor-lift20-prior-0p03.json)
- [20 mm overrides](../reviews/v44-training-20260905/common-floor-lift20.json)

### Actuator and optimizer caveats

The nominal implicit drive uses stiffness 30 and damping approximately 4.58366,
an old simulator setting retained from the July/August setup, not a new V44
typo. No hardware system identification was found to establish this damping.
The simple ratio `damping/stiffness ≈ 0.153 s` shows that the existing plant
already adds substantial dynamic lag before the new 30 ms response hypothesis.
Do not blindly remove damping or stack the full measured cyclic lag on top.
The current effective nominal peak torque is 2.6478 Nm, not the initial import
table's older 0.8826 Nm limit; runtime randomization/setup changes it.

Source inspection ruled out shared actor/critic gradient clipping: installed
RSL-RL clips each network separately. Its fixed +/-0.2 **value** clipping could
slow fitting of fresh critics with value MSE around 9,000–10,000, but clipping
fractions are not logged, so this is not an established training failure cause.
Neither global reward scaling nor value clipping was changed in pilots 01/02.

Across the first five matched/reference diagnostic evaluations, delivered-target
limiter occupancy and held-packet fraction are zero; pooled joint-time velocity-
ceiling occupancy is at most approximately 0.32% (worst per-joint approximately
0.95%). These results do **not** support
relaxing target-rate/step guards as the current remedy. Rated/soft-torque
threshold exceedance is not a measured stall or hard-cap saturation fraction.
Simulator actuator telemetry is not hardware torque measurement. Similarly,
these simulated evaluations cannot validate Pi IMU acquisition or USB feedback
timing; that remains a separate later hardware check.

### Common-floor checkpoint 50 comparison

These learned-policy evaluations use the same nominal physics, seed 3101,
phase-zero start, 0.04 m/s command and 15 s run / 0.9 s ramp / 3 s hold protocol.
Steady-run metrics exclude the first 1.5 seconds. Both pilots start fresh with
the same seed and PPO configuration; the intended change is sagittal residual
authority, not actuator strength or a larger target-rate limit.

| Candidate | Measured forward speed | Result |
| --- | --- | --- |
| Narrow authority, checkpoint 50 | 0.00186 m/s | Rear feet still drag; no fall; 4.16 mm maximum hold drift |
| Wider authority, checkpoint 50 | 0.00332 m/s | Rear scheduled release improves to 23.5% / 14.0%; core-swing corner p05 remains near floor; no fall; 3.71 mm maximum hold drift |

The wider candidate's support-sole p95 angles are approximately 3.5–5.3 degrees,
contact-slip RMS 0.012–0.021 m/s and pooled joint-tracking p95 9.38 degrees.
This is a small measurable improvement, **not an acceptable walking/running
policy**. Better release fractions without adequate clearance/travel do not
satisfy the acceptance criteria. Training reward alone is not used to select
the candidate.

- [Narrow checkpoint 50 metrics](../reviews/v44-training-20260905/common-floor-model50-nominal.json)
- [Wider checkpoint 50 metrics](../reviews/v44-training-20260905/authority-model50-nominal.json)
- [Wider checkpoint 50 diagnostic video](../reviews/v44-training-20260905/authority-model50-video/videos/play/rl-video-step-0.mp4)

The second video is a separate seed-1701 nominal playback of the wider candidate,
not a recording of the seed-3101 evaluation episode. It is 20.000 seconds,
1,200 frames at 60 fps, 1280 x 720; a few extracted frames were inspected. Its
playback exports remain isolated in the review directory, not onboard models.

Checkpoint 100 was then evaluated for **two** nominal episodes under the same
0.04 m/s run/stop protocol. Actual speed improved to 0.00635 m/s (15.9% of command),
rear release to 53.5% / 52.0%, and rear core-corner p05 to 3.02 / 2.18 mm. Front
core-corner p05 remained approximately 0.09 / 0.06 mm: the front feet still skim
the floor. Pooled tracking p95 increased to 11.57 degrees; per-front-knee p95
reached approximately 12.65 / 14.98 degrees. No falls; all four feet contact
during hold; worst hold drift 1.16 mm.

There is now learning progress, but the candidate still fails speed, per-leg
release, pair quality, core clearance, slip and effort filters. Target limiting
at checkpoint 100 is no longer exactly zero, but remains small (approximately
0.051% of joint-time; corrected front-knee velocity-ceiling occupancy up to
2.57%). This still does not establish the packet limits as the primary cause.

[Wider checkpoint 100, two-episode metrics](../reviews/v44-training-20260905/authority-model100-nominal.json).

### Cadence diagnostic

A separate zero-actor experiment time-scales the exact same 20 mm reference
tables from 0.30 Hz / 0.030 m/s to 0.15 Hz / 0.015 m/s. The command and blending
bands are halved together, preserving reference geometry, stroke and nominal
no-slip speed/cadence relationship. Physical masses, COM, friction, drive gains,
response hypothesis and target/torque limits are unchanged. The original table
metadata still describes its generation cadence; the evaluation override is the
explicit time-scaling record, not a regenerated or relabeled table.

The 25-second run provides more than three complete cycles after the
1.5-second startup exclusion. A detailed diagnostic trace distinguishes
delivered packet targets, internal response-filtered drive targets, actual joint
state, torque estimates, root motion, foot corners and contacts. Slower motion
restoring rear clearance would support a response-bandwidth explanation; rear
feet remaining grounded would direct investigation toward loaded equilibrium,
body sag and load transfer. Neither result alone identifies the physical servo
model or justifies relaxing its limits. Results will be recorded before making
that inference.

[Half-cadence overrides](../reviews/v44-training-20260905/common-floor-lift20-half-cadence.json).

The half-cadence run completed: speed 0.00348 m/s at a 0.015 m/s command, front
core-corner p05 7.56 / 6.91 mm, rear p05 approximately -0.18 / -0.18 mm, rear
release 29.2% / 27.9%, pooled tracking p95 7.21 degrees, no fall and stable hold.
Slowing the reference improves front clearance but **does not restore useful
rear clearance**. A same-duration full-cadence trace was also collected to
compare loaded tracking and hard-cap occupancy with exact phase labels.

- [Half-cadence report](../reviews/v44-training-20260905/lift20-half-cadence-prior.json)
- [Half-cadence step trace](../reviews/v44-training-20260905/lift20-half-cadence-trace.jsonl)

### Retrospective reproducibility snapshot

[Snapshot manifest](../reviews/v44-training-20260905/retrospective-snapshot/manifest.json)
records 72 files / 9,325,894 bytes captured at **14:23:11 PDT**, during pilot 02.
It contains Python/PowerShell source, configurations, reference tables, then
available reports, and the simulator robot asset. This is explicitly a
**retrospective** snapshot, not evidence that source hashes were recorded at
training launch. Launch-time effective YAML, evaluation-time source hashes and
these captured bytes are distinct evidence.

The USD asset `quadruped_robot_floating.usdc` is self-contained: standalone USD
dependency inspection found only its root layer, no external assets and no
unresolved dependencies. Its 13 material references are internal. Captured
SHA-256: `71b639bd877913bffeac47a1cfcb6f3dcabbbd1e25c6fba90b8b87e7ea96c6b8`.
No simulator was launched for that metadata inspection.

Current environment source after the logging-only correction hashes to
`8a0ab4638ddbb75a907f8ee15368463280e6e7b8bd52c5ffe48a14ad44c21c4f`;
environment configuration and PPO source retain their launch hashes above.
Later trace metadata additions and analyses are not retroactively included in
this frozen snapshot. New evaluation reports retain their own source hashes.

Future training should persist launch evidence before entering the simulator;
the launcher used for pilots 01/02 only flushed its in-memory manifest on normal
return. The subsequent pre-import preparation fix is described below. A resumed
job also restarts physics/RNG/environment counters: a plain
resume would restart the 19,200-step smoothness curriculum. Any continuation
must explicitly preserve/document curriculum position, not silently weaken
smoothness penalties. No continuation has been launched at this point.

## Final completed pilot and disposition

Pilot `common-floor-authority-02` completed **200 PPO updates (labels 0–199)**
at approximately 14:32 PDT. The log reports `Training time: 2440.47 seconds`;
the calculated rollout count is 200 x 192 x 256 = **9,830,400 transitions**.
This is the completed pilot budget, not completion of the earlier abandoned
1,500-update original-reference run.

Final checkpoint:
`logs/rsl_rl/drobot_commanded_run_v44_flat_contact_hardware_matched_stop_external_rear_payload__ablation_common-floor-authority-02/2026-09-05_13-51-06_common-floor-fresh/model_199.pt`.

SHA-256: `7afb77e9a440cf139a8862a37956bb9172dc36e163c2ad7a62b39b72f8aaa769`.

Two nominal episodes were evaluated at each trained command, seed 3101,
phase-zero start, the same run/ramp/hold protocol and startup exclusion:

| Requested speed | Actual speed | Rear release RL / RR | Worst core-corner p05 FL / FR / RL / RR | Worst hold drift |
| --- | --- | --- | --- | --- |
| 0.030 m/s | 0.00967 m/s (32.2%) | 61.3% / 65.2% | 2.14 / 3.29 / 1.35 / 1.96 mm | 0.99 mm |
| 0.040 m/s | 0.01077 m/s (26.9%) | 67.1% / 67.0% | 0.74 / 2.18 / 3.01 / 4.15 mm | 2.60 mm |

All four episodes stopped and held stably without falling. Mean final heading
error was approximately 0.90 / 1.35 degrees at the two commands; lateral/forward
ratios were 3.85% / 2.31%. At 0.04 m/s, body linear/angular acceleration RMS was
0.257 m/s² / 1.251 rad/s², satisfying the existing smoothness filters. These are
positive results, but do not compensate for deficient locomotion.

Both command evaluations still fail the same six filters: actual speed,
every-leg release, paired-release quality, 5 mm core-corner clearance, contact
slip and soft/rated effort usage. The target limiter is not absent in this later
policy: approximately 1.48% of joint-time at 0.04 m/s, with front-knee corrected
velocity-ceiling occupancy up to 12.36%. That is an emerging constraint to model
faithfully, not permission to remove the hardware envelope. Pooled tracking p95
remains 10.80 degrees at 0.04 m/s.

**Decision: retain this checkpoint as an unqualified research result; do not
deploy or promote it.** It is genuinely trained and measurably improved over
its own checkpoint 50, but is not a successful running model. More updates
could still improve it; 200 updates do not prove convergence. Nevertheless,
continuing blindly before resolving the contact/actuator uncertainty is not
justified by reward improvement alone. No extended continuation was started.
Randomized robustness qualification was not run because nominal acceptance
already fails. The original baseline's randomized training is not equivalent
to a robustness evaluation of this final nominal pilot.

- [0.03 m/s, two-episode report](../reviews/v44-training-20260905/authority-model199-0p03.json)
- [0.04 m/s, two-episode report](../reviews/v44-training-20260905/authority-model199-0p04.json)
- [Final 20-second simulation video](../reviews/v44-training-20260905/authority-model199-video/videos/play/rl-video-step-0.mp4)
- [Final learned-policy step trace](../reviews/v44-training-20260905/authority-model199-0p04-trace.jsonl)
- [Offline trace analysis](../reviews/v44-training-20260905/authority-model199-0p04-trace-analysis.json)

The final video is separate nominal seed-1701 playback: 20.000 seconds, 1,200
frames at 60 fps, 1280 x 720. Frames were extracted and inspected. It is not a
time-synchronized recording of either seed-3101 evaluation episode. Automatic
playback exports remain in the review directory, not a deployment package.

### What the detailed traces establish

The matched 25-second zero-actor 20 mm-lift trials produced 1,764 recorded steps
each. Halving cadence from 0.30 to 0.15 Hz reduced rear-knee filtered-target to
joint RMS error from approximately 7.2 to 4.4 degrees. The additional delivered
target to response-filter gap was only approximately 1.48 to 1.05 degrees.
Rear core-swing mean corner heights nevertheless stayed at the floor (roughly
-0.07 to -0.15 mm). There were no rear **core-swing** snapshots above 98% of
estimated hard torque capacity, although hard-cap occupancy was substantial in
other support/swing phases. Lower force contact alone is not proof that a foot
is airborne: a nearly unloaded foot can still graze the floor.

The final 0.5-second hold windows had constant targets and negligible response-
filter error. Static simulated sagittal deflections remained 1.10–3.57 degrees
after the full-cadence prior, and 1.67–5.46 degrees after the half-cadence prior.
Thus the simulated error is not exclusively the extra first-order filter or
dynamic timing. These observations do not uniquely identify stiffness, damping,
torque capacity, geometric error or contact/load distribution.

The final learned checkpoint also has larger drive-to-joint errors than its
additional filter gap: front-knee RMS 7.69 / 9.33 degrees versus filter gaps
1.07 / 1.06 degrees. Estimated front-knee hard-cap occupancy reaches 37.7% /
47.6% of steady-run snapshots. Its final hold is different from the prior's
hold: some joints track within a degree while others retain several degrees
of deflection. Contact history and load distribution matter; do not pool these
different equilibria into a single identified servo parameter.

- [Full-cadence report](../reviews/v44-training-20260905/lift20-full-cadence-prior25.json)
- [Full-cadence trace analysis](../reviews/v44-training-20260905/lift20-full-cadence-trace-analysis.json)
- [Half-cadence trace analysis](../reviews/v44-training-20260905/lift20-half-cadence-trace-analysis.json)

The existing real trial provides a short near-neutral tail with maximum error
1.005 degrees / pooled p95 0.851 degrees over 26 distinct feedback timestamps
in the last 0.5 seconds. The neutral target coordinates match the simulator.
However, support/loading is unknown, and the IMU still shows some motion. This
is suggestive evidence to investigate simulated loaded deflection, **not** a
matched static calibration or justification to pick a new drive stiffness.
See the [real-trial follow-up](../../debug/walk-trials/20260905T193441588801Z-2a2a9832/ANALYSIS.md).

### Completion evidence and shutdown limitation

The installed Isaac Sim uses `fast_shutdown=True`: `SimulationApp.close()` can
terminate the interpreter directly. Consequently `run_train_cli()` need not
return to the Python wrapper, even after successful training. This explains
why the completed pilot has no `ablation-manifest.json`; it is not evidence
that its PPO updates failed. The production wrapper's analogous post-return
provenance hook has the same lifecycle risk and must not be assumed reliable.

[Retrospective completion evidence](../reviews/v44-training-20260905/authority-02-retrospective-completion.json)
binds the final checkpoint, saved configurations, logs, reports and video by
hash, with the observed final iteration and training-time marker. It is clearly
retrospective and is not a fabricated launch manifest or production certificate.

For future ablation invocations, `train_v44_ablation.py` now snapshots source,
inputs, references and the root USD **before** importing the simulator. It
publishes an exclusive atomic `PREPARED_LAUNCH_EVIDENCE_ONLY` record outside
checkpoint-discovery directories. It explicitly does not claim training has
started, completed or qualified. This added preparation path received source
review only; **it was not exercised by another training invocation or a test**.
Reliable automatic completion recording still requires a non-Kit supervisor
or an equivalent pre-shutdown completion hook. Default fast shutdown was not
disabled, and no production provenance was manufactured.

### Next decision and boundaries

Before another long continuation, constrain loaded actuator/contact response
with a controlled, separately authorized measurement or suitable additional
recorded segments. Compare a genuinely held/loaded pose and a slow joint sweep
using completed targets, distinct encoder timestamps and known support; do not
fit the 30 ms filter to total cyclic lag or infer physical torque from simulator
estimates. In parallel, improve unloaded-foot clearance and body load transfer
with the existing torque/rate limits. A slower three-support-foot contact
curriculum is a possible separate experiment before returning to diagonal-pair
running, not an already tested solution.

Only after nominal contact/tracking improves should training expand to broader
physical randomization, multiple seeds, speed tracking and graceful-stop
qualification. Keep the previous robot selection until a candidate passes
those checks and deployment is explicitly approved.

All jobs launched for this session have finished or were deliberately stopped.
Performed: authorized Isaac Lab PPO training, checkpoint/reference simulation
evaluation, diagnostic recording, local video playback/frame review, and
offline numerical analysis. No separate test suite, linter, type checker,
compiler/build check, browser validation, Pi connection, motor check, hardware
motion, deployment, Git commit/push or hosted review was performed. Unrelated
worktree changes remain preserved.

The [final source supplement](../reviews/v44-training-20260905/final-source-supplement/manifest.json)
preserves later evaluator/analyzer/launcher source, shared runtime, completion
evidence and report snapshots separately from the earlier 14:23 capture. Both
are retrospective; neither is mislabeled as training-launch attestation.
