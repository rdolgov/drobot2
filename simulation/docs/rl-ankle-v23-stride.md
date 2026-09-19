# V2.3 R3 solid-sole ankle: longer-stride PPO

September 12 user request: train a policy with longer strides while using the
ankle to keep good sole contact. This run uses the latest V2.3 R3 printed foot
and the retained original lower leg. Training, asset import, policy selection
and example recording continue the user's authorized RL task. No standalone
test suite, mesh check, slicer, or hardware operation is requested or run.

Status: **completed September 12, 2026 at approximately 19:04 PDT. No candidate
passed all selection criteria.** Both 3000-iteration, 256-environment PPO runs
finished successfully, totaling 98,304,000 transitions. Learning took about
2 hours 58 minutes, followed by selection and two 48-second recordings.
V2.1 checkpoints/results remain unchanged and are not results for this model.

Completion state: [`workflow.json`](../reviews/ankle-v23-stride/workflow.json).
Seed 2301 initialized only its actor from the best V2.1 policy; seed 2302
started fresh. The best available attempts were archived and recorded despite
failing the gate. The completed run does not establish a successful four-leg
longer-stride gait or show that longer strides are safer.

## Completed recordings and measured results

Both nominal clips command 0.03 m/s, stop the command after 40 seconds, and
finish at 48 seconds (1440 frames, 30 fps, 1280 x 720). Speed and contact
statistics below describe the moving-command portion; displacement and drift
describe the complete recording. A qualified touchdown requires more than
0.10 seconds in the air and more than 8 mm clearance before touchdown, with
the spacing and lateral constraints in the implementation.

| Nominal recording metric | Best available: seed 2301, iteration 2250 | Fresh initialization: seed 2302, iteration 1500 |
| --- | ---: | ---: |
| Forward displacement | 78.8 cm | 6.1 cm |
| Mean forward speed while commanded to move | 1.99 cm/s | 0.20 cm/s |
| Qualified touchdown stride | 8.87 cm, front feet only | No qualifying strides |
| Qualified steps: front-left, rear-left, front-right, rear-right | 8, 0, 8, 0 | 0, 0, 0, 0 |
| Mid-swing clearance fraction | 40.6% | 0.0% |
| Expected stance contact fraction | 93.8% | 100.0% |
| Flat/planted fraction of contacting stance feet | 85.6% | 84.6% |
| Mean stance sole tilt | 1.74 degrees | 1.62 degrees |
| Mean stance slip speed | 2.11 mm/s | 1.74 mm/s |
| Signed lateral drift | +2.02 cm | -0.38 cm |
| Mean speed after stopping | 4.76 mm/s | 3.27 mm/s |
| Falls | 0 | 0 |
| Ankle excursions across the four legs | 18.9-29.2 degrees | 21.0-40.0 degrees |

The 8.87 cm result measures qualifying front-foot steps, not a four-leg average
with successful rear-foot stepping. Zero qualified rear steps does not mean
the rear feet never moved: they did not complete steps meeting the measurement
criteria. Flat/planted percentage is conditional on contacting stance feet;
it does not mean every sole was flat throughout the motion. The fresh policy
moved and rocked but made little net forward progress and had no qualifying
steps. No falls in these clips establishes neither physical safety nor reliable
walking. Saved frames at 4 and 20 seconds from each clip were visually reviewed.

- [Best available recording](../reviews/ankle-v23-stride/walk-seed2301/model_2250-0.03mps-learned.mp4), [measurements](../reviews/ankle-v23-stride/walk-seed2301/report.json), [trace](../reviews/ankle-v23-stride/walk-seed2301/trace.json), [archived weights](../isaac/models/ankle-v23-stride/seed2301-model_2250.pt).
- [Fresh-initialization recording](../reviews/ankle-v23-stride/walk-seed2302/model_1500-0.03mps-learned.mp4), [measurements](../reviews/ankle-v23-stride/walk-seed2302/report.json), [trace](../reviews/ankle-v23-stride/walk-seed2302/trace.json), [archived weights](../isaac/models/ankle-v23-stride/seed2302-model_1500.pt).

## Selection outcome

All six candidate checkpoints failed the unchanged gate. Selection covered
12 command/checkpoint trials with 16 environments each, or 192 episodes.
These are model-selection cases, not a held-out generalization study. The
selected seed 2301 checkpoint scored highest, but its mean minimum per-leg
qualified step count was zero at both commands.

| Selected seed 2301 checkpoint | Command 2.4 cm/s | Command 3.4 cm/s |
| --- | ---: | ---: |
| Achieved speed | 1.31 cm/s; fails 70% target | 1.82 cm/s; fails 70% target |
| Qualified stride | 6.91 cm; fails 8 cm target | 8.59 cm |
| Mid-swing clearance | 23.4%; fails 50% target | 29.5%; fails 50% target |
| Mean minimum qualified steps per leg | 0; fails 3-step target | 0; fails 3-step target |
| Flat/planted fraction of contacting stance feet | 90.6% | 91.0% |
| Expected stance contact | 96.2% | 95.6% |
| Mean stance slip | 2.60 mm/s | 3.85 mm/s |
| Mean absolute lateral drift | 0.74 cm | 1.43 cm |
| Mean stopped speed | 3.62 mm/s | 3.83 mm/s |
| Fall fraction | 0% | 0% |

The selected fresh policy had zero qualifying strides and zero mid-swing
clearance at both commands, with mean speed only 0.12 and 0.23 cm/s. Detailed
selection evidence is in [the selection report](../reviews/ankle-v23-stride/selection/report.json).
Completion records for [seed 2301](../reviews/ankle-v23-stride/20260912-seed2301/train.metrics.completion.json)
and [seed 2302](../reviews/ankle-v23-stride/20260912-seed2302/train.metrics.completion.json)
confirm iteration 2999 and successful training return.

The next training change should address rear-foot unloading and swing/weight
transfer before increasing the stride target further. That is a proposed next
step, not an additional run performed here. The prior V2.1 model used different
mechanics, cadence and objectives, so these runs do not isolate an ankle benefit
or prove improved speed or safety. No policy was deployed to hardware.

## Mechanical source and assumptions

Canonical simulation URDF: `cad/exports/urdf/quadruped_robot_v23_solid_sole.urdf`.
Retained shank source: `cad/drobot_cad/parts/upper_arm.py`; new foot source:
`cad/drobot_cad/parts/ankle_v23_existing_leg.py`, R3 solid sole without groove.
Link-frame STEP/STL exports are generated from those existing sources.

The added motor case and its mass now belong to the moving foot. Its output
wheels attach to the retained fork. Geometric ankle joint coordinates retain
the existing URDF axes; a hardware motor command sign/zero needs new calibration
because the case moves relative to the output. No hardware sign is inferred.
The bare sole is 42.2117 mm from the ankle axis, 43.2117 mm with a 1 mm tread.
The single foot print remains 100 x 60 x 6 mm at its sole. Knee-to-ankle spacing
remains 159.896689 mm. CAD-derived mass uses the prior effective printed density
806 kg/m3, 55 g per motor, and explicit hardware/tread allowances; it is unmeasured.
External collision boxes approximate the parts; detailed cavity/screw contact
and cable routing are omitted. Existing mesh triangulation warnings do not
become collision geometry. Physical range, printability and strength are unverified.

## Longer strides and contact objective

Final requested stance sweep is 80-110 mm, with same-foot touchdown spacing
approximately 107-147 mm when tracking without slip. The V2.1 recording's
reference stance sweep was 39 mm. A 1500-iteration curriculum grows sweep and
command together from 60% to 100%, with swing lift increasing from 12 to 18 mm.
Duty factor is 0.75 with quarter-cycle staggered crawl phases.

Longer steps require a slower cadence with the existing 100 deg/s target-rate
limit. Commands are 0.024-0.034 m/s, with frequency about 0.225-0.232 Hz, chosen
from the analytic IK trajectory. The final reference nominal maximum knee
target speed is about 89 deg/s. Actual PD tracking, random weaker motors and
body motion can still produce rate limiting and loss of contact.

Reward terms explicitly value scheduled foot lift, clearance and foot-path
tracking, and normalize contact slip by a 20 mm/s scale so a few cm/s of
sliding is materially penalized. Stance flatness is conditional on stance,
and its denominator includes missing stance contacts. Qualified touchdown
stride needs preceding air time and clearance; failed episodes remain failures.
The controller uses actual hip/knee angles for ankle leveling. Its motor
torque, packet-rate and travel limits stay at the prior conservative settings.

Training used two seeds with 256 environments and 3000 iterations of 64
steps (98,304,000 transitions combined). One starts from the previous best
V2.1 actor only; one starts fresh. New critic/optimizer/iteration state is used
for the changed model and reward. Neither initialization is a matched physical
ankle/no-ankle control or proof that longer steps are safer.

Completion selection considered measured stride, foot lift, contact slip,
falls, lateral drift, speed tracking and stopping. The best available attempt
from each seed was recorded, with the failed criteria reported above.

## Selection criteria and artifacts

Candidates are iterations 1500, 2250 and 2999 from each seed, each evaluated
at commands 0.024 and 0.034 m/s for 48 seconds, stopping after 40 seconds.
Each trial has 16 environments: one nominal plus 15 with independent motor
strength/rate variation from 85% to 100%. Gate requirements at both commands:
fall fraction <=10%, speed >=70% of command, planted contacting-stance fraction
>=90%, stance-contact fraction >=85%, mid-swing clearance fraction >=50%,
mean slip <=0.012 m/s, absolute lateral drift <=0.08 m, stop speed <=0.012 m/s,
qualified touchdown stride >=0.08 m, and mean minimum per-leg qualified step
count >=3. A qualified step follows >0.10 s air time and >8 mm clearance.

This is intentionally stricter than the earlier selection on flat contacts.
Missing/zero qualified strides fail the gate. Failure masks remain after the
first fall; reset bodies do not contribute successful continued travel. Each
seed's selected nominal recording runs at 0.03 m/s with actual contact and
command telemetry, including failures. Selected weights are archived under
`simulation/isaac/models/ankle-v23-stride/`.

Inputs and inherited source modules were copied under
[`provenance/`](../reviews/ankle-v23-stride/provenance/) before launch, including
SHA-256 records and the actor initialization checkpoint hash. Estimated robot
mass is 3.444019909 kg. The import manifest records URDF/USD hashes and preserved
foot bodies. No unrelated automated tests, mesh checks, slicer checks or
hardware verification were run. Requested work completed: STEP/STL asset
production, mass calculations and URDF authoring, Isaac import, both full PPO
runs, checkpoint selection, two recordings, and static presentation images.

The direct URDF snapshot could not resolve parent-relative mesh paths. A local
presentation copy with sibling meshes resolved that renderer issue; the
canonical URDF and training assets were not changed. The saved illustrations
show the latest geometry; they do not verify fit or motion.

- [Current robot illustration](../../cad/reviews/ankle-v23-stride/robot_20260912T224822Z.png)
- [Retained shank link illustration](../../cad/reviews/ankle-v23-stride/shank-link_20260912T224545Z.png)
- [Solid-sole foot link illustration](../../cad/reviews/ankle-v23-stride/foot-link_20260912T224545Z.png)
- [Local-machine CAD Viewer](http://127.0.0.1:4178/?file=exports%2Furdf%2Fquadruped_robot_v23_solid_sole.urdf)

## Reproduction

From the repository root, first produce/import the model using the commands
in [the CAD document](../../cad/docs/ankle-v23-existing-leg-foot.md). Do not
duplicate active jobs. The exact two launches are:

```powershell
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v23_stride/train.py --rl_library rsl_rl --task Drobot-Ankle-V23-Stride --num_envs 256 --seed 2301 --max_iterations 3000 --run_name stride-2301 --warm-start C:/Users/roman/Documents/dev/drobot2/simulation/isaac/models/ankle-v21-planted/seed2101-model_1500.pt --visualizer none env.metrics_path=C:/Users/roman/Documents/dev/drobot2/simulation/reviews/ankle-v23-stride/20260912-seed2301/train.metrics.jsonl
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v23_stride/train.py --rl_library rsl_rl --task Drobot-Ankle-V23-Stride --num_envs 256 --seed 2302 --max_iterations 3000 --run_name stride-2302 --visualizer none env.metrics_path=C:/Users/roman/Documents/dev/drobot2/simulation/reviews/ankle-v23-stride/20260912-seed2302/train.metrics.jsonl
& ./cad/.venv/Scripts/python.exe simulation/isaac/rl/ankle_v23_stride/finish_training.py
```
