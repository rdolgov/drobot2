# V2.3 rear-foot unloading and weight-transfer PPO

September 13, 2026. The user authorized the next training change after the
previous longer-stride run produced qualifying front steps but no qualifying
rear steps. This task teaches weight transfer and supported rear-foot swing
before attempting greater speed or stride. The solid sole, retained lower leg,
USD model and hardware motor limits are unchanged.

Status: **training, selection and both recordings completed September 13,
2026 at approximately 19:12 PDT. No candidate passed all selection criteria.**
Both 3000-iteration jobs completed successfully, using 98,304,000 transitions
combined. Completion state is recorded in
[`workflow.json`](../reviews/ankle-v23-transfer/workflow.json).
Previous [V2.3 stride results](rl-ankle-v23-stride.md) remain separate.

## Completed recordings

The policy initialized from the previous actor learned supported rear-right
steps alongside its front steps, but still had no qualifying rear-left steps.
The fresh policy retained a front-only qualifying-step pattern. Neither is a
successful four-leg longer-stride gait. Zero qualified steps means the step
criteria were not met; it does not mean a foot never moved or briefly lifted.

Both clips last 48 seconds (1440 frames, 30 fps, 1280 x 720). Command is
**0.018 m/s**, ramped up initially and ramped down after 40 seconds. The filename
rounds that value to `0.02mps`; the overlays and reports retain the exact command.
Moving-command averages below exclude stopping, while displacement/drift cover
the complete clip. Stopped speed is measured from 42 seconds onward.

| Nominal recording metric | Previous-actor initialization, seed 2311 | Fresh start, seed 2312 |
| --- | ---: | ---: |
| Selected checkpoint | Iteration 2999 | Iteration 2999 |
| Forward displacement | 21.1 cm | 50.0 cm |
| Mean achieved forward speed while commanded to move | 0.57 cm/s | 1.24 cm/s |
| Signed lateral drift | -10.15 cm | -0.04 cm |
| Average qualifying stride across feet that qualified | 3.76 cm | 8.42 cm, front feet only |
| Overall swing-clear fraction | 64.1% | 48.1% |
| Rear supported swing-clear fraction | 47.5%, rear-right only | 0.0% |
| Rear supported unloading fraction | 8.0% | 0.0% |
| Expected stance contact fraction | 95.9% | 96.9% |
| Flat/planted fraction of contacting stance feet | 70.4% | 97.3% |
| Mean stance sole tilt | 3.64 degrees | 1.13 degrees |
| Mean stance slip speed | 1.31 mm/s | 1.39 mm/s |
| Mean speed after stopping | 4.26 mm/s | 4.44 mm/s |
| Falls | 0 | 0 |
| Target-rate-limited fraction | 0.84% | 0.50% |
| Ankle excursions across the four legs | 18.3-35.4 degrees | 24.7-42.1 degrees |

| Leg | Seed 2311 supported steps / qualifying stride | Seed 2312 supported steps / qualifying stride |
| --- | ---: | ---: |
| Front-left | 5 / 4.33 cm | 5 / 8.24 cm |
| Rear-left | 0 / no qualifying stride | 0 / no qualifying stride |
| Front-right | 3 / 3.29 cm | 5 / 8.60 cm |
| Rear-right | 3 / 3.27 cm | 0 / no qualifying stride |

Supported and ordinary qualifying step counts match in these two clips.
In seed 2311, supported mid-swing clearance by leg was 83.5/0/77.1/95.1%;
late-transfer supported unloading was 0/0/12.6/15.9%, ordered front-left,
rear-left, front-right, rear-right. In seed 2312, those values were
98.7/0/76.7/0% and 75.9/0/100/0%, respectively. The rear-left in seed 2311
still carried a mean 21.5% of body weight in its late transfer intervals,
against the unloading criterion of less than 6%.

Flat/planted percentages are conditional on contacting stance feet; neither
clip keeps every sole planted throughout. The more stable-looking fresh clip
still fails the required rear-foot stepping. These slower-command runs do not
demonstrate faster walking, a controlled benefit over the prior policy, or
hardware safety. The reference's 8-11 cm sweep is a target, not a measured
four-leg outcome. No policy was deployed to hardware.

- [Previous-actor initialization video](../reviews/ankle-v23-transfer/walk-seed2311/model_2999-0.02mps-learned.mp4), [recording report](../reviews/ankle-v23-transfer/walk-seed2311/report.json), [trace](../reviews/ankle-v23-transfer/walk-seed2311/trace.json), [policy weights](../isaac/models/ankle-v23-transfer/seed2311-model_2999.pt).
- [Fresh-start video](../reviews/ankle-v23-transfer/walk-seed2312/model_2999-0.02mps-learned.mp4), [recording report](../reviews/ankle-v23-transfer/walk-seed2312/report.json), [trace](../reviews/ankle-v23-transfer/walk-seed2312/trace.json), [policy weights](../isaac/models/ankle-v23-transfer/seed2312-model_2999.pt).

Both encoders finalized 1440 frames. Saved frames at 4 and 20 seconds from each
clip were visually inspected. The workflow records the archived policy hashes.
No unrelated automated tests, mesh/slicer checks, or hardware checks were run.

## Completed selection results

Six checkpoints were selected across 12 trials with 16 environments each,
192 episodes total. No episode achieved three supported steps on every leg.
Both seeds selected their final checkpoint, iteration 2999. Seed 2311, which
initialized its actor from the previous V2.3 policy, ranked highest.

| Selected seed 2311, mean across 16 environments | Command 1.5 cm/s | Command 2.1 cm/s |
| --- | ---: | ---: |
| Achieved forward speed | 0.63 cm/s | 1.11 cm/s |
| Qualified stride across legs that qualified | 4.44 cm | 6.47 cm |
| Supported steps: front-left, rear-left, front-right, rear-right | 4.56, 0, 4.31, 3.75 | 4.38, 0, 4.38, 3.94 |
| Rear supported swing-clear fraction | 39.0% | 41.8% |
| Rear supported unloading fraction | 14.8% | 20.5% |
| Overall swing-clear fraction | 61.0% | 64.9% |
| Flat/planted fraction of contacting stance feet | 76.3% | 75.5% |
| Expected stance contact fraction | 98.0% | 97.9% |
| Mean stance sole tilt | 3.12 degrees | 3.25 degrees |
| Mean stance slip | 0.83 mm/s | 0.98 mm/s |
| Mean absolute lateral drift | 10.0 cm | 12.0 cm |
| Mean speed after stopping | 4.65 mm/s | 3.79 mm/s |
| Falls | 0/16 | 1/16 |

The new result includes supported rear-right steps. The rear-left completed
zero qualifying steps in both selected cases, with zero supported clearance;
rear averages therefore hide a large difference between the two rear legs.
No qualifying rear-left step does not mean that foot never moved or lifted.
Speed tracking, stride, planted contact, drift, all-four stepping and the rear
unloading threshold failed. Low average slip and a positive mean support
margin do not compensate for those failures.

The selected fresh policy, seed 2312, again relied on front-foot stepping:
mean supported counts were 4.0/0/3.5/0 at 1.5 cm/s and 4.81/0/4.81/0 at
2.1 cm/s. Its corresponding forward speeds were 0.53 and 0.85 cm/s, planted
fractions 99.2% at both commands, and no falls. Rear supported clearance and
unloading were zero at both commands. Flat feet by themselves therefore did
not establish the desired four-leg gait.

These are checkpoint-selection cases, not held-out generalization results.
One earlier seed-2311 checkpoint fell in 4/16 higher-command cases; all failed
trials remain in the [selection report](../reviews/ankle-v23-transfer/selection/report.json).
Training completion records for [seed 2311](../reviews/ankle-v23-transfer/20260913-seed2311/train.metrics.completion.json)
and [seed 2312](../reviews/ankle-v23-transfer/20260913-seed2312/train.metrics.completion.json)
confirm successful return at iteration 2999. The reported learning wall times
were 9871 and 9651 seconds, respectively. No training was repeated.

## What changed

- Each quarter-cycle now includes a four-foot transfer interval: duty factor
  0.82, transfer phase 0.75-0.82, then swing 0.82-1.0. Phase advances by the
  clock, including when unloading fails; privileged simulator feedback cannot
  pause the clock to hide failed lift attempts.
- A smooth reference body shift precedes swing: 26 mm forward for a rear-foot
  swing, 12 mm backward for a front-foot swing, and 18 mm toward the opposite
  side. These are hip joint targets through IK, not forces or teleportation.
  The ankle continues leveling the sole using measured hip/knee angles.
- Dense reward favors transferring load onto three other feet with the
  whole-robot COM above their center triangle, reducing the selected foot's
  vertical force, and then clearing the ground. Rear lift receives 1.7 times
  the front lift weight. A leg that is behind on its step count receives a
  larger touchdown reward; completing another round on all four earns a bonus.
- Supported touchdown needs a real >0.10-second flight and >8 mm clearance,
  three other contacts during lift, an upright body, and forward touchdown
  displacement. Ground contact clears the flight evidence so scraping cannot
  retain a previous lift qualification.
- The actor keeps 82 proprioceptive/clock observations. The 101-input critic
  additionally sees vertical load fractions, relative COM, support margin,
  and selected leg. No force sensor or exact COM input is added to the actor.
- Lateral penalties allow the intended 18 mm body sway. Residual authority
  increases to 0.14/0.24/0.30/0.20 rad for abduction/hip/knee/ankle. Torque,
  joint travel, packet step and actuator rate limits remain unchanged.

The old neutral reference places the tread 373.91 mm below base_link, based
on read-only forward kinematics of the canonical URDF. The old reward target
was 342.21 mm, a 31.70 mm mismatch. This task corrects the height relationship
and bends the legs 10 mm more to leave reach for the body shift: stance down
275 mm, target base height 363.91 mm, starting height 377.91 mm. This correction
and the other changes are combined; this run cannot isolate their individual
effects.

The final reference stance sweep remains **80-110 mm**, reached through the
existing 60%-to-100% curriculum over 1500 iterations. Commands are reduced to
**0.015-0.021 m/s** to allow approximately 0.45 seconds for transfer and
1.15-1.17 seconds for swing. Reference lift is 18 mm. These are learning targets,
not achieved strides or proof of safer motion.

Analytic sampling of the reference equations at the two final commands found
maximum nominal joint target rates of about 21/38/95/76 deg/s for
abduction/hip/knee/ankle. This calculation does not exercise the simulator or
show actual tracking; weaker randomized motors may still be rate limited.
At the old 285 mm stance down, the added shift and longest sweep would exceed
two-link reach, which is why the 10 mm additional bend is included.

## Training and completion workflow

Two seeds, 256 environments each, 3000 iterations of 64 steps, totaling
98,304,000 planned transitions. Seed 2311 initializes only the prior V2.3
seed-2301 iteration-2250 actor. Its critic, optimizer and iteration counter
start fresh. Seed 2312 starts completely fresh. Both use the same new task.
Motor strength and target rate vary independently from 85% to 100%. There are
no changes to hardware or physical print files.

Candidate checkpoints are 1500, 2250 and 2999 from each seed. Each candidate
is selected using 48-second trials at 0.015 and 0.021 m/s, one nominal plus
15 environments with weaker motors per trial. Command stops after 40 seconds.
The best attempt per seed is recorded at 0.018 m/s, even if no candidate passes.
The recording overlays supported per-leg step counts and selected foot load.

At both commands, selection requires:

- Fall fraction <=10%, mean achieved speed >=70% of command, planted fraction
  of contacting stance feet >=90%, expected stance contact >=85%.
- Mean slip <=12 mm/s, absolute lateral drift <=8 cm, stopped speed <=12 mm/s.
- Mean qualifying stride >=8 cm, mean minimum per-leg qualifying stride >=8 cm,
  mean minimum per-leg qualifying step count >=3, and swing clearance >=50%.
- Mean minimum per-leg supported steps >=3; at least 80% of episodes must
  achieve three supported steps on every leg without falling.
- Mean minimum per-leg supported swing-clear fraction >=50%; rear supported
  unloading fraction >=30% during the latter half of the transfer intervals.

Supported unloading means selected foot load <6% of robot weight, three other
contacts carrying >65% of weight, upright posture, and COM projection inside
the other foot centers' triangle. The triangle is a conservative proxy; it
does not include the wider sole patches or prove dynamic stability. Missing
observations and zero steps produce zero fractions, not a passing result.
Falls remain failures after simulator resets. Aggregate and per-leg metrics
are retained in the selection and recording reports.

This is a slower command envelope than the previous run. A future pass would
demonstrate the new criteria at those slower commands, not faster walking or a
controlled ankle benefit. Hardware safety and printability remain unverified.
No unrelated tests, mesh checks, slicers or hardware operations are requested.

## Reproduction and provenance

Sources and the existing model are snapshotted before launch under
[`provenance/`](../reviews/ankle-v23-transfer/provenance/), with SHA-256 records
and the initialization checkpoint hash. Do not duplicate an active run.

```powershell
& ./cad/.venv/Scripts/python.exe simulation/isaac/rl/ankle_v23_transfer/save_provenance.py
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v23_transfer/train.py --task Drobot-Ankle-V23-Transfer --num_envs 256 --seed 2311 --max_iterations 3000 --run_name transfer-2311 --warm-start C:/Users/roman/Documents/dev/drobot2/simulation/isaac/models/ankle-v23-stride/seed2301-model_2250.pt env.metrics_path=C:/Users/roman/Documents/dev/drobot2/simulation/reviews/ankle-v23-transfer/20260913-seed2311/train.metrics.jsonl
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v23_transfer/train.py --task Drobot-Ankle-V23-Transfer --num_envs 256 --seed 2312 --max_iterations 3000 --run_name transfer-2312 env.metrics_path=C:/Users/roman/Documents/dev/drobot2/simulation/reviews/ankle-v23-transfer/20260913-seed2312/train.metrics.jsonl
& ./cad/.venv/Scripts/python.exe simulation/isaac/rl/ankle_v23_transfer/finish_training.py
```

The training supervisor records `.completion.json` beside each metrics file.
`finish_training.py` owns selection, recording, archiving and the workflow state.
The task heartbeat stays quiet during healthy progress and delivers completed
results and both MP4 recordings, then pauses. This run's completed results and
recordings are above.
