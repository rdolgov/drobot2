# Mechanical V2 ankle PPO training, September 12

User authorization: **"Run RL training with new designs."** The necessary
asset import and the PPO training jobs were run. No separate automated test
suite, URDF validator, rendered review, held-out evaluation or hardware trial
was requested or run.

Status: **all three first-seed, equal-budget PPO jobs completed** at 07:52 PDT.
Each finished 500 iterations and saved `model_499.pt`, totaling 12,288,000
environment transitions across the comparison. All launch processes returned
exit code 0. Saved checkpoint metadata records iteration 499 for all three.

The active ankle did **not demonstrate a speed advantage in this pilot**.
All three training speeds remained below the 0.06-0.14 m/s cruise commands.
This is an inconclusive morphology experiment: no usable running policy or
held-out maximum sustained speed has been established.

## Recorded results

The following are unweighted means of the **last 50 PPO logger summaries**,
including stand episodes. They describe stochastic training policies, not
independent evaluation episodes or performance of the final deterministic policy.

| Design | Forward speed | Logged moving command | Fall fraction | Targets rate limited |
| --- | ---: | ---: | ---: | ---: |
| Active V2 ankles | 0.0151 m/s | 0.0820 m/s | 2.66% | 84.00% |
| Rigid baseline | 0.0233 m/s | 0.0873 m/s | 1.34% | 86.31% |
| Locked V2 ankles | 0.0287 m/s | 0.0843 m/s | 21.90% | 88.89% |

The locked design logged more forward motion but substantially more falls.
The active policy lagged both controls in forward-speed telemetry. The logged
all-feet-airborne fraction was zero in the final 50 iterations for every design;
there is no evidence of aerial running here. Different stochastic episodes and
only one training seed prevent a causal or statistically reliable ranking.

![Training curves from the saved console logs](../reviews/ankle-v2-training/20260912-071916-seed1901/learning-curves.png)

Full numeric results and checkpoint hashes are in
[training_summary.json](../reviews/ankle-v2-training/20260912-071916-seed1901/training_summary.json).
The archived [policy weights and contracts](../isaac/models/ankle-v2-pilot/20260912-071916-seed1901/README.md)
include actor, critic and optimizer state, actual configuration, named action
mapping and source/asset provenance. These checkpoints have not been deployed.

## Jobs

All variants use seed 1901, 128 environments, 64 rollout steps and 500 PPO
iterations: 4,096,000 environment transitions per design. Training starts from
scratch with bounded Beta actions and separate 256 x 256 actor/critic networks.

| Task | Actions | Actor / critic observations | Nominal mass |
| --- | ---: | ---: | ---: |
| `Drobot-Ankle-V2-Active` | 16 | 62 / 70 | 3.50 kg |
| `Drobot-Ankle-V2-Baseline` | 12 | 50 / 58 | 3.18 kg |
| `Drobot-Ankle-V2-Locked` | 12 | 50 / 58 | 3.50 kg |

Batch directory:
[`20260912-071916-seed1901`](../reviews/ankle-v2-training/20260912-071916-seed1901/).
The active run started first. The controls were subsequently launched alongside
it on the available RTX 5090; the supervisor joins these same jobs when the
original queue reaches them, so it does not train duplicate controls.

The two initial control startups encountered a Windows stdout encoding error
before PPO iterations. Their logs are retained as `*.startup-encoding.log`.
The supervisor was fixed to use UTF-8 and restarted. Mathematical environment
and PPO settings were unchanged. The active job used the original thin
entrypoint; controls use the supervising entrypoint. They share the same
environment, configuration, action semantics and PPO source.

## Implemented pilot scope

`import_assets.py` converts the three source URDFs into isolated floating-base
USD assets under `simulation/exports/isaac/ankle_v2/`. Their physical foot
bodies are retained for contact sensing even in the fixed controls. A massless
optical frame is omitted from training inputs to avoid creating an artificial
dynamic body. Directly connected neighbors are collision filtered; general
self-collision remains enabled. Source/model hashes and imported body names
are in [assets.json](../exports/isaac/ankle_v2/assets.json).

The controller has a 60 Hz action step and 120 Hz implicit position servos.
Targets are limited to 60 deg/s for abduction, 100 deg/s for hip/knee/ankle and
at most 2 degrees per packet. All motors share the 270 deg/s no-load physical
velocity ceiling in the pilot; the ankle URDF's conservative 100 deg/s value
is enforced as a **target** rate by this task rather than as a backdrive cap.
At each physics step, effort is clipped by the conservative 0.8826 N m cap
and a linear torque-speed envelope. The implicit torque telemetry used for
mechanical-work logging is an estimate, not a calibrated battery-energy measure.

The pilot randomizes independent motor effort and target rate between 85% and
100%. It uses common fixed contact parameters, common geometric/mass assumptions
and the same reward terms. It does **not** yet implement backlash, identified
loaded servo response, communication delay, mass/COM/friction randomization,
thermal state or correlated 16-motor supply sag. It cannot qualify hardware
transfer or conclusively settle the morphology comparison.

Episodes last approximately 12 seconds. Ten percent are stand commands; other
episodes request 0.06-0.14 m/s with startup and final stop periods. Rewards
cover actual speed, heading/corridor, posture, slip, target smoothness, foot
release, stop behavior and estimated mechanical work. Moving foot pitch is
not forced flat; quiet stop-foot orientation is rewarded. Corner clearance
uses the separate foot transforms.

## Interpreting the logs

`active.log`, `baseline.parallel.log`, and `locked.parallel.log` are the full
training console records. `*.metrics.jsonl` contains sparse training episode
summaries. `training_summary.json` reads these saved records and
the final checkpoint iteration metadata without executing a policy.

Reported speeds are training episode summaries from stochastic, changing
policies, including stand commands. A last-50-iteration average is descriptive
training telemetry, not a held-out deterministic evaluation or an independent
multi-seed estimate. No speed gain, gait quality, graceful-stop acceptance or
running claim follows merely from completing 500 iterations.

The observation/reward history uses finite differences for sole slip. The
runner randomizes initial episode ages, so the very first transition may have
an initialization artifact in that slip proxy; ordinary reset steps suppress
that term until history is available. This shared first-step limitation should
be removed before a larger study, without retroactively claiming this pilot
used the corrected code.

Frequent target-rate limiting is an experiment-design signal, not evidence of
physical torque saturation. The next training revision should address that
interaction between stochastic position actions and the target limiter, fix
the initial history artifact, and establish consistent stepping before raising
speed commands. Preserve these pilot weights as the original comparison.
Any later deterministic evaluation, additional training seeds or hardware work
would be separate work; none was run as part of this pilot.

## Reproduction

```powershell
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v2/import_assets.py
& ./simulation/isaac/rl/ankle_v2/run_pilot.ps1 -Iterations 500 -NumEnvs 128 -Seed 1901
```

This creates a fresh run directory. Exact launch arguments, source snapshots
and SHA-256 records for this run are retained in the batch directory. The
source URDFs and existing V43/V44 training/hardware code were not changed.
