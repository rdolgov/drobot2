# Reinforcement-learning walking task

## Current status

The first trainable locomotion pipeline is implemented as
`Drobot-Quadruped-Walk-v1`. It runs Stable-Baselines3 PPO directly against the
validated Isaac Sim 6.0.1 manual world. A `2,002,944`-step checkpoint now
demonstrates repeatable simulated walking: a 20-episode deterministic
evaluation completed every eight-second episode without tipping and averaged
`0.1445 m/s` forward speed against the `0.15 m/s` command.

This is a useful simulated walking result, not evidence of sim-to-real
transfer. The policy still drifts consistently toward negative lateral Y.

Isaac Lab is NVIDIA's preferred high-throughput robot-learning framework, but
the Isaac Sim 6.0-compatible Isaac Lab 3 line is still beta. The current
direct Gymnasium environment avoids making the project depend on that beta
API. Its main tradeoff is one robot environment per Isaac process, so it will
train much more slowly than a future vectorized Isaac Lab task.

## Source of truth

| File | Responsibility |
| --- | --- |
| `simulation/isaac/rl/quadruped_walk_v1.yaml` | Task, reward, reset, action, and PPO configuration |
| `simulation/isaac/rl/_rl_contract.py` | Pure NumPy 48-value observation and reward contract |
| `simulation/isaac/rl/_quadruped_rl_env.py` | Gymnasium environment bound to the Isaac articulation |
| `simulation/isaac/rl/train_ppo.py` | PPO training, checkpoints, TensorBoard, and run report |
| `simulation/isaac/rl/play_ppo.py` | Deterministic evaluation and optional camera screenshot |
| `simulation/isaac/rl/record_ppo.py` | One-episode deterministic external or onboard H.264 recording |
| `scripts/setup_isaac_rl.ps1` | Tested Isaac Python dependency installation |
| `exports/isaac/quadruped_robot_manual_world.usda` | Validated gravity, contact, servo, camera, and IMU world |

Generated models and logs stay under `simulation/isaac/output/rl/` and are
ignored by Git. Promote a checkpoint deliberately only after evaluation.

## Tested software

- Isaac Sim `6.0.1`, Python `3.12`;
- PyTorch `2.10.0+cu128`;
- Gymnasium `1.2.0`;
- Stable-Baselines3 `2.9.0`;
- TensorBoard `2.21.0`;
- NVIDIA GeForce RTX 5090 in the current workstation.

The tiny MLP policy trains faster on CPU in the present single-environment
setup. Isaac/PhysX and rendering still use the NVIDIA simulator stack.
`--device cuda` is available, but Stable-Baselines3 warns that PPO MLP
optimization can be slower on GPU until rollouts are parallelized.

## Install the RL dependencies

From `robot-cad/`:

```powershell
.\scripts\setup_isaac_rl.ps1
```

The setup script installs the CUDA 12.8 PyTorch build and the pinned packages
from `simulation/isaac/rl/requirements.txt` into Isaac Sim's bundled Python.

## Policy interface

The action is 12 normalized joint-position offsets in the articulation's
reported DOF-name order. The environment maps them to conservative offsets
from the validated standing pose:

| Joint kind | Maximum policy offset |
| --- | ---: |
| hip abduction | `0.12 rad` |
| hip flexion | `0.30 rad` |
| knee | `0.40 rad` |

Targets are clamped to URDF limits and rate-limited by the verified ST3215
no-load velocity. Every joint remains capped at the sustainable
`0.980665 N·m` rated torque.

The 48-value observation contains:

1. commanded forward/lateral/yaw velocity: 3;
2. IMU angular velocity, projected gravity, and acceleration: 9;
3. joint position error from the nominal stance: 12;
4. joint velocity normalized by each URDF speed limit: 12;
5. previous normalized policy action: 12.

Simulator-only base linear velocity is excluded from the policy observation.
It is used only to calculate the training reward, keeping the observation
reproducible on hardware from commands, BNO085 data, servo feedback, and the
previous action.

## Reward and termination

The primary positive reward tracks the configured `0.15 m/s` forward body
speed. Smaller terms reward upright posture and remaining alive. Penalties
cover lateral/vertical motion, roll/pitch/yaw rate, body-height error, action
rate, action magnitude, and joint velocity.

An episode ends when:

- body height drops below `0.22 m`;
- upright cosine drops below `0.78`;
- the configured eight-second time limit is reached.

A non-finite simulator or sensor value aborts the training run and leaves a
failed `training_report.json`; it is not silently converted into a normal
episode reset.

Every term and threshold is visible in `quadruped_walk_v1.yaml`; do not tune a
hidden constant solely to make a report pass.

## Run a pipeline smoke test

This validates the complete code path without claiming policy convergence:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\train_ppo.py `
  --smoke-test `
  --output-dir simulation\isaac\output\rl\smoke-v1
```

Expected artifacts:

- `training_report.json`;
- `monitor.csv`;
- `tensorboard/`;
- `checkpoints/*.zip`;
- `drobot_walk_ppo_final.zip`.

### Recorded smoke validation: 2026-07-26

The tested 512-step smoke run passed:

- two PPO update batches completed and a `1,951,476` byte checkpoint was saved;
- one stochastic rollout reached the eight-second time limit without falling;
- the deterministic checkpoint evaluation also reached eight seconds without
  falling, with `0.234 degrees` maximum tilt;
- deterministic forward displacement was only `3.42 mm`;
- the mounted camera evaluation path wrote a `198,772` byte onboard PNG.

These numbers validate reset, rollout, reward, optimization, checkpoint,
reload, deterministic evaluation, and camera capture. The displacement is too
small to call the checkpoint a walking policy.

## Run a full training job

The default configuration requests 500,000 simulator steps:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\train_ppo.py `
  --output-dir simulation\isaac\output\rl\ppo-walk-v1
```

Resume a saved checkpoint:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\train_ppo.py `
  --resume simulation\isaac\output\rl\ppo-walk-v1\checkpoints\drobot_walk_ppo_250000_steps.zip `
  --total-timesteps 250000 `
  --output-dir simulation\isaac\output\rl\ppo-walk-v1-resumed
```

Monitor learning:

```powershell
& C:\isaacsim\python.bat -m tensorboard.main `
  --logdir simulation\isaac\output\rl\ppo-walk-v1\tensorboard
```

## Evaluate a checkpoint

Headless deterministic evaluation:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\play_ppo.py `
  --model simulation\isaac\output\rl\ppo-walk-v1\drobot_walk_ppo_final.zip `
  --episodes 3 `
  --report simulation\isaac\output\rl\ppo-walk-v1\evaluation_report.json
```

Open Isaac and watch from the mounted camera:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\play_ppo.py `
  --model simulation\isaac\output\rl\ppo-walk-v1\drobot_walk_ppo_final.zip `
  --episodes 1 `
  --gui `
  --camera-view onboard
```

The camera is intentionally excluded from the version-1 policy observation.
It remains available for monitoring and evaluation. Image-based locomotion
would require rendering every rollout, a feature encoder, substantially more
VRAM/time, and a separately documented reward/observation version.

Record one deterministic episode from a static external review camera:

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\record_ppo.py `
  --model simulation\isaac\output\rl\ppo-walk-v1-2m\drobot_walk_ppo_final.zip `
  --video reviews\ppo-walk-v1-2m-evaluation.mp4 `
  --thumbnail reviews\ppo-walk-v1-2m-evaluation.png `
  --report simulation\isaac\output\rl\ppo-walk-v1-2m\recording_report.json
```

The recorder runs the same deterministic policy loop as `play_ppo.py`, reads
an offscreen RTX camera at 30 FPS, and encodes H.264 directly through Isaac's
video encoder. The external camera is static so forward progress and lateral
drift remain visible. Use `--camera-view onboard` to record the mounted camera
instead.

### Recorded two-million-step evaluation: 2026-07-27

The final `ppo-walk-v1-2m` checkpoint passed a 20-episode deterministic
evaluation from seed `43`:

- all `20/20` episodes reached the eight-second time limit without tipping;
- mean forward displacement was `1.156212 m`, or `0.144527 m/s`;
- mean lateral displacement was `-0.249181 m`, exposing a repeatable left
  drift that still needs correction;
- mean maximum body tilt was `7.985806 degrees`, with a `9.159999 degree`
  worst case;
- minimum base height across the evaluation was `0.357794 m`;
- mean episode return was `1130.111581`.

The [external review video](../reviews/ppo-walk-v1-2m-evaluation.mp4) records
one representative deterministic episode. It contains `240` decoded H.264
frames at `960 x 540`, `30 FPS`, and eight seconds. That episode moved
`1.160433 m` forward, drifted `-0.199756 m` laterally, reached
`8.378761 degrees` maximum tilt, and did not tip. The adjacent
[thumbnail](../reviews/ppo-walk-v1-2m-evaluation.png) captures the camera
framing. The ignored `recording_report.json` retains the exact runtime
contract without publishing workstation-local paths.

## Acceptance before calling a policy useful

A full run should be evaluated across multiple seeds and should show:

- increasing mean episodic return rather than only decreasing loss;
- sustained positive forward displacement;
- low termination rate and no non-finite state;
- acceptable body tilt, height, and lateral drift;
- adherence to rated torque and joint limits;
- robustness after mass, friction, voltage/torque, latency, IMU noise, and
  initial-pose randomization.

## Known limitations and next work

- The current task has one simulated robot, not Isaac Lab parallel clones.
- The distal fork-tip contact proxy is still standing in for a printed foot.
- Motor backlash, battery voltage sag, thermal derating, communication delay,
  dropped packets, and measured IMU noise are not yet randomized.
- The reset pose always starts on flat ground with zero body velocity.
- The two-million-step policy demonstrates useful flat-ground walking in the
  exact training simulator, but it has only been evaluated in one seeded reset
  stream and retains about `0.25 m` of left drift per eight-second episode.
- No sim-to-real transfer has yet been demonstrated.
- A trained checkpoint must be tested first in simulation at rated torque,
  then on suspended/guarded hardware with an emergency stop before floor use.

## Upstream references

- [Isaac Lab reinforcement-learning training guide](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/training_guide.html)
- [Isaac Lab 3 / Isaac Sim 6 installation](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/setup/installation/index.html)
- [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
