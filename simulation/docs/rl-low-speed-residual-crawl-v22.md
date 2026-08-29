# V22 low-speed residual crawl

## Goal

V22 responds to the August 29 V20 real-walk trial in
`debug/walk-trials/20260829T204215017854Z-299dd310`. That trial remained
upright but oscillated laterally, repeatedly reversed yaw, saturated several
actions, and moved too aggressively for the requested smooth crawl. The V22
selection priority is smooth, straight, three-foot-supported motion; maximum
speed is deliberately secondary.

## Controller and simulation contract

The selected design does not ask PPO to rediscover a crawl from zero. It uses
the proven hardware distributed-push sequence as a deterministic base:

1. rear-right swing;
2. front-right swing;
3. rear-left swing; and
4. front-left swing.

The base reference includes weight transfer, lift, forward swing, slow lower,
firm plant, weight return, a four-foot push, and settle. Rectangular support
shoes remain flat during planted phases. PPO emits a bounded residual with
only 25% of the normal action scale. Training and deployment both form the
target as `crawl reference + 0.25 * policy residual`, clamp it to joint limits,
and apply a 120 deg/s target slew limit (two degrees at 60 Hz).

The low-speed command range is `0.003-0.015 m/s`. Cadence scales from
`0.06-0.30 Hz`, retaining the full 50 mm reference stride. Startup blends from
the trained neutral pose over 1.5 seconds. The simulator retains V20's external
rear payload and mass/COM uncertainty.

The ONNX JSON sidecar embeds the same 2,048-sample reference table used by
Isaac, plus cadence, stride scale, residual authority, neutral pose, target
slew, and startup fields. An ONNX file without that sidecar is not a complete
V22 model.

## Training search and rejected formulations

All runs use experiment
`drobot_commanded_walk_v22_low_speed_crawl_external_rear_payload`.

| Run | Formulation | Outcome |
| --- | --- | --- |
| `2026-08-29_14-51-37_manual-headless` | V20 continuation, sequential contact objective | Rejected: rear-right never achieved qualified touchdowns. |
| `2026-08-29_14-57-51_manual-headless` | Stronger equal-leg continuation | Rejected: front-leg activity remained dominant and rear participation collapsed. |
| `2026-08-29_14-59-41_manual-headless` and `2026-08-29_15-02-47_manual-headless` | Fresh direct-action crawl | Rejected: standing/front-only local optima and inadequate four-leg stepping. |
| `2026-08-29_15-08-49_manual-headless` and `2026-08-29_15-11-54_manual-headless` | Direct action with distributed-push imitation | Rejected: the policy still regressed away from the full four-leg reference. |
| `2026-08-29_15-17-36_manual-headless` | Short fresh residual proof | Promising: all four legs produced qualified touchdowns without falls. Used only to validate the formulation. |
| `2026-08-29_15-20-35_manual-headless` | Fresh 1,000-iteration residual run | Qualification candidate; final checkpoint selection is recorded below. |

Rejected training logs remain local diagnostic artifacts and are not deployed.
The architectural conclusion is important: contact rewards and imitation alone
did not stop a direct-action policy from abandoning one or more legs. Keeping
the known-good crawl in the controller and learning only bounded corrections
preserves gait structure by construction.

## Reward priorities

V22 rewards commanded forward tracking, sustained low-speed progress, the
distributed reference, scheduled stance/swing, balanced qualified touchdowns,
and three-or-four-foot support. It penalizes stalls, two-or-fewer-foot support,
target-limiter backlog, lateral/yaw motion, tilt, action rate and acceleration,
joint/body acceleration, slip, touchdown impact, saturation, and termination.
The smoothness curriculum reaches full strength before checkpoint selection.

## Qualification gate

A checkpoint is deployable only if independent deterministic evaluation shows:

- zero falls at `0.003`, `0.005`, `0.010`, and `0.015 m/s`;
- touchdowns from all four legs, with no persistently inactive leg;
- predominantly three-or-four-foot support and little two-foot support;
- low requested-to-applied target-limiter gap and low action saturation;
- sustained progress appropriate to the command without a burst-and-stall
  pattern;
- low lateral displacement and absolute yaw travel; and
- visibly smooth motion in a recorded Isaac preview.

## Selected checkpoint and evaluation

The fresh run completed 1,000 iterations. Checkpoints 500, 750, and 999 were
screened independently at `0.010 m/s`. Checkpoint 500 was selected because its
joint RMS acceleration was `6.31 rad/s2`, versus `8.40` and `8.56 rad/s2`; it
also had the lowest lateral displacement and absolute yaw travel. The later
checkpoints moved faster but violated the smoothness-first selection goal.

Checkpoint 500 then completed three deterministic 30-second episodes at each
command:

| Command | Mean actual speed | Joint RMS accel. | Mean lateral displacement | >=3-foot support | Falls / stalls |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.003 m/s` | `0.00535 m/s` | `4.29 rad/s2` | `0.0293 m` | `96.3%` | `0 / 0` |
| `0.005 m/s` | `0.00709 m/s` | `5.42 rad/s2` | `0.0298 m` | `95.0%` | `0 / 0` |
| `0.010 m/s` | `0.01293 m/s` | `6.71 rad/s2` | `0.0344 m` | `86.4%` | `0 / 0` |
| `0.015 m/s` | `0.01736 m/s` | `8.36 rad/s2` | `0.0313 m` | `80.2%` | `0 / 0` |

Every episode had touchdowns from all four legs, zero normalized-action
saturation, and low target-limiter backlog. The minimum command overshoots by
about `0.00235 m/s`; this is retained as a disclosed limitation rather than
claiming exact speed tracking. Hardware rollout should begin at
`0.003-0.005 m/s`, where support and smoothness are strongest.

Selected artifacts:

- training checkpoint:
  `simulation/isaac/models/parallel-walking-v22-low-speed-residual-crawl/model_500.pt`;
- deployable actor and sidecar:
  `onboard/models/parallel-walking-v22-low-speed-residual-crawl/model_500.onnx`
  and `model_500.json`;
- 20-second `0.005 m/s` review video:
  `simulation/reviews/parallel-walking-v22-low-speed-residual-crawl-model500-20s.mp4`;
- ONNX SHA-256:
  `ddbc2fa70661a5a81342eb76b884888e3da2a1b88c63ef241cd77afbc95ccfe0`.

## Real-robot rollout

After qualification, begin with the robot supported: prepare the
model-declared neutral stance, confirm all 12 encoders are settled, and run a
short `0.003 m/s` trial. Continue with a floor trial only after reviewing the
automatic IMU/joint recording. Hardware evidence remains the final authority;
Isaac qualification reduces risk but does not prove floor safety.
