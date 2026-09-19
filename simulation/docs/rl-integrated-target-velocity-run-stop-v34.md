# V34 integrated-target-velocity run and graceful stop

## Status

V34 is a corrective research policy under nominal training. It is not yet
approved for Raspberry Pi deployment. V30 remains the known hardware fallback;
V31, V32, and V33 are diagnostic/rejected run-policy stages.

Identifiers:

- task: `Drobot-Commanded-Run-Integrated-Target-Velocity-Stop-External-Rear-Payload`;
- command set: `integrated-target-velocity-run-stop-external-rear-payload`;
- experiment: `drobot_commanded_run_v34_integrated_target_velocity_stop_external_rear_payload`;
- bootstrap: reset-head checkpoint derived from V33 `model_2200.pt`.

The motivating real trace and controller analysis are in
`debug/walk-trials/20260902T040348286640Z-9f4db9ba/ANALYSIS.md`.

## Why V33 was rejected

V33 retained the 50-value observation size but replaced the previous raw
network action with the previous controller-applied normalized target. This
removed an observation ambiguity but left the network free to request an
unreachable absolute target every packet.

At `0.06 m/s`, five deterministic nominal episodes from V33 iteration 2200
had zero falls, `0.06287 m/s` mean speed, `0.0506` absolute lateral distance
per forward distance, four-leg scheduled release of at least `75.9%`, and
`99.2%` stable stopped hold. It nevertheless failed the hardware-feasibility
contract:

- `8.56%` of run actions were at or above `|0.9|`;
- `21.02%` of run joint requests differed from the rate-limited target by more
  than `0.5 deg`;
- `27.03%` of run joint motion occupied at least 98% of the packet slew budget;
  and
- `33.95%` of run joint samples and `21.37%` of stopped-hold samples exceeded
  the documented continuous-rated torque approximation.

The target mismatch was not materially better than V32, so V33 will not be
exported or tested on the robot.

## Action contract

V34 changes the meaning of each of the 12 actor outputs. An output `a` in
`[-1, 1]` advances the previous executable target by:

```text
delta_target = a * min(target_velocity_limit * elapsed_time,
                       maximum_packet_step)
```

The result is clipped only to the joint position envelope. At the nominal
`120 deg/s`, 60 Hz contract, `a=1` means a `2 deg` target increment. A missed
or late controller cycle uses real elapsed time but remains bounded by the
same `2 deg` packet cap. No queued catch-up target is replayed.

The final 12 observation values remain the previous applied target normalized
around the calibrated 30-degree neutral pose. Consequently the network knows
the integration state used by both Isaac Lab and the Raspberry Pi runtime.

For one-frame actuator delay randomization, V34 delays the velocity request
rather than an already integrated absolute target. This prevents a delayed
queue from creating a two-step jump.

## Reward interpretation

Under the V34 action contract:

- action magnitude is target speed;
- action rate is target acceleration;
- action second difference is target jerk;
- action saturation is continuous use of nearly the full target-speed budget;
- gait-reference error compares the applied integrated target with the
  diagonal reference pose; and
- during stopped hold, the neutral score measures applied target position
  while stopped-action magnitude damps target velocity to zero.

The positive velocity/progress, straight-line, four-leg release, paired gait,
support, impact, body/joint acceleration, and effort objectives remain from
V32. Requested-to-applied limiter penalty is zero because an over-budget
request is unrepresentable in this action space.

## Bootstrap

V33 and V34 share observation semantics, so V34 reuses V33's two hidden actor
layers as feature extraction. The output layer is reset to zero because an
absolute joint-position output cannot safely be interpreted as target
velocity. Zero raw Beta logits give deterministic mean velocity zero while
retaining stochastic exploration. The critic output and optimizer state are
also reset, and training iteration/curriculum age start at zero.

The reproducible bootstrap and its hashes are documented in
`simulation/isaac/models/parallel-running-v34-velocity-bootstrap/README.md`.
The bootstrap itself must never be exported or installed on hardware.

## First nominal checkpoint

Iteration 200 is the first deterministic confirmation that the new action
contract fixes the V31-V33 failure mechanism. Across five nominal `0.06 m/s`
episodes it achieved:

- zero falls and `0.06149 m/s` mean run speed (`102.5%` of command);
- `0.0164` absolute lateral distance per forward distance and `0.43 deg` mean
  final heading error;
- scheduled release of `88.8% / 91.8% / 84.0% / 98.0%` for FL/RL/FR/RR and
  `0.910` paired-release quality;
- exactly zero action saturation, target-limiter occupancy, and target-rate
  ceiling occupancy;
- `15.17 rad/s2` joint, `0.243 m/s2` body-linear, and `1.39 rad/s2`
  body-angular RMS acceleration; and
- a `0.60 s` settle, all feet down for the complete hold, and a fully stable
  hold with `0.00128 m/s` final speed.

The checkpoint is not accepted because the original aggregate torque gate
still fails: `32.84%` of run samples and `27.37%` of hold samples exceed the
continuous-rated approximation. Per-joint data shows that the stopped policy
damps velocity before every accumulated target returns to calibrated center.
Training therefore continues from iteration 200 with an explicit quadratic
stopped target-position cost. The evaluator now reports normalized hold-target
error and effort by phase and joint. No Raspberry Pi deployment is authorized
by this early result.

## Qualification sequence

1. Train in the exact nominal mass, payload, contact, and actuator domain.
2. Evaluate fixed `0.04`, `0.06`, `0.08`, and `0.10 m/s` commands with both a
   `0.60 s` ramp and an abrupt stop.
3. Reject falls, wrong-direction motion, excessive line/heading error, any
   underused leg, poor paired release, excessive acceleration, target-velocity
   saturation, or unstable stopping.
4. Inspect effort by run/deceleration/hold and by joint. Separate brief
   two-leg support torque from a wasteful stopped pose; do not hide them in one
   aggregate number.
5. Only a passing nominal checkpoint may continue in the randomized payload,
   supply, friction, actuator-rate, actuator-strength, delay, assembly-offset,
   and body-wrench domain.
6. Export ONNX only after robust deterministic evaluation. The first physical
   trial must be short, supported, at the selected policy's minimum speed, with
   recording enabled and V30 available for rollback.

## Commands

Nominal training:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet integrated-target-velocity-run-stop-external-rear-payload `
  -Iterations 1200 -NumEnvs 4096 -Seed 3401 -V25Phase nominal
```

Deterministic run-stop evaluation:

```powershell
& .\simulation\isaac\rl\parallel_walking\evaluate_run_stop.ps1 `
  -CommandSet integrated-target-velocity-run-stop-external-rear-payload `
  -Checkpoint <candidate.pt> -Speed 0.06 `
  -RunSeconds 4 -RampSeconds 0.60 -HoldSeconds 3 `
  -Episodes 5 -DomainMode nominal -Output <report.json>
```

Set `-RampSeconds 0` for abrupt stopping. Completion of a PPO iteration budget
is not policy acceptance.
