# V31 fast run and graceful stop

## Status

V31 is a rejected real-hardware research policy for faster forward motion
followed by a controlled stop. The selected simulation candidate is
`model_1650.pt`. It was deployed for one guarded trial on 2026-09-02 and did
not produce a run: the legs oscillated laterally while loaded joints remained
far behind their targets. It does not replace the V30 crawl policy or its
checkpoints, and V30 remains the fallback.

Identifiers:

- task: `Drobot-Commanded-Run-Fast-Stop-External-Rear-Payload-Direct`;
- command set: `fast-run-stop-external-rear-payload`;
- experiment: `drobot_commanded_run_v31_fast_stop_external_rear_payload`.

The target behavior is a grounded diagonal run: move straight at the requested
speed without a flight phase, tolerate either a short deceleration command or
an immediate zero-speed command, settle without falling, and remain in a
four-foot stance after stopping.

## Why this is a separate policy

The validated crawl is deliberately statically stable and normally releases
only one foot. Optimizing that topology for maximum speed would trade away the
behavior that makes it useful as a fallback. V31 therefore has its own task,
experiment directory, evaluator, preview profile, and eventual export folder.
Nothing in the V31 workflow selects or deploys it on hardware automatically.

## Run and stop schedule

Each ten-second training episode samples a positive forward-speed target,
runs for `4.5-6.5 s`, and then changes the target to zero. Sixty-five percent
of episodes use a cubic smooth deceleration lasting `0.35-0.90 s`; the other
thirty-five percent issue an immediate zero command. The abrupt cases prevent
the actor from relying on a controller-side ramp, while the smooth cases teach
a lower-jerk normal stop.

The command curriculum begins at `0.05-0.09 m/s` and expands to
`0.05-0.20 m/s` over 48,000 policy steps. Keeping the mature lower bound at
`0.05 m/s` is intentional: an earlier `0.10 m/s` mature minimum produced a
good high-speed checkpoint that forgot rear-left release at `0.08 m/s`.
Uniform sampling across the repaired range still left that leg marginal, so
the final curriculum is a mixture: half of environments sample the full
`0.05-0.20 m/s` range and half deliberately sample `0.05-0.10 m/s`. High-speed
commands remain present in every batch while guarded-trial speeds receive
enough practice.

The gait clock is integrated rather than recomputed from wall-clock time, so
cadence can change continuously as the speed command ramps down. At zero
command the clock parks and the analytic reference requests all four feet on
the ground in the symmetric 80 mm stance, with a level-body target. This park
pose is deliberately distinct from the lower, forward-biased running stance
because the first-stage effort audit found that reusing the run geometry made
a stationary hold more expensive than the run itself.

The evaluator can reproduce a fixed schedule exactly:

```powershell
& .\simulation\isaac\rl\parallel_walking\evaluate_run_stop.ps1 `
  -Checkpoint <candidate.pt> -Speed 0.12 `
  -RunSeconds 4 -RampSeconds 0.60 -HoldSeconds 3 `
  -Episodes 5 -DomainMode nominal `
  -Output .\result.json
```

Set `-RampSeconds 0` for the separate abrupt-stop case.

## Gait and hardware contract

V31 uses a `60%` duty-factor diagonal pattern with phase offsets
`(0.0, 0.5, 0.5, 0.0)`. This gives two short four-foot overlap intervals per
cycle without requiring an airborne body phase. Frequency scales from
`0.9-2.4 Hz`, reference stride is `65 mm`, foot lift is `26 mm`, and the target
slew limit remains `240 deg/s` at the 60 Hz deployment control rate.

The actor remains the same 50-input, 12-output direct-action network used by
the deployable walking family. It receives commanded planar velocity and yaw,
gait-clock sine/cosine, IMU angular velocity, projected gravity and linear
acceleration, joint position error, joint velocity, and previous action. It
does not require simulator-only state at deployment.

The environment retains V30's measured rear battery assembly, robot mass and
inertia, low forward-biased stance, padded-foot contact assumptions, servo
limits, and targeted domains for payload placement, chassis tilt, servo zero,
servo strength/rate, supply sag, and traction. Nominal training pins those
domains while the gait topology adapts; robust continuation enables them only
after a nominal candidate works.

## Reward design

V31 retains the walking policy's upright, body-height, rear-payload pitch,
straight-path, slip, touchdown-impact, action/joint/body acceleration, target
limiter, and sustained-effort objectives. Its motion terms give more weight to
commanded forward velocity, instant progress, and sustained progress, while
explicit overspeed and backward-motion costs prevent a single lunge from being
scored as a run.

Sequential-crawl-only rewards are disabled. The replacement gait terms score
diagonal timing, scheduled stance/swing, touchdown quality, and proximity to
the continuous run reference.

The first direct-action continuation exposed a contact loophole: both front
feet released reliably while the rear-left foot remained loaded through most
of its scheduled swing. V31 now evaluates each scheduled diagonal pair, uses
the weaker foot's soft release quality as a gate on positive motion reward,
rewards paired release directly, and penalizes mismatch inside the pair. The
final repair also rewards completion of a cycle in which all four feet release
and applies signed shaping to whichever foot has accumulated the least release
activity. This prevents forward speed from hiding a dragged or inactive leg.

After the command reaches zero, additional terms:

- reward low planar velocity and four-foot contact;
- penalize travel beyond the predicted ramp endpoint;
- penalize residual yaw and joint motion;
- keep the normal upright, acceleration, effort, and contact objectives active.

The predicted endpoint is captured at the stop trigger and includes the
integral of the sampled deceleration ramp. For an abrupt stop, it is the trigger
position itself.

## Bootstrap decision

The first implementation tried to continue V30 as a residual policy around the
new diagonal reference. That branch was rejected. With the residual fixed at
zero, the reference moved only about `0.0028 m/s`; training mostly rocked in
place and the large stop reward could dominate the weak motion signal. Its run
directories are prefixed `_rejected-v30-residual-` so automatic checkpoint
resolution cannot select them.

V18 `model_299.pt` was then tested as a direct-action bootstrap under V31's
current physics. At an `0.08 m/s` command it remained upright, averaged
`0.1112 m/s`, and ended at `0.0014 m/s`, proving that the network and action
topology can produce useful forward motion and stop. It was not acceptable as
the V31 result: final heading error was `0.4396 rad`, speed RMSE was
`0.0662 m/s`, RMS joint acceleration was `17.07 rad/s2`, and above-soft-limit
effort occupied `0.557` of joint samples. The baseline report is:

```text
logs/rsl_rl/drobot_commanded_run_v31_fast_stop_external_rear_payload/
  v18-direct-bootstrap-run-stop.json
```

V31 nominal training therefore starts from V18 direct-action weights and
adapts them to the straight, smooth, speed-conditioned run/stop task.

## First nominal run/stop checkpoint

The paired-contact continuation reached `model_799.pt` after 32,000 command
curriculum steps. A held-out three-episode nominal evaluation at `0.12 m/s`
with a `0.60 s` ramp was fall-free, averaged `0.1273 m/s`, settled in
`0.622 s`, ended at `0.0015 m/s`, held all four feet down for the full stopped
window, and finished with `0.0093 rad` mean absolute heading error. Every leg
released on at least `94.8%` of its scheduled swings and paired release quality
was `0.978`.

It passed every behavioral, contact, path, stop, and acceleration gate but not
the rated-effort gate: `47.1%` of joint-time samples exceeded the rated-torque
soft threshold. The report is:

```text
logs/rsl_rl/drobot_commanded_run_v31_fast_stop_external_rear_payload/
  2026-09-01_00-08-41_manual-headless/evaluation-0p12-ramp-nominal.json
```

The final nominal curriculum continuation therefore increases the above-rated
effort cost, adds a stopped-only pull toward the calibrated neutral action, and
records effort separately for the run, deceleration, and hold phases. The
behaviorally passing `model_799.pt` remains preserved as a comparison rather
than being overwritten.

## Selected V31 simulation candidate

`model_1650.pt` is the selected V31 simulation candidate. It was chosen from a
contact-focused nominal continuation followed by a 200-iteration robust
continuation with 25% nominal anchor environments and 75% randomized
environments. Its curriculum sidecar records 86,464 policy steps. Training
domains include payload placement, body torque, joint-zero bias, actuator
strength and rate, supply-strength proxy, and padded-foot contact variation.

Held-out results for the exact selected checkpoint are below. Ramped and
abrupt nominal cases use three or five episodes as recorded in each JSON file;
the randomized case uses five episodes. Every listed case had zero falls,
settled in every episode, and passed the speed, heading, line, all-leg release,
paired-contact, final-speed, stop-position, planted-hold, action-acceleration,
joint-acceleration, and body-acceleration gates.

| Case | Actual speed | Minimum leg release | Heading | Lateral / forward | Settle | Final speed | Stable hold | RMS joint acceleration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.08 m/s, 0.60 s ramp, nominal (5) | 0.1006 m/s | 0.870 | 0.12 deg | 0.044 | 0.650 s | 0.0001 m/s | 0.984 | 11.60 rad/s2 |
| 0.08 m/s, abrupt, nominal (3) | 0.1010 m/s | 0.852 | 0.16 deg | 0.044 | 0.211 s | 0.0001 m/s | 0.930 | 11.83 rad/s2 |
| 0.12 m/s, 0.60 s ramp, nominal (3) | 0.1411 m/s | 0.896 | 0.21 deg | 0.014 | 0.600 s | 0.0003 m/s | 1.000 | 13.38 rad/s2 |
| 0.12 m/s, abrupt, nominal (3) | 0.1398 m/s | 0.892 | 0.33 deg | 0.003 | 0.122 s | 0.0002 m/s | 0.959 | 13.25 rad/s2 |
| 0.20 m/s, 0.60 s ramp, nominal (5) | 0.1970 m/s | 0.830 | 0.33 deg | 0.010 | 0.600 s | 0.0002 m/s | 1.000 | 15.63 rad/s2 |
| 0.20 m/s, abrupt, nominal (3) | 0.1988 m/s | 0.854 | 0.40 deg | 0.017 | 0.094 s | 0.0001 m/s | 0.969 | 15.35 rad/s2 |
| 0.16 m/s, 0.60 s ramp, randomized (5) | 0.1503 m/s | 0.927 | 0.12 deg | 0.020 | 0.600 s | 0.0014 m/s | 1.000 | 15.23 rad/s2 |

The remaining failed gate is sustained effort. Above-rated joint-time
occupancy ranges from `0.425-0.532` across this matrix versus the conservative
`0.10` limit. At 0.12 m/s it averages `0.456` while running, `0.421` while
decelerating, and `0.547` while holding. This is not hidden by the otherwise
passing behavior: the candidate remains a simulation research result until
servo current/voltage measurements establish a defensible effort baseline.

## Real-hardware outcome

The first `0.08 m/s`, five-second trial demonstrated that the simulation-only
effort warning corresponded to an actuator-feasibility failure. V31 drove every
hip and knee to its `240 deg/s` target ceiling, while the loaded encoders
reached roughly `77-115 deg/s` and lagged their periodic targets by
`0.27-0.47 s`. Intended diagonal pairs therefore lost physical phase even
though their policy targets were synchronized. Action saturation was `12.1%`
during the run and higher during the zero-command hold.

The policy loop also achieved only `50.24 Hz`: its target callback contended
with servo-bus work through the shared session lock. Feedback itself remained
fresh, integrated phase was correct, all joint correlations became positive
after allowing lag, and the IMU acceleration convention matched Isaac Lab.
These findings rule out a simple motor reversal, phase-clock bug, or gravity
sign error.

The complete numerical analysis and corrective V32 contract are in
`debug/walk-trials/20260902T040348286640Z-9f4db9ba/ANALYSIS.md`. V31 must not be
used free-standing again. V32 first removes target-handoff blocking, limits
training to `120 deg/s`, slows cadence, broadens actuator-response domains, and
adds hard selection gates for saturation and target-limit occupancy.

Selected artifacts are under
`simulation/isaac/models/parallel-running-v31-fast-stop/`. Checkpoint SHA-256
is `7ef75a7e5da7fd4eb7d6863eb5a4c8f9913fef213b5a5ad02adebe42f5ab620b`;
ONNX SHA-256 is
`9faa1336cbe351b94b2dbdc70509d0d140c8288df2d279c74aebf00ce1cae174`.
PyTorch/ONNX maximum absolute action error is `2.205371856689453e-06`
over 256 bounded observations using the repository's numerically stable ONNX
reference verifier. The nine-second review video runs at 0.16 m/s for five
seconds, ramps to zero over 0.60 seconds, and holds the stopped stance.

## Candidate acceptance

`evaluate_run_stop.py` records per-episode phase metrics rather than relying on
PPO return. A release candidate must be fall-free, track between 65% and 135%
of requested run speed, settle in every episode within 1.5 seconds, end below
`0.03 m/s`, remain stable for at least 80% of the stopped hold, keep all four
feet down for at least 90% of that hold, stay within the stop/line/heading
limits, remain under joint and body acceleration limits, and avoid sustained
above-rated effort.

Selection requires fixed-speed nominal cases at `0.08` and `0.12 m/s`, both a
`0.60 s` graceful ramp and an abrupt stop, followed by randomized-domain
evaluation. A visually convincing video is useful for review but cannot
override a failed quantitative gate.

## Training and preview commands

Nominal continuation:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet fast-run-stop-external-rear-payload `
  -Iterations 500 -NumEnvs 4096 -Seed 3121 -V25Phase nominal `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v18-coordinated\model_299.pt
```

Once a nominal checkpoint passes the deterministic cases, continue it with
`-V25Phase robust` and a new seed. Do not start robust training merely because
the nominal job reached its requested iteration count.

Preview a run, deceleration, and stopped hold without changing the Pi:

```powershell
& .\simulation\isaac\rl\parallel_walking\preview_walking.ps1 `
  -CommandSet fast-run-stop-external-rear-payload `
  -Checkpoint <candidate.pt> -Command forward -ForwardSpeed 0.12 `
  -RunThenStop -RunSeconds 5 -StopRampSeconds 0.60 `
  -RecordSeconds 9 -NoTimeLimit
```

## Deployment boundary

Training completion is not deployment approval. The selected V31 checkpoint
must be exported with a sidecar that records the run/stop task, speed range,
integrated speed-scaled gait clock, cadence range, and standstill behavior.
The Raspberry Pi UI should expose V31 as a distinct run mode with the V30 crawl
still available for immediate rollback. A physical trial should begin tethered
or supported, at the lowest trained speed, with a short duration and a clear
stop/disarm path.
