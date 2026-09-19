# V44 study35: offline Pi integration, not deployment

The user authorized preparing the exact selected candidate and running offline
mapping/timing checks before deployment. No SSH, bus access, motor commands,
Pi service changes, training, simulation reruns, UI validation, or Git publication
are part of this step. Existing hardware-tested modes remain the fallback.

## Prospective checks

Use the complete saved study35 canonical trace (two nominal episodes, 3,528
controller rows) and the hash-bound study34 checkpoint used for that evaluation.
No new physics or model training is needed.

- ONNX versus saved GPU actions and the CPU checkpoint actor: maximum absolute
  action error <= 3e-6. Actor normalizer is identity, deterministic Beta mean.
- Ported reference/corrections/F5 projection/5 mm lift/requested and delivered
  targets: maximum absolute error <= 2e-6 rad against saved stages. Compare
  every available row, including startup, phase transitions, deceleration, hold.
- Exercise the actual WalkingPolicyLoop with replay IMU/encoder samples and
  an offline sink. Independently check observation assembly and phase timing;
  state clearly which sensor channels come from recordings rather than hardware.
- Check clock behavior under delayed intervals, unchanged packet/rate limits,
  acknowledged rather than queued target history, and fresh-instance reset.
- Reject missing/unknown/altered experimental mapping, mismatched model/reference
  assets, nonfinite inputs and invalid geometry. Default model loading and a
  non-offline sink must reject this experimental bundle.
- Measure local CPU inference plus mapping time on the recorded workload.
  Report p50/p95/p99/max, environment and sample count; no Pi deadline claim.
- Check existing legacy target-mode behavior only where it intersects this port;
  do not run unrelated tests, builds, simulators or hardware checks.

If a check fails, preserve the failed report, fix the implementation, and rerun
the same checks. Do not loosen the tolerances or change the selected policy.

## Implementation boundaries

The new mapper uses cached NumPy reference arrays and a scalar analytic F5
inner-domain projector. It has no Torch, Isaac, serial, IMU or ROS dependency.
Joint-frame matrices and soft limits were extracted from the unchanged canonical
URDF with SHA256 `cfae29e48a29d5d63db99a9f4443f05c0074d7100c890b227553d35a3aa6b474`.
The URDF skill's frame rules informed this extraction; no robot geometry was
edited, no URDF validator/viewer was run.

Pipeline: reference + scaled residual -> front stance correction -> rear swing
correction -> common-F5 projection -> 5 mm fixed-pitch lift -> soft limits ->
elapsed-time and 2-degree packet limits. The simulated 30 ms / 0.97 actuator
response is a plant model and is intentionally NOT added as another Pi filter.

The selected configuration also evaluated heading feedback: gain 0.75/s, maximum
correction 0.12 rad/s. This changes the actor's yaw observation, not the nominal
turn request that controls residual authority. At zero forward command, heading
correction alone must not reactivate leg motion. Only forward/stop nominal
commands from zero through 0.02227272727272727 m/s are accepted; the intervening
values support the stop ramp, not a claim that the whole range is qualified.

Experimental phase integration uses the previous interval's command frequency;
legacy models retain their existing code path. The first simulated stop sample
already includes one control interval; the selected adapter preserves that
convention. Last host-completed bus-write acknowledgements remain the history
source; these are not proof of encoder tracking or servo receipt. Measured
joint positions separately supply the causal tracking correction.

`onboard/models/experimental-v44-study35` is explicitly offline-only. Loading it
requires `allow_experimental_offline=True`, and WalkingPolicyLoop additionally
requires a sink declaring `offline_only=True`. Existing web/hardware callers
do neither and cannot activate this candidate accidentally. Production export
guards and the original failed/qualified experiment decisions are unchanged.

## Results: 2026-09-06 PDT / 2026-09-07 UTC

**Offline integration passed; deployment remains disabled.** This is the selected
V44 study35 combination (study34 checkpoint396 plus study35 controller mapping),
not a newly trained policy and not the old production V35 model.

[Final machine-readable report](../../simulation/reviews/v44-actuator-20260905/offline-pi-integration/20260907T024938011837Z/report.json)
binds bundle manifest SHA256
`d82e9be64162e0d88e3c3c885278eda128a673f6f34ff6ca5487dc5e7bdd60cc`.
The manifest binds the ONNX graph, external weights, both reference assets,
sidecar, runtime sources and preparation script. It records the selected
checkpoint/configuration and canonical trace hashes. The checker records its
own hash and rejects changed bound files before inference.

| Offline check | Result |
| --- | --- |
| ONNX vs CPU checkpoint, 3,528 samples | Max action difference 4.77e-7; limit 3e-6 |
| ONNX vs saved GPU actions, 3,528 samples | Max action difference 5.36e-7; limit 3e-6 |
| Direct target mapping, all 3,528 samples | Max delivered difference 1.19e-7 rad; limit 2e-6 |
| Contact schedule and recursively integrated phase | Exact agreement |
| Actual loop with its own 25 s run / aligned 0.9 s ramp / 3 s hold | Both full episodes pass; max delivered difference 1.79e-7 rad |
| F5 scalar vs frozen Torch calculation | Exact targets in 100 cases; 20 identity and 80 changed rear-leg cases |
| Delay, packet/rate caps, unacknowledged history, fresh instance | Pass; slow fake sink skips deadlines without catch-up bursts |
| Invalid metadata/assets/input/geometry and accidental hardware loading | All named rejection cases pass |
| Existing residual/clock/stop intersection tests | 8 passed in 0.13 s |

The runtime replay supplies recorded angular velocity, projected gravity,
acceleration and pre-step joint positions/velocities. Heading is independently
reconstructed from the previous recorded root quaternion. Its own predicted
targets recursively feed the fake completed-write acknowledgements. The final
scheduled replay does NOT feed golden commands, gait phases, delivered targets
or stop triggers into the loop. The dynamics and sensors are still recorded
exogenous data, not closed-loop real hardware or a new physics evaluation.

The 100 additional projection cases span 20 recorded contexts and deterministic
hip/knee perturbations. They verify identity and nonidentity implementation
branches and partial-lift geometry; they do not certify every possible state.
The full nominal trace contains 2,730 active rear projection contexts. Its
positive 5 mm lift envelope matches within 1.22e-17 m numerical difference.

Local CPU ONNX + mapping time over 3,528 samples: p50 0.199 ms, p95 0.376 ms,
p99 0.423 ms, maximum 0.627 ms. Windows 11, Python 3.12.13, NumPy 2.3.1,
Torch 2.11.0, ONNX Runtime 1.24.1, one inference thread, CPU provider.
This excludes real IMU, serial traffic, OS scheduling and Pi CPU performance;
it cannot establish a Raspberry Pi 60 Hz deadline or end-to-end reliability.

### Failures retained, not hidden

- [Attempt 1](../../simulation/reviews/v44-actuator-20260905/offline-pi-integration/20260907T024306234901Z/report.json):
  `(lift * 5) / 5` could round slightly above the certified maximum and correctly
  trip its guard. Changed arithmetic to `lift * (stage / 5)`, whose final
  multiplier is exactly one. No guard or limit was relaxed.
- [Attempt 2](../../simulation/reviews/v44-actuator-20260905/offline-pi-integration/20260907T024323523778Z/report.json):
  replay initially treated corrected yaw as the nominal turn request, keeping
  residual motion active during stop. The initial sidecar had also omitted the
  evaluated heading feedback. Corrected both and bound the saved configuration.
  Live previous-tick stop alignment introduced an extra scheduling delay;
  selected-only alignment now uses the simulator's nominal reset-time phase
  and nearest-control-tick convention. Generic model behavior was not changed.
- Subsequent passing reports remain alongside the final report. The final
  version includes independent full-scheduler replay and projection edge cases;
  an earlier passing report did not yet include those additional checks.

No tolerance, reward, learned weight, physical limit, current/voltage/temperature
threshold or original study qualification gate was changed to make these pass.

## Reproduction (offline only)

Run from the repository root. The preparation step refreshes the local
experimental bundle; it never connects to the Pi. The checker imports the pure
Torch projector for arithmetic comparison but never imports Isaac SimulationApp
or starts a simulator. Every checker invocation creates a new report directory.

```powershell
# The isolated dependency was installed for this authorized offline check.
# Needed only if outputs/v44-study35-offline-deps is absent:
& 'C:/isaacsim/kit/python/python.exe' -m pip install --no-deps --target outputs/v44-study35-offline-deps onnxruntime==1.24.1

& 'C:/isaacsim/kit/python/python.exe' onboard/scripts/prepare-v44-study35.py
& 'C:/isaacsim/kit/python/python.exe' onboard/scripts/check-v44-study35-offline.py

# Only the legacy intersection file, not the full test suite:
$env:PYTHONPATH = (Resolve-Path 'onboard/policy-runtime/src').Path
& '.venv/Scripts/python.exe' -m pytest onboard/policy-runtime/tests/test_residual_contract.py -q
```

The checker requires the saved canonical trace, selected checkpoint and frozen
arithmetic projector retained in this development repository. The deployment
runtime itself needs only NumPy/ONNX Runtime and packaged runtime sources, not
Torch, training logs or Isaac. No Pi installer or service selection was changed.

## Remaining boundary before a real trial

This does not change study35's qualification: 28/30 nominal gates, narrow
clearance/release margins, remaining slip/effort-proxy failures, approximately
0.0154 m/s achieved motion. It is **not a qualified running model**.

Next, with separate authorization: stage an explicitly experimental Pi option,
measure Pi inference/controller timing with no motor writes, and validate the
current calibration/IMU and completed-write semantics. Keep the known walking
fallback selected until a supervised, fall-supported short trial is arranged.
Do not simply remove `offline_only` checks or overwrite the active policy.
No SSH, hardware checks, Pi deployment/restart, physical motion, retraining,
new simulation, Git commit or push was performed during this integration step.
