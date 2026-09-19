# V44 flat-contact / hardware-matched corrective candidate

Date: 2026-09-05. **Actual Isaac Lab pilots trained and evaluated; final candidate
fails nominal acceptance and is not deployable.** See the [training record](rl-v44-training-20260905.md)
for completed steps and results. V43's weights, JSON sidecar and reference files
remain unchanged. No claim of running performance is made.

The baseline is the [September 5 trial investigation](../../debug/walk-trials/20260905T193441588801Z-2a2a9832/ANALYSIS.md).
It found approximately 25-degree front sole tilt in the reference, 310–328 ms
front motion lag, 48.7 Hz policy updates, and a training/deployment packet-limit
mismatch. The similar-run video was not time-aligned with that ZIP.

## What changed

### Real controller

- The 8080 motor dashboard uses `BackgroundImuSource`. One worker owns BNO085
  construction, reads, and cleanup; the policy reads a copied complete cache.
  Startup can be cancelled and has a 30-second caller deadline. A 120 ms freshness
  guard rejects stale samples. Cleanup waits at most 0.5 s; a blocked driver
  retains ownership until it exits, preventing overlapping sensor sessions.
- Freshness is **host complete-acquisition age**, not a promise that the three
  Adafruit sensor properties have synchronized hardware timestamps. Acquisition
  failures still propagate; the change does not hide sensor faults.
- Optional applied-target snapshots report the last **host-completed bus
  command**, not servo acknowledgment or encoder position. Policy observations
  use that snapshot instead of treating a queued target as already sent.
  Recordings include target, sequence, and timestamp for later analysis.
- The producer and handoff step from that last-sent target. Producer rate
  budget is the larger of elapsed production time and the age of the last
  sent target; this avoids charging queue latency twice. The actual RL bus
  worker independently enforces the selected per-joint elapsed-time rates
  and packet cap, including after queue replacement. That bus guard is
  authoritative. Recordings label `target_limit_base_rad` and
  `target_elapsed_semantics` so producer budget is not confused with bus time.
- The motor safety envelope and normal stop/neutral-hold behavior are retained.
  This is not a change to manual hardcoded crawl geometry or its speed.

See [IMU lifecycle documentation](../../onboard/docs/imu-session-lifecycle.md)
for ownership, cancellation, failure handling, and remaining limitations.

### Separate training task

Profile: `flat-contact-hardware-matched-run-stop-external-rear-payload`.

Task: `Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload`.

| Setting | V44 candidate |
| --- | --- |
| Command range / recommended | 0.03–0.05 / 0.04 m/s |
| Cadence | 0.35–0.45 Hz |
| Pattern | Diagonal pairs FL+RR / FR+RL; 55% duty |
| Startup reference ramp | 1.5 s from existing symmetric neutral |
| Target rates | 60 degrees/s abduction, 90 degrees/s sagittal |
| Packet increment | At most 2 degrees, also bounded by elapsed-time target rate |
| Random packet holds | Probability 0.12 in randomized environments; at most one consecutive held step |
| Nominal actuator response hypothesis | First-order 30 ms, gain 0.97 |
| Random response hypotheses | 10–70 ms depending on joint; gain 0.90–1.00 |
| Residual scales | 0.06 abduction, 0.20 hip/knee, multiplied by 0.12/0.30/0.40 rad action scales |
| Nominal swing low-edge lift | 10 mm, smoothly phase-shaped; not verified floor clearance |
| New cost weights | Stance sole tilt 12, corner-clearance deficit 6, contact-point slip 4 |

The speed/cadence band is intentionally conservative. This stage addresses
foot placement and tracking first; it is **not** the promised final faster
running policy. Keeping V43's fast cadence while adding whole-foot lift would
request joint rates outside the new envelope. The generator has now executed
its analytical reachability, joint-limit and sampled-rate checks successfully;
those checks do not establish dynamic walking quality.

## Foot trajectory and rewards

`generate_flat_contact_reference.py` constructs two new 512-row references:
`references/v44_flat_contact_low.json` and `v44_flat_contact_high.json`.
Both outputs were generated on September 5 before the fresh training run.

During nominal stance the canonical hip and knee are equal and opposite,
making the sole level relative to the body. Front/rear neutral placements are
again approximately 80 mm ahead/behind their hips instead of preserving V43's
nearly-zero front offset. The old 22 mm uniform shortening is not carried over.

The fore/aft sweep uses a quintic curve. Swing solves for the **lowest fore/aft
edge**, accounting for rigid-shoe tilt, rather than just lifting its center.
The lift and reward use the same quintic enter/leave envelope, with a 0.5
transition fraction: position, velocity and acceleration join stance
continuously in the analytical trajectory. The generator rejects unreachable
poses, soft-limit violations and sampled reference rates above 90 degrees/s;
it does not silently project them. It refuses to overwrite existing tables.

Important geometry limitation: there is no independently actuated ankle.
Specifying two-dimensional foot position already determines sole angle.
Flat body-relative soles therefore require a varying support depth, and the
two diagonal support legs can request different depths. Body pitch and bounded
residual corrections must reconcile real contact. The generated 10 mm clearance
is relative to the local flat-support locus, **not** a guarantee against the
world floor. Table interpolation, body motion, actuator lag and compliance
can reduce it. Do not deploy this untrained reference directly as a gait.

Simulation now scores actual world-frame geometry for this opt-in task:

- Sole tilt during scheduled or force-detected stance, so lifting a badly
  pitched support foot does not evade the orientation cost.
- Clearance of the lowest of eight bottom corners across the structural sole
  and tread. The phase-shaped deficit is based on actual link pose, not targets.
- Tangential contact-point slip using link-origin velocity plus `omega × r`,
  scored on low corners of force-confirmed feet.

These metrics currently support a plane terrain only; other terrain is rejected
when the new costs are enabled. Per-leg episode logs include sole tilt, mean/min
core-swing corner height and contact-point slip. Existing straightness,
four-leg participation, acceleration, torque/speed and stop objectives remain.
Reference reward and speed pressure are reduced so the actor can favor contact
quality without being penalized for missing the old 0.115 m/s threshold.

## Executable limits and actuator uncertainty

The simulation's delivered target uses the same 2-degree cap as deployment,
in addition to elapsed-time rate limits. Held packets keep the delivered target
and its observation history unchanged; physics continues integrating. The next
packet can use the longer elapsed time but cannot exceed its packet cap.

A separate per-joint first-order response target feeds the simulated plant.
It does **not** replace the delivered-target observation history. State resets
to the actual spawn pose and evolves each physics substep. Both nominal and
randomized environments include the enabled response model. Existing artificial
command delay is disabled in V44 to avoid stacking it on this hypothesis.

The measured 310 ms cyclic motion lag is not interpreted as a 310 ms transport
delay. It includes load, tracking attenuation, phase distortion and existing
actuator/contact dynamics. The much smaller added response ranges are initial
uncertainty hypotheses, not calibrated servo identification. A future separately
authorized supported-motor measurement should refine them. Legacy tasks retain
disabled defaults for the new envelope/response/contact features.

## Preparation, training and export

Run from the repository root only after the corresponding work is explicitly
authorized. Reference generation and the fresh training command below have now
been launched; export remains a later step. No prior checkpoint is reused.

Generate the candidate reference files with ordinary project Python:

```powershell
& .\.venv\Scripts\python.exe .\simulation\isaac\rl\parallel_walking\generate_flat_contact_reference.py
```

Start a separate fresh Isaac Lab PPO run; this iteration count is an initial
training budget, not a promise of convergence:

```powershell
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet flat-contact-hardware-matched-run-stop-external-rear-payload `
  -Fresh -Iterations 1500 -NumEnvs 128 -Seed 1701
```

The V44 training entrypoint attempts to write hash-bound `model_N.training.json`
provenance after training returns. **Installed Isaac Sim fast shutdown can bypass
this hook even after successful training.** Do not assume provenance was written
or fabricate it; reliable automatic completion recording still needs a non-Kit
supervisor or pre-shutdown hook. The separate ablation launcher now preserves
pre-import preparation evidence, which is not production qualification. Keep the original
run's `params/env.yaml`, `params/agent.yaml`, references and provenance. The
provenance binds the actual saved environment/agent configuration, generator
metadata and table hashes to the checkpoint. Copying V43 into a V44 directory
does not qualify it for export.

After explicit simulation/ONNX qualification and selection of a real trained
checkpoint, export to a **new** package path:

```powershell
$v44Checkpoint = 'PATH_TO_THE_SELECTED_V44_RUN/model_N.pt'
$v44Output = 'onboard/models/parallel-running-v44-flat-contact-candidate/model_N.onnx'
& .\simulation\isaac\rl\parallel_walking\export_v44_policy.ps1 `
  -Checkpoint $v44Checkpoint -Output $v44Output
```

The exporter requires explicit paths, refuses existing output packages or a
V43 destination, and requires provenance plus matching training parameters and
references. If the checkpoint was copied, use `-TrainingProvenance` pointing
to its original run's `.training.json` beside `params/`. Isaac Python needs
PyYAML for safe inspection of selected YAML nodes; no Python-tagged objects
are constructed. Export metadata labels this an
`unqualified_corrective_stage_candidate`: successful training/export alone
does not certify real walking.

## Acceptance and remaining work

Before deployment, explicitly authorize and complete:

1. Focused controller/generator/contract tests and startup/stop lifecycle tests.
2. Nominal and randomized simulation at 0.03/0.04/0.05 m/s, including 20-second
   movement and graceful stop, with per-leg corner/contact diagnostics and
   video. Reject stalled, edge-supported or dragging solutions even if upright.
3. Export parity against the selected training checkpoint and exact contract.
4. Separately authorized Pi/controller and supported physical trials, comparing
   actual sent targets, encoder response and synchronized fixed-camera video.

Track p95 joint error and lag, whole-foot clearance, stance sole tilt,
contact-point slip, roll/pitch, heading and externally measured forward speed.
Do not qualify a run solely because reward is high or the robot did not fall.
Preserve the September 5 before-fix ZIP/video as the comparison baseline.

New focused test sources include
`onboard/policy-runtime/tests/test_background_imu.py`,
`onboard/policy-runtime/tests/test_applied_motor_history.py`, and
`simulation/tests/test_v44_flat_contact_reference.py`.
The dashboard test file also contains an authored coalesced-output packet-cap
case; it has not been executed.
Suggested unit-test command, **not executed**:

```powershell
& .\.venv\Scripts\python.exe -m pytest onboard/policy-runtime/tests/test_background_imu.py onboard/policy-runtime/tests/test_applied_motor_history.py simulation/tests/test_v44_flat_contact_reference.py
```

The initial source-implementation turn performed no execution checks. The user
subsequently authorized actual training and simulation evaluation; their outcomes
are recorded separately in the training record. Unrelated unit tests, linters,
hardware connections, Pi deployment, commits, pushes and hosted reviews remain
outside this training/evaluation turn.
