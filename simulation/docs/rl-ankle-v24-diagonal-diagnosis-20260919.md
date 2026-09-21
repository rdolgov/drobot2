# Diagonal walking: timeout diagnosis and revision 2

September 19, 2026. Requested after the completed seed-2419 r3 training run
failed to produce qualifying diagonal steps. The initial revision-2 analysis
below used source/log inspection and independent IK arithmetic. The user then
explicitly authorized continuing local diagnostics and training without asking
again; the executed diagnostic follow-up is recorded at the end. No real robot
operation or deployment was performed.

## What the saved evidence establishes

Run: `simulation/reviews/ankle-v24-diagonal/20260919-seed2419-r3`.
All 3000 iteration reports recorded zero qualifying paired landings. Final
100 iteration-report means were:

| Metric | Mean of logged iteration values |
| --- | ---: |
| Gate timeout fraction | 89.95% |
| Qualifying paired landings / minimum qualifying steps on any leg | 0 / 0 |
| Forward velocity / command | -0.20 / 13.54 mm/s |
| Foot slip | 1.35 mm/s |
| Body angular velocity RMS | 0.09156 rad/s |
| Falls | 0% |

These are unweighted means of logged training summaries, not independent
episode-weighted evaluation estimates. Many episodes ended early at a gate;
zero falls does not establish a successful or stable walk. No per-step trace
or timeout-stage snapshots were saved by r3. `last_diagnostics` existed only
in process memory. The exact dominant joint, gate, or blocker cannot be
reconstructed from these logs.

The final checkpoint is
`logs/rsl_rl/drobot_ankle_v24_diagonal/2026-09-19_14-08-49_diagonal-2419-r3/model_2999.pt`,
SHA-256 `27ac34b78a3a7c82a50084b28d4b18c25e11a3f513cd7ebe31e49cbf3dc75d89`.
It remains unchanged. New training uses a separate output directory.

## Confirmed source problems and adjustments

1. **The gate and commanded pose had different definitions.** The controller
   requested `reference + filtered_action * residual_scale`, but the gate
   required measured joints within 5 degrees of `reference`. Available
   corrections were 5.73/10.31/12.61/8.02 degrees for abduction/hip/knee/ankle.
   Thus tracking an allowed corrected target could still fail the nominal
   joint-pose gate. This proves a contradictory condition is possible; the
   saved logs do not prove which episodes failed for that reason.

   Revision 2 compares joint tracking with the delivered target and separately
   requires the requested target to have reached delivery within 2 degrees.
   An independent encoder-based foot-path condition prevents a residual that
   cancels the lift or advance from passing solely because its command was
   tracked. Forward error must be below 15 mm; down-axis error below 8 mm at
   lift/advance and 10 mm at settle. Joint speed, IMU, 150 ms continuous ready
   time, and the 1.5-second additional-wait timeout remain. Advance now also
   has an endpoint gate before lowering.

   The foot-path condition uses the existing planar two-link FK of hip and
   knee. It is an ankle-pivot proxy, not world sole clearance or floor contact;
   it ignores body translation/roll. Simulator contacts and true sole geometry
   remain outside the controller and actor. No gate timeout forces progression.

2. **Height shaping arrived after the failing gate.** `mid_swing` meant stage
   2 (advance), while the first gate was at the end of stage 1 (lift). The
   path reward measured only forward position, which stays constant during
   lift. A policy failing the lift gate therefore never reached its lift-height
   reward. Revision 2 shapes height during lift and lowering too, includes
   vertical error in the path reward, and gives a bounded bonus only for
   improvements in the best supported pair lift achieved in that placement.
   Bobbing at the same height cannot repeatedly earn that progress bonus.

   A small, once-per-foot-per-placement qualifying touchdown reward bridges
   the sparse full-pair reward. It does not increment successful paired-step
   counts. Both feet must still meet the original real-flight, clearance,
   supported-flight, distance, lateral, and flat/quiet landing criteria for
   paired credit. Failed placements and extended waits now have explicit costs.

3. **The long-stride reference exceeded the ankle envelope.** For a front
   leg at the 100 mm stride lift endpoint, the old equations use
   `x = .070 - .050 - .014 = .006 m` and `down = .275 - .022 = .253 m`.
   With 159.896689 mm equal links, these imply hip 39.05 degrees, knee
   -75.37 degrees, and level-sole ankle 36.33 degrees. That exceeds the
   unchanged 33.25-degree soft ankle bound by 3.08 degrees. At 80 mm stride
   the same endpoint asks for 33.94 degrees. Reference clamping concealed
   this incompatibility, and no clipping metric was recorded.

   Revision 2 uses an 80 mm forward / 280 mm down stance and the corresponding
   368.908389 mm base-height target. The same maximum-stride endpoint asks
   for 32.52 degrees of ankle leveling. Endpoint arithmetic for front/rear,
   offsets -50/0/+50 mm, and heights 0/22 mm gave maximum absolute
   hip/knee/ankle angles 52.01/72.14/32.52 degrees. This is nominal endpoint
   arithmetic only: it does not establish trajectory rates, collision freedom,
   loaded tracking, or angular headroom under body pitch. Reference clipping
   is now recorded. Physical/model limits were not widened.

4. **The rate-limit metric measured any remaining target error.** The old
   `abs(delta - applied_change) > 1e-5` marked acceleration shaping and output
   holds as rate saturation. The reported approximately 96% cannot by itself
   prove the motors spent 96% of time at their speed cap. Revision 2 separately
   logs rate/packet-cap limiting, acceleration shaping, reference clipping,
   and held output cycles. The command limiter and its safety caps are retained.

## Diagnostics now retained

The revised environment records stage entries and wait times, timeout stage
and diagonal pair, and blockers observed during each timed-out wait: tracking,
delivery, forward position, height, joint speed, body tilt, and angular speed.
Moving-episode timeout fraction is separate from the mix that includes standing
episodes. Qualified single-foot touchdowns remain distinct from completed pairs.

`train.metrics.gate-totals.json` accumulates all completed episodes and is
flushed at worker shutdown. Initial resets are excluded. Reasons can overlap.
`train.metrics.gate-events.jsonl` saves up to two representative timeout
snapshots per 640 control steps, including joint order, targets, measured pose,
velocities, nominal reference, clipping, feet, forces, gate values, and ready
hold time. Gate values precede the physics tick; the pose/contact snapshot
follows that tick. These snapshots are sampled evidence, not an exhaustive trace.

## Bounded reference diagnostic

`simulation/isaac/rl/ankle_v24_diagonal/diagnose_reference.py` runs at most
48 simulated seconds by default, with four trials: both initial diagonals
with nominal motors, then both with 85% strength/rate. Actions are zero so it
isolates the reference, gates, actuator mechanics, and ankle leveling from
learned residuals. It writes stage transitions, periodic samples, terminal
snapshots, and first-episode outcomes. Auto-reset episodes never erase a failed
first trial. There is no learning, policy deployment, rendering, or hardware.

The user subsequently authorized execution, and this command was run:

```powershell
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v24_diagonal/diagnose_reference.py --output simulation/reviews/ankle-v24-diagonal/reference-v2 --seconds 48 --speed .015
```

## Executed diagnostic follow-up and revision 3

Each diagnostic retained four first-episode trials (both starting pairs, each
at 100% and 85% motor strength/rate), zero learned residuals, 15 mm/s command,
full 90 mm stride reference, and at most 48 simulated seconds. Reports and
traces are under `simulation/reviews/ankle-v24-diagonal/`:

| Directory | Reference | First-episode outcomes |
| --- | --- | --- |
| `reference-v2` | 80/280 mm stance, 14 mm shift, original startup | Four gate timeouts at 4.85–6.17 s, lift/advance |
| `reference-v3-upright` | 30/305 mm, 4 mm shift | Two falls, two timeouts by 4.85 s |
| `reference-v3-upright2` | 20/312 mm, 4 mm shift, radial reach cap | Four falls by 4.52 s |
| `reference-v3-startup` | Same upright stance, corrected startup | Four falls by 4.67 s |
| `reference-v3-wide-startup` | 80/280 mm, 14 mm shift, corrected startup | Four gate timeouts at 4.85–7.23 s; one reached settle |

All trials recorded **zero full qualifying paired landings**. Completion in a
diagnostic report means the diagnostic executed, not that the gait passed.
The upright alternatives were rejected; the retained stance is 80/280 mm.
The broad stance stayed upright in these bounded trials, but load tracking
and clearance remain unresolved. At the terminal snapshot, selected feet often
remained on the floor despite the 22 mm reference lift. True support-leg
compression consumes lift clearance; nominal endpoint IK alone missed this.

Startup also commanded the selected feet backward by half a stride while all
four feet were planted. Those closed-chain targets cannot all be followed
without slip or deformation. Revision 3 removes that preload: the initial
placement starts at the real neutral foot offsets and advances half a stride.
The subsequent all-foot push leads continuously into the ordinary full-stride
alternating sequence. A successful startup landing earns only its own once-per-
placement bonus, never a full paired-step count; subsequent criteria are unchanged.

The reference now explicitly retains its actual per-foot down target after a
317 mm radial reach cap. Foot-path gates and rewards use that same target.
The retained broad stance does not reach this cap in its nominal trajectory.
Torque limits, motor randomization, physical asset, joint limits, contact-based
success criteria, and timeout handling are unchanged. Effort and body-position
snapshots were added for load diagnosis.

The next run, `20260919-seed2419-r4`, deliberately remains experimental residual
learning, not a proven open-loop reference. It retains revision 2's lift-stage
height/path rewards, bounded progress milestones, command-consistent gates,
and blocker telemetry. The learner must still correct support/load tracking;
the diagnostics do not establish that it will succeed. Check actual clearance,
full paired landings, per-leg stride and gate blockers during training. Repeated
zero progress warrants adjustment, not deployment or reporting the model ready.

The latest authorization covers necessary local diagnosis, adjustments and
training in this continuing walking-policy work without repeated permission.
The existing ten-minute monitor is reused for the new run. The Pi remains on
its prior policy, and no hardware motion is authorized by this local run.
