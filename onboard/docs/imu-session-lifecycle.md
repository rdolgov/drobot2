# BNO085 initialization and cleanup — 2026-09-05

## Findings

Hardcoded crawl does not require the IMU. RL requires accelerometer, gyro, and
game-rotation-vector reports before it can start. The reported feature `1`
timeout means that the driver's accelerometer report did not arrive; it does
not identify the underlying electrical or protocol cause.

The Pi was still running service PID 1063 when the error recurred. The earlier
startup patch was on disk but had not been activated by a service restart.
Software I2C bus 8 and its GPIO 2/3 configuration were present. The dashboard
reported all 12 motors holding torque, with crawl and RL inactive. A service
restart would disarm the robot, so activation requires the user to support and
disarm it first.

The previous runtime retained the IMU object after a run, including after a
sensor read failure. Idle report streams were not stopped, initialization could
overlap HTTP requests, and a failed post-initialization stance check could leave
an unused sensor session open.

## Changes

- The BNO085 source has one process-local owner. A competing source cannot reset
  an existing session. This lock does not exclude unrelated external processes.
- Initialization makes at most two fresh bus/driver attempts. Each fresh driver
  performs its own reset and identification. A missing-first-report timeout can
  retry once in that session. A failed session closes its bus before reopening.
- Startup requires three complete, finite samples. This checks sample contents;
  it does not prove report freshness or the achieved sample frequency.
- Errors identify the failed startup stage, bus/address, and original cause.
  They remain visible if recovery fails. Sensor failure does not start RL.
- Stop cancels pending preparation. Overlapping preparation and preparation
  during a running policy are rejected. The browser disables duplicate starts
  immediately and displays initialization status. Client version is now 11.
- Failed preparation, rejected stance checks, normal completion, faults,
  explicit stop, and idle model switches release the sensor session.
- Cleanup uses the driver's public soft reset to stop reporting, then closes
  I2C in a `finally` path. A cleanup error is logged and does not retain ownership.
  Motor finalization happens first, so sensor reset does not delay the existing
  center hold or disarm behavior. Cleanup does not reset the sensor mid-walk.

Two attempts bound the number of retries, not total wall time: the underlying
driver and Linux I2C calls can each block. A sensor latched electrically may
still require a full power cycle. No stale/synthetic sample fallback was added.

## Deployment and activation

The changed source/UI files are uploaded to `/home/rd/drobot2`. The previous
files are preserved under
`/home/rd/drobot-deployments/v43-20260905/before-imu-lifecycle.tar.gz`.
The running service was left untouched while the motors held torque.

Activation completed on 2026-09-05 at approximately 11:15 local time after the
dashboard reported all 12 motors disarmed and no active gait. The service was
restarted successfully as PID 3193 with `active/running`, zero restarts, and
exit status zero. This activates the lifecycle changes; successful IMU startup
still requires a subsequent operator attempt.

Once the robot is supported and disarmed, activate with:

```bash
sudo systemctl restart drobot-manual-web.service
```

Reload the dashboard. The updated page displays `INITIALIZING IMU…` while Start
is pending. If both initialization sessions fail, preserve the new complete
error and service log for diagnosis rather than repeatedly pressing Start.

## Verification boundary

Source inspection and existing service logs/status were used. No automated
tests, compilers, UI validation, IMU probes, policy inference, or motion checks
were run, following the user's verification restrictions. The physical startup
failure is not claimed resolved until a subsequent authorized attempt succeeds.

## Background acquisition and completed-output history (local follow-up)

The recorded V43 trial `20260905T193441588801Z-2a2a9832` subsequently completed.
Its steady policy loop averaged 48.72 Hz: observation acquisition took 11.80 ms
on average and 20.60 ms at p95, compared with 0.234 ms mean policy inference.
The following changes are local source changes, not yet deployed or verified.

- `BackgroundImuSource` runs source construction, sensor reads, reset, and I2C
  close on one owner thread. The policy only copies its newest complete sample;
  it never waits on a device read. Requested acquisition cadence is 100 Hz,
  with missed acquisition slots skipped instead of replayed.
- Existing BNO085 fresh-session initialization and three-complete-sample warmup
  remain inside that owner. Dashboard preparation then waits for a complete
  cached sample before allowing the motor policy to start. Startup is cancellable
  and the caller stops waiting after 30 seconds.
- Each cached sample preserves its acquisition timestamp. Reads reject samples
  older than 120 ms, invalid/future/backwards timestamps, incomplete vectors,
  and non-finite values. Acquisition failures propagate immediately, even if an
  older cached sample has not reached the age limit. Cached vectors are copied
  so a consumer cannot mutate a later observation.
- Sensor reset/close stays on the owner thread. Caller cleanup waits at most
  0.5 seconds for it; a blocked Linux/driver call cannot be forcibly interrupted
  safely. If the call is still pending, the session retains ownership and blocks
  reinitialization/model switching until that owner finishes cleanup. The
  dashboard status includes `imu_cleanup_pending`.
- Motor finalization still happens before sensor cleanup. Existing graceful
  phase-aligned stopping/center hold and the 2-degree target guard remain.
- The motor sink can now expose `read_applied_target()` returning
  `AppliedMotorTarget(joint_target_rad, monotonic_time_s, sequence)`. The real
  dashboard updates this snapshot only after the bus write completes and its
  joint targets are committed. It is read under a short separate target lock,
  never under the shared servo-bus lock.
- `applied_target` actor history uses that completed snapshot, or the armed
  initial pose before its first write. Enqueued/replaced targets do not become
  applied history. Existing sinks without acknowledgements keep their previous
  synchronous behavior. The producer's rate limiter also starts from the
  acknowledged pose rather than accumulating unsent target increments.
  Its elapsed request budget is the larger of the actual production interval
  and time since the acknowledged write. This avoids subtracting the normal
  queue delay twice; it does not predict when a future bus write will occur.
- The motor worker independently applies the selected model's **per-joint
  velocity limit multiplied by actual elapsed time since the last completed
  packet**, plus its packet-step limit (2° for V43/V44). A queue replacement
  cannot authorize a larger catch-up broadcast. This override only applies to
  active, authorized RL motor output; manual/stance/final-center motion keeps
  its existing behavior.
- Recorded samples include the actual snapshot used for history in
  `applied_motor_target` and the limiter's base in `target_limit_base_rad`.
  `control_timing.target_elapsed_s` is explicitly labeled as the producer's
  request budget, not a measured bus interval. Trial metadata records acquisition mode, requested
  rate, age limit, and timestamp semantics. A host-completed broadcast is not
  proof that the servo has physically reached its command.

Freshness here means **complete host acquisition/cache age**. The existing
Adafruit property interface does not expose a hardware acquisition timestamp
for each combined acceleration/gyro/quaternion report. These fields can still
arrive asynchronously; this change does not claim hardware-synchronized sensor
channels or infer report freshness just because their numeric values changed.

Focused regression tests were added for owner-thread lifecycle, cached reads
during a blocked device call, freshness/error propagation, cancellation and
timeout cleanup, and acknowledged action history. They were **not run** under
the user's current verification restrictions. No tests, hardware checks,
inference, simulation, training, or deployment were performed for this update.
Additional authored cases cover held-output target accumulation, producer
budget timing, and actual 2-degree/per-joint bus limits after queue coalescing.

### Operator-requested start after activation

The user subsequently requested running the robot. The dashboard reported V43,
the prepared stance ready, no active gait, and no motor fault. One Start request
was submitted for `0.10 m/s` and `3 seconds`. It returned HTTP 400 after both
fresh initialization sessions failed to enable the accelerometer on bus 8 at
`0x4A`. Policy execution did not begin. The software recovery changes are active
but have not resolved this sensor's report-start failure. A physical sensor
power cycle and, if necessary, wiring/power diagnosis remain outstanding.
