# RL real-walk recordings

The hardware dashboard automatically starts a recording whenever an RL walk
starts. Recording does not open the servo bus and does not perform additional
IMU or encoder reads. It observes the same synchronized values already used by
the 60 Hz policy loop and places them on a bounded in-memory queue. A separate
writer thread performs all JSON and filesystem work.

If storage cannot keep up, the control loop drops recording items rather than
waiting. `metadata.json` reports `dropped_samples` and `dropped_events`; do not
use an incomplete trial for timing-sensitive analysis without accounting for
those gaps. A recorder failure is shown in the RL recorder status but does not
alter motor-control behavior.

## Storage and dashboard controls

The default directory on the Raspberry Pi is:

```text
~/.local/share/drobot2/rl-recordings
```

Override it when starting the dashboard with:

```bash
drobot-four-leg-web --recordings-dir /path/to/recordings ...
```

Open **Settings → Trial recordings** on port 8080 to refresh the list, assign a
human-readable name, download a ZIP, or delete a finalized trial. A trial is
listed as `FINALIZING` briefly while the background writer drains its queue and
builds the archive. Folder IDs never change when a display name is edited.

Each trial folder and ZIP contains:

- `metadata.json`: schema version, result, timestamps, command, model and
  hardware-file hashes, sensor configuration, counts, and drop statistics;
- `samples.jsonl`: one object for each successfully commanded policy step;
- `events.jsonl`: start/finalization events and staggered full motor diagnostics.

New recordings store the measured 3.175 kg total robot mass plus the V20
simulation payload estimate: a 523.18 g battery/holder centered at
`[-0.1315, 0, 0.05] m` in the body frame. The payload number combines the
measured 416 g battery with CAD-derived box/lid mass; fasteners, foam, and wire
remain uncertain. Keep that distinction when fitting simulation dynamics.

## `samples.jsonl`

Every line contains:

- sequence number, policy elapsed time, and monotonic timestamps for the policy,
  IMU sample, and joint sample;
- commanded forward/lateral/yaw velocity, the effective yaw input seen by the
  actor, and gait-clock sine/cosine;
- body-frame angular velocity, projected gravity, and linear acceleration;
- BNO085 game-world yaw plus relative-heading reference, desired heading,
  error, bounded correction, and whether model metadata enabled heading hold;
- 12 measured joint positions and derived joint velocities in policy order;
- the exact 50-value observation passed to ONNX and the 12-value policy action;
- the action-derived requested target and the velocity-limited target sent to
  the dashboard motor-target sink;
- actual elapsed time used by target limiting, cumulative missed deadlines,
  and the model-declared gait frequency; and
- feedback transport in diagnostic events (`group_sync_read` or sequential
  fallback).

The semantic joint and physical servo-ID order is stored in `metadata.json`.
The target is the control-loop target, not proof that the servo reached it;
compare it with the subsequent measured joint position. New controller builds
send one synchronous group-write packet for the 12 targets, while only one full
motor status is sampled per diagnostic period, so `events.jsonl` remains much
lower rate than `samples.jsonl`.

## Initial uses

Start with internal data only. It can answer whether falls correlate with body
tilt/acceleration, joint tracking lag, voltage sag, current, a particular leg,
or policy action spikes. It is also suitable for:

1. simulation-to-real comparison using identical observation and action fields;
2. identifying latency, noise, actuator-rate, and voltage-dependent parameter
   ranges for Isaac domain randomization;
3. replaying observations through a candidate model without moving hardware;
4. training a residual or system-identification model after enough clean trials.

Do not directly use a fallen or faulted real-world trajectory as a walking
demonstration. Keep its result label and use it for failure classification or
dynamics identification. Prefer improving simulation randomization and policy
evaluation before any real-world fine-tuning.

Video is optional at this stage. Internal signals are enough to implement the
pipeline and diagnose many control/power issues. Later, synchronized video is
valuable for labeling foot contact, slip, actual travel, and fall direction—
quantities the current hardware does not sense. A simple future convention is
to film continuously and show the recording ID in the frame or name the video
with that ID.

## ROS 2 migration

The policy loop publishes complete step samples to a small recorder interface;
the JSONL implementation only consumes that interface. A future ROS 2 adapter
can publish the same sample fields plus IMU, joint state, target, and diagnostic
topics, while a `rosbag2` implementation replaces the JSONL writer. Keep the
servo-bus owner and its safety checks in the existing hardware process. The
recorder must remain a passive consumer and must never become a motor-command
path.
