# Drobot V22 live-transfer support audit

V22 extends the accepted camera-blind single-foot 190 mm lift into a continuous
front-right placement, inter-leg transfer, and front-left lift. Every run used
the four-tread Isaac scene with exact 250 mm tread depth, 180 mm rise, floating
base, gravity, collisions, and the real-test 0.8825985 N*m effort cap.

The accepted V17 result remains unchanged: 5/5 deterministic isolated-foot
trials reached 205.55-207.75 mm lift with at most 3.27 mm support slip and 2.33
degrees body tilt. Its review video is
[`reviews/ppo-stairs-v17-single-foot-190mm-lift-success.mp4`](../../reviews/ppo-stairs-v17-single-foot-190mm-lift-success.mp4).

## V22 implementation

- Added per-joint implicit PD demand, capped demand, tracking error, explicit
  actuation input, and projected reaction-load telemetry. This prevents the
  zero explicit-force channel used by position drives from being mislabeled as
  delivered motor torque.
- Reset the runtime IMU between teleported Gym episodes so filter/history state
  does not leak into later policy rollouts.
- Added `search_stairs_transfer_pose.py` for reproducible timing, stance,
  friction, effort-cap, and feedback searches from either a cached phase or the
  full continuous sequence.
- Added a V22 configuration with a 3.0 second left-foot lift and a searched 50
  mm extension of the already-placed front-right support reference.
- Added `--phase-disable-snapshot-cache` so phase PPO can replay the real
  precursor transfer on every reset instead of training on one favorable
  restored state.

## Findings

The original transferred baseline requested up to 17.27 N*m from its implicit
PD drive, 19.56 times the 0.8825985 N*m cap, and spent 36.86% of joint samples
at or above 95% of the cap. Position-drive demand is a controller diagnostic,
not a claim that the capped servo delivered that torque.

The searched pose showed that the second lift is physically possible. A full
continuous seed reached 221.0 mm with 13.5 mm slip and 9.7 degrees tilt. In the
cached sensitivity matrix, the 40 mm precursor pose passed at static/dynamic
friction pairs 0.90/0.75, 1.05/0.90, and 1.20/1.00 when the real-test torque cap
was retained. It failed at 0.82 N*m and 0.75 N*m, so this phase is torque- and
posture-sensitive rather than traction-limited.

The repeated end-to-end gate remains below acceptance:

| Controller | Strict successes | Successful lift | Successful slip | Successful tilt |
| --- | ---: | ---: | ---: | ---: |
| Frozen V10 + V17, before IMU reset | 2/5 | 219.9 mm | 13.4-15.2 mm | 10.7-11.1 deg |
| Frozen V10 + V17, IMU reset | 3/5 | 218.8-221.0 mm | 13.4-15.2 mm | 9.6-11.1 deg |
| 4,096-step live-prefix support-residual PPO | 3/5 | 217.7-219.3 mm | 14.2-15.4 mm | 11.0-11.1 deg |

The unsuccessful trials did not fall or exceed the slip threshold. They
remained stable but stalled at 166-174 mm, so neither the fixed support pose nor
the small PPO run is promoted as a completed two-foot skill. The trained
residual is retained as audit evidence, not as the default policy.

## Next training step

Train the transfer and support controller with explicit per-foot normal-load,
COM-target error, and torque-saturation observations. Randomize the live
post-transfer state, effort cap near 0.8826 N*m, and friction near measured
rubber-pad values. Keep the V17 swing policy frozen until the support controller
passes at least 5/5 fresh continuous episodes, then add tread contact and the
rear-foot phases. Vision remains unnecessary for this fixed known stair; add
depth/ToF or camera input later for unknown riser localization and landing.
