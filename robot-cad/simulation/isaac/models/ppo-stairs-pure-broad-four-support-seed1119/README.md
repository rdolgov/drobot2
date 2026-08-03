# Pure-PPO broad four-support first-tread landing

This package contains model 556, initialized from exact-landing model 477 and
trained in 128 parallel Isaac Lab simulations against the exact 180 mm rise by
250 mm tread. PPO controls all 12 joints directly without a gait phase,
prescribed leg order, inverse kinematics, or action trajectory.

## Result

- Stochastic PPO: 23/7,283 episodes (0.3158%) over 245,760 transitions.
- Deterministic unseen noisy-sensor seed 1120: 3/1,245 (0.2410%).
- Centered-contact transfer: 0/5,433 over another 184,320 transitions.
- SHA-256 of `model_556.pt`:
  `94d13d2fe7472e72d5f6ab1123664f1a0dec1f9b3f26084940082cc5f89424e9`.

Success requires one foot to remain force-verified on the first tread for three
policy steps while all four feet are simultaneously supported and the body is
upright. Contact may occur anywhere in the tread surface band; this is not the
stricter centered-contact gate. It is a rare first-tread landing result, not a
body transfer, completed step, or stair climb.

The actor receives only hardware-representable IMU, VL53L5CX depth, joint
position/velocity, previous action, and four foot load/contact channels. RGB is
used only for review video. Simulator pose and stair coordinates are used only
for reward and evaluation. Joint effort remains capped at 0.8825985 N*m.
