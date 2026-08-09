# Pure-PPO width105 four-support first-tread landing

This package contains model 635, initialized from broad four-support model 556
and trained in 128 parallel Isaac Lab simulations against the exact 180 mm rise
by 250 mm tread. The width curriculum accepts a 210 mm band centered on the
first tread, excluding the outer 20 mm at each end.

## Result

- Stochastic PPO: 7/7,322 episodes (0.0956%) over 245,760 transitions.
- Deterministic unseen noisy-sensor seed 1123: 3/1,248 (0.2404%).
- Following +/-90 mm stage: 1/6,983 stochastic and 0/1,153 deterministic.
- Following contact-gated 10 mm body-rise stage: 0/5,379.
- SHA-256 of `model_635.pt`:
  `aba06edff4188061c31d9c235784592a3bfffe81b9d2e53b60d2916baf232db8`.

Success requires one foot to remain force-verified inside the +/-105 mm tread
band for three policy steps while all four feet are simultaneously supported
and the body is upright. This is a rare first-tread landing, not a body transfer,
completed step, or stair climb.

PPO controls all 12 joints without a gait phase, prescribed leg order, inverse
kinematics, or action trajectory. The actor receives only hardware-representable
IMU, VL53L5CX depth, joint position/velocity, previous action, and four foot
load/contact channels. RGB is review-only. Joint effort remains capped at
0.8825985 N*m.
