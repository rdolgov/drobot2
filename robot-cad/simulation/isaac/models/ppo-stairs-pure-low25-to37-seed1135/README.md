# Pure-PPO Low25-to-Low37.5 bridge checkpoint

This package contains iteration 776 from the correlated lower-reset bridge.
Each episode starts between 0.42 m / 25% fold and 0.40 m / 37.5% fold,
using one random interpolation value for both body height and all hip/knee
reset angles. The policy still controls all 12 joints on the exact 180 mm rise
by 250 mm tread without a gait clock, prescribed leg order, IK, or trajectory.

## Measured result

- Mixed bridge training: 7/2,199 strict stochastic events (0.3183%) over
  368,640 transitions. Checkpoint 776 was selected before late policy drift.
- Model 776 deterministic seed 1136: 3/251 (1.1952%).
- Model 776 deterministic seed 1139: 1/528 (0.1894%); the reset-bin audit
  found 0/255 in the easier half and 1/273 in the harder 31.25%-to-37.5% half.
- Pooled model 776 deterministic result: 4/779 (0.5135%).
- Final model 853 pooled deterministic result: 3/915 (0.3279%); rejected in
  favor of model 776. Its second seed also contained one harder-half success.
- Fixed 0.40 m / 37.5% continuation: 2/1,421 stochastic events (0.1407%) over
  245,760 transitions, followed by 0/300 deterministic; rejected.
- One-env RGB review seed 1138: 0/10. The video is an honest qualitative
  review, not a successful episode recording.
- SHA-256 of `model_776.pt`:
  `2c4d291a385fbad879c9101fbcbc9f35210b09c8033ae9f031fde7d14c8ee132`.

This proves rare supported tread success from the harder half of the lower
reset range, but not repeatable fixed-40-cm behavior, body transfer, a complete
step, sideways climbing, full fold, or a full stair climb.

Actor inputs remain hardware-representable IMU, VL53L5CX 8 x 8 depth, joint
position/velocity, previous action, and four foot load/contact channels. RGB is
review-only. Hip/knee authority remains 0.90/1.20 rad and every joint remains
capped at 0.8825985 N*m.
