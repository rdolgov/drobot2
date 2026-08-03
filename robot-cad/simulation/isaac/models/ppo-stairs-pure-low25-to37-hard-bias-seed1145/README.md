# Pure-PPO Low25-to-Low37.5 hard-biased checkpoint

This package contains iteration 895 from a 128-environment continuation of
the correlated 0.42 m / 25% fold to 0.40 m / 37.5% fold reset bridge. Reset
interpolation uses `sqrt(U)`, so 75% of starts fall in the harder half of the
range. The policy still controls all 12 joints on the exact 180 mm rise by
250 mm tread without a gait clock, prescribed leg order, IK, or trajectory.

## Measured result

- Hard-biased continuation from model 776: 368,640 transitions and 3/2,039
  strict stochastic successes (0.1471%). The reset audit measured 505 easier
  episodes with 1 success and 1,534 harder episodes with 2 successes, so
  75.23% of completed episodes came from the intended harder half.
- Matched deterministic seed 1146: source model 776 scored 1/563 overall and
  1/417 hard-half; selected model 895 scored 2/522 overall and 2/383
  hard-half.
- Matched deterministic seed 1147: source model 776 scored 0/527 overall and
  0/393 hard-half; selected model 895 scored 3/461 overall, 2/349 hard-half,
  and 1/112 easier-half.
- Pooled deterministic comparison: model 895 scored 5/983 (0.5086%) overall
  and 4/732 (0.5464%) in the hard half, versus model 776 at 1/1,090 (0.0917%)
  overall and 1/810 (0.1235%) in the hard half.
- Early model 796 scored 0/577 on matched seed 1146 and was rejected.
- A second 128-environment continuation from model 895 added another 368,640
  transitions and produced 5/2,013 stochastic successes (0.2484%): 1/483 in
  the easier half and 4/1,530 in the harder half. On deterministic seed 1150,
  retained model 895 scored 2/494, while event-rich intermediate models 938
  and 987 scored 1/534 and 0/494, and final model 1014 scored 0/473. All three
  continuation candidates were rejected as mean-policy drift.
- One-environment RGB review seed 1148 scored 0/11. It is qualitative review
  only; RGB is never an actor input or a training signal.
- SHA-256 of `model_895.pt`:
  `bcc6e8d431539fbfd5e5849d579a1e11645a30510b06580efc5b7825d7bdcf57`.

The measured gain is real across two matched populations but remains sparse.
It proves improved rare supported tread success from lower starts, not a
repeatable step, body transfer, sideways climbing, full fold, or a full stair
climb.

Actor inputs remain hardware-representable IMU, VL53L5CX 8 x 8 depth, joint
position/velocity, previous action, and four foot load/contact channels. Hip
and knee action authority remains 0.90/1.20 rad, and every joint remains
capped at 0.8825985 N*m.

## Supported body-rise audit

Five additional 128-environment pure-PPO runs added 1,843,200 transitions
while testing progressively simpler outcome gates. None prescribed a joint
pose, action trajectory, gait phase, leg order, or inverse-kinematics target.

- Direct centered-tread plus four-support transfer: 0/1,948 successes.
- Four-support 10 mm stand-up precursor: 0/2,327 successes.
- Three-support 10 mm precursor: 0/2,357 successes; the maximum gated body
  rise was 1.27977 mm.
- Ungated upright plus 10 mm continuation from model 895: 244/2,599
  stochastic rise events before the final simultaneous-failure exclusion,
  with a 60.48828 mm maximum. Event-rich model 990 scored 18/167 on the first
  deterministic comparison versus 29/176 for retained model 895, so it was
  rejected.
- Fresh ungated upright plus 10 mm PPO: 135/2,801 stochastic events before
  the same final exclusion, with a 58.55331 mm maximum. It also remained below
  the retained policy and was rejected.

The final success definition explicitly requires the rise hold and `not
failed` on the same step. Under that stricter gate, retained model 895 scored
17/169 deterministic episodes on seed 1160, including 15/135 from the harder
reset half. This proves limited body-extension authority under the real effort
cap, not supported transfer or stair climbing.

Third-person seed-1176 review clips are tracked at
`reviews/ppo-upright-rise-retained-model895-seed1176.mp4` and
`reviews/ppo-upright-rise-fresh-model119-seed1176.mp4`. Both are honest
ordinary failed attempts (0/5 and 0/1 respectively), not selected successes.
