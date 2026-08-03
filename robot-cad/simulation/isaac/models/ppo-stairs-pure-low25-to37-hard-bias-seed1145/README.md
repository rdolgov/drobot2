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
