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

## Two-support 5 mm rise bridge

Three further 128-environment pure-PPO stages added 1,064,960 transitions.
The strict outcome remained unchanged: hold at least 5 mm body rise for four
steps with at least two supporting feet, remain upright, and do not fail.

- A binary two-support reward gate produced 5/2,561 stochastic successes. Its
  maximum two-support rise sample was 51.80928 mm, but event checkpoints 935,
  963, 999, and final 1014 each scored 0 deterministic successes on seed 1179;
  retained model 895 scored 1/144 on the same population.
- A soft dense reward gave zero, one, and two supporting feet 0%, 50%, and
  100% rise credit while leaving the strict success gate at two supports. It
  produced 11/2,513 stochastic successes, all in the harder reset half.
  Models 924 and 1003 reproduced one strict success on seed 1181 (1/170 and
  1/150), although both scored zero in the seed-1182 256-environment tie-break.
- A 64-step, zero-entropy, 2e-5 learning-rate consolidation from model 1003
  added 327,680 transitions and 6/2,180 stochastic successes. Selected model
  1016 scored 1/145 on deterministic seed 1185, with its success in the hard
  lower-reset half; source model 1003 scored 0/146. Models 1021 and 1042
  regressed to zero and were rejected.

The experimental bridge checkpoint is
`model_two_support_rise5_consolidated_1016.pt`, SHA-256
`c34ce4dcca7c4659467231c45c95dc95dc875402ea0074da90b1bba7c2c0a4f2`.
It is not promoted over retained model 895 as the safety baseline: its gain is
rare and not robust across the 256-environment tie-break. Third-person clips
`reviews/ppo-two-support-rise5-soft-model1003-seed1183.mp4` and
`reviews/ppo-two-support-rise5-consolidated-model1016-seed1186.mp4` scored 0/5
and 0/6. They are ordinary periodic attempts, not selected successes.

## Full-fold and sideways hip-leverage audit

The next pure-PPO round tested whether a lower start and more lateral hip
authority make the 5 mm, two-support rise precursor easier. The actor inputs,
12-joint action space, exact 180 mm rise by 250 mm tread, and 0.8825985 N*m
effort cap stayed unchanged.

- The full-fold reset uses the hardware-representable 100% folded pose at a
  0.30 m base height. A 60-step settling window and 12-step success hold prevent
  spawn depenetration from being counted as learned body rise. The earlier
  ungated measurement of 5,411/5,422 was therefore rejected as an artifact.
- Under the corrected gate, the zero-action control scored 17/148 (11.49%) and
  source model 1016 scored 23/143 (16.08%) on seed 1187.
- Full-fold continuation added 368,640 transitions and 272/2,260 stochastic
  successes. Selected model 1030 pooled 163/774 deterministic successes
  (21.06%) on seeds 1189 and 1190 versus 140/854 (16.39%) for source model
  1016: +4.67 percentage points, or about 28.5% relative improvement.
- `model_full_fold_rise5_1030.pt` is that selected checkpoint. SHA-256:
  `46d6f0c000531dc4d6e22ca4f16d06b080d9510903e42d38295739e9099d61f9`.
- The ordinary seed-1191 third-person review scored 0/4. This is intentionally
  not a searched success clip.
- A true 90-degree sideways task remounted the same simulated VL53L5CX toward
  the stair and expanded hip-abduction action scale from 0.30 to 0.42 rad.
  Source model 1030 scored 1/265 during the baseline population. The 368,640
  transition continuation found 7/4,165 stochastic successes, but source and
  event checkpoints 1037, 1070, and 1093 all scored 0 on held-out seed 1197.
  The sideways continuation was rejected rather than packaged.
- The ordinary lateral seed-1198 third-person review scored 0/6. Its SHA-256
  is `f0a8f7120d4a62e09d365fc0ee40d913836d9b6a967e31c146ff97eb92194b46`.

The lower full-fold start is a verified improvement for this small supported
rise precursor. Direct 90-degree lateral transfer is not. The next justified
curriculum is a gradual 0-to-45-to-90-degree yaw bridge while keeping the
single depth sensor physically aimed at the stair. Neither result demonstrates
foot placement, a complete 180 mm step, or a stair climb.
