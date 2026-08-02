# Measured result

The constant-action search first established that the dynamic post-transfer
state is recoverable. The zero-action baseline tipped after `105/120` control
steps (`1.75 s`). One of 64 bounded candidates completed the full `120/120`
steps (`2.0 s`) without failure, with minimum upright cosine `0.948718` and
reward `+2560.95`. That candidate initialized the PPO actor mean:

```text
[-0.011061003, 0.035005786, 0.036190730, 0.120000000,
 -0.009826610, 0.003399614, -0.055805374, 0.017830297,
 -0.001008772]
```

The 4,096-step seed-840 PPO run passed in `159.24 s`. It completed 27 target
transfer holds, restored 28 cached snapshots with zero failed restores, and
saved model SHA-256
`4b45219e9e2d2d071c4373f152278e8e5fb67a9d00a5d74a469af875b3dffef8`.

The independent composed seed-832 run completed the front-right-to-front-left
and front-left-to-rear-right transfers, then remained upright for the full
65-second horizon with no termination reason. Maximum body tilt was
`16.946836 deg`, minimum base clearance was `0.342217 m`, and the rear-right
foot finished at `0.157804 m` versus `0.007756 m` initially: a physical lift
of `150.049 mm`. The 80-second extension also remained upright and held the
same pose.

## What this proves

- A small support-only residual can catch the specific dynamic state that
  tipped under zero residual.
- The real-test effort cap and exact `250 mm` tread geometry remain active.
- The policy is camera-blind; RGB appears only in the review video.

## What this does not prove

- It did not reach the requested `190 mm` rear-right lift in the composed run.
- The rear-right foot did not land on the tread, and the robot did not climb a
  stair.
- Fresh seed `841` tipped at `37.55 s` during the older precursor transfer,
  before this post-transfer catch policy became active.
- The composed seed-832 pose plateaued because rear-right hip-flexion tracking
  error reached about `1.324 rad` under the measured effort cap. The legacy
  composed-policy `maximum_foot_lift_m_by_leg` counter reports zero here, so
  the result uses the recorded initial/final physical foot-tip positions.

The next experiment should train support recovery jointly with the frozen V35
rear-right swing policy, rewarding both the 190 mm clearance gate and upright
hold before adding tread lowering/contact. More friction or RGB vision is not
the first change: the current evidence points to dynamic support and
torque-limited swing tracking.
