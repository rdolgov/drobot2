# Pure-PPO exact first-tread landing transfer

This package contains model-477, transferred from the consolidated 190 mm
supported-foot-lift policy and trained in 128 parallel simulations against an
exact 180 mm rise by 250 mm tread.

## Result

- Stochastic landing training: 24/5,495 episodes (0.4368%).
- Deterministic unseen seed 1108: 2/1,357 (0.1474%).
- Following 10 mm body-rise stage: 0/5,443.
- Fully folded 300 mm-start adaptation: 0/506.
- True 90-degree sideways adaptation: 0/2,888.
- Low-entropy landing consolidation: 27/5,481 stochastic (0.4926%) and
  2/1,336 deterministic (0.1497%); no material improvement over model 477.
- SHA-256 of `model_477.pt`:
  `3475f961ce0225a957c7fb64732bcdb4e2d858a3cdf371fcc006dda98868ebc2`.

The landing success requires a force-verified foot contact inside the tread
surface band while retaining support and uprightness. It is a rare first-contact
result, not a completed step or stair climb. The 10 mm bridge failed, so higher
body-rise stages were not attempted.

The actor input remains hardware-representable IMU, VL53L5CX depth, joint state,
previous action, and four foot load/contact channels. RGB and simulator pose are
not policy inputs. Joint effort remains capped at 0.8825985 N*m.
