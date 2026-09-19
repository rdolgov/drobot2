# V2.3 rear-foot unloading and weight transfer

Owning document: `simulation/docs/rl-ankle-v23-transfer.md`.

This is a new PPO task using the existing solid-sole V2.3 USD asset. It preserves
the previous stride task and its completed evidence. It inherits only the V2
actuator/scene mechanics and V2.1 target filtering and reset logging.

The reference has a four-foot transfer interval before each swing, a small
body translation through hip targets, and actual-joint ankle leveling. Forces
and whole-robot COM appear only in reward/critic diagnostics. They do not gate
the phase or actuators. Per-leg supported steps, unloading, and clearance are
reported so front-foot success cannot hide rear-foot failure.

The unchanged 8-11 cm stance sweep uses a slower 1.5-2.1 cm/s crawl. The actor
still has 82 inputs; the critic has 101. Seed 2311 initializes only the prior
V2.3 best actor, with a fresh critic and optimizer; seed 2312 starts fresh.
Each run has 256 environments and 3000 iterations of 64 transitions per env.

`train.py` locks each metrics path and records completion; `finish_training.py`
waits for both runs, selects six checkpoints and records the best available
attempt from each seed. A failed gate is preserved and reported. No hardware
deployment is part of this task. The owning document contains launch commands.
