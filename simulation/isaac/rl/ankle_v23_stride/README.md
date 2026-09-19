# V2.3 longer strides

Owning document: `simulation/docs/rl-ankle-v23-stride.md`.

`env.py` inherits actuator limits, scene construction and termination mechanics
from `ankle_v2`, and the target filter/rate limiter and reset logging from
`ankle_v21`. It supplies V2.3 contact geometry, a longer-step reference and a
new contact/slip/swing objective. These dependencies are snapshotted with runs.

`_train_worker.py` creates full PPO runs, optionally loading only an actor.
`train.py` serializes duplicate launches by metrics path and records completion.
`finish_training.py` waits for the two runs, selects six checkpoints, records
one attempt per seed, archives weights and writes `workflow.json`.

No hardware interfaces. Selection criteria are task-specific simulation gates,
not a safety qualification. Run commands are documented in the owning file.
