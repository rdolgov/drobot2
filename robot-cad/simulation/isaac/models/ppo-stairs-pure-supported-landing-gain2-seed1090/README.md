# Pure-PPO supported first-tread landing and 2 cm transfer

This package preserves iteration 411 of the pure-RL curriculum for the exact
`180 mm` rise and `250 mm` tread. Training used 128 parallel Isaac simulations.
The actor receives only 70 deployable IMU, VL53L5CX depth, joint position and
velocity, previous-action, and foot-load/contact values. RGB is evaluation only.

The curriculum is phase-free and symmetric: no leg is selected, and there is no
scripted gait, inverse kinematics, pose target, or privileged actor input. A
narrow landing-surface reward first learned a supported three-step tread hold;
that checkpoint was then transferred to a 20 mm body-rise objective.

The 2 cm transfer processed 245,760 transitions. Its best logged reset batch
averaged 0.50 force-verified tread contacts. Maximum logged supported tread hold
was 0.1333 s, maximum body gain was 0.0116 m, and success remained 0% because
contact and the required 0.020 m body gain were not combined in one episode.
This is meaningful landing progress, not a completed stair step.

