# Pure-PPO 19 cm foot-lift continuation checkpoint

This package contains iteration 359 from a second 128-environment continuation
of the symmetric supported-foot-lift task. PPO controls all 12 joints directly;
there is no prescribed foot, gait phase, inverse kinematics, action replay, or
reference trajectory.

The run processed 307,200 transitions. It sampled strict 100% success reset
batches at iterations 260 and 355, reached 0.3891 m maximum fork-tip clearance,
and held the supported lift for the full 0.2667 s gate. Other batches—including
the final one—remained at 0%, so this is strong stochastic exploration evidence,
not a converged policy. Deterministic unseen-seed playback still folds its legs
and resets rather than showing a clean supported lift.

The actor input remains deployable: IMU, VL53L5CX depth, joint position and
velocity, previous action, and four foot load/contact values. RGB and simulator
ground truth are excluded from the actor.

- `model.pt`: final iteration-359 RSL-RL checkpoint
- `agent.yaml`: resolved PPO configuration
- `env.yaml`: resolved Isaac Lab environment configuration
- `training-report.json`: exact continuation and review summary

