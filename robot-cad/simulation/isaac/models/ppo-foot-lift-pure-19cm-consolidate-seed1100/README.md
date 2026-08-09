# Pure-PPO 19 cm foot-lift consolidation checkpoint

This package contains iteration 260 from a continuation of the symmetric
supported-foot-lift curriculum. PPO directly controlled all 12 joints in 128
parallel Isaac Lab environments. No leg order, gait phase, inverse kinematics,
reference action, or scripted trajectory was supplied.

The continuation processed 138,240 transitions. Strict supported-lift success
remained intermittent, with a best logged reset batch of 50% at iteration 262,
maximum fork-tip clearance of 0.2400 m, and maximum supported lift hold of
0.1333 s. This is stronger repeated stochastic evidence than the preceding
single run, but it is not a converged deterministic policy and it does not
establish stair climbing.

The actor receives only IMU, VL53L5CX depth, joint position and velocity,
previous action, and four foot load/contact values. RGB and simulator ground
truth are excluded from the actor. The associated six-second RGB recording is
an honest deterministic evaluation: it shows an autonomous lift attempt,
collapse, and reset rather than a selected pass.

- `model.pt`: selected iteration-260 RSL-RL checkpoint
- `agent.yaml`: resolved PPO configuration
- `env.yaml`: resolved Isaac Lab environment configuration
- `training-report.json`: exact continuation and review summary

