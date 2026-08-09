# Pure-PPO supported 19 cm foot-lift checkpoint

This package contains the selected iteration-220 checkpoint from the final
stage of a symmetric `100 mm -> 140 mm -> 190 mm` foot-lift curriculum. PPO
controlled all 12 joints in 128 parallel Isaac Lab environments. There is no
scripted leg choice, gait phase, inverse kinematics, reference trajectory, or
action replay.

Success requires any one foot to remain at least `190 mm` above its local
terrain for eight 30 Hz control steps while at least three feet remain in
force-verified support and the body stays upright. The final stage processed
460,800 transitions. Its best logged reset batch had a 50% success rate at
iteration 149; success was intermittent rather than converged. Maximum logged
clearance was 0.3126 m.

The actor input remains deployable: IMU, VL53L5CX depth, joint position and
velocity, previous action, and four foot load/contact values. RGB and simulator
ground truth are not actor inputs. The tracked deterministic model-220 video is
an evaluation sample, not proof of robust 19 cm success or stair climbing.

- `model.pt`: selected iteration-220 RSL-RL checkpoint
- `agent.yaml`: resolved PPO configuration
- `env.yaml`: resolved Isaac Lab environment configuration
- `training-report.json`: exact staged training and review summary

