# Pure parallel stair PPO — lift-hold checkpoint

This package contains the first pure, 12-joint, 128-environment Isaac Lab PPO
experiment for the fixed `180 mm` rise by `250 mm` tread stair. It is an
evaluation checkpoint, not a release stair-climbing policy.

`model.pt` is iteration 600 from the lift-hold continuation and is the model
used for the 8-second deterministic review recording. During training, the
force/height tread metric reached two simultaneous valid tread contacts, but
the success rate remained zero and deterministic playback did not climb the
first step.

`model-nearest-first-two-contact-event.pt` is iteration 400, immediately after
the first logged two-contact event at iteration 399. The event was stochastic
training exploration and did not reproduce as a deterministic climb.

The actor and critic are `70 -> 256 -> 256 -> 12/1` ELU networks. The 70 actor
inputs are IMU angular velocity and projected gravity, 12 joint positions, 12
joint velocities, the previous 12 actions, four foot-load channels, and 24
compressed VL53L5CX depth values. RGB and simulator stair coordinates are not
policy inputs.

See `training-report.json` for hashes and measured results. The editable task
and reproduction commands are documented in
`docs/rl-stairs-pure-parallel/README.md`.

