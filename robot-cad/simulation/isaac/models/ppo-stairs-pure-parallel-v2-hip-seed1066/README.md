# Pure parallel stair PPO — hip-authority checkpoint

This is the best review checkpoint from the second pure-PPO stair round. The
task keeps the fixed 180 mm rise by 250 mm tread, 128 parallel environments,
the measured 0.8825985 N·m effort cap, and the deployable 70-value observation.

The round tested three unscripted reset conditions: a normal forward stance
with wider hip authority, a 300 mm-high nearly fully folded stance, and a true
90-degree sideways stance. The forward hip-authority variant was strongest.
It produced two simultaneous force/height-verified tread contacts during
stochastic training, but the full-climb success rate remained zero.

`model.pt` is iteration 260, the checkpoint used for the short deterministic
review video. `model-after-two-contact-event.pt` is iteration 280, immediately
after the two-contact event at iteration 272. Neither is a release stair
policy; deterministic playback approaches and articulates at the first riser
but does not climb it.

The actor uses IMU, joint state, prior action, foot load/contact, and compressed
VL53L5CX depth. RGB and simulator stair coordinates are not actor inputs.

See `training-report.json` and `docs/rl-stairs-pure-parallel/README.md`.
