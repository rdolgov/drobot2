# R6 completed, gait failed

The run completed all 3,000 iterations and 49,152,000 transitions. All recorded
iteration reports had zero full paired landings. Completed episodes contained
1,598 qualified front-left and 8,203 front-right touchdowns, but zero rear-left
or rear-right qualified touchdowns. Every one of the 72,656 completed moving
episodes ended in a lift, advance or settle timeout. Source and asset hashes
still match the recorded contract. See `result-summary.json` for exact counts.

The final checkpoint's independent default-controller evaluation failed all
four first-episode trials: both starting pairs, nominal and 85% motors, full
curriculum. Three timed out during advance around six seconds; the nominal
FL+RR trial timed out during settle at 12.17 seconds. None completed a full
paired landing. No trial reached a meaningful stop/settle evaluation. Absence
of falls before these early timeouts is not sustained walking qualification.

Every actor action in all 322 final diagnostic trace samples was exactly zero.
The reference controller still commands motion; zero here means the learned
residual correction disappeared, not that the motors received zero targets.

## Late optimizer collapse

The installed policy uses Beta concentrations `softplus(raw) + 1` in float32.
When both raw outputs become sufficiently negative, both concentrations round
to one. That produces uniform sampled actions on [-1,1], zero deterministic
mean and vanishing gradients through the raw outputs. The final training
logger reports zero entropy loss and surrogate loss, and action standard
deviation about 0.577, consistent with this failure.

`optimizer-collapse-audit.json` records saved checkpoint values and TensorBoard
learning-rate history. Late adaptive learning rate reached 0.01, 33.3 times the
configured initial 0.0003. At checkpoint 2999, 99.85% of individual concentration
parameters hit the exact floor across a zero input and 256 seeded synthetic
normalized-input probes. Those probes are not actual rollout observations.
The final diagnostic trace separately establishes zero mean actions on actual
observations. The checkpoint weights and normalization statistics did update;
the earlier normalizer audit remains valid.

These findings identify a late learning failure. They do not establish that
learning rate alone caused it, or explain away the earlier rear-foot support
and clearance failures. Supporting knees were already near simulated torque
caps with large tracking errors before this collapse.

## Bounded next experiment

`../20260920-optimizer-pilot-r7` restores checkpoint 1500's actor, critic and
normalizers, starts fresh Adam at fixed 0.0001, and uses 250 iterations at full
curriculum. The environment, rewards, physics, controller and physical limits
are unchanged. Per-iteration diagnostics record the fraction of actual final
update-minibatch action distributions with both concentrations equal to one.
The pilot stops if that fraction stays at least 95% for ten iterations.

An independent evaluation with the matching default controller must follow.
The pilot is an optimizer recovery experiment, not a qualified walking policy.
No Pi connection, motor motion, model deployment or physical-limit change was
performed.
