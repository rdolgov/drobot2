# V93 frozen-base held-unload residual

Status: **negative experiment; do not deploy as a stair policy**.

- Steps: 8,192, seed 1046
- Frozen base: V90 sustained-unload controller
- Trainable correction: zero-initialized 12-joint residual, scale 0.10
- Curriculum: 6, 4, then 1 N; one continuous 0.50 s success per level
- Completed transfers: 0
- Minimum front-left load: 8.792 N
- Minimum upright cosine: 0.953237
- Maximum pitch magnitude: 9.708 degrees
- Maximum support slip: 35.217 mm
- Model SHA-256: `d7a8b499fae8a71d9ab82898cd0991f82dc53a93a4f1027ab717a6b81887b099`

The bounded residual preserved the frozen-controller structure but did not
produce a held 6 N unload. See `training_report.json` and
`simulation/isaac/rl/stairs/REAR_RIGHT_LANDING_SEARCH.md` for the full contract
and interpretation. V91 remains the verified 190 mm raise-and-hold policy.
