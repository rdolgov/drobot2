# V92 unrestricted held-unload transfer

Status: **negative experiment; do not deploy as a stair policy**.

- Steps: 8,192, seed 1045
- Curriculum: 8, 6, 4, then 1 N; one continuous 0.50 s success per level
- Completed transfers: 0
- Minimum front-left load: 8.921 N
- Minimum upright cosine: 0.953835
- Maximum support slip: 36.247 mm
- Model SHA-256: `aac1a7b1bb80fe64fcecd670910425492134a07cdc906353abef3c554b2a04e9`

This run initialized all 12 transfer actions from V90 and fine-tuned them
together. It regressed the inherited behavior and did not clear the easiest
8 N gate. See `training_report.json` and
`simulation/isaac/rl/stairs/REAR_RIGHT_LANDING_SEARCH.md` for the full contract
and interpretation.
