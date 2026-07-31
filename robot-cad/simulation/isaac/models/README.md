# Tracked Isaac policy artifacts

This directory contains intentional evaluation dependencies and release
checkpoints. Bulk training products, TensorBoard logs, and checkpoints remain
ignored under `simulation/isaac/output/`.

| Directory | Purpose | SHA-256 |
| --- | --- | --- |
| `ppo-walk-v1-2m/` | Frozen flat-walking base used by the v5 residual stair environment | `eac4fd89066bf9483900188f8a5b8c047e848dfa2b3845f90c148967fdea56be` |
| `ppo-stairs-v5-10mm-four-step/` | Source-equivalent residual stair policy, current schema-2 manifest, packaging report, and 10-episode deterministic evaluation | `a61ebec6b02b366b48928cacf8aab70bba39cf660fc3a8f4ac0d39db0374fcfc` |
| `ppo-stairs-v6-180mm-25cm-small/` | Bounded full-size stair evaluation policy, schema-2 manifest, packaging/training/evaluation/recording reports; objective failed | `c29cb71ab596392c36292a57ca98473e7c4807a9de700bc7824bf2e0d73f91bc` |

The stair checkpoint has one verified stochastic four-step `10 mm` climb but
scored `0/10` deterministic completions. It is an evaluation artifact, not a
hardware-deployment checkpoint. See `docs/rl-stairs-v5/README.md` for exact
commands, measured results, provenance, and limitations.

The v6 checkpoint uses four `180 mm` rises and fixed `250 mm` treads. Its
pipeline and contracts pass, but it completed `0/10` deterministic episodes
and reached at most stair one. It is also evaluation-only and must not be
deployed. See `docs/rl-stairs-v6-180mm/README.md`.
