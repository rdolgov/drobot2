# Tracked Isaac policy artifacts

This directory contains intentional evaluation dependencies and release
checkpoints. Bulk training products, TensorBoard logs, and checkpoints remain
ignored under `simulation/isaac/output/`.

| Directory | Purpose | SHA-256 |
| --- | --- | --- |
| `ppo-walk-v1-2m/` | Frozen flat-walking base used by the v5 residual stair environment | `eac4fd89066bf9483900188f8a5b8c047e848dfa2b3845f90c148967fdea56be` |
| `ppo-stairs-v5-10mm-four-step/` | Source-equivalent residual stair policy, current schema-2 manifest, packaging report, and 10-episode deterministic evaluation | `a61ebec6b02b366b48928cacf8aab70bba39cf660fc3a8f4ac0d39db0374fcfc` |
| `ppo-stairs-v6-180mm-25cm-small/` | Bounded full-size stair evaluation policy, schema-2 manifest, packaging/training/evaluation/recording reports; objective failed | `c29cb71ab596392c36292a57ca98473e7c4807a9de700bc7824bf2e0d73f91bc` |
| `ppo-stairs-v12-front-right-190mm-lift-small/` | Support-only mixed-height lift residual, manifest, training/evaluation/recording reports; strict 190 mm objective failed | `e0147f16ea942d751c8cec49616fa3aed62fc6b5f6a263aa0625600ee75e5f62` |
| `ppo-stairs-v13-front-right-190mm-lift-small/` | Direct front-right lift PPO residual, manifest, training/evaluation/recording reports; 3/3 strict 190 mm lift-hold successes | `b53a8d3d0087f816403461f328c758657d85ab6b19a08d79e951a8f4b85a9494` |

The stair checkpoint has one verified stochastic four-step `10 mm` climb but
scored `0/10` deterministic completions. It is an evaluation artifact, not a
hardware-deployment checkpoint. See `docs/rl-stairs-v5/README.md` for exact
commands, measured results, provenance, and limitations.

The v6 checkpoint uses four `180 mm` rises and fixed `250 mm` treads. Its
pipeline and contracts pass, but it completed `0/10` deterministic episodes
and reached at most stair one. It is also evaluation-only and must not be
deployed. See `docs/rl-stairs-v6-180mm/README.md`.

The v12 checkpoint was trained for `2,048` steps on one `250 mm` tread after a
verified left-foot placement and unloaded transfer. It preserved all support
contacts and stayed below the measurable-slip gate, but reached only
`142.08 mm` before exceeding the strict lateral corridor by `2.12 mm`. It is
an evaluation artifact, not a successful or hardware-deployment policy. See
`docs/rl-stairs-v12-lift-hold/README.md`.

The v13 checkpoint isolates front-right lift from a four-foot stance beside
the same fixed `250 mm` tread. Its `8,192`-step mastery run passed `28/28`
training episodes, and strict deterministic evaluation passed `3/3` with
`204.61-205.05 mm` lift and no measurable support slip. It is a successful
simulation prerequisite, not yet a tread-placement, ascent, or hardware
deployment policy. See `docs/rl-stairs-v13-direct-lift/README.md`.
