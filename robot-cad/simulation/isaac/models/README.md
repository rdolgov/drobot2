# Tracked Isaac policy artifacts

This directory contains intentional evaluation dependencies and release
checkpoints. Bulk training products, TensorBoard logs, and checkpoints remain
ignored under `simulation/isaac/output/`.

| Directory | Purpose | SHA-256 |
| --- | --- | --- |
| `ppo-walk-v1-2m/` | Frozen flat-walking base used by the v5 residual stair environment | `eac4fd89066bf9483900188f8a5b8c047e848dfa2b3845f90c148967fdea56be` |
| `parallel-walking-v15/` | Selected stable 128-environment RSL-RL transfer checkpoint for the corrected implicit servo task | `995dc00e4603da91386f80976fcec3da9df0b50cff73798a056c86deade0a887` |
| `ppo-stairs-v5-10mm-four-step/` | Source-equivalent residual stair policy, current schema-2 manifest, packaging report, and 10-episode deterministic evaluation | `a61ebec6b02b366b48928cacf8aab70bba39cf660fc3a8f4ac0d39db0374fcfc` |
| `ppo-stairs-v6-180mm-25cm-small/` | Bounded full-size stair evaluation policy, schema-2 manifest, packaging/training/evaluation/recording reports; objective failed | `c29cb71ab596392c36292a57ca98473e7c4807a9de700bc7824bf2e0d73f91bc` |
| `ppo-stairs-v12-front-right-190mm-lift-small/` | Support-only mixed-height lift residual, manifest, training/evaluation/recording reports; strict 190 mm objective failed | `e0147f16ea942d751c8cec49616fa3aed62fc6b5f6a263aa0625600ee75e5f62` |
| `ppo-stairs-v13-front-right-190mm-lift-small/` | Direct front-right lift PPO residual, manifest, training/evaluation/recording reports; 3/3 strict 190 mm lift-hold successes | `b53a8d3d0087f816403461f328c758657d85ab6b19a08d79e951a8f4b85a9494` |
| `ppo-foot-lift-v3-rear-right-190mm-small/` | Fresh 512-step rear-right unsupported-balance PPO, manifest, 5/5 strict evaluation, and recording report | `376d1e02e09ea9d5d25eacdc38bf46432c8ed9bf6b3efc65cccdbec6214e636b` |
| `ppo-foot-lift-v3-rear-right-190mm-seed941-small/` | Independent fresh 512-step rear-right no-fall rerun, manifest, 5/5 strict evaluation, and recording report | `7f3ccb0a159140de47eb99d8ad71c0eeabf3692a6dd712e36c44206c4e9d279c` |
| `ppo-foot-lift-pure-19cm-consolidate-seed1100/` | Pure-PPO 128-environment supported-lift continuation, repeated intermittent strict success, resolved configs, and deterministic review metadata | `1b5aa01b1c36c7c3a283a9149c35b88b3db0f81eaf34fbcc71f8c9b3b94c7162` |
| `ppo-foot-lift-pure-19cm-continue-seed1102/` | Second pure-PPO continuation, two 100% stochastic success batches, final checkpoint, resolved configs, and honest unseen-seed review metadata | `13a8e70ae3ff77ed4119f06f34303d512fd2c6045d94f0e83ef6d0a416c00676` |
| `ppo-stairs-v45-rear-left-dynamic-transfer-4096/` | Exact-snapshot 4,096-step rear-left dynamic-transfer diagnostic, deterministic failed replay, and controller-authority evidence; objective failed | `c97cb2d5c4f1f7cb1cf70c011026b36033c4e71317c82e416f6d4529a48ab5c0` |
| `ppo-stairs-v46-rear-right-sidestep-transfer-4096/` | Accepted post-landing rear-right sidestep snapshot plus rejected 4,096-step rear-left transfer PPO and deterministic failure recording | `9b8cf6dec9f2da8f43dab9fb65f72b4a32999340d43356cf748c3a8498164977` |

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

The foot-lift V3 checkpoint isolates the rear-right leg on flat ground with
three physical support feet and no torso pose pinning. Five fresh deterministic
episodes all passed the strict `190 mm`/`0.75 s` gate with `199.85-203.66 mm`
maximum lift and at most `2.12 deg` tilt. This proves clearance and balance in
the simple pose, not the mixed-height stair transfer, landing, or ascent.

The independent seed-941 V3 rerun passed `5/5` fresh deterministic episodes
at `201.006-204.345 mm` lift and at most `2.218 deg` tilt. Its separately
recorded seed reached `202.907 mm` without falling. It reproduces the raw
clearance conclusion under the same real-test effort cap; it does not add
stair-transfer evidence.

The V45 checkpoint starts after the accepted V44 rear-right tread landing and
controls all 12 joint residuals during the rear-right-to-rear-left transfer.
Its 4,096-step seed-875 run completed zero transfers, and deterministic replay
tipped after 3.583 s. It is retained to make the rejected controller direction
reproducible; it is not a stair-climbing or hardware-deployment checkpoint.

V46 physically widened the rear-right foothold by `9.215 mm` while preserving
`65.201 mm` replacement support margin, but its new transfer still started
about `102.8 mm` outside the requested support geometry. The seed-939 policy
completed zero transfers and deterministic seed 940 tipped after 93 steps.
This rejects sidestep-only control; an all-four-feet COM settle/preload phase is
the next gate.
