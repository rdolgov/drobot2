# Migration record

## Source consolidation

Source repository:

`C:\Users\roman\Documents\dev\fork\text-to-cad`

All local branches were consolidated into a dedicated `develop` worktree at:

`C:\Users\roman\Documents\dev\fork\text-to-cad-develop-merge`

The consolidation retained the canonical symlink layout and excluded
materialized Windows checkout duplicates.

- `5ef2303` - merged the robot CAD and orthographic markup feature histories
  into `develop`
- `01b5c37` - recorded `main` release history without importing its
  publish-only deletion of development models

Every local branch (`codex/cad-viewer-markup`,
`codex/so101-st3215-models`, `codex/so101-upper-arm`,
`codex/upper-arm-shared-motor-bay`, `develop`, `main`, and `wip`) is an ancestor
of the resulting local `develop`.

The consolidated branch is local and was not pushed. At migration time it was
31 commits ahead of `upstream/develop`.

## Migrated model sources

- `simple_drobot/components/st3215_motor_bay/ST3215_rear_motor_bay.py`
  became `drobot_cad/parts/st3215_motor_bay.py`.
- `simple_drobot/components/so101_upper_arm/Upper_arm_SO101.py`
  became `drobot_cad/parts/upper_arm.py`.
- Both labeled fit-preview generators moved under `drobot_cad/assembly/`.
- The exact ST3215 catalog STEP and immutable original SO-101 upper-arm STEP
  moved under `vendor/`.
- The original upper-arm change-request markup sheet moved to
  `reviews/legacy/`.

Only repository-relative paths and Python package imports were adapted. The
fit-critical cavity construction and upper-arm geometry parameters were
preserved.

## Geometry equivalence

Fresh source and migrated outputs were generated with the same CAD runtime.
For both the motor bay and upper arm:

- bounding boxes were identical;
- solid volumes were identical;
- the bidirectional boolean-difference volume was zero.

The upper-arm STEP serialization reported six additional edges in the migrated
artifact, but no face-count, bounding-box, volume, or solid-set difference.
Selector references must therefore be resolved from the new generated artifact
rather than copied from the source repository.

Repeated in-memory OCCT boolean evaluation can move a tolerant boundary by up
to about 0.031 mm and volume by less than 0.001%. Automated upper-arm geometry
baselines use a 0.05 mm / 0.001% tolerance while exported
source-versus-migration comparison remains a zero-volume solid-set difference.
