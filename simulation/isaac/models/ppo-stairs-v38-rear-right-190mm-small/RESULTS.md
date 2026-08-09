# V38 result

## Accepted

The seed-848 deterministic replay passed the complete model and PPO contracts,
finished the front-right-to-front-left and front-left-to-rear-right transfers,
and terminated successfully after rear-right held `217.319 mm` physical lift.
There was no fall or other failure reason. The final rear transfer had
`38.519 mm` support margin and `11.2504 deg` body tilt.

## Rejected during development

Two separate `1,024`-step transfer-residual PPO pilots were not promoted. The
seed-843 actor left only `3.74 N` of rear-right preload, and the lower-authority
seed-845 actor left `3.24 N`; both timed out at the strict pre-unload gate. A
zero-residual analytic target remained stable, so V38 keeps analytic transfer
and trains PPO only after that handoff.

The uninterrupted evaluator also exposed that the old global `5 N` rear-right
preload threshold blocked a physically stable `3.87 N` contact state. V38 uses
a rear-right-only `3 N` threshold held for `0.50 s`; the other legs retain the
global `5 N` preload threshold, and the post-unload requirement remains
`<= 1 N`.

## Next gate

Train only rear-right advance/lowering and require force-backed contact on the
first `250 mm` tread. Then search an independently positive support-margin
target for rear-left before learning residuals. Do not interpret this package
as a completed stair or four-stair climb.
