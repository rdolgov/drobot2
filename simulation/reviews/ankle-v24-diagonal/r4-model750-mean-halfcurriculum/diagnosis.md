# R4 checkpoint 750: deterministic lift failure

Executed during the September 19 training run (September 20 UTC). The running
R4 training source and contract were not changed. The separate helper is
`simulation/isaac/diagnostics/diagonal_checkpoint.py`; its exact source,
training provenance, checkpoint copy and hashes are retained in this folder.

Four first-episode trials used deterministic policy means, both starting
diagonals, nominal and 85% motor strength/rate, curriculum fraction 0.5 (the
checkpoint's curriculum), 15 mm/s base command, and a 48-second upper bound.
Every trial timed out at the first lift gate at 4.85 seconds. No qualifying
touchdowns or full paired steps occurred. These are diagnostic trials, not a
hardware qualification or a complete statistical evaluation.

This rules out stochastic exploration as the sole explanation for the lift
failure. Delivery errors were below the 2-degree gate in all four final
snapshots, but height, tracking or speed still prevented progression. See
`diagnostic-summary.json` for nominal, requested and measured planar foot-down
positions per selected leg, and `report.json` for complete terminal snapshots.

The policy often requests a pose that cancels most of the intended 22 mm lift.
At nominal-strength pair 0, both hip and knee residuals approach their allowed
range in the straightening direction. The resulting selected feet remain on
the floor. This is a commanded cancellation as well as a physical load-tracking
problem; waiting for quieter output alone will not correct it.

Source inspection identifies a likely reward/authority mismatch: selected
hip/knee residuals can negate the reference lift, while quiet contact and
support can still earn positive dense rewards. Height reward is Gaussian,
giving very little shaping near zero clearance, and the once-per-placement
lift bonus uses the minimum of the pair, so lifting only one foot gives no
pair progress. This is a supported hypothesis about learning, not proof of
which reward term caused the policy choice.

If R4 remains unsuccessful, the next bounded adjustment should constrain
swing residuals in foot coordinates so they can refine but cannot erase lift
and advance, while leaving support-leg corrections available. Compare actual
clearance under that constraint before another full run. Also consider
monotonic per-foot lift shaping with separately retained strict full-pair
success criteria. Keep physical torque, joint travel and contact qualification
unchanged; never make timeout release the next pair. Preserve R4 rather than
editing its running source or overwriting its checkpoints.
