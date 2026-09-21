# R6 checkpoint 750: rear feet still do not clear

Four deterministic first episodes used the matching revision-4 controller,
both starting diagonals, nominal/85% motors, a 15 mm/s base command and
curriculum 0.5. Effective command/stride were 80% of full values. The exact
checkpoint and diagnostic source are copied in this folder; checkpoint SHA-256:
`d7d7e5e5795f102aa64abb962e76cdfb79770726eb1520c514a5cb5b32117b83`.

All four passed lift and advance, but timed out at settle at 7.13–7.60 s.
There were zero full pairs and no physical falls before these early exits.
Front-right produced one qualifying touchdown in each pair-1 trial; no other
foot produced a qualifying touchdown. Front peak clearance was 22.5–32.9 mm.
Rear-right remained in contact; rear-left peak clearance was approximately
0.01 mm. This is no improvement toward actual diagonal clearance compared with
checkpoint 250. Settle blockers included joint tracking, fore/aft path error,
and sometimes height or speed; simply extending the wait would not establish
a real rear-foot step.

The offline support-plane audit uses recorded encoders, IMU gravity, exact URDF
FK and an assumption that both scheduled support feet are planted on one floor.
Its body-height estimate was within 1.5–3.3 mm of the measured simulator height
in four late-lift snapshots. Holding each recorded body/support pose fixed,
and flattening the hypothetical swing ankle, an additional 10.6–16.8 mm of
planar lift predicted the intended rear clearance. Rear-right hip soft limits
slightly constrained two of those static requests. This calculation ignores
the body's response to unloading and is not a feasibility proof.

`support-plane-audit.json` records the calculation, and copied audit/FK source
files preserve the method. A batched exact-FK implementation matched the
previous audited implementation to 2.8e-16 m on four recorded poses. Its
subsequent dynamic trial **failed**: see the adjacent
`r6-model750-support-plane-comparison.md`. Do not infer that a numerically
correct pose estimate makes this compensation dynamically stable.

R6 training remains unchanged. No Pi operation or hardware qualification.

## Follow-up support-load audit

`support-load-audit.json` examines the same recorded lift/advance samples,
without running another simulation. In each trial, at least one supporting
knee's reported torque stayed at or above 95% of the configured, strength-scaled
torque-speed cap in every sampled frame. Mean absolute tracking error for that
knee was approximately 9.0–12.9 degrees. Other supporting knees and ankles also
spent substantial portions of the samples near the cap.

This points to loaded support-joint saturation as a contributor to posture and
tracking failure. PhysX implicit-drive torque is approximate telemetry, and
these are short first-episode samples; they do not establish that a different
pose or learned support strategy is impossible, or measure real servo torque.
Future corrections should account for support load and posture before adding
more swing retraction. The physical actuator limits remain unchanged.
