# Pure stair foot-lift CEM candidate (seed 1341)

This package contains the strongest robust deterministic policy from the
reward-ranked episode-bias search. It is a partial stair result, not a climbing
policy: from a literal 0.30 m fully folded, 90-degree sideways reset it can
occasionally hold one unnamed foot at least 50 mm above its reset height while
three other feet remain force-loaded. It has not placed a foot on the 250 mm
tread and has not climbed the 180 mm riser.

The actor still consumes exactly 70 deployable values: IMU, 12 joint positions,
12 joint velocities, the previous 12 actions, four foot loads, and a compressed
24-value VL53L5CX depth observation. RGB is used only for review. No selected
leg, gait phase, IK target, reference motion, or simulator-only actor input is
present. Joint effort remains capped at 0.8825985 Nm.

`model.pt` is population 0 after a 20-generation, 128-environment,
two-population winner-centered CEM refinement with configured 0.01 rad joint
and 0.01 m lateral reset randomization. Four fresh seeded screens scored
19/583 strict successes (3.26%). The paired population scored 18/568 (3.17%).
With reset variability increased to 0.02 rad joint noise and 0.015 m lateral
jitter, three later screens scored 34/509 (6.68%). A conservative PPO
continuation and a local CEM refinement both regressed against this source, so
the packaged checkpoint remains unchanged.
The accepted 30-second third-person rollout is
`reviews/ppo-stairs-robust-cem-pop0-env10-seed1351-30s.mp4`; telemetry confirms
that filmed environment 10 crossed the strict gate.

The next promotion gate remains a repeatable 100 mm held lift, followed by
190 mm clearance, tread placement, weight transfer, and only then ascent.
