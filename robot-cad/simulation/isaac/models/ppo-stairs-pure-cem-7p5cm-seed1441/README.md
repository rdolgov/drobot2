# Pure stair 7.5 cm CEM bridge candidate (seed 1441)

This package preserves population 1 from the reward-ranked 7.5 cm bridge
search. It is an experimental authority checkpoint, not a promoted stair or
repeatable lift policy. From the literal 0.30 m fully folded, 90-degree
sideways reset it achieved one fresh strict 75 mm held-foot success in 308
completed episodes across two seeds. The source and other baked candidates
were zero on the same bridge screens.

The strict gate requires one unnamed foot to remain at least 75 mm above its
reset height for five post-settle control steps while the other three feet are
force-loaded. The actor still consumes exactly 70 deployable IMU, joint,
previous-action, foot-load, and compressed VL53L5CX depth values. There is no
selected leg, gait phase, IK target, reference motion, RGB input, or
simulator-only actor observation. Joint effort remains capped at 0.8825985 Nm.

The 128-environment, two-population search used 614,400 transitions and logged
15 strict successes among 5,603 completed episodes including automatic resets.
A population center passed once during search. Fresh screens were 0/165 and
1/143 for the packaged center. Two rendered 30-second batches each produced
one strict success in a non-filmed environment; the archived ordinary attempt
therefore must not be described as a visual success.

The next work should increase repeatability at 75 mm before returning to the
rare 100 mm gate, then proceed to 190 mm clearance, 250 mm tread placement,
weight transfer, and ascent.

