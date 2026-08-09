# V88 frozen V84 support-hip-flexion transfer

V88 is a controlled negative result, not a stair-climbing policy. It expands
the V84 six-output transfer policy to nine outputs: all three front-left joints
plus hip abduction and hip flexion on each support leg. The V84 actor and its
six inherited output rows were frozen; only the three new support hip-flexion
rows and the value network trained. This isolates whether support hip flexion
alone crosses the learned load-transfer floor.

The simulator contract remains the exact `180 mm` rise by `250 mm` tread, with
the measured `0.8825985 N m` applied joint-effort cap. The 82 policy inputs are
camera-blind: IMU, joint/proprioceptive state, previous action, contact/load,
and known analytic stair geometry. RGB is recording-only.

After 8,192 PPO steps on seed 1034, V88 repeated the V84 result: two accepted
transfers at the 8 N curriculum gate and zero at 4 N. Minimum sampled
front-left load was `5.145 N`, minimum upright cosine was `0.975542`, and
maximum support slip was `34.951 mm`. The model SHA-256 is
`6a02ec152836c832147dc56dc696d9fe463becf8484988d7601abbce51fbf8cf`.

Fresh deterministic strict evaluation on seed 1041 failed `0/3`; every episode
exceeded the support-slip gate. Observed maximum support slip was
`35.010-42.692 mm`, maximum body tilt was `11.279 deg`, mean forward motion was
`-7.725 mm`, and mean elevation change was `-11.758 mm`. The recorded first
episode lifted the front-right foot `198.085 mm` but rear-left support slip
reached `42.692 mm` before front-left transfer. The video is diagnostic failure
evidence, not a successful climb.

This rules out more support-hip authority by itself as the next priority. The
next bounded experiment should expose support-knee extension and train an
explicit pre-unload/vertical-COM phase while keeping the inherited V84 actor
frozen. More camera vision or friction is still lower priority than measured
load sharing and upright force coordination.
