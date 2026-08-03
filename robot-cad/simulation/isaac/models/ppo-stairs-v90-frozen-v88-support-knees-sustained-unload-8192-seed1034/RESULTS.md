# V90 frozen support-knee sustained-unload transfer

V90 is a controlled negative result, not a stair-climbing policy. It expands
V88 from nine to twelve actions while freezing the inherited actor and action
rows 0-8. Only the three support-knee rows 9-11 and the value network train.
The reward includes vertical balance-target progress plus persistent costs for
active swing-foot load and vertical balance error.

The physical contract remains the exact `180 mm` rise by `250 mm` tread and
the measured `0.8825985 N m` applied effort cap. The 82 policy inputs are
camera-blind: IMU, joint/proprioceptive state, prior action, contact/load, and
known analytic stair geometry. RGB is recording-only.

After 8,192 PPO steps on seed 1034, V90 still completed only the two inherited
8 N transfers and none at 4 N. Minimum sampled front-left load was unchanged at
`5.145 N`; minimum upright cosine was `0.975542` and maximum support slip was
`34.951 mm`. The persistent state-cost contribution was `-405,039.928`, proving
the new objective was active, but it did not alter the physical load floor.

Fresh strict evaluation on seed 1042 failed `0/3`, all by support slip. Slip
was `35.093-38.372 mm`, maximum tilt was `11.029 deg`, front-left lift was only
`4.668-5.828 mm`, mean forward displacement was `-12.703 mm`, and mean
elevation change was `-10.836 mm`. The recorded first episode reaches
`198.213 mm` front-right lift but `38.372 mm` support slip before second-foot
transfer. It is diagnostic failure evidence, not a climb.

The model SHA-256 is
`c78b50a516fd013a1f22da1ecfaca61dfc59f596d0d33c8558cb4e43983ec4d5`.
This result rules out missing support-joint authority as the main limitation of
the current direct-transfer formulation. The next experiment should separate a
stationary pre-unload hold from XY transfer: require front-left load below 4 N
for at least 0.5 s with all three supports loaded and bounded slip, then freeze
that checkpoint and compose the proven V80 190 mm lift primitive.
