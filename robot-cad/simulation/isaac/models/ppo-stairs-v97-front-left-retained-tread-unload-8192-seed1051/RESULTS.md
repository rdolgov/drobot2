# V97 retained-tread front-left unload

This package is a reproducible negative transfer result, not a stair-climbing
policy. It uses the real-test `0.8825985 N m` effort cap and exact `180 mm` rise
by `250 mm` tread geometry. RGB is excluded from policy observations.

The verified phase snapshot starts from a stable first-foot placement boundary.
V97 freezes the already-achieved analytic unload pose, enables full 12-joint PPO
residual authority, requires front-left unloading, and penalizes loss of the
already-placed front-right tread load.

Results from 8,192 PPO steps, seed 1051:

- zero completed transfer gates; curriculum remained at 20 N;
- best transient front-left load: `18.279501 N`;
- minimum front-right completed-tread load: `0 N`;
- minimum upright cosine: `0.984620`;
- maximum support slip: `2.666 mm`;
- minimum support margin: `103.692 mm`;
- model SHA-256:
  `e0e7dd3810782cb4cb6cd6e3148425d5d6cf5921386b78984d5bae45bef18fca`.

The package includes the policy, contract manifest, full training report, and
the verified seed-1047 phase snapshot plus capture report. The next experiment
should change stance/body-height geometry to retain front-right tread contact;
traction and camera vision are not the limiting variables in this stationary
test.
