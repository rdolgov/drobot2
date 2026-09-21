# R4 checkpoint 1250: bounded controller comparison

All three diagnostics used the same checkpoint, seed, curriculum fraction
1250/1500, two starting diagonals, and nominal/85% motors. Only first episodes
count. Full training R4 continued unchanged throughout. Each report retains
its actual helper source, optional projection source, checkpoint and provenance.
`r4-model1250-comparison.json` summarizes stage outcomes and sampled clearance.

- **Unchanged controller:** all four trials timed out at lift after 4.85 s.
- **Swing projection:** requested planar forward positions stayed within
  +/-8 mm and down positions within -10/+2 mm of the nominal swing reference.
  All four passed the lift-position gate, three timed out at advance, one at
  settle. None earned a qualifying touchdown or complete pair. Passing a
  planar encoder gate does not establish actual floor clearance.
- **Projection plus support-compression compensation:** an encoder-only,
  150 ms filtered estimate of supporting-leg shortening raised the selected
  swing reference by up to 25 mm, blended with lift fraction. One trial still
  timed out at lift, two at advance and one at settle. Again no qualifying
  touchdowns or complete pairs. This added correction did not solve the
  loaded-clearance failure and is not adopted in training.

All joint, actuator, output-rate and acceleration limits were unchanged.
Contact-based success requirements and timeout handling were unchanged.
These candidates are diagnostic prototypes, not a deployable controller or
evidence that a new training run will succeed. Projection directly prevents
the observed cancellation, but support/load compensation and real clearance
remain unresolved. Do not report stage progression as gait progress or adopt
the added compression feedback merely because its encoder gate progresses.

The retained hypothesis for a future training adjustment is constrained swing
corrections together with monotonic per-foot clearance shaping, leaving the
support-leg policy free to learn load transfer. The compression prototype is
not supported as an additional improvement by this comparison.
