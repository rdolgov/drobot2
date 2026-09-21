# Lower-load reference diagnostics: rejected candidates

These isolated reference-only trials ran while R4 training continued without
source changes. Checkpoint 2250 was copied for provenance but **not used**;
all actions were zero. Both starting diagonals were exercised with nominal
and 85% motor strength/rate, full curriculum and 15 mm/s base command.

The candidate used front/rear stance offsets +30/-20 mm, 305 mm down,
15 mm pair-dependent lateral transfer, a correctly matched initial joint
pose, and the existing startup half-step and full-stride scheduler. Model,
torque limits, physical travel, command limiter and success criteria were
unchanged. References and traces are retained in the named folders:

- `load-stance-reference-v1`: all four trials terminated during lift at
  4.40–4.45 s with large body tilt. No complete steps.
- `load-stance-attitude-reference-v1`: nominal planar foot vectors were
  transformed using measured body pitch, with abduction roll compensation.
  Two trials terminated early; two hit the lift timeout. Large body tilt
  persisted. No complete steps.
- `load-stance-ankle-posture-v1`: instead of that attitude transformation,
  supporting ankles received a filtered pitch-restoring position offset.
  All four trials terminated during lift at 4.45–4.53 s. No complete steps.

These variants are rejected as training defaults. The earlier restricted
vertical-force static screen did not predict loaded transient behavior.
Do not adopt any of these reference changes or raise motor limits to make
them pass. The broad R4 stance remains the retained source.

The stronger supported correction for the next training attempt remains:
prevent selected swing corrections from erasing lift, give monotonic actual
clearance shaping to each selected foot (including the rear foot), and keep
strict full-pair landing criteria separate. Projection diagnostics already
showed improved scheduler progression, but not real paired clearance; the
reward and support policy must address that remaining failure. Use a new
run directory after R4 ends, and keep its evidence intact. No Pi changes.
