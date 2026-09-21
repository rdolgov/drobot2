# Full-curriculum checkpoint 1500 and a taller broad stance

Four deterministic first-episode trials per condition ran September 20 UTC,
with both starting pairs, nominal/85% motors, 15 mm/s command and full stride
curriculum. The candidate kept the existing 80 mm fore/aft footprint and 14 mm
body shift, increasing only nominal downward reach from 280 to 290 mm and
target body height by 10 mm. Inherited initial-pose configuration uses those
values, so the candidate starts at its own matched stance. Exact sources,
checkpoint and metadata are saved in each diagnostic directory.

The change was motivated by supporting-knee torque saturation in earlier
traces. `broad-stance-load-screen.json` retains the restricted vertical-force
static calculation: minimum peak torque at the initial lifted pose decreases
from approximately 0.824 to 0.744 Nm. The latter leaves very little margin
against the 0.750 Nm weak-motor cap and does not include dynamic behavior or
contact-force constraints beyond vertical nonnegative forces.

| Four first episodes | Original 280 mm stance | Candidate 290 mm stance |
| --- | --- | --- |
| End stages | Two settle, two lift timeouts | Three settle, one lift timeout |
| End times | 4.85–10.20 s | 7.12–11.63 s |
| Full paired landings | 0 | 0 |
| Rear qualifying touchdowns | 0 | 0 |
| Peak selected rear-foot clearance | Under 0.5 mm | Under 2 mm |
| Falls before timeout | 0 | 0 |

Both conditions sometimes advanced the scheduler into the other pair without
a qualified physical paired landing. That progression is not a successful
step. The comparison JSON groups clearance only while each leg is actually
selected, accounting for pair changes within an episode.

The candidate improved some front touchdowns and produced small rear
clearance, but did not meet the unchanged 8 mm clearance or full-step criteria.
It is **not adopted for training or deployment**. The actor was trained on the
original stance; this result does not prove retraining on another posture can
never work, but the static torque screen alone is insufficient reason to start
another long run. All actuator limits, physical joint limits, reward, output
limiter and success criteria were retained. R6 source and training continue
unchanged. No Pi operation occurred.
