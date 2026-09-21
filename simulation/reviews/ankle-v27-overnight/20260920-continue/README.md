# Overnight diagonal training results

Selected for review: **model_2549**. Qualified selection trials: 4/4.
The user-approved checkpoint is retained. No Pi deployment or physical motion occurred.

[Comparison video](comparison.mp4) · [First diagonal](video-pair-0/walk.mp4) · [Other starting diagonal](video-pair-1/walk.mp4)

[Phone review, same local URL](http://192.168.1.54:8765/7c89c5858ecf176b03a14b81/)

The four selection cases traveled **49.1-51.6 cm** in 32 seconds, with
**1.36-4.99 cm** sideways drift and **2.37-2.56 degrees RMS** body tilt
(maximum 6.47 degrees). Every leg completed **five full steps** in every case.
Both selected swing feet cleared the ground at **100% of 10 Hz advance-stage
samples**; per-leg peak clearance ranged from **38.2 to 48.7 mm**.

The separate full recordings use another seed and nominal motors. The first
traveled 48.65 cm with 2.12 cm drift and passed; complete steps FL/RL/FR/RR
were 5/4/4/5. The second traveled 48.90 cm with 5.72 cm drift and **failed the
unchanged 5 cm drift limit**, despite five complete steps on every leg.
Both recordings had no falls and 100% sampled paired advance clearance.
Lateral drift remains the main observed weakness. The weaker-motor second
start in the selection batch passed the drift limit by only 0.13 mm.

These are simulation results. The actor export and matching-controller
requirements are documented in the [hardware handoff](hardware-handoff.md).
The matching Pi controller is not yet implemented, so the candidate is not
ready for upload into the installed V23 policy slot.

| Candidate | Qualified trials | Falls | Complete pair events | Both-clear fractions, four trials |
|---|---:|---:|---:|---|
| model_2549 | 4/4 | 0 | 40 | 100.0%, 100.0%, 100.0%, 100.0% |
| model_1400 | 3/4 | 0 | 35 | 95.7%, 100.0%, 95.7%, 85.1% |
| model_2200 | 2/4 | 0 | 39 | 100.0%, 100.0%, 100.0%, 100.0% |
| model_1800 | 2/4 | 0 | 31 | 100.0%, 100.0%, 100.0%, 70.2% |
| model_1000 | 0/4 | 0 | 7 | 95.7%, 91.5%, 95.7%, 89.4% |
| model_600 | 0/4 | 0 | 2 | 19.1%, 6.4%, 8.5%, 0.0% |
| user-approved starting candidate | 0/4 | 0 | 1 | 8.5%, 4.3%, 6.4%, 0.0% |

All selection trials use the first 32-second episode, both starting diagonals,
and nominal/85% motors. Standing, dragging and body translation do not count as full steps.
Complete walking needs at least two full steps per leg, acceptable drift and body tilt,
and no falls. Ranking never changes these criteria.

The exported actor is the Gaussian mean network, not the deployed V23 Beta policy.
See [hardware handoff](hardware-handoff.md) before attempting a physical trial.
