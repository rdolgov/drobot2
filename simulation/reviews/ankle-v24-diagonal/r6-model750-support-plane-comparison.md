# Support-plane clearance compensation: rejected diagnostic prototype

This comparison ran September 20 UTC alongside R6, without changing its
training source or contract. Both conditions used checkpoint 750, curriculum
0.5, deterministic means, both diagonal starts and nominal/85% motors. The
prototype's source, URDF input, metadata and traces are retained in
`r6-model750-support-plane`; the full comparison is in the adjacent JSON file.

The prototype estimated floor-relative body height using exact URDF FK of
encoder positions, IMU gravity and the assumption that the scheduled supporting
soles remain on a horizontal floor. It used no simulator contacts, forces,
body position or link poses in the reference. It added up to 25 mm of filtered
swing lift and requested a flat swing ankle from the requested hip/knee pose.
Supporting ankle behavior, broad stance, physical limits, output limiter and
step qualification stayed unchanged. This policy was not trained with the
intervention, so the comparison tests immediate behavior only.

| Result across four first episodes | Matching baseline | Support-plane prototype |
| --- | --- | --- |
| End stage | Four settle timeouts | Four lift timeouts |
| End time | 7.13–7.60 s | 4.85 s |
| Full paired landings | 0 | 0 |
| Rear qualifying touchdowns | 0 | 0 |
| Peak rear-left sole clearance | About 0.01 mm | Below 0.01 mm |
| Rear-right clearance | Remained in contact | Remained in contact |
| Peak front sole clearance | 22.5–32.9 mm | 31.7–59.5 mm |

The correction saturated at nearly 25 mm on every selected rear leg. It
increased body tilt; pair-1 tilt sine reached 0.175–0.186 and violated the
unchanged 0.16 gate. Reference joint clipping appeared in 12–14 sampled swing
poses per trial. The weaker pair-0 body dropped as low as 0.329 m. The support
height estimator itself remained within roughly 2.0–4.1 mm of simulator body
height during sampled swing, so an accurate instantaneous estimate did not
prevent the body's response from cancelling the extra rear lift.

**Do not adopt this prototype or start training with it as-is.** Extra swing
retraction alone did not solve load support and instead aggravated pitch and
joint constraints. Preserve the evidence, continue the unchanged R6 run, and
use support/torque/posture behavior to guide any later bounded correction.
No motor-limit increase, success-criterion relaxation, asset edit or Pi change
was made.
