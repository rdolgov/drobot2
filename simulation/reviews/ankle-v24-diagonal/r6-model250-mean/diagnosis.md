# R6 checkpoint 250: commanded lift survives, rear clearance remains inadequate

Four deterministic first-episode diagnostics ran on September 20 UTC while
R6 training continued unchanged. They used the default matching revision-4
controller, both starting diagonals, nominal and 85% strength/rate, a 15 mm/s
base command and curriculum fraction 250/1500. At this curriculum the effective
command and stride are two-thirds of their full values. The duration bound was
48 seconds. The copied checkpoint SHA-256 is
`2c3e3f8d4f07fb1328bf17df2e6b28621e500a8105a26b5f8c34e594259b326b`.

All four passed the lift gate. Three timed out at settle, and the weaker
front-left/rear-right trial timed out at advance, at 6.25–8.03 seconds.
There were no physical falls, qualifying touchdowns or full paired landings.
These short gate-terminated trials do not establish sustained balance or a
successful stop. The final instant of a gate timeout may show all checks clear
without having held them clear for the required continuous 0.15 seconds.

The approximately 10 Hz trace shows front-foot peak sole clearance of
25.2–32.2 mm during lift/advance. Rear-left peaks were 3.8 mm at nominal strength
and 0.8 mm with weaker motors. Rear-right remained in contact, with measured
sole minima slightly below zero from simulated contact penetration. None of
the selected rear feet reached the required 8 mm minimum clearance.

At the last sampled lift instant, requested planar downward distances were
258.0–260.0 mm against a 258.0 mm reference. The previous 19–20 mm commanded
lift cancellation is absent. Measured rear downward distances were
259.8–263.2 mm; this body-relative tracking is not world sole clearance.
Relative to the pre-lift sample around 2.4 seconds, the selected rear hip moved
down approximately 9.3–16.1 mm. Body pitch was about -2.1 to -3.5 degrees,
and rear sole orientation also reduced corner clearance in the weaker pair-1
trial. These measured changes explain why a body-relative lift can pass while
the rear sole remains near the floor. They do not prove a particular reward
change or controller intervention would solve the support problem.

`diagnostic-summary.json` retains the per-trial clearance, force, planar
request/tracking and root-height summaries; `trace.jsonl` and `report.json`
retain measured geometry, gates, actions and terminal states. No reference
overlay, physical-limit change, training-source edit or Pi operation occurred.

R6 is only at an early checkpoint. Continue the existing run and compare later
rear clearance, supported flight, actual touchdowns and full paired steps.
Improved gate progression and front-foot lift alone are insufficient. Do not
relax the clearance/landing criteria to turn this result into a pass.
