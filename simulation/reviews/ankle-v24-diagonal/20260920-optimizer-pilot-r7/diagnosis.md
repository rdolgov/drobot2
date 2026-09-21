# Optimizer recovery succeeded; walking still failed

R7 completed 250 iterations (4,096,000 transitions) in 630.93 seconds.
It restored the R6 checkpoint1500 actor/critic/normalizers, used fresh Adam
at fixed 0.0001 and full curriculum, and retained revision4 physics, rewards,
controller, physical joint/actuator limits and success criteria. All source
and asset hashes still match the recorded contract.

The learning rate remained exactly 0.0001. The fraction of actual last-update
minibatch actions with both Beta concentrations at one averaged 11.55% over
the final 50 iterations. Mean absolute deterministic correction was 0.192
over those minibatches. No ten-iteration collapse stop triggered. This avoids
the late R6 failure; 250 iterations do not establish indefinite optimizer
stability, nor does this isolate learning rate from checkpoint/optimizer reset.

Training still recorded zero full paired landings. Completed episodes included
41 front-left, one rear-left, 1,320 front-right and zero rear-right qualified
touchdowns. All 5,950 completed moving episodes timed out. The single rear-left
touchdown was insufficient and was not reproduced in the independent trials.

## Independent evaluation

`../r7-model249-mean` contains the copied checkpoint, matching controller source,
provenance and trace. Four first-episode trials used deterministic actor means,
both starting diagonals, nominal/85% motors and full curriculum. All timed out
at 7.87–11.75 seconds: two at advance and two at settle. None completed a full
paired landing or a rear qualifying touchdown. None reached stop/settle at
40 seconds. No physical fall before timeout is not walking qualification.

Actor outputs are no longer zero (maximum sampled absolute action 0.9843), but
rear clearance remains effectively zero: maximum 0.061 mm across selected
rear feet, versus the unchanged 8 mm qualification threshold. Selected front
feet reached approximately 15–41 mm in the sampled traces. Scheduler changes
of pair without physical qualified landings are not successful steps.

The offline `recovery-comparison.json` compares these traces against the R6
checkpoint1500 initialization under the same diagnostic. In every R7 trial,
at least one supporting knee was near its configured strength-scaled cap in
all sampled lift/advance frames for that leg. Associated mean tracking errors
were roughly 9 degrees, reaching 19.5 degrees on one weak-motor rear support.
Torque-cap comparisons use instantaneous sampled speed; the implicit-drive
effort telemetry is approximate, not measured hardware torque.

## Decision and next dependency

Do not deploy this policy or extend this setup blindly. The optimizer pilot
separates the late policy collapse from the persistent support/clearance
failure. Several earlier isolated posture, lift, timing and preload prototypes
also failed; they are preserved and must not be silently adopted or repeated.

Further scheduled runs are paused while the physical model is reconciled with
the actual 16-servo robot. The pending user question asks for current total
weight including battery and all 16 servos. Current 3.444 kg mass is estimated;
the earlier measured 7 lb value belongs to the pre-upgrade 12-servo robot.
Loaded actuator response is also not calibrated. This uncertainty matters
because the observed failure occurs near simulated load limits. These results
do not prove the real robot cannot walk diagonally and do not justify raising
torque limits, weakening success criteria or inventing a replacement mass.

When current physical information is available, review
`../model-assumptions-review.md`, reconcile the model, and select a bounded
diagnostic before another training run. The user's standing authorization for
local training and adjustments remains in force; no renewed permission is
needed. No Pi connection, motor motion, deployment, stair policy or physical
limit change was performed. Training and diagnostic artifacts remain local.
