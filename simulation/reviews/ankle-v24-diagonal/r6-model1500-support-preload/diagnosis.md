# Model-based support preload: rejected diagnostic

The isolated helper biases supporting-joint position targets using a restricted
vertical-force static support solution divided by the simulation's 30 Nm/rad
position stiffness. It solves at 10 Hz and filters offsets over 0.15 s. Offset
torque is capped at the existing nominal 0.8825985 Nm; actual actuator strength,
torque-speed limits, joint bounds and command shaping remain unchanged.
Reference inputs are nominal geometry, the fixed URDF mass ledger and the
scheduled support pair. No simulated contact or force is used in the reference.

Four first-episode trials used checkpoint 1500, deterministic means, full
curriculum, both diagonal starts and nominal/85% motors. All timed out at
6.53–7.63 s: three at settle and one at advance. No full paired landing or rear
qualifying touchdown occurred. Rear-right peaked at 1.37 mm in one trial;
rear-left remained effectively in contact. No fall preceded the timeouts.
The actor was not trained with the intervention.

Supporting knees still spent 93–100% of their sampled lift/advance frames near
the configured cap, with mean tracking errors of approximately 5.8–16.1 degrees.
Implicit-drive effort is approximate telemetry. The comparison to the default
controller is preserved in `../r6-model1500-support-preload-comparison.json`;
exact helper/static-solver sources and URDF input are copied here.

This preload did not fix support tracking or rear clearance and is not adopted
for training or hardware. The static model assumes level posture, flat soles
and freely allocatable nonnegative vertical corner forces; it does not prove
that real contact forces can realize its optimum. Simulation position stiffness
is not a calibrated real-servo torque gain. Do not deploy this helper or raise
motor limits to make it pass. R6 training remains unchanged.
