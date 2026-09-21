# Sixteen-motor inverse-kinematics walk

This documents the initial small-stride implementation and its recorded
results. The diagonal button now uses the separate, not-yet-run
[larger V2 profile](ankle-ik-diagonal-v2.md); crawl retains the parameters below.

Local implementation requested September 20, 2026. Hardware deployment and
powered validation have not been performed. This is a separate deterministic
controller; it does not load RL or stair-climbing weights.

The dashboard's crawl and diagonal-pair buttons automatically use
`ankle_ik_gait.py` when all sixteen calibrated motors are configured. The
existing supported-commissioning acknowledgment, bus ownership, software
travel, heartbeat and disarm controls remain in force. Original 12-motor
configurations retain their original gait implementation.

Each hip-to-knee and knee-to-ankle segment is 159.896689 mm. IK targets the
ankle pivot separately from the V2.3 sole's 43.2117 mm offset. Front and rear
legs use opposite knee branches. The ankle target is `-(hip + knee)`, giving
zero target sole pitch relative to the chassis. Abduction introduces sole
roll that the pitch-only ankle cannot cancel. This is a target orientation,
not proof of actual sole level or contact. Gait
ankle targets are restricted to ±33.25 degrees even though manually captured
individual servo travel is wider. Reach is capped at 317 mm.

Defaults in `[ankle_ik]` are a 16-second reference cycle, 60 mm stride, 25 mm
lift, 305 mm ankle-pivot depth and 60 mm opposed fore/aft stance. A 10 mm
forward body transfer is requested for rear swings and diagonal pairs;
front single-leg swings use no forward transfer. Single-leg mode also
requests 30 mm lateral body transfer away from the selected foot. Diagonal
mode has no lateral transfer. Exact hip offsets and a common height
adjustment retain reach for every foot. The body geometry and rear battery remain
unchanged. Quintic profiles have zero endpoint velocity and acceleration.

The single-leg order remains rear-right, front-right, rear-left, front-left.
Diagonal mode alternates front-left/rear-right then front-right/rear-left.
Each placement has transfer, lift, swing, lower, plant, all-foot push and
settle phases. A shared encoder-gated clock stops at lift, swing and plant
endpoints, requiring all joint errors within 5 degrees, delivered target
errors within 0.3 degrees and joint speeds below 18 degrees/s continuously
for 0.15 seconds. It faults after two seconds of controller ticks waiting;
bus delays can make this longer in wall-clock time. Startup positioning and
settling have a separate 15-second wall-clock timeout. Contact is not inferred
from encoder tracking. Every selected foot must be independently evaluated
for lift, advance and landing in simulation.

The motor worker requests at most 60 Hz, skips missed output slots and never
catches up the trajectory clock after bus delays. Position packets retain
the two-degree cap; abduction is limited to 60 degrees/s and the other axes
to 100 degrees/s. Feedback is read at most every 50 ms; read failures or a
transaction over 200 ms fault. STOP during swing completes lowering and the
plant gate, then holds the resulting pose. STOP while all feet are scheduled
planted holds the current target. DISARM remains the immediate torque-off
operation. Feedback errors and gait timeouts use the existing fault/disarm
path. A held partial-cycle pose must be repositioned or disarmed before
a new walk when it differs from the start stance.

Simulation results and reproduction commands are recorded in
[`simulation/reviews/ankle-ik-walk/diagnosis.md`](../../../../simulation/reviews/ankle-ik-walk/diagnosis.md).
The latest six trials all timed out at the first lift with zero qualifying
steps. This implementation is not qualified for a floor walk and has not
been deployed to the Pi. The working total-mass estimate is 3.395 kg, from the recorded
7 lb 12-servo robot plus four specified 55 g motors. Net mount/foot changes
and baseline measurement precision remain estimates.
