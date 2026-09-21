# V24 completed diagonal-pair walking retraining

Requested September 19, 2026 after the timing-corrected V23 hardware policy
began walking but still made short, shaky, jerky movements. The user selected
front-left + rear-right, then front-right + rear-left. This is flat-ground
walking; it does not use a stair policy or stair rewards.

**Status:** both `20260919-seed2419-r3` and revision-3
`20260919-seed2419-r4` completed all 3000 iterations with zero qualifying full
paired landings in every iteration report. Neither is a usable diagonal gait.
R4's result and final-checkpoint hash are retained in its `result-summary.json`.
It produced no qualifying rear-foot touchdowns. Independent diagnostics found
lift-cancelling residuals and unresolved support/load tracking, and rejected
several alternative stances.

Current source is **revision 4**, described in
[preserve swing and reward both feet](rl-ankle-v24-diagonal-revision4.md).
It constrains selected swing corrections and uses monotonic per-foot actual
clearance shaping while retaining strict full-pair success criteria. A bounded
candidate diagnostic executed; it is not evidence of a trained successful gait.
R5 stopped around iteration 120 after a Windows progress-file access error
broke its supervisor and worker logging pipe. Its evidence is preserved.
Progress writes now retry sharing conflicts and worker output goes directly
to a file. The restart directory is `20260919-seed2419-r6`, initialized from
the original V23 walking actor, with 256 environments, seed 2419, and 3000
planned iterations. Read its `status.json` for actual launch and completion.
The user explicitly authorized continuing local diagnosis and training without
asking again. The [earlier diagnosis](rl-ankle-v24-diagonal-diagnosis-20260919.md)
records the revision-2/3 gate, reward, geometry and startup adjustments.

Revisions 3 and 4 use the 80/280 mm
stance and 14 mm body shift, and caps radial leg reach at 317 mm without
expanding joint limits. Startup keeps every foot at its reset stance before
lifting. The first placement advances half a stride; after the all-foot push,
the ordinary alternating full-stride sequence is continuous. The startup
placement has a separate one-time reward and is excluded from full paired-step
and per-leg step/stride counts. Touchdown diagnostics include startup touchdowns.

The remaining sections record the **historical revision-1 training contract**;
the revision-4 document above takes precedence for current behavior.
The original run, source snapshots, calibration, and checkpoint are preserved.

## Training contract

One actor-initialized PPO run: seed 2419, 256 environments, 3000 iterations,
64 steps per iteration (49,152,000 transitions). The initial actor and its
normalizer come from `ankle-v23-transfer/seed2311-model_2999.pt`; the critic,
optimizer, and iteration start fresh. The unchanged actor has 82 inputs and
16 actions. The meaning of its phase clock and reference has changed, so the
result is **not compatible with the deployed V23 reference controller**.

The existing V2.3 solid-sole USD, masses, joint travel, torque-speed envelope,
60 Hz control, 120 Hz physics, and 2-degree packet limit remain. The current
ankle calibration and measured travel are captured in provenance. Each ankle
uses the intersection of its original model soft bounds and the recorded
travel minus the existing five-degree margin. The captured range is not used
to expand the model's approximately ±33.25-degree ankle envelope.

Training inputs are snapshotted and hashed before launch. The run directory
contains `status.json`, `contract.json`, `command.json`, `train.log`, metrics,
the initial actor, and source/calibration provenance. Training completion is
recorded only after the worker returns and the final checkpoint exists.

## Pair placement

One placement has six separate stages, using quintic position interpolation
with zero endpoint velocity and acceleration:

| Stage | Fraction of nominal placement time | Reference behavior |
| --- | ---: | --- |
| Transfer | 10% | Four-foot support and time to establish the stance |
| Lift | 18% | Raise only the selected diagonal to 22 mm |
| Advance | 24% | Move that pair forward at full reference clearance |
| Lower | 18% | Lower the selected feet at their forward positions |
| Settle | 10% | Hold all four feet down |
| All-foot push | 20% | Shift all foot targets backward by half a stride |

Then the opposite pair starts. The reference closes over two placements;
there is no periodic jump back to a neutral pose. Startup blends into the
initial reference over 2.5 seconds. A stop request finishes the current
placement through settle and holds that stance instead of cutting a swing
short. Episodes reserve eight seconds for this landing/stop behavior.

Final reference advances are 80–100 mm at commands 0.012–0.018 m/s. Both
stride and command grow from 60% to 100% over 1500 iterations. Nominal placement
time is final stride divided by twice commanded cruise speed. Waiting at a
gate extends that time; neither the target distance nor the command is a
measured walking result.

At the end of lift, selected joints must approach their reference. At settle,
all joints must approach it. Both gates require encoder error below 5 degrees,
joint speed below 18 degrees/second, low body tilt and angular speed, held for
150 ms. An additional 1.5-second wait that still fails ends the episode as a
failure; it never releases the next pair by timeout.

These gates use only encoders, IMU, and controller state. They cannot prove
floor contact. Contact forces, sole corners, and whole-robot COM are critic
and reward information, and are absent from the actor and gate conditions.
Any future hardware controller must preserve this distinction.

## Rewards and smoothness

- A completed-pair reward requires genuine liftoff, at least 100 ms of flight,
  8 mm minimum clearance reached during flight, forward advance of at least
  65% of the current stride target, and less than 35 mm sideways displacement.
  The opposite diagonal must remain supporting throughout flight. Both feet
  must qualify, then all four must remain flat and quiet for 120 ms. Tiny
  motions, scraping, or a single successful foot do not earn that reward.
- Longer advances earn a bounded bonus up to the reference stride. A bonus
  for completing another round on every leg discourages one favored pair
  from doing all the work. Counts and strides are logged separately per leg.
- Dense terms reward clearance, the support-foot contact area around COM,
  stance contact, flat soles, and forward progress. The convex hull uses the
  two finite sole patches rather than assuming two foot centers form a
  support triangle. This remains a static support proxy, not a guarantee of
  dynamic stability.
- Penalties address lost support, scuffing, slip, body angular velocity and
  angular acceleration, target acceleration, action changes, lateral drift,
  yaw, power, and ankle-limit saturation. Progress reward allows the intended
  pauses during lift and landing instead of demanding constant body velocity.
- Active ankle references level the sole from measured hip/knee angles and
  body pitch, add pitch-rate damping, and filter the reference. All 16 action
  residuals are filtered. Target requests include acceleration/braking limits;
  joint bounds and the two-degree packet cap take precedence.
- Training retains 85–100% randomized motor strength/rate and adds occasional
  single-cycle output holds to expose 16.7/33.3 ms target-delivery intervals.

## Scope and status

This request authorizes necessary local diagnostics and retraining in this
walking-policy work without renewed permission. No hardware checks, Pi changes,
or automatic deployment are part of the launcher. Training metrics are not held-out results
and cannot establish that the gait is stable on the real robot.

The Pi continues running the timing-corrected V23 policy. A later hardware
handoff needs the new stateful reference, filters, target limiter, calibration
identity, and stopping behavior along with its matching weights; copying the
checkpoint into the old runtime would be incorrect.

Launch once from the repository root:

```powershell
& ./.venv/Scripts/python.exe simulation/isaac/rl/ankle_v24_diagonal/train.py --output simulation/reviews/ankle-v24-diagonal/20260919-seed2419 --warm-start simulation/isaac/models/ankle-v23-transfer/seed2311-model_2999.pt --num-envs 256 --iterations 3000 --seed 2419 --run-name diagonal-2419
```

The supervisor rejects duplicate or changed contracts in an existing run
directory. A failed attempt remains on disk; a correction uses a fresh run
directory so the failed evidence is preserved.

The initial attempt ended before iteration 0. The second attempt preserved
the previously suppressed exception: standing curriculum episodes have zero
stride, which caused a division by zero in the touchdown-distance reward.
The denominator is now bounded; standing episodes still earn no completed-pair
bonus. The corrected run uses `20260919-seed2419-r3`. Both failed attempts and
their source snapshots remain available.
