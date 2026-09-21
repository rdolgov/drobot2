# Diagonal IK V6: ankle lead with measured onset before further lift

Requested September 20, 2026 after another operator report of lift preceding
visible ankle movement. No tests, compilation, simulation, browser validation
or hardware checks were run. Existing recorded data was read; no new motion
or servo-bus transaction was requested for diagnosis.

## Evidence from the actual V5 trial

The latest completed Pi recording is `20260921T005426064068Z-925c59f3`, copied
to `debug/walk-trials/ik-immediate-v5-20260920/`. Metadata identifies
`ankle_ik_v5_immediate_lift_push` and its deployed source hash. It completed
two cycles and ended holding torque.

The record does not show ankle commands postponed until lift is complete.
Both desired and delivered support-ankle targets advance during lift, and
measured ankles also move before the lift endpoint. In the first lift:

| Placement reference time | FR ankle delivered / measured delta | RL ankle delivered / measured delta | RR lifting knee delivered / measured delta |
| --- | --- | --- | --- |
| 0.433 s | +1.12 / 0.00 deg | +1.14 / 0.00 deg | +10.27 / +1.05 deg |
| 0.483 s | +2.61 / +0.97 deg | +2.75 / +0.79 deg | +19.98 / +5.10 deg |
| 0.550 s | +3.81 / +2.02 deg | +4.10 / +2.02 deg | +25.90 / +10.46 deg |
| 0.720 s | +6.34 / +5.27 deg | +7.20 / +6.68 deg | +36.67 / +33.66 deg |

Deltas are relative to the last pre-lift recorded sample (0.317 s). Lift
starts at 0.36 s and its scheduled endpoint is 0.72 s. Approximately 67 ms
sample spacing cannot determine exact physical onset. Subsequent ankle
motion during endpoint holds includes a growing IMU leveling correction:
the last placement's correction rose from roughly -1 to +7.6 degrees while
the lift reference was stopped. That larger late movement is consistent
with the operator's observation, but the recording cannot establish sole
contact, actual body translation, torque reserve or mechanical causes.

V5 synchronized command timing but did not require measured ankle onset
before continuing the lift. The endpoint-only gate could therefore let the
lifting pair progress while support ankles initially lagged.

## Control change

`support_onset_fraction = 0.30` moves 30% of the existing 20 mm lift-phase
push into the first 10% of lift time. The reference is continuous:

- For lift progress p <= 0.10, push fraction is `0.30 * p / 0.10`.
- Subsequently it is `0.30 + 0.70 * f((p - 0.10) / 0.90)`, where
  `f(t) = t + t² - t³` remains the V5 lift profile.
- At p=0.10: push is 6 mm; requested lift is `35 * f(0.10) = 3.815 mm`.
- At p=1: push remains 20 mm; lift remains 35 mm. Swing/landing and the
  remaining 30 mm all-feet advance retain their previous endpoints.

At p=0.10 the clock enters `support_ankle_onset`. It holds that entire
reference pose until both planted ankles have moved at least 0.5 logical
degrees in the nominal push direction from their actual pre-lift encoder
angles, and command delivery is within 0.3 degrees of the current pose.
The direction is derived from the nominal IK, without IMU correction.
For smaller configurable onset targets the threshold is half the nominal
angular move, capped at 0.5 degrees; targets below 0.2 degrees are rejected.
The deployed configuration uses the full 0.5-degree threshold for both ankles.

This is an onset gate: it does not require a zero-velocity settle before
resuming lift. Failure to observe onset faults after two accumulated control
seconds under the existing no-catch-up clock; wall time can be longer when
bus cycles are delayed. Existing fault handling disarms. Existing lift/swing/
plant/settle endpoint gates retain their position, speed and timeout criteria.

The gate confirms encoder rotation, not contact, force or forward travel.
IMU leveling stays active during the gate and afterward, so later balancing
motions remain possible. The controller does not increase torque or disable
leveling to conceal that motion. The existing actual target rate/packet
limits and shared sync-write continue to apply, including at ramp joins.

Only propulsive diagonal mode with a nonzero onset fraction enables this
trajectory/gate. The default onset fraction is zero for other configurations.
The normal 100 mm world stride, 4-second reference cycle, two-cycle stop,
calibrations, directions, ankle range, servo firmware settings and model
weights remain unchanged. Actual cycle time includes the new onset wait.

Client 18 labels the control `ANKLE LEAD V6` and displays an explicit
`LIFT HELD LOW` status while waiting for the support encoders. Status and
recordings include each support ankle's signed measured onset displacement,
required displacement and `support_onset_confirmed`. Those recorded values
describe the onset gate, not continuing motion tracking after it passes.

Deployment record: `debug/deployments/ik-diagonal-measured-onset-20260920/`.
The source installer retains V5 and prerequisite hashes, creates a rollback
backup and never starts a walk. Activation requires current supported/disarmed
readiness because the dashboard restart releases torque.
