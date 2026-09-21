# Diagonal IK V5: immediate planted-ankle push during lift

Requested September 20, 2026 after the operator observed delayed support-ankle
movement with V4. Source changes and analysis of existing recordings only;
no tests, compilation, simulation, UI validation or hardware checks were run.
Deployment status: `debug/deployments/ik-diagonal-immediate-push-20260920/`.

## Recorded evidence

Two existing Pi recordings were downloaded without contacting the servo bus:
`20260921T004236623929Z-ee0018c9` and
`20260921T004314914106Z-6a9f2521`, stored under
`debug/walk-trials/ik-sync-v4-20260920/`. Their metadata identifies V4 and
the expected installed gait-source hash. Both reached the two-cycle limit.

In the latest recording's first lift, relative to the last pre-lift sample:

| Reference time within placement | Scheduled body push | FR ankle target / measured change | RL ankle target / measured change |
| --- | --- | --- | --- |
| 0.383 s | 0.02 mm | +0.13 / 0.00 deg | -0.11 / 0.00 deg |
| 0.450 s | 1.04 mm | +0.70 / +0.09 deg | -0.04 / -0.18 deg |
| 0.550 s | 5.52 mm | +2.43 / +1.41 deg | +1.16 / -0.26 deg |
| 0.600 s | 7.90 mm | +3.25 / +2.29 deg | +1.93 / +0.18 deg |

Lift spans reference time 0.36–0.72 s. Samples are approximately 67 ms apart,
so these are observations at sample times, not exact motion-onset timestamps.
The rear-left leveling correction decayed from about 1.8 degrees before lift
to 1.25 degrees at time 0.50 s, opposing the initially small positive push.
At other placements the IMU correction also changed direction during endpoint
holds. These records do not establish body displacement, contact, available
torque, or the mechanical cause of the measured target lag.

## Change

- `support_lift_fraction = 1.0` moves the entire existing 20 mm early advance
  into lift. It is held through swing and landing; the existing remaining
  30 mm follows with all four feet scheduled down. No extra stride is added.
- Lift and support push both use `f(t) = t + t² - t³`, clamped to [0, 1].
  Position is continuous, initial velocity is nonzero, and endpoint velocity
  is zero. This intentionally removes the previous slow onset. The existing
  common motor interpolation fraction, packet limits and sync-write remain.
- The upcoming planted pair uses planned hip/knee ankle feedforward already
  during weight transfer, instead of switching from measured-angle leveling
  precisely when lift starts. IMU compensation and its bounds remain active.
- Raised feet translate forward with the body during lift, retaining their
  body-relative X. Swing then starts from the carried 20 mm world advance
  and completes the original 100 mm world stride. This avoids pulling the
  raised front ankle beyond its flat-sole limit with the stronger early push.
- Client 17 labels the button `IMMEDIATE ANKLE PUSH V5`. New recordings expose
  `ankle_leveling_correction_deg` explicitly alongside desired, delivered and
  measured joint angles; controller identifier is
  `ankle_ik_v5_immediate_lift_push`.

The 4-second reference cycle, 35 mm lift, two-cycle limit, 200 deg/s walking
target rate, 4-degree packet cap, initial stance rate, joint/ankle limits,
calibrations, directions, motor firmware settings and fault guards remain.
Single-leg crawl retains its original quintic lift and trajectory.

## Independent calculations (not controller execution)

For the configured support positions X=+80/-80 mm and down=288 mm, using
the two equal 159.896689 mm links and zero body tilt:

| Time into nominal lift | V4 body push | V5 body push | V5 support ankle advance, front / rear |
| --- | --- | --- | --- |
| 16.67 ms | 0.009 mm | 0.967 mm | 0.308 / 0.310 deg |
| 90 ms | 1.035 mm | 5.938 mm | 1.862 / 1.935 deg |
| 180 ms | 5.000 mm | 12.500 mm | 3.844 / 4.171 deg |
| 360 ms | 10.000 mm | 20.000 mm | 6.018 / 6.873 deg |

These are scalar geometry calculations without importing the controller;
actual first-frame phase, output limits, IMU compensation and servo load
affect the observed movement. The lifted front endpoint becomes X=30 mm,
down=253 mm, ankle=30.424 degrees (previously 32.957); the lifted rear
endpoint is X=-130 mm, down=253 mm, ankle=0.002 degrees. Limits remain in force.
No claim of immediate physical movement or successful forward travel can be
made until an operator trial supplies new feedback.

The source-only installer retains the established file/prerequisite hashes
and rollback backup. It does not start a walk. Restart releases torque, so
activation requires the operator's current supported/disarmed readiness.
