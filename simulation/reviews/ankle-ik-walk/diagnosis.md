# Sixteen-motor IK walking attempt, September 20, 2026

The local dashboard now has a deterministic 16-motor walking alternative to
RL: one leg at a time or alternating diagonal pairs, with active ankle pitch,
smooth foot trajectories and encoder-gated placement. **The simulated walk
is not successful yet. Nothing from this attempt has been deployed to the Pi.**

The current implementation is described in
[`ankle-ik-gait-v1.md`](../../../hardware/test-apps/four-leg-dashboard/specs/ankle-ik-gait-v1.md).
The current manifest requests a 60 mm stride, 25 mm lift and 16-second cycle
before encoder waits. Body depth is 305 mm, opposed fore/aft stance offsets
are 60 mm, forward transfer is 10 mm, and crawl lateral transfer is 30 mm.
Reach correction can reduce the requested depth. These are trajectory
parameters, not achieved physical stride or clearance.

## Evidence

The user-requested [video recordings](20260920-current-video/README.md)
show the current V4 controller with foot-force/clearance overlays and slow
replay. All six recording trials reproduce the prior failure times and
messages; no gait or physical limits were changed to make the recordings.

Each completed diagnostic used six first episodes: crawl at nominal/85%
motor strength and both diagonal starting pairs at nominal/85% strength.
All used the working 3.39514659 kg total-mass estimate, scaling existing
mass and inertia together without changing their relative distribution.
Effort caps remained 0.8825985 Nm nominal and 0.750208725 Nm at 85% strength.
The existing asset, joint limits and PD gains were retained.

| Run | Change | Result |
| --- | --- | --- |
| `20260920-v1` | Two-link ankle-pivot IK, 295 mm depth, 80 mm stance offsets, 25 mm forward transfer | All six lift gates timed out; zero qualified steps |
| `20260920-v2-lateral` | Add 25 mm lateral transfer | Runner rejected an unreachable pose; no successful trial |
| `20260920-v3-lateral-reach` | Exact roll-plane reach bound replaces heuristic reserve | All six lift gates timed out; zero qualified steps |
| `20260920-v4-upright` | 305 mm depth, 60 mm stance offsets, 10 mm forward and 30 mm lateral transfer | All six lift gates timed out; zero qualified steps |

Latest [report](20260920-v4-upright/report.json) and
[trace](20260920-v4-upright/trace.jsonl):

| Mode/start | Strength | Failure time | Largest tracking error at gate |
| --- | ---: | ---: | ---: |
| Crawl / rear-right | 100% | 5.97 s | 7.4 degrees |
| Crawl / rear-right | 85% | 5.97 s | 7.8 degrees |
| Diagonal / FL + RR | 100% | 7.40 s | 9.2 degrees |
| Diagonal / FL + RR | 85% | 7.40 s | 9.6 degrees |
| Diagonal / FR + RL | 100% | 7.40 s | 8.6 degrees |
| Diagonal / FR + RL | 85% | 7.40 s | 22.4 degrees |

In nominal crawl, the selected rear-right sole remained on the floor while
the unintended front-left sole lifted about 5.3 mm. At timeout their vertical
forces were approximately 2.8 N and 0 N respectively; rear-left and front-right
each carried about 15.3 N. Body tilt was 4.5 degrees. The rear-left hip and
knee, front-right knee, selected rear-right knee and both supporting ankles
were at their configured effort caps. This is a load-transfer/tracking
failure, not a successful slow step.

Offline vertical-force equilibrium calculations suggested the more upright
pose could reduce the best achievable peak static torque from approximately
0.83 to 0.67 Nm. Dynamic simulation still failed. That calculation is an
optimistic bound: it does not establish realizable contact-force distribution,
balance, PD tracking or dynamic feasibility. It must not be used to override
the failed physical traces or increase allowed torque.

The separate [R7 mass comparison](../ankle-v24-diagonal/r7-model249-mass3395/report.json)
also produced zero full pairs and zero qualifying rear touchdowns. The weight
estimate resolves the old request for a new weigh-in; it does not resolve
the walking failure.

## Limits of the evidence

The simulator runs the same pure trajectory/clock and equivalent rate and
packet bounds, but does not emulate the USB bus, encoder quantization or
complete dashboard startup/fault handling. Feedback is sampled every 50 ms
with ideal read latency. A single initialization writes the starting pose;
there are no walking state corrections. Failed trials end before the scheduled
stop, so stop/settle behavior is not physically qualified. Failed environments
hold their last simulated target for diagnosis; the dashboard's fault path
disarms instead. Per-leg landing criteria alone would also be insufficient
to qualify paired walking without paired timing and support evaluation.

Seven focused pure/controller tests passed earlier in this attempt, covering
IK geometry, trajectory continuity, tracking gates, stop and fake-bus 16-motor
delivery. The final startup-timeout, held-progress and restart-state edits
were inspected but not rerun. No broad suite, browser validation, hardware
connection or motion check was run. Saved provenance records the exact
sources used for each diagnostic, rather than claiming the latest source
is identical to an earlier test snapshot.

The next useful read-only investigation is to compare actual sole attitude,
support ankle moments and available contact patches through transfer/lift.
The controller currently levels commanded pitch relative to its chassis; it
does not use IMU/world attitude or measured hip/knee deflection to level the
sole. Neither that missing feedback nor an unmeasured actuator response has
yet been demonstrated to explain or fix the failure. Preserve the current
traces as the comparison baseline; do not extend training blindly or weaken
the landing/tracking criteria.

That read-only follow-up is now complete; see
[`sole-attitude-findings-20260920.md`](sole-attitude-findings-20260920.md).
In nominal FL + RR lift, rear-leg joint motion contributes +22.7 mm of sole
rise, body rotation removes 15.7 mm and body drop removes 4.9 mm, leaving only
2.1 mm of sole-center rise and no lowest-corner clearance. Support soles stay
closer to floor level than the requested chassis-relative targets while
support-joint errors and torque caps persist. This quantifies the body-support
failure; it does not qualify a corrective controller or identify actual
contact-pressure distribution. No new rollout was run for these calculations.

## Reproduction commands (not additional executions)

From the repository root, use a new output directory for each run:

```powershell
C:/isaacsim/python.bat simulation/isaac/diagnostics/ankle_ik_walk.py --output simulation/reviews/ankle-ik-walk/NEW-RUN --seconds 60
```

Focused controller checks:

```powershell
$env:PYTHONPATH = 'hardware/test-apps/four-leg-dashboard/src;hardware/test-apps/one-leg-testbed/src;onboard/policy-runtime/src'
.venv/Scripts/python.exe -m pytest hardware/test-apps/four-leg-dashboard/tests/test_ankle_ik_gait.py -q
```

Honor the user's verification instruction before executing either command.
