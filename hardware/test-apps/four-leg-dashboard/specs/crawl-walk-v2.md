# Crawl walk V2: kinematics, gait, and test plan

> Historical design record: this mirrored 28 mm gait passed its Isaac model,
> but the assembled-robot trial reported opposite visible leg directions and
> motion that was too small. The active supported commissioning profile is
> documented in [`crawl-walk-v3.md`](crawl-walk-v3.md).

## Goal

Crawl V2 turns calibrated joint motion into deliberate foot-space motion. The
controller should move the body forward because planted feet push rearward,
not merely because the legs execute a visible sequence.

The selected first hardware profile is still open-loop. It is a commissioning
gait, not an autonomous balance controller. Keep the body supported for the
first run and keep the physical power cutoff within reach.

## What kinematics already exist

Each leg has analytic inverse kinematics (IK). The dashboard and Isaac runtime
use the same equations and the same measured link length:

- upper and lower link length `L = 0.159896689 m`;
- desired foot height below the hip `d`;
- desired foot fore/aft coordinate `x`;
- reach `r² = d² + x²`;
- knee angle `q_k = s * acos((r² - 2L²) / (2L²))`;
- hip angle `q_h = atan2(x, d) - atan2(L sin(q_k), L + L cos(q_k))`.

`s` selects the two-link knee branch. The safe symmetric support stance uses
equal bend magnitudes but mirrored signed branches: `s = +1` for the front
pair and `s = -1` for the rear pair. The signs look opposite in joint space,
but the knees and feet open away from the chassis center and create a useful
support polygon. Left/right electrical direction differences remain in the
servo profiles and are applied when calibrated degrees are converted to raw
servo ticks.

Hip abduction supplies the lateral part of the 3-DOF solution. A requested
body shift is converted into a lateral foot displacement, then into a new
abduction angle with `atan2(lateral, vertical)`. The sagittal two-link IK is
solved at the resulting leg length.

The solver rejects an unreachable foot point. Before a crawl starts, the
dashboard samples a complete cycle and also converts every result through the
real calibration and motor limit checks. This catches geometric reach and
configured joint-limit violations before torque is enabled.

## Why the old crawl moved legs but barely moved the body

The previous sequence placed rear-right, front-right, rear-left, and
front-left in turn. The feet were then reset together during only 10% of the
cycle. That short reset was the only meaningful propulsion phase. On real
feet, compliance, controller lag, or partial contact can consume that pulse,
leaving a convincing leg animation with little net body travel.

## Selected distributed-push gait

V2 keeps the same one-foot-at-a-time order but divides propulsion into four
smaller events. Each quarter-cycle has these phases:

1. shift weight toward the three-foot support triangle;
2. lift the selected foot;
3. move it from its rear workspace point to its front workspace point;
4. lower and confirm the commanded touchdown interval;
5. return the body shift toward center;
6. move all four planted foot targets rearward by one quarter stride;
7. settle briefly before the next foot.

After four steps, every foot has moved forward by one full stride during its
swing and backward by four quarter-stride pushes while planted. Its target is
therefore periodic, while ideal body advance is one stride per cycle. The push
is spread across four all-feet-down intervals instead of one abrupt reset.

The tracked profile is:

| Parameter | Value | Reason |
| --- | ---: | --- |
| Cycle period | 20 s | slow enough for rated-torque tracking |
| Cycles | 1 | guarded 20 s first physical trial; raise only after success |
| Foot stride | 28 mm | longest tested profile that cleared every gate |
| Foot lift | 14 mm | clears the fork-tip proxy without excess knee demand |
| Stance height | 305 mm | preserves rated-torque mechanical advantage |
| Nominal fore/aft footprint | 25 mm | more centered workspace than the old 35 mm setting |
| Forward weight shift | 16 mm | keeps all three support feet loaded during swing |
| Lateral weight shift | 0 mm | avoided unverified open-loop side loading |
| Servo torque setting | 30% | unchanged |
| Command ramp | 45 deg/s | unchanged |

The manifest accepts up to 50 mm stride and 25 mm lift so experiments are not
artificially blocked, but values above the tracked profile are not thereby
validated. A full-cycle limit preflight still runs before arming.

## The 45-degree stance request

With the current 25 mm nominal fore/aft position, knee magnitude is about:

- `33.75°` at the selected 305 mm stance;
- `39.44°` at 300 mm;
- `44.43°` at 295 mm;
- `47.18°` at 292 mm.

The 295 mm mirrored stance matches the desired 45° magnitude closely and
remained upright, but a 35 mm gait reached `0.293 rad` maximum tracking error
at rated torque and advanced only `4.70 mm` over two cycles. A 292 mm stance
with every knee forced onto the same world-direction branch could stand at
rest, but tipped after a foot lifted even with a 25 mm stride and no artificial
weight shift. It is therefore not the automatic walking default.

This is a torque and support tradeoff, not a lack of IK. A lower body bends the
knees more and creates geometric workspace, but also increases the moment arm
that each servo must hold. The next step toward a reliable 45° walking stance
is closed-loop contact/IMU support or a verified sustainable torque/current
increase, not merely larger angles.

## Why diagonal pairs are not the default

Moving front-left with rear-right and front-right with rear-left is a diagonal
pair gait, usually closer to a trot than a crawl. During each pair swing the
robot has only the opposite diagonal contact line. It is not statically stable;
small center-of-mass, floor-friction, or timing errors create roll/pitch motion.

Diagonal motion becomes reasonable after the controller has IMU attitude
feedback, foot-contact evidence, and a fast stop/recovery rule. Until then,
one-foot swing plus three-foot support is the safer foundation. Opposite legs
already cooperate in V2 as loaded support legs while the selected foot moves.

A one-foot-at-a-time diagonal ordering was also simulated as an intermediate
experiment. It moved `48.37 mm` over two cycles with `0.116 rad` tracking error,
but only rear-right passed the complete three-support-foot contact gate and
loaded-tip slip reached `14.91 mm`. Adding an 8 mm lateral shift increased slip
to `22.24 mm`. Both variants were rejected; simultaneous diagonal swing would
remove still more static support.

## Isaac selection results

All results below use the floating 4.526 kg robot, provisional printed-PLA
contact, and the sustainable `0.980665 Nm` rated-torque cap.

| Candidate | Result |
| --- | --- |
| 300 mm stance, 40 mm continuous wave | `-11.00 mm`; tracking `0.156 rad`; rejected |
| 285 mm stance, 50 mm continuous wave | `+33.90 mm`; tracking `0.529 rad`; rejected |
| 295 mm stance, 35 mm distributed push | `+4.70 mm`; tracking `0.293 rad`; rejected |
| 305 mm stance, 30 mm distributed push | `+19.91 mm`; tracking `0.159 rad`; rejected by the `0.150 rad` gate |
| 305 mm stance, 28 mm distributed push, 12 mm shift | `+17.07 mm`; front-right support gate failed |
| 305 mm stance, 28 mm distributed push, 16 mm shift | **PASS**; `+19.35 mm`; tracking `0.144 rad` |

The selected two-cycle run also recorded `1.94 mm` lateral drift, `2.18°`
maximum tilt, `99.92%` expected support contact, `5.32 mm` maximum loaded-tip
slip, `1.663 rad/s` maximum joint speed, and all four contact-verified foot
steps complete. Isaac is evidence for the selected
software/contact model; it does not certify the wiring, printed feet, supply,
or physical floor.

## Hardware test ladder

1. Stop any old dashboard process and launch the updated app from
   `hardware\test-apps`.
2. Support the body with every foot clear. Confirm all IDs, corner labels,
   measured angles, voltage, temperature, and current.
3. Confirm `cycles = 1` remains set in `hardware\robot-runtime\four-leg.toml`.
   Keep the 28 mm/14 mm/305 mm tracked geometry unchanged.
4. Run one cycle in the air and confirm rear-right, front-right, rear-left,
   front-left order. During each `ALL FEET PUSH` phase, all feet must command
   rearward relative to the body.
5. Lower the support until the feet barely load. Repeat one cycle and stop on
   dragging, reversed motion, supply sag, a current warning, heat, noise, or
   unexpected body roll.
6. Only then try one floor cycle with a tether or overhead catch. Mark the
   starting body position and measure net travel and yaw.
7. Restore `cycles = 4` only after a single cycle produces forward travel and
   no warning.

Do not increase stride and lower the stance in the same experiment. Change one
parameter at a time, rerun Isaac, then repeat the supported hardware ladder.

## Commands

Install and start the hardware dashboard:

```powershell
cd C:\Users\roman\Documents\dev\drobot2\hardware\test-apps
.\install-test-apps.ps1
.\start-four-leg-web.ps1 -Port COM4
```

Reproduce the selected two-cycle Isaac run from the repository root:

```powershell
& C:\isaacsim\python.bat simulation\isaac\run_crawl.py `
  --usd simulation\exports\isaac\quadruped_robot_floating.usdc `
  --headless --gait-mode distributed-push --torque-cap rated `
  --cycles 2 --period 20 --stride 0.028 --lift 0.014 `
  --weight-shift-forward 0.016 --weight-shift-lateral 0 `
  --stance-down 0.305 --stance-fore-aft 0.025 `
  --abduction-deg 0 --start-z 0.455 `
  --min-forward-displacement 0.010 `
  --report hardware\test-apps\four-leg-dashboard\validation\isaac-distributed-push-crawl.json `
  --screenshot hardware\test-apps\four-leg-dashboard\validation\isaac-distributed-push-crawl.png
```

## Next controller layer

The IK and gait scheduler are now separate, so closed-loop control can build on
the same foot targets. The next useful inputs are body roll/pitch from an IMU,
per-foot contact or load estimates, and measured-vs-commanded joint lag. Those
signals can pause swing until unloading, end lowering on contact, reduce stride
when a support foot slips, and later enable diagonal-pair motion with active
balance.
