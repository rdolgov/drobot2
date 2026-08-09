# Crawl walk V3: common-direction stance and coordinated support push

> Historical failed profile. V4 replaced this walking geometry after the
> physical robot fell backward. See [crawl-walk-v4.md](crawl-walk-v4.md).

## Status

V3 is a **supported hardware commissioning motion**, not a validated floor
walk. It exists to resolve two direct observations from the assembled robot:

1. front and rear legs bent in opposite visible directions;
2. the V2 support-leg motion was too small to see and appeared to move only one
   leg at a time.

The controller now produces the requested common-direction 45-degree stance
and coordinated multi-leg commands. Isaac reproduced the exact joint targets
but the simulated robot fell during the aggressive floor trial. The web app
therefore labels the sequence as a feet-clear test.

## Root cause in V2

The V2 IK selected `+acos(...)` for front knees and `-acos(...)` for rear
knees. At the configured 305 mm height, its start targets were approximately:

| Pair | Hip flexion | Knee |
| --- | ---: | ---: |
| Front | -10 to -14 degrees | +33 to +34 degrees |
| Rear | +9 to +12 degrees | -32 to -34 degrees |

That is intentionally mirrored in the Isaac URDF, but it did not match the
assembled robot's requested convention. The 28 mm distributed push was split
into four 7 mm events, producing only a few degrees of planted-leg motion.
Only the selected swing leg had a large, obvious change.

## Uniform ready stance

For equal two-link lengths `L`, a centered foot target and knee bend `q` have
vertical reach:

```text
down = 2 L cos(q / 2)
```

With `L = 0.159896689 m` and `q = 45 degrees`, the selected height is
`0.295447 m`. The common-direction IK branch produces the same calibrated
joint targets for all four legs:

| Joint | Target |
| --- | ---: |
| Hip abduction | 0.00 degrees |
| Hip flexion | -22.50 degrees |
| Knee | +45.00 degrees |

Electrical direction signs remain in each `leg-N.toml` profile and are applied
only when calibrated degrees are converted to raw servo ticks. The gait does
not rewrite servo IDs, EEPROM, direction signs, or calibration centers.

The dashboard provides **SET UNIFORM 45 STANCE** as a separate first test. It
requires all motors disarmed, validates all targets, arms at measured
positions, and ramps to the stance. Torque remains enabled until **STOP +
DISARM**.

## Coordinated support-push gait

Only one foot is commanded up at once. On every step:

1. the selected foot lifts;
2. it advances by one 60 mm stride;
3. its diagonal planted partner moves rearward by 30 mm;
4. both adjacent planted legs move rearward by 15 mm each;
5. the selected foot lowers and a touchdown interval is held;
6. the next leg begins.

The step order is:

```text
rear-right -> front-left -> front-right -> rear-left
```

Across one cycle, each leg advances 60 mm during its swing and accumulates
60 mm of rearward support motion: 30 mm as the diagonal partner plus two
15 mm adjacent pushes. Targets are therefore continuous and periodic. During
the `swing_push` phase, all four hip-flexion commands change simultaneously,
while only the swing knee performs the extra lift bend.

The tracked supported-test profile is:

| Parameter | Value |
| --- | ---: |
| Cycle period | 12 s |
| Cycles | 1 |
| Foot stride | 60 mm |
| Foot lift | 25 mm |
| Stance height | 295.447 mm |
| Nominal fore/aft offset | 0 mm |
| Hip abduction | 0 degrees |
| Forward/lateral open-loop shift | 0 mm / 0 mm |
| Servo torque setting | 30% |
| Command ramp | 45 degrees/s |

Over the sampled cycle, all knee targets remain positive. The centered stance
is exactly `+45 degrees`; walking targets remain roughly `+43.6` to
`+64.5 degrees`. Hip-flexion targets remain negative and range roughly
`-16.0` to `-38.0 degrees`. These are materially larger than V2 while remaining
inside the configured motor ranges.

## Isaac result and interpretation

The two-cycle rated-torque test used the same target generator, the floating
4.526 kg robot, provisional fork-tip contacts, and a `0.980665 Nm` per-joint
cap. It failed:

| Metric | Result |
| --- | ---: |
| Minimum base height | 0.112 m |
| Maximum body tilt | 97.12 degrees |
| Forward displacement | -0.489 m |
| Maximum joint tracking error | 0.424 rad |
| Expected support contact | 4.42% |
| Maximum loaded-tip slip | 55.14 mm |
| Completed contact-verified steps | 0 / 4 |

The simulation settles in the uniform stance but collapses after motion begins.
This means the profile must not be presented as a floor-ready gait. It does not
prove that the software direction convention is wrong for the rewired physical
robot; it proves that the current URDF mounting convention, rated torque, and
contact model cannot support this aggressive common-branch sequence.

The complete negative report is
[`../validation/isaac-coordinated-support-push.json`](../validation/isaac-coordinated-support-push.json).
The earlier mirrored V2 gait remains a passing simulation reference, but the
assembled-robot trial rejected its visible direction and amplitude.

## Required physical test ladder

1. Support the chassis rigidly with every foot clear of the floor.
2. Start the dashboard and confirm IDs 1-12, the corner map, nominal voltage,
   temperature, current, and `0 / 12` armed.
3. Check the safety confirmation and press **SET UNIFORM 45 STANCE**.
4. Verify all four knees bend in the same intended physical direction. Verify
   the dashboard converges near `-22.5 degrees` hip and `+45 degrees` knee for
   every leg.
5. Press **STOP + DISARM**. Do not continue if one leg is reversed, binds,
   draws abnormal current, sags the supply, heats, or makes unexpected noise.
6. With the chassis still rigidly supported and every foot clear, press
   **TEST COORDINATED MOTION**. Confirm rear-right, front-left, front-right,
   rear-left lift order. During each `SWING PUSH`, visually confirm all four
   hip-flexion joints move and the displayed diagonal `PUSH` partner moves
   rearward most.
7. Press **STOP + DISARM** after the single cycle. Record which physical joint,
   if any, moved opposite to its displayed logical angle.

Do not place this V3 profile on the floor. Floor work resumes only after the
physical direction observations are reflected in the robot description and a
new floating-base validation passes contact, tilt, tracking, slip, and forward
travel gates.

## Commands

Start the hardware dashboard:

```powershell
cd C:\Users\roman\Documents\dev\drobot2\hardware\test-apps
.\install-test-apps.ps1
.\start-four-leg-web.ps1 -Port COM4
```

Reproduce the failing Isaac boundary from the repository root:

```powershell
& C:\isaacsim\python.bat simulation\isaac\run_crawl.py `
  --usd simulation\exports\isaac\quadruped_robot_floating.usdc `
  --headless --gait-mode coordinated-push --torque-cap rated `
  --cycles 2 --period 12 --stride 0.060 --lift 0.025 `
  --weight-shift-forward 0 --weight-shift-lateral 0 `
  --stance-down 0.295447 --stance-fore-aft 0 `
  --abduction-deg 0 --start-z 0.455 `
  --min-forward-displacement 0.010 `
  --report hardware\test-apps\four-leg-dashboard\validation\isaac-coordinated-support-push.json `
  --screenshot hardware\test-apps\four-leg-dashboard\validation\isaac-coordinated-support-push.png
```
