# One-leg wall-mounted Isaac range revisit

Date: 2026-07-28

Status: completed range investigation; stair PPO remains gated

## Question

A physical one-leg ST3215 test showed useful free motion, while earlier
NVIDIA Isaac Sim work appeared unable to move an arm even with gravity
disabled. The requested investigation was:

1. reproduce the home setup with one leg mounted to a wall;
2. test whether the three joints have useful range in Isaac;
3. determine what was wrong with the earlier simulator result;
4. decide whether stair-climbing simulation should resume.

## Hardware evidence used

The ignored local hardware files were read but not committed:

- `hardware/one-leg-testbed/leg.toml`;
- `hardware/one-leg-testbed/calibration.json`.

Their 2026-07-28 configuration was:

| Motor | Joint | Configured range | Encoder direction | Center |
| ---: | --- | ---: | ---: | ---: |
| 1 | hip abduction | `-45 to +45 deg` | `+1` | `2048` |
| 2 | hip flexion | `-90 to +90 deg` | `-1` | `2048` |
| 3 | knee | `-120 to +120 deg` | `-1` | `2048` |

The local torque-limit register was `300/1000`. For comparison, the range
runner uses `0.8825985 N*m`, or 30% of published stall torque. This is only a
nominal mapping because the register is not a calibrated linear
joint-torque sensor.

## Recreated fixture

The dedicated URDF contains:

- one fixed wall/fixture root;
- the exact printable 76 x 76 mm hip body mount;
- one hip link and two arm links;
- three ST3215 revolute joints;
- the same moving-link meshes, frames, collision proxies, masses, and
  inertias as the quadruped URDF;
- no virtual foot sphere.

The body-mount CAD transform places its plate back face exactly
`85.084989 mm` behind the hip-abduction axis. The vertical collision wall is
flush with that face. Only the two intentional moving-pivot overlaps are
filtered; wall contact remains enabled for every moving link.

![Wall-mounted one-leg Isaac fixture](../reviews/isaac-one-leg-wall-range.png)

The imported Isaac 6.0.1 asset passed its structural contract:

| Property | Result |
| --- | ---: |
| articulation roots | 1 |
| rigid bodies | 4 |
| revolute drives | 3 |
| filtered moving-pivot pairs | 2 |
| self-collision | enabled |
| wall contact | enabled |

## Measured range

The automatic runner tested each joint two degrees inside both configured
endpoints and tested two combined poses.

| Test | Zero gravity | Earth gravity |
| --- | ---: | ---: |
| hip abduction toward wall, target `-43 deg` | stopped at `-10.250 deg` | stopped at `-10.250 deg` |
| hip abduction away from wall, target `+43 deg` | `42.999 deg` | `41.377 deg` after 2 s |
| hip flexion, targets `+/-88 deg` | maximum error `0.0019 deg` | maximum error `1.4932 deg` after 2 s |
| knee, targets `+/-118 deg` | maximum error `0.0038 deg` | maximum error `0.3519 deg` after 2 s |
| combined `(30, 60, -90) deg` | maximum error `0.0020 deg` | maximum error `1.0537 deg` after 2 s |

The reverse combined pose stopped only on inward hip abduction. A diagnostic
run that disabled only the wall collision made all eight zero-gravity
endpoint and combined poses pass, with a worst error of `0.00382 deg`.

This means the negative-abduction stop was geometric contact with the flush
wall. It was not an Isaac motor or articulation failure. A real fixture that
allows more inward motion must differ by using a wall edge, narrower support,
or additional standoff; that physical boundary should be measured before
changing the simulated wall.

## Full-quadruped cross-check

The unchanged fixed quadruped asset was then tested with zero gravity, no
floor, and no stair:

| Joint | Target | Final | Error | Time to 90% |
| --- | ---: | ---: | ---: | ---: |
| front-left hip flexion | `40 deg` | `39.999070 deg` | `0.000930 deg` | `0.383 s` |
| front-left knee | `70 deg` | `69.998039 deg` | `0.001961 deg` | `0.383 s` |

These probes establish that the full Isaac articulation can move its
sagittal joints. The earlier failure should not be interpreted as a frozen
arm.

## Why stair training remains gated

The prior stair IK targets already fit the narrower quadruped limits. Its
failure was dynamic:

- the `4.53 kg` floating robot lost three-foot support;
- support tips slipped;
- the body tipped toward the stair;
- requested PD effort saturated at the rated cap;
- the commanded front foot never achieved the necessary lift and tread hold.

An unloaded wall-mounted range pass does not prove that the remaining three
legs can support and transfer the robot's weight.

![Rated-torque full-quadruped standing reference](../reviews/isaac-standing-rated-final.png)

PPO stair training was therefore not restarted. The next physical simulation
gate should be:

1. add the intended flat, grippy physical foot and its real contact patch;
2. pass one-foot stair placement with the body externally supported;
3. pass the same motion with three-foot support and measured weight transfer;
4. verify rated-current, voltage-sag, and thermal margin;
5. only then resume stair curriculum training.

## Reproduce

Run from `robot-cad`:

```powershell
$projectPython = '.\.venv\Scripts\python.exe'
$urdfTool = 'C:\Users\roman\.codex\plugins\cache\text-to-cad\cad\0.3.9\skills\urdf\scripts\urdf'
$isaacPython = 'C:\isaacsim\python.bat'

& $projectPython $urdfTool `
  robot_cad\urdf\one_leg_wall_testbed.py `
  -o exports\urdf\one_leg_wall_testbed.urdf

& $isaacPython simulation\isaac\import_one_leg_wall.py `
  --urdf exports\urdf\one_leg_wall_testbed.urdf `
  --output exports\isaac\one_leg_wall_testbed.usdc `
  --report simulation\isaac\output\one-leg-wall\import_report.json

& $isaacPython simulation\isaac\run_one_leg_wall.py `
  --gravity both `
  --screenshot reviews\isaac-one-leg-wall-range.png
```

For direct joint control:

```powershell
& $isaacPython simulation\isaac\run_one_leg_wall.py --interactive
```

Use `1`, `2`, or `3` to select a joint, `Up`/`Down` to move, `Z` to zero,
`G` to toggle gravity, `R` to reset, `C` to print state, and `X` or `Esc` to
save and exit.

The wall-disabled comparison is diagnostic only:

```powershell
& $isaacPython simulation\isaac\run_one_leg_wall.py `
  --gravity zero `
  --disable-wall-contact `
  --report simulation\isaac\output\one-leg-wall\range_report_no_wall_contact.json
```

## Durable files

- `robot_cad/urdf/one_leg_wall_testbed.py`
- `simulation/isaac/import_one_leg_wall.py`
- `simulation/isaac/run_one_leg_wall.py`
- `tests/test_one_leg_wall_urdf.py`
- `exports/urdf/one_leg_wall_testbed.urdf`
- `exports/isaac/one_leg_wall_testbed.usdc`
- `reviews/isaac-one-leg-wall-range.png`
- `simulation/isaac/README.md`
- `docs/stair-feasibility/README.md`

Run products under `simulation/isaac/output/` remain intentionally ignored.

## Validation performed

- URDF generation-time validation: passed.
- Isaac import structural contract: passed.
- Automatic wall-contact range sweep: completed; inward abduction correctly
  reported wall contact.
- Zero-gravity wall-disabled diagnostic: all eight poses passed.
- Full-quadruped zero-gravity flexion and knee probes: passed.
- Focused Python tests: 26 passed.
- Repository tests: 186 passed and 1 skipped; the one Windows temp-fixture
  test that encountered a permission error passed when rerun with a
  workspace-local temp directory.
- Focused Ruff checks and `compileall`: passed.
