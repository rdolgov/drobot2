# Isaac Sim 6.0 quadruped checks

These runners use the generated URDF as the physical source of truth and must
be launched with Isaac Sim's bundled Python. `monolithic` produces one
self-contained binary USDC, which is the preferred handoff format:

```powershell
$isaacPython = 'C:\isaacsim\python.bat'
$urdf = (Resolve-Path 'exports\urdf\quadruped_robot.urdf').Path

& $isaacPython simulation\isaac\import_quadruped.py `
  --urdf $urdf `
  --output-dir simulation\isaac\output\monolithic\fixed `
  --mode fixed `
  --asset-layout monolithic `
  --report simulation\isaac\output\monolithic\fixed-import.json

& $isaacPython simulation\isaac\import_quadruped.py `
  --urdf $urdf `
  --output-dir simulation\isaac\output\monolithic\floating `
  --mode floating `
  --asset-layout monolithic `
  --report simulation\isaac\output\monolithic\floating-import.json
```

Each import report must show one articulation root, 13 rigid bodies, 12
revolute joints, and 12 angular drives. Use its `root_usd` value for validation.
`--asset-layout packaged` remains available when a multi-file, multi-physics
payload package is specifically needed; keep that entire generated directory.

Self-collision is enabled by default. The importer preserves collision between
different legs and between non-adjacent robot links, while authoring exactly
12 filtered pairs for directly connected joint neighbors whose CAD envelopes
overlap at their pivots. Use `--disable-self-collision` only as a troubleshooting
comparison; it permits legs to pass through each other.

```powershell
& $isaacPython simulation\isaac\validate_quadruped.py `
  --usd <fixed-root.usda> `
  --mode fixed `
  --torque-cap rated `
  --report simulation\isaac\output\fixed-rated.json `
  --screenshot reviews\isaac-fixed-rated.png

& $isaacPython simulation\isaac\validate_quadruped.py `
  --usd <floating-root.usda> `
  --mode floating `
  --torque-cap rated `
  --report simulation\isaac\output\standing-rated.json `
  --screenshot reviews\isaac-standing-rated.png

& $isaacPython simulation\isaac\run_crawl.py `
  --usd <floating-root.usda> `
  --headless `
  --torque-cap rated `
  --cycles 4 `
  --report simulation\isaac\output\crawl-rated.json `
  --screenshot reviews\isaac-crawl-rated.png
```

The contact-verified quasi-static trial uses a straighter stance and explicit
weight transfer before each foot is lifted:

```powershell
& $isaacPython simulation\isaac\run_crawl.py `
  --usd <floating-root.usda> `
  --headless `
  --gait-mode quasi-static `
  --torque-cap rated `
  --cycles 2 `
  --period 20 `
  --stride 0.015 `
  --lift 0.010 `
  --weight-shift-forward 0.030 `
  --weight-shift-lateral 0 `
  --stance-down 0.310 `
  --stance-fore-aft 0.025 `
  --abduction-deg 0 `
  --start-z 0.460 `
  --review-phase 0.11 `
  --report simulation\isaac\output\quasistatic-rated.json `
  --screenshot reviews\isaac-quasistatic-rated.png
```

In this mode, all four distal links have ground-filtered contact tracking.
Acceptance requires the three support contacts to remain loaded, each swing
foot to unload and clear the floor, each touchdown to persist, and planted
fork tips to stay within the configured slip limit.

Repeat the standing and crawl runs with `--torque-cap stall` only as a short
peak-torque comparison. The rated profile is the sustainable ST3215 test.
Both runners accept `--effort-limit-nm` for an explicit custom cap.

The default acceptance thresholds are deliberately strict: ten seconds of
stable standing, four crawl cycles, at least 20 mm forward travel, less than
50 mm lateral drift, less than 25 degrees body tilt, and less than 0.15 rad
maximum joint error. Do not weaken them merely to turn a shuffle into a pass.

The present model has no printed feet. Its distal fork-tip collision contacts
are intentionally recorded as an approximation in every report, so a simulated
pass does not replace printed-foot, floor-friction, or endurance testing.

## Manual articulation

The portable handoff files are:

- `exports/isaac/quadruped_robot_fixed.usdc`
- `exports/isaac/quadruped_robot_floating.usdc`
- `exports/isaac/quadruped_robot_manual_world.usda`

The world references the floating USDC beside it, so keep those two files in
the same directory. It already contains Earth gravity, a high-friction floor,
the conservative standing targets, and the sustainable `0.980665 N·m`
ST3215 cap.

```powershell
& C:\isaacsim\python.bat simulation\isaac\open_articulation.py `
  --world exports\isaac\quadruped_robot_manual_world.usda
```

When Isaac opens, press **Play**, open **Physics > Articulation Inspector**,
select `/World/Robot`, and move the named joints. The launcher sets the initial
stand once and then leaves control to Isaac; it does not continuously overwrite
Inspector commands.
