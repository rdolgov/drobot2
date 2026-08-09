# CAD generation, preview, and markup workflow

## Tooling contract

This repository is designed to use the same installed text-to-cad skills as the
source project:

- `cad:cad` generates, inspects, measures, compares, and snapshots STEP-first
  geometry.
- `cad:cad-viewer` opens the local 3D review application.
- `cad:step-parts` resolves exact purchasable components before placeholders
  are considered.

The CAD generation helper discovers the installed CAD skill. The Viewer helper
uses the repository-owned runtime under `tools/cad-viewer-markup/runtime/`, so
3D review and markup do not depend on a neighboring `text-to-cad` checkout.
Set `ROBOT_CAD_SKILL_ROOT` only when CAD generation cannot find the installed
skill.

## Live Build123d preview

Open a Python generator that defines `gen_step()`, start OCP CAD Viewer from
its VS Code sidebar, and press `Ctrl+Shift+B`. The default task,
**CAD: Preview current Build123d file in OCP**, imports the selected module,
builds its in-memory object, and sends it to the viewer on OCP's configured
port.

The equivalent command-line workflow is:

```powershell
.\.venv\Scripts\python.exe .\scripts\preview_build123d.py `
  .\drobot_cad\parts\upper_arm.py
```

Preview is deliberately independent of the validated STEP workflow below. It
does not run the module's `__main__` block and therefore does not rewrite
anything under `exports/`.

## Generate

```powershell
.\scripts\generate_cad.ps1 -Force
```

This regenerates the standalone ST3215 bay, fused upper arm, two fit-preview
assemblies, and secondary STL files for the two printable parts. It also makes
the skill-local `cadpy` assembly package available without adding it as a
published project dependency.

## Validate

For every visibly changed STEP:

1. Generate the explicit Python target.
2. Inspect refs with facts, major planes, and positioning.
3. Measure the changed fit, clearance, hole, or mating datum.
4. Create a snapshot and visually review it.
5. Open the artifact in CAD Viewer.

The current migrated geometry baselines are recorded in
`specs/st3215-motor-bay.yaml` and `specs/upper-arm.yaml`. The automated tests
also verify connected-solid counts, bounds, vendor hashes, and specification
values.

## 3D review

Open the complete upper-arm fit preview:

```powershell
.\scripts\start_cad_viewer.ps1 `
  -File exports/step/upper_arm_st3215_fit_preview.step
```

The script prints a reusable local review URL. It launches the committed
project runtime and reuses only a running Viewer with the same
`drobot2-markup` runtime identity. Older installed Viewer releases and
neighboring source checkouts are not used.

## Fusion 360 handoff

`exports/step/quadruped_robot_fusion360.step` is the single-file full-assembly
handoff. `exports/step/quadruped_robot.step` is kept as a byte-identical
compatibility copy for existing Viewer and simulator links. Its editable
source of truth is
`drobot_cad/assembly/quadruped_robot.py`; battery, controller, camera, IMU, body,
and leg placements remain authored in the Python generators and specifications.

Regenerate the Fusion handoff explicitly:

```powershell
$cadSkill = "$env:USERPROFILE\.codex\plugins\cache\text-to-cad\cad\0.3.9\skills\cad"
$env:PYTHONPATH = "$PWD;$cadSkill\scripts\packages\cadpy\src"

.\.venv\Scripts\python.exe "$cadSkill\scripts\step" `
  drobot_cad/assembly/quadruped_robot.py=exports/step/quadruped_robot_fusion360.step `
  --force

Copy-Item `
  exports/step/quadruped_robot_fusion360.step `
  exports/step/quadruped_robot.step `
  -Force
```

In Autodesk Fusion, use **File > Open > Open from my computer**, or upload the
STEP from the Data Panel. Large STEP files can also be uploaded through the
Fusion Team Hub. Fusion converts the interchange file into a native design.
The exported hierarchy is intended for visibility and fit review: it carries
labels and resolved occurrence positions, but not Build123d feature history,
source joints, servo articulation, or electrical validation.

The current body-internal review includes:

- LeKiwi 12 V battery reference proxy, measured from the upstream 5/5.2 Ah
  mesh as 70 x 66 x 40 mm in the installed orientation.
- Waveshare Bus Servo Adapter (A) on four M2 standoffs at 37 x 28 mm pitch.
  The exact 117-solid board is a standalone STEP; the full robot uses its
  measured 42.024 x 33.816 x 14.600 mm envelope because nesting the detailed
  vendor assembly makes the XCAF writer fail. This is USB/UART serial bus,
  not CAN.
- Exact Adafruit BNO085 reference.
- LeKiwi camera mount and Arducam reference proxies.

Fit validation is geometric. The current battery's BMS/discharge capability,
the selected cable and connector bend envelopes, heat, EMI, and the electrical
load of twelve ST3215 servos still require hardware-specific engineering.

### Current Fusion handoff validation

Validated on 2026-07-26:

- `quadruped_robot_fusion360.step` was generated through `scripts/step` and
  copied byte-for-byte to `quadruped_robot.step`; SHA-256
  `e48ea77c1e2fd4964fd5a895dfb72a34de31e76148d189824785670a8e393e8d`.
- Direct STEP re-import returned 174 solids, eleven labeled top-level
  components, and bounds 519.235 x 635.397 x 432.609 mm. The CAD Viewer GLB
  sidecar reports larger nested bounds, so Fusion/STEP re-import is the
  authoritative interchange check.
- Battery bounds are 70 x 66 x 40 mm at Z 4 to 44 mm: 1 mm side clearance in
  the 68 mm bay and 12 mm below the tray.
- The controller fit envelope is 42.024 x 33.816 x 14.600 mm at Z 59.4 to
  74.0 mm. It has 0.4 mm underside clearance above the tray, 19 mm below the
  lid locator, and more than 14 mm X clearance from the body-centred IMU.
- Full `pytest -q` passed. Ruff passed for every Python file changed by this
  handoff; the committed Viewer runtime retains unrelated pre-existing lint
  findings when the entire repository is linted.
- The mandatory full-robot `scripts/snapshot` attempt closed its browser page
  while loading the 80 MB review mesh. A lighter exact-placement
  `quadruped_body_hardware_fit_preview.step` was therefore generated and its
  full-body, open-electronics, and battery-bay snapshots were reviewed.

## Orthographic markup

```powershell
.\scripts\start_cad_viewer.ps1 `
  -File exports/step/upper_arm.step `
  -Markup
```

In CAD Viewer, choose **Orthographic markup** from the floating toolbar if the
workspace is not already open. The workspace provides front, back, top,
bottom, right, and left views.

The exact source patch and the ready-to-run Viewer runtime are retained under
`tools/cad-viewer-markup/`. The patch remains the editable source record; the
runtime makes the feature immediately usable from a standalone clone.

Use these established colors:

- Red: remove material.
- Green: add material.
- Blue: move or reposition.
- Purple: hardware or vendor-part concern.
- Amber: note, question, or clarification.

Export editable markup JSON and the annotated PNG view to `reviews/`. Keep the
JSON whenever a future edit must remain adjustable. A retained source-project
review sheet is available at
`reviews/legacy/upper_arm_change_markup_sheet.png`.

## Adding vendor parts

Search step.parts by exact model number and common aliases. If an exact match is
found, store the immutable STEP under `vendor/` and record its catalog page,
retrieval date, and SHA-256 in `vendor/README.md`.

If no exact model exists, record the miss. A simplified envelope may only be
used when clearly labeled as an approximation. Never use an adjacent model
number as an exact fit reference.
