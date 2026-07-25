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
  .\robot_cad\parts\upper_arm.py
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
