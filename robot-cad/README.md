# robot-cad

Code-driven CAD area for the Drobot/SO-101 robot inside the Drobot2 monorepo.
Its Python package, specifications, tests, vendor models, and review tooling
remain self-contained under `robot-cad/`.

The current model set was migrated from `simple_drobot` after all local
`text-to-cad` branches were consolidated into its `develop` branch. The
fit-critical ST3215 motor bay and the edited SO-101 upper arm are preserved,
including exact vendor/reference STEP geometry and labeled fit-preview
assemblies.

## Current CAD targets

| Target | Generator | Purpose |
| --- | --- | --- |
| `exports/step/st3215_motor_bay.step` | `robot_cad/parts/st3215_motor_bay.py` | Printable keyed rear motor bay |
| `exports/step/upper_arm.step` | `robot_cad/parts/upper_arm.py` | Printable SO-101 upper arm with fused bay |
| `exports/step/st3215_motor_bay_fit_preview.step` | `robot_cad/assembly/st3215_motor_bay_fit_preview.py` | Bay plus exact catalog servo |
| `exports/step/upper_arm_st3215_fit_preview.step` | `robot_cad/assembly/upper_arm_st3215_fit_preview.py` | Upper arm plus installed servo |

STEP is the primary output. STL files are secondary manufacturing exports.
Everything under `exports/` is generated and intentionally ignored by Git.

## Setup

Python 3.11 or newer and Git LFS are required.

```powershell
git lfs install
git lfs pull
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

## Preview a Python generator in OCP CAD Viewer

The workspace includes an OCP CAD Viewer task for modules that expose
`gen_step()`.

1. Start **OCP CAD Viewer** from its VS Code sidebar.
2. Open a generator such as `robot_cad/parts/st3215_motor_bay.py`.
3. Press `Ctrl+Shift+B` and run
   **CAD: Preview current Build123d file in OCP**.

The task builds the in-memory Build123d object and sends it directly to OCP CAD
Viewer. It does not export or overwrite a STEP file. The same preview can be
run from PowerShell:

```powershell
.\.venv\Scripts\python.exe .\scripts\preview_build123d.py `
  .\robot_cad\parts\st3215_motor_bay.py
```

## Generate the migrated models

The generation helper discovers the installed text-to-cad CAD skill, makes its
assembly runtime available, and regenerates all four explicit targets.

```powershell
.\scripts\generate_cad.ps1 -Force
```

## Open 3D preview and markup

```powershell
.\scripts\start_cad_viewer.ps1 `
  -File exports/step/upper_arm_st3215_fit_preview.step
```

Open directly in the six-view markup workspace:

```powershell
.\scripts\start_cad_viewer.ps1 `
  -File exports/step/upper_arm.step `
  -Markup
```

The markup workspace supports front, back, top, bottom, right, and left
orthographic views. Save editable markup as JSON and review sheets as PNG under
`reviews/`. The launcher uses the markup-capable Viewer runtime committed under
`tools/cad-viewer-markup/runtime/`; it does not require another
`text-to-cad` checkout.

See [docs/cad-workflow.md](docs/cad-workflow.md) for the complete generation,
inspection, snapshot, and markup workflow. See
[docs/migration.md](docs/migration.md) for source-history and geometry
provenance.

## Layout

```text
robot-cad/
|-- specs/                 Mechanical design contracts
|-- vendor/                Immutable catalog/reference STEP models
|-- robot_cad/
|   |-- parts/             Printable part generators
|   `-- assembly/          Labeled, non-printable fit previews
|-- scripts/               CAD generation and Viewer launch helpers
|-- tools/                 Project-owned markup patch and Viewer runtime
|-- reviews/               Editable markup and retained design intent
|-- tests/                 Mechanical and provenance checks
`-- exports/               Regenerated STEP, STL, and snapshots
```

## Vendor-model warning

The migrated source uses the exact Waveshare Feetech **ST3215** catalog model.
No STS3212/ST3212 model was found in step.parts. Do not treat ST3215 as an exact
STS3212 substitute. If the physical robot actually uses STS3212, add its
verified manufacturer STEP and revalidate the cavity before fabrication.

## Monorepo layout

Run the commands in this document from the `robot-cad/` directory. Future
areas such as simulation and controls live in sibling directories at the
Drobot2 repository root.
