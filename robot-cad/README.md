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
| `exports/step/st3215_servo_visual.step` | `robot_cad/parts/st3215_servo_visual.py` | Unmodified exact catalog ST3215 geometry for URDF/Isaac visuals |
| `exports/step/upper_arm.step` | `robot_cad/parts/upper_arm.py` | Complete printable SO-101 upper arm with fused bay and integral output fork |
| `exports/step/st3215_servo_output_fork.step` | `robot_cad/parts/st3215_servo_output_fork.py` | Reusable ST3215 output fork with a 30 mm negative-X full-edge fusion cap |
| `exports/step/st3215_hip.step` | `robot_cad/parts/st3215_hip.py` | Printable fused hip in the approved left/down perpendicular pose |
| `exports/step/st3215_hip_body_mount.step` | `robot_cad/parts/st3215_hip_body_mount.py` | Body-side hip fork with oversized four-bolt mounting plate |
| `exports/step/hip_orientation_preview.step` | `robot_cad/assembly/hip_orientation_preview.py` | Two-body left/down hip pose checkpoint before printable fusion |
| `exports/step/st3215_motor_bay_fit_preview.step` | `robot_cad/assembly/st3215_motor_bay_fit_preview.py` | Bay plus exact catalog servo |
| `exports/step/upper_arm_st3215_fit_preview.step` | `robot_cad/assembly/upper_arm_st3215_fit_preview.py` | Complete upper arm and installed servo |
| `exports/step/robot_arm.step` | `robot_cad/assembly/robot_arm.py` | Two complete upper arms joined by the elbow ST3215 |
| `exports/step/robot_leg.step` | `robot_cad/assembly/robot_leg.py` | Body hip mount, perpendicular hip, three exact ST3215 servos, and two linked upper arms |
| `exports/step/quadruped_body_base.step` | `robot_cad/parts/quadruped_body.py` | One-piece X2D-safe body tub with hip reinforcement and battery bay |
| `exports/step/quadruped_body_lid.step` | `robot_cad/parts/quadruped_body_lid.py` | Removable ventilated body lid with direct LeKiwi camera-mount pattern |
| `exports/step/quadruped_electronics_tray.step` | `robot_cad/parts/quadruped_electronics_tray.py` | Removable electronics tray above the battery |
| `exports/step/quadruped_robot.step` | `robot_cad/assembly/quadruped_robot.py` | Complete body with four mirrored seven-component ST3215 legs |
| `exports/step/lekiwi_camera_body_fit_preview.step` | `robot_cad/assembly/lekiwi_camera_body_fit_preview.py` | Lid positioning preview for the upstream LeKiwi mount and Arducam envelope |

The front lid interface accepts the unchanged LeKiwi
`base_camera_mount.stl`: three M3 clearance holes at 20 mm pitch and a separate
20 x 12 mm cable opening.  The selected compatibility target is LeKiwi's
Arducam 5 MP wide-angle USB option (ASIN `B0972KK7BC`), retained under
`vendor/references/lekiwi/` with upstream license and checksums.

## Robot description and Isaac articulation

| Target | Purpose |
| --- | --- |
| `exports/urdf/quadruped_robot.urdf` | SI-unit 13-link/12-joint robot description using current printable meshes and exact ST3215 visual geometry |
| `exports/isaac/quadruped_robot_fixed.usdc` | Self-contained fixed-base articulation for safe joint commissioning |
| `exports/isaac/quadruped_robot_floating.usdc` | Self-contained floating-base articulation with guarded inter-leg collision |
| `exports/isaac/quadruped_robot_manual_world.usda` | Portable Isaac world referencing the self-collision-enabled floating asset, with Earth gravity, floor contact, rated-torque caps, and standing targets |

For manual control, launch the prepared world:

```powershell
& C:\isaacsim\python.bat simulation\isaac\open_articulation.py `
  --world exports\isaac\quadruped_robot_manual_world.usda
```

Isaac opens the robot at `/World/Robot`. Press **Play**, open
**Physics > Articulation Inspector**, select `/World/Robot`, and command the
12 joints by name. The launcher applies the sustainable ST3215 rated cap
(`0.980665 N·m`) once; the imported hard limit remains the verified
`2.941995 N·m` stall torque.

STEP is the primary output. STL, 3MF, and intentional GLB files are secondary
manufacturing or interchange exports. Commit these 3D deliverables and regenerate
every affected file whenever its Python generator or YAML specification changes.
Review-only renders, snapshots, Viewer caches, and temporary files stay local and
are ignored by Git.

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
assembly runtime available, and regenerates every explicit target above.

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
|   |-- assembly/          Labeled, non-printable fit previews
|   `-- urdf/              Generated robot-description source
|-- scripts/               CAD generation and Viewer launch helpers
|-- simulation/isaac/      Import, validation, gait, and manual-control runners
|-- tools/                 Project-owned markup patch and Viewer runtime
|-- reviews/               Editable markup and retained design intent
|-- tests/                 Mechanical and provenance checks
`-- exports/               Versioned 3D deliverables; local review output
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
