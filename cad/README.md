# cad

Code-driven CAD area for the Drobot/SO-101 robot inside the Drobot2 monorepo.
Its Python package, specifications, tests, vendor models, and review tooling
remain self-contained under `cad/`.

The current model set was migrated from `simple_drobot` after all local
`text-to-cad` branches were consolidated into its `develop` branch. The
fit-critical ST3215 motor bay and the edited SO-101 upper arm are preserved,
including exact vendor/reference STEP geometry and labeled fit-preview
assemblies.

## Current CAD targets

| Target | Generator | Purpose |
| --- | --- | --- |
| `exports/step/st3215_motor_bay.step` | `drobot_cad/parts/st3215_motor_bay.py` | Printable keyed rear motor bay |
| `exports/step/st3215_servo_visual.step` | `drobot_cad/parts/st3215_servo_visual.py` | Unmodified exact catalog ST3215 geometry for URDF/Isaac visuals |
| `exports/step/upper_arm.step` | `drobot_cad/parts/upper_arm.py` | Complete printable SO-101 upper arm with fused bay and integral output fork |
| `exports/step/st3215_servo_output_fork.step` | `drobot_cad/parts/st3215_servo_output_fork.py` | Reusable ST3215 output fork with a 30 mm negative-X full-edge fusion cap |
| `exports/step/st3215_hip.step` | `drobot_cad/parts/st3215_hip.py` | Printable fused hip in the approved left/down perpendicular pose |
| `exports/step/st3215_hip_body_mount.step` | `drobot_cad/parts/st3215_hip_body_mount.py` | Body-side hip fork with oversized four-bolt mounting plate |
| `exports/step/hip_orientation_preview.step` | `drobot_cad/assembly/hip_orientation_preview.py` | Two-body left/down hip pose checkpoint before printable fusion |
| `exports/step/st3215_motor_bay_fit_preview.step` | `drobot_cad/assembly/st3215_motor_bay_fit_preview.py` | Bay plus exact catalog servo |
| `exports/step/upper_arm_st3215_fit_preview.step` | `drobot_cad/assembly/upper_arm_st3215_fit_preview.py` | Complete upper arm and installed servo |
| `exports/step/robot_arm.step` | `drobot_cad/assembly/robot_arm.py` | Two complete upper arms joined by the elbow ST3215 |
| `exports/step/robot_leg.step` | `drobot_cad/assembly/robot_leg.py` | Body hip mount, perpendicular hip, three exact ST3215 servos, and two linked upper arms |
| `exports/step/rigid_fork_shoe.step` | `drobot_cad/parts/rigid_fork_shoe.py` | Shallow rigid circular shoe with a fork-wrapping saddle and recessed traction pad |
| `exports/step/rigid_fork_shoe_fit_preview.step` | `drobot_cad/assembly/rigid_fork_shoe_fit_preview.py` | Lower-leg, rigid shoe, and two diagonal M3 rod clearance envelopes |
| `exports/step/rectangular_fork_shoe.step` | `drobot_cad/parts/rectangular_fork_shoe.step.py` | 100 x 60 mm flat PLA shoe with a raised fork attachment for increased stability and clearance |
| `exports/step/rectangular_fork_shoe_fit_preview.step` | `drobot_cad/assembly/rectangular_fork_shoe_fit_preview.step.py` | Generated lower-leg preview with four M3 rods and bilateral driver-access envelopes |
| `exports/step/quadruped_body_base.step` | `drobot_cad/parts/quadruped_body.py` | One-piece X2D-safe body tub with hip reinforcement, protected battery rail, four-wall M3 grid, full floor grid, and paired wire ports |
| `exports/step/quadruped_body_lid.step` | `drobot_cad/parts/quadruped_body_lid.py` | Removable ventilated body lid with a 10 mm-pitch universal M3 grid and direct LeKiwi camera pattern |
| `exports/step/quadruped_electronics_tray.step` | `drobot_cad/parts/quadruped_electronics_tray.py` | Optional electronics tray with four floor-grid-aligned M3 mounting locations |
| `exports/step/cm5202_battery_box.step` | `drobot_cad/parts/cm5202_battery_box.step.py` | Main CM5202 LiPo box with four floor-grid M3 mounts, lid-screw towers, and one upper-left wire port |
| `exports/step/cm5202_battery_box_lid.step` | `drobot_cad/parts/cm5202_battery_box_lid.step.py` | Solid screw-on CM5202 box lid with matching wire relief |
| `exports/step/cm5202_battery_box_fit_preview.step` | `drobot_cad/assembly/cm5202_battery_box_fit_preview.step.py` | Exploded box, measured battery envelope, and lid review assembly |
| `exports/step/cm5202_battery_cradle.step` | `drobot_cad/parts/cm5202_battery_cradle.step.py` | Open CM5202 strap cradle with four floor-grid M3 mounts and wire-side clearance |
| `exports/step/raspberry_pi_5_enclosure_base.step` | `drobot_cad/parts/raspberry_pi_5_enclosure_base.step.py` | Printable Pi 5 box with exact board standoffs and four body-floor grid mounts |
| `exports/step/raspberry_pi_5_enclosure_lid.step` | `drobot_cad/parts/raspberry_pi_5_enclosure_lid.step.py` | Screw-on ventilated lid with exact BNO085 mounting pattern |
| `exports/step/raspberry_pi_5_imu_cover.step` | `drobot_cad/parts/raspberry_pi_5_imu_cover.step.py` | Open-sided removable BNO085 protection roof |
| `exports/step/raspberry_pi_5_enclosure_fit_preview.step` | `drobot_cad/assembly/raspberry_pi_5_enclosure_fit_preview.step.py` | Enclosure, exact Pi 5, lid, exact BNO085, and protection roof review assembly |
| `exports/step/lekiwi_12v_battery_reference.step` | `drobot_cad/parts/lekiwi_12v_battery_reference.py` | Measured 70 x 66 x 40 mm LeKiwi 12 V pack fit proxy |
| `exports/step/waveshare_bus_servo_adapter_a.step` | `drobot_cad/parts/waveshare_bus_servo_adapter_a.py` | Exact Waveshare USB/UART serial-bus controller reference |
| `exports/step/adafruit_bno085_stemma_qt.step` | `drobot_cad/parts/adafruit_bno085.py` | Exact Adafruit BNO085 reference centred on its sensing package |
| `exports/step/quadruped_imu_cover.step` | `drobot_cad/parts/quadruped_imu_cover.py` | Printable open-sided BNO085 roof with four aligned M2 nylon through-bolt sleeves |
| `exports/step/quadruped_imu_tray_fit_preview.step` | `drobot_cad/assembly/quadruped_imu_tray_fit_preview.py` | Installed tray, BNO085, and removable protection-cover fit preview |
| `exports/step/quadruped_body_hardware_fit_preview.step` | `drobot_cad/assembly/quadruped_body_hardware_fit_preview.py` | Lighter body, internals, lid, and camera review at full-assembly placements |
| `exports/step/quadruped_robot_fusion360.step` | `drobot_cad/assembly/quadruped_robot.py` | Primary Fusion-ready body, battery, controller, IMU, camera, and four mirrored seven-component ST3215 legs |
| `exports/step/quadruped_robot.step` | `drobot_cad/assembly/quadruped_robot.py` | Byte-identical compatibility copy of the Fusion handoff |
| `exports/step/lekiwi_camera_body_fit_preview.step` | `drobot_cad/assembly/lekiwi_camera_body_fit_preview.py` | Lid positioning preview for the upstream LeKiwi mount and Arducam envelope |

The front lid interface accepts the unchanged LeKiwi
`base_camera_mount.stl`: three M3 clearance holes at 20 mm pitch and a separate
20 x 12 mm cable opening.  The selected compatibility target is LeKiwi's
Arducam 5 MP wide-angle USB option (ASIN `B0972KK7BC`), retained under
`vendor/references/lekiwi/` with upstream license and checksums.

### Rectangular PLA fork shoe

The new rectangular fork-shoe source replaces the hollow TPU rocker with a
fully flat `100 x 60 x 6 mm` PLA contact plate. Its upper sole face is moved to
local `X=24 mm`, leaving `19.05 mm` from the nearest forward M3 hole center to
the plate. A narrow raised spine connects the attachment hub while leaving the
bottom of the fork open; the inherited outer cup was removed after it was found
to collide with the real fork. Even the closest reinforcing rib remains `10.55 mm` outside the
modeled `9 mm` nut-driver envelope. All four existing M3 positions remain
available and are recommended for the longer lever arm.

The owning design record is
[`docs/rectangular-fork-shoe.md`](docs/rectangular-fork-shoe.md), and the focused
generation entry is `scripts/generate_rectangular_fork_shoe.ps1`. The source
and STEP/STL/3MF outputs were regenerated on 2026-08-13. A focused installed
geometry check found zero solid overlap between the new shoe and fork; physical
fit, slicing, and printing remain unverified.

### Universal body-wall mounting grid and wire ports

The one-piece body base now provides M3 clearance holes on all four vertical
walls. The front and rear faces each expose 83 usable locations on a 10 x
10 mm grid after ventilation keep-outs. The left and right walls each expose
12 locations in the protected 36 mm center strip between the hip backing
plates, for 190 body-wall mounting locations in total. These are 3.4 mm
through-holes for bolts, washers, nuts, and standoffs; they are not printed
threads.

Both rounded side wire ports were enlarged from 24 x 14 mm to 32 x 20 mm.
Their centered position is unchanged, and each port retains 2 mm of material
before the neighboring hip reinforcement field. The wall-local hole cutters
do not pass into the battery rail, hip plates, ventilation slots, or lid
interface.

Editable inputs are `drobot_cad/parts/quadruped_body.py` and
`specs/quadruped-body.yaml`. Regenerate every affected committed artifact and
run the focused checks from `cad/` with:

```powershell
.\scripts\generate_cad.ps1 -Force

$cadSkillRoot = Get-ChildItem `
  "$env:USERPROFILE\.codex\plugins\cache\text-to-cad\cad" `
  -Directory | Sort-Object Name -Descending | Select-Object -First 1
$cadSkill = Join-Path $cadSkillRoot.FullName "skills\cad"
$cadpy = Join-Path $cadSkill "scripts\packages\cadpy\src"
$env:PYTHONPATH = ((Resolve-Path ".").Path, $cadpy) -join `
  [IO.Path]::PathSeparator

.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_quadruped_body.py `
  tests/test_quadruped_body_hardware_fit_preview.py `
  tests/test_quadruped_robot_assembly.py `
  tests/test_quadruped_urdf.py `
  tests/test_lekiwi_camera_body_fit_preview.py
.\.venv\Scripts\python.exe -m ruff check `
  drobot_cad/parts/quadruped_body.py `
  tests/test_quadruped_body.py
.\.venv\Scripts\python.exe "$cadSkill\scripts\inspect" refs `
  exports/step/quadruped_body_base.step `
  --facts --planes --positioning --format text
```

The current combined enclosure regression is documented with the body-floor
grid below. Wall-specific inspection still confirms 190 wall-grid cylinders
at 1.7 mm radius, 10.0 mm pitch in each local grid direction, and a 32 x 20 mm
rounded-slot envelope for each side wire port.

### Universal body-floor mounting grid

The four 56 mm-tall molded electronics-tray posts have been removed from the
body. The body floor now provides 255 new M3 clearance holes on a 10 x 10 mm
grid: 119 through the center inside the battery-retaining rail and 136 in the
outer floor bands. Together with four existing M3 and four existing M2 floor
openings, the body has 263 floor mounting locations. A 2 mm minimum web keeps
the raised battery-retaining rail itself solid and uncut.

The optional electronics tray's four M3 holes now align with floor-grid points
at `(+/-80, +/-60) mm`. Its current `Z = 56 mm` assembly placement is only a
clearance reference: no integrated supports, detachable standoff height, or
specific fasteners are selected or modeled. Choose the final tray height with
removable standoffs after the electronics stack and wire bend radii are known.

The new 3.4 mm openings are through-holes, not printed threads. Use bolts,
washers, nuts, or detachable standoffs. Because holes now pass under the
battery envelope, use a nonconductive liner and keep all bolt ends below the
battery. The grid improves mounting flexibility but reduces dust and splash
resistance; floor strength, vibration, fastener retention, and real component
fit still require a physical prototype.

Editable inputs are `drobot_cad/parts/quadruped_body.py`,
`drobot_cad/parts/quadruped_electronics_tray.py`, and
`specs/quadruped-body.yaml`. The focused regression command now covers:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_quadruped_body.py `
  tests/test_quadruped_body_hardware_fit_preview.py `
  tests/test_quadruped_robot_assembly.py `
  tests/test_quadruped_urdf.py `
  tests/test_lekiwi_body_hardware.py `
  tests/test_quadruped_imu_cover.py `
  tests/test_lekiwi_camera_body_fit_preview.py
.\.venv\Scripts\python.exe -m ruff check `
  drobot_cad/parts/quadruped_body.py `
  drobot_cad/parts/quadruped_electronics_tray.py `
  tests/test_quadruped_body.py
```

Validated on 2026-08-01: all 61 focused tests passed and Ruff passed. STEP
inspection returned one valid body part with 610 faces, 1,795 edges, and the
unchanged 220 x 170 x 96 mm envelope. Baseline comparison confirmed geometry
and topology changed while the envelope did not. The export contains 259 M3
floor cylinders at 1.7 mm radius: 255 grid locations plus four legacy M3
holes. The center hole is open, representative X and Y measurements both
returned 10.0 mm pitch, and the four tray coordinates remain members of the
floor grid. The hardware preview inspected at 2,416 faces and 6,585 edges; the
full robot inspected at 22,128 faces and 60,541 edges with its previous bounds
unchanged. Opposed isometric, top, and installed hardware snapshots were
reviewed successfully. The 107 MB full-robot review mesh closed the snapshot
page during load, so its successful STEP inspection and the lighter
exact-placement hardware preview were used for final enclosure review.

### Universal lid mounting grid

The removable lid now provides 215 usable M3 clearance locations on a 10 x
10 mm pitch across the central 180 x 120 mm mounting field. Of those, 212 are
new holes and the three existing LeKiwi camera holes also land on grid points.
The generator omits local points where a hole would leave less than 2 mm of
material around ventilation slots, cable ports, legacy M2/M3 utility holes, or
the four lid fasteners. The exterior 220 x 170 mm footprint, 4 mm top-plate
thickness, locator lip, camera interface, and body fit are unchanged.

These are 3.4 mm through-holes, not printed threads. Mount a Raspberry Pi,
fuse block, power-distribution board, sensors, or other hardware with M3 bolts,
washers, nuts, and suitable standoffs or with a small adapter plate that picks
up convenient grid points. Hole density and geometric clearance were checked;
payload strength, vibration endurance, component-specific footprints, thermal
behavior, and cable routing still require hardware-level validation.

Editable inputs are `drobot_cad/parts/quadruped_body_lid.py` and
`specs/quadruped-body.yaml`. Regenerate every affected committed artifact and
run the focused checks from `cad/` with:

```powershell
.\scripts\generate_cad.ps1 -Force

.\.venv\Scripts\python.exe -m pytest tests/test_quadruped_body.py -q
.\.venv\Scripts\python.exe -m ruff check `
  drobot_cad/parts/quadruped_body_lid.py `
  tests/test_quadruped_body.py

$cadSkill = "$env:USERPROFILE\.codex\plugins\cache\text-to-cad\cad\0.3.9\skills\cad"
.\.venv\Scripts\python.exe "$cadSkill\scripts\inspect" refs `
  exports/step/quadruped_body_lid.step --facts --planes --positioning --format text
```

Validated on 2026-07-31: the focused body suite passed all 16 tests, a 61-test
body/camera/full-assembly/URDF regression set passed, and Ruff passed. STEP
inspection returned one labeled part shape, 311 faces, 915 edges,
and unchanged 220 x 170 x 7 mm total bounds including the 3 mm underside lip.
The exported topology contains 221 cylindrical M3 faces at 1.7 mm radius (the
215 grid locations plus four lid-fastener and two rear utility holes), and
representative X/Y grid measurements both returned 10.0 mm. Lid, camera-fit,
and installed-body snapshots were reviewed successfully. The full-robot STEP
generated and inspected, but its 80 MB review mesh again closed the snapshot
page during load; the lighter exact-placement body preview remains the visual
review artifact for the enclosure. The full repository test suite was also
attempted but retains unrelated pre-existing failures in the Isaac script-map
documentation and Windows pytest temporary-directory permissions.

Generated outputs are the lid STEP/STL/3MF, the LeKiwi camera fit preview, the
body-hardware fit preview, and the Fusion/compatibility full-robot STEP files
listed above.

The body fit assembly includes the LeKiwi 12 V battery reference and a measured
envelope of the Waveshare Bus Servo Adapter (A). The exact 117-solid controller
is also exported as its own STEP. The battery proxy is 70 x 66 x 40 mm and
sits on the body floor with 1 mm side clearance and 12 mm below the tray's
current reference placement. The body no longer has molded tray supports; the
optional tray requires removable standoffs of a height selected for the final
electronics stack. The adapter sits on four separate M2 board-to-tray
standoffs at the official 37 x 28 mm pattern. LeKiwi calls this a motor control
board; electrically it is a USB/UART
half-duplex serial-bus adapter, not a CAN controller. Pack discharge current,
BMS behavior, connector bend radii, and twelve-servo electrical suitability
remain unvalidated. The proposed fused power, Raspberry Pi, and four-leg data
architecture is documented separately in
[`electrical/README.md`](../electrical/README.md).

The body-centred BNO085 now has a separate printable roof cover. Four
`2.4 mm` holes share the board and tray mounting axes, so four M2 x 20 mm
nylon through bolts and nylon nuts clamp the cover, board, and tray standoffs
as one removable stack. Integrated sleeves seat on the PCB mounting zones,
the roof clears the tallest modeled component by `3.0 mm`, and all four sides
remain open for STEMMA QT wiring and airflow. Use non-magnetic nylon hardware;
an exact M2 x 20 nylon fastener was not available in the checked step.parts
catalog, so fastener geometry is not included in the assembly.

## Review the full assembly in Fusion

Download `exports/step/quadruped_robot_fusion360.step`, then in Fusion use
**File > Open > Open from my computer**, or upload the STEP through the Data
Panel. Fusion translates the STEP into a Fusion design; expand the browser
tree to hide the body lid and inspect the battery, electronics tray,
controller, IMU cover, IMU, and camera references. The STEP contains resolved
static placements and component labels, but not the source-level parametric
history or active servo joints.

## Robot description

| Target | Purpose |
| --- | --- |
| `exports/urdf/quadruped_robot.urdf` | SI-unit description with 15 physical links, one optical frame, 12 actuated joints, camera frames, and the body-centred BNO085 frame |

The URDF generator stays with CAD because its link geometry, inertials, joint
frames, and mesh references are derived directly from the mechanical model.
Isaac import, validation, manual control, walking, stair training, trained
models, and simulator assets live in the sibling
[`simulation/`](../simulation/README.md) package.

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
2. Open a generator such as `drobot_cad/parts/st3215_motor_bay.py`.
3. Press `Ctrl+Shift+B` and run
   **CAD: Preview current Build123d file in OCP**.

The task builds the in-memory Build123d object and sends it directly to OCP CAD
Viewer. It does not export or overwrite a STEP file. The same preview can be
run from PowerShell:

```powershell
.\.venv\Scripts\python.exe .\scripts\preview_build123d.py `
  .\drobot_cad\parts\st3215_motor_bay.py
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
provenance. The [documentation index](docs/README.md) identifies the Markdown
owner that must be updated with each CAD or robot-description change.

## Layout

```text
cad/
|-- specs/                 Mechanical design contracts
|-- vendor/                Immutable catalog/reference STEP models
|-- drobot_cad/
|   |-- parts/             Printable part generators
|   |-- assembly/          Labeled, non-printable fit previews
|   `-- urdf/              CAD-derived robot-description generators
|-- scripts/               CAD generation and Viewer launch helpers
|-- tools/                 Project-owned markup patch and Viewer runtime
|-- reviews/               CAD markup and retained design intent
|-- tests/                 CAD, URDF, and provenance checks
`-- exports/               Versioned CAD and URDF deliverables
```

## Vendor-model warning

The migrated source uses the exact Waveshare Feetech **ST3215** catalog model.
No STS3212/ST3212 model was found in step.parts. Do not treat ST3215 as an exact
STS3212 substitute. If the physical robot actually uses STS3212, add its
verified manufacturer STEP and revalidate the cavity before fabrication.

## Monorepo layout

Run the commands in this document from the `cad/` directory. Simulation,
hardware control, and electrical design live in sibling directories at the
Drobot2 repository root; see the
[repository layout and ownership guide](../docs/repository-layout.md).
