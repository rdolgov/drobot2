# V2.3 R3 motor bay on a solid sole, for the existing lower leg

## Design brief

Keep the user's existing lower leg and distal ST3215 output fork. Replace
only its fixed shoe with a single printed motor socket and flat sole. The
new servo's output wheels attach to the existing fork using the same joint
hardware arrangement as the existing knee. The motor case is held by the
printed foot: driving the motor rotates its case and the attached foot
relative to the lower leg. No new lower leg or separate ankle cheek prints
are required, and there are no sole-to-bracket bolts or captive nuts.

This supersedes the V2.2 fused-cheek option, which still assumed the replacement
V2.1 lower leg. The V2.1 and V2.2 files are retained as previous alternatives.

Units are mm. Origin is the existing distal fork axis. +X points toward the
ground, +Y runs along the 100 mm foot length, +Z follows the axle. The existing
`upper_arm.py` and its `DISTAL_FORK_AXIS_MM` define the retained lower leg.
The ST3215 socket imports `st3215_motor_bay.gen_step()` unchanged.

The established knee-style servo placement is reused: catalog translation
(25.5, 0, 7.475) with +90 degrees about X. This places the motor-bay attachment
datum at (36.6117, 0, -1.95). The motor points down toward the sole, clearing
the existing fork root instead of occupying the leg above the ankle.

R3 retains the direct mounting introduced in R2 and removes its sole groove.
There are no posts or air gap. The motor socket joins directly to the
100 x 60 x 6 mm sole, with 0.4 mm
overlap at its attachment rim. This shortens the load path and supports a
larger portion of the socket perimeter. Increased strength is a design
expectation, not an FEA or physical-test result.

The sole is the full 6 mm thickness throughout, with flat exposed upper and
ground faces and no cable groove or through-passage. The previous groove was
intended for the added servo's power/data lead; the user requested its removal.
The imported socket cavity and stop rim stay unchanged. The solid sole closes
the passage beneath the socket's rear window. Cable routing is not prescribed
or verified by this revision.

The sole upper face is 36.2117 mm from the ankle axis; the bare contact face
is 42.2117 mm away, or 43.2117 mm with the separate 1 mm tread. This lowers
the axis-to-ground offset by 8.4 mm versus the raised V2.3 design, and adds
12.2117 mm compared with the previous fixed shoe.

The nominal print envelope is 60 x 100 x 21.6 mm. The sole lies flat on the
build plate and the motor socket opens upward. Motor-case screw accesses
come from the unchanged socket source.

## Hardware and installation

Per leg, print **one `ankle_v23_existing_leg_foot`** and use one added exact
ST3215 servo. Reuse the existing lower leg. Remove its fixed shoe and long
shoe rods; the rods must not pass through the added servo.

1. With power disconnected, seat the motor case without pinching its lead,
   and retain it with the same four case fixings used in the existing bays.
   This solid-sole revision provides no cable passage beneath the motor bay.
2. Attach the servo wheels to the existing lower-leg fork with the same
   short fork-to-servo screws and wheel hardware as the existing knee joint.
   Reuse that established assembly method, including any supplied hub or
   spacer components; do not force the fork over mismatched wheel spacing.
3. Clock the output wheel pattern to the fork's four diagonal M3 locations.
   Both use a 14 mm pitch circle; the catalog wheel is drawn with cardinal
   holes while the fork pattern is clocked 45 degrees. Retain the original
   center screws. Establish the new motor zero after physical installation.
4. Bond a roughly 94 x 54 x 1 mm traction sheet under the sole after assembly.

The source reuses the repository's established fork-to-servo datum; that is
not a new physical fit check. Actual wheel spacing, access, seating and screw
length must match the user's existing knee hardware. The reference preview
does not simulate spline indexing or claim all hardware is already aligned.

Print one fit prototype first in mm at 100% scale, sole down. PLA/PLA+, 0.2 mm
layers, 5 walls and 40% infill remain suggested, unvalidated starting settings.
The socket now rests on the sole. Its cavity overhangs
have not been assessed for support requirements.
No slicing, G-code, physical fit, strength analysis or motion check is run.

## RL relationship

The user's longer-stride training request now has a separate V2.3 R3 task:
[longer-stride model, completed results and recordings](../../simulation/docs/rl-ankle-v23-stride.md).
Both training seeds completed, but no candidate passed all walking criteria;
the best recording had qualifying front-foot strides and no qualifying rear-foot steps.
Its canonical robot description is
[`quadruped_robot_v23_solid_sole.urdf`](../exports/urdf/quadruped_robot_v23_solid_sole.urdf),
with retained-leg and foot link-frame meshes from `ankle_v23_shank_link.step.py`
and `ankle_v23_foot_link.step.py`. The printed geometry is unchanged. CAD-derived
mass estimates give 3.444019909 kg for the modeled robot; these are unmeasured.
The ankle motor case belongs to the moving foot in this new model. Standalone
URDF/mesh validation was not run; Isaac import and training are user-authorized.

Reproduce those simulation assets from the repository root:

```powershell
& ./cad/scripts/export_ankle_v23_training_meshes.ps1
$env:PYTHONPATH = (Resolve-Path cad).Path + ';C:/Users/roman/.codex/plugins/cache/text-to-cad/cad/0.4.21/skills/cad/scripts/packages/cadgen/src'
& ./cad/.venv/Scripts/python.exe cad/reviews/ankle-v23-stride/author_model.py
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v23_stride/import_assets.py
```

The model-authoring script is one-time calculation scaffolding; the written
URDF XML is canonical. Calculation inputs and the mass report are retained
under `cad/reviews/ankle-v23-stride/`. Both visual mesh exports retain the
two-skipped-faces warning; physical collisions use the documented boxes.

The completed V2.1 PPO jobs and recordings use their saved model; see the
[V2.1 results and videos](../../simulation/docs/rl-ankle-v21-planted.md).
They **do not train this mounting arrangement**: V2.3 moves the ankle motor mass onto
the foot, reverses which side of the servo is fixed, and increases the sole
offset. Its inertia, ankle command sign/calibration and contact geometry need
a separate simulation update before applying a policy. Do not deploy the
V2.1 policy to this foot based on the earlier results.

## Downloads and local review

- [STL, one foot per leg](../exports/stl/ankle_v23_existing_leg_foot.stl)
- [3MF](../exports/3mf/ankle_v23_existing_leg_foot.3mf)
- [Primary STEP](../exports/step/ankle_v23_existing_leg_foot.step)
- [Assembly STEP with retained leg and motors](../exports/step/ankle_v23_existing_leg_assembly.step)
- [Part illustration, solid-sole R3](../reviews/ankle-v23-existing-leg/one-piece-foot_20260912T220659Z.png)
- [Assembly illustration, solid-sole R3](../reviews/ankle-v23-existing-leg/existing-leg-assembly_20260912T220659Z.png)
- [Local part viewer](http://127.0.0.1:4178/?file=drobot_cad%2Fparts%2Fankle_v23_existing_leg_foot.step.py)
- [Local assembly viewer](http://127.0.0.1:4178/?file=exports%2Fstep%2Fankle_v23_existing_leg_assembly.step)

The yellow component is the only new print; the blue lower leg is retained.
Use these V2.3 files, not the older replacement-leg print kit.

**Export limitation:** generation completed, but the tessellator reported
`Warning: 2 faces have been skipped due to null triangulation` for the new
foot mesh. Mesh closure has not been checked, so the exported STL/3MF are
unverified and must not be described as confirmed watertight or ready to slice.
The static illustrations were reviewed for presentation only; they do not
resolve the tessellation warning. See the
[R3 generation log](../reviews/ankle-v23-existing-leg/generation-solid-sole.log).
STEP, STL and 3MF were regenerated successfully from the revised source;
the same two-face mesh warning remains. Earlier raised-foot and grooved-sole
illustrations and logs remain under `reviews/` as historical records.

## Source and reproduction

Source helper: `drobot_cad/parts/ankle_v23_existing_leg.py`.
Part entry: `drobot_cad/parts/ankle_v23_existing_leg_foot.step.py`.
Assembly entry: `drobot_cad/assembly/ankle_v23_existing_leg_assembly.step.py`.
Specification: `specs/ankle-v23-existing-leg-foot.yaml`.

```powershell
& ./cad/scripts/generate_ankle_v23_existing_leg.ps1

# Optional reproduction of the local illustrations (artifact production).
$env:PYTHONPATH = (Resolve-Path cad).Path
& ./cad/.venv/Scripts/python.exe C:/Users/roman/.codex/plugins/cache/text-to-cad/cad/0.4.21/skills/cad/scripts/snapshot --job cad/reviews/ankle-v23-existing-leg/render-jobs.json
```

Generation exports STEP as primary geometry and STL/3MF from the same source.
No automated geometry, mesh-closure, interference or hardware verification is
run for this edit under the user's verification rule. Static illustrations
are created to show the requested part and its attachment to the existing leg.
