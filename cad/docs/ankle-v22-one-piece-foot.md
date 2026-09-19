# V2.2 one-piece ankle foot

Superseded for the user's retained-leg request by the
[V2.3 motor-socket foot](ankle-v23-existing-leg-foot.md). This V2.2 alternative
still requires the replacement V2.1 lower leg and is not the latest print.

## Design brief

Combine the V2.1 sole and both ankle cheeks into one fused printed part.
Remove the four sole-to-cheek screw holes, head recesses and nut pockets.
Retain the eight M3 wheel attachments and two shaft-center screw accesses.
No printed spline: the foot still bolts to the actual ST3215 metal wheels.

Units are millimetres. Mechanical origin is the ankle axis; +X points toward
the ground, +Y runs along the 100 mm sole, +Z follows the axle. Preserve the
100 x 60 x 6 mm sole, its top face X=24 mm and bare contact face X=30 mm.
The wheel inner mating planes are Z=+/-18.625 mm, giving a fixed 37.25 mm
clear span from the immutable servo's measured outer wheel faces. Unlike
the earlier adjustable assembly, this has no axial adjustment slots. Nominal
face spacing is a catalog dimension, not a physically verified printed fit.

Print orientation puts the flat sole on Z=0 and the two cheeks upright.
Expected envelope is 60 x 100 x 42.5 mm in print coordinates. Each cheek has
a 7 mm thick upright and a solid 10 mm thick, 30 mm wide root flange. The
flanges overlap the sole by 0.4 mm before fusion, creating an intentional
material connection rather than three touching mesh shells.

Source helper: `drobot_cad/parts/ankle_v22_one_piece.py`. Part entry:
`drobot_cad/parts/ankle_v22_one_piece_foot.step.py`. Assembly entry:
`drobot_cad/assembly/ankle_v22_one_piece_assembly.step.py`.
The parameter ledger is `specs/ankle-v22-one-piece-foot.yaml`.

## Assembly and printing

Use **one one-piece foot per leg**, replacing one V2.1 sole and two cheeks.
Keep eight M3 x 6 wheel screws and the stock wheel-retaining center screws.
Delete four M3 x 10 sole screws and four M3 nuts per leg from the earlier BOM.
The existing V2.1 replacement lower leg and its case fixings are reused.

With power disconnected, slide the servo's stock wheels between the open
uprights and align both wheel centers. Attach each side with four M3 x 6
screws. Check the actual fixed spacing before tightening: the cheeks should
seat against both wheel faces without forcing the servo sideways. If the
printed span differs from the actual servo, adjust `WHEEL_FACE_GAP_MM` in
the new helper and regenerate; do not use the screws to force a bad fit.
The original shaft-retaining screws remain accessible through the center holes.
Fit the one-piece foot before completing the lower-leg assembly if access is
easier that way. A separate 94 x 54 x 1 mm traction sheet can be bonded below.

Print at 100% scale in mm with the sole down. The earlier PLA/PLA+, 0.2 mm,
5-wall, 40% infill settings remain unvalidated starting settings. Upright
horizontal bores may need support or cleanup; no slicer or G-code is supplied.
Print one foot for a physical fit trial before duplicating it for four legs.

## Scope and status

This is a separate physical revision. The running V2.1 PPO jobs continue with
their frozen, bolted-foot assets and source snapshots. Axis and footprint are
unchanged, but the fused material and removed hardware change mass/inertia;
those policies are not an exact validation of this one-piece revision.

No automated geometry validation, mesh-closure check, interference sweep,
slicing, strength analysis or hardware motion is run for this edit, following
the user's verification rule. Generation/export and static illustrations are
part of producing and showing the requested artifact.

The unrelated lower-leg STL's earlier two-face export warning remains open.
This new foot does not fix or replace that lower-leg mesh.
