# Printable ankle V2.1 prototype

**Latest print option:** the [V2.3 motor-socket foot](ankle-v23-existing-leg-foot.md)
is a single print that retains the user's original lower leg. Use it instead
of this replacement-leg kit. This page retains the earlier bolted design used
by the currently running RL model.

## Design brief

Create a removable, motorized ankle kit for the existing robot: one replacement
lower leg, one flat sole, and two identical adjustable horn cheeks per leg.
Use the exact ST3215 already used by the robot. Keep the 159.896689 mm nominal
knee-to-ankle distance and the 100 x 60 x 6 mm sole with its upper face 24 mm
below the ankle axis. This is a physical prototype revision of the earlier
box-envelope concept, not the geometry used by the completed PPO pilot.

The lower leg imports the existing fit-critical `st3215_motor_bay.gen_step()`
unchanged at both ends. A hollow, windowed spine joins the sockets. The knee
socket retains its existing mounting pose. The ankle motor points toward the
knee and stays fixed to the new lower leg. Its supplied metal wheels drive the
two foot cheeks; no printed spline or new bearing purchase is required.

The cheeks bolt independently to slotted holes in the sole, allowing their
axial spacing to be adjusted to the actual servo wheel faces without squeezing
the gearbox. Captive M3 nuts and recessed underside screws retain the cheeks.
The joint's centered shaft screw remains accessible through each cheek.

Units are mm. The assembly origin is the ankle axis, +X points down/outward,
+Y follows the foot's 100 mm direction, and +Z follows the ankle axle.
Print entry generators rotate parts into build-plate orientations; the
assembly source uses their original mechanical frames.

## Source references and assumptions

- The existing ST3215 bay source and `specs/st3215-motor-bay.yaml` own all
  motor-body cavity allowances and case screw locations.
- `UPPER_ARM_INTERFACES` owns the current knee/distal axis spacing.
- The immutable ST3215 catalog STEP has four 2.5 mm unthreaded representations
  on a 14 mm pitch circle on each metal wheel; these represent the M3 threaded
  mounting holes. Use the supplied wheels, not bare splines.
- Read-only reference extraction is saved in
  `reviews/ankle-v2-printable/reference-interface-facts.json`. The catalog wheel
  outer faces are at local Y=9.6 and Y=-27.65 mm. Their midpoint determines the
  new ankle's axial centering. The cheeks start 0.2 mm beyond each outer face;
  the sole slots allow that allowance to be removed during assembly.
- The current servo STEP record was queried again from
  [step.parts](https://api.step.parts/v1/parts/waveshare_feetech_st3215_servo).
  Its SHA-256 is unchanged from `vendor/README.md`. The manufacturer describes
  the dual-shaft construction and aluminum servo wheels on its
  [ST3215 product page](https://www.waveshare.com/product/st3215-servo.htm).
- PLA/PLA+ is the initial structural material, matching the existing shoe.
  A 1 mm bonded traction sheet is separate from the printed sole.

## Files to print

Editable helper: `drobot_cad/parts/ankle_v2_printable.py`.
The [parameter ledger](../specs/ankle-v2-printable.yaml) documents the interfaces.

| Printed part | One leg | Complete robot | 3MF | STL | Editable STEP source |
| --- | ---: | ---: | --- | --- | --- |
| Replacement lower leg | 1 | 4 | [3MF](../exports/3mf/ankle_v2_lower_leg.3mf) | [STL](../exports/stl/ankle_v2_lower_leg.stl) | [source](../drobot_cad/parts/ankle_v2_lower_leg.step.py) |
| Flat sole | 1 | 4 | [3MF](../exports/3mf/ankle_v2_sole.3mf) | [STL](../exports/stl/ankle_v2_sole.stl) | [source](../drobot_cad/parts/ankle_v2_sole.step.py) |
| Identical horn cheek | 2 | 8 | [3MF](../exports/3mf/ankle_v2_cheek.3mf) | [STL](../exports/stl/ankle_v2_cheek.stl) | [source](../drobot_cad/parts/ankle_v2_cheek.step.py) |

Primary STEP files share those names under `exports/step/`. The
[one-leg assembly STEP](../exports/step/ankle_v2_printable_assembly.step)
shows the existing knee motor, added ankle motor and four printed pieces.
It is an assembly reference, not a combined print object. Fasteners and the
bonded traction sheet are omitted from that preview.

## Hardware

| Item | One leg | Complete robot |
| --- | ---: | ---: |
| Additional ST3215, with its two metal wheels and original shaft-retaining screws | 1 | 4 |
| M3 x 6 ISO 4762 socket-head screws, cheek to metal wheel | 8 | 32 |
| M3 x 10 ISO 7380 button-head screws, underside of sole to cheek | 4 | 16 |
| M3 ISO 4032 hex nuts for the accessible cheek pockets | 4 | 16 |
| Case screws of the same type/length used in the existing ST3215 bays | 4 for the added motor; reuse 4 knee fixings | 16 added; reuse 16 knee fixings |
| Approximately 94 x 54 x 1 mm bonded rubber/TPU tread | 1 | 4 |

Standard-fastener catalog selections are recorded in
`reviews/ankle-v2-printable/selected-hardware.json`. The motor's own center
screws retain its metal wheels; the eight M3 x 6 screws attach the printed
cheeks to those wheels. Do not substitute long screws through the motor.
The nominal wheel-screw engagement is 2.0 mm after adjusting the cheek flush
to the wheel. The source wheel holes are about 2.2-2.5 mm deep, so confirm
actual screw length and wheel depth when assembling the physical prototype.

## Print and assembly sequence

First print **one cheek** to check the wheel interface. Then complete one leg
kit with a second cheek, one sole and one lower leg after resolving the
lower-leg export warning described below.
The supplied print entries place parts on Z=0 in the intended orientations:
sole contact face down, cheek flat inner face down, and lower leg on its broad
side. Suggested starting settings are PLA/PLA+, 0.2 mm layers, 5 walls and 40%
infill. These are unvalidated starting settings. The lower-leg sockets may
need local support; keep support material out of fitted grooves and screw
accesses. No G-code or slicer profile is supplied.

1. With power disconnected, fit one cheek to an actual supplied metal wheel.
   Check all four M3 holes and access to its center-retaining screw before
   committing to the rest of the kit. The metal wheel carries the spline.
2. Transfer the existing knee motor into the matching socket of the new lower
   leg. Its socket pose is unchanged relative to the existing knee interface.
   Seat the added ankle motor in the reversed socket at the other end.
3. Reuse the established bay case fixings. Route the cable through the socket
   windows and open spine, securing it to a rail with a loose strain-relief loop.
   Keep it away from the moving wheel and sole.
4. Insert two M3 nuts per cheek through the exposed outer-face loading slots.
   Orient each nut with flats along the slot walls and seat it at the bottom
   of its hex pocket. These are open loading channels, not enclosed nut cavities.
5. Attach each cheek to its metal wheel using four M3 x 6 screws. Retain the
   original shaft-center screws. Both cheeks use the same print; turn one over
   for the opposite side.
6. Install the four M3 x 10 button-head screws through the underside recesses
   of the sole into the cheek nuts. Leave these loose while adjusting the
   cheek spacing in the slots. Let the cheeks sit flush to the wheel faces;
   do not pull the servo sideways by tightening an incorrectly spaced U-frame.
7. Tighten the sole fixings, then bond the traction sheet only after access to
   the recessed heads is no longer needed. The bare plate is 30 mm from the
   ankle axis; the nominal tread increases this to 31 mm.

The assembly is shown at zero ankle pitch. There are **no designed mechanical
hard stops or qualified travel limits**. The +/-35-degree source joint range
is a concept annotation, not a tested operating range. Physical clearance,
motor zero, wiring and 16-motor control must be established before powered
locomotion. This CAD task does not operate the robot or install a policy.

## Reproduction and status

The subsequently requested [V2.1 flat-stance RL training](../../simulation/docs/rl-ankle-v21-planted.md)
uses the actual printable geometry in a new canonical
[`quadruped_robot_v21_printable.urdf`](../exports/urdf/quadruped_robot_v21_printable.urdf).
The two `ankle_v21_*_link.step.py` entries provide simulation visuals in link
frames, preserving the separate print-oriented parts above. CAD-native mass
properties use a documented, unmeasured effective density. Training and
recording are authorized; they do not resolve the printing warning or qualify
physical motion. No visible printed geometry changed in this integration, so
the reviewed part illustrations above remain current.

[Download the local prototype package](../exports/ankle_v2_print_kit.zip).
It contains STEP, STL and 3MF files, this assembly guide and the parameter
ledger. Its lower-leg mesh has the unresolved warning described below.

[Open the assembly in CAD Viewer](http://127.0.0.1:4178/?dir=C%3A%5CUsers%5Croman%5CDocuments%5Cdev%5Cdrobot2%5Ccad&file=drobot_cad%2Fassembly%2Fankle_v2_printable_assembly.step.py).
This URL works only on this machine; no remote Viewer is published.
Individual local views: [lower leg](http://127.0.0.1:4178/?file=drobot_cad%2Fparts%2Fankle_v2_lower_leg.step.py),
[sole](http://127.0.0.1:4178/?file=drobot_cad%2Fparts%2Fankle_v2_sole.step.py),
[cheek](http://127.0.0.1:4178/?file=drobot_cad%2Fparts%2Fankle_v2_cheek.step.py).

Saved CAD illustrations of the current geometry were reviewed for the handoff:
[assembled leg](../reviews/ankle-v2-printable/assembly-upright_20260912T190139Z.png),
[lower leg](../reviews/ankle-v2-printable/ankle_v2_lower_leg_20260912T190139Z.png),
[sole](../reviews/ankle-v2-printable/ankle_v2_sole_20260912T190139Z.png),
[cheek with accessible nut slots](../reviews/ankle-v2-printable/ankle_v2_cheek_20260912T190139Z.png).
These images do not establish mesh closure, fit or structural adequacy.

From the repository root:

```powershell
& ./cad/scripts/generate_ankle_v2_printable.ps1
```

The review illustrations were created with the CAD skill's `scripts/snapshot
--job cad/reviews/ankle-v2-printable/render-jobs.json`; this file records the
camera and output settings. Generation and rendering logs are in the same
review directory.

This generates the three print parts through the CAD skill's STEP workflow,
then writes STL and 3MF from the same generators and creates the assembly STEP.
The motor bay and immutable vendor files are unchanged. The original V2 URDFs,
simulation assets, and trained policies are also unchanged; this replacement
leg has different material distribution. The new V2.1 simulation is described
above; original pilot results do not predict this revision's performance.

The initial lower-leg STL export reported **two faces skipped due to null
triangulation**. A finer export also reported the warning. A targeted mesh-closure
check has been requested under the user's explicit-verification rule and is
pending. Do not treat the lower-leg mesh as cleared for printing until that
warning has been resolved. The sole and cheek exports reported no such warning.

Generation is authorized by the request for printable parts. Separate
geometry validation, interference sweeps, DfAM checks, slicing, simulation,
training, and hardware checks are not requested and will not be run. The
first print remains a fit prototype; no strength or running claim is made.
