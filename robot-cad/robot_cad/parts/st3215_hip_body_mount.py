"""Body-side ST3215 hip mount made from the reusable motor fork and a plate.

The existing extended ST3215 servo-output fork is preserved in its local frame:

    - the fork output joint extends toward local +X
    - the rounded fusion end extends toward local -X
    - the body mounting plate lies in the YZ plane at the negative-X end

The plate is deliberately larger than the complete fork attachment edge.  Four
M4 clearance holes near its corners provide a symmetric bolt pattern for a
future robot-body interface.  The body-side plate face remains flat.
"""

from __future__ import annotations

from pathlib import Path

from build123d import Align, Axis, Box, Cylinder, Location, Shape, fillet

from robot_cad.parts import st3215_servo_output_fork

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT_EXTENSION_RADIUS_Z_MM = (
    st3215_servo_output_fork.ROOT_EXTENSION_RADIUS_Z_MM
)
ROOT_EXTENSION_WIDTH_Y_MM = st3215_servo_output_fork.ROOT_EXTENSION_WIDTH_Y_MM

# First-pass robot-body mounting interface.  The 76 mm square plate exceeds
# both the fork's 24 mm width and its 63.4 mm full oval-edge height.
MOUNTING_PLATE_WIDTH_Y_MM = 76.0
MOUNTING_PLATE_HEIGHT_Z_MM = 76.0
MOUNTING_PLATE_THICKNESS_X_MM = 6.0
MOUNTING_PLATE_CORNER_RADIUS_MM = 6.0

# The fork ends at X=-30 mm.  Extending the plate's fork-side face to X=-26 mm
# creates 4 mm of controlled overlap with the rounded fusion envelope.
MOUNTING_PLATE_FORK_SIDE_FACE_X_MM = -26.0
MOUNTING_PLATE_BODY_SIDE_FACE_X_MM = (
    MOUNTING_PLATE_FORK_SIDE_FACE_X_MM - MOUNTING_PLATE_THICKNESS_X_MM
)
MOUNTING_PLATE_CENTER_X_MM = (
    MOUNTING_PLATE_FORK_SIDE_FACE_X_MM
    + MOUNTING_PLATE_BODY_SIDE_FACE_X_MM
) / 2.0
FORK_PLATE_OVERLAP_X_MM = 4.0

# Normal-clearance M4 body bolts on a symmetric 60 x 60 mm pattern.
BODY_BOLT_NOMINAL = "M4"
BODY_BOLT_CLEARANCE_DIAMETER_MM = 4.5
BODY_BOLT_PATTERN_Y_MM = 60.0
BODY_BOLT_PATTERN_Z_MM = 60.0
BODY_BOLT_CENTER_OFFSET_Y_MM = BODY_BOLT_PATTERN_Y_MM / 2.0
BODY_BOLT_CENTER_OFFSET_Z_MM = BODY_BOLT_PATTERN_Z_MM / 2.0
BODY_BOLT_HOLE_CENTERS_YZ_MM = (
    (-BODY_BOLT_CENTER_OFFSET_Y_MM, -BODY_BOLT_CENTER_OFFSET_Z_MM),
    (-BODY_BOLT_CENTER_OFFSET_Y_MM, BODY_BOLT_CENTER_OFFSET_Z_MM),
    (BODY_BOLT_CENTER_OFFSET_Y_MM, -BODY_BOLT_CENTER_OFFSET_Z_MM),
    (BODY_BOLT_CENTER_OFFSET_Y_MM, BODY_BOLT_CENTER_OFFSET_Z_MM),
)
HOLE_CUTTER_OVERTRAVEL_MM = 1.0


def _one_valid_solid(shape: Shape, label: str) -> Shape:
    solids = list(shape.solids())
    if len(solids) != 1 or not solids[0].is_valid:
        raise RuntimeError(
            f"{label} must be one connected valid solid; found {len(solids)} solids"
        )
    result = solids[0]
    result.label = label
    return result


def make_mounting_plate_blank() -> Shape:
    """Return the centered rounded plate before drilling body-bolt holes."""
    plate = Box(
        MOUNTING_PLATE_THICKNESS_X_MM,
        MOUNTING_PLATE_WIDTH_Y_MM,
        MOUNTING_PLATE_HEIGHT_Z_MM,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).moved(Location((MOUNTING_PLATE_CENTER_X_MM, 0.0, 0.0)))
    rounded_plate = fillet(
        plate.edges().filter_by(Axis.X),
        MOUNTING_PLATE_CORNER_RADIUS_MM,
    )
    return _one_valid_solid(rounded_plate, "hip_body_mount_plate_blank")


def make_body_bolt_hole_tools() -> tuple[Shape, ...]:
    """Return four overtravel M4 through-hole cutters along local X."""
    cutter_length = (
        MOUNTING_PLATE_THICKNESS_X_MM + 2.0 * HOLE_CUTTER_OVERTRAVEL_MM
    )
    hole_radius = BODY_BOLT_CLEARANCE_DIAMETER_MM / 2.0
    return tuple(
        Cylinder(
            hole_radius,
            cutter_length,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        .rotate(Axis.Y, 90.0)
        .moved(Location((MOUNTING_PLATE_CENTER_X_MM, y_mm, z_mm)))
        for y_mm, z_mm in BODY_BOLT_HOLE_CENTERS_YZ_MM
    )


def make_mounting_plate() -> Shape:
    """Return the rounded body plate with four M4 clearance holes."""
    drilled_plate = make_mounting_plate_blank() - make_body_bolt_hole_tools()
    return _one_valid_solid(drilled_plate, "hip_body_mount_plate")


def placed_motor_fork() -> Shape:
    """Return the reusable full motor fork without reorientation."""
    return st3215_servo_output_fork.gen_step()


def gen_step() -> Shape:
    """Return the STEP-ready fork and body mounting plate as one solid."""
    fused = placed_motor_fork().fuse(make_mounting_plate())
    return _one_valid_solid(fused, "st3215_hip_body_mount")


if __name__ == "__main__":
    from build123d import export_step

    output_path = (
        PROJECT_ROOT / "exports" / "step" / "st3215_hip_body_mount.step"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
