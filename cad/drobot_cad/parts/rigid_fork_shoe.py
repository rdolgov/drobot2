"""Low-profile rigid shoe with a close-fitting distal-fork saddle.

The printable structural part wraps around the rounded fork nose with a
shallow annular cup, fills the space between the fork cheeks with a central
hub, and reuses the existing four-hole M3 pattern.  Its floor-facing circular
sole includes a shallow recess for a thin adhesive traction pad.

Coordinate convention:
    - origin: distal fork revolute-axis centre
    - +X: outward from the lower leg and normal to the floor-contact face
    - +Y: inherited upper-arm transverse direction
    - +Z: along the fork axis and attachment rods
"""

from __future__ import annotations

from pathlib import Path

from build123d import Align, Axis, Cylinder, Location, Shape

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Existing distal-fork attachment interface.
FORK_ROD_PATTERN_OFFSET_MM = 4.949747
FORK_ROD_HOLE_CENTERS_XY_MM = tuple(
    (x_sign * FORK_ROD_PATTERN_OFFSET_MM, y_sign * FORK_ROD_PATTERN_OFFSET_MM)
    for x_sign in (-1.0, 1.0)
    for y_sign in (-1.0, 1.0)
)
RECOMMENDED_ROD_HOLE_CENTERS_XY_MM = (
    (-FORK_ROD_PATTERN_OFFSET_MM, -FORK_ROD_PATTERN_OFFSET_MM),
    (FORK_ROD_PATTERN_OFFSET_MM, FORK_ROD_PATTERN_OFFSET_MM),
)
M3_CLEARANCE_DIAMETER_MM = 3.4
RECOMMENDED_HARDWARE = (
    "2x M3 x 75 mm threaded rods, M3 washers, and M3 nylon lock nuts"
)

# Central hub fits between the fork cheeks.  The small pads approach the
# locally flatter M3-hole faces while preserving 0.4 mm clearance per side.
FORK_INNER_BOSS_FACE_Z_MM = 16.7
FORK_LOCAL_ROD_FACE_Z_MM = 18.2
FORK_CLEARANCE_PER_SIDE_MM = 0.4
HUB_RADIUS_MM = 11.5
HUB_HALF_WIDTH_Z_MM = FORK_INNER_BOSS_FACE_Z_MM - FORK_CLEARANCE_PER_SIDE_MM
ROD_PAD_FACE_Z_MM = FORK_LOCAL_ROD_FACE_Z_MM - FORK_CLEARANCE_PER_SIDE_MM
ROD_PAD_RADIUS_MM = 2.6

# Shallow rigid shoe.  The fork reference extends about 13.43 mm beyond its
# distal axis; the 20 mm contact face therefore adds only about 6.6 mm before
# the optional traction pad.  The annular rear rim cups the fork nose while the
# central spine occupies the existing gap between its cheeks.
SOLE_RADIUS_MM = 24.0
SOLE_THICKNESS_MM = 6.0
SOLE_BACK_X_MM = 14.0
SOLE_FACE_X_MM = SOLE_BACK_X_MM + SOLE_THICKNESS_MM
FORK_NOSE_ENVELOPE_RADIUS_MM = 13.5
FORK_WRAP_RADIAL_CLEARANCE_MM = 0.5
FORK_WRAP_INNER_RADIUS_MM = (
    FORK_NOSE_ENVELOPE_RADIUS_MM + FORK_WRAP_RADIAL_CLEARANCE_MM
)
FORK_WRAP_BACK_X_MM = 8.5
FORK_WRAP_DEPTH_MM = SOLE_BACK_X_MM - FORK_WRAP_BACK_X_MM
CENTRAL_SPINE_RADIUS_MM = HUB_RADIUS_MM
CENTRAL_SPINE_END_X_MM = SOLE_BACK_X_MM + 0.5

# A 44 mm adhesive rubber or TPU disk sits in this recess.  A 1.5 mm pad
# projects only 0.7 mm beyond the rigid sole face.
TRACTION_PAD_DIAMETER_MM = 44.0
TRACTION_PAD_RECESS_DEPTH_MM = 0.8
RECOMMENDED_TRACTION_PAD_THICKNESS_MM = 1.5
BOOLEAN_OVERTRAVEL_MM = 1.0


def _one_valid_solid(shape: Shape, label: str) -> Shape:
    solids = list(shape.solids())
    if len(solids) != 1 or not solids[0].is_valid:
        raise RuntimeError(
            f"{label} must be one connected valid solid; found {len(solids)} solids"
        )
    result = solids[0]
    result.label = label
    return result


def _x_cylinder(radius_mm: float, length_mm: float, start_x_mm: float) -> Shape:
    """Return a +X cylinder positioned by its rear face."""
    return (
        Cylinder(
            radius_mm,
            length_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        .rotate(Axis.Y, 90.0)
        .moved(Location((start_x_mm, 0.0, 0.0)))
    )


def make_attachment_hub() -> Shape:
    """Return the fork-gap hub with local anti-crush M3 pads."""
    hub = Cylinder(
        HUB_RADIUS_MM,
        2.0 * HUB_HALF_WIDTH_Z_MM,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    pad_height = ROD_PAD_FACE_Z_MM - HUB_HALF_WIDTH_Z_MM
    pads: list[Shape] = []
    for x_mm, y_mm in FORK_ROD_HOLE_CENTERS_XY_MM:
        pads.extend(
            (
                Cylinder(
                    ROD_PAD_RADIUS_MM,
                    pad_height + BOOLEAN_OVERTRAVEL_MM,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                ).moved(
                    Location(
                        (
                            x_mm,
                            y_mm,
                            HUB_HALF_WIDTH_Z_MM - BOOLEAN_OVERTRAVEL_MM,
                        )
                    )
                ),
                Cylinder(
                    ROD_PAD_RADIUS_MM,
                    pad_height + BOOLEAN_OVERTRAVEL_MM,
                    align=(Align.CENTER, Align.CENTER, Align.MAX),
                ).moved(
                    Location(
                        (
                            x_mm,
                            y_mm,
                            -HUB_HALF_WIDTH_Z_MM + BOOLEAN_OVERTRAVEL_MM,
                        )
                    )
                ),
            )
        )
    return _one_valid_solid(hub.fuse(*pads), "rigid_fork_shoe_attachment_hub")


def make_fork_saddle_and_sole() -> Shape:
    """Return the central spine, fork-wrapping cup, and shallow sole plate."""
    sole = _x_cylinder(SOLE_RADIUS_MM, SOLE_THICKNESS_MM, SOLE_BACK_X_MM)
    wrap_outer = _x_cylinder(
        SOLE_RADIUS_MM,
        FORK_WRAP_DEPTH_MM + BOOLEAN_OVERTRAVEL_MM,
        FORK_WRAP_BACK_X_MM,
    )
    wrap_inner = _x_cylinder(
        FORK_WRAP_INNER_RADIUS_MM,
        FORK_WRAP_DEPTH_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM,
        FORK_WRAP_BACK_X_MM - BOOLEAN_OVERTRAVEL_MM,
    )
    wrap_rim = wrap_outer - wrap_inner
    spine = _x_cylinder(
        CENTRAL_SPINE_RADIUS_MM,
        CENTRAL_SPINE_END_X_MM,
        0.0,
    )
    return _one_valid_solid(
        sole.fuse(wrap_rim, spine),
        "rigid_fork_shoe_saddle_and_sole",
    )


def make_rod_hole_tools() -> tuple[Shape, ...]:
    """Return four overshooting M3 clearance cutters along local Z."""
    cutter_height = 2.0 * (ROD_PAD_FACE_Z_MM + BOOLEAN_OVERTRAVEL_MM)
    return tuple(
        Cylinder(
            M3_CLEARANCE_DIAMETER_MM / 2.0,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        ).moved(Location((x_mm, y_mm, 0.0)))
        for x_mm, y_mm in FORK_ROD_HOLE_CENTERS_XY_MM
    )


def make_traction_pad_recess_tool() -> Shape:
    """Return the shallow cutter for a bonded high-friction outsole disk."""
    return _x_cylinder(
        TRACTION_PAD_DIAMETER_MM / 2.0,
        TRACTION_PAD_RECESS_DEPTH_MM + BOOLEAN_OVERTRAVEL_MM,
        SOLE_FACE_X_MM - TRACTION_PAD_RECESS_DEPTH_MM,
    )


def make_rigid_fork_shoe() -> Shape:
    """Return the printable rigid structural shoe."""
    blank = make_attachment_hub().fuse(make_fork_saddle_and_sole())
    finished = blank - (*make_rod_hole_tools(), make_traction_pad_recess_tool())
    return _one_valid_solid(finished, "rigid_fork_shoe")


def gen_step() -> Shape:
    """Return the STEP-ready rigid fork shoe."""
    return make_rigid_fork_shoe()


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "rigid_fork_shoe.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
