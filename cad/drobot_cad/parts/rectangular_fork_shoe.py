"""Rigid rectangular PLA shoe for the distal lower-leg fork.

The shoe replaces the compliant TPU rocker with a long, flat fore/aft contact
plate.  Its attachment retains the existing four-hole M3 pattern, while the
plate and its reinforcing ribs begin far enough beyond the hole axes to leave
room for washers, lock nuts, and a small nut driver.

Coordinate convention:
    - origin: distal fork revolute-axis centre
    - +X: outward from the lower leg and normal to the floor-contact face
    - +Y: fore/aft direction of the rectangular sole
    - +Z: lateral direction, along the fork axis and attachment rods
"""

from __future__ import annotations

from build123d import (
    Align,
    Axis,
    Box,
    BuildSketch,
    Cylinder,
    Location,
    Plane,
    RectangleRounded,
    Shape,
    extrude,
)

# Existing distal-fork attachment interface.
FORK_ROD_PATTERN_OFFSET_MM = 4.949747
FORK_ROD_HOLE_CENTERS_XY_MM = tuple(
    (x_sign * FORK_ROD_PATTERN_OFFSET_MM, y_sign * FORK_ROD_PATTERN_OFFSET_MM)
    for x_sign in (-1.0, 1.0)
    for y_sign in (-1.0, 1.0)
)
RECOMMENDED_ROD_HOLE_CENTERS_XY_MM = FORK_ROD_HOLE_CENTERS_XY_MM
M3_CLEARANCE_DIAMETER_MM = 3.4
RECOMMENDED_HARDWARE = (
    "4x M3 x 75 mm threaded rods, M3 washers, and M3 nylon lock nuts"
)

# The hub fits between the fork cheeks.  Local pads approach the rod faces
# while preserving the same 0.4 mm clearance used by the earlier shoes.
FORK_INNER_BOSS_FACE_Z_MM = 16.7
FORK_LOCAL_ROD_FACE_Z_MM = 18.2
FORK_CLEARANCE_PER_SIDE_MM = 0.4
HUB_RADIUS_MM = 11.5
HUB_HALF_WIDTH_Z_MM = FORK_INNER_BOSS_FACE_Z_MM - FORK_CLEARANCE_PER_SIDE_MM
ROD_PAD_FACE_Z_MM = FORK_LOCAL_ROD_FACE_Z_MM - FORK_CLEARANCE_PER_SIDE_MM
ROD_PAD_RADIUS_MM = 2.6

# Flat PLA sole.  The sole moves another 6 mm outward while the intervening
# central spine acts as a raised neck.  The external circular cup was removed,
# leaving useful open space beneath the real fork nose.
PREVIOUS_SOLE_BACK_X_MM = 18.0
ADDED_FORK_TO_SOLE_CLEARANCE_MM = 6.0
SOLE_LENGTH_FORE_AFT_MM = 100.0
SOLE_WIDTH_LATERAL_MM = 60.0
SOLE_THICKNESS_MM = 6.0
SOLE_CORNER_RADIUS_MM = 2.0
SOLE_BACK_X_MM = PREVIOUS_SOLE_BACK_X_MM + ADDED_FORK_TO_SOLE_CLEARANCE_MM
SOLE_FACE_X_MM = SOLE_BACK_X_MM + SOLE_THICKNESS_MM

# Two upper-side ribs connect the long plate back into the central spine.  The
# rib face remains outside the declared tool envelope around every rod axis.
RIB_BACK_X_MM = 20.0
RIB_OVERLAP_INTO_SOLE_MM = 0.5
RIB_LENGTH_FORE_AFT_MM = 80.0
RIB_WIDTH_LATERAL_MM = 4.0
RIB_CENTER_Z_MM = 10.0

# Installation-access contract.  A 9 mm cylinder is a conservative envelope
# for a small M3 nut driver or socket around each rod axis.
FASTENER_TOOL_ENVELOPE_DIAMETER_MM = 9.0
FASTENER_TOOL_ENVELOPE_RADIUS_MM = FASTENER_TOOL_ENVELOPE_DIAMETER_MM / 2.0
MIN_HOLE_CENTER_TO_SOLE_BACK_X_MM = (
    SOLE_BACK_X_MM - FORK_ROD_PATTERN_OFFSET_MM
)
MIN_HOLE_CENTER_TO_REINFORCEMENT_X_MM = (
    RIB_BACK_X_MM - FORK_ROD_PATTERN_OFFSET_MM
)
MIN_TOOL_ENVELOPE_TO_REINFORCEMENT_X_MM = (
    MIN_HOLE_CENTER_TO_REINFORCEMENT_X_MM
    - FASTENER_TOOL_ENVELOPE_RADIUS_MM
)

# The previous circular outer cup was removed after an installed boolean check
# showed that it intersected the real fork nose.  The hub, four rods, and the
# narrow central spine now locate the shoe while leaving the fork bottom open.
CENTRAL_SPINE_RADIUS_MM = HUB_RADIUS_MM
CENTRAL_SPINE_END_X_MM = SOLE_BACK_X_MM + RIB_OVERLAP_INTO_SOLE_MM
BOOLEAN_OVERTRAVEL_MM = 1.0

# A thin bonded tread is deliberately not structural geometry.  Keeping the
# complete PLA face flat allows rubber sheet thickness to be selected after
# the first print without creating a perimeter rocker.
RECOMMENDED_TRACTION_PAD_THICKNESS_MM = 1.0
RECOMMENDED_TRACTION_PAD_INSET_MM = 3.0


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


def _x_rounded_prism(
    length_y_mm: float,
    width_z_mm: float,
    depth_x_mm: float,
    start_x_mm: float,
    corner_radius_mm: float,
) -> Shape:
    """Extrude a rounded YZ profile in +X."""
    yz_plane = Plane(
        origin=(start_x_mm, 0.0, 0.0),
        x_dir=(0.0, 1.0, 0.0),
        z_dir=(1.0, 0.0, 0.0),
    )
    with BuildSketch(yz_plane) as profile:
        RectangleRounded(length_y_mm, width_z_mm, corner_radius_mm)
    return extrude(profile.sketch, amount=depth_x_mm)


def make_attachment_hub() -> Shape:
    """Return the fork-gap hub with four local anti-crush M3 pads."""
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
    return _one_valid_solid(hub.fuse(*pads), "rectangular_shoe_attachment_hub")


def make_rectangular_sole() -> Shape:
    """Return the full flat rectangular contact plate."""
    sole = _x_rounded_prism(
        SOLE_LENGTH_FORE_AFT_MM,
        SOLE_WIDTH_LATERAL_MM,
        SOLE_THICKNESS_MM,
        SOLE_BACK_X_MM,
        SOLE_CORNER_RADIUS_MM,
    )
    sole.label = "flat_rectangular_pla_sole"
    return sole


def make_spine_and_ribs() -> Shape:
    """Return the raised central neck and reinforced load path into the sole."""
    spine = _x_cylinder(
        CENTRAL_SPINE_RADIUS_MM,
        CENTRAL_SPINE_END_X_MM,
        0.0,
    )

    rib_depth_x_mm = (
        SOLE_BACK_X_MM - RIB_BACK_X_MM + RIB_OVERLAP_INTO_SOLE_MM
    )
    ribs = tuple(
        Box(
            rib_depth_x_mm,
            RIB_LENGTH_FORE_AFT_MM,
            RIB_WIDTH_LATERAL_MM,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        ).moved(
            Location(
                (
                    RIB_BACK_X_MM + rib_depth_x_mm / 2.0,
                    0.0,
                    z_sign * RIB_CENTER_Z_MM,
                )
            )
        )
        for z_sign in (-1.0, 1.0)
    )
    return _one_valid_solid(
        spine.fuse(*ribs),
        "rectangular_shoe_spine_and_ribs",
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


def make_rectangular_fork_shoe() -> Shape:
    """Return the printable one-piece PLA shoe."""
    blank = make_attachment_hub().fuse(
        make_rectangular_sole(),
        make_spine_and_ribs(),
    )
    finished = blank - (*make_rod_hole_tools(),)
    return _one_valid_solid(finished, "rectangular_fork_shoe")
