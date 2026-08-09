"""Compliant oval-rocker shoe for the free distal ST3215 fork.

The monolithic TPU part fits between the two distal fork cheeks and reuses
their four existing M3-pattern through-holes.  Two diagonal M3 threaded rods
are sufficient to prevent rotation; all four holes remain available for
alternate hardware layouts.

Coordinate convention:
    - origin: distal fork revolute-axis centre
    - +X: outward from the end of the lower leg toward the contact pad
    - +Y: inherited upper-arm transverse direction
    - +Z: along the fork axis and attachment rods
"""

from __future__ import annotations

from pathlib import Path

from build123d import (
    Align,
    Axis,
    BuildSketch,
    Cylinder,
    Ellipse,
    Location,
    Mode,
    Plane,
    Rectangle,
    Shape,
    revolve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Measured from the immutable SO-101 distal fork geometry.  The four axes form
# a 9.899494 mm square around the nominal ST3215 output axis.
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

M3_TPU_CLEARANCE_DIAMETER_MM = 3.4
RECOMMENDED_HARDWARE = (
    "2x M3 x 75 mm threaded rods, M3 washers, and M3 nylon lock nuts"
)

# The closest inward fork bosses terminate at Z=+/-16.7 mm.  The main hub
# leaves 0.4 mm clearance per side.  Small pads around the M3 axes reach the
# flatter local cheek faces at Z=+/-18.2 mm while preserving the same gap.
FORK_INNER_BOSS_FACE_Z_MM = 16.7
FORK_LOCAL_ROD_FACE_Z_MM = 18.2
FORK_CLEARANCE_PER_SIDE_MM = 0.4
HUB_RADIUS_MM = 11.5
HUB_HALF_WIDTH_Z_MM = FORK_INNER_BOSS_FACE_Z_MM - FORK_CLEARANCE_PER_SIDE_MM
ROD_PAD_FACE_Z_MM = FORK_LOCAL_ROD_FACE_Z_MM - FORK_CLEARANCE_PER_SIDE_MM
ROD_PAD_RADIUS_MM = 2.6

# The load core runs from the attachment hub into the hollow contact body
# without extending to the compliant outer nose.
LOAD_CORE_RADIUS_MM = 8.0
LOAD_CORE_END_X_MM = 48.0

CONTACT_CENTER_X_MM = 42.5
CONTACT_OUTER_AXIAL_RADIUS_X_MM = 30.0
CONTACT_OUTER_RADIAL_RADIUS_MM = 24.0
SHELL_WALL_MM = 4.0
CONTACT_INNER_AXIAL_RADIUS_X_MM = (
    CONTACT_OUTER_AXIAL_RADIUS_X_MM - SHELL_WALL_MM
)
CONTACT_INNER_RADIAL_RADIUS_MM = (
    CONTACT_OUTER_RADIAL_RADIUS_MM - SHELL_WALL_MM
)
VENT_RADIUS_MM = 5.5
VENT_OFFSET_Y_MM = 11.0
VENT_OFFSET_Z_MM = 11.0
BOOLEAN_OVERTRAVEL_MM = 2.0


def _one_valid_solid(shape: Shape, label: str) -> Shape:
    solids = list(shape.solids())
    if len(solids) != 1 or not solids[0].is_valid:
        raise RuntimeError(
            f"{label} must be one connected valid solid; found {len(solids)} solids"
        )
    result = solids[0]
    result.label = label
    return result


def make_attachment_hub() -> Shape:
    """Return the fork-fitting hub with local anti-crush rod pads."""
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
                ).moved(Location((x_mm, y_mm, HUB_HALF_WIDTH_Z_MM - BOOLEAN_OVERTRAVEL_MM))),
                Cylinder(
                    ROD_PAD_RADIUS_MM,
                    pad_height + BOOLEAN_OVERTRAVEL_MM,
                    align=(Align.CENTER, Align.CENTER, Align.MAX),
                ).moved(Location((x_mm, y_mm, -HUB_HALF_WIDTH_Z_MM + BOOLEAN_OVERTRAVEL_MM))),
            )
        )
    return _one_valid_solid(hub.fuse(*pads), "tpu_fork_shoe_attachment_hub")


def _make_ellipsoid(axial_radius_x_mm: float, radial_radius_mm: float) -> Shape:
    """Revolve a half ellipse into an analytic, STEP-stable ellipsoid."""
    profile_margin_mm = 1.0
    with BuildSketch(Plane.XY) as half_profile:
        Ellipse(axial_radius_x_mm, radial_radius_mm)
        Rectangle(
            2.0 * (axial_radius_x_mm + profile_margin_mm),
            radial_radius_mm + profile_margin_mm,
            align=(Align.CENTER, Align.MIN),
            mode=Mode.INTERSECT,
        )
    return revolve(half_profile.sketch, axis=Axis.X).moved(
        Location((CONTACT_CENTER_X_MM, 0.0, 0.0))
    )


def make_ball_shell() -> Shape:
    """Return the hollow, vented ellipsoidal rocker contact body."""
    outer = _make_ellipsoid(
        CONTACT_OUTER_AXIAL_RADIUS_X_MM,
        CONTACT_OUTER_RADIAL_RADIUS_MM,
    )
    inner = _make_ellipsoid(
        CONTACT_INNER_AXIAL_RADIUS_X_MM,
        CONTACT_INNER_RADIAL_RADIUS_MM,
    )
    shell = outer - inner

    vent_length = (
        2.0
        * max(
            CONTACT_OUTER_AXIAL_RADIUS_X_MM,
            CONTACT_OUTER_RADIAL_RADIUS_MM,
        )
        + 2.0 * BOOLEAN_OVERTRAVEL_MM
    )
    vents: list[Shape] = []
    for z_mm in (-VENT_OFFSET_Z_MM, VENT_OFFSET_Z_MM):
        vents.append(
            Cylinder(
                VENT_RADIUS_MM,
                vent_length,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
            )
            .rotate(Axis.X, 90.0)
            .moved(Location((CONTACT_CENTER_X_MM, 0.0, z_mm)))
        )
    for y_mm in (-VENT_OFFSET_Y_MM, VENT_OFFSET_Y_MM):
        vents.append(
            Cylinder(
                VENT_RADIUS_MM,
                vent_length,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
            ).moved(Location((CONTACT_CENTER_X_MM, y_mm, 0.0)))
        )
    return _one_valid_solid(shell - tuple(vents), "tpu_fork_shoe_ball_shell")


def make_load_core() -> Shape:
    """Return the axial load path from the fork hub to the contact pad."""
    return Cylinder(
        LOAD_CORE_RADIUS_MM,
        LOAD_CORE_END_X_MM,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    ).rotate(Axis.Y, 90.0)


def make_rod_hole_tools() -> tuple[Shape, ...]:
    """Return four overshooting M3 clearance cutters along local Z."""
    cutter_height = 2.0 * (
        ROD_PAD_FACE_Z_MM + BOOLEAN_OVERTRAVEL_MM
    )
    return tuple(
        Cylinder(
            M3_TPU_CLEARANCE_DIAMETER_MM / 2.0,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        ).moved(Location((x_mm, y_mm, 0.0)))
        for x_mm, y_mm in FORK_ROD_HOLE_CENTERS_XY_MM
    )


def make_tpu_fork_shoe() -> Shape:
    """Return the printable one-piece TPU shoe."""
    blank = make_attachment_hub().fuse(make_load_core(), make_ball_shell())
    finished = blank - make_rod_hole_tools()
    return _one_valid_solid(finished, "tpu_fork_shoe")


def gen_step() -> Shape:
    """Return the STEP-ready TPU fork shoe."""
    return make_tpu_fork_shoe()


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "tpu_fork_shoe.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
