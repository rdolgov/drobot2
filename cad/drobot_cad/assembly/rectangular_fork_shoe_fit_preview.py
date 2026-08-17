"""Installed lower-leg preview for the rectangular PLA fork shoe.

The gray cylinders represent the four M3 threaded rods.  The purple cylinders
represent a 9 mm-diameter nut-driver approach envelope outside both ends of
the rods, where the retaining nuts are installed.  Rod and tool envelopes are
review geometry only.
"""

from __future__ import annotations

from build123d import Align, Color, Compound, Cylinder, Location, Shape

from drobot_cad.parts import rectangular_fork_shoe, upper_arm

ROD_ENVELOPE_DIAMETER_MM = 3.0
ROD_ENVELOPE_LENGTH_MM = 75.0
TOOL_APPROACH_LENGTH_MM = 12.0


def shoe_location() -> Location:
    """Place shoe-local origin on the lower leg's distal fork axis."""
    return Location(upper_arm.DISTAL_FORK_AXIS_MM)


def _styled(shape: Shape, label: str, color: Color) -> Shape:
    shape.label = label
    shape.color = color
    return shape


def make_rod_envelope(x_mm: float, y_mm: float) -> Shape:
    """Return one M3 x 75 mm threaded-rod fit envelope."""
    return Cylinder(
        ROD_ENVELOPE_DIAMETER_MM / 2.0,
        ROD_ENVELOPE_LENGTH_MM,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).moved(shoe_location() * Location((x_mm, y_mm, 0.0)))


def make_tool_approach_envelope(
    x_mm: float,
    y_mm: float,
    z_sign: float,
) -> Shape:
    """Return one external driver envelope approaching along a rod axis."""
    center_z_mm = z_sign * (
        ROD_ENVELOPE_LENGTH_MM / 2.0 + TOOL_APPROACH_LENGTH_MM / 2.0
    )
    return Cylinder(
        rectangular_fork_shoe.FASTENER_TOOL_ENVELOPE_RADIUS_MM,
        TOOL_APPROACH_LENGTH_MM,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).moved(shoe_location() * Location((x_mm, y_mm, center_z_mm)))


def make_fit_preview() -> Compound:
    """Return the leg, shoe, rods, and external tool-access envelopes."""
    children: list[Shape] = [
        _styled(
            upper_arm.gen_step(),
            "existing_distal_lower_leg",
            Color(0.22, 0.55, 0.82),
        ),
        _styled(
            rectangular_fork_shoe.make_rectangular_fork_shoe().moved(shoe_location()),
            "printable_rectangular_fork_shoe",
            Color(0.88, 0.55, 0.12),
        ),
    ]
    for index, (x_mm, y_mm) in enumerate(
        rectangular_fork_shoe.RECOMMENDED_ROD_HOLE_CENTERS_XY_MM,
        start=1,
    ):
        children.append(
            _styled(
                make_rod_envelope(x_mm, y_mm),
                f"m3x75_threaded_rod_envelope_{index}",
                Color(0.72, 0.74, 0.78),
            )
        )
        for z_sign, side_name in ((-1.0, "negative_z"), (1.0, "positive_z")):
            children.append(
                _styled(
                    make_tool_approach_envelope(x_mm, y_mm, z_sign),
                    f"m3_driver_access_{side_name}_{index}",
                    Color(0.58, 0.26, 0.76),
                )
            )

    preview = Compound(children=children)
    preview.label = "rectangular_fork_shoe_fit_preview"
    return preview
