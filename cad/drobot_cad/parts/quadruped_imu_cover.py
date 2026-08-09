"""Removable bolt-on protection cover for the body-centred BNO085.

The cover shares the exact Adafruit board and electronics-tray four-hole M2
pattern. Integrated sleeves seat on the PCB mounting zones while an
open-sided roof protects the sensing package and connectors.

Coordinate convention:
    - origin X/Y: BNO085 sensing-package centre
    - origin Z: top surface of the BNO085 PCB
    - +X/+Y: aligned to the quadruped body axes
    - +Z: toward the protective roof and body lid
"""

from __future__ import annotations

from pathlib import Path

from build123d import Align, Cylinder, Location, Shape

from drobot_cad.parts import (
    adafruit_bno085,
    quadruped_body,
    quadruped_electronics_tray,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

M2_CLEARANCE_DIAMETER_MM = 2.4
COVER_MARGIN_X_MM = 1.5
COVER_MARGIN_Y_MM = 1.5
COVER_LENGTH_X_MM = (
    adafruit_bno085.BOARD_SIZE_XYZ_MM[0] + 2.0 * COVER_MARGIN_X_MM
)
COVER_WIDTH_Y_MM = (
    adafruit_bno085.BOARD_SIZE_XYZ_MM[1] + 2.0 * COVER_MARGIN_Y_MM
)
COVER_CENTER_XY_MM = (
    adafruit_bno085.BOARD_ENVELOPE_CENTER_FROM_SENSOR_XYZ_MM[0],
    adafruit_bno085.BOARD_ENVELOPE_CENTER_FROM_SENSOR_XYZ_MM[1],
)
COVER_CORNER_RADIUS_MM = 3.0
COVER_ROOF_THICKNESS_MM = 2.4
COMPONENT_CLEARANCE_MM = 3.0
COVER_ROOF_UNDERSIDE_Z_MM = (
    adafruit_bno085.BOARD_SIZE_XYZ_MM[2]
    - adafruit_bno085.PCB_THICKNESS_MM
    + COMPONENT_CLEARANCE_MM
)
COVER_TOTAL_HEIGHT_MM = COVER_ROOF_UNDERSIDE_Z_MM + COVER_ROOF_THICKNESS_MM
SPACER_OUTER_DIAMETER_MM = 5.0
SPACER_ROOF_OVERLAP_MM = 0.4
MOUNT_CENTERS_XY_MM = (
    quadruped_electronics_tray.IMU_MOUNT_CENTERS_XY_MM
)

# The bolt passes through the cover, PCB, tray standoff, and tray plate.
RECOMMENDED_FASTENER = "M2 x 20 mm nylon through bolt with nylon nut"


def _one_valid_solid(shape: Shape, label: str) -> Shape:
    solids = list(shape.solids())
    if len(solids) != 1 or not solids[0].is_valid:
        raise RuntimeError(
            f"{label} must be one connected valid solid; found {len(solids)} solids"
        )
    result = solids[0]
    result.label = label
    return result


def make_roof() -> Shape:
    """Return the rounded protective roof above the component envelope."""
    return quadruped_body._rounded_prism(
        COVER_LENGTH_X_MM,
        COVER_WIDTH_Y_MM,
        COVER_ROOF_THICKNESS_MM,
        minimum_z_mm=COVER_ROOF_UNDERSIDE_Z_MM,
        corner_radius_mm=COVER_CORNER_RADIUS_MM,
    ).moved(Location((*COVER_CENTER_XY_MM, 0.0)))


def make_spacer_sleeves() -> tuple[Shape, ...]:
    """Return four roof-connected sleeves that seat on the PCB."""
    height = COVER_ROOF_UNDERSIDE_Z_MM + SPACER_ROOF_OVERLAP_MM
    return tuple(
        Cylinder(
            SPACER_OUTER_DIAMETER_MM / 2.0,
            height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, 0.0)))
        for x_mm, y_mm in MOUNT_CENTERS_XY_MM
    )


def make_mount_hole_tools() -> tuple[Shape, ...]:
    """Return four overshooting M2 clearance-hole cutters."""
    cutter_height = (
        COVER_TOTAL_HEIGHT_MM
        + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    return tuple(
        Cylinder(
            M2_CLEARANCE_DIAMETER_MM / 2.0,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    x_mm,
                    y_mm,
                    -quadruped_body.BOOLEAN_OVERTRAVEL_MM,
                )
            )
        )
        for x_mm, y_mm in MOUNT_CENTERS_XY_MM
    )


def make_imu_cover() -> Shape:
    """Return the printable open-sided protective cover."""
    blank = make_roof().fuse(*make_spacer_sleeves())
    finished = blank - make_mount_hole_tools()
    return _one_valid_solid(finished, "quadruped_imu_cover")


def gen_step() -> Shape:
    """Return the STEP-ready BNO085 cover."""
    return make_imu_cover()


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "quadruped_imu_cover.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
