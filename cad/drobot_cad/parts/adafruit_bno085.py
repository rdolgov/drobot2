"""Exact Adafruit BNO085 STEMMA QT breakout reference geometry.

The immutable manufacturer STEP uses a lower-left PCB origin.  This wrapper
recentres X/Y on the BNO085 sensing package while keeping the PCB bottom at
Z=0.  That makes the generated mesh convenient for both the electronics-tray
fit preview and the URDF ``imu_link`` sensor frame.

Coordinate convention:
    - origin X/Y: centre of the BNO085 package in the official CAD
    - origin Z: bottom of the PCB
    - +X/+Y: aligned to the quadruped body axes when installed
    - +Z: component side of the board
"""

from __future__ import annotations

from pathlib import Path

from build123d import Compound, Location, Shape, import_step

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_STEP = (
    PROJECT_ROOT / "vendor" / "sensors" / "adafruit_bno085_stemma_qt.step"
)

# Measured from Adafruit's official product-4754 STEP model.
BOARD_SIZE_XYZ_MM = (25.4, 22.86, 4.53)
PCB_THICKNESS_MM = 1.57
PRODUCT_MASS_KG = 0.0025
BNO085_SENSOR_CENTER_FROM_VENDOR_ORIGIN_XYZ_MM = (12.703, 12.235, 2.160)
BOARD_ENVELOPE_CENTER_FROM_SENSOR_XYZ_MM = (-0.003, -0.805, 0.105)
MOUNT_HOLE_DIAMETER_MM = 2.5
MOUNT_HOLE_CENTERS_VENDOR_XY_MM = (
    (2.54, 2.54),
    (2.54, 20.32),
    (22.86, 2.54),
    (22.86, 20.32),
)
MOUNT_HOLE_CENTERS_SENSOR_XY_MM = tuple(
    (
        x_mm - BNO085_SENSOR_CENTER_FROM_VENDOR_ORIGIN_XYZ_MM[0],
        y_mm - BNO085_SENSOR_CENTER_FROM_VENDOR_ORIGIN_XYZ_MM[1],
    )
    for x_mm, y_mm in MOUNT_HOLE_CENTERS_VENDOR_XY_MM
)


def gen_step() -> Shape:
    """Return the exact board with X/Y centred on the sensing package."""
    vendor_board = import_step(VENDOR_STEP)
    sensor_centering = Location(
        (
            -BNO085_SENSOR_CENTER_FROM_VENDOR_ORIGIN_XYZ_MM[0],
            -BNO085_SENSOR_CENTER_FROM_VENDOR_ORIGIN_XYZ_MM[1],
            0.0,
        )
    )
    # Realize the transform on each top-level imported component.  A location
    # applied only to the outer imported compound can be discarded by some
    # STEP exporters when they preserve the vendor assembly hierarchy.
    board = Compound(
        children=[
            child.moved(sensor_centering) for child in vendor_board.children
        ]
    )
    board.label = "adafruit_bno085_stemma_qt"
    return board


if __name__ == "__main__":
    from build123d import export_step

    output_path = (
        PROJECT_ROOT / "exports" / "step" / "adafruit_bno085_stemma_qt.step"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
