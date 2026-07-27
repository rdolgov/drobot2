"""Exact Waveshare Bus Servo Adapter (A) reference geometry.

This is the USB/UART half-duplex serial-bus controller used by the standard
Feetech LeKiwi setup.  It is not a CAN-bus controller.

Coordinate convention:
    - origin X/Y: nominal 42 x 33 mm PCB center
    - origin Z: vendor PCB mounting datum
    - +Z: component side
"""

from __future__ import annotations

from pathlib import Path

from build123d import Compound, Location, Shape, import_step

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_STEP = (
    PROJECT_ROOT
    / "vendor"
    / "electronics"
    / "waveshare_bus_servo_adapter_a.step"
)

# Official Waveshare product dimensions and mounting pattern.
NOMINAL_BOARD_SIZE_XY_MM = (42.0, 33.0)
MOUNT_HOLE_DIAMETER_MM = 2.5
MOUNT_HOLE_SPACING_XY_MM = (37.0, 28.0)
MOUNT_HOLE_CENTERS_XY_MM = tuple(
    (x_sign * MOUNT_HOLE_SPACING_XY_MM[0] / 2.0,
     y_sign * MOUNT_HOLE_SPACING_XY_MM[1] / 2.0)
    for x_sign in (-1.0, 1.0)
    for y_sign in (-1.0, 1.0)
)

# Measured from Waveshare's official STEP.  Connectors and underside
# components extend beyond the nominal PCB outline and datum.
DETAIL_BOUNDS_SIZE_XYZ_MM = (42.02417808, 33.81628938, 14.59999982)
DETAIL_MIN_Z_MM = -3.600003485
DETAIL_MAX_Z_MM = 10.999996334
PRODUCT_MASS_KG = 0.016


def gen_step() -> Shape:
    """Return the exact board centered on its nominal PCB footprint."""
    vendor_board = import_step(VENDOR_STEP)
    centering = Location(
        (
            -NOMINAL_BOARD_SIZE_XY_MM[0] / 2.0,
            -NOMINAL_BOARD_SIZE_XY_MM[1] / 2.0,
            0.0,
        )
    )
    if vendor_board.children:
        board = Compound(
            children=[child.moved(centering) for child in vendor_board.children]
        )
    else:
        board = Compound(children=[vendor_board.moved(centering)])
    board.label = "waveshare_bus_servo_adapter_a"
    return board


if __name__ == "__main__":
    from build123d import export_step

    output_path = (
        PROJECT_ROOT
        / "exports"
        / "step"
        / "waveshare_bus_servo_adapter_a.step"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
