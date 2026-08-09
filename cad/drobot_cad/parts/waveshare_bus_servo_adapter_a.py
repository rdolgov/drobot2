"""Exact Waveshare Bus Servo Adapter (A) reference geometry.

This is the USB/UART half-duplex serial-bus controller used by the standard
Feetech LeKiwi setup.  It is not a CAN-bus controller.

Coordinate convention:
    - origin X/Y: nominal 42 x 33 mm PCB center
    - origin Z: vendor PCB mounting datum
    - +Z: component side
"""

from __future__ import annotations

import warnings
from pathlib import Path

from build123d import Align, Box, Compound, Location, Shape, import_step

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
DETAIL_BOUNDS_CENTER_XY_MM = (-0.00000644, -0.39606949)
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
    # Flatten the vendor occurrence tree into solids and bake every resulting
    # placement into its B-rep. Preserving non-identity occurrence locations
    # causes some STEP writers/readers to apply connector transforms twice.
    baked_solids = []
    for solid in vendor_board.solids():
        centered_solid = solid.moved(centering)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            centered_solid.relocate(Location())
        baked_solids.append(centered_solid)
    # Keep these as one topological multi-solid part. Nesting 117 individual
    # XCAF occurrences inside the full robot can exceed the writer's practical
    # hierarchy depth even though the same board exports successfully alone.
    board = Compound(baked_solids)
    board.label = "waveshare_bus_servo_adapter_a"
    return board


def make_fit_proxy() -> Shape:
    """Return the exact detailed envelope for reliable full-robot export."""
    proxy = Box(
        *DETAIL_BOUNDS_SIZE_XYZ_MM,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    ).moved(
        Location(
            (
                *DETAIL_BOUNDS_CENTER_XY_MM,
                DETAIL_MIN_Z_MM,
            )
        )
    )
    proxy.label = "waveshare_bus_servo_adapter_a_fit_envelope"
    return proxy


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
