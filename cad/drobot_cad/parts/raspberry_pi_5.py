"""Exact Raspberry Pi 5 reference geometry for enclosure fit previews.

The step.parts model uses X for board length, Z for board width, and Y for
height.  This wrapper rotates it onto the robot's XY mounting plane while
keeping the PCB bottom at local Z=0.

Coordinate convention:
    - origin: vendor model datum projected onto the PCB bottom
    - +X: board length
    - +Y: board width after installation
    - +Z: component side of the board
"""

from __future__ import annotations

from pathlib import Path

from build123d import Compound, Location, Shape, import_step

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_STEP = PROJECT_ROOT / "vendor" / "electronics" / "raspberry_pi_5.step"

BOARD_LENGTH_X_MM = 85.0
BOARD_WIDTH_Y_MM = 56.0
MOUNT_HOLE_DIAMETER_MM = 2.7
MOUNT_HOLE_CENTERS_XY_MM = (
    (-39.0, -24.5),
    (-39.0, 24.5),
    (19.0, -24.5),
    (19.0, 24.5),
)


def gen_step() -> Shape:
    """Return the exact catalog Pi rotated into its installed orientation."""
    vendor_pi = import_step(VENDOR_STEP)
    installed_orientation = Location((0.0, 0.0, 0.0), (90.0, 0.0, 0.0))
    children = list(vendor_pi.children) or [vendor_pi]
    pi = Compound(
        children=[child.moved(installed_orientation) for child in children]
    )
    pi.label = "raspberry_pi_5_exact_reference"
    return pi
