"""Two-material AprilTag body marker for external robot pose tracking.

The marker geometry is tag36h11 ID 0. The code value and bit-coordinate order
come from AprilRobotics' BSD-2-Clause AprilTag implementation:
https://github.com/AprilRobotics/apriltag/blob/master/tag36h11.c

Coordinate convention:
- XY is the visible marker plane.
- +Z points out of the marker face.
- +Y is the marker's documented robot-forward direction.
- The origin is the center of the 80 x 80 mm marker image.
"""

from __future__ import annotations

from build123d import (
    Align,
    Box,
    BuildSketch,
    Color,
    Compound,
    Polygon,
    Pos,
    RectangleRounded,
    Shape,
    extrude,
)

TAG_FAMILY = "tag36h11"
TAG_ID = 0
TAG_CODE = 0x0000000D7E00984B

MODULE_SIZE_MM = 8.0
IMAGE_MODULES = 10
BLACK_BORDER_MODULES = 8
TAG_IMAGE_SIZE_MM = MODULE_SIZE_MM * IMAGE_MODULES
TAG_SIZE_MM = MODULE_SIZE_MM * BLACK_BORDER_MODULES

PLATE_WIDTH_MM = 104.0
PLATE_HEIGHT_MM = 80.0
PLATE_THICKNESS_MM = 2.0
BLACK_LAYER_MM = 0.24
CORNER_RADIUS_MM = 4.0

ZIP_SLOT_WIDTH_MM = 4.0
ZIP_SLOT_LENGTH_MM = 16.0
ZIP_SLOT_X_MM = 46.0
ZIP_SLOT_Y_MM = 20.0
FRONT_ARROW_CENTER_X_MM = 46.0
FRONT_ARROW_CENTER_Y_MM = 34.0
FRONT_ARROW_LENGTH_MM = 8.0
FRONT_ARROW_HEAD_WIDTH_MM = 7.0
FRONT_ARROW_HEAD_LENGTH_MM = 3.2
FRONT_ARROW_SHAFT_WIDTH_MM = 2.4

# Bit coordinates are listed in the exact decode order used by tag36h11.c.
BIT_COORDINATES = (
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (2, 2),
    (3, 2),
    (4, 2),
    (3, 3),
    (6, 1),
    (6, 2),
    (6, 3),
    (6, 4),
    (6, 5),
    (5, 2),
    (5, 3),
    (5, 4),
    (4, 3),
    (6, 6),
    (5, 6),
    (4, 6),
    (3, 6),
    (2, 6),
    (5, 5),
    (4, 5),
    (3, 5),
    (4, 4),
    (1, 6),
    (1, 5),
    (1, 4),
    (1, 3),
    (1, 2),
    (2, 5),
    (2, 4),
    (2, 3),
    (3, 4),
)


def marker_pixels() -> tuple[tuple[int, ...], ...]:
    """Return the official 10 x 10 marker image (1 white, 0 black)."""
    pixels = [[0 for _ in range(IMAGE_MODULES)] for _ in range(IMAGE_MODULES)]

    # AprilTag's total image includes a one-module white quiet border.
    for index in range(IMAGE_MODULES):
        pixels[0][index] = 1
        pixels[IMAGE_MODULES - 1][index] = 1
        pixels[index][0] = 1
        pixels[index][IMAGE_MODULES - 1] = 1

    border_start = (IMAGE_MODULES - BLACK_BORDER_MODULES) // 2
    for bit_index, (bit_x, bit_y) in enumerate(BIT_COORDINATES):
        if TAG_CODE & (1 << (len(BIT_COORDINATES) - bit_index - 1)):
            pixels[bit_y + border_start][bit_x + border_start] = 1

    return tuple(tuple(row) for row in pixels)


def _rounded_plate(thickness_mm: float) -> Shape:
    with BuildSketch() as plate_profile:
        RectangleRounded(PLATE_WIDTH_MM, PLATE_HEIGHT_MM, CORNER_RADIUS_MM)
    return extrude(plate_profile.sketch, amount=thickness_mm)


def _slot_cutters() -> list[Shape]:
    cutters: list[Shape] = []
    for x_mm in (-ZIP_SLOT_X_MM, ZIP_SLOT_X_MM):
        for y_mm in (-ZIP_SLOT_Y_MM, ZIP_SLOT_Y_MM):
            cutter = Box(
                ZIP_SLOT_WIDTH_MM,
                ZIP_SLOT_LENGTH_MM,
                PLATE_THICKNESS_MM + 2.0,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )
            cutters.append(Pos(x_mm, y_mm, -1.0) * cutter)

    half_length = FRONT_ARROW_LENGTH_MM / 2.0
    half_head_width = FRONT_ARROW_HEAD_WIDTH_MM / 2.0
    half_shaft_width = FRONT_ARROW_SHAFT_WIDTH_MM / 2.0
    head_base_y = (
        FRONT_ARROW_CENTER_Y_MM
        + half_length
        - FRONT_ARROW_HEAD_LENGTH_MM
    )
    arrow_points = (
        (
            FRONT_ARROW_CENTER_X_MM - half_shaft_width,
            FRONT_ARROW_CENTER_Y_MM - half_length,
        ),
        (
            FRONT_ARROW_CENTER_X_MM + half_shaft_width,
            FRONT_ARROW_CENTER_Y_MM - half_length,
        ),
        (FRONT_ARROW_CENTER_X_MM + half_shaft_width, head_base_y),
        (FRONT_ARROW_CENTER_X_MM + half_head_width, head_base_y),
        (FRONT_ARROW_CENTER_X_MM, FRONT_ARROW_CENTER_Y_MM + half_length),
        (FRONT_ARROW_CENTER_X_MM - half_head_width, head_base_y),
        (FRONT_ARROW_CENTER_X_MM - half_shaft_width, head_base_y),
    )
    with BuildSketch() as arrow_profile:
        Polygon(*arrow_points)
    arrow_cutter = extrude(
        arrow_profile.sketch,
        amount=PLATE_THICKNESS_MM + 2.0,
    )
    cutters.append(Pos(0.0, 0.0, -1.0) * arrow_cutter)
    return cutters


def make_white_plate() -> Shape:
    """Return the white structural plate with mounting openings."""
    plate = _rounded_plate(PLATE_THICKNESS_MM) - _slot_cutters()
    plate.label = "white_plate"
    plate.color = Color(1.0, 1.0, 1.0)
    return plate


def _black_runs() -> list[tuple[int, int, int]]:
    """Return (row, first column, run length) for contiguous black cells."""
    runs: list[tuple[int, int, int]] = []
    for row_index, row in enumerate(marker_pixels()):
        start: int | None = None
        for column_index, value in enumerate((*row, 1)):
            if value == 0 and start is None:
                start = column_index
            elif value != 0 and start is not None:
                runs.append((row_index, start, column_index - start))
                start = None
    return runs


def make_black_tag() -> Compound:
    """Return the raised black marker regions as one labeled multi-solid body."""
    cells: list[Shape] = []
    half_grid = (IMAGE_MODULES - 1) / 2.0
    for row, first_column, run_length in _black_runs():
        center_column = first_column + (run_length - 1) / 2.0
        x_mm = (center_column - half_grid) * MODULE_SIZE_MM
        y_mm = (half_grid - row) * MODULE_SIZE_MM
        run = Box(
            run_length * MODULE_SIZE_MM,
            MODULE_SIZE_MM,
            BLACK_LAYER_MM,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        run = Pos(x_mm, y_mm, PLATE_THICKNESS_MM) * run
        run.label = f"black_row_{row:02d}_column_{first_column:02d}"
        run.color = Color(0.0, 0.0, 0.0)
        cells.append(run)

    marker = Compound(children=cells)
    marker.label = "black_tag36h11_id_0"
    return marker


def make_apriltag_body_marker() -> Compound:
    """Return aligned white and black bodies for multi-material export."""
    marker = Compound(children=[make_white_plate(), make_black_tag()])
    marker.label = "apriltag_body_marker_tag36h11_id_0"
    return marker
