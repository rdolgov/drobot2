"""Pure geometry helpers shared by stair world generation and tests."""

from __future__ import annotations

from collections.abc import Mapping

from _stair_rl_contract import validate_staircase_config


def stair_layer_boxes(
    staircase: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Return stacked cube layers with no cracks beneath exposed treads."""

    validate_staircase_config(staircase)
    start = float(staircase["start_x_m"])
    count = int(staircase["step_count"])
    tread = float(staircase["tread_depth_m"])
    rise = float(staircase["rise_m"])
    width = float(staircase["width_m"])
    end = (
        start
        + count * tread
        + float(staircase["top_platform_depth_m"])
    )
    boxes: list[dict[str, object]] = []
    for index in range(count):
        x_min = start + index * tread
        x_length = end - x_min
        boxes.append(
            {
                "name": f"StepLayer_{index + 1:02d}",
                "center_xyz_m": (
                    x_min + x_length / 2.0,
                    0.0,
                    (index + 0.5) * rise,
                ),
                "size_xyz_m": (x_length, width, rise),
                "exposed_top_z_m": (index + 1) * rise,
                "exposed_tread_start_x_m": x_min,
                "exposed_tread_end_x_m": (
                    start + (index + 1) * tread
                    if index + 1 < count
                    else end
                ),
            }
        )
    return tuple(boxes)
