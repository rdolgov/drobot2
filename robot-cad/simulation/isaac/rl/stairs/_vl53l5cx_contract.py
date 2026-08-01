"""Pure NumPy contract for a VL53L5CX-style stair depth observation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

VL53L5CX_MODE = "vl53l5cx_raycast"
VL53L5CX_LANE_NAMES = ("left", "center", "right")


def _finite_vector(value, length: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain {length} finite values")
    return vector


def validate_vl53l5cx_config(
    config: Mapping[str, object],
    *,
    control_hz: int | None = None,
) -> None:
    """Reject sensor settings that do not match the modeled observation."""

    mode = str(config.get("mode", VL53L5CX_MODE))
    if mode != VL53L5CX_MODE:
        raise ValueError(f"Unsupported terrain perception mode: {mode}")
    rows = int(config["rows"])
    columns = int(config["columns"])
    if rows != 8 or columns != 8:
        raise ValueError("VL53L5CX stair perception requires an 8 x 8 grid")
    _finite_vector(
        config["optical_origin_from_base_xyz_m"],
        3,
        "optical_origin_from_base_xyz_m",
    )
    pitch = float(config["pitch_down_deg"])
    horizontal_fov = float(config["horizontal_fov_deg"])
    vertical_fov = float(config["vertical_fov_deg"])
    if not 0.0 < pitch < 90.0:
        raise ValueError("pitch_down_deg must be within (0, 90)")
    if not 0.0 < horizontal_fov < 180.0:
        raise ValueError("horizontal_fov_deg must be within (0, 180)")
    if not 0.0 < vertical_fov < 180.0:
        raise ValueError("vertical_fov_deg must be within (0, 180)")
    if pitch - vertical_fov / 2.0 <= 0.0:
        raise ValueError("Every modeled row must point below horizontal")

    minimum = float(config["minimum_range_m"])
    maximum = float(config["maximum_range_m"])
    useful = float(config["useful_normalization_range_m"])
    if not 0.0 < minimum < useful <= maximum:
        raise ValueError(
            "Sensor ranges must satisfy 0 < minimum < useful <= maximum"
        )
    near_limit = float(config["near_accuracy_limit_m"])
    near_bound = float(config["near_accuracy_bound_m"])
    far_bound = float(config["far_relative_accuracy_bound"])
    if not minimum < near_limit <= maximum:
        raise ValueError("near_accuracy_limit_m must be inside the sensor range")
    if near_bound < 0.0 or far_bound < 0.0:
        raise ValueError("Accuracy bounds cannot be negative")
    dropout = float(config["dropout_probability"])
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("dropout_probability must be within [0, 1]")

    update_rate = int(config["update_rate_hz"])
    latency_frames = int(config["latency_frames"])
    if update_rate <= 0:
        raise ValueError("update_rate_hz must be positive")
    if update_rate > 15:
        raise ValueError("VL53L5CX 8 x 8 mode is limited to 15 Hz")
    if latency_frames < 0:
        raise ValueError("latency_frames cannot be negative")
    if control_hz is not None:
        if int(control_hz) <= 0 or int(control_hz) % update_rate:
            raise ValueError(
                "control_hz must be a positive integer multiple of update_rate_hz"
            )

    lanes = tuple(tuple(int(column) for column in lane) for lane in config["lane_columns"])
    if len(lanes) != len(VL53L5CX_LANE_NAMES) or any(not lane for lane in lanes):
        raise ValueError("lane_columns must define non-empty left/center/right lanes")
    flattened = tuple(column for lane in lanes for column in lane)
    if sorted(flattened) != list(range(columns)):
        raise ValueError("lane_columns must cover every column exactly once")


def vl53l5cx_ray_directions(config: Mapping[str, object]) -> np.ndarray:
    """Return unit rays in base coordinates, shaped ``(row, column, xyz)``.

    Base +X is forward, +Y is left, and +Z is up. Row zero is the
    least-downward row; column zero is the left-most column.
    """

    validate_vl53l5cx_config(config)
    rows = int(config["rows"])
    columns = int(config["columns"])
    horizontal_fov = np.deg2rad(float(config["horizontal_fov_deg"]))
    vertical_fov = np.deg2rad(float(config["vertical_fov_deg"]))
    pitch = np.deg2rad(float(config["pitch_down_deg"]))
    horizontal = (
        horizontal_fov / 2.0
        - (np.arange(columns, dtype=np.float64) + 0.5)
        * horizontal_fov
        / columns
    )
    downward = (
        pitch
        - vertical_fov / 2.0
        + (np.arange(rows, dtype=np.float64) + 0.5) * vertical_fov / rows
    )
    directions = np.empty((rows, columns, 3), dtype=np.float64)
    for row, down_angle in enumerate(downward):
        horizontal_projection = np.cos(down_angle)
        directions[row, :, 0] = horizontal_projection * np.cos(horizontal)
        directions[row, :, 1] = horizontal_projection * np.sin(horizontal)
        directions[row, :, 2] = -np.sin(down_angle)
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    return directions.astype(np.float32)


def vl53l5cx_observation_fields(
    config: Mapping[str, object],
) -> tuple[str, ...]:
    """Return the stable 24-value lane-by-row depth field order."""

    validate_vl53l5cx_config(config)
    return tuple(
        f"tof_depth_{lane}_row_{row:02d}_normalized"
        for lane in VL53L5CX_LANE_NAMES
        for row in range(int(config["rows"]))
    )


def apply_vl53l5cx_noise(
    depth_grid_m,
    config: Mapping[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply bounded range error and modeled dropout to one depth grid.

    The manufacturer accuracy figures are represented as uniform error bounds.
    Dropout is an explicit simulation assumption rather than a datasheet claim.
    """

    validate_vl53l5cx_config(config)
    rows = int(config["rows"])
    columns = int(config["columns"])
    depth = np.asarray(depth_grid_m, dtype=np.float64)
    if depth.shape != (rows, columns):
        raise ValueError(f"depth_grid_m must have shape ({rows}, {columns})")
    minimum = float(config["minimum_range_m"])
    maximum = float(config["maximum_range_m"])
    valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    noisy = np.full(depth.shape, np.nan, dtype=np.float64)
    near_limit = float(config["near_accuracy_limit_m"])
    absolute_bound = float(config["near_accuracy_bound_m"])
    relative_bound = float(config["far_relative_accuracy_bound"])
    error_bound = np.where(depth <= near_limit, absolute_bound, depth * relative_bound)
    error = rng.uniform(-1.0, 1.0, size=depth.shape) * error_bound
    noisy[valid] = np.clip(depth[valid] + error[valid], minimum, maximum)
    dropout = rng.random(depth.shape) < float(config["dropout_probability"])
    noisy[dropout & valid] = np.nan
    return noisy.astype(np.float32)


def compress_vl53l5cx_depth_grid(
    depth_grid_m,
    config: Mapping[str, object],
) -> np.ndarray:
    """Compress 64 zones to median left/center/right values for each row."""

    validate_vl53l5cx_config(config)
    rows = int(config["rows"])
    columns = int(config["columns"])
    depth = np.asarray(depth_grid_m, dtype=np.float64)
    if depth.shape != (rows, columns):
        raise ValueError(f"depth_grid_m must have shape ({rows}, {columns})")
    minimum = float(config["minimum_range_m"])
    maximum = float(config["maximum_range_m"])
    normalization = float(config["useful_normalization_range_m"])
    valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    values: list[float] = []
    for lane in config["lane_columns"]:
        columns_in_lane = tuple(int(column) for column in lane)
        for row in range(rows):
            row_values = depth[row, columns_in_lane]
            row_valid = valid[row, columns_in_lane]
            median = (
                float(np.median(row_values[row_valid]))
                if np.any(row_valid)
                else normalization
            )
            values.append(float(np.clip(median / normalization, 0.0, 1.0)))
    return np.asarray(values, dtype=np.float32)
