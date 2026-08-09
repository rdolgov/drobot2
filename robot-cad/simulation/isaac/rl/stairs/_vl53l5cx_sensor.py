"""Isaac PhysX raycast runtime for the VL53L5CX stair observation."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import numpy as np
from _vl53l5cx_contract import (
    apply_vl53l5cx_noise,
    compress_vl53l5cx_depth_grid,
    validate_vl53l5cx_config,
    vl53l5cx_observation_fields,
    vl53l5cx_ray_directions,
)
from omni.physx import get_physx_scene_query_interface
from pxr import Gf


def _rotate_wxyz(quaternion_wxyz, vectors) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("base_orientation_wxyz must be finite")
    magnitude = float(np.linalg.norm(quaternion))
    if magnitude <= 0.0:
        raise ValueError("base_orientation_wxyz cannot be zero")
    quaternion /= magnitude
    w = quaternion[0]
    quaternion_vector = quaternion[1:]
    value = np.asarray(vectors, dtype=np.float64)
    return (
        value
        + 2.0 * w * np.cross(quaternion_vector, value)
        + 2.0
        * np.cross(
            quaternion_vector,
            np.cross(quaternion_vector, value),
        )
    )


class VL53L5CXRaycastSensor:
    """Sample 64 closest-hit rays at the real sensor's 8 x 8 cadence."""

    def __init__(
        self,
        config: Mapping[str, object],
        *,
        control_hz: int,
    ) -> None:
        self.config = dict(config)
        validate_vl53l5cx_config(self.config, control_hz=control_hz)
        self.control_hz = int(control_hz)
        self.update_rate_hz = int(self.config["update_rate_hz"])
        self.control_frames_per_measurement = (
            self.control_hz // self.update_rate_hz
        )
        self.latency_frames = int(self.config["latency_frames"])
        self.origin_from_base_m = np.asarray(
            self.config["optical_origin_from_base_xyz_m"],
            dtype=np.float64,
        ).reshape(3)
        self.ray_directions_from_base = vl53l5cx_ray_directions(self.config)
        self.observation_fields = vl53l5cx_observation_fields(self.config)
        self.scene_query = get_physx_scene_query_interface()
        self.reset()

    def reset(self) -> None:
        rows = int(self.config["rows"])
        columns = int(self.config["columns"])
        empty_observation = np.ones(len(self.observation_fields), dtype=np.float32)
        self._latency_queue: deque[np.ndarray] = deque(
            empty_observation.copy() for _ in range(self.latency_frames)
        )
        self.latest_observation = empty_observation
        self.latest_true_depth_grid_m = np.full(
            (rows, columns),
            np.nan,
            dtype=np.float32,
        )
        self.latest_noisy_depth_grid_m = self.latest_true_depth_grid_m.copy()
        self.latest_hit_paths = tuple("" for _ in range(rows * columns))
        self.control_frame_index = 0
        self.measurement_count = 0
        self.last_measurement_control_frame: int | None = None

    def _measure(
        self,
        *,
        base_position_world_m,
        base_orientation_wxyz,
        rng: np.random.Generator,
    ) -> np.ndarray:
        base_position = np.asarray(
            base_position_world_m,
            dtype=np.float64,
        ).reshape(3)
        if not np.all(np.isfinite(base_position)):
            raise ValueError("base_position_world_m must be finite")
        origin_world = base_position + _rotate_wxyz(
            base_orientation_wxyz,
            self.origin_from_base_m,
        )
        directions_world = _rotate_wxyz(
            base_orientation_wxyz,
            self.ray_directions_from_base.reshape(-1, 3),
        )
        maximum_range = float(self.config["maximum_range_m"])
        depth = np.full(directions_world.shape[0], np.nan, dtype=np.float32)
        hit_paths: list[str] = []
        origin = Gf.Vec3f(*(float(value) for value in origin_world))
        for index, direction_world in enumerate(directions_world):
            direction = Gf.Vec3f(
                *(float(value) for value in direction_world)
            )
            hit_info = self.scene_query.raycast_closest(
                origin,
                direction,
                maximum_range,
            )
            if hit_info and bool(hit_info.get("hit", False)):
                position = np.asarray(hit_info["position"], dtype=np.float64)
                depth[index] = float(np.linalg.norm(position - origin_world))
                hit_paths.append(str(hit_info.get("collision", "")))
            else:
                hit_paths.append("")
        rows = int(self.config["rows"])
        columns = int(self.config["columns"])
        self.latest_true_depth_grid_m = depth.reshape(rows, columns)
        self.latest_noisy_depth_grid_m = apply_vl53l5cx_noise(
            self.latest_true_depth_grid_m,
            self.config,
            rng,
        )
        self.latest_hit_paths = tuple(hit_paths)
        return compress_vl53l5cx_depth_grid(
            self.latest_noisy_depth_grid_m,
            self.config,
        )

    def observe(
        self,
        *,
        base_position_world_m,
        base_orientation_wxyz,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return the held, latency-delayed sensor observation."""

        if self.control_frame_index % self.control_frames_per_measurement == 0:
            measurement = self._measure(
                base_position_world_m=base_position_world_m,
                base_orientation_wxyz=base_orientation_wxyz,
                rng=rng,
            )
            self._latency_queue.append(measurement)
            self.latest_observation = self._latency_queue.popleft()
            self.measurement_count += 1
            self.last_measurement_control_frame = self.control_frame_index
        self.control_frame_index += 1
        return self.latest_observation.copy()

    @property
    def metrics(self) -> dict[str, object]:
        finite = np.isfinite(self.latest_noisy_depth_grid_m)
        return {
            "measurement_count": self.measurement_count,
            "last_measurement_control_frame": self.last_measurement_control_frame,
            "control_frames_per_measurement": self.control_frames_per_measurement,
            "latency_frames": self.latency_frames,
            "latency_seconds": self.latency_frames / self.update_rate_hz,
            "valid_zone_count": int(np.count_nonzero(finite)),
            "closest_valid_depth_m": (
                float(np.min(self.latest_noisy_depth_grid_m[finite]))
                if np.any(finite)
                else None
            ),
            "hit_path_count": int(sum(bool(path) for path in self.latest_hit_paths)),
        }
