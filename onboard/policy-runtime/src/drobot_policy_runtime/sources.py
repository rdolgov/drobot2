"""Replaceable IMU and joint-state sources for script and future ROS use."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .contract import DEFAULT_JOINT_POSITION_RAD


@dataclass(frozen=True)
class ImuSample:
    angular_velocity_body_rad_s: np.ndarray
    projected_gravity_body: np.ndarray
    linear_acceleration_body_m_s2: np.ndarray
    heading_yaw_rad: float
    monotonic_time_s: float


@dataclass(frozen=True)
class JointStateSample:
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    monotonic_time_s: float


class ImuSource(Protocol):
    def read(self) -> ImuSample: ...


class JointStateSource(Protocol):
    def read(self) -> JointStateSample: ...


class LevelImuSource:
    """Stationary level input for dependency and policy-pipeline checks."""

    def read(self) -> ImuSample:
        return ImuSample(
            angular_velocity_body_rad_s=np.zeros(3, dtype=np.float32),
            projected_gravity_body=np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
            linear_acceleration_body_m_s2=np.asarray((0.0, 0.0, 9.81), dtype=np.float32),
            heading_yaw_rad=0.0,
            monotonic_time_s=time.monotonic(),
        )


class NeutralJointStateSource:
    """Print-only bring-up source; it is not valid for closed-loop actuation."""

    def read(self) -> JointStateSample:
        return JointStateSample(
            position_rad=DEFAULT_JOINT_POSITION_RAD.copy(),
            velocity_rad_s=np.zeros(12, dtype=np.float32),
            monotonic_time_s=time.monotonic(),
        )


def _quaternion_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Return the sensor-to-game-world rotation represented by an XYZW quaternion."""

    x, y, z, w = (float(value) for value in quaternion_xyzw)
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        raise RuntimeError("BNO085 returned a non-finite quaternion")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-6:
        raise RuntimeError("BNO085 returned an invalid zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def _quaternion_inverse_rotate(
    quaternion_xyzw: np.ndarray, vector_world: np.ndarray
) -> np.ndarray:
    rotation = _quaternion_rotation_matrix(quaternion_xyzw)
    return rotation.T @ vector_world


def _body_forward_heading_rad(
    quaternion_xyzw: np.ndarray,
    axis_map: tuple[tuple[int, float], ...],
) -> float:
    """Project mapped body +X into the BNO game-world horizontal plane."""

    # ``axis_map`` declares body[i] = sign * sensor[index].  Its first entry
    # therefore gives the sensor-frame vector corresponding to body-forward.
    sensor_forward = np.zeros(3, dtype=np.float32)
    sensor_index, sign = axis_map[0]
    sensor_forward[sensor_index] = sign
    world_forward = _quaternion_rotation_matrix(quaternion_xyzw) @ sensor_forward
    horizontal_norm = math.hypot(float(world_forward[0]), float(world_forward[1]))
    if horizontal_norm < 1.0e-4:
        raise RuntimeError("BNO085 body-forward heading is undefined near vertical")
    return math.atan2(float(world_forward[1]), float(world_forward[0]))


def parse_axis_map(text: str) -> tuple[tuple[int, float], ...]:
    axes = {"x": 0, "y": 1, "z": 2}
    result: list[tuple[int, float]] = []
    for token in text.lower().replace(" ", "").split(","):
        if len(token) != 2 or token[0] not in "+-" or token[1] not in axes:
            raise ValueError("axis map must look like +x,+y,+z or +y,-x,+z")
        result.append((axes[token[1]], 1.0 if token[0] == "+" else -1.0))
    if len(result) != 3 or len({axis for axis, _ in result}) != 3:
        raise ValueError("axis map must use x, y, and z exactly once")
    transform = np.zeros((3, 3), dtype=np.float32)
    for body_axis, (sensor_axis, sign) in enumerate(result):
        transform[body_axis, sensor_axis] = sign
    if not math.isclose(float(np.linalg.det(transform)), 1.0, abs_tol=1.0e-6):
        raise ValueError(
            "axis map must be a right-handed rigid rotation so gyro yaw and "
            "quaternion heading use the same sign"
        )
    return tuple(result)


class Bno085ImuSource:
    """Read body-frame policy inputs directly from the Pi's BNO085."""

    def __init__(
        self,
        address: int = 0x4A,
        axis_map: str = "+x,+y,+z",
        i2c_bus: int | None = None,
    ) -> None:
        try:
            from adafruit_bno08x import (
                BNO_REPORT_ACCELEROMETER,
                BNO_REPORT_GAME_ROTATION_VECTOR,
                BNO_REPORT_GYROSCOPE,
            )
            from adafruit_bno08x.i2c import BNO08X_I2C
        except ImportError as exc:
            raise RuntimeError(
                "BNO085 dependencies are missing; install the package with [bno085]"
            ) from exc

        self._axis_map = parse_axis_map(axis_map)
        if i2c_bus is None:
            configured_bus = os.environ.get("DROBOT_BNO085_I2C_BUS", "").strip()
            if configured_bus:
                try:
                    i2c_bus = int(configured_bus, 10)
                except ValueError as exc:
                    raise RuntimeError(
                        "DROBOT_BNO085_I2C_BUS must be a non-negative integer"
                    ) from exc
                if i2c_bus < 0:
                    raise RuntimeError(
                        "DROBOT_BNO085_I2C_BUS must be a non-negative integer"
                    )

        if i2c_bus is None:
            import board
            import busio

            self._i2c = busio.I2C(board.SCL, board.SDA)
        else:
            try:
                from adafruit_extended_bus import ExtendedI2C
            except ImportError as exc:
                raise RuntimeError(
                    "Software I2C support is missing; reinstall the package with [bno085]"
                ) from exc
            self._i2c = ExtendedI2C(i2c_bus)

        self._imu = BNO08X_I2C(self._i2c, address=address)
        self._imu.enable_feature(BNO_REPORT_ACCELEROMETER)
        self._imu.enable_feature(BNO_REPORT_GYROSCOPE)
        self._imu.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)
        time.sleep(1.0)

    def _to_body(self, sensor_vector: np.ndarray) -> np.ndarray:
        return np.asarray(
            [sign * sensor_vector[index] for index, sign in self._axis_map],
            dtype=np.float32,
        )

    def read(self) -> ImuSample:
        acceleration = self._imu.acceleration
        gyro = self._imu.gyro
        quaternion = self._imu.game_quaternion
        if acceleration is None or gyro is None or quaternion is None:
            raise RuntimeError("BNO085 sample is incomplete")
        acceleration_sensor = np.asarray(acceleration, dtype=np.float32)
        gyro_sensor = np.asarray(gyro, dtype=np.float32)
        quaternion_xyzw = np.asarray(quaternion, dtype=np.float32)
        gravity_sensor = _quaternion_inverse_rotate(
            quaternion_xyzw, np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
        )
        heading_yaw_rad = _body_forward_heading_rad(
            quaternion_xyzw,
            self._axis_map,
        )
        values = np.concatenate(
            (
                acceleration_sensor,
                gyro_sensor,
                gravity_sensor,
                quaternion_xyzw,
                np.asarray((heading_yaw_rad,), dtype=np.float32),
            )
        )
        if not np.all(np.isfinite(values)):
            raise RuntimeError("BNO085 sample contains a non-finite value")
        return ImuSample(
            angular_velocity_body_rad_s=self._to_body(gyro_sensor),
            projected_gravity_body=self._to_body(gravity_sensor),
            linear_acceleration_body_m_s2=self._to_body(acceleration_sensor),
            heading_yaw_rad=heading_yaw_rad,
            monotonic_time_s=time.monotonic(),
        )
