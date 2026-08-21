"""Replaceable IMU and joint-state sources for script and future ROS use."""

from __future__ import annotations

import math
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


def _quaternion_inverse_rotate(
    quaternion_xyzw: np.ndarray, vector_world: np.ndarray
) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-6:
        raise RuntimeError("BNO085 returned an invalid zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )
    return rotation.T @ vector_world


def parse_axis_map(text: str) -> tuple[tuple[int, float], ...]:
    axes = {"x": 0, "y": 1, "z": 2}
    result: list[tuple[int, float]] = []
    for token in text.lower().replace(" ", "").split(","):
        if len(token) != 2 or token[0] not in "+-" or token[1] not in axes:
            raise ValueError("axis map must look like +x,+y,+z or +y,-x,+z")
        result.append((axes[token[1]], 1.0 if token[0] == "+" else -1.0))
    if len(result) != 3 or len({axis for axis, _ in result}) != 3:
        raise ValueError("axis map must use x, y, and z exactly once")
    return tuple(result)


class Bno085ImuSource:
    """Read body-frame policy inputs directly from the Pi's BNO085."""

    def __init__(self, address: int = 0x4A, axis_map: str = "+x,+y,+z") -> None:
        try:
            import board
            import busio
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
        i2c = busio.I2C(board.SCL, board.SDA)
        self._imu = BNO08X_I2C(i2c, address=address)
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
        values = np.concatenate((acceleration_sensor, gyro_sensor, gravity_sensor))
        if not np.all(np.isfinite(values)):
            raise RuntimeError("BNO085 sample contains a non-finite value")
        return ImuSample(
            angular_velocity_body_rad_s=self._to_body(gyro_sensor),
            projected_gravity_body=self._to_body(gravity_sensor),
            linear_acceleration_body_m_s2=self._to_body(acceleration_sensor),
            monotonic_time_s=time.monotonic(),
        )

