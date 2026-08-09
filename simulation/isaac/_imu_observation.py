"""Pure NumPy IMU observation adapter shared by validation and training code."""

from __future__ import annotations

import numpy as np

EARTH_GRAVITY_M_S2 = 9.81
IMU_OBSERVATION_FIELDS = (
    "angular_velocity_x_rad_s",
    "angular_velocity_y_rad_s",
    "angular_velocity_z_rad_s",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    "linear_acceleration_x_g",
    "linear_acceleration_y_g",
    "linear_acceleration_z_g",
)


def _vector(value, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def normalize_quaternion_wxyz(value) -> np.ndarray:
    """Return a finite unit quaternion in Isaac's wxyz order."""
    quaternion = _vector(value, 4, "orientation_wxyz")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-8:
        raise ValueError("orientation_wxyz must have non-zero norm")
    return quaternion / norm


def rotate_world_vector_into_body(vector_world, orientation_wxyz) -> np.ndarray:
    """Rotate a world-frame vector into an IMU/body frame.

    Isaac reports the sensor orientation as the world-from-sensor quaternion.
    The inverse rotation is therefore used for projected gravity.
    """
    vector = _vector(vector_world, 3, "vector_world")
    quaternion = normalize_quaternion_wxyz(orientation_wxyz)
    scalar = quaternion[0]
    xyz = quaternion[1:]
    return (
        vector * (2.0 * scalar * scalar - 1.0)
        - 2.0 * scalar * np.cross(xyz, vector)
        + 2.0 * xyz * np.dot(xyz, vector)
    ).astype(np.float32)


def pack_imu_observation(
    *,
    linear_acceleration,
    angular_velocity,
    orientation_wxyz,
) -> np.ndarray:
    """Return the nine-value walking-policy IMU observation.

    Field order is angular velocity, projected unit gravity, then linear
    acceleration normalized by Earth gravity.
    """
    angular_velocity_vector = _vector(
        angular_velocity,
        3,
        "angular_velocity",
    )
    linear_acceleration_vector = _vector(
        linear_acceleration,
        3,
        "linear_acceleration",
    )
    projected_gravity = rotate_world_vector_into_body(
        (0.0, 0.0, -1.0),
        orientation_wxyz,
    )
    return np.concatenate(
        (
            angular_velocity_vector,
            projected_gravity,
            linear_acceleration_vector / EARTH_GRAVITY_M_S2,
        )
    ).astype(np.float32)


def pack_imu_frame(frame: dict[str, object]) -> np.ndarray:
    """Pack an ``IMUSensor.get_data()`` dictionary."""
    return pack_imu_observation(
        linear_acceleration=frame["linear_acceleration"],
        angular_velocity=frame["angular_velocity"],
        orientation_wxyz=frame["orientation"],
    )
