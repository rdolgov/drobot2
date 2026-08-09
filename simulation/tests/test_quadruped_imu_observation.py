"""Tests for the simulator-to-training IMU observation contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from simulation.isaac import _imu_observation as imu


def test_identity_orientation_projects_world_gravity_down_body_z() -> None:
    observation = imu.pack_imu_observation(
        linear_acceleration=(0.0, 0.0, imu.EARTH_GRAVITY_M_S2),
        angular_velocity=(0.1, -0.2, 0.3),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    assert observation.shape == (9,)
    assert observation[:3] == pytest.approx((0.1, -0.2, 0.3))
    assert observation[3:6] == pytest.approx((0.0, 0.0, -1.0))
    assert observation[6:] == pytest.approx((0.0, 0.0, 1.0))
    assert imu.IMU_OBSERVATION_FIELDS[3:6] == (
        "projected_gravity_x",
        "projected_gravity_y",
        "projected_gravity_z",
    )


def test_positive_ninety_degree_roll_projects_gravity_to_negative_y() -> None:
    half_angle = math.pi / 4.0
    projected = imu.rotate_world_vector_into_body(
        (0.0, 0.0, -1.0),
        (
            math.cos(half_angle),
            math.sin(half_angle),
            0.0,
            0.0,
        ),
    )

    assert projected == pytest.approx((0.0, -1.0, 0.0), abs=1e-6)


def test_invalid_or_nonfinite_frames_fail_fast() -> None:
    with pytest.raises(ValueError):
        imu.normalize_quaternion_wxyz((0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        imu.pack_imu_observation(
            linear_acceleration=(0.0, 0.0, np.nan),
            angular_velocity=(0.0, 0.0, 0.0),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
