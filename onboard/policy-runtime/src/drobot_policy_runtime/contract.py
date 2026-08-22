"""The exact deployable observation, action, and joint-target contract."""

from __future__ import annotations

import numpy as np


OBSERVATION_SIZE = 50
ACTION_SIZE = 12
SERVO_VELOCITY_LIMIT_RAD_S = 4.5836625
GRAVITY_M_S2 = 9.81
GAIT_PERIOD_S = 0.8

ACTION_NAMES = (
    "front_left_hip_abduction",
    "rear_left_hip_abduction",
    "front_right_hip_abduction",
    "rear_right_hip_abduction",
    "front_left_hip_flexion",
    "rear_left_hip_flexion",
    "front_right_hip_flexion",
    "rear_right_hip_flexion",
    "front_left_knee",
    "rear_left_knee",
    "front_right_knee",
    "rear_right_knee",
)

STANCE_ANGLE_RAD = 0.5239596454
DEFAULT_JOINT_POSITION_RAD = np.asarray(
    (
        0.0,
        0.0,
        0.0,
        0.0,
        STANCE_ANGLE_RAD,
        -STANCE_ANGLE_RAD,
        STANCE_ANGLE_RAD,
        -STANCE_ANGLE_RAD,
        -STANCE_ANGLE_RAD,
        STANCE_ANGLE_RAD,
        -STANCE_ANGLE_RAD,
        STANCE_ANGLE_RAD,
    ),
    dtype=np.float32,
)
ACTION_SCALE_RAD = np.asarray(
    (0.12, 0.12, 0.12, 0.12, 0.30, 0.30, 0.30, 0.30, 0.40, 0.40, 0.40, 0.40),
    dtype=np.float32,
)
JOINT_LOWER_RAD = np.asarray(
    (-0.436,) * 4 + (-1.047,) * 4 + (-1.571,) * 4,
    dtype=np.float32,
)
JOINT_UPPER_RAD = -JOINT_LOWER_RAD

# Physical manifest order is FL, FR, RL, RR, while Isaac's tensor order is
# FL, RL, FR, RR inside each joint type.
SERVO_ID_BY_ACTION_NAME = {
    "front_left_hip_abduction": 1,
    "front_left_hip_flexion": 2,
    "front_left_knee": 3,
    "front_right_hip_abduction": 4,
    "front_right_hip_flexion": 5,
    "front_right_knee": 6,
    "rear_left_hip_abduction": 7,
    "rear_left_hip_flexion": 8,
    "rear_left_knee": 9,
    "rear_right_hip_abduction": 10,
    "rear_right_hip_flexion": 11,
    "rear_right_knee": 12,
}


def normalized_action_to_joint_target(
    action: np.ndarray,
    previous_target_rad: np.ndarray,
    control_hz: float,
) -> np.ndarray:
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    desired = np.clip(
        DEFAULT_JOINT_POSITION_RAD + ACTION_SCALE_RAD * action,
        JOINT_LOWER_RAD,
        JOINT_UPPER_RAD,
    )
    max_delta = SERVO_VELOCITY_LIMIT_RAD_S / control_hz
    return np.clip(
        desired,
        previous_target_rad - max_delta,
        previous_target_rad + max_delta,
    ).astype(np.float32)

