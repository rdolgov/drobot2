"""Pure NumPy observation, reward, and termination contract for walking RL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from _imu_observation import IMU_OBSERVATION_FIELDS

COMMAND_FIELDS = (
    "command_forward_velocity_m_s",
    "command_lateral_velocity_m_s",
    "command_yaw_rate_rad_s",
)
JOINT_COUNT = 12
POLICY_OBSERVATION_FIELDS = (
    COMMAND_FIELDS
    + tuple(f"imu_{name}" for name in IMU_OBSERVATION_FIELDS)
    + tuple(f"joint_position_error_{index}" for index in range(JOINT_COUNT))
    + tuple(f"joint_velocity_normalized_{index}" for index in range(JOINT_COUNT))
    + tuple(f"previous_action_{index}" for index in range(JOINT_COUNT))
)
POLICY_OBSERVATION_SIZE = len(POLICY_OBSERVATION_FIELDS)
POLICY_OBSERVATION_CLIP = 20.0


def _finite_vector(value, length: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain {length} finite values")
    return vector


def pack_policy_observation(
    *,
    command_velocity_xyz,
    imu_observation,
    joint_positions,
    nominal_joint_positions,
    joint_velocities,
    joint_max_velocities,
    previous_action,
) -> np.ndarray:
    """Pack the deployable 48-value walking-policy observation.

    The policy does not receive simulator-only base linear velocity. That value
    is used by the reward during training, while the observation remains
    reproducible on hardware from commands, the body IMU, servo feedback, and
    the previous policy action.
    """

    command = _finite_vector(command_velocity_xyz, 3, "command_velocity_xyz")
    imu = _finite_vector(
        imu_observation,
        len(IMU_OBSERVATION_FIELDS),
        "imu_observation",
    )
    positions = _finite_vector(joint_positions, JOINT_COUNT, "joint_positions")
    nominal = _finite_vector(
        nominal_joint_positions,
        JOINT_COUNT,
        "nominal_joint_positions",
    )
    velocities = _finite_vector(
        joint_velocities,
        JOINT_COUNT,
        "joint_velocities",
    )
    max_velocities = _finite_vector(
        joint_max_velocities,
        JOINT_COUNT,
        "joint_max_velocities",
    )
    if np.any(max_velocities <= 0.0):
        raise ValueError("joint_max_velocities must be positive")
    previous = _finite_vector(previous_action, JOINT_COUNT, "previous_action")

    position_error = np.clip(positions - nominal, -2.0, 2.0)
    normalized_velocity = np.clip(velocities / max_velocities, -2.0, 2.0)
    observation = np.concatenate(
        (command, imu, position_error, normalized_velocity, previous)
    ).astype(np.float32)
    observation = np.clip(
        observation,
        -POLICY_OBSERVATION_CLIP,
        POLICY_OBSERVATION_CLIP,
    )
    if observation.shape != (POLICY_OBSERVATION_SIZE,):
        raise AssertionError(f"Unexpected policy observation shape: {observation.shape}")
    return observation


def walking_reward_terms(
    *,
    command_velocity_xyz,
    body_linear_velocity_xyz,
    body_angular_velocity_xyz,
    projected_gravity_xyz,
    base_height_m: float,
    joint_velocities_normalized,
    action,
    previous_action,
    terminated: bool,
    reward_config: Mapping[str, float],
) -> dict[str, float]:
    """Return individually reviewable locomotion reward terms."""

    command = _finite_vector(command_velocity_xyz, 3, "command_velocity_xyz")
    linear_velocity = _finite_vector(
        body_linear_velocity_xyz,
        3,
        "body_linear_velocity_xyz",
    )
    angular_velocity = _finite_vector(
        body_angular_velocity_xyz,
        3,
        "body_angular_velocity_xyz",
    )
    gravity = _finite_vector(
        projected_gravity_xyz,
        3,
        "projected_gravity_xyz",
    )
    joint_velocity = _finite_vector(
        joint_velocities_normalized,
        JOINT_COUNT,
        "joint_velocities_normalized",
    )
    current_action = _finite_vector(action, JOINT_COUNT, "action")
    prior_action = _finite_vector(previous_action, JOINT_COUNT, "previous_action")
    sigma = float(reward_config["velocity_tracking_sigma_m_s"])
    if sigma <= 0.0:
        raise ValueError("velocity_tracking_sigma_m_s must be positive")

    forward_error = float(linear_velocity[0] - command[0])
    velocity_tracking = float(np.exp(-((forward_error / sigma) ** 2)))
    upright_cosine = float(np.clip(-gravity[2], 0.0, 1.0))
    height_error = float(
        base_height_m - float(reward_config["target_body_height_m"])
    )
    terms = {
        "forward_velocity_tracking": (
            float(reward_config["forward_velocity_tracking"]) * velocity_tracking
        ),
        "upright": float(reward_config["upright"]) * upright_cosine,
        "alive": float(reward_config["alive"]),
        "lateral_velocity": (
            float(reward_config["lateral_velocity"])
            * float(linear_velocity[1] ** 2)
        ),
        "vertical_velocity": (
            float(reward_config["vertical_velocity"])
            * float(linear_velocity[2] ** 2)
        ),
        "roll_pitch_rate": (
            float(reward_config["roll_pitch_rate"])
            * float(np.sum(np.square(angular_velocity[:2])))
        ),
        "yaw_rate": (
            float(reward_config["yaw_rate"])
            * float((angular_velocity[2] - command[2]) ** 2)
        ),
        "body_height": float(reward_config["body_height"]) * height_error**2,
        "action_rate": (
            float(reward_config["action_rate"])
            * float(np.mean(np.square(current_action - prior_action)))
        ),
        "action_magnitude": (
            float(reward_config["action_magnitude"])
            * float(np.mean(np.square(current_action)))
        ),
        "joint_velocity": (
            float(reward_config["joint_velocity"])
            * float(np.mean(np.square(joint_velocity)))
        ),
        "termination": (
            float(reward_config["termination"]) if terminated else 0.0
        ),
    }
    terms["total"] = float(sum(terms.values()))
    return terms


def termination_reasons(
    *,
    base_height_m: float,
    projected_gravity_xyz: Sequence[float],
    minimum_base_height_m: float,
    minimum_upright_cosine: float,
    finite_state: bool = True,
) -> tuple[str, ...]:
    """Return deterministic fall/non-finite termination reasons."""

    gravity = _finite_vector(
        projected_gravity_xyz,
        3,
        "projected_gravity_xyz",
    )
    reasons: list[str] = []
    if not finite_state:
        reasons.append("non_finite_state")
    if float(base_height_m) < float(minimum_base_height_m):
        reasons.append("base_too_low")
    if float(-gravity[2]) < float(minimum_upright_cosine):
        reasons.append("body_tipped")
    return tuple(reasons)
