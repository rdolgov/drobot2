"""Pure observation, reward, and gate contract for one-foot lift training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from _rl_contract import (
    POLICY_OBSERVATION_CLIP,
    POLICY_OBSERVATION_FIELDS,
    POLICY_OBSERVATION_SIZE,
)

FOOT_LIFT_EXTRA_OBSERVATION_FIELDS = (
    "desired_swing_foot_lift_normalized",
    "measured_swing_foot_lift_normalized",
    "swing_foot_lift_error_normalized",
    "maximum_swing_foot_lift_normalized",
    "base_height_error_normalized",
    "base_forward_displacement_normalized",
    "base_lateral_displacement_normalized",
    "maximum_support_foot_lift_normalized",
)
FOOT_LIFT_OBSERVATION_FIELDS = (
    *POLICY_OBSERVATION_FIELDS,
    *FOOT_LIFT_EXTRA_OBSERVATION_FIELDS,
)
FOOT_LIFT_OBSERVATION_SIZE = len(FOOT_LIFT_OBSERVATION_FIELDS)


def _finite_vector(value: Sequence[float], length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {length} finite values")
    return result


def smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def desired_foot_lift_m(
    elapsed_seconds: float,
    *,
    target_lift_m: float,
    ramp_start_seconds: float,
    ramp_duration_seconds: float,
) -> float:
    """Return the commanded vertical lift along a smooth one-way ramp."""

    if target_lift_m <= 0.0:
        raise ValueError("target_lift_m must be positive")
    if ramp_start_seconds < 0.0 or ramp_duration_seconds <= 0.0:
        raise ValueError("lift ramp timing must be non-negative and non-zero")
    ramp_fraction = (float(elapsed_seconds) - float(ramp_start_seconds)) / float(
        ramp_duration_seconds
    )
    return float(target_lift_m) * smoothstep(ramp_fraction)


def pack_foot_lift_observation(
    *,
    walking_observation: Sequence[float],
    target_lift_m: float,
    desired_lift_m: float,
    measured_lift_m: float,
    maximum_lift_m: float,
    base_height_error_m: float,
    base_displacement_xy_m: Sequence[float],
    maximum_support_foot_lift_m: float,
) -> np.ndarray:
    """Append hardware-reproducible skill state to the 48-value walk input."""

    walking = _finite_vector(
        walking_observation,
        POLICY_OBSERVATION_SIZE,
        "walking_observation",
    )
    displacement = _finite_vector(
        base_displacement_xy_m,
        2,
        "base_displacement_xy_m",
    )
    target = float(target_lift_m)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_lift_m must be positive and finite")
    extras = np.asarray(
        [
            float(desired_lift_m) / target,
            float(measured_lift_m) / target,
            (float(desired_lift_m) - float(measured_lift_m)) / target,
            float(maximum_lift_m) / target,
            float(base_height_error_m) / 0.10,
            float(displacement[0]) / 0.10,
            float(displacement[1]) / 0.10,
            float(maximum_support_foot_lift_m) / target,
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(extras)):
        raise ValueError("foot-lift observation inputs must be finite")
    observation = np.concatenate((walking, extras)).astype(np.float32)
    return np.clip(
        observation,
        -POLICY_OBSERVATION_CLIP,
        POLICY_OBSERVATION_CLIP,
    ).astype(np.float32)


def foot_lift_failure_reasons(
    *,
    base_height_m: float,
    projected_gravity_xyz: Sequence[float],
    base_displacement_xy_m: Sequence[float],
    maximum_support_foot_lift_m: float,
    minimum_base_height_m: float,
    minimum_upright_cosine: float,
    maximum_base_displacement_m: float,
    maximum_support_foot_lift_allowed_m: float,
) -> tuple[str, ...]:
    gravity = _finite_vector(
        projected_gravity_xyz,
        3,
        "projected_gravity_xyz",
    )
    displacement = _finite_vector(
        base_displacement_xy_m,
        2,
        "base_displacement_xy_m",
    )
    reasons: list[str] = []
    if float(base_height_m) < float(minimum_base_height_m):
        reasons.append("base_too_low")
    if float(-gravity[2]) < float(minimum_upright_cosine):
        reasons.append("body_tipped")
    if float(np.linalg.norm(displacement)) > float(maximum_base_displacement_m):
        reasons.append("base_drifted")
    if float(maximum_support_foot_lift_m) > float(maximum_support_foot_lift_allowed_m):
        reasons.append("support_foot_lost")
    return tuple(reasons)


def foot_lift_success_reached(
    *,
    desired_lift_m: float,
    measured_lift_m: float,
    target_lift_m: float,
    minimum_success_lift_m: float,
    projected_gravity_xyz: Sequence[float],
    base_height_error_m: float,
    base_displacement_xy_m: Sequence[float],
    maximum_support_foot_lift_m: float,
    minimum_upright_cosine: float,
    maximum_base_height_error_m: float,
    maximum_base_displacement_m: float,
    maximum_support_foot_lift_allowed_m: float,
) -> bool:
    gravity = _finite_vector(
        projected_gravity_xyz,
        3,
        "projected_gravity_xyz",
    )
    displacement = _finite_vector(
        base_displacement_xy_m,
        2,
        "base_displacement_xy_m",
    )
    return bool(
        float(desired_lift_m) >= float(target_lift_m) - 1e-6
        and float(measured_lift_m) >= float(minimum_success_lift_m)
        and float(-gravity[2]) >= float(minimum_upright_cosine)
        and abs(float(base_height_error_m)) <= float(maximum_base_height_error_m)
        and float(np.linalg.norm(displacement)) <= float(maximum_base_displacement_m)
        and float(maximum_support_foot_lift_m) <= float(maximum_support_foot_lift_allowed_m)
    )


def foot_lift_reward_terms(
    *,
    desired_lift_m: float,
    measured_lift_m: float,
    lift_progress_m: float,
    tracking_target_reached: bool,
    base_height_error_m: float,
    base_displacement_xy_m: Sequence[float],
    maximum_support_foot_lift_m: float,
    body_angular_velocity_xyz: Sequence[float],
    projected_gravity_xyz: Sequence[float],
    joint_velocities_normalized: Sequence[float],
    action: Sequence[float],
    previous_action: Sequence[float],
    failed: bool,
    succeeded: bool,
    reward_config: Mapping[str, float],
) -> dict[str, float]:
    displacement = _finite_vector(
        base_displacement_xy_m,
        2,
        "base_displacement_xy_m",
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
        12,
        "joint_velocities_normalized",
    )
    current_action = _finite_vector(action, 12, "action")
    prior_action = _finite_vector(previous_action, 12, "previous_action")
    sigma = float(reward_config["lift_tracking_sigma_m"])
    if sigma <= 0.0:
        raise ValueError("lift_tracking_sigma_m must be positive")
    tracking_error = float(desired_lift_m) - float(measured_lift_m)
    tracking_score = math.exp(-0.5 * (tracking_error / sigma) ** 2)
    upright_cosine = float(np.clip(-gravity[2], -1.0, 1.0))
    terms = {
        "lift_tracking": float(reward_config["lift_tracking"]) * tracking_score,
        "lift_progress": float(reward_config["lift_progress"]) * max(0.0, float(lift_progress_m)),
        "target_hold": float(reward_config["target_hold"]) if tracking_target_reached else 0.0,
        "upright_deviation": float(reward_config["upright_deviation"]) * (1.0 - upright_cosine),
        "alive": float(reward_config["alive"]),
        "base_height": float(reward_config["base_height"]) * abs(float(base_height_error_m)),
        "base_drift": float(reward_config["base_drift"]) * float(np.linalg.norm(displacement)),
        "support_foot_lift": float(reward_config["support_foot_lift"])
        * max(0.0, float(maximum_support_foot_lift_m)),
        "roll_pitch_rate": float(reward_config["roll_pitch_rate"])
        * float(np.dot(angular_velocity[:2], angular_velocity[:2])),
        "yaw_rate": float(reward_config["yaw_rate"]) * float(angular_velocity[2] ** 2),
        "action_rate": float(reward_config["action_rate"])
        * float(np.mean(np.square(current_action - prior_action))),
        "action_magnitude": float(reward_config["action_magnitude"])
        * float(np.mean(np.square(current_action))),
        "joint_velocity": float(reward_config["joint_velocity"])
        * float(np.mean(np.square(joint_velocity))),
        "failure": float(reward_config["failure"]) if failed else 0.0,
        "success": float(reward_config["success"]) if succeeded else 0.0,
    }
    terms["total"] = float(sum(terms.values()))
    return terms
