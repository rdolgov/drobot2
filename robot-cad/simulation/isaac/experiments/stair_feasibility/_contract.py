"""Pure kinematic and acceptance helpers for real-stair feasibility."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from _quadruped_runtime import (
    LEGS,
    LINK_LENGTH_M,
    leg_ik,
    pose_by_name,
)

HARD_HIP_ABDUCTION_LIMIT_RAD = math.radians(25.0)
HARD_HIP_FLEXION_LIMIT_RAD = math.radians(60.0)
HARD_KNEE_LIMIT_RAD = math.radians(90.0)


def smoothstep(value: float) -> float:
    value = min(max(float(value), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported stair-feasibility schema")
    experiment = dict(config["experiment"])
    positive_fields = (
        "tread_depth_m",
        "step_width_m",
        "reset_base_z_m",
        "virtual_foot_radius_m",
        "edge_overtravel_m",
        "landing_margin_m",
        "swing_clearance_m",
    )
    for field in positive_fields:
        if float(experiment[field]) <= 0.0:
            raise ValueError(f"experiment.{field} must be positive")
    heights = tuple(float(value) for value in experiment["riser_heights_m"])
    if not heights or any(value <= 0.0 for value in heights):
        raise ValueError("riser_heights_m must contain positive values")
    if any(right <= left for left, right in zip(heights, heights[1:])):
        raise ValueError("riser_heights_m must be strictly increasing")
    if heights[-1] >= float(experiment["stance"]["down_m"]):
        raise ValueError("Riser height must be below nominal stance down")
    if experiment["swing_leg"] not in LEGS:
        raise ValueError("swing_leg is unknown")
    if not str(experiment["swing_leg"]).startswith("front_"):
        raise ValueError("v1 feasibility script supports a front swing leg")
    physics = dict(experiment["physics"])
    if int(physics["physics_hz"]) % int(physics["control_hz"]):
        raise ValueError("physics_hz must be divisible by control_hz")
    if float(physics["effort_cap_nm"]) <= 0.0:
        raise ValueError("effort_cap_nm must be positive")
    for value in dict(experiment["timing"]).values():
        if float(value) <= 0.0:
            raise ValueError("All phase durations must be positive")
    return experiment


def step_targets(
    experiment: Mapping[str, object],
    *,
    riser_height_m: float,
    shifted_base_x_m: float,
) -> dict[str, object]:
    """Return edge-clearance and tread-placement Cartesian/IK targets."""

    height = float(riser_height_m)
    stance_down = float(experiment["stance"]["down_m"])
    clearance = float(experiment["swing_clearance_m"])
    mid_down = stance_down - height - clearance
    landing_down = stance_down - height
    if mid_down <= 0.0:
        raise ValueError("Riser plus clearance exceeds nominal stance")
    hip_x = float(experiment["swing_hip_x_from_base_m"])
    step_start = float(experiment["step_start_x_m"])
    foot_radius = float(experiment["virtual_foot_radius_m"])
    mid_forward = (
        step_start
        + float(experiment["edge_overtravel_m"])
        - shifted_base_x_m
        - hip_x
        - foot_radius
    )
    landing_forward = (
        step_start
        + float(experiment["landing_margin_m"])
        - shifted_base_x_m
        - hip_x
    )
    swing_leg = str(experiment["swing_leg"])
    mid_hip, mid_knee = leg_ik(swing_leg, mid_down, mid_forward)
    landing_hip, landing_knee = leg_ik(
        swing_leg,
        landing_down,
        landing_forward,
    )
    return {
        "riser_height_m": height,
        "shifted_base_x_m": float(shifted_base_x_m),
        "edge_clearance": {
            "down_m": mid_down,
            "forward_m": mid_forward,
            "hip_flexion_rad": mid_hip,
            "knee_rad": mid_knee,
        },
        "landing": {
            "down_m": landing_down,
            "forward_m": landing_forward,
            "hip_flexion_rad": landing_hip,
            "knee_rad": landing_knee,
        },
    }


def target_limit_failures(
    targets: Mapping[str, object],
    *,
    margin_rad: float,
) -> tuple[str, ...]:
    failures: list[str] = []
    margin = float(margin_rad)
    for target_name in ("edge_clearance", "landing"):
        target = dict(targets[target_name])
        hip = abs(float(target["hip_flexion_rad"]))
        knee = abs(float(target["knee_rad"]))
        if hip >= HARD_HIP_FLEXION_LIMIT_RAD - margin:
            failures.append(
                f"{target_name}.hip_flexion={hip:.6f}>="
                f"{HARD_HIP_FLEXION_LIMIT_RAD - margin:.6f}"
            )
        if knee >= HARD_KNEE_LIMIT_RAD - margin:
            failures.append(
                f"{target_name}.knee={knee:.6f}>="
                f"{HARD_KNEE_LIMIT_RAD - margin:.6f}"
            )
    return tuple(failures)


def shifted_stance_pose(
    experiment: Mapping[str, object],
    *,
    transfer: float,
    swing_down_m: float | None = None,
    swing_forward_m: float | None = None,
) -> dict[str, float]:
    """Return a stance shifted backward/right for a front-left swing."""

    amount = smoothstep(transfer)
    stance = dict(experiment["stance"])
    down = float(stance["down_m"])
    fore_aft = float(stance["fore_aft_m"])
    abduction = math.radians(float(stance["abduction_deg"]))
    swing_leg = str(experiment["swing_leg"])
    side_sign = 1.0 if swing_leg.endswith("_left") else -1.0
    body_shift_forward = (
        -float(experiment["weight_shift"]["backward_m"]) * amount
    )
    body_shift_lateral = (
        -side_sign
        * float(experiment["weight_shift"]["away_from_swing_leg_m"])
        * amount
    )
    down_by_leg = {leg: down for leg in LEGS}
    forward_by_leg = {
        leg: (fore_aft if leg.startswith("front_") else -fore_aft)
        - body_shift_forward
        for leg in LEGS
    }
    foot_delta_lateral = -body_shift_lateral
    abduction_by_leg: dict[str, float] = {}
    for leg in LEGS:
        leg_side = 1.0 if leg.endswith("_left") else -1.0
        vertical = down * math.cos(abduction)
        outward = down * math.sin(abduction)
        shifted_outward = outward + leg_side * foot_delta_lateral
        down_by_leg[leg] = math.hypot(vertical, shifted_outward)
        abduction_by_leg[leg] = math.degrees(
            math.atan2(shifted_outward, vertical)
        )
    if swing_down_m is not None:
        down_by_leg[swing_leg] = float(swing_down_m)
    if swing_forward_m is not None:
        forward_by_leg[swing_leg] = float(swing_forward_m)
    return pose_by_name(
        down_by_leg_m=down_by_leg,
        forward_by_leg_m=forward_by_leg,
        abduction_by_leg_deg=abduction_by_leg,
    )


def planar_foot_down_m(hip_flexion_rad: float, knee_rad: float) -> float:
    hip = float(hip_flexion_rad)
    knee = float(knee_rad)
    return LINK_LENGTH_M * (
        math.cos(hip) + math.cos(hip + knee)
    )


def current_policy_front_lift_m(
    experiment: Mapping[str, object],
    *,
    hip_action_scale_rad: float = 0.36,
    knee_action_scale_rad: float = 0.48,
) -> float:
    """Return the current PPO upper-corner front-foot lift estimate."""

    stance = dict(experiment["stance"])
    hip, knee = leg_ik(
        str(experiment["swing_leg"]),
        float(stance["down_m"]),
        float(stance["fore_aft_m"]),
    )
    raised_down = planar_foot_down_m(
        hip + float(hip_action_scale_rad),
        knee + float(knee_action_scale_rad),
    )
    return float(stance["down_m"]) - raised_down


def signed_support_margin_m(
    point_xy: Sequence[float],
    support_points_xy: Sequence[Sequence[float]],
) -> float:
    """Return minimum signed edge distance for a triangular support polygon."""

    point = np.asarray(point_xy, dtype=float).reshape(2)
    points = np.asarray(support_points_xy, dtype=float).reshape(-1, 2)
    if points.shape != (3, 2):
        raise ValueError("Exactly three support points are required")
    center = np.mean(points, axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    polygon = points[order]

    def cross_2d(left: np.ndarray, right: np.ndarray) -> float:
        return float(left[0] * right[1] - left[1] * right[0])

    area_twice = sum(
        cross_2d(
            polygon[(index + 1) % 3] - polygon[index],
            center - polygon[index],
        )
        for index in range(3)
    )
    if area_twice < 0.0:
        polygon = polygon[::-1]
    distances = []
    for index in range(3):
        start = polygon[index]
        edge = polygon[(index + 1) % 3] - start
        length = float(np.linalg.norm(edge))
        if length <= 1e-9:
            raise ValueError("Support points must form a non-degenerate triangle")
        distances.append(cross_2d(edge, point - start) / length)
    return min(distances)


def trial_gate_failures(
    metrics: Mapping[str, object],
    acceptance: Mapping[str, object],
) -> tuple[str, ...]:
    failures: list[str] = []
    minimum_fields = (
        ("minimum_edge_clearance_m", "edge_clearance_m"),
        ("minimum_tread_contact_hold_s", "tread_contact_hold_s"),
        ("minimum_support_contact_fraction", "support_contact_fraction"),
        ("minimum_support_polygon_margin_m", "minimum_support_polygon_margin_m"),
    )
    for threshold_name, metric_name in minimum_fields:
        value = float(metrics[metric_name])
        threshold = float(acceptance[threshold_name])
        if value < threshold:
            failures.append(f"{metric_name}={value:.6f}<{threshold:.6f}")
    maximum_fields = (
        ("maximum_landing_height_error_m", "landing_height_error_m"),
        ("maximum_support_tip_slip_m", "maximum_support_tip_slip_m"),
        ("maximum_body_tilt_deg", "maximum_body_tilt_deg"),
        ("maximum_base_drop_m", "maximum_base_drop_m"),
        ("maximum_abs_joint_error_rad", "maximum_abs_joint_error_rad"),
        ("maximum_pd_saturation_fraction", "pd_saturation_fraction"),
    )
    for threshold_name, metric_name in maximum_fields:
        value = float(metrics[metric_name])
        threshold = float(acceptance[threshold_name])
        if value > threshold:
            failures.append(f"{metric_name}={value:.6f}>{threshold:.6f}")
    if bool(metrics["riser_strike"]):
        failures.append("riser_strike=true")
    if bool(metrics["nonfoot_step_collision"]):
        failures.append("nonfoot_step_collision=true")
    if not bool(metrics["tread_contact_achieved"]):
        failures.append("tread_contact_achieved=false")
    return tuple(failures)
