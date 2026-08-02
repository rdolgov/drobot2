"""Pure NumPy contract for the separate stair-climbing policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np
from _rl_contract import (
    JOINT_COUNT,
    POLICY_OBSERVATION_CLIP,
    POLICY_OBSERVATION_FIELDS,
    POLICY_OBSERVATION_SIZE,
)

STAIR_FOOT_NAMES = (
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
)
PLACEMENT_PHASES = (
    "weight_shift",
    "lift",
    "advance",
    "lower",
    "hold",
)
PLACEMENT_REFERENCE_OBSERVATION_FIELDS = (
    *(f"placement_phase_{phase}" for phase in PLACEMENT_PHASES),
    "placement_desired_swing_height_normalized",
    "placement_measured_swing_height_normalized",
    "placement_swing_x_error_normalized",
    "placement_swing_z_error_normalized",
    "placement_tread_normal_load_normalized",
    "placement_support_contact_fraction",
    "placement_support_margin_normalized",
    "placement_maximum_support_slip_normalized",
)
SUPPORT_REGULATION_OBSERVATION_FIELDS = (
    *(
        f"placement_total_normal_load_{name}_normalized"
        for name in STAIR_FOOT_NAMES
    ),
    "placement_com_target_error_x_normalized",
    "placement_com_target_error_y_normalized",
    *(
        f"placement_pd_effort_cap_ratio_{name}"
        for name in STAIR_FOOT_NAMES
    ),
    *(
        f"placement_pd_saturated_joint_fraction_{name}"
        for name in STAIR_FOOT_NAMES
    ),
)
STAIR_LEG_DOF_INDICES = (
    (0, 4, 8),
    (2, 6, 10),
    (1, 5, 9),
    (3, 7, 11),
)


def config_for_height_stage(
    config: Mapping[str, object],
    stage_id: str | None,
) -> dict[str, object]:
    """Return a config copy with one declared stair-height stage applied."""

    resolved = deepcopy(dict(config))
    if stage_id is None:
        return resolved
    stages = list(resolved.get("stair_height_stages", ()))
    matches = [stage for stage in stages if str(stage["id"]) == stage_id]
    if len(matches) != 1:
        available = [str(stage["id"]) for stage in stages]
        raise ValueError(
            f"Unknown stair height stage {stage_id!r}; available={available}"
        )
    stage = dict(matches[0])
    task = dict(resolved["task"])
    staircase = dict(task["staircase"])
    rise_m = float(stage["rise_m"])
    if rise_m <= 0.0:
        raise ValueError("Stair height stage rise_m must be positive")
    task["id"] = str(stage["task_id"])
    task["world"] = str(stage["world"])
    staircase["rise_m"] = rise_m
    task["staircase"] = staircase
    resolved["task"] = task
    resolved["selected_stair_height_stage"] = stage
    return resolved


def _finite_vector(value, length: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain {length} finite values")
    return vector


def validate_staircase_config(staircase: Mapping[str, object]) -> None:
    """Reject stair descriptions that cannot define a physical staircase."""

    positive_fields = (
        "step_count",
        "tread_depth_m",
        "rise_m",
        "width_m",
        "top_platform_depth_m",
        "terrain_height_normalization_m",
        "goal_distance_normalization_m",
    )
    for field in positive_fields:
        if float(staircase[field]) <= 0.0:
            raise ValueError(f"staircase.{field} must be positive")
    if int(staircase["step_count"]) != float(staircase["step_count"]):
        raise ValueError("staircase.step_count must be an integer")
    offsets = np.asarray(
        staircase["terrain_sample_offsets_m"],
        dtype=np.float32,
    ).reshape(-1)
    if offsets.size < 2 or not np.all(np.isfinite(offsets)):
        raise ValueError(
            "staircase.terrain_sample_offsets_m needs at least two finite samples"
        )
    if np.any(np.diff(offsets) <= 0.0):
        raise ValueError(
            "staircase.terrain_sample_offsets_m must be strictly increasing"
        )


def stair_height_at_x(
    world_x_m: float,
    staircase: Mapping[str, object],
) -> float:
    """Return the authored walking-surface height at a world-X coordinate."""

    validate_staircase_config(staircase)
    x = float(world_x_m)
    start = float(staircase["start_x_m"])
    if x < start:
        return 0.0
    step_count = int(staircase["step_count"])
    tread = float(staircase["tread_depth_m"])
    rise = float(staircase["rise_m"])
    stair_end = start + step_count * tread
    platform_end = stair_end + float(staircase["top_platform_depth_m"])
    if x >= platform_end - 1e-9:
        return 0.0
    if x >= stair_end - 1e-9:
        return step_count * rise
    step_index = int(np.floor((x - start) / tread + 1e-9))
    return (step_index + 1) * rise


def stair_index_at_x(
    world_x_m: float,
    staircase: Mapping[str, object],
) -> int:
    """Return zero on the approach and 1..N on stairs/top platform."""

    x = float(world_x_m)
    start = float(staircase["start_x_m"])
    if x < start:
        return 0
    step_count = int(staircase["step_count"])
    tread = float(staircase["tread_depth_m"])
    stair_end = start + step_count * tread
    platform_end = stair_end + float(staircase["top_platform_depth_m"])
    if x >= platform_end - 1e-9:
        return 0
    if x >= stair_end - 1e-9:
        return step_count
    return min(step_count, int(np.floor((x - start) / tread + 1e-9)) + 1)


def curriculum_active_steps(
    progress_fraction: float,
    levels: Sequence[Mapping[str, object]],
    *,
    maximum_steps: int,
) -> int:
    """Select the deterministic curriculum level for training progress."""

    progress = float(np.clip(progress_fraction, 0.0, 1.0))
    if not levels:
        raise ValueError("curriculum levels cannot be empty")
    active = None
    previous_start = -1.0
    for level in levels:
        start = float(level["start_fraction"])
        step_count = int(level["active_steps"])
        if start < 0.0 or start > 1.0 or start <= previous_start:
            raise ValueError(
                "curriculum start_fraction values must increase within [0, 1]"
            )
        if step_count < 1 or step_count > int(maximum_steps):
            raise ValueError("curriculum active_steps is outside the staircase")
        if progress >= start:
            active = step_count
        previous_start = start
    if active is None:
        raise ValueError("the first curriculum level must start at fraction 0")
    return active


def placement_curriculum_level(
    progress_fraction: float,
    levels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select one ordered single-tread placement stage."""

    progress = float(np.clip(progress_fraction, 0.0, 1.0))
    if not levels:
        raise ValueError("placement curriculum levels cannot be empty")
    selected: dict[str, object] | None = None
    previous_start = -1.0
    for raw_level in levels:
        level = dict(raw_level)
        start = float(level["start_fraction"])
        if start < 0.0 or start > 1.0 or start <= previous_start:
            raise ValueError(
                "placement curriculum start fractions must increase within [0, 1]"
            )
        apex = float(level["apex_lift_m"])
        landing = float(level["landing_lift_m"])
        forward = float(level["swing_forward_offset_m"])
        lift_forward = float(level.get("lift_forward_offset_m", min(forward, 0.11)))
        landing_forward = float(level.get("landing_forward_offset_m", forward))
        success_mode = str(level.get("success_mode", "tread_contact"))
        tread_fraction = float(level["target_tread_fraction"])
        if apex <= 0.0 or landing <= 0.0 or landing > apex:
            raise ValueError("placement lift heights must satisfy 0 < landing <= apex")
        if forward < 0.0 or (
            forward == 0.0 and success_mode != "swing_lift_hold"
        ):
            raise ValueError(
                "placement swing_forward_offset_m must be positive unless "
                "the stage is a pure swing lift hold"
            )
        if lift_forward < 0.0 or lift_forward > forward:
            raise ValueError(
                "placement lift_forward_offset_m must be within "
                "[0, swing forward]"
            )
        if landing_forward < 0.0 or landing_forward > forward:
            raise ValueError(
                "placement landing_forward_offset_m must be within "
                "[0, swing forward]"
            )
        if tread_fraction <= 0.0 or tread_fraction >= 1.0:
            raise ValueError("placement target_tread_fraction must be within (0, 1)")
        if progress >= start:
            selected = level
        previous_start = start
    if selected is None:
        raise ValueError("the first placement curriculum level must start at zero")
    return selected


def placement_success_mode(
    *,
    swing_leg: str,
    default_mode: str = "tread_contact",
    mode_by_leg: Mapping[str, object] | None = None,
    active_level: Mapping[str, object] | None = None,
) -> str:
    """Resolve level, leg, then task success mode for a placement stage."""

    leg_modes = dict(mode_by_leg or {})
    level = dict(active_level or {})
    mode = str(
        level.get(
            "success_mode",
            leg_modes.get(str(swing_leg), default_mode),
        )
    )
    if mode not in {"tread_contact", "swing_lift_hold"}:
        raise ValueError(
            "placement success mode must be tread_contact or swing_lift_hold"
        )
    return mode


def placement_reference_state(
    elapsed_seconds: float,
    *,
    timing: Mapping[str, float],
    level: Mapping[str, object],
) -> dict[str, object]:
    """Return the explicit shift, lift/advance, lower, and hold reference."""

    elapsed = max(0.0, float(elapsed_seconds))
    shift_start = float(timing["shift_start_seconds"])
    shift_duration = float(timing["shift_duration_seconds"])
    lift_start = float(timing["lift_start_seconds"])
    lift_duration = float(timing["lift_duration_seconds"])
    advance_start = float(timing["advance_start_seconds"])
    advance_duration = float(timing["advance_duration_seconds"])
    lower_start = float(timing["lower_start_seconds"])
    lower_duration = float(timing["lower_duration_seconds"])
    if min(shift_start, lift_start, advance_start, lower_start) < 0.0:
        raise ValueError("placement phase start times cannot be negative")
    if min(shift_duration, lift_duration, advance_duration, lower_duration) <= 0.0:
        raise ValueError("placement phase durations must be positive")
    if lift_start < shift_start + shift_duration - 1e-9:
        raise ValueError("placement lift must start after the weight shift")
    if advance_start < lift_start + lift_duration - 1e-9:
        raise ValueError("placement advance must start after lift")
    if lower_start < advance_start + advance_duration - 1e-9:
        raise ValueError("placement lower must start after advance")

    def smoothstep(value: float) -> float:
        clipped = float(np.clip(value, 0.0, 1.0))
        return clipped * clipped * (3.0 - 2.0 * clipped)

    shift_fraction = smoothstep((elapsed - shift_start) / shift_duration)
    lift_fraction = smoothstep((elapsed - lift_start) / lift_duration)
    advance_fraction = smoothstep(
        (elapsed - advance_start) / advance_duration
    )
    lower_fraction = smoothstep((elapsed - lower_start) / lower_duration)
    apex_lift = float(level["apex_lift_m"])
    landing_lift = float(level["landing_lift_m"])
    desired_lift = apex_lift * lift_fraction
    if elapsed >= lower_start:
        desired_lift = apex_lift + lower_fraction * (landing_lift - apex_lift)
    final_forward = float(level["swing_forward_offset_m"])
    lift_forward = min(
        final_forward,
        float(level.get("lift_forward_offset_m", 0.11)),
    )
    desired_forward = lift_forward * lift_fraction
    if elapsed >= advance_start:
        desired_forward = lift_forward + advance_fraction * (
            final_forward - lift_forward
        )
    if elapsed >= lower_start:
        landing_forward = float(
            level.get("landing_forward_offset_m", final_forward)
        )
        desired_forward = final_forward + lower_fraction * (
            landing_forward - final_forward
        )
    forward_fraction = (
        0.0 if final_forward == 0.0 else desired_forward / final_forward
    )
    if elapsed < lift_start:
        phase = "weight_shift"
    elif elapsed < advance_start:
        phase = "lift"
    elif elapsed < lower_start:
        phase = "advance"
    elif elapsed < lower_start + lower_duration:
        phase = "lower"
    else:
        phase = "hold"
    return {
        "elapsed_seconds": elapsed,
        "phase": phase,
        "phase_one_hot": tuple(float(phase == name) for name in PLACEMENT_PHASES),
        "shift_fraction": shift_fraction,
        "lift_fraction": lift_fraction,
        "advance_fraction": advance_fraction,
        "forward_fraction": forward_fraction,
        "lower_fraction": lower_fraction,
        "desired_lift_m": desired_lift,
        "desired_forward_offset_m": desired_forward,
        "contact_expected": phase in {"lower", "hold"},
    }


def placement_advance_clearance_gate_state(
    *,
    candidate_phase: str,
    measured_clearance_m: float,
    minimum_clearance_m: float,
    held_steps: int,
    maximum_hold_steps: int,
) -> dict[str, object]:
    """Gate forward swing on measured clearance with a bounded safe hold."""

    phase = str(candidate_phase)
    if phase not in PLACEMENT_PHASES:
        raise ValueError(f"unknown placement phase: {phase}")
    measured = float(measured_clearance_m)
    minimum = float(minimum_clearance_m)
    if not np.isfinite(measured) or not np.isfinite(minimum):
        raise ValueError("clearance values must be finite")
    if minimum <= 0.0:
        raise ValueError("minimum clearance must be positive")
    held = int(held_steps)
    maximum = int(maximum_hold_steps)
    if held < 0 or maximum < 1:
        raise ValueError("clearance hold steps must be non-negative and bounded")

    advance_due = phase in {"advance", "lower", "hold"}
    released = bool(advance_due and measured >= minimum)
    hold_reference = bool(advance_due and not released)
    next_held = held + int(hold_reference)
    return {
        "advance_due": advance_due,
        "released": released,
        "hold_reference": hold_reference,
        "held_steps": next_held,
        "timed_out": bool(hold_reference and next_held >= maximum),
        "clearance_error_m": max(0.0, minimum - measured),
    }


def inter_leg_transfer_state(
    elapsed_seconds: float,
    *,
    duration_seconds: float,
    unload_duration_seconds: float = 0.0,
    unload_elapsed_seconds: float | None = None,
) -> dict[str, object]:
    """Return a smooth transfer with an optionally gated unload clock."""

    duration = float(duration_seconds)
    if duration <= 0.0:
        raise ValueError("inter-leg transfer duration must be positive")
    unload_duration = float(unload_duration_seconds)
    if unload_duration < 0.0:
        raise ValueError("inter-leg unload duration cannot be negative")
    elapsed = max(0.0, float(elapsed_seconds))
    linear_fraction = float(
        np.clip(elapsed / duration, 0.0, 1.0)
    )
    transfer_fraction = linear_fraction * linear_fraction * (
        3.0 - 2.0 * linear_fraction
    )
    unload_elapsed = (
        max(0.0, elapsed - duration)
        if unload_elapsed_seconds is None
        else max(0.0, float(unload_elapsed_seconds))
    )
    unload_linear_fraction = (
        float(np.clip(unload_elapsed / unload_duration, 0.0, 1.0))
        if unload_duration > 0.0
        else 1.0
    )
    unload_fraction = unload_linear_fraction * unload_linear_fraction * (
        3.0 - 2.0 * unload_linear_fraction
    )
    return {
        "phase": "weight_shift",
        "phase_one_hot": tuple(
            float(name == "weight_shift") for name in PLACEMENT_PHASES
        ),
        "shift_fraction": 1.0,
        "lift_fraction": 0.0,
        "advance_fraction": 0.0,
        "forward_fraction": 0.0,
        "lower_fraction": 0.0,
        "desired_lift_m": 0.0,
        "desired_forward_offset_m": 0.0,
        "contact_expected": False,
        "transfer_fraction": transfer_fraction,
        "unload_fraction": unload_fraction,
        "unload_elapsed_seconds": unload_elapsed,
        "transfer_stage": (
            "shift"
            if transfer_fraction < 1.0 - 1e-9
            else (
                "pre_unload_settle"
                if unload_fraction <= 1e-9
                else ("unload" if unload_fraction < 1.0 - 1e-9 else "gate")
            )
        ),
    }


def inter_leg_pre_unload_gate_failures(
    *,
    transfer_fraction: float,
    support_contact_fraction: float,
    completed_tread_loaded: bool,
    next_swing_total_load_n: float,
    minimum_next_swing_preload_n: float,
    support_margin_m: float,
    minimum_support_margin_m: float,
    balance_target_error_m: float,
    maximum_balance_target_error_m: float,
    base_speed_m_s: float,
    maximum_base_speed_m_s: float,
    body_rate_rad_s: float,
    maximum_body_rate_rad_s: float,
    upright_cosine: float,
    minimum_upright_cosine: float,
) -> tuple[str, ...]:
    """Return reasons a four-foot transfer state is not ready to unload."""

    values = np.asarray(
        [
            transfer_fraction,
            support_contact_fraction,
            next_swing_total_load_n,
            minimum_next_swing_preload_n,
            support_margin_m,
            minimum_support_margin_m,
            balance_target_error_m,
            maximum_balance_target_error_m,
            base_speed_m_s,
            maximum_base_speed_m_s,
            body_rate_rad_s,
            maximum_body_rate_rad_s,
            upright_cosine,
            minimum_upright_cosine,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("pre-unload gate inputs must be finite")
    if any(
        value < 0.0
        for value in (
            minimum_next_swing_preload_n,
            maximum_balance_target_error_m,
            maximum_base_speed_m_s,
            maximum_body_rate_rad_s,
        )
    ):
        raise ValueError("pre-unload gate thresholds cannot be negative")

    failures: list[str] = []
    if float(transfer_fraction) < 1.0 - 1e-6:
        failures.append("transfer_incomplete")
    if float(support_contact_fraction) < 1.0 - 1e-6:
        failures.append("support_contact_lost")
    if not bool(completed_tread_loaded):
        failures.append("placed_tread_unloaded")
    if float(next_swing_total_load_n) < float(minimum_next_swing_preload_n):
        failures.append("next_swing_not_preloaded")
    if float(support_margin_m) < float(minimum_support_margin_m):
        failures.append("support_margin_low")
    if float(balance_target_error_m) > float(maximum_balance_target_error_m):
        failures.append("balance_target_error_high")
    if float(base_speed_m_s) > float(maximum_base_speed_m_s):
        failures.append("base_not_settled")
    if float(body_rate_rad_s) > float(maximum_body_rate_rad_s):
        failures.append("body_rate_high")
    if float(upright_cosine) < float(minimum_upright_cosine):
        failures.append("body_not_upright")
    return tuple(failures)


def support_triangle_incenter_xy(
    support_points_xy_m: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return the point with equal distance to all three support edges."""

    points = np.asarray(support_points_xy_m, dtype=np.float64)
    if points.shape != (3, 2) or not np.all(np.isfinite(points)):
        raise ValueError("support_points_xy_m must be a finite 3x2 triangle")
    first_edge = points[1] - points[0]
    second_edge = points[2] - points[0]
    twice_area = abs(
        float(
            first_edge[0] * second_edge[1]
            - first_edge[1] * second_edge[0]
        )
    )
    if twice_area <= 1e-9:
        raise ValueError("support triangle must have nonzero area")
    opposite_edge_lengths = np.asarray(
        [
            np.linalg.norm(points[1] - points[2]),
            np.linalg.norm(points[0] - points[2]),
            np.linalg.norm(points[0] - points[1]),
        ],
        dtype=np.float64,
    )
    perimeter = float(np.sum(opposite_edge_lengths))
    if perimeter <= 0.0:
        raise ValueError("support triangle perimeter must be positive")
    return (
        np.sum(points * opposite_edge_lengths[:, None], axis=0) / perimeter
    ).astype(np.float32)


def bounded_support_incenter_target_xy(
    *,
    reference_point_xy_m: Sequence[float],
    support_points_xy_m: Sequence[Sequence[float]],
    incenter_blend: float,
    target_offset_xy_m: Sequence[float] = (0.0, 0.0),
    maximum_shift_xy_m: Sequence[float] = (0.12, 0.12),
) -> np.ndarray:
    """Return a bounded balance target inside a three-foot support polygon."""

    reference = _finite_vector(
        reference_point_xy_m,
        2,
        "reference_point_xy_m",
    )
    offset = _finite_vector(
        target_offset_xy_m,
        2,
        "target_offset_xy_m",
    )
    maximum_shift = _finite_vector(
        maximum_shift_xy_m,
        2,
        "maximum_shift_xy_m",
    )
    blend = float(incenter_blend)
    if blend <= 0.0 or blend > 1.0:
        raise ValueError("incenter_blend must be within (0, 1]")
    if np.any(maximum_shift <= 0.0):
        raise ValueError("maximum_shift_xy_m values must be positive")
    incenter = np.asarray(
        support_triangle_incenter_xy(support_points_xy_m),
        dtype=np.float64,
    )
    desired_delta = blend * (incenter - reference) + offset
    return (
        reference
        + np.clip(desired_delta, -maximum_shift, maximum_shift)
    ).astype(np.float64)


def support_margin_constrained_target_xy(
    *,
    desired_target_xy_m: Sequence[float],
    support_points_xy_m: Sequence[Sequence[float]],
    minimum_margin_m: float,
) -> np.ndarray:
    """Clip a desired COM target to a safe three-foot support inset.

    The returned point is the farthest point toward ``desired_target_xy_m``
    on the line from the support triangle's incenter that retains the requested
    edge clearance. This keeps a forward transfer command useful while making
    the support-margin contract authoritative instead of relying on an
    unconstrained open-loop body shift.
    """

    desired = _finite_vector(
        desired_target_xy_m,
        2,
        "desired_target_xy_m",
    ).astype(np.float64)
    points = np.asarray(support_points_xy_m, dtype=np.float64)
    if points.shape != (3, 2) or not np.all(np.isfinite(points)):
        raise ValueError("support_points_xy_m must be a finite 3x2 triangle")
    minimum_margin = float(minimum_margin_m)
    if not np.isfinite(minimum_margin) or minimum_margin < 0.0:
        raise ValueError("minimum_margin_m must be finite and nonnegative")

    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]

    def signed_margin(point: np.ndarray) -> float:
        margins: list[float] = []
        for index in range(3):
            start = ordered[index]
            edge = ordered[(index + 1) % 3] - start
            edge_length = float(np.linalg.norm(edge))
            if edge_length <= 1e-9:
                raise ValueError("support triangle points must be distinct")
            relative = point - start
            margins.append(
                float(
                    (edge[0] * relative[1] - edge[1] * relative[0])
                    / edge_length
                )
            )
        return min(margins)

    incenter = np.asarray(
        support_triangle_incenter_xy(points),
        dtype=np.float64,
    )
    incenter_margin = signed_margin(incenter)
    if minimum_margin > incenter_margin + 1e-9:
        raise ValueError(
            "minimum_margin_m exceeds the support triangle inradius"
        )
    if signed_margin(desired) >= minimum_margin:
        return desired

    low = 0.0
    high = 1.0
    for _ in range(48):
        midpoint = 0.5 * (low + high)
        candidate = incenter + midpoint * (desired - incenter)
        if signed_margin(candidate) >= minimum_margin:
            low = midpoint
        else:
            high = midpoint
    return (incenter + low * (desired - incenter)).astype(np.float64)


def touchdown_load_lift_correction_m(
    *,
    measured_tread_load_n: float,
    target_tread_load_n: float,
    proportional_gain_m_per_n: float,
    maximum_lift_correction_m: float,
) -> float:
    """Return an upward swing-foot correction for excess touchdown load."""

    measured = float(measured_tread_load_n)
    target = float(target_tread_load_n)
    gain = float(proportional_gain_m_per_n)
    maximum = float(maximum_lift_correction_m)
    if not np.isfinite(measured) or measured < 0.0:
        raise ValueError("measured_tread_load_n must be finite and nonnegative")
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("target_tread_load_n must be finite and positive")
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError(
            "proportional_gain_m_per_n must be finite and positive"
        )
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(
            "maximum_lift_correction_m must be finite and positive"
        )
    return float(np.clip((measured - target) * gain, 0.0, maximum))


def balance_target_error_xy(
    *,
    balance_position_xy_m: Sequence[float],
    target_position_xy_m: Sequence[float],
) -> np.ndarray:
    """Return the signed whole-robot balance error in the target frame."""

    balance = _finite_vector(
        balance_position_xy_m,
        2,
        "balance_position_xy_m",
    )
    target = _finite_vector(
        target_position_xy_m,
        2,
        "target_position_xy_m",
    )
    return balance.astype(np.float64) - target.astype(np.float64)


def support_load_share_vertical_corrections(
    *,
    support_points_xy_m: Sequence[Sequence[float]],
    target_position_xy_m: Sequence[float],
    measured_normal_loads_n: Sequence[float],
    proportional_gain_m: float,
    maximum_correction_m: float,
    minimum_total_load_n: float = 1.0,
    minimum_desired_fraction: float = 0.05,
) -> np.ndarray:
    """Return zero-sum stance-leg extension corrections from load error."""

    points = np.asarray(support_points_xy_m, dtype=np.float64)
    if points.shape != (3, 2) or not np.all(np.isfinite(points)):
        raise ValueError("support_points_xy_m must contain three finite XY points")
    target = _finite_vector(
        target_position_xy_m,
        2,
        "target_position_xy_m",
    ).astype(np.float64)
    loads = _finite_vector(
        measured_normal_loads_n,
        3,
        "measured_normal_loads_n",
    ).astype(np.float64)
    if np.any(loads < 0.0):
        raise ValueError("measured_normal_loads_n must be nonnegative")
    gain = float(proportional_gain_m)
    maximum = float(maximum_correction_m)
    minimum_total = float(minimum_total_load_n)
    minimum_fraction = float(minimum_desired_fraction)
    if gain <= 0.0 or not np.isfinite(gain):
        raise ValueError("proportional_gain_m must be finite and positive")
    if maximum <= 0.0 or not np.isfinite(maximum):
        raise ValueError("maximum_correction_m must be finite and positive")
    if minimum_total < 0.0 or not np.isfinite(minimum_total):
        raise ValueError("minimum_total_load_n must be finite and nonnegative")
    if not 0.0 <= minimum_fraction < 1.0 / 3.0:
        raise ValueError("minimum_desired_fraction must be within [0, 1/3)")
    total_load = float(np.sum(loads))
    if total_load < minimum_total:
        return np.zeros(3, dtype=np.float64)

    barycentric_matrix = np.asarray(
        [
            [points[0, 0], points[1, 0], points[2, 0]],
            [points[0, 1], points[1, 1], points[2, 1]],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    determinant = float(np.linalg.det(barycentric_matrix))
    if abs(determinant) < 1e-9:
        raise ValueError("support points do not form a triangle")
    desired_fractions = np.linalg.solve(
        barycentric_matrix,
        np.asarray([target[0], target[1], 1.0], dtype=np.float64),
    )
    desired_fractions = np.maximum(desired_fractions, minimum_fraction)
    desired_fractions /= float(np.sum(desired_fractions))
    measured_fractions = loads / total_load
    correction = gain * (desired_fractions - measured_fractions)
    correction -= float(np.mean(correction))
    correction = np.clip(correction, -maximum, maximum)
    correction -= float(np.mean(correction))
    correction = np.clip(correction, -maximum, maximum)
    correction[np.abs(correction) < 1e-9] = 0.0
    return correction.astype(np.float64)


def joint_effort_telemetry_sample(
    *,
    target_joint_positions_rad: Sequence[float],
    measured_joint_positions_rad: Sequence[float],
    joint_velocities_rad_s: Sequence[float] | None = None,
    drive_stiffness_nm_rad: Sequence[float] | None = None,
    drive_damping_nm_s_rad: Sequence[float] | None = None,
    effort_cap_nm: float,
    reported_actuation_effort_nm: Sequence[float] | None = None,
    projected_joint_reaction_load_nm: Sequence[float] | None = None,
) -> dict[str, np.ndarray | float]:
    """Return finite tracking, implicit-drive demand, and load telemetry."""

    target = _finite_vector(
        target_joint_positions_rad,
        JOINT_COUNT,
        "target_joint_positions_rad",
    )
    measured = _finite_vector(
        measured_joint_positions_rad,
        JOINT_COUNT,
        "measured_joint_positions_rad",
    )
    cap = float(effort_cap_nm)
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("effort_cap_nm must be finite and positive")
    result: dict[str, np.ndarray | float] = {
        "joint_tracking_error_rad": (target - measured).astype(np.float64),
    }
    pd_inputs = (
        joint_velocities_rad_s,
        drive_stiffness_nm_rad,
        drive_damping_nm_s_rad,
    )
    if any(value is not None for value in pd_inputs):
        if not all(value is not None for value in pd_inputs):
            raise ValueError(
                "joint velocities, drive stiffness, and drive damping must "
                "be provided together"
            )
        velocities = _finite_vector(
            joint_velocities_rad_s,
            JOINT_COUNT,
            "joint_velocities_rad_s",
        )
        stiffness = _finite_vector(
            drive_stiffness_nm_rad,
            JOINT_COUNT,
            "drive_stiffness_nm_rad",
        )
        damping = _finite_vector(
            drive_damping_nm_s_rad,
            JOINT_COUNT,
            "drive_damping_nm_s_rad",
        )
        if np.any(stiffness < 0.0) or np.any(damping < 0.0):
            raise ValueError("drive gains must be non-negative")
        requested = stiffness * (target - measured) - damping * velocities
        result["requested_pd_effort_nm"] = requested.astype(np.float64)
        result["capped_pd_effort_nm"] = np.clip(
            requested,
            -cap,
            cap,
        ).astype(np.float64)
        result["requested_pd_effort_nm_peak_to_cap_ratio"] = float(
            np.max(np.abs(requested)) / cap
        )
        result["requested_pd_effort_nm_95pct_cap_fraction"] = float(
            np.mean(np.abs(requested) >= 0.95 * cap - 1e-6)
        )
    for label, values in (
        ("reported_actuation_effort_nm", reported_actuation_effort_nm),
        (
            "projected_joint_reaction_load_nm",
            projected_joint_reaction_load_nm,
        ),
    ):
        if values is None:
            continue
        vector = _finite_vector(values, JOINT_COUNT, label).astype(np.float64)
        result[label] = vector
    return result


def placement_phase_ready(
    *,
    sequence_legs: Sequence[str],
    completed_legs: Sequence[str],
    active_leg: str,
    transfer_active: bool,
    target_leg: str,
) -> bool:
    """Return whether a post-transfer target-leg phase is ready for PPO."""

    sequence = tuple(str(leg) for leg in sequence_legs)
    if target_leg not in sequence:
        raise ValueError(f"Unknown placement phase target: {target_leg}")
    target_position = sequence.index(target_leg)
    completed = set(str(leg) for leg in completed_legs)
    return bool(
        not transfer_active
        and active_leg == target_leg
        and all(leg in completed for leg in sequence[:target_position])
    )


def placement_transfer_ready(
    *,
    sequence_legs: Sequence[str],
    completed_legs: Sequence[str],
    active_leg: str,
    transfer_active: bool,
    target_leg: str,
) -> bool:
    """Return whether PPO should control the transfer into ``target_leg``."""

    sequence = tuple(str(leg) for leg in sequence_legs)
    if target_leg not in sequence:
        raise ValueError(f"Unknown placement transfer target: {target_leg}")
    target_position = sequence.index(target_leg)
    if target_position == 0:
        return False
    completed = tuple(str(leg) for leg in completed_legs)
    return bool(
        transfer_active
        and active_leg == target_leg
        and completed == sequence[:target_position]
    )


def compose_bounded_residual_action(
    base_action: Sequence[float],
    residual_action: Sequence[float],
    *,
    residual_scale: float,
    residual_mask: Sequence[float] | None = None,
) -> np.ndarray:
    """Add a bounded corrective action to a frozen base-policy action."""

    base = np.asarray(base_action, dtype=np.float32)
    residual = np.asarray(residual_action, dtype=np.float32)
    if base.shape != residual.shape or base.ndim != 1:
        raise ValueError("base and residual actions must be matching vectors")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(residual)):
        raise ValueError("base and residual actions must be finite")
    mask = (
        np.ones_like(base)
        if residual_mask is None
        else np.asarray(residual_mask, dtype=np.float32)
    )
    if mask.shape != base.shape or not np.all(np.isfinite(mask)):
        raise ValueError("residual_mask must match the finite action vector")
    if np.any(mask < 0.0) or np.any(mask > 1.0):
        raise ValueError("residual_mask values must be within [0, 1]")
    scale = float(residual_scale)
    if scale <= 0.0 or scale > 1.0:
        raise ValueError("residual_scale must be within (0, 1]")
    return np.clip(base + scale * mask * residual, -1.0, 1.0).astype(np.float32)


def overlay_masked_action(
    base_action: Sequence[float],
    overlay_action: Sequence[float],
    action_mask: Sequence[float],
) -> np.ndarray:
    """Replace selected joints in ``base_action`` with an overlay policy.

    This keeps a swing-leg policy active while a disjoint support policy spans
    the controller handoff immediately after an inter-leg transfer.
    """

    base = np.asarray(base_action, dtype=np.float32)
    overlay = np.asarray(overlay_action, dtype=np.float32)
    mask = np.asarray(action_mask, dtype=np.float32)
    if base.ndim != 1 or overlay.shape != base.shape or mask.shape != base.shape:
        raise ValueError("base, overlay, and action_mask must be matching vectors")
    if not all(np.all(np.isfinite(value)) for value in (base, overlay, mask)):
        raise ValueError("masked action inputs must be finite")
    if np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("action_mask must be binary")
    return np.clip(
        base * (1.0 - mask) + overlay * mask,
        -1.0,
        1.0,
    ).astype(np.float32)


def expand_compact_masked_action(
    compact_action: Sequence[float],
    action_mask: Sequence[float],
) -> np.ndarray:
    """Expand a policy action onto the active entries of a full joint mask."""

    compact = np.asarray(compact_action, dtype=np.float32)
    mask = np.asarray(action_mask, dtype=np.float32)
    if compact.ndim != 1 or mask.ndim != 1:
        raise ValueError("compact action and action_mask must be vectors")
    if not np.all(np.isfinite(compact)) or not np.all(np.isfinite(mask)):
        raise ValueError("compact action and action_mask must be finite")
    if np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("action_mask must be binary")
    active_indices = np.flatnonzero(mask)
    if compact.shape != active_indices.shape:
        raise ValueError(
            "compact action size must match the active action_mask entries: "
            f"{compact.shape} != {active_indices.shape}"
        )
    expanded = np.zeros(mask.shape, dtype=np.float32)
    expanded[active_indices] = compact
    return expanded


def placement_policy_action_mask(
    dof_names: Sequence[str],
    *,
    target_leg: str,
    mode: str,
) -> np.ndarray:
    """Select the joints a phase-specific placement policy may command."""

    names = tuple(str(name) for name in dof_names)
    target_prefix = f"{target_leg}_"
    if not any(name.startswith(target_prefix) for name in names):
        raise ValueError(f"unknown placement target leg: {target_leg}")
    if mode == "support_only":
        selected = [not name.startswith(target_prefix) for name in names]
    elif mode == "swing_only":
        selected = [name.startswith(target_prefix) for name in names]
    elif mode == "support_abduction_only":
        selected = [
            not name.startswith(target_prefix)
            and name.endswith("_hip_abduction")
            for name in names
        ]
    elif mode == "swing_plus_support_abduction":
        selected = [
            name.startswith(target_prefix)
            or name.endswith("_hip_abduction")
            for name in names
        ]
    else:
        raise ValueError(f"unknown placement action mask mode: {mode}")
    return np.asarray(selected, dtype=np.float32)


def stabilized_support_reference_base_delta(
    *,
    desired_base_delta_m: Sequence[float],
    actual_base_delta_m: Sequence[float],
    anchor_follow_gain: float | Sequence[float],
    error_feedback_gain_xyz: Sequence[float],
) -> np.ndarray:
    """Return a support reference that actively rejects post-transfer drift.

    A zero feedback gain preserves the existing anchor-follow blend. Positive
    feedback moves the virtual support-foot target beyond the desired body
    pose, increasing the joint-position error that restores the base. This is
    deliberately a proprioceptive controller: it uses the simulated base pose,
    not camera input or privileged stair geometry.
    """

    desired = _finite_vector(
        desired_base_delta_m,
        3,
        "desired_base_delta_m",
    )
    actual = _finite_vector(
        actual_base_delta_m,
        3,
        "actual_base_delta_m",
    )
    feedback = _finite_vector(
        error_feedback_gain_xyz,
        3,
        "error_feedback_gain_xyz",
    )
    follow = (
        np.full(3, float(anchor_follow_gain), dtype=np.float64)
        if np.isscalar(anchor_follow_gain)
        else _finite_vector(
            anchor_follow_gain,
            3,
            "anchor_follow_gain",
        )
    )
    if np.any(follow < 0.0) or np.any(follow > 1.0):
        raise ValueError("anchor_follow_gain values must be within [0, 1]")
    if np.any(feedback < 0.0) or np.any(feedback > 2.0):
        raise ValueError("error feedback gains must be within [0, 2]")
    tracking_error = actual - desired
    return (
        desired + (follow - feedback) * tracking_error
    ).astype(np.float64)


def staged_swing_reference_base_delta(
    *,
    base_delta_m: Sequence[float],
    advance_fraction: float,
    end_scale_xyz: Sequence[float],
) -> np.ndarray:
    """Release a swing foot from its world anchor only after clearance.

    The placement controller normally subtracts the desired base motion from
    the swing reference so the raised foot remains fixed in world space.  A
    scale of one preserves that behavior. During the advance phase this helper
    smoothly approaches ``end_scale_xyz``; a forward end scale of zero lets
    body translation carry the cleared foot without changing the lift
    trajectory or actuator authority.
    """

    base_delta = _finite_vector(base_delta_m, 3, "base_delta_m").astype(
        np.float64
    )
    end_scale = _finite_vector(end_scale_xyz, 3, "end_scale_xyz").astype(
        np.float64
    )
    fraction = float(advance_fraction)
    if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError("advance_fraction must be finite and within [0, 1]")
    if np.any(end_scale < 0.0) or np.any(end_scale > 1.0):
        raise ValueError("end_scale_xyz values must be within [0, 1]")
    scale = 1.0 + fraction * (end_scale - 1.0)
    return (base_delta * scale).astype(np.float64)


def split_post_clearance_advance_fractions(
    *,
    advance_fraction: float,
    body_shift_fraction_of_advance: float,
    sequence: str = "body_then_swing",
) -> tuple[float, float]:
    """Sequence body shift and swing advance inside one clearance gate."""

    fraction = float(advance_fraction)
    split = float(body_shift_fraction_of_advance)
    if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError("advance_fraction must be finite and within [0, 1]")
    if not np.isfinite(split) or split <= 0.0 or split >= 1.0:
        raise ValueError(
            "body_shift_fraction_of_advance must be finite and within (0, 1)"
        )
    if sequence not in {"body_then_swing", "swing_then_body"}:
        raise ValueError(
            "sequence must be body_then_swing or swing_then_body"
        )

    def smoothstep(value: float) -> float:
        clipped = float(np.clip(value, 0.0, 1.0))
        return clipped * clipped * (3.0 - 2.0 * clipped)

    if sequence == "body_then_swing":
        body_shift_fraction = smoothstep(fraction / split)
        swing_advance_fraction = smoothstep(
            (fraction - split) / (1.0 - split)
        )
    else:
        swing_fraction = 1.0 - split
        swing_advance_fraction = smoothstep(fraction / swing_fraction)
        body_shift_fraction = smoothstep(
            (fraction - swing_fraction) / split
        )
    return body_shift_fraction, swing_advance_fraction


def support_pitch_vertical_corrections(
    *,
    support_legs: Sequence[str],
    projected_gravity_x: float,
    proportional_gain_m: float,
    maximum_correction_m: float,
) -> dict[str, float]:
    """Level sagittal attitude by differentially extending stance legs.

    For this robot's IMU/joint convention, negative projected gravity X is
    corrected by shortening front stance legs and extending rear stance legs.
    This creates the restoring pitch moment without camera input.
    """

    gravity_x = float(projected_gravity_x)
    gain = float(proportional_gain_m)
    maximum = float(maximum_correction_m)
    legs = tuple(str(leg) for leg in support_legs)
    if not np.isfinite(gravity_x):
        raise ValueError("projected_gravity_x must be finite")
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("proportional_gain_m must be finite and positive")
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_correction_m must be finite and positive")
    if not legs or any(leg not in STAIR_FOOT_NAMES for leg in legs):
        raise ValueError("support_legs must contain known robot legs")

    correction = float(np.clip(gain * gravity_x, -maximum, maximum))
    return {
        leg: correction if leg.startswith("front_") else -correction
        for leg in legs
    }


def staged_support_rear_pitch_scale(
    *,
    elapsed_seconds: float,
    front_only_seconds: float,
    blend_seconds: float,
) -> float:
    """Smoothly restore rear-stance pitch correction after a front-only catch."""

    elapsed = float(elapsed_seconds)
    hold = float(front_only_seconds)
    blend = float(blend_seconds)
    if not np.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("elapsed_seconds must be finite and nonnegative")
    if not np.isfinite(hold) or hold < 0.0:
        raise ValueError("front_only_seconds must be finite and nonnegative")
    if not np.isfinite(blend) or blend < 0.0:
        raise ValueError("blend_seconds must be finite and nonnegative")
    if elapsed <= hold:
        return 0.0
    if blend == 0.0:
        return 1.0
    fraction = float(np.clip((elapsed - hold) / blend, 0.0, 1.0))
    return fraction * fraction * (3.0 - 2.0 * fraction)


def placement_contact_reached(
    *,
    swing_tip_position_m: Sequence[float],
    swing_tread_normal_load_n: float,
    support_ground_normal_loads_n: Sequence[float],
    projected_gravity_xyz: Sequence[float],
    staircase: Mapping[str, object],
    target_tread_fraction: float,
    target_x_tolerance_m: float,
    target_z_tolerance_m: float,
    contact_on_threshold_n: float,
    minimum_upright_cosine: float,
) -> bool:
    """Require force-backed top contact inside the reviewed tread window."""

    validate_staircase_config(staircase)
    tip = _finite_vector(swing_tip_position_m, 3, "swing_tip_position_m")
    support_loads = _finite_vector(
        support_ground_normal_loads_n,
        len(STAIR_FOOT_NAMES) - 1,
        "support_ground_normal_loads_n",
    )
    gravity = _finite_vector(projected_gravity_xyz, 3, "projected_gravity_xyz")
    threshold = float(contact_on_threshold_n)
    if threshold <= 0.0:
        raise ValueError("contact_on_threshold_n must be positive")
    target_x = float(staircase["start_x_m"]) + float(
        target_tread_fraction
    ) * float(staircase["tread_depth_m"])
    target_z = float(staircase["rise_m"])
    return bool(
        float(swing_tread_normal_load_n) >= threshold
        and np.all(support_loads >= threshold)
        and abs(float(tip[0]) - target_x) <= float(target_x_tolerance_m)
        and abs(float(tip[2]) - target_z) <= float(target_z_tolerance_m)
        and float(-gravity[2]) >= float(minimum_upright_cosine)
    )


def placement_lift_hold_reached(
    *,
    swing_tip_height_m: float,
    initial_swing_tip_height_m: float,
    support_normal_loads_n: Sequence[float],
    support_margin_m: float,
    projected_gravity_xyz: Sequence[float],
    minimum_lift_m: float,
    contact_on_threshold_n: float,
    minimum_support_margin_m: float,
    minimum_upright_cosine: float,
) -> bool:
    """Require a force-supported upright hold above the requested foot lift."""

    support_loads = _finite_vector(
        support_normal_loads_n,
        len(STAIR_FOOT_NAMES) - 1,
        "support_normal_loads_n",
    )
    gravity = _finite_vector(projected_gravity_xyz, 3, "projected_gravity_xyz")
    lift = float(swing_tip_height_m) - float(initial_swing_tip_height_m)
    threshold = float(contact_on_threshold_n)
    minimum_lift = float(minimum_lift_m)
    if threshold <= 0.0:
        raise ValueError("contact_on_threshold_n must be positive")
    if minimum_lift <= 0.0:
        raise ValueError("minimum_lift_m must be positive")
    return bool(
        lift >= minimum_lift
        and np.all(support_loads >= threshold)
        and float(support_margin_m) >= float(minimum_support_margin_m)
        and float(-gravity[2]) >= float(minimum_upright_cosine)
    )


def pack_placement_reference_observation(
    *,
    stair_observation: Sequence[float],
    phase_one_hot: Sequence[float],
    desired_swing_height_m: float,
    measured_swing_height_m: float,
    swing_x_error_m: float,
    swing_z_error_m: float,
    tread_normal_load_n: float,
    support_contact_fraction: float,
    support_margin_m: float,
    maximum_support_slip_m: float,
    staircase: Mapping[str, object],
    contact_load_normalization_n: float,
) -> np.ndarray:
    """Append phase, target error, force, support, and slip state."""

    base = np.asarray(stair_observation, dtype=np.float32).reshape(-1)
    if base.size == 0 or not np.all(np.isfinite(base)):
        raise ValueError("stair_observation must contain finite values")
    phases = _finite_vector(phase_one_hot, len(PLACEMENT_PHASES), "phase_one_hot")
    if not np.isclose(float(np.sum(phases)), 1.0, atol=1e-6):
        raise ValueError("phase_one_hot must select exactly one phase")
    rise = float(staircase["rise_m"])
    tread = float(staircase["tread_depth_m"])
    load_scale = float(contact_load_normalization_n)
    if load_scale <= 0.0:
        raise ValueError("contact_load_normalization_n must be positive")
    extras = np.asarray(
        [
            *phases,
            float(desired_swing_height_m) / rise,
            float(measured_swing_height_m) / rise,
            float(swing_x_error_m) / tread,
            float(swing_z_error_m) / rise,
            float(tread_normal_load_n) / load_scale,
            float(support_contact_fraction),
            float(support_margin_m) / 0.10,
            float(maximum_support_slip_m) / 0.05,
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(extras)):
        raise ValueError("placement observation inputs must be finite")
    return np.clip(
        np.concatenate((base, extras)),
        -POLICY_OBSERVATION_CLIP,
        POLICY_OBSERVATION_CLIP,
    ).astype(np.float32)


def pack_support_regulation_observation(
    *,
    stair_observation: Sequence[float],
    total_foot_normal_loads_n: Sequence[float],
    com_target_error_xy_m: Sequence[float],
    requested_pd_effort_nm: Sequence[float],
    effort_cap_nm: float,
    contact_load_normalization_n: float,
    com_error_normalization_m: float = 0.10,
) -> np.ndarray:
    """Append load distribution, COM error, and per-leg drive saturation."""

    base = np.asarray(stair_observation, dtype=np.float32).reshape(-1)
    if base.size == 0 or not np.all(np.isfinite(base)):
        raise ValueError("stair_observation must contain finite values")
    loads = _finite_vector(
        total_foot_normal_loads_n,
        len(STAIR_FOOT_NAMES),
        "total_foot_normal_loads_n",
    )
    com_error = _finite_vector(
        com_target_error_xy_m,
        2,
        "com_target_error_xy_m",
    )
    requested_effort = _finite_vector(
        requested_pd_effort_nm,
        JOINT_COUNT,
        "requested_pd_effort_nm",
    )
    effort_cap = float(effort_cap_nm)
    load_scale = float(contact_load_normalization_n)
    com_scale = float(com_error_normalization_m)
    if effort_cap <= 0.0 or not np.isfinite(effort_cap):
        raise ValueError("effort_cap_nm must be finite and positive")
    if load_scale <= 0.0 or not np.isfinite(load_scale):
        raise ValueError(
            "contact_load_normalization_n must be finite and positive"
        )
    if com_scale <= 0.0 or not np.isfinite(com_scale):
        raise ValueError("com_error_normalization_m must be finite and positive")

    effort_cap_ratios: list[float] = []
    saturated_joint_fractions: list[float] = []
    for indices in STAIR_LEG_DOF_INDICES:
        leg_effort = np.abs(requested_effort[list(indices)])
        effort_cap_ratios.append(float(np.max(leg_effort) / effort_cap))
        saturated_joint_fractions.append(
            float(np.mean(leg_effort >= 0.95 * effort_cap - 1e-6))
        )
    extras = np.asarray(
        [
            *(loads / load_scale),
            *(com_error / com_scale),
            *effort_cap_ratios,
            *saturated_joint_fractions,
        ],
        dtype=np.float32,
    )
    return np.clip(
        np.concatenate((base, extras)),
        -POLICY_OBSERVATION_CLIP,
        POLICY_OBSERVATION_CLIP,
    ).astype(np.float32)


def progress_gate_failures(
    *,
    completed_episodes: int,
    first_step_climb_episodes: int,
    minimum_completed_episodes: int,
    minimum_first_step_climb_episodes: int,
    minimum_first_step_climb_rate: float,
) -> tuple[str, ...]:
    """Return explicit reasons that an early stair-progress gate failed."""

    completed = int(completed_episodes)
    climbs = int(first_step_climb_episodes)
    minimum_completed = int(minimum_completed_episodes)
    minimum_climbs = int(minimum_first_step_climb_episodes)
    minimum_rate = float(minimum_first_step_climb_rate)
    if min(completed, climbs, minimum_completed, minimum_climbs) < 0:
        raise ValueError("Progress-gate episode counts cannot be negative")
    if climbs > completed:
        raise ValueError("First-step climbs cannot exceed completed episodes")
    if minimum_rate < 0.0 or minimum_rate > 1.0:
        raise ValueError("minimum_first_step_climb_rate must be within [0, 1]")
    climb_rate = climbs / completed if completed else 0.0
    failures: list[str] = []
    if completed < minimum_completed:
        failures.append(f"completed_episodes={completed}<{minimum_completed}")
    if climbs < minimum_climbs:
        failures.append(
            f"first_step_climb_episodes={climbs}<{minimum_climbs}"
        )
    if climb_rate < minimum_rate:
        failures.append(
            f"first_step_climb_rate={climb_rate:.4f}<{minimum_rate:.4f}"
        )
    return tuple(failures)


def goal_x_for_active_steps(
    staircase: Mapping[str, object],
    active_steps: int,
) -> float:
    """Place each curriculum goal near the far side of its highest tread."""

    step_count = int(staircase["step_count"])
    if active_steps < 1 or active_steps > step_count:
        raise ValueError("active_steps is outside the staircase")
    tread_end = (
        float(staircase["start_x_m"])
        + active_steps * float(staircase["tread_depth_m"])
    )
    margin = float(staircase["goal_margin_m"])
    if active_steps == step_count:
        return tread_end + margin
    return tread_end - margin


def stair_goal_reached(
    *,
    base_world_x_m: float,
    base_elevation_gain_m: float,
    goal_world_x_m: float,
    minimum_base_elevation_gain_m: float,
    current_foot_steps: Sequence[int],
    active_steps: int,
    required_feet_on_goal_tread: int = 0,
) -> bool:
    """Return whether the body and required feet simultaneously reached the goal."""

    foot_steps = np.asarray(current_foot_steps, dtype=np.int32).reshape(-1)
    required_feet = int(required_feet_on_goal_tread)
    if int(active_steps) < 1:
        raise ValueError("active_steps must be positive")
    if required_feet < 0 or required_feet > foot_steps.size:
        raise ValueError(
            "required_feet_on_goal_tread must be within the foot-step vector"
        )
    feet_on_goal = int(np.count_nonzero(foot_steps >= int(active_steps)))
    return bool(
        float(base_world_x_m) >= float(goal_world_x_m)
        and float(base_elevation_gain_m)
        >= float(minimum_base_elevation_gain_m)
        and feet_on_goal >= required_feet
    )


def foot_tread_progress(
    *,
    foot_tip_positions_m,
    highest_foot_steps: Sequence[int],
    staircase: Mapping[str, object],
    active_steps: int,
    approach_distance_m: float,
    landing_fraction: float = 0.35,
) -> np.ndarray:
    """Return continuous per-foot progress toward each next stair tread."""

    validate_staircase_config(staircase)
    tips = np.asarray(foot_tip_positions_m, dtype=np.float32)
    if tips.ndim != 2 or tips.shape[1] != 3 or not np.all(np.isfinite(tips)):
        raise ValueError("foot_tip_positions_m must have finite shape (N, 3)")
    completed = np.asarray(highest_foot_steps, dtype=np.int32).reshape(-1)
    if completed.shape != (tips.shape[0],):
        raise ValueError("highest_foot_steps must match the foot-tip count")
    maximum_steps = int(staircase["step_count"])
    active = int(active_steps)
    if active < 1 or active > maximum_steps:
        raise ValueError("active_steps is outside the staircase")
    approach = float(approach_distance_m)
    landing = float(landing_fraction)
    if approach <= 0.0:
        raise ValueError("approach_distance_m must be positive")
    if not 0.0 < landing < 1.0:
        raise ValueError("landing_fraction must be within (0, 1)")

    start = float(staircase["start_x_m"])
    tread = float(staircase["tread_depth_m"])
    rise = float(staircase["rise_m"])
    progress = np.zeros(tips.shape[0], dtype=np.float32)
    for index, tip in enumerate(tips):
        completed_steps = int(np.clip(completed[index], 0, active))
        if completed_steps >= active:
            progress[index] = float(active)
            continue
        next_step = completed_steps + 1
        riser_x = start + (next_step - 1) * tread
        target_x = riser_x + landing * tread
        lower_surface_z = completed_steps * rise
        x_fraction = np.clip(
            (float(tip[0]) - (riser_x - approach))
            / (target_x - (riser_x - approach)),
            0.0,
            1.0,
        )
        z_fraction = np.clip(
            (float(tip[2]) - lower_surface_z) / rise,
            0.0,
            1.0,
        )
        progress[index] = completed_steps + min(x_fraction, z_fraction)
    return progress


def next_foot_target_index(
    highest_foot_steps: Sequence[int],
    *,
    active_steps: int,
    sequence_indices: Sequence[int],
) -> int | None:
    """Choose the next foot in a repeatable one-tread-at-a-time sequence."""

    completed = np.asarray(highest_foot_steps, dtype=np.int32).reshape(-1)
    if completed.size == 0:
        raise ValueError("highest_foot_steps cannot be empty")
    active = int(active_steps)
    if active < 1:
        raise ValueError("active_steps must be positive")
    sequence = tuple(int(value) for value in sequence_indices)
    if sorted(sequence) != list(range(completed.size)):
        raise ValueError("sequence_indices must be a permutation of the feet")
    target_step = min(int(np.min(completed)) + 1, active)
    for index in sequence:
        if int(completed[index]) < target_step:
            return index
    return None


def stair_observation_fields(
    terrain_sample_offsets_m: Sequence[float],
    *,
    include_navigation_observation: bool = False,
    include_foot_progress_observation: bool = False,
    include_placement_reference_observation: bool = False,
    include_support_regulation_observation: bool = False,
    terrain_observation_fields: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if terrain_observation_fields is None:
        terrain_fields = tuple(
            f"terrain_height_delta_at_{float(offset):+.3f}_m"
            for offset in terrain_sample_offsets_m
        )
    else:
        terrain_fields = tuple(str(field) for field in terrain_observation_fields)
        if not terrain_fields or any(not field for field in terrain_fields):
            raise ValueError("terrain_observation_fields cannot be empty")
        if len(set(terrain_fields)) != len(terrain_fields):
            raise ValueError("terrain_observation_fields must be unique")
    fields = POLICY_OBSERVATION_FIELDS + terrain_fields + (
        "goal_distance_normalized",
    )
    if include_navigation_observation:
        fields += (
            "lateral_offset_normalized",
            "heading_error_sin",
            "heading_error_cos",
        )
    if include_foot_progress_observation:
        fields += tuple(
            f"foot_tread_progress_{name}" for name in STAIR_FOOT_NAMES
        )
        fields += tuple(
            f"next_foot_target_{name}" for name in STAIR_FOOT_NAMES
        )
    if include_placement_reference_observation:
        fields += PLACEMENT_REFERENCE_OBSERVATION_FIELDS
    if include_support_regulation_observation:
        if not include_placement_reference_observation:
            raise ValueError(
                "support regulation observation requires placement reference"
            )
        fields += SUPPORT_REGULATION_OBSERVATION_FIELDS
    return fields


def pack_stair_policy_observation(
    *,
    walking_observation,
    base_world_x_m: float,
    base_world_y_m: float = 0.0,
    heading_error_rad: float = 0.0,
    goal_world_x_m: float,
    staircase: Mapping[str, object],
    include_navigation_observation: bool = False,
    include_foot_progress_observation: bool = False,
    foot_progress_normalized=None,
    next_foot_target_one_hot=None,
    terrain_observation_values=None,
) -> np.ndarray:
    """Append a terrain observation and curriculum goal distance."""

    validate_staircase_config(staircase)
    base = _finite_vector(
        walking_observation,
        POLICY_OBSERVATION_SIZE,
        "walking_observation",
    )
    if terrain_observation_values is None:
        offsets = np.asarray(
            staircase["terrain_sample_offsets_m"],
            dtype=np.float32,
        ).reshape(-1)
        local_height = stair_height_at_x(base_world_x_m, staircase)
        heights = np.asarray(
            [
                stair_height_at_x(base_world_x_m + float(offset), staircase)
                for offset in offsets
            ],
            dtype=np.float32,
        )
        height_scale = float(staircase["terrain_height_normalization_m"])
        terrain_profile = np.clip(
            (heights - local_height) / height_scale,
            -2.0,
            2.0,
        )
    else:
        terrain_profile = np.asarray(
            terrain_observation_values,
            dtype=np.float32,
        ).reshape(-1)
        if terrain_profile.size == 0 or not np.all(np.isfinite(terrain_profile)):
            raise ValueError(
                "terrain_observation_values must contain finite values"
            )
    goal_distance = np.clip(
        (float(goal_world_x_m) - float(base_world_x_m))
        / float(staircase["goal_distance_normalization_m"]),
        -2.0,
        2.0,
    )
    appended = [*terrain_profile, goal_distance]
    if include_navigation_observation:
        half_width = float(staircase["width_m"]) / 2.0
        lateral_offset = np.clip(float(base_world_y_m) / half_width, -2.0, 2.0)
        heading = float(heading_error_rad)
        appended.extend((lateral_offset, np.sin(heading), np.cos(heading)))
    if include_foot_progress_observation:
        progress = _finite_vector(
            foot_progress_normalized,
            len(STAIR_FOOT_NAMES),
            "foot_progress_normalized",
        )
        target = _finite_vector(
            next_foot_target_one_hot,
            len(STAIR_FOOT_NAMES),
            "next_foot_target_one_hot",
        )
        if np.any(progress < 0.0) or np.any(progress > 1.0):
            raise ValueError("foot_progress_normalized must be within [0, 1]")
        if np.any(target < 0.0) or np.any(target > 1.0):
            raise ValueError("next_foot_target_one_hot must be within [0, 1]")
        if not np.isclose(float(np.sum(target)), 1.0, atol=1e-6):
            raise ValueError("next_foot_target_one_hot must select exactly one foot")
        appended.extend(progress.tolist())
        appended.extend(target.tolist())
    observation = np.concatenate(
        (base, np.asarray(appended, dtype=np.float32))
    ).astype(np.float32)
    return np.clip(
        observation,
        -POLICY_OBSERVATION_CLIP,
        POLICY_OBSERVATION_CLIP,
    )


def stair_reward_terms(
    *,
    command_velocity_xyz,
    body_linear_velocity_xyz,
    body_angular_velocity_xyz,
    projected_gravity_xyz,
    base_clearance_m: float,
    lateral_position_m: float,
    forward_progress_m: float,
    base_height_gain_m: float,
    terrain_height_gain_m: float,
    heading_error_rad: float,
    joint_velocities_normalized,
    action,
    previous_action,
    failed: bool,
    succeeded: bool,
    reward_config: Mapping[str, float],
    foot_lift_progress_m: float = 0.0,
    foot_step_placement_progress: int = 0,
    foot_tread_progress: float = 0.0,
    foot_tread_support_count: int = 0,
    swing_target_distance_m: float = 0.0,
    tread_contact_reached: bool = False,
    support_contact_fraction: float = 0.0,
    support_slip_m: float = 0.0,
    support_margin_m: float = 0.0,
    balance_target_error_xy_m: Sequence[float] | None = None,
    support_normal_loads_n: Sequence[float] | None = None,
    requested_pd_effort_nm: Sequence[float] | None = None,
    effort_cap_nm: float = 1.0,
    contact_load_normalization_n: float = 50.0,
    swing_height_error_m: float | None = None,
    clearance_gate_deficit_m: float = 0.0,
) -> dict[str, float]:
    """Return individually reviewable stair-climbing reward terms."""

    command = _finite_vector(command_velocity_xyz, 3, "command_velocity_xyz")
    linear = _finite_vector(
        body_linear_velocity_xyz,
        3,
        "body_linear_velocity_xyz",
    )
    angular = _finite_vector(
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
    placement_sigma = float(
        reward_config.get("swing_target_tracking_sigma_m", 0.05)
    )
    if placement_sigma <= 0.0:
        raise ValueError("swing_target_tracking_sigma_m must be positive")
    height_sigma = float(
        reward_config.get("swing_height_tracking_sigma_m", 0.02)
    )
    if height_sigma <= 0.0:
        raise ValueError("swing_height_tracking_sigma_m must be positive")
    height_error = (
        None
        if swing_height_error_m is None
        else float(swing_height_error_m)
    )
    gate_deficit = float(clearance_gate_deficit_m)
    if height_error is not None and not np.isfinite(height_error):
        raise ValueError("swing_height_error_m must be finite")
    if not np.isfinite(gate_deficit) or gate_deficit < 0.0:
        raise ValueError(
            "clearance_gate_deficit_m must be finite and nonnegative"
        )
    balance_error = (
        np.zeros(2, dtype=np.float32)
        if balance_target_error_xy_m is None
        else _finite_vector(
            balance_target_error_xy_m,
            2,
            "balance_target_error_xy_m",
        )
    )
    support_loads = np.asarray(
        () if support_normal_loads_n is None else support_normal_loads_n,
        dtype=np.float32,
    ).reshape(-1)
    if not np.all(np.isfinite(support_loads)) or np.any(support_loads < 0.0):
        raise ValueError("support_normal_loads_n must be finite and nonnegative")
    effort_cap = float(effort_cap_nm)
    load_scale = float(contact_load_normalization_n)
    if effort_cap <= 0.0 or not np.isfinite(effort_cap):
        raise ValueError("effort_cap_nm must be finite and positive")
    if load_scale <= 0.0 or not np.isfinite(load_scale):
        raise ValueError(
            "contact_load_normalization_n must be finite and positive"
        )
    requested_effort = np.asarray(
        () if requested_pd_effort_nm is None else requested_pd_effort_nm,
        dtype=np.float32,
    ).reshape(-1)
    if requested_effort.size not in (0, JOINT_COUNT) or not np.all(
        np.isfinite(requested_effort)
    ):
        raise ValueError(
            f"requested_pd_effort_nm must contain {JOINT_COUNT} finite values"
        )

    velocity_error = float(linear[0] - command[0])
    velocity_tracking = float(np.exp(-((velocity_error / sigma) ** 2)))
    if bool(reward_config.get("subtract_zero_velocity_tracking", False)):
        velocity_tracking -= float(np.exp(-((command[0] / sigma) ** 2)))
    upright_cosine = float(np.clip(-gravity[2], 0.0, 1.0))
    clearance_error = (
        float(base_clearance_m)
        - float(reward_config["target_body_clearance_m"])
    )
    terms = {
        "forward_velocity_tracking": (
            float(reward_config["forward_velocity_tracking"]) * velocity_tracking
        ),
        "forward_progress": (
            float(reward_config["forward_progress"]) * float(forward_progress_m)
        ),
        "base_height_gain": (
            float(reward_config.get("base_height_gain", 0.0))
            * float(base_height_gain_m)
        ),
        "terrain_height_gain": (
            float(reward_config["terrain_height_gain"])
            * float(terrain_height_gain_m)
        ),
        "upright": float(reward_config["upright"]) * upright_cosine,
        "upright_deviation": (
            float(reward_config.get("upright_deviation", 0.0))
            * (1.0 - upright_cosine)
        ),
        "alive": float(reward_config["alive"]),
        "centerline": (
            float(reward_config["centerline"]) * float(lateral_position_m**2)
        ),
        "lateral_velocity": (
            float(reward_config.get("lateral_velocity", 0.0))
            * float(linear[1] ** 2)
        ),
        "heading_error": (
            float(reward_config.get("heading_error", 0.0))
            * float(1.0 - np.cos(float(heading_error_rad)))
        ),
        "vertical_velocity": (
            float(reward_config["vertical_velocity"]) * float(linear[2] ** 2)
        ),
        "roll_pitch_rate": (
            float(reward_config["roll_pitch_rate"])
            * float(np.sum(np.square(angular[:2])))
        ),
        "yaw_rate": (
            float(reward_config["yaw_rate"])
            * float((angular[2] - command[2]) ** 2)
        ),
        "body_clearance": (
            float(reward_config["body_clearance"]) * clearance_error**2
        ),
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
        "foot_lift_progress": (
            float(reward_config.get("foot_lift_progress", 0.0))
            * max(0.0, float(foot_lift_progress_m))
        ),
        "foot_step_placement": (
            float(reward_config.get("foot_step_placement", 0.0))
            * max(0, int(foot_step_placement_progress))
        ),
        "foot_tread_progress": (
            float(reward_config.get("foot_tread_progress", 0.0))
            * max(0.0, float(foot_tread_progress))
        ),
        "foot_tread_support": (
            float(reward_config.get("foot_tread_support", 0.0))
            * max(0, int(foot_tread_support_count))
        ),
        "swing_target_tracking": (
            float(reward_config.get("swing_target_tracking", 0.0))
            * float(
                np.exp(
                    -0.5
                    * (float(swing_target_distance_m) / placement_sigma) ** 2
                )
            )
        ),
        "swing_height_tracking": (
            float(reward_config.get("swing_height_tracking", 0.0))
            * (
                float(np.exp(-0.5 * (height_error / height_sigma) ** 2))
                if height_error is not None
                else 0.0
            )
        ),
        "clearance_gate_deficit": (
            float(reward_config.get("clearance_gate_deficit", 0.0))
            * gate_deficit
        ),
        "tread_contact": (
            float(reward_config.get("tread_contact", 0.0))
            if tread_contact_reached
            else 0.0
        ),
        "support_contact": (
            float(reward_config.get("support_contact", 0.0))
            * float(np.clip(support_contact_fraction, 0.0, 1.0))
        ),
        "support_slip": (
            float(reward_config.get("support_slip", 0.0))
            * max(0.0, float(support_slip_m))
        ),
        "support_margin": (
            float(reward_config.get("support_margin", 0.0))
            * float(np.clip(support_margin_m, -0.10, 0.10))
        ),
        "balance_target_error": (
            float(reward_config.get("balance_target_error", 0.0))
            * float(np.sum(np.square(balance_error / 0.10)))
        ),
        "minimum_support_load": (
            float(reward_config.get("minimum_support_load", 0.0))
            * (
                float(np.clip(np.min(support_loads) / load_scale, 0.0, 1.0))
                if support_loads.size
                else 0.0
            )
        ),
        "pd_effort_saturation": (
            float(reward_config.get("pd_effort_saturation", 0.0))
            * (
                float(
                    np.mean(
                        np.clip(
                            np.abs(requested_effort) / effort_cap - 0.95,
                            0.0,
                            2.0,
                        )
                    )
                )
                if requested_effort.size
                else 0.0
            )
        ),
        "failure": float(reward_config["failure"]) if failed else 0.0,
        "success": float(reward_config["success"]) if succeeded else 0.0,
    }
    terms["total"] = float(sum(terms.values()))
    return terms


def stair_failure_reasons(
    *,
    base_clearance_m: float,
    lateral_position_m: float,
    world_x_m: float,
    projected_gravity_xyz: Sequence[float],
    minimum_base_clearance_m: float,
    minimum_upright_cosine: float,
    maximum_lateral_deviation_m: float,
    minimum_world_x_m: float,
    support_slip_m: float = 0.0,
    maximum_support_slip_m: float | None = None,
    finite_state: bool = True,
) -> tuple[str, ...]:
    """Return deterministic fall, corridor, and non-finite failure reasons."""

    gravity = _finite_vector(
        projected_gravity_xyz,
        3,
        "projected_gravity_xyz",
    )
    reasons: list[str] = []
    if not finite_state:
        reasons.append("non_finite_state")
    if float(base_clearance_m) < float(minimum_base_clearance_m):
        reasons.append("base_clearance_too_low")
    if float(-gravity[2]) < float(minimum_upright_cosine):
        reasons.append("body_tipped")
    if abs(float(lateral_position_m)) > float(maximum_lateral_deviation_m):
        reasons.append("left_stair_corridor")
    if (
        maximum_support_slip_m is not None
        and float(support_slip_m) > float(maximum_support_slip_m)
    ):
        reasons.append("support_slip_exceeded")
    if float(world_x_m) < float(minimum_world_x_m):
        reasons.append("moved_too_far_backward")
    return tuple(reasons)
