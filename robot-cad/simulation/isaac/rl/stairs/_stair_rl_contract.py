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
        tread_fraction = float(level["target_tread_fraction"])
        if apex <= 0.0 or landing <= 0.0 or landing > apex:
            raise ValueError("placement lift heights must satisfy 0 < landing <= apex")
        if forward <= 0.0:
            raise ValueError("placement swing_forward_offset_m must be positive")
        if lift_forward <= 0.0 or lift_forward > forward:
            raise ValueError(
                "placement lift_forward_offset_m must be within (0, swing forward]"
            )
        if landing_forward <= 0.0 or landing_forward > forward:
            raise ValueError(
                "placement landing_forward_offset_m must be within (0, swing forward]"
            )
        if tread_fraction <= 0.0 or tread_fraction >= 1.0:
            raise ValueError("placement target_tread_fraction must be within (0, 1)")
        if progress >= start:
            selected = level
        previous_start = start
    if selected is None:
        raise ValueError("the first placement curriculum level must start at zero")
    return selected


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
    forward_fraction = desired_forward / final_forward
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


def inter_leg_transfer_state(
    elapsed_seconds: float,
    *,
    duration_seconds: float,
    unload_duration_seconds: float = 0.0,
) -> dict[str, object]:
    """Return a smooth all-feet-loaded transfer encoded as weight shift."""

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
    unload_linear_fraction = (
        float(np.clip((elapsed - duration) / unload_duration, 0.0, 1.0))
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
    }


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
