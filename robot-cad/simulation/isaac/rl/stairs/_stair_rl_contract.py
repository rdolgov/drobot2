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
    if float(world_x_m) < float(minimum_world_x_m):
        reasons.append("moved_too_far_backward")
    return tuple(reasons)
