"""Gymnasium environment for the separate Drobot stair-climbing policy.

Import this module only after constructing ``isaacsim.SimulationApp``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from _quadruped_rl_env import QuadrupedWalkEnv
from _quadruped_runtime import LEGS, LINK_LENGTH_M
from _rl_contract import POLICY_OBSERVATION_CLIP
from _stair_rl_contract import (
    curriculum_active_steps,
    goal_x_for_active_steps,
    pack_stair_policy_observation,
    stair_failure_reasons,
    stair_height_at_x,
    stair_index_at_x,
    stair_observation_fields,
    stair_reward_terms,
    validate_staircase_config,
)
from gymnasium import spaces
from isaacsim.core.experimental.prims import RigidPrim
from pxr import UsdPhysics
from stable_baselines3 import PPO

STAIRS_EXPECTED_DOF_ORDER = (
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
FOOT_CONTACT_RADIUS_M = 0.0125
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaw_from_wxyz(orientation_wxyz) -> float:
    w, x, y, z = (
        float(value)
        for value in np.asarray(orientation_wxyz, dtype=np.float64).reshape(4)
    )
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    quaternion_vector = np.asarray([x, y, z], dtype=np.float64)
    return (
        vector
        + 2.0 * w * np.cross(quaternion_vector, vector)
        + 2.0
        * np.cross(
            quaternion_vector,
            np.cross(quaternion_vector, vector),
        )
    )
class QuadrupedStairsEnv(QuadrupedWalkEnv):
    """One floating quadruped learning a curriculum over a fixed staircase."""

    def __init__(
        self,
        simulation_app,
        *,
        world_path: str,
        task_config: Mapping[str, object],
        render_mode: str | None = None,
    ) -> None:
        self.staircase_config = dict(task_config["staircase"])
        validate_staircase_config(self.staircase_config)
        self.curriculum_config = dict(task_config["curriculum"])
        self.curriculum_levels = tuple(self.curriculum_config["levels"])
        self.active_step_count = curriculum_active_steps(
            0.0,
            self.curriculum_levels,
            maximum_steps=int(self.staircase_config["step_count"]),
        )
        self.pending_active_step_count = self.active_step_count
        self.current_goal_x_m = goal_x_for_active_steps(
            self.staircase_config,
            self.active_step_count,
        )
        self.curriculum_progress = 0.0
        self.curriculum_transitions: list[dict[str, object]] = []
        super().__init__(
            simulation_app,
            world_path=world_path,
            task_config=task_config,
            render_mode=render_mode,
        )
        self.physics_steps_per_control = self.physics_hz // self.control_hz
        if self.physics_steps_per_control < 1:
            raise ValueError("control_hz cannot exceed physics_hz")
        if tuple(self.dof_names) != STAIRS_EXPECTED_DOF_ORDER:
            raise RuntimeError(
                "Stair policy requires the reviewed DOF order: "
                f"{self.dof_names} != {list(STAIRS_EXPECTED_DOF_ORDER)}"
            )
        self._validate_stair_prims()
        self.residual_policy_config = dict(
            self.config.get("residual_policy", {})
        )
        self.residual_policy_enabled = bool(
            self.residual_policy_config.get("enabled", False)
        )
        self.base_policy = None
        self.base_policy_path: Path | None = None
        self.base_policy_sha256: str | None = None
        self.base_action_scale = np.zeros(12, dtype=np.float32)
        self.latest_walking_observation = np.zeros(48, dtype=np.float32)
        self.previous_residual_action = np.zeros(12, dtype=np.float32)
        if self.residual_policy_enabled:
            configured_path = Path(
                str(self.residual_policy_config["base_model"])
            )
            self.base_policy_path = (
                configured_path.resolve()
                if configured_path.is_absolute()
                else (PROJECT_ROOT / configured_path).resolve()
            )
            if not self.base_policy_path.is_file():
                raise FileNotFoundError(self.base_policy_path)
            self.base_policy_sha256 = _sha256_file(self.base_policy_path)
            self.base_policy = PPO.load(
                str(self.base_policy_path),
                device="cpu",
            )
            if (
                tuple(self.base_policy.observation_space.shape) != (48,)
                or tuple(self.base_policy.action_space.shape) != (12,)
            ):
                raise RuntimeError(
                    "Residual base policy must have observation/action "
                    "shapes (48,)/(12,)"
                )
            base_scale_by_kind = dict(
                self.residual_policy_config["base_action_scale_rad"]
            )
            self.base_action_scale = np.asarray(
                [
                    float(
                        base_scale_by_kind[
                            next(
                                kind
                                for kind in base_scale_by_kind
                                if name.endswith(kind)
                            )
                        ]
                    )
                    for name in self.dof_names
                ],
                dtype=np.float32,
            )
        link_path_by_name = dict(
            zip(self.robot.link_names, self.robot.link_paths[0], strict=True)
        )
        self.foot_prim = RigidPrim(
            [link_path_by_name[f"{leg}_distal_link"] for leg in LEGS]
        )

        offsets = tuple(
            float(value)
            for value in self.staircase_config["terrain_sample_offsets_m"]
        )
        self.include_navigation_observation = bool(
            self.config.get("include_navigation_observation", False)
        )
        self.observation_fields = stair_observation_fields(
            offsets,
            include_navigation_observation=self.include_navigation_observation,
        )
        self.observation_size = len(self.observation_fields)
        self.observation_space = spaces.Box(
            low=-POLICY_OBSERVATION_CLIP,
            high=POLICY_OBSERVATION_CLIP,
            shape=(self.observation_size,),
            dtype=np.float32,
        )
        self.success_hold_steps = int(
            round(float(self.config["success_hold_seconds"]) * self.control_hz)
        )
        if self.success_hold_steps < 1:
            raise ValueError("success_hold_seconds is shorter than one control step")
        self.previous_base_x_m = 0.0
        self.previous_base_z_m = 0.0
        self.previous_terrain_height_m = 0.0
        self.maximum_base_elevation_gain_m = 0.0
        self.maximum_terrain_height_m = 0.0
        self.minimum_base_clearance_m = float("inf")
        self.highest_step_reached = 0
        self.goal_hold_step_count = 0
        self.initial_foot_bottom_z_m = np.zeros(len(LEGS), dtype=np.float32)
        self.maximum_foot_lift_m = np.zeros(len(LEGS), dtype=np.float32)
        self.highest_foot_step = np.zeros(len(LEGS), dtype=np.int32)

    def _validate_stair_prims(self) -> None:
        expected = int(self.staircase_config["step_count"])
        for index in range(expected):
            prim_path = f"/World/Stairs/StepLayer_{index + 1:02d}"
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(
                    f"Stair collision layer is missing or invalid: {prim_path}"
                )

    def _sample_foot_tips(self) -> np.ndarray:
        positions_raw, orientations_raw = self.foot_prim.get_world_poses()
        positions = np.asarray(
            positions_raw.numpy()
            if hasattr(positions_raw, "numpy")
            else positions_raw,
            dtype=np.float64,
        ).reshape(len(LEGS), 3)
        orientations = np.asarray(
            orientations_raw.numpy()
            if hasattr(orientations_raw, "numpy")
            else orientations_raw,
            dtype=np.float64,
        ).reshape(len(LEGS), 4)
        if not np.all(np.isfinite(positions)) or not np.all(
            np.isfinite(orientations)
        ):
            raise RuntimeError("Foot poses contain non-finite values")
        tip_positions = positions.copy()
        local_tip = np.asarray([LINK_LENGTH_M, 0.0, 0.0], dtype=np.float64)
        for index in range(len(LEGS)):
            tip_positions[index] += _rotate_wxyz(
                orientations[index],
                local_tip,
            )
        tip_positions[:, 2] -= FOOT_CONTACT_RADIUS_M
        return tip_positions.astype(np.float32)

    def _foot_progress(
        self,
        foot_tips: np.ndarray,
    ) -> tuple[float, int, np.ndarray, np.ndarray]:
        rise = float(self.staircase_config["rise_m"])
        lift_cap = 2.0 * rise
        foot_lifts = np.maximum(
            0.0,
            foot_tips[:, 2] - self.initial_foot_bottom_z_m,
        )
        prior_maximum = self.maximum_foot_lift_m.copy()
        stair_start = float(self.staircase_config["start_x_m"])
        active_end = (
            stair_start
            + self.active_step_count
            * float(self.staircase_config["tread_depth_m"])
        )
        lift_eligible = np.logical_and(
            foot_tips[:, 0] >= stair_start - 0.18,
            foot_tips[:, 0] <= active_end,
        )
        capped_lifts = np.where(
            lift_eligible,
            np.minimum(foot_lifts, lift_cap),
            prior_maximum,
        )
        new_maximum = np.maximum(prior_maximum, capped_lifts)
        lift_progress = float(np.sum(new_maximum - prior_maximum))

        placement_tolerance = min(
            float(self.config.get("foot_placement_tolerance_m", 0.0125)),
            0.25 * rise,
        )
        current_steps = np.zeros(len(LEGS), dtype=np.int32)
        for index, foot_tip in enumerate(foot_tips):
            step_index = min(
                self.active_step_count,
                stair_index_at_x(float(foot_tip[0]), self.staircase_config),
            )
            surface_height = stair_height_at_x(
                float(foot_tip[0]),
                self.staircase_config,
            )
            if (
                step_index > 0
                and float(foot_tip[2])
                >= surface_height - placement_tolerance
            ):
                current_steps[index] = step_index
        prior_steps = self.highest_foot_step.copy()
        new_steps = np.maximum(prior_steps, current_steps)
        placement_progress = int(np.sum(new_steps - prior_steps))
        return lift_progress, placement_progress, new_maximum, new_steps

    def set_training_progress(self, progress_fraction: float) -> None:
        """Schedule a curriculum level; it becomes active at the next reset."""

        progress = float(np.clip(progress_fraction, 0.0, 1.0))
        active = curriculum_active_steps(
            progress,
            self.curriculum_levels,
            maximum_steps=int(self.staircase_config["step_count"]),
        )
        if active != self.pending_active_step_count:
            self.curriculum_transitions.append(
                {
                    "progress_fraction": progress,
                    "active_steps": active,
                }
            )
        self.curriculum_progress = progress
        self.pending_active_step_count = active

    def set_evaluation_level(self, active_steps: int) -> None:
        """Pin evaluation to a requested number of stairs."""

        maximum = int(self.staircase_config["step_count"])
        if active_steps < 1 or active_steps > maximum:
            raise ValueError(f"active_steps must be within 1..{maximum}")
        self.pending_active_step_count = int(active_steps)
        self.active_step_count = int(active_steps)
        self.current_goal_x_m = goal_x_for_active_steps(
            self.staircase_config,
            self.active_step_count,
        )

    def set_training_level(
        self,
        active_steps: int,
        *,
        reason: str,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        """Schedule a mastery-selected level for the next episode reset."""

        maximum = int(self.staircase_config["step_count"])
        if active_steps < 1 or active_steps > maximum:
            raise ValueError(f"active_steps must be within 1..{maximum}")
        if int(active_steps) == self.pending_active_step_count:
            return
        transition: dict[str, object] = {
            "active_steps": int(active_steps),
            "reason": str(reason),
        }
        if evidence:
            transition["evidence"] = dict(evidence)
        self.curriculum_transitions.append(transition)
        self.pending_active_step_count = int(active_steps)

    def _read_state(self) -> dict[str, np.ndarray | float]:
        state = super()._read_state()
        self.latest_walking_observation = np.asarray(
            state["observation"],
            dtype=np.float32,
        ).copy()
        base_position = np.asarray(state["base_position"])
        heading_error = _yaw_from_wxyz(state["base_orientation"])
        state["observation"] = pack_stair_policy_observation(
            walking_observation=state["observation"],
            base_world_x_m=float(base_position[0]),
            base_world_y_m=float(base_position[1]),
            heading_error_rad=heading_error,
            goal_world_x_m=self.current_goal_x_m,
            staircase=self.staircase_config,
            include_navigation_observation=self.include_navigation_observation,
        )
        state["heading_error_rad"] = heading_error
        return state

    def _reset_robot(self) -> None:
        reset_noise = float(self.config["reset_joint_noise_rad"])
        joint_noise = self.np_random.uniform(
            -reset_noise,
            reset_noise,
            size=12,
        ).astype(np.float32)
        initial_positions = np.clip(
            self.nominal_positions + joint_noise,
            self.lower_limits + 1e-3,
            self.upper_limits - 1e-3,
        )
        x_range = tuple(float(value) for value in self.config["reset_start_x_range_m"])
        y_range = tuple(float(value) for value in self.config["reset_start_y_range_m"])
        yaw_range = tuple(
            math.radians(float(value))
            for value in self.config["reset_start_yaw_range_deg"]
        )
        if len(x_range) != 2 or len(y_range) != 2 or len(yaw_range) != 2:
            raise ValueError("reset position/yaw ranges need exactly two endpoints")
        x = float(self.np_random.uniform(*x_range))
        y = float(self.np_random.uniform(*y_range))
        yaw = float(self.np_random.uniform(*yaw_range))
        orientation_wxyz = [
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ]
        self.robot.set_world_poses(
            positions=[[x, y, float(self.config["reset_start_z_m"])]],
            orientations=[orientation_wxyz],
        )
        self.robot.set_velocities(
            linear_velocities=[[0.0, 0.0, 0.0]],
            angular_velocities=[[0.0, 0.0, 0.0]],
        )
        self.robot.set_dof_positions(initial_positions)
        self.robot.set_dof_velocities(np.zeros(12, dtype=np.float32))
        self.robot.set_dof_position_targets(self.nominal_positions)
        self.previous_target = self.nominal_positions.copy()
        self.previous_action.fill(0.0)
        self.previous_residual_action.fill(0.0)
        for _ in range(self.reset_settle_steps):
            self.robot.set_dof_position_targets(self.nominal_positions)
            self._update(self.physics_steps_per_control)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        self.active_step_count = self.pending_active_step_count
        self.current_goal_x_m = goal_x_for_active_steps(
            self.staircase_config,
            self.active_step_count,
        )
        observation, info = super().reset(seed=seed, options=options)
        self.previous_base_x_m = float(self.episode_origin[0])
        self.previous_base_z_m = float(self.episode_origin[2])
        self.previous_terrain_height_m = stair_height_at_x(
            self.previous_base_x_m,
            self.staircase_config,
        )
        initial_clearance = float(
            self.episode_origin[2] - self.previous_terrain_height_m
        )
        self.minimum_base_clearance_m = initial_clearance
        self.maximum_base_elevation_gain_m = 0.0
        self.maximum_terrain_height_m = self.previous_terrain_height_m
        self.highest_step_reached = stair_index_at_x(
            self.previous_base_x_m,
            self.staircase_config,
        )
        self.goal_hold_step_count = 0
        foot_tips = self._sample_foot_tips()
        self.initial_foot_bottom_z_m = foot_tips[:, 2].copy()
        self.maximum_foot_lift_m.fill(0.0)
        self.highest_foot_step.fill(0)
        info.update(
            {
                "task_id": self.config["id"],
                "active_step_count": self.active_step_count,
                "goal_world_x_m": self.current_goal_x_m,
                "observation_fields": self.observation_fields,
                "physics_steps_per_control": self.physics_steps_per_control,
            }
        )
        return observation, info

    def step(
        self,
        action,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        clipped_action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(12),
            -1.0,
            1.0,
        )
        base_action = np.zeros(12, dtype=np.float32)
        if self.residual_policy_enabled:
            if self.base_policy is None:
                raise RuntimeError("Residual base policy was not loaded")
            predicted, _ = self.base_policy.predict(
                self.latest_walking_observation,
                deterministic=True,
            )
            base_action = np.clip(
                np.asarray(predicted, dtype=np.float32).reshape(12),
                -1.0,
                1.0,
            )
            prior_action = self.previous_residual_action.copy()
            action_offset = (
                self.base_action_scale * base_action
                + self.action_scale * clipped_action
            )
        else:
            prior_action = self.previous_action.copy()
            action_offset = self.action_scale * clipped_action
        desired_target = np.clip(
            self.nominal_positions + action_offset,
            self.lower_limits + 1e-3,
            self.upper_limits - 1e-3,
        )
        maximum_delta = self.max_velocities * self.control_dt_s
        target = np.clip(
            desired_target,
            self.previous_target - maximum_delta,
            self.previous_target + maximum_delta,
        )
        self.robot.set_dof_position_targets(target.astype(np.float32))
        self._update(self.physics_steps_per_control)
        self.previous_action = (
            base_action.copy()
            if self.residual_policy_enabled
            else clipped_action.copy()
        )
        state = self._read_state()
        base_position = np.asarray(state["base_position"])
        base_x = float(base_position[0])
        base_y = float(base_position[1])
        terrain_height = stair_height_at_x(base_x, self.staircase_config)
        base_clearance = float(base_position[2] - terrain_height)
        foot_tips = self._sample_foot_tips()
        (
            foot_lift_progress,
            foot_placement_progress,
            next_maximum_foot_lift,
            next_highest_foot_step,
        ) = self._foot_progress(foot_tips)
        imu_observation = np.asarray(state["imu_observation"])
        projected_gravity = imu_observation[3:6]
        failure_reasons = list(
            stair_failure_reasons(
                base_clearance_m=base_clearance,
                lateral_position_m=base_y,
                world_x_m=base_x,
                projected_gravity_xyz=projected_gravity,
                minimum_base_clearance_m=float(
                    self.termination_config["minimum_base_clearance_m"]
                ),
                minimum_upright_cosine=float(
                    self.termination_config["minimum_upright_cosine"]
                ),
                maximum_lateral_deviation_m=float(
                    self.termination_config["maximum_lateral_deviation_m"]
                ),
                minimum_world_x_m=float(
                    self.termination_config["minimum_world_x_m"]
                ),
            )
        )
        stall_config = dict(self.config.get("stall_termination", {}))
        if bool(stall_config.get("enabled", False)):
            stall_step = int(
                round(
                    float(stall_config["after_seconds"])
                    * self.control_hz
                )
            )
            forward_displacement = float(base_x - self.episode_origin[0])
            if (
                self.episode_step + 1 >= stall_step
                and forward_displacement
                < float(stall_config["minimum_forward_progress_m"])
            ):
                failure_reasons.append("no_forward_progress")
        failure_reasons = tuple(failure_reasons)
        failed = bool(failure_reasons)
        minimum_success_elevation = (
            self.active_step_count
            * float(self.staircase_config["rise_m"])
            * float(
                self.config.get(
                    "success_minimum_base_elevation_fraction",
                    0.0,
                )
            )
        )
        current_base_elevation = float(
            base_position[2] - self.episode_origin[2]
        )
        if (
            base_x >= self.current_goal_x_m
            and current_base_elevation >= minimum_success_elevation
            and not failed
        ):
            self.goal_hold_step_count += 1
        else:
            self.goal_hold_step_count = 0
        succeeded = self.goal_hold_step_count >= self.success_hold_steps
        terminated = failed or succeeded

        self.episode_step += 1
        truncated = self.episode_step >= self.max_episode_steps
        forward_progress = base_x - self.previous_base_x_m
        base_height_gain = float(base_position[2] - self.previous_base_z_m)
        terrain_height_gain = terrain_height - self.previous_terrain_height_m
        normalized_joint_velocity = (
            np.asarray(state["joint_velocities"]) / self.max_velocities
        )
        reward_terms = stair_reward_terms(
            command_velocity_xyz=self.command_velocity,
            body_linear_velocity_xyz=state["body_linear_velocity"],
            body_angular_velocity_xyz=imu_observation[:3],
            projected_gravity_xyz=projected_gravity,
            base_clearance_m=base_clearance,
            lateral_position_m=base_y,
            forward_progress_m=forward_progress,
            base_height_gain_m=base_height_gain,
            terrain_height_gain_m=terrain_height_gain,
            heading_error_rad=float(state["heading_error_rad"]),
            joint_velocities_normalized=normalized_joint_velocity,
            action=clipped_action,
            previous_action=prior_action,
            failed=failed,
            succeeded=succeeded,
            reward_config=self.reward_config,
            foot_lift_progress_m=(
                0.0 if failed else foot_lift_progress
            ),
            foot_step_placement_progress=(
                0 if failed else foot_placement_progress
            ),
        )
        reward = float(reward_terms["total"])
        self.episode_return += reward
        self.minimum_base_clearance_m = min(
            self.minimum_base_clearance_m,
            base_clearance,
        )
        upright_cosine = float(np.clip(-projected_gravity[2], -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(upright_cosine)))
        self.maximum_tilt_deg = max(self.maximum_tilt_deg, tilt_deg)
        self.highest_step_reached = max(
            self.highest_step_reached,
            stair_index_at_x(base_x, self.staircase_config),
        )
        self.maximum_base_elevation_gain_m = max(
            self.maximum_base_elevation_gain_m,
            float(base_position[2] - self.episode_origin[2]),
        )
        self.maximum_terrain_height_m = max(
            self.maximum_terrain_height_m,
            terrain_height,
        )
        displacement = base_position - self.episode_origin
        info: dict[str, object] = {
            "reward_terms": reward_terms,
            "base_position_m": base_position.copy(),
            "body_linear_velocity_m_s": np.asarray(
                state["body_linear_velocity"]
            ).copy(),
            "failure_reasons": failure_reasons,
            "succeeded": succeeded,
            "active_step_count": self.active_step_count,
            "highest_step_reached": self.highest_step_reached,
            "terrain_height_m": terrain_height,
            "base_clearance_m": base_clearance,
            "goal_world_x_m": self.current_goal_x_m,
            "heading_error_rad": float(state["heading_error_rad"]),
            "maximum_base_elevation_gain_m": self.maximum_base_elevation_gain_m,
            "foot_tip_positions_m": foot_tips.copy(),
            "maximum_foot_lift_m_by_leg": dict(
                zip(LEGS, next_maximum_foot_lift.tolist(), strict=True)
            ),
            "highest_foot_step_by_leg": dict(
                zip(LEGS, next_highest_foot_step.tolist(), strict=True)
            ),
            "target_joint_positions_rad": target.copy(),
            "base_policy_action": base_action.copy(),
            "residual_policy_action": clipped_action.copy(),
        }
        if terminated or truncated:
            episode_metrics = {
                "return": self.episode_return,
                "length_steps": self.episode_step,
                "duration_s": self.episode_step / self.control_hz,
                "active_step_count": self.active_step_count,
                "highest_step_reached": self.highest_step_reached,
                "stairs_completed": succeeded,
                "forward_displacement_m": float(displacement[0]),
                "lateral_displacement_m": float(displacement[1]),
                "elevation_gain_m": float(displacement[2]),
                "maximum_base_elevation_gain_m": (
                    self.maximum_base_elevation_gain_m
                ),
                "maximum_foot_lift_m_by_leg": dict(
                    zip(LEGS, next_maximum_foot_lift.tolist(), strict=True)
                ),
                "highest_foot_step_by_leg": dict(
                    zip(LEGS, next_highest_foot_step.tolist(), strict=True)
                ),
                "final_terrain_height_m": terrain_height,
                "maximum_terrain_height_m": self.maximum_terrain_height_m,
                "minimum_base_clearance_m": self.minimum_base_clearance_m,
                "maximum_body_tilt_deg": self.maximum_tilt_deg,
                "goal_hold_duration_s": (
                    self.goal_hold_step_count / self.control_hz
                ),
                "terminated": terminated,
                "truncated": truncated,
                "failure_reasons": failure_reasons,
            }
            info["episode_metrics"] = episode_metrics
            self.completed_episode_metrics.append(episode_metrics)
            self.completed_episode_metrics = self.completed_episode_metrics[-20:]

        self.previous_target = target.copy()
        self.previous_base_x_m = base_x
        self.previous_base_z_m = float(base_position[2])
        self.previous_terrain_height_m = terrain_height
        self.previous_residual_action = clipped_action.copy()
        self.maximum_foot_lift_m = next_maximum_foot_lift
        self.highest_foot_step = next_highest_foot_step
        return (
            np.asarray(state["observation"]).copy(),
            reward,
            terminated,
            truncated,
            info,
        )

    @property
    def contract(self) -> dict[str, object]:
        contract = dict(super().contract)
        contract.update(
            {
                "task_id": self.config["id"],
                "dof_names": list(STAIRS_EXPECTED_DOF_ORDER),
                "observation_fields": list(self.observation_fields),
                "observation_size": self.observation_size,
                "walking_observation_size": 48,
                "include_navigation_observation": (
                    self.include_navigation_observation
                ),
                "terrain_input_note": (
                    "Analytic forward terrain profile; replace with a "
                    "camera/depth estimator before hardware deployment."
                ),
                "physics_steps_per_control": self.physics_steps_per_control,
                "control_action_mode": (
                    "residual_over_flat"
                    if self.residual_policy_enabled
                    else "direct"
                ),
                "residual_policy": (
                    {
                        **self.residual_policy_config,
                        "base_model_resolved": str(self.base_policy_path),
                        "base_model_sha256": self.base_policy_sha256,
                        "base_action_scale_by_dof_rad": (
                            self.base_action_scale.tolist()
                        ),
                    }
                    if self.residual_policy_enabled
                    else None
                ),
                "staircase": self.staircase_config,
                "curriculum_levels": list(self.curriculum_levels),
            }
        )
        return contract
