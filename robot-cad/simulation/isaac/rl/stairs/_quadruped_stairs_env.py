"""Gymnasium environment for the separate Drobot stair-climbing policy.

Import this module only after constructing ``isaacsim.SimulationApp``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from _quadruped_rl_env import QuadrupedWalkEnv
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
from pxr import UsdPhysics

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


def _yaw_from_wxyz(orientation_wxyz) -> float:
    w, x, y, z = (
        float(value)
        for value in np.asarray(orientation_wxyz, dtype=np.float64).reshape(4)
    )
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
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

    def _validate_stair_prims(self) -> None:
        expected = int(self.staircase_config["step_count"])
        for index in range(expected):
            prim_path = f"/World/Stairs/StepLayer_{index + 1:02d}"
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(
                    f"Stair collision layer is missing or invalid: {prim_path}"
                )

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
        prior_action = self.previous_action.copy()
        desired_target = np.clip(
            self.nominal_positions + self.action_scale * clipped_action,
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
        self.previous_action = clipped_action.copy()
        state = self._read_state()
        base_position = np.asarray(state["base_position"])
        base_x = float(base_position[0])
        base_y = float(base_position[1])
        terrain_height = stair_height_at_x(base_x, self.staircase_config)
        base_clearance = float(base_position[2] - terrain_height)
        imu_observation = np.asarray(state["imu_observation"])
        projected_gravity = imu_observation[3:6]
        failure_reasons = stair_failure_reasons(
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
        failed = bool(failure_reasons)
        if base_x >= self.current_goal_x_m and not failed:
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
            "target_joint_positions_rad": target.copy(),
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
                "staircase": self.staircase_config,
                "curriculum_levels": list(self.curriculum_levels),
            }
        )
        return contract
