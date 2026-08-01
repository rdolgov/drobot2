"""Isaac Gymnasium environment for the 190 mm single-foot-lift skill."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from _foot_lift_contract import (
    FOOT_LIFT_OBSERVATION_FIELDS,
    FOOT_LIFT_OBSERVATION_SIZE,
    desired_foot_lift_m,
    foot_lift_failure_reasons,
    foot_lift_reward_terms,
    foot_lift_success_reached,
    pack_foot_lift_observation,
    smoothstep,
)
from _quadruped_rl_env import QuadrupedWalkEnv
from _quadruped_runtime import LEGS, LINK_LENGTH_M, pose_by_name, targets_for_order
from _rl_contract import POLICY_OBSERVATION_CLIP
from gymnasium import spaces
from isaacsim.core.experimental.prims import RigidPrim

FOOT_CONTACT_RADIUS_M = 0.0125
EXPECTED_DOF_ORDER = (
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


class QuadrupedFootLiftEnv(QuadrupedWalkEnv):
    """Track one vertical foot reference while PPO stabilizes the body."""

    def __init__(
        self,
        simulation_app,
        *,
        world_path: str,
        task_config: Mapping[str, object],
        render_mode: str | None = None,
    ) -> None:
        self.lift_config = dict(task_config["foot_lift"])
        self.success_config = dict(task_config["success"])
        self.base_support_config = dict(task_config["base_support"])
        self.base_support_mode = str(self.base_support_config["mode"])
        if self.base_support_mode not in {"none", "pose_hold"}:
            raise ValueError(f"Unsupported base support mode: {self.base_support_mode}")
        self.swing_leg = str(self.lift_config["swing_leg"])
        if self.swing_leg not in LEGS:
            raise ValueError(f"Unknown swing leg: {self.swing_leg}")
        self.target_lift_m = float(self.lift_config["target_lift_m"])
        self.reference_lift_m = float(self.lift_config.get("reference_lift_m", self.target_lift_m))
        if self.reference_lift_m < self.target_lift_m:
            raise ValueError("reference_lift_m cannot be below target_lift_m")
        self.ramp_start_seconds = float(self.lift_config["ramp_start_seconds"])
        self.ramp_duration_seconds = float(self.lift_config["ramp_duration_seconds"])
        self.target_forward_offset_m = float(self.lift_config["target_forward_offset_m"])
        self.weight_shift_config = dict(task_config["weight_shift"])
        super().__init__(
            simulation_app,
            world_path=world_path,
            task_config=task_config,
            render_mode=render_mode,
        )
        if tuple(self.dof_names) != EXPECTED_DOF_ORDER:
            raise RuntimeError(f"Foot-lift task requires the reviewed DOF order: {self.dof_names}")
        self.physics_steps_per_control = self.physics_hz // self.control_hz
        if self.physics_steps_per_control < 1:
            raise ValueError("control_hz cannot exceed physics_hz")
        link_path_by_name = dict(zip(self.robot.link_names, self.robot.link_paths[0], strict=True))
        self.foot_prim = RigidPrim([link_path_by_name[f"{leg}_distal_link"] for leg in LEGS])
        self.swing_leg_index = LEGS.index(self.swing_leg)
        self.support_leg_indices = tuple(
            index for index, leg in enumerate(LEGS) if leg != self.swing_leg
        )
        residual_scale = dict(self.config["residual_action_scale_rad"])
        for index, name in enumerate(self.dof_names):
            role = "swing" if name.startswith(f"{self.swing_leg}_") else "support"
            kind = next(candidate for candidate in residual_scale[role] if name.endswith(candidate))
            self.action_scale[index] = float(residual_scale[role][kind])
        self.observation_space = spaces.Box(
            low=-POLICY_OBSERVATION_CLIP,
            high=POLICY_OBSERVATION_CLIP,
            shape=(FOOT_LIFT_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.success_hold_steps = int(
            round(float(self.config["success_hold_seconds"]) * self.control_hz)
        )
        if self.success_hold_steps < 1:
            raise ValueError("success_hold_seconds is shorter than one control step")
        self.initial_foot_tip_z_m = np.zeros(len(LEGS), dtype=np.float32)
        self.episode_base_origin = np.zeros(3, dtype=np.float32)
        self.episode_base_orientation = np.asarray(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self.maximum_swing_foot_lift_m = 0.0
        self.maximum_support_foot_lift_m = 0.0
        self.current_swing_foot_lift_m = 0.0
        self.desired_swing_foot_lift_m = 0.0
        self.goal_hold_step_count = 0

    def _sample_foot_tips(self) -> np.ndarray:
        positions_raw, orientations_raw = self.foot_prim.get_world_poses()
        positions = np.asarray(
            positions_raw.numpy() if hasattr(positions_raw, "numpy") else positions_raw,
            dtype=np.float64,
        ).reshape(len(LEGS), 3)
        orientations = np.asarray(
            orientations_raw.numpy() if hasattr(orientations_raw, "numpy") else orientations_raw,
            dtype=np.float64,
        ).reshape(len(LEGS), 4)
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(orientations)):
            raise RuntimeError("Foot poses contain non-finite values")
        tips = positions.copy()
        local_tip = np.asarray([LINK_LENGTH_M, 0.0, 0.0], dtype=np.float64)
        for index in range(len(LEGS)):
            tips[index] += _rotate_wxyz(orientations[index], local_tip)
        tips[:, 2] -= FOOT_CONTACT_RADIUS_M
        return tips.astype(np.float32)

    def _reference_targets(
        self,
        *,
        elapsed_seconds: float,
        desired_lift_m: float,
    ) -> np.ndarray:
        stance = dict(self.config["nominal_stance"])
        nominal_down = float(stance["down_m"])
        fore_aft = float(stance["fore_aft_m"])
        nominal_abduction = np.deg2rad(float(stance["abduction_deg"]))
        shift_start = float(self.weight_shift_config["start_seconds"])
        shift_duration = float(self.weight_shift_config["duration_seconds"])
        shift_fraction = smoothstep((float(elapsed_seconds) - shift_start) / shift_duration)
        swing_front_sign = 1.0 if self.swing_leg.startswith("front_") else -1.0
        swing_side_sign = 1.0 if self.swing_leg.endswith("_left") else -1.0
        body_shift_forward_m = (
            -swing_front_sign * float(self.weight_shift_config["forward_m"]) * shift_fraction
        )
        body_shift_lateral_m = (
            -swing_side_sign * float(self.weight_shift_config["lateral_m"]) * shift_fraction
        )
        down_by_leg = {leg: nominal_down for leg in LEGS}
        down_by_leg[self.swing_leg] = nominal_down - float(desired_lift_m)
        forward_by_leg = {
            leg: (fore_aft if leg.startswith("front_") else -fore_aft) - body_shift_forward_m
            for leg in LEGS
        }
        forward_by_leg[self.swing_leg] += (
            self.target_forward_offset_m * float(desired_lift_m) / self.reference_lift_m
        )
        foot_delta_lateral_m = -body_shift_lateral_m
        abduction_by_leg_deg: dict[str, float] = {}
        for leg in LEGS:
            side_sign = 1.0 if leg.endswith("_left") else -1.0
            vertical = down_by_leg[leg] * np.cos(nominal_abduction)
            outward = down_by_leg[leg] * np.sin(nominal_abduction)
            shifted_outward = outward + side_sign * foot_delta_lateral_m
            down_by_leg[leg] = float(np.hypot(vertical, shifted_outward))
            abduction_by_leg_deg[leg] = float(np.degrees(np.arctan2(shifted_outward, vertical)))
        pose = pose_by_name(
            down_by_leg_m=down_by_leg,
            forward_by_leg_m=forward_by_leg,
            abduction_by_leg_deg=abduction_by_leg_deg,
        )
        targets = np.asarray(
            targets_for_order(self.dof_names, pose),
            dtype=np.float32,
        )
        return np.clip(
            targets,
            self.lower_limits + 1e-3,
            self.upper_limits - 1e-3,
        )

    def _skill_observation(
        self,
        state: Mapping[str, object],
        *,
        desired_lift_m: float,
        measured_lift_m: float,
        maximum_lift_m: float,
        base_height_error_m: float,
        base_displacement_xy_m: np.ndarray,
        support_foot_lift_m: float,
    ) -> np.ndarray:
        return pack_foot_lift_observation(
            walking_observation=state["observation"],
            target_lift_m=self.target_lift_m,
            desired_lift_m=desired_lift_m,
            measured_lift_m=measured_lift_m,
            maximum_lift_m=maximum_lift_m,
            base_height_error_m=base_height_error_m,
            base_displacement_xy_m=base_displacement_xy_m,
            maximum_support_foot_lift_m=support_foot_lift_m,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        _, info = super().reset(seed=seed, options=options)
        state = super()._read_state()
        foot_tips = self._sample_foot_tips()
        self.initial_foot_tip_z_m = foot_tips[:, 2].copy()
        self.episode_base_origin = np.asarray(state["base_position"]).copy()
        self.episode_base_orientation = np.asarray(state["base_orientation"]).copy()
        self.maximum_swing_foot_lift_m = 0.0
        self.maximum_support_foot_lift_m = 0.0
        self.current_swing_foot_lift_m = 0.0
        self.desired_swing_foot_lift_m = 0.0
        self.goal_hold_step_count = 0
        observation = self._skill_observation(
            state,
            desired_lift_m=0.0,
            measured_lift_m=0.0,
            maximum_lift_m=0.0,
            base_height_error_m=0.0,
            base_displacement_xy_m=np.zeros(2, dtype=np.float32),
            support_foot_lift_m=0.0,
        )
        info.update(
            {
                "task_id": self.config["id"],
                "swing_leg": self.swing_leg,
                "target_lift_m": self.target_lift_m,
                "base_support_mode": self.base_support_mode,
                "observation_fields": FOOT_LIFT_OBSERVATION_FIELDS,
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
        elapsed_seconds = (self.episode_step + 1) * self.control_dt_s
        desired_lift = desired_foot_lift_m(
            elapsed_seconds,
            target_lift_m=self.reference_lift_m,
            ramp_start_seconds=self.ramp_start_seconds,
            ramp_duration_seconds=self.ramp_duration_seconds,
        )
        reference_target = self._reference_targets(
            elapsed_seconds=elapsed_seconds,
            desired_lift_m=desired_lift,
        )
        desired_target = np.clip(
            reference_target + self.action_scale * clipped_action,
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
        if self.base_support_mode == "pose_hold":
            self.robot.set_world_poses(
                positions=[self.episode_base_origin.tolist()],
                orientations=[self.episode_base_orientation.tolist()],
            )
            self.robot.set_velocities(
                linear_velocities=[[0.0, 0.0, 0.0]],
                angular_velocities=[[0.0, 0.0, 0.0]],
            )
        self.previous_action = clipped_action.copy()
        state = super()._read_state()
        foot_tips = self._sample_foot_tips()
        foot_lifts = foot_tips[:, 2] - self.initial_foot_tip_z_m
        measured_lift = max(0.0, float(foot_lifts[self.swing_leg_index]))
        support_lift = max(
            0.0,
            float(np.max(foot_lifts[list(self.support_leg_indices)])),
        )
        previous_maximum_lift = self.maximum_swing_foot_lift_m
        maximum_lift = max(previous_maximum_lift, measured_lift)
        lift_progress = max(0.0, maximum_lift - previous_maximum_lift)
        maximum_support_lift = max(
            self.maximum_support_foot_lift_m,
            support_lift,
        )
        base_position = np.asarray(state["base_position"])
        base_displacement = base_position - self.episode_base_origin
        base_height_error = float(base_displacement[2])
        base_displacement_xy = base_displacement[:2]
        imu_observation = np.asarray(state["imu_observation"])
        projected_gravity = imu_observation[3:6]
        failure_reasons = foot_lift_failure_reasons(
            base_height_m=float(base_position[2]),
            projected_gravity_xyz=projected_gravity,
            base_displacement_xy_m=base_displacement_xy,
            maximum_support_foot_lift_m=support_lift,
            minimum_base_height_m=float(self.termination_config["minimum_base_height_m"]),
            minimum_upright_cosine=float(self.termination_config["minimum_upright_cosine"]),
            maximum_base_displacement_m=float(
                self.termination_config["maximum_base_displacement_m"]
            ),
            maximum_support_foot_lift_allowed_m=float(
                self.termination_config["maximum_support_foot_lift_m"]
            ),
        )
        failed = bool(failure_reasons)
        success_now = not failed and foot_lift_success_reached(
            desired_lift_m=desired_lift,
            measured_lift_m=measured_lift,
            target_lift_m=self.target_lift_m,
            minimum_success_lift_m=float(self.success_config["minimum_lift_m"]),
            projected_gravity_xyz=projected_gravity,
            base_height_error_m=base_height_error,
            base_displacement_xy_m=base_displacement_xy,
            maximum_support_foot_lift_m=support_lift,
            minimum_upright_cosine=float(self.success_config["minimum_upright_cosine"]),
            maximum_base_height_error_m=float(self.success_config["maximum_base_height_error_m"]),
            maximum_base_displacement_m=float(self.success_config["maximum_base_displacement_m"]),
            maximum_support_foot_lift_allowed_m=float(
                self.success_config["maximum_support_foot_lift_m"]
            ),
        )
        self.goal_hold_step_count = self.goal_hold_step_count + 1 if success_now else 0
        succeeded = self.goal_hold_step_count >= self.success_hold_steps
        terminated = failed or succeeded
        self.episode_step += 1
        truncated = self.episode_step >= self.max_episode_steps
        normalized_joint_velocity = np.asarray(state["joint_velocities"]) / self.max_velocities
        reward_terms = foot_lift_reward_terms(
            desired_lift_m=desired_lift,
            measured_lift_m=measured_lift,
            lift_progress_m=lift_progress,
            tracking_target_reached=success_now,
            base_height_error_m=base_height_error,
            base_displacement_xy_m=base_displacement_xy,
            maximum_support_foot_lift_m=support_lift,
            body_angular_velocity_xyz=imu_observation[:3],
            projected_gravity_xyz=projected_gravity,
            joint_velocities_normalized=normalized_joint_velocity,
            action=clipped_action,
            previous_action=prior_action,
            failed=failed,
            succeeded=succeeded,
            reward_config=self.reward_config,
        )
        reward = float(reward_terms["total"])
        self.episode_return += reward
        self.minimum_height_m = min(self.minimum_height_m, float(base_position[2]))
        upright_cosine = float(np.clip(-projected_gravity[2], -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(upright_cosine)))
        self.maximum_tilt_deg = max(self.maximum_tilt_deg, tilt_deg)
        self.maximum_swing_foot_lift_m = maximum_lift
        self.maximum_support_foot_lift_m = maximum_support_lift
        self.current_swing_foot_lift_m = measured_lift
        self.desired_swing_foot_lift_m = desired_lift
        observation = self._skill_observation(
            state,
            desired_lift_m=desired_lift,
            measured_lift_m=measured_lift,
            maximum_lift_m=maximum_lift,
            base_height_error_m=base_height_error,
            base_displacement_xy_m=base_displacement_xy,
            support_foot_lift_m=support_lift,
        )
        info: dict[str, object] = {
            "reward_terms": reward_terms,
            "base_position_m": base_position.copy(),
            "base_displacement_m": base_displacement.copy(),
            "foot_tip_positions_m": foot_tips.copy(),
            "desired_swing_foot_lift_m": desired_lift,
            "measured_swing_foot_lift_m": measured_lift,
            "maximum_swing_foot_lift_m": maximum_lift,
            "maximum_support_foot_lift_m": maximum_support_lift,
            "goal_hold_duration_s": self.goal_hold_step_count / self.control_hz,
            "failure_reasons": failure_reasons,
            "succeeded": succeeded,
            "target_joint_positions_rad": target.copy(),
            "reference_joint_positions_rad": reference_target.copy(),
        }
        if terminated or truncated:
            episode_metrics = {
                "return": self.episode_return,
                "length_steps": self.episode_step,
                "duration_s": self.episode_step / self.control_hz,
                "swing_leg": self.swing_leg,
                "base_support_mode": self.base_support_mode,
                "target_lift_m": self.target_lift_m,
                "reference_lift_m": self.reference_lift_m,
                "maximum_swing_foot_lift_m": maximum_lift,
                "final_swing_foot_lift_m": measured_lift,
                "maximum_support_foot_lift_m": maximum_support_lift,
                "base_displacement_m": base_displacement.tolist(),
                "minimum_base_height_m": self.minimum_height_m,
                "maximum_body_tilt_deg": self.maximum_tilt_deg,
                "goal_hold_duration_s": (self.goal_hold_step_count / self.control_hz),
                "skill_completed": succeeded,
                "terminated": terminated,
                "truncated": truncated,
                "failure_reasons": failure_reasons,
            }
            info["episode_metrics"] = episode_metrics
            self.completed_episode_metrics.append(episode_metrics)
            self.completed_episode_metrics = self.completed_episode_metrics[-20:]
        self.previous_target = target.copy()
        return observation, reward, terminated, truncated, info

    @property
    def contract(self) -> dict[str, object]:
        contract = dict(super().contract)
        contract.update(
            {
                "task_id": self.config["id"],
                "observation_fields": list(FOOT_LIFT_OBSERVATION_FIELDS),
                "observation_size": FOOT_LIFT_OBSERVATION_SIZE,
                "walking_observation_size": 48,
                "action_size": 12,
                "physics_steps_per_control": self.physics_steps_per_control,
                "control_action_mode": "ik_reference_plus_ppo_residual",
                "swing_leg": self.swing_leg,
                "target_lift_m": self.target_lift_m,
                "foot_lift": self.lift_config,
                "weight_shift": self.weight_shift_config,
                "base_support": self.base_support_config,
                "success": self.success_config,
                "rgb_camera_policy_input": False,
                "terrain_perception_policy_input": False,
                "reference_note": (
                    "Analytic leg IK commands a raise-forward arc that stays "
                    "inside the measured knee limit; PPO controls bounded "
                    "residuals on all 12 joints for balance."
                ),
            }
        )
        return contract
