"""Pure-RL vectorized commanded-walking environment for Drobot."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu

from . import preview_control
from .commanded_walking_env_cfg import (
    SERVO_VELOCITY_LIMIT_RAD_S,
    DrobotCommandedWalkingForwardEnvCfg,
)


class DrobotCommandedWalkingEnv(DirectRLEnv):
    """Learn locomotion from velocity commands without a prescribed gait phase."""

    cfg: DrobotCommandedWalkingForwardEnvCfg
    def __init__(
        self,
        cfg: DrobotCommandedWalkingForwardEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_actions = torch.zeros_like(self._actions)
        self._previous_targets = self._robot.data.default_joint_pos.torch.clone()
        self._commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._joint_scale = torch.tensor(
            [
                self.cfg.action_scale_abduction_rad
                if name.endswith("hip_abduction")
                else self.cfg.action_scale_hip_rad
                if name.endswith("hip_flexion")
                else self.cfg.action_scale_knee_rad
                for name in self._robot.joint_names
            ],
            dtype=torch.float32,
            device=self.device,
        )
        if len(self._robot.joint_names) != 12:
            raise RuntimeError(f"Expected 12 joints, got {self._robot.joint_names}")
        self._base_sensor_ids, _ = self._contact_sensor.find_sensors("base_link")
        if len(self._base_sensor_ids) == 0:
            raise RuntimeError("Could not find base_link in the contact sensor")

        self._steps_since_reset = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_start_position = torch.zeros((self.num_envs, 3), device=self.device)
        self._episode_velocity_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_yaw_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_commanded_distance = torch.zeros(self.num_envs, device=self.device)
        self._episode_action_saturation_sum = torch.zeros(
            self.num_envs, device=self.device
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._imu_sensor = Imu(self.cfg.imu_sensor)
        self.scene.sensors["imu_sensor"] = self._imu_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        positions = cloner.grid_transforms(
            self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device
        )[0]
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, self.device, positions)
        cloner.replicate(plan, stage=self.scene.stage)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.82, 0.82, 0.82))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions.copy_(self._actions)
        self._actions = torch.clamp(actions, -1.0, 1.0)
        desired_targets = (
            self._robot.data.default_joint_pos.torch + self._joint_scale * self._actions
        )
        limits = self._robot.data.soft_joint_pos_limits.torch
        desired_targets = torch.clamp(
            desired_targets, limits[:, :, 0], limits[:, :, 1]
        )
        max_delta = SERVO_VELOCITY_LIMIT_RAD_S * self.step_dt
        self._processed_actions = torch.clamp(
            desired_targets,
            self._previous_targets - max_delta,
            self._previous_targets + max_delta,
        )
        self._previous_targets.copy_(self._processed_actions)

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self._processed_actions)

    def _base_contact_force(self) -> torch.Tensor:
        force = self._contact_sensor.data.net_forces_w.torch[:, self._base_sensor_ids]
        return torch.linalg.norm(force, dim=-1).amax(dim=1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observation = torch.cat(
            (
                self._commands,
                self._imu_sensor.data.ang_vel_b.torch * 0.25,
                self._robot.data.projected_gravity_b.torch,
                self._imu_sensor.data.lin_acc_b.torch / 9.81,
                self._robot.data.joint_pos.torch - self._robot.data.default_joint_pos.torch,
                self._robot.data.joint_vel.torch / SERVO_VELOCITY_LIMIT_RAD_S,
                self._previous_actions,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(observation, -20.0, 20.0)}

    def _get_rewards(self) -> torch.Tensor:
        linear_velocity = self._robot.data.root_lin_vel_b.torch
        angular_velocity = self._robot.data.root_ang_vel_b.torch
        linear_error = linear_velocity[:, :2] - self._commands[:, :2]
        velocity_error_squared = torch.sum(torch.square(linear_error), dim=1)
        yaw_error = angular_velocity[:, 2] - self._commands[:, 2]
        velocity_tracking = torch.exp(
            -velocity_error_squared / self.cfg.velocity_tracking_sigma_m_s**2
        )
        yaw_tracking = torch.exp(
            -torch.square(yaw_error / self.cfg.yaw_tracking_sigma_rad_s)
        )

        command_speed = torch.linalg.norm(self._commands[:, :2], dim=1)
        command_direction = self._commands[:, :2] / torch.clamp(
            command_speed.unsqueeze(1), min=1.0e-4
        )
        commanded_velocity = torch.sum(linear_velocity[:, :2] * command_direction, dim=1)
        commanded_progress = torch.where(
            command_speed > 0.03,
            torch.clamp(commanded_velocity, -0.25, 0.35),
            torch.zeros_like(commanded_velocity),
        )
        upright_cosine = torch.clamp(
            -self._robot.data.projected_gravity_b.torch[:, 2], 0.0, 1.0
        )
        height_error = self._robot.data.root_pos_w.torch[:, 2] - (
            self._terrain.env_origins[:, 2] + self.cfg.target_base_height_m
        )
        action_rate = torch.mean(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        action_magnitude = torch.mean(torch.square(self._actions), dim=1)
        joint_speed = torch.mean(
            torch.square(self._robot.data.joint_vel.torch / SERVO_VELOCITY_LIMIT_RAD_S),
            dim=1,
        )
        effort = torch.mean(
            torch.square(self._robot.data.applied_torque.torch / 0.8825985), dim=1
        )
        base_contact = (self._base_contact_force() > 5.0) & (
            self._steps_since_reset > self.cfg.base_contact_grace_steps
        )

        next_commanded_distance = (
            self._episode_commanded_distance + commanded_velocity * self.step_dt
        )
        expected_distance = command_speed * self.cfg.episode_length_s
        success_distance = self.cfg.distance_success_fraction * expected_distance
        active_translation = command_speed > 0.03
        terminal_progress_fraction = torch.where(
            active_translation,
            torch.clamp(
                next_commanded_distance / torch.clamp(success_distance, min=1.0e-4),
                0.0,
                1.0,
            ),
            torch.zeros_like(command_speed),
        )
        distance_success = (
            self.reset_time_outs
            & active_translation
            & (next_commanded_distance >= success_distance)
            & ~self._failed
        )
        stationary = active_translation & (
            commanded_velocity < 0.25 * command_speed
        )
        terminal_progress_reward = self.reset_time_outs.float() * (~self._failed).float() * (
            self.cfg.terminal_progress_reward_scale * terminal_progress_fraction
            + self.cfg.distance_success_reward * distance_success.float()
        )

        reward = (
            3.0 * velocity_tracking
            + 0.35 * yaw_tracking
            + 5.0 * commanded_progress
            + 0.20 * upright_cosine
            + 0.01
            + terminal_progress_reward
            - 0.12 * torch.square(linear_velocity[:, 2])
            - 0.04 * torch.sum(torch.square(angular_velocity[:, :2]), dim=1)
            - 1.5 * torch.square(height_error)
            - 0.012 * action_rate
            - 0.002 * action_magnitude
            - 0.001 * joint_speed
            - 0.0005 * effort
            - 0.35 * stationary.float()
            - 2.0 * base_contact.float()
        )
        reward = torch.where(self._failed, reward - 5.0, reward)

        self._episode_velocity_error_sum += torch.sqrt(velocity_error_squared)
        self._episode_yaw_error_sum += torch.abs(yaw_error)
        self._episode_commanded_distance.copy_(next_commanded_distance)
        self._episode_action_saturation_sum += torch.mean(
            (torch.abs(self._actions) >= 0.98).float(), dim=1
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._steps_since_reset += 1
        root_position = self._robot.data.root_pos_w.torch
        local_position = root_position - self._terrain.env_origins
        upright_cosine = -self._robot.data.projected_gravity_b.torch[:, 2]
        base_contact = (self._base_contact_force() > 5.0) & (
            self._steps_since_reset > self.cfg.base_contact_grace_steps
        )
        finite = torch.isfinite(root_position).all(dim=1) & torch.isfinite(
            self._robot.data.joint_pos.torch
        ).all(dim=1)
        self._failed = (
            ~finite
            | (local_position[:, 2] < self.cfg.minimum_base_height_m)
            | (upright_cosine < self.cfg.minimum_upright_cosine)
            | (
                torch.linalg.norm(local_position[:, :2], dim=1)
                > self.cfg.maximum_distance_from_origin_m
            )
            | base_contact
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._failed, time_out

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        self._commands[env_ids] = 0.0
        override = preview_control.COMMAND_OVERRIDE
        if override is not None:
            self._commands[env_ids] = torch.tensor(
                override, dtype=torch.float32, device=self.device
            )
            return

        if self.cfg.command_profile == "forward":
            curriculum_fraction = min(
                float(self.common_step_counter) / self.cfg.command_curriculum_steps,
                1.0,
            )
            speed_min = self.cfg.initial_forward_speed_min_m_s + curriculum_fraction * (
                self.cfg.forward_speed_min_m_s
                - self.cfg.initial_forward_speed_min_m_s
            )
            speed_max = self.cfg.initial_forward_speed_max_m_s + curriculum_fraction * (
                self.cfg.forward_speed_max_m_s
                - self.cfg.initial_forward_speed_max_m_s
            )
            self._commands[env_ids, 0] = torch.empty(
                count, device=self.device
            ).uniform_(speed_min, speed_max)
            return
        if self.cfg.command_profile != "directional":
            raise ValueError(f"Unknown command profile: {self.cfg.command_profile}")

        # 50% forward, 15% backward, 15% left, 15% right, 5% stop.
        selector = torch.rand(count, device=self.device)
        forward = selector < 0.50
        backward = (selector >= 0.50) & (selector < 0.65)
        left = (selector >= 0.65) & (selector < 0.80)
        right = (selector >= 0.80) & (selector < 0.95)
        random_unit = torch.rand(count, device=self.device)
        self._commands[env_ids[forward], 0] = (
            self.cfg.forward_speed_min_m_s
            + random_unit[forward]
            * (self.cfg.forward_speed_max_m_s - self.cfg.forward_speed_min_m_s)
        )
        self._commands[env_ids[backward], 0] = -(
            self.cfg.backward_speed_min_m_s
            + random_unit[backward]
            * (self.cfg.backward_speed_max_m_s - self.cfg.backward_speed_min_m_s)
        )
        turn_speed = torch.rand(count, device=self.device) * self.cfg.turn_forward_speed_max_m_s
        turn_rate = self.cfg.turn_rate_min_rad_s + torch.rand(count, device=self.device) * (
            self.cfg.turn_rate_max_rad_s - self.cfg.turn_rate_min_rad_s
        )
        turning = left | right
        self._commands[env_ids[turning], 0] = turn_speed[turning]
        self._commands[env_ids[left], 2] = turn_rate[left]
        self._commands[env_ids[right], 2] = -turn_rate[right]

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if len(env_ids) > 0:
            completed = self._steps_since_reset[env_ids] > 0
            completed_steps = torch.clamp(self._steps_since_reset[env_ids], min=1).float()
            log = self.extras.setdefault("log", {})
            log["Metrics/mean_velocity_error_m_s"] = (
                self._episode_velocity_error_sum[env_ids] / completed_steps
            ).mean().item()
            log["Metrics/mean_yaw_error_rad_s"] = (
                self._episode_yaw_error_sum[env_ids] / completed_steps
            ).mean().item()
            log["Metrics/commanded_distance_m"] = self._episode_commanded_distance[
                env_ids
            ].mean().item()
            episode_duration = completed_steps * self.step_dt
            log["Metrics/mean_commanded_speed_m_s"] = (
                self._episode_commanded_distance[env_ids] / episode_duration
            ).mean().item()
            motion_command = torch.linalg.norm(
                self._commands[env_ids, :2], dim=1
            ) > 0.03
            success_distance = (
                self.cfg.distance_success_fraction
                * torch.linalg.norm(self._commands[env_ids, :2], dim=1)
                * self.cfg.episode_length_s
            )
            distance_success = (
                completed
                & motion_command
                & ~self._failed[env_ids]
                & (self._episode_commanded_distance[env_ids] >= success_distance)
            )
            completed_motion = completed & motion_command
            log["Metrics/distance_success_rate"] = (
                distance_success.float().sum()
                / torch.clamp(completed_motion.float().sum(), min=1.0)
            ).item()
            log["Metrics/action_saturation_rate"] = (
                self._episode_action_saturation_sum[env_ids] / completed_steps
            ).mean().item()
            log["Metrics/fall_rate"] = (
                (self._failed[env_ids] & completed).float().sum()
                / torch.clamp(completed.float().sum(), min=1.0)
            ).item()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        joint_position = self._robot.data.default_joint_pos.torch[env_ids].clone()
        joint_position += self.cfg.reset_joint_position_noise_rad * (
            2.0 * torch.rand_like(joint_position) - 1.0
        )
        joint_velocity = torch.zeros_like(self._robot.data.default_joint_vel.torch[env_ids])
        root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        root_pose[:, :2] += self.cfg.reset_xy_jitter_m * (
            2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
        )
        root_velocity = torch.zeros_like(self._robot.data.default_root_vel.torch[env_ids])

        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=env_ids
        )
        self._robot.write_joint_position_to_sim_index(
            position=joint_position, env_ids=env_ids
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=joint_velocity, env_ids=env_ids
        )

        self._sample_commands(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_targets[env_ids] = joint_position
        self._steps_since_reset[env_ids] = 0
        self._failed[env_ids] = False
        self._episode_start_position[env_ids] = root_pose[:, :3]
        self._episode_velocity_error_sum[env_ids] = 0.0
        self._episode_yaw_error_sum[env_ids] = 0.0
        self._episode_commanded_distance[env_ids] = 0.0
        self._episode_action_saturation_sum[env_ids] = 0.0
