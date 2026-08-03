"""Pure-reward vectorized stair environment for Drobot."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply

from .pure_stairs_env_cfg import (
    LOW_FOLD_HIP_RAD,
    LOW_FOLD_KNEE_RAD,
    NORMAL_HIP_RAD,
    NORMAL_KNEE_RAD,
    DrobotPureStairsEnvCfg,
)

DISTAL_LINK_LENGTH_M = 0.159896689
FOOT_CONTACT_RADIUS_M = 0.0125


class DrobotPureStairsEnv(DirectRLEnv):
    """Learn stair climbing without prescribed phases, leg order, or trajectories."""

    cfg: DrobotPureStairsEnvCfg

    def __init__(
        self,
        cfg: DrobotPureStairsEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_actions = torch.zeros_like(self._actions)
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

        self._base_ids, _ = self._contact_sensor.find_sensors("base_link")
        self._foot_sensor_ids, self._foot_sensor_names = self._contact_sensor.find_sensors(
            ".*_distal_link"
        )
        self._foot_body_ids, self._foot_body_names = self._robot.find_bodies(
            ".*_distal_link"
        )
        if len(self._foot_sensor_ids) != 4 or len(self._foot_body_ids) != 4:
            raise RuntimeError(
                "Expected four distal foot bodies; got "
                f"sensor={self._foot_sensor_names}, bodies={self._foot_body_names}"
            )

        self._depth_pending = torch.ones((self.num_envs, 24), device=self.device)
        self._depth_observation = torch.ones_like(self._depth_pending)
        self._depth_frame = 0
        self._previous_root_x = torch.zeros(self.num_envs, device=self.device)
        self._previous_root_z = torch.zeros(self.num_envs, device=self.device)
        self._previous_foot_tip_z = torch.zeros((self.num_envs, 4), device=self.device)
        self._episode_start_root_x = torch.zeros(self.num_envs, device=self.device)
        self._episode_start_root_z = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_progress = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_base_gain = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_supported_base_gain = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_max_four_support_base_gain = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_max_support_rise_base_gain = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_max_foot_clearance = torch.zeros(self.num_envs, device=self.device)
        self._episode_best_tread_contacts = torch.zeros(self.num_envs, device=self.device)
        self._episode_best_centered_tread_contacts = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_best_supported_center_approach = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_best_tread_potential = torch.zeros(self.num_envs, device=self.device)
        self._episode_best_narrow_tread_potential = torch.zeros(
            self.num_envs, device=self.device
        )
        self._tread_hold_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_max_tread_hold_steps = torch.zeros_like(self._tread_hold_steps)
        self._lift_hold_steps = torch.zeros_like(self._tread_hold_steps)
        self._episode_max_lift_hold_steps = torch.zeros_like(self._tread_hold_steps)
        self._support_rise_hold_steps = torch.zeros_like(self._tread_hold_steps)
        self._episode_max_support_rise_hold_steps = torch.zeros_like(
            self._tread_hold_steps
        )
        self._steps_since_reset = torch.zeros_like(self._tread_hold_steps)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._failed = torch.zeros_like(self._success)
        self._completed_episode_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._successful_episode_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._easy_reset_episode_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._easy_reset_success_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._hard_reset_episode_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._hard_reset_success_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._supported_base_gain_episode_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._supported_base_gain_sum_m = torch.zeros((), device=self.device)
        self._supported_base_gain_max_m = torch.zeros((), device=self.device)
        self._four_support_gain_episode_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._four_support_gain_sum_m = torch.zeros((), device=self.device)
        self._four_support_gain_max_m = torch.zeros((), device=self.device)
        self._support_rise_gain_episode_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._support_rise_gain_sum_m = torch.zeros((), device=self.device)
        self._support_rise_gain_max_m = torch.zeros((), device=self.device)
        self._reset_fold_fraction = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._depth_sensor = RayCaster(self.cfg.depth_sensor)
        self.scene.sensors["depth_sensor"] = self._depth_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        positions = cloner.grid_transforms(
            self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device
        )[0]
        plan = cloner.clone_plan_from_env_0(
            src, dest, self.scene.num_envs, self.device, positions
        )
        cloner.replicate(plan, stage=self.scene.stage)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)

    def close(self):
        """Report exact episode totals for bounded deterministic evaluations."""

        if getattr(self.cfg, "report_episode_totals_on_close", False) and hasattr(
            self, "_completed_episode_count"
        ):
            completed = int(self._completed_episode_count.item())
            successful = int(self._successful_episode_count.item())
            rate = successful / max(completed, 1)
            easy_completed = int(self._easy_reset_episode_count.item())
            easy_successful = int(self._easy_reset_success_count.item())
            hard_completed = int(self._hard_reset_episode_count.item())
            hard_successful = int(self._hard_reset_success_count.item())
            supported_gain_episodes = int(self._supported_base_gain_episode_count.item())
            supported_gain_mean_m = float(
                self._supported_base_gain_sum_m.item() / max(completed, 1)
            )
            supported_gain_max_m = float(self._supported_base_gain_max_m.item())
            four_support_gain_episodes = int(self._four_support_gain_episode_count.item())
            four_support_gain_mean_m = float(
                self._four_support_gain_sum_m.item() / max(completed, 1)
            )
            four_support_gain_max_m = float(self._four_support_gain_max_m.item())
            support_rise_gain_episodes = int(
                self._support_rise_gain_episode_count.item()
            )
            support_rise_gain_mean_m = float(
                self._support_rise_gain_sum_m.item() / max(completed, 1)
            )
            support_rise_gain_max_m = float(self._support_rise_gain_max_m.item())
            print(
                "[DROBOT_EPISODE_TOTALS] "
                f"completed={completed} successful={successful} rate={rate:.8f} "
                f"easy_completed={easy_completed} easy_successful={easy_successful} "
                f"hard_completed={hard_completed} hard_successful={hard_successful} "
                f"supported_gain_episodes={supported_gain_episodes} "
                f"supported_gain_mean_m={supported_gain_mean_m:.8f} "
                f"supported_gain_max_m={supported_gain_max_m:.8f} "
                f"four_support_gain_episodes={four_support_gain_episodes} "
                f"four_support_gain_mean_m={four_support_gain_mean_m:.8f} "
                f"four_support_gain_max_m={four_support_gain_max_m:.8f} "
                f"support_rise_gain_episodes={support_rise_gain_episodes} "
                f"support_rise_gain_mean_m={support_rise_gain_mean_m:.8f} "
                f"support_rise_gain_max_m={support_rise_gain_max_m:.8f}"
            )
        super().close()

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions.copy_(self._actions)
        self._actions = torch.clamp(actions, -1.0, 1.0)
        self._processed_actions = (
            self._robot.data.default_joint_pos.torch + self._joint_scale * self._actions
        )
        limits = self._robot.data.soft_joint_pos_limits.torch
        self._processed_actions = torch.clamp(
            self._processed_actions, limits[:, :, 0], limits[:, :, 1]
        )

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self._processed_actions)

    def _foot_forces(self) -> torch.Tensor:
        forces = self._contact_sensor.data.net_forces_w.torch[:, self._foot_sensor_ids]
        return torch.linalg.norm(forces, dim=-1)

    def _base_forces(self) -> torch.Tensor:
        forces = self._contact_sensor.data.net_forces_w.torch[:, self._base_ids]
        return torch.linalg.norm(forces, dim=-1).amax(dim=1)

    def _terrain_height(self, x_from_origin: torch.Tensor) -> torch.Tensor:
        step = torch.floor(
            (x_from_origin - self.cfg.stair_start_from_origin_m)
            / self.cfg.stair_tread_depth_m
        ) + 1.0
        step = torch.clamp(step, 0.0, float(self.cfg.stair_step_count))
        return step * self.cfg.stair_rise_m

    def _foot_tip_positions(self) -> torch.Tensor:
        """Return the four modeled fork-tip contact points in world coordinates."""
        body_pos = self._robot.data.body_pos_w.torch[:, self._foot_body_ids]
        body_quat = self._robot.data.body_quat_w.torch[:, self._foot_body_ids]
        local_tip = torch.zeros_like(body_pos)
        local_tip[:, :, 0] = DISTAL_LINK_LENGTH_M
        rotated_tip = quat_apply(
            body_quat.reshape(-1, 4), local_tip.reshape(-1, 3)
        ).reshape(self.num_envs, 4, 3)
        tip_pos = body_pos + rotated_tip
        tip_pos[:, :, 2] -= FOOT_CONTACT_RADIUS_M
        return tip_pos

    def _read_depth(self) -> torch.Tensor:
        hits = self._depth_sensor.data.ray_hits_w.torch
        starts = self._depth_sensor.data.pos_w.torch.unsqueeze(1)
        distances = torch.linalg.norm(hits - starts, dim=-1)
        distances = torch.nan_to_num(
            distances, nan=4.0, posinf=4.0, neginf=4.0
        ).clamp(0.02, 4.0)
        grid = distances.reshape(self.num_envs, 8, 8)

        accuracy = torch.where(grid <= 0.20, 0.015, 0.05 * grid)
        grid = grid + (2.0 * torch.rand_like(grid) - 1.0) * accuracy
        dropout = torch.rand_like(grid) < 0.05
        grid = torch.where(dropout, torch.full_like(grid, 4.0), grid).clamp(0.02, 4.0)

        lanes = [grid[:, :, 0:3], grid[:, :, 3:5], grid[:, :, 5:8]]
        compressed = torch.cat([lane.mean(dim=2) for lane in lanes], dim=1)
        return (compressed / 1.5).clamp(0.0, 1.0)

    def _update_depth_latency(self):
        # Control is 30 Hz; every second control frame is one 15 Hz sensor frame.
        self._depth_frame += 1
        if self._depth_frame % 2 == 0:
            self._depth_observation.copy_(self._depth_pending)
            self._depth_pending.copy_(self._read_depth())

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_depth_latency()
        forces = self._foot_forces()
        contacts = torch.tanh(forces / 20.0)
        obs = torch.cat(
            (
                self._robot.data.root_ang_vel_b.torch * 0.25,
                self._robot.data.projected_gravity_b.torch,
                self._robot.data.joint_pos.torch
                - self._robot.data.default_joint_pos.torch,
                self._robot.data.joint_vel.torch * 0.20,
                self._previous_actions,
                contacts,
                self._depth_observation,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -10.0, 10.0)}

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self._robot.data.root_pos_w.torch
        root_x = root_pos[:, 0]
        root_z = root_pos[:, 2]

        foot_pos = self._foot_tip_positions()
        foot_x = foot_pos[:, :, 0] - self._terrain.env_origins[:, 0:1]
        foot_ground = self._terrain_height(foot_x)
        foot_clearance = foot_pos[:, :, 2] - self._terrain.env_origins[:, 2:3] - foot_ground
        max_clearance = foot_clearance.max(dim=1).values

        forces = self._foot_forces()
        contact = forces > 1.0
        on_tread = contact & (torch.abs(foot_clearance) < 0.025) & (foot_ground > 0.0)
        tread_contacts = torch.sum(on_tread.float(), dim=1)
        supported = contact & (torch.abs(foot_clearance) < 0.03)
        support_count = torch.sum(supported.float(), dim=1)
        ground_contacts = torch.sum((supported & (foot_ground == 0.0)).float(), dim=1)
        retained_support = torch.clamp(tread_contacts, 0.0, 1.0) * torch.clamp(
            ground_contacts, 0.0, 2.0
        )
        base_contact = (self._base_forces() > 5.0) & (
            self._steps_since_reset > self.cfg.base_contact_grace_steps
        )

        # Symmetric, phase-free placement potential. Every foot is compared with
        # every tread; the actor never receives these simulator coordinates.
        tread_index = torch.arange(
            1, self.cfg.reward_tread_count + 1, device=self.device, dtype=torch.float32
        )
        tread_center_x = self.cfg.stair_start_from_origin_m + (
            tread_index - 0.5
        ) * self.cfg.stair_tread_depth_m
        tread_height = tread_index * self.cfg.stair_rise_m
        tip_x_error = foot_x.unsqueeze(-1) - tread_center_x.view(1, 1, -1)
        tip_z_local = foot_pos[:, :, 2] - self._terrain.env_origins[:, 2:3]
        tip_z_error = tip_z_local.unsqueeze(-1) - tread_height.view(1, 1, -1)
        centered_on_tread = on_tread & (
            torch.abs(tip_x_error).amin(dim=2) <= self.cfg.centered_tread_half_width_m
        )
        centered_tread_contacts = torch.sum(centered_on_tread.float(), dim=1)
        placement_score = torch.exp(
            -torch.square(tip_x_error / 0.35) - torch.square(tip_z_error / 0.20)
        )
        tread_potential = placement_score.amax(dim=(1, 2))
        new_tread_potential = torch.clamp(
            tread_potential - self._episode_best_tread_potential, 0.0, 0.10
        )
        # The broad potential preserves discovery from low foot poses. This
        # narrower surface band distinguishes a landing approach from merely
        # lifting near the stair. It remains symmetric over every foot/tread.
        narrow_placement_score = torch.exp(
            -torch.square(tip_x_error / 0.13) - torch.square(tip_z_error / 0.055)
        )
        narrow_tread_potential = narrow_placement_score.amax(dim=(1, 2))
        new_narrow_tread_potential = torch.clamp(
            narrow_tread_potential - self._episode_best_narrow_tread_potential,
            0.0,
            0.15,
        )
        other_support_count = support_count.unsqueeze(1) - supported.float()
        three_other_supports = torch.clamp(other_support_count - 2.0, 0.0, 1.0)
        per_foot_center_approach = narrow_placement_score.amax(dim=2)
        supported_center_approach = (
            per_foot_center_approach * three_other_supports
        ).amax(dim=1)
        foot_tip_descent = torch.clamp(
            self._previous_foot_tip_z - foot_pos[:, :, 2], 0.0, 0.03
        ) / 0.03
        above_tread_band = (tip_z_error >= 0.0) & (tip_z_error <= 0.12)
        descending_pair_score = (
            torch.exp(-torch.square(tip_x_error / 0.10))
            * torch.exp(-torch.square(tip_z_error / 0.07))
            * above_tread_band.float()
        )
        descending_center_approach = (
            descending_pair_score.amax(dim=2)
            * foot_tip_descent
            * three_other_supports
        ).amax(dim=1)

        progress_delta = torch.clamp(root_x - self._previous_root_x, -0.02, 0.03)
        height_delta = torch.clamp(root_z - self._previous_root_z, -0.02, 0.03)
        new_clearance = torch.clamp(
            max_clearance - self._episode_max_foot_clearance, 0.0, 0.04
        )
        lift_hold = torch.clamp(max_clearance / 0.19, 0.0, 1.0)
        supported_lift = lift_hold * torch.clamp(support_count - 2.0, 0.0, 1.0)
        tread_binary = torch.clamp(tread_contacts, 0.0, 1.0)
        centered_tread_binary = torch.clamp(centered_tread_contacts, 0.0, 1.0)
        required_tread_binary = (
            centered_tread_binary
            if self.cfg.first_step_require_centered_contact
            else tread_binary
        )
        retained_ground_support = required_tread_binary * torch.clamp(
            ground_contacts / 3.0, 0.0, 1.0
        )
        tread_hold_fraction = torch.clamp(
            self._tread_hold_steps.float() / max(self.cfg.first_step_hold_steps, 1),
            0.0,
            1.0,
        )
        base_gain_fraction = torch.clamp(
            (root_z - self._episode_start_root_z)
            / max(self.cfg.first_step_min_base_gain_m, 1.0e-6),
            0.0,
            1.0,
        )
        tread_transfer = required_tread_binary * base_gain_fraction
        full_support = torch.clamp(support_count - 3.0, 0.0, 1.0)
        supported_transfer_gate = required_tread_binary * full_support
        supported_transfer = supported_transfer_gate * base_gain_fraction
        supported_base_gain = supported_transfer_gate * torch.clamp(
            root_z - self._episode_start_root_z, 0.0, 0.18
        )
        support_rise_gain_fraction = torch.clamp(
            (root_z - self._episode_start_root_z)
            / max(self.cfg.support_rise_min_base_gain_m, 1.0e-6),
            0.0,
            1.0,
        )
        if self.cfg.support_rise_min_support_count > 0:
            support_rise_gate = torch.clamp(
                support_count - float(self.cfg.support_rise_min_support_count - 1),
                0.0,
                1.0,
            )
        else:
            support_rise_gate = torch.ones_like(support_count)
        support_rise = support_rise_gate * support_rise_gain_fraction
        support_rise_base_gain = support_rise_gate * torch.clamp(
            root_z - self._episode_start_root_z, 0.0, 0.18
        )
        four_support_base_gain = full_support * torch.clamp(
            root_z - self._episode_start_root_z, 0.0, 0.18
        )
        narrow_transfer = narrow_tread_potential * base_gain_fraction
        upright_error = torch.sum(
            torch.square(self._robot.data.projected_gravity_b.torch[:, :2]), dim=1
        )
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        effort = torch.sum(
            torch.square(self._robot.data.applied_torque.torch / 0.8825985), dim=1
        )
        body_rate = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b.torch[:, :2]), dim=1
        )
        first_step_completion = self._success.float()

        reward = (
            0.01
            + self.cfg.progress_delta_reward_scale * progress_delta
            + self.cfg.height_delta_reward_scale * height_delta
            + self.cfg.clearance_reward_scale * new_clearance
            + self.cfg.lift_hold_reward_scale * lift_hold
            + self.cfg.new_tread_potential_reward_scale * new_tread_potential
            + self.cfg.tread_potential_reward_scale * tread_potential
            + self.cfg.new_narrow_tread_potential_reward_scale
            * new_narrow_tread_potential
            + self.cfg.narrow_tread_potential_reward_scale * narrow_tread_potential
            + self.cfg.support_reward_scale * support_count
            + self.cfg.supported_lift_reward_scale * supported_lift
            + self.cfg.tread_contact_reward_scale * tread_contacts
            + self.cfg.centered_tread_contact_reward_scale * centered_tread_contacts
            + self.cfg.supported_center_approach_reward_scale
            * supported_center_approach
            + self.cfg.descending_center_approach_reward_scale
            * descending_center_approach
            + self.cfg.retained_ground_support_reward_scale
            * retained_ground_support
            + 0.20 * retained_support
            + self.cfg.tread_hold_reward_scale * tread_hold_fraction
            + self.cfg.tread_transfer_reward_scale * tread_transfer
            + self.cfg.supported_transfer_reward_scale * supported_transfer
            + self.cfg.support_rise_reward_scale * support_rise
            + self.cfg.narrow_transfer_reward_scale * narrow_transfer
            + self.cfg.first_step_completion_reward_scale * first_step_completion
            + self.cfg.tread_height_delta_scale * required_tread_binary * height_delta
            + self.cfg.supported_tread_height_delta_scale
            * supported_transfer_gate
            * height_delta
            + self.cfg.support_rise_height_delta_scale
            * support_rise_gate
            * height_delta
            - 0.08 * upright_error
            - 0.002 * action_rate
            - 0.0005 * effort
            - 0.002 * body_rate
            - 1.0 * base_contact.float()
        )
        reward = torch.where(self._failed, reward - 5.0, reward)
        reward = torch.where(
            self._success,
            reward + self.cfg.success_completion_reward_scale,
            reward,
        )

        self._previous_root_x.copy_(root_x)
        self._previous_root_z.copy_(root_z)
        self._previous_foot_tip_z.copy_(foot_pos[:, :, 2])
        self._episode_max_progress = torch.maximum(
            self._episode_max_progress, root_x - self._episode_start_root_x
        )
        self._episode_max_base_gain = torch.maximum(
            self._episode_max_base_gain,
            root_z - self._episode_start_root_z,
        )
        self._episode_max_supported_base_gain = torch.maximum(
            self._episode_max_supported_base_gain, supported_base_gain
        )
        self._episode_max_four_support_base_gain = torch.maximum(
            self._episode_max_four_support_base_gain, four_support_base_gain
        )
        self._episode_max_support_rise_base_gain = torch.maximum(
            self._episode_max_support_rise_base_gain, support_rise_base_gain
        )
        self._episode_max_foot_clearance = torch.maximum(
            self._episode_max_foot_clearance, max_clearance
        )
        self._episode_best_tread_contacts = torch.maximum(
            self._episode_best_tread_contacts, tread_contacts
        )
        self._episode_best_centered_tread_contacts = torch.maximum(
            self._episode_best_centered_tread_contacts, centered_tread_contacts
        )
        self._episode_best_supported_center_approach = torch.maximum(
            self._episode_best_supported_center_approach, supported_center_approach
        )
        self._episode_best_tread_potential = torch.maximum(
            self._episode_best_tread_potential, tread_potential
        )
        self._episode_best_narrow_tread_potential = torch.maximum(
            self._episode_best_narrow_tread_potential, narrow_tread_potential
        )
        self._episode_max_tread_hold_steps = torch.maximum(
            self._episode_max_tread_hold_steps, self._tread_hold_steps
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._steps_since_reset += 1
        root = self._robot.data.root_pos_w.torch
        local = root - self._terrain.env_origins
        upright_cosine = -self._robot.data.projected_gravity_b.torch[:, 2]
        foot_pos = self._foot_tip_positions()
        foot_x = foot_pos[:, :, 0] - self._terrain.env_origins[:, 0:1]
        foot_ground = self._terrain_height(foot_x)
        foot_clearance = (
            foot_pos[:, :, 2] - self._terrain.env_origins[:, 2:3] - foot_ground
        )
        foot_contact = self._foot_forces() > 1.0
        supported = foot_contact & (torch.abs(foot_clearance) < 0.03)
        tread_contacts = torch.sum(
            (supported & (foot_ground > 0.0)).float(), dim=1
        )
        support_count = torch.sum(supported.float(), dim=1)
        tread_index = torch.arange(
            1, self.cfg.reward_tread_count + 1, device=self.device, dtype=torch.float32
        )
        tread_center_x = self.cfg.stair_start_from_origin_m + (
            tread_index - 0.5
        ) * self.cfg.stair_tread_depth_m
        centered_contact_count = torch.sum(
            (
                supported
                & (foot_ground > 0.0)
                & (
                    torch.abs(foot_x.unsqueeze(-1) - tread_center_x.view(1, 1, -1))
                    .amin(dim=2)
                    <= self.cfg.centered_tread_half_width_m
                )
            ).float(),
            dim=1,
        )
        required_tread_contacts = (
            centered_contact_count
            if self.cfg.first_step_require_centered_contact
            else tread_contacts
        )
        stable_tread = (required_tread_contacts >= 1.0) & (
            support_count >= self.cfg.first_step_min_support_count
        )
        self._tread_hold_steps = torch.where(
            stable_tread, self._tread_hold_steps + 1, torch.zeros_like(self._tread_hold_steps)
        )
        stable_lift = (
            (foot_clearance.max(dim=1).values >= self.cfg.foot_lift_height_m)
            & (support_count >= 3.0)
            & (upright_cosine >= 0.70)
        )
        self._lift_hold_steps = torch.where(
            stable_lift, self._lift_hold_steps + 1, torch.zeros_like(self._lift_hold_steps)
        )
        self._episode_max_lift_hold_steps = torch.maximum(
            self._episode_max_lift_hold_steps, self._lift_hold_steps
        )
        stable_support_rise = (
            (support_count >= self.cfg.support_rise_min_support_count)
            & (
                root[:, 2] - self._episode_start_root_z
                >= self.cfg.support_rise_min_base_gain_m
            )
            & (upright_cosine >= 0.70)
        )
        self._support_rise_hold_steps = torch.where(
            stable_support_rise,
            self._support_rise_hold_steps + 1,
            torch.zeros_like(self._support_rise_hold_steps),
        )
        self._episode_max_support_rise_hold_steps = torch.maximum(
            self._episode_max_support_rise_hold_steps,
            self._support_rise_hold_steps,
        )
        base_contact = (self._base_forces() > 5.0) & (
            self._steps_since_reset > self.cfg.base_contact_grace_steps
        )
        base_contact_failure = base_contact & (
            (root[:, 2] < self._episode_start_root_z - self.cfg.base_contact_failure_drop_m)
            | (upright_cosine < self.cfg.base_contact_failure_upright_cosine)
        )
        self._failed = (
            (local[:, 2] < self.cfg.minimum_base_height_m)
            | (upright_cosine < self.cfg.minimum_upright_cosine)
            | (torch.abs(local[:, 1]) > self.cfg.maximum_lateral_deviation_m)
            | (local[:, 0] < -0.40)
            | base_contact_failure
        )
        full_climb_success = (
            (local[:, 0] >= self.cfg.top_success_x_from_origin_m)
            & (local[:, 2] >= self.cfg.minimum_top_base_height_m)
            & (upright_cosine >= 0.75)
        )
        base_gain_ok = (
            root[:, 2] - self._episode_start_root_z >= self.cfg.first_step_min_base_gain_m
            if self.cfg.first_step_require_base_gain
            else torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        )
        first_step_success = (
            (self._tread_hold_steps >= self.cfg.first_step_hold_steps)
            & base_gain_ok
            & (upright_cosine >= 0.70)
        )
        foot_lift_success = self._lift_hold_steps >= self.cfg.foot_lift_hold_steps
        support_rise_success = (
            self._support_rise_hold_steps >= self.cfg.support_rise_hold_steps
        ) & ~self._failed
        if self.cfg.support_rise_curriculum:
            self._success = support_rise_success
        elif self.cfg.foot_lift_curriculum:
            self._success = foot_lift_success
        elif self.cfg.first_step_curriculum:
            self._success = first_step_success
        else:
            self._success = full_climb_success
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._failed | self._success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if len(env_ids) > 0:
            completed = self._steps_since_reset[env_ids] > 0
            completed_count = completed.sum()
            successful_count = (self._success[env_ids] & completed).sum()
            self._completed_episode_count += completed_count
            self._successful_episode_count += successful_count
            episode_supported_gain = self._episode_max_supported_base_gain[env_ids]
            supported_gain_reached = completed & (
                episode_supported_gain
                >= max(self.cfg.first_step_min_base_gain_m, 1.0e-6)
            )
            self._supported_base_gain_episode_count += supported_gain_reached.sum()
            self._supported_base_gain_sum_m += torch.sum(
                episode_supported_gain * completed.float()
            )
            if torch.any(completed):
                self._supported_base_gain_max_m = torch.maximum(
                    self._supported_base_gain_max_m,
                    episode_supported_gain[completed].max(),
                )
            episode_four_support_gain = self._episode_max_four_support_base_gain[env_ids]
            four_support_gain_reached = completed & (
                episode_four_support_gain >= self.cfg.support_rise_min_base_gain_m
            )
            self._four_support_gain_episode_count += four_support_gain_reached.sum()
            self._four_support_gain_sum_m += torch.sum(
                episode_four_support_gain * completed.float()
            )
            if torch.any(completed):
                self._four_support_gain_max_m = torch.maximum(
                    self._four_support_gain_max_m,
                    episode_four_support_gain[completed].max(),
                )
            episode_support_rise_gain = self._episode_max_support_rise_base_gain[env_ids]
            support_rise_gain_reached = completed & (
                episode_support_rise_gain >= self.cfg.support_rise_min_base_gain_m
            )
            self._support_rise_gain_episode_count += support_rise_gain_reached.sum()
            self._support_rise_gain_sum_m += torch.sum(
                episode_support_rise_gain * completed.float()
            )
            if torch.any(completed):
                self._support_rise_gain_max_m = torch.maximum(
                    self._support_rise_gain_max_m,
                    episode_support_rise_gain[completed].max(),
                )
            fold_min = self.cfg.reset_fold_fraction_min
            fold_max = self.cfg.reset_fold_fraction_max
            if fold_min is not None and fold_max is not None:
                reset_midpoint = 0.5 * (fold_min + fold_max)
                hard_reset = self._reset_fold_fraction[env_ids] >= reset_midpoint
                easy_completed = completed & ~hard_reset
                hard_completed = completed & hard_reset
                self._easy_reset_episode_count += easy_completed.sum()
                self._easy_reset_success_count += (
                    self._success[env_ids] & easy_completed
                ).sum()
                self._hard_reset_episode_count += hard_completed.sum()
                self._hard_reset_success_count += (
                    self._success[env_ids] & hard_completed
                ).sum()
            cumulative_success_rate = self._successful_episode_count.float() / torch.clamp(
                self._completed_episode_count, min=1
            ).float()
            self.extras.setdefault("log", {})["Metrics/max_progress_m"] = (
                self._episode_max_progress[env_ids].mean().item()
            )
            self.extras.setdefault("log", {})["Metrics/max_base_gain_m"] = (
                self._episode_max_base_gain[env_ids].mean().item()
            )
            self.extras.setdefault("log", {})["Metrics/max_supported_base_gain_m"] = (
                self._episode_max_supported_base_gain[env_ids].mean().item()
            )
            self.extras.setdefault("log", {})["Metrics/max_four_support_base_gain_m"] = (
                self._episode_max_four_support_base_gain[env_ids].mean().item()
            )
            self.extras.setdefault("log", {})["Metrics/max_support_rise_base_gain_m"] = (
                self._episode_max_support_rise_base_gain[env_ids].mean().item()
            )
            self.extras.setdefault("log", {})["Metrics/max_foot_clearance_m"] = (
                self._episode_max_foot_clearance[env_ids].mean().item()
            )
            self.extras.setdefault("log", {})["Metrics/best_tread_contacts"] = (
                self._episode_best_tread_contacts[env_ids].mean().item()
            )
            self.extras.setdefault("log", {})[
                "Metrics/best_centered_tread_contacts"
            ] = self._episode_best_centered_tread_contacts[env_ids].mean().item()
            self.extras.setdefault("log", {})[
                "Metrics/best_supported_center_approach"
            ] = self._episode_best_supported_center_approach[env_ids].mean().item()
            self.extras.setdefault("log", {})["Metrics/best_tread_potential"] = (
                self._episode_best_tread_potential[env_ids].mean().item()
            )
            self.extras.setdefault("log", {})[
                "Metrics/best_narrow_tread_potential"
            ] = self._episode_best_narrow_tread_potential[env_ids].mean().item()
            self.extras.setdefault("log", {})["Metrics/max_tread_hold_s"] = (
                self._episode_max_tread_hold_steps[env_ids].float().mean().item()
                * self.step_dt
            )
            self.extras.setdefault("log", {})["Metrics/max_lift_hold_s"] = (
                self._episode_max_lift_hold_steps[env_ids].float().mean().item()
                * self.step_dt
            )
            self.extras.setdefault("log", {})["Metrics/max_support_rise_hold_s"] = (
                self._episode_max_support_rise_hold_steps[env_ids].float().mean().item()
                * self.step_dt
            )
            self.extras.setdefault("log", {})["Metrics/reset_success_rate"] = (
                successful_count.float() / torch.clamp(completed_count, min=1).float()
            ).item()
            self.extras.setdefault("log", {})["Metrics/reset_episode_count"] = (
                completed_count.item()
            )
            self.extras.setdefault("log", {})["Metrics/success_rate"] = (
                cumulative_success_rate.item()
            )
            self.extras.setdefault("log", {})["Metrics/successful_episodes"] = (
                self._successful_episode_count.item()
            )
            self.extras.setdefault("log", {})["Metrics/completed_episodes"] = (
                self._completed_episode_count.item()
            )

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        joint_pos = self._robot.data.default_joint_pos.torch[env_ids].clone()
        fold_min = self.cfg.reset_fold_fraction_min
        fold_max = self.cfg.reset_fold_fraction_max
        reset_alpha = None
        if fold_min is not None and fold_max is not None:
            reset_alpha = torch.rand(len(env_ids), device=self.device).pow(
                self.cfg.reset_alpha_power
            )
            reset_fraction = fold_min + reset_alpha * (fold_max - fold_min)
            reset_hip = NORMAL_HIP_RAD + reset_fraction * (
                LOW_FOLD_HIP_RAD - NORMAL_HIP_RAD
            )
            reset_knee = NORMAL_KNEE_RAD + reset_fraction * (
                LOW_FOLD_KNEE_RAD - NORMAL_KNEE_RAD
            )
            for joint_index, joint_name in enumerate(self._robot.joint_names):
                if joint_name.endswith("hip_abduction"):
                    joint_pos[:, joint_index] = 0.0
                elif joint_name.startswith("front_") and joint_name.endswith("hip_flexion"):
                    joint_pos[:, joint_index] = -reset_hip
                elif joint_name.startswith("rear_") and joint_name.endswith("hip_flexion"):
                    joint_pos[:, joint_index] = reset_hip
                elif joint_name.startswith("front_") and joint_name.endswith("knee"):
                    joint_pos[:, joint_index] = reset_knee
                elif joint_name.startswith("rear_") and joint_name.endswith("knee"):
                    joint_pos[:, joint_index] = -reset_knee
            self._reset_fold_fraction[env_ids] = reset_fraction
        joint_pos += 0.01 * (2.0 * torch.rand_like(joint_pos) - 1.0)
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel.torch[env_ids])
        root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        height_min = self.cfg.reset_base_height_min_m
        height_max = self.cfg.reset_base_height_max_m
        if reset_alpha is not None and height_min is not None and height_max is not None:
            root_pose[:, 2] = (
                self._terrain.env_origins[env_ids, 2]
                + height_max
                + reset_alpha * (height_min - height_max)
            )
        root_pose[:, 0] += self.cfg.reset_forward_offset_m
        root_pose[:, 0] += self.cfg.reset_forward_jitter_m * (
            2.0 * torch.rand(len(env_ids), device=self.device) - 1.0
        )
        root_pose[:, 1] += 0.01 * (2.0 * torch.rand(len(env_ids), device=self.device) - 1.0)
        root_vel = torch.zeros_like(self._robot.data.default_root_vel.torch[env_ids])

        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._depth_pending[env_ids] = 1.0
        self._depth_observation[env_ids] = 1.0
        self._previous_root_x[env_ids] = root_pose[:, 0]
        self._previous_root_z[env_ids] = root_pose[:, 2]
        self._previous_foot_tip_z[env_ids] = 0.0
        self._episode_start_root_x[env_ids] = root_pose[:, 0]
        self._episode_start_root_z[env_ids] = root_pose[:, 2]
        self._episode_max_progress[env_ids] = 0.0
        self._episode_max_base_gain[env_ids] = 0.0
        self._episode_max_supported_base_gain[env_ids] = 0.0
        self._episode_max_four_support_base_gain[env_ids] = 0.0
        self._episode_max_support_rise_base_gain[env_ids] = 0.0
        self._episode_max_foot_clearance[env_ids] = 0.0
        self._episode_best_tread_contacts[env_ids] = 0.0
        self._episode_best_centered_tread_contacts[env_ids] = 0.0
        self._episode_best_supported_center_approach[env_ids] = 0.0
        self._episode_best_tread_potential[env_ids] = 0.0
        self._episode_best_narrow_tread_potential[env_ids] = 0.0
        self._tread_hold_steps[env_ids] = 0
        self._episode_max_tread_hold_steps[env_ids] = 0
        self._lift_hold_steps[env_ids] = 0
        self._episode_max_lift_hold_steps[env_ids] = 0
        self._support_rise_hold_steps[env_ids] = 0
        self._episode_max_support_rise_hold_steps[env_ids] = 0
        self._steps_since_reset[env_ids] = 0
        self._success[env_ids] = False
        self._failed[env_ids] = False
