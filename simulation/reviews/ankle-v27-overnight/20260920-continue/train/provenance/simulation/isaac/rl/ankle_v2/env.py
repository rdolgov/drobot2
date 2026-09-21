"""From-scratch PPO pilot for physically distinct rigid/locked/active feet.

This is an isolated simulator task, not a deployable 16-servo hardware policy.
Training telemetry is not held-out performance evaluation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu
from pxr import Gf, PhysxSchema, UsdPhysics, UsdShade

from .env_cfg import LEGS, ActiveEnvCfg


def rotate(q, v):
    """Isaac Lab current backend uses xyzw quaternions."""
    xyz, v = torch.broadcast_tensors(q[..., :3], v)
    cross = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., 3:4] * cross + torch.cross(xyz, cross, dim=-1)


class AnkleEnv(DirectRLEnv):
    cfg: ActiveEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        kinds = ("hip_abduction", "hip_flexion", "knee")
        if cfg.variant == "active":
            kinds += ("ankle_pitch",)
        self.action_order = [f"{leg}_{kind}" for kind in kinds for leg in LEGS]
        ids, names = self.robot.find_joints(self.action_order, preserve_order=True)
        if list(names) != self.action_order or self.robot.num_joints != len(names):
            raise RuntimeError(f"Ankle action mapping mismatch: {names}, {self.robot.joint_names}")
        self.ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        foot_names = [f"{leg}_foot_link" for leg in LEGS]
        foot_ids, found = self.robot.find_bodies(foot_names, preserve_order=True)
        if list(found) != foot_names:
            raise RuntimeError(f"Foot body mapping mismatch: {found}")
        self.feet = foot_ids
        self.foot_sensors, found = self.contacts.find_bodies(foot_names, preserve_order=True)
        if list(found) != foot_names:
            raise RuntimeError(f"Foot sensor mapping mismatch: {found}")
        self.base_sensor, _ = self.contacts.find_bodies("base_link")
        self.neutral = self.robot.data.default_joint_pos.torch[:, self.ids].clone()
        self.limits = self.robot.data.soft_joint_pos_limits.torch[:, self.ids].clone()
        self.scales = torch.tensor([cfg.action_scales_rad[i] for i in range(len(kinds)) for _ in LEGS], device=self.device)
        self.rates = torch.tensor([math.radians(cfg.target_rates_deg_s[i]) for i in range(len(kinds)) for _ in LEGS], device=self.device)
        self.targets = self.neutral.clone()
        self.actions = torch.zeros_like(self.targets)
        self.last_actions = torch.zeros_like(self.targets)
        self.target_delta = torch.zeros_like(self.targets)
        self.rate_limited = torch.zeros_like(self.targets, dtype=torch.bool)
        self.strength = torch.ones_like(self.targets)
        self.rate_strength = torch.ones_like(self.targets)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.cruise = torch.zeros(self.num_envs, device=self.device)
        self.phase_offset = torch.zeros(self.num_envs, device=self.device)
        self.air_time = torch.zeros((self.num_envs, 4), device=self.device)
        self.previous_contact = torch.zeros_like(self.air_time, dtype=torch.bool)
        self.previous_sole = torch.zeros((self.num_envs, 4, 3), device=self.device)
        self.previous_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.start_position = torch.zeros_like(self.previous_position)
        self.fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.sums = {name: torch.zeros(self.num_envs, device=self.device) for name in
                     ("move_time", "move_distance", "command_integral", "rate_limited", "work", "flight_time", "contact_quality")}
        self.contact_counts = torch.zeros((self.num_envs, 4), device=self.device)
        self.episode_count = 0
        self.last_log_step = -1
        self.metrics_file = Path(cfg.metrics_path) if cfg.metrics_path else None
        if self.metrics_file:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.corners = torch.tensor([[0.031, y, z] for y in (-0.05, 0.05) for z in (-0.03, 0.03)], device=self.device)
        print("ANKLE_TRAINING_CONTRACT=" + json.dumps({
            "variant": cfg.variant, "action_order": self.action_order,
            "native_order": self.robot.joint_names, "foot_bodies": foot_names,
            "actor_inputs": cfg.observation_space, "critic_inputs": cfg.state_space,
            "pilot_only": True, "nominal_mass_kg": 3.18 if cfg.variant == "baseline" else 3.5,
        }), flush=True)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        # The imported assets already own their feet and rear payload. No legacy
        # geometry insertion or mass normalization is called here.
        material = UsdShade.Material.Define(self.scene.stage, "/World/Materials/AnkleTread")
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr().Set(1.0)
        api.CreateDynamicFrictionAttr().Set(0.9)
        api.CreateRestitutionAttr().Set(0.0)
        physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx.CreateCompliantContactStiffnessAttr().Set(12000.0)
        physx.CreateCompliantContactDampingAttr().Set(120.0)
        for prim in self.scene.stage.Traverse():
            if "_foot_link/" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
        self.contacts = ContactSensor(self.cfg.contacts)
        self.imu = Imu(self.cfg.imu)
        self.scene.sensors["contacts"] = self.contacts
        self.scene.sensors["imu"] = self.imu
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        positions = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.clone_plan_from_env_0("/World/envs/env_0", "/World/envs/env_{}", self.scene.num_envs, self.device, positions)
        cloner.replicate(plan, stage=self.scene.stage)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=1500.0)
        light.func("/World/Light", light)

    def _set_commands(self):
        age = self.episode_length_buf.float() * self.step_dt
        # Stand initially, then ramp up; reserve the last 2 seconds for stopping.
        start = ((age - 0.5) / 0.75).clamp(0, 1)
        stop = ((self.cfg.episode_length_s - 2.0 - age) / 0.75).clamp(0, 1)
        self.commands[:, 0] = self.cruise * start * stop

    def _pre_physics_step(self, actions):
        self._set_commands()
        self.last_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1, 1))
        desired = self.neutral + self.actions * self.scales
        # Quiet stance is still learned; do not overwrite ankle actions at stop.
        desired = torch.maximum(torch.minimum(desired, self.limits[:, :, 1]), self.limits[:, :, 0])
        delta = desired - self.targets
        step_limit = torch.minimum(self.rates * self.rate_strength * self.step_dt,
                                   torch.full_like(delta, math.radians(2.0)))
        self.rate_limited = delta.abs() > step_limit
        self.target_delta = torch.maximum(torch.minimum(delta, step_limit), -step_limit)
        self.targets += self.target_delta

    def _apply_action(self):
        # Apply a conservative coupled torque-speed envelope at each physics tick.
        velocity = self.robot.data.joint_vel.torch[:, self.ids].abs()
        cap = torch.minimum(torch.full_like(velocity, self.cfg.effort_cap_nm),
                            self.cfg.stall_effort_nm * (1.0 - velocity / self.cfg.no_load_speed_rad_s).clamp(min=0.0))
        cap *= self.strength
        native_caps = torch.empty_like(cap)
        native_caps[:, self.ids] = cap
        actuator = self.robot.actuators["servos"]
        actuator.effort_limit.copy_(native_caps)
        actuator.effort_limit_sim.copy_(native_caps)
        self.robot.write_joint_effort_limit_to_sim_index(limits=native_caps)
        self.robot.set_joint_position_target_index(target=self.targets, joint_ids=self.ids)

    def _contact(self):
        forces = self.contacts.data.net_forces_w.torch[:, self.foot_sensors]
        return torch.linalg.vector_norm(forces, dim=-1) > 1.0

    def _foot_geometry(self):
        q = self.robot.data.body_link_quat_w.torch[:, self.feet]
        p = self.robot.data.body_link_pos_w.torch[:, self.feet]
        corners = p.unsqueeze(2) + rotate(q.unsqueeze(2), self.corners)
        sole = p + rotate(q, torch.tensor([0.0305, 0.0, 0.0], device=self.device))
        up = rotate(q, torch.tensor([-1.0, 0.0, 0.0], device=self.device))
        return sole, corners[..., 2].amin(dim=2), up[..., 2]

    def _get_observations(self):
        self._set_commands()
        phase = 2 * torch.pi * (self.episode_length_buf.float() * self.step_dt + self.phase_offset)
        clock = torch.stack((phase.sin(), phase.cos()), dim=1)
        q = self.robot.data.joint_pos.torch[:, self.ids]
        qd = self.robot.data.joint_vel.torch[:, self.ids]
        policy = torch.cat((self.commands * 10.0, clock, self.imu.data.ang_vel_b.torch,
                            self.robot.data.projected_gravity_b.torch,
                            self.imu.data.lin_acc_b.torch / 9.81,
                            q - self.neutral, qd * 0.25,
                            ((self.targets - self.neutral) / self.scales).clamp(-1, 1)), dim=1)
        critic = torch.cat((policy, self.robot.data.root_lin_vel_b.torch,
                            self.robot.data.root_pos_w.torch[:, 2:3], self._contact().float()), dim=1)
        return {"policy": policy, "critic": critic}

    def _get_dones(self):
        q = self.robot.data.root_quat_w.torch
        finite = torch.isfinite(self.robot.data.joint_pos.torch).all(dim=1) & torch.isfinite(q).all(dim=1)
        base_force = torch.linalg.vector_norm(self.contacts.data.net_forces_w.torch[:, self.base_sensor], dim=-1).amax(dim=1)
        self.fallen = ((self.robot.data.root_pos_w.torch[:, 2] < 0.22)
                       | (self.robot.data.projected_gravity_b.torch[:, 2] > -0.65)
                       | (base_force > 5.0) | ~finite)
        return self.fallen, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        velocity = self.robot.data.root_lin_vel_b.torch
        position = self.robot.data.root_pos_w.torch
        q = self.robot.data.root_quat_w.torch
        x, y, z, w = q.unbind(dim=1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        moving = self.commands[:, 0] > 0.02
        stopped = ~moving
        contact = self._contact()
        sole, clearance, normal_z = self._foot_geometry()
        sole_velocity = (sole - self.previous_sole) / self.step_dt
        valid = self.episode_length_buf > 1
        slip = (sole_velocity[..., :2].square().sum(dim=-1) * contact).mean(dim=1) * valid
        self.air_time += (~contact).float() * self.step_dt
        touchdown = contact & ~self.previous_contact
        air_reward = ((self.air_time - 0.08).clamp(0, 0.30) * touchdown).sum(dim=1)
        self.air_time[contact] = 0.0
        self.contact_counts += touchdown.float()
        effort = self.robot.data.applied_torque.torch[:, self.ids]
        qd = self.robot.data.joint_vel.torch[:, self.ids]
        # PhysX implicit drive torque is approximate telemetry, hence work is a proxy.
        power = (effort * qd).clamp(min=0).sum(dim=1)
        tracking = torch.exp(-((velocity[:, 0] - self.commands[:, 0]) / 0.06).square())
        posture = self.robot.data.projected_gravity_b.torch[:, :2].square().sum(dim=1)
        lateral = position[:, 1] - self.start_position[:, 1]
        action_rate = (self.actions - self.last_actions).square().mean(dim=1)
        flat_stop = (1.0 - normal_z.clamp(-1, 1)).mean(dim=1) * stopped
        height_error = ((position[:, 2] - 0.355) / 0.05).square()
        abduction = (self.robot.data.joint_pos.torch[:, self.ids[:4]]).square().mean(dim=1)
        release = ((~contact).float() * (clearance / 0.02).clamp(0, 1)).mean(dim=1)
        reward = (6.0 * tracking + 1.0 * (velocity[:, 0] / 0.14).clamp(-1, 1) * moving
                  + 0.5 * air_reward * moving + 0.3 * release * moving
                  - 3.0 * velocity[:, 1].square() - 1.0 * yaw.square()
                  - 4.0 * lateral.square() - 2.0 * posture - 0.2 * height_error
                  - 2.0 * slip.clamp(max=4.0) - 0.05 * power - 0.1 * action_rate
                  - 1.0 * abduction - 1.0 * flat_stop
                  - 4.0 * velocity[:, :2].square().sum(dim=1) * stopped
                  - 100.0 * self.fallen.float()) * self.step_dt
        delta = position - self.previous_position
        self.sums["move_time"] += moving.float() * self.step_dt
        self.sums["move_distance"] += delta[:, 0] * moving * valid
        self.sums["command_integral"] += self.commands[:, 0] * moving * self.step_dt
        self.sums["rate_limited"] += self.rate_limited.float().mean(dim=1)
        self.sums["work"] += power * self.step_dt
        self.sums["flight_time"] += (~contact.any(dim=1)).float() * self.step_dt
        self.sums["contact_quality"] += contact.float().mean(dim=1)
        self.previous_position.copy_(position)
        self.previous_sole.copy_(sole)
        self.previous_contact.copy_(contact)
        return reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if hasattr(self, "sums") and len(env_ids):
            age = self.episode_length_buf[env_ids].float().clamp(min=1)
            move_time = self.sums["move_time"][env_ids].clamp(min=self.step_dt)
            log = {
                "Metrics/train_forward_speed_m_s": (self.sums["move_distance"][env_ids] / move_time).mean().item(),
                "Metrics/train_command_m_s": (self.sums["command_integral"][env_ids] / move_time).mean().item(),
                "Metrics/train_fall_fraction": self.fallen[env_ids].float().mean().item(),
                "Metrics/train_episode_seconds": (age * self.step_dt).mean().item(),
                "Metrics/train_target_rate_limited": (self.sums["rate_limited"][env_ids] / age).mean().item(),
                "Metrics/train_positive_work_proxy_j": self.sums["work"][env_ids].mean().item(),
                "Metrics/train_flight_fraction": (self.sums["flight_time"][env_ids] / (age * self.step_dt)).mean().item(),
                "Metrics/train_min_leg_touchdowns": self.contact_counts[env_ids].amin(dim=1).mean().item(),
            }
            self.extras["log"] = log
            self.episode_count += len(env_ids)
            if self.metrics_file and self.common_step_counter - self.last_log_step >= 640:
                with self.metrics_file.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": self.common_step_counter, "variant": self.cfg.variant,
                                             "episode_count": self.episode_count, **log}) + "\n")
                self.last_log_step = self.common_step_counter
        super()._reset_idx(env_ids)
        if not hasattr(self, "neutral"):
            return
        count = len(env_ids)
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_vel = torch.zeros_like(self.robot.data.default_root_vel.torch[env_ids])
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
        self.targets[env_ids] = self.neutral[env_ids]
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.target_delta[env_ids] = 0.0
        self.rate_limited[env_ids] = False
        # Draw a common 16-column domain for every morphology to preserve pairing.
        strength = self.cfg.pilot_strength_min + (1 - self.cfg.pilot_strength_min) * torch.rand((count, 16), device=self.device)
        rate = self.cfg.pilot_strength_min + (1 - self.cfg.pilot_strength_min) * torch.rand((count, 16), device=self.device)
        self.strength[env_ids] = strength[:, :len(self.action_order)]
        self.rate_strength[env_ids] = rate[:, :len(self.action_order)]
        self.cruise[env_ids] = self.cfg.command_min + (self.cfg.command_max - self.cfg.command_min) * torch.rand(count, device=self.device)
        stand = torch.rand(count, device=self.device) < 0.1
        self.cruise[env_ids[stand]] = 0.0
        self.phase_offset[env_ids] = torch.rand(count, device=self.device)
        self.previous_position[env_ids] = root_pose[:, :3]
        self.start_position[env_ids] = root_pose[:, :3]
        self.previous_sole[env_ids] = 0.0
        self.previous_contact[env_ids] = False
        self.air_time[env_ids] = 0.0
        self.contact_counts[env_ids] = 0.0
        for values in self.sums.values():
            values[env_ids] = 0.0
