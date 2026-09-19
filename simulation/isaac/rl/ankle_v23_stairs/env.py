"""Recurrent residual PPO using V2.3 mechanics and camera-only terrain input."""
import json
import math
import torch
from pxr import Gf, UsdGeom, UsdPhysics
from ankle_v2.env import AnkleEnv, rotate
from ankle_v21.env import PlantedEnv
from .depth import D405Depth
from .env_cfg import StairsCfg


class StairsEnv(AnkleEnv):
    cfg: StairsCfg

    def _setup_scene(self):
        # Ground/robot cloning uses the existing V2.3 physics implementation.
        super()._setup_scene()
        cfg = self.cfg
        origins = self.scene.env_origins.detach().cpu().tolist()
        for env_id, origin in enumerate(origins):
            for index, (low, high) in enumerate(self._box_coordinates()):
                prim = UsdGeom.Cube.Define(self.scene.stage, f"/World/Stairs/env_{env_id}/step_{index}")
                prim.CreateSizeAttr(1.0)
                prim.AddTranslateOp().Set(Gf.Vec3d(*[origin[j] + (low[j] + high[j]) / 2 for j in range(3)]))
                prim.AddScaleOp().Set(Gf.Vec3d(*[high[j] - low[j] for j in range(3)]))
                prim.CreateDisplayColorAttr().Set([Gf.Vec3f(.34, .38, .43)])
                UsdPhysics.CollisionAPI.Apply(prim.GetPrim())

    def _box_coordinates(self):
        cfg = self.cfg
        result = []
        for index in range(cfg.stair_count):
            start = cfg.stair_start_m + index * cfg.stair_tread_m
            end = start + (cfg.platform_depth_m if index == cfg.stair_count - 1 else cfg.stair_tread_m)
            result.append(((start, -cfg.stair_width_m / 2, 0.),
                           (end, cfg.stair_width_m / 2, (index + 1) * cfg.stair_rise_m)))
        return result

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.filtered = torch.zeros_like(self.actions)
        self.residual_scale = torch.tensor([v for v in cfg.residual_scales for _ in range(4)], device=self.device)
        self.ref = self.neutral.clone()
        self.expected_stance = torch.ones((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self.ref_clearance = torch.zeros((self.num_envs, 4), device=self.device)
        self.phase_bias = torch.tensor([0., .5, .75, .25], device=self.device)
        self.front_sign = torch.tensor([1., -1., 1., -1.], device=self.device)
        self.corners = torch.tensor([[cfg.contact_x_m, y, z] for y in (-.047, .047) for z in (-.027, .027)], device=self.device)
        self.history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reached = torch.zeros((self.num_envs, 4), device=self.device)
        self.swing_peak = torch.zeros_like(self.reached)
        self.best_front = torch.zeros(self.num_envs, device=self.device)
        self.best_rear = torch.zeros_like(self.best_front)
        self.hold = torch.zeros_like(self.best_front)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.all_first_seen = torch.zeros_like(self.success)
        self.landing_qualified = torch.zeros_like(self.reached, dtype=torch.bool)
        self.depth = D405Depth(cfg, self.num_envs, self.device,
                              [(torch.tensor(a, device=self.device), torch.tensor(b, device=self.device))
                               for a, b in self._box_coordinates()])
        self.totals = {"episodes": 0, "successes": 0, "falls": 0,
                       "front_first": 0, "rear_first": 0, "all_first": 0}
        self.recent = []
        self.summary_path = self.metrics_file.with_suffix(".episodes.json") if self.metrics_file else None
        for name in ("slip", "depth_valid", "ankle_motion", "swing_clear"):
            self.sums[name] = torch.zeros(self.num_envs, device=self.device)
        print("STAIRS_D405_CONTRACT=" + json.dumps({"model": "V2.3_R3_solid_sole_retained_leg",
            "rise_m": cfg.stair_rise_m, "tread_m": cfg.stair_tread_m, "steps": cfg.stair_count,
            "actions": self.action_order, "actor_observations": cfg.observation_space,
            "critic_observations": cfg.state_space, "camera": "sparse_pinhole_D405_proxy",
            "camera_position_m": cfg.camera_position_m, "camera_pitch_deg": cfg.camera_pitch_deg,
            "camera_range_m": [cfg.camera_min_m, cfg.camera_max_m], "camera_hz": cfg.camera_hz,
            "camera_mass_included": False, "robot_mass_kg": cfg.nominal_mass_kg,
            "training_only": True}), flush=True)

    def _set_commands(self):
        age = self.episode_length_buf.float() * self.step_dt
        self.commands[:, 0] = self.cruise * ((age - .8) / 1.5).clamp(0, 1)

    def _reference(self):
        age = self.episode_length_buf.float() * self.step_dt
        phase = (age[:, None] * self.cfg.frequency_hz + self.phase_offset[:, None] + self.phase_bias) % 1
        stance = phase < self.cfg.duty_factor
        u = ((phase - self.cfg.duty_factor) / (1 - self.cfg.duty_factor)).clamp(0, 1)
        smooth = u ** 3 * (10 + u * (-15 + 6 * u))
        sweep = torch.where(stance, .5 - phase / self.cfg.duty_factor, -.5 + smooth)
        stride = self.commands[:, 0] * self.cfg.duty_factor / self.cfg.frequency_hz
        amplitude = (self.commands[:, 0] / .025).clamp(0, 1)
        lift = self.cfg.lift_m * torch.sin(torch.pi * u).square() * (~stance) * amplitude[:, None]
        forward = self.front_sign * self.cfg.stance_forward_m + sweep * stride[:, None]
        down = self.cfg.stance_down_m - lift
        length = .159896689
        cosine = ((forward.square() + down.square() - 2 * length ** 2) / (2 * length ** 2)).clamp(-.999, .999)
        knee = -self.front_sign * torch.acos(cosine)
        hip = torch.atan2(forward, down) - torch.atan2(torch.sin(knee), 1 + torch.cos(knee))
        ref = self.neutral.clone()
        ref[:, 4:8], ref[:, 8:12] = hip, knee
        quat = self.robot.data.root_quat_w.torch
        pitch = torch.asin((2 * (quat[:, 3] * quat[:, 1] - quat[:, 2] * quat[:, 0])).clamp(-1, 1))
        joints = self.robot.data.joint_pos.torch[:, self.ids]
        ref[:, 12:16] = pitch[:, None] - joints[:, 4:8] - joints[:, 8:12]
        self.expected_stance = stance | (self.commands[:, :1] < .005)
        self.ref_clearance, self.ref = lift, ref
        return ref

    _pre_physics_step = PlantedEnv._pre_physics_step

    def _terrain_height(self, points):
        cfg = self.cfg
        height = torch.zeros_like(points[..., 0])
        for low, high in self._box_coordinates():
            inside = ((points[..., 0] >= low[0]) & (points[..., 0] <= high[0]) &
                      (points[..., 1] >= low[1]) & (points[..., 1] <= high[1]))
            height = torch.maximum(height, inside * high[2])
        return height

    def _geometry(self):
        pos = self.robot.data.body_link_pos_w.torch[:, self.feet]
        quat = self.robot.data.body_link_quat_w.torch[:, self.feet]
        corners = pos[:, :, None] + rotate(quat[:, :, None], self.corners)
        local = corners - self.scene.env_origins[:, None, None]
        surface = self._terrain_height(local)
        gap = local[..., 2] - surface
        # Upward load + every traction corner close to one horizontal tread.
        forces = self.contacts.data.net_forces_w.torch[:, self.foot_sensors]
        contact = forces[..., 2] > 1.0
        planted = contact & (gap.abs().amax(2) < .008) & ((surface.amax(2) - surface.amin(2)) < .003)
        levels = (surface.amin(2) / self.cfg.stair_rise_m).round()
        return local, gap, surface, contact, planted, levels

    def _get_observations(self):
        self._set_commands()
        phase = 2 * torch.pi * (self.episode_length_buf.float() * self.step_dt * self.cfg.frequency_hz + self.phase_offset)
        joints = self.robot.data.joint_pos.torch[:, self.ids]
        velocity = self.robot.data.joint_vel.torch[:, self.ids]
        pos = self.robot.data.root_pos_w.torch - self.scene.env_origins
        depth = self.depth.update(self.common_step_counter, pos, self.robot.data.root_quat_w.torch, self.step_dt)
        policy = torch.cat((self.commands * 10, torch.stack((phase.sin(), phase.cos()), 1),
            self.imu.data.ang_vel_b.torch, self.robot.data.projected_gravity_b.torch,
            self.imu.data.lin_acc_b.torch / 9.81, joints - self.neutral, velocity * .25,
            ((self.targets - self.neutral) / self.scales).clamp(-1, 1), self.filtered,
            self.expected_stance.float(), depth), 1)
        corners, gap, surface, contact, planted, levels = self._geometry()
        critic = torch.cat((policy, self.robot.data.root_lin_vel_b.torch,
            pos[:, 2:3] - self._terrain_height(pos)[:, None], contact.float(),
            surface.mean(2), self.reached / self.cfg.stair_count), 1)
        return {"policy": policy, "critic": critic}

    def _get_dones(self):
        pos = self.robot.data.root_pos_w.torch - self.scene.env_origins
        quat = self.robot.data.root_quat_w.torch
        finite = torch.isfinite(self.robot.data.joint_pos.torch).all(1) & torch.isfinite(pos).all(1) & torch.isfinite(quat).all(1)
        force = torch.linalg.vector_norm(self.contacts.data.net_forces_w.torch[:, self.base_sensor], dim=-1).amax(1)
        self.fallen = ((pos[:, 2] - self._terrain_height(pos) < .20) |
                       (self.robot.data.projected_gravity_b.torch[:, 2] > -.65) |
                       (force > 5) | (pos[:, 1].abs() > self.cfg.stair_width_m / 2) | ~finite)
        _, _, _, contact, planted, levels = self._geometry()
        top = (planted & (levels >= self.cfg.stair_count)).all(1)
        stable = top & (~self.fallen) & (self.robot.data.root_lin_vel_b.torch.norm(dim=1) < .08)
        self.hold = torch.where(stable, self.hold + self.step_dt, 0)
        self.success = self.hold >= self.cfg.success_hold_s
        return self.fallen | self.success, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        pos = self.robot.data.root_pos_w.torch
        vel = self.robot.data.root_lin_vel_b.torch
        joints = self.robot.data.joint_pos.torch[:, self.ids]
        qd = self.robot.data.joint_vel.torch[:, self.ids]
        local, gap, surface, contact, planted, levels = self._geometry()
        sole = local.mean(2)
        low = gap.amin(2)
        slip = ((sole - self.previous_sole)[..., :2] / self.step_dt).norm(dim=2) * self.history_valid[:, None]
        self.air_time += (~contact) * self.step_dt
        self.swing_peak = torch.maximum(self.swing_peak, low * (~contact))
        touchdown = contact & ~self.previous_contact & self.history_valid[:, None]
        lifted = (self.air_time > .08) & (self.swing_peak > .008)
        self.landing_qualified |= lifted & touchdown
        new_level = (levels - self.reached).clamp(min=0) * planted * self.landing_qualified
        self.reached = torch.maximum(self.reached, levels * planted)
        self.landing_qualified[planted | (~contact)] = False
        current_levels = levels * planted
        self.best_front = torch.maximum(self.best_front, current_levels[:, [0, 2]].amin(1))
        self.best_rear = torch.maximum(self.best_rear, current_levels[:, [1, 3]].amin(1))
        self.air_time[contact] = 0
        self.swing_peak[contact] = 0
        progress = (pos[:, 0] - self.previous_position[:, 0]) * self.history_valid
        tracking = torch.exp(-((vel[:, 0] - self.commands[:, 0]) / .025).square())
        swing = ~self.expected_stance
        clear = swing & (~contact) & (low > .015)
        stance_count = self.expected_stance.sum(1).clamp(min=1)
        match = (contact == self.expected_stance).float().mean(1)
        posture = self.robot.data.projected_gravity_b.torch[:, :2].square().sum(1)
        support_height = surface.mean((1, 2))
        base_height = pos[:, 2] - self.scene.env_origins[:, 2] - support_height
        power = (self.robot.data.applied_torque.torch[:, self.ids] * qd).clamp(min=0).sum(1)
        local_pos = pos - self.scene.env_origins
        q = self.robot.data.root_quat_w.torch
        x, y, z, w = q.unbind(1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        reward = (4 * tracking + match + 1.5 * (planted * self.expected_stance).sum(1) / stance_count
            + 2 * clear.float().sum(1) + .5
            - .8 * ((slip / .02).square().clamp(max=25) * contact).mean(1)
            - 1.5 * (contact & swing).float().sum(1)
            - 6 * posture - 2 * (local_pos[:, 1] / .12).square().clamp(max=9)
            - (yaw / .25).square().clamp(max=9)
            - .4 * ((base_height - self.cfg.target_base_height_m) / .05).square().clamp(max=9)
            - .02 * power - .04 * (self.actions - self.last_actions).square().mean(1)
            - .03 * self.filtered.square().mean(1) - 50 * self.fallen.float()) * self.step_dt
        reward += 25 * progress.clamp(-.005, .005) + 2 * new_level.sum(1) + 20 * self.success.float()
        # A first-tread measure needs simultaneous four-foot support; historic
        # front/rear maxima are deliberately reported as separate milestones.
        all_first = (planted & (levels >= 1)).all(1)
        self.all_first_seen |= all_first
        self.sums["move_time"] += self.step_dt
        self.sums["move_distance"] += progress
        self.sums["command_integral"] += self.commands[:, 0] * self.step_dt
        self.sums["rate_limited"] += self.rate_limited.float().mean(1)
        self.sums["work"] += power * self.step_dt
        self.sums["slip"] += (slip * contact).sum(1) / contact.sum(1).clamp(min=1)
        self.sums["depth_valid"] += self.depth.frame[:, self.cfg.camera_rows * self.cfg.camera_cols:].mean(1)
        self.sums["ankle_motion"] += qd[:, 12:16].abs().mean(1) * self.step_dt
        self.sums["swing_clear"] += clear.float().mean(1)
        self.contact_counts += touchdown.float()
        self.previous_position.copy_(pos)
        self.previous_sole.copy_(sole)
        self.previous_contact.copy_(contact)
        self.history_valid[:] = True
        return reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if hasattr(self, "depth"):
            finished = env_ids[self.episode_length_buf[env_ids] > 0]
            if len(finished):
                age = self.episode_length_buf[finished].float().clamp(min=1)
                rows = torch.stack((self.success[finished], self.fallen[finished],
                    self.best_front[finished] >= 1, self.best_rear[finished] >= 1,
                    self.all_first_seen[finished],
                    self.sums["move_distance"][finished] / (age * self.step_dt),
                    self.sums["depth_valid"][finished] / age,
                    self.sums["slip"][finished] / age,
                    self.sums["ankle_motion"][finished] / (age * self.step_dt),
                    self.sums["rate_limited"][finished] / age), 1).float().cpu().tolist()
                self.recent = (self.recent + rows)[-512:]
                self.totals["episodes"] += len(rows)
                for column, key in enumerate(("successes", "falls", "front_first", "rear_first", "all_first")):
                    self.totals[key] += sum(int(row[column]) for row in rows)
                recent = torch.tensor(self.recent).mean(0).tolist()
                record = {"step": self.common_step_counter, "rise_m": self.cfg.stair_rise_m,
                    "totals": self.totals, "recent_episodes": len(self.recent),
                    "recent_success_fraction": recent[0], "recent_fall_fraction": recent[1],
                    "recent_front_first_fraction": recent[2], "recent_rear_first_fraction": recent[3],
                    "recent_all_first_fraction": recent[4], "recent_speed_m_s": recent[5],
                    "recent_depth_valid_fraction": recent[6], "recent_contact_slip_m_s": recent[7],
                    "recent_ankle_abs_velocity_rad_s": recent[8], "recent_target_rate_limited": recent[9],
                    "evidence": "stochastic_training_episodes_not_held_out_evaluation"}
                if self.summary_path:
                    self.summary_path.write_text(json.dumps(record, indent=2) + "\n")
                if self.metrics_file:
                    with self.metrics_file.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                stair_log = {"Stairs/" + key: value for key, value in record.items() if key.startswith("recent_")}
            else:
                stair_log = {}
        else:
            stair_log = {}
        super()._reset_idx(env_ids)
        if not hasattr(self, "depth"):
            return
        self.filtered[env_ids] = 0
        self.history_valid[env_ids] = False
        self.expected_stance[env_ids] = True
        self.ref[env_ids] = self.neutral[env_ids]
        self.reached[env_ids] = 0
        self.swing_peak[env_ids] = 0
        self.best_front[env_ids] = 0
        self.best_rear[env_ids] = 0
        self.hold[env_ids] = 0
        self.success[env_ids] = False
        self.landing_qualified[env_ids] = False
        if hasattr(self, "all_first_seen"):
            self.all_first_seen[env_ids] = False
        self.depth.reset(env_ids)
        # Every episode starts before the first riser, never already on a tread.
        count = len(env_ids)
        self.cruise[env_ids] = self.cfg.command_min + (self.cfg.command_max - self.cfg.command_min) * torch.rand(count, device=self.device)
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        pose[:, 0] += (torch.rand(count, device=self.device) * 2 - 1) * .025
        pose[:, 1] += (torch.rand(count, device=self.device) * 2 - 1) * .01
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.previous_position[env_ids] = pose[:, :3]
        self.start_position[env_ids] = pose[:, :3]
        self.extras.setdefault("log", {}).update(stair_log)
