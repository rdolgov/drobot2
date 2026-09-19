"""Free joint exploration, turning and staged foot-placement reward combinations.

Every arm uses the same mechanics, resets, episode budget, observations and
actuator controller. Absolute terrain and contact geometry are reward/critic
information only. Phase commands are clock-driven, never oracle-gated.
"""
import json
import math
import torch
from ankle_v2.env import AnkleEnv
from ankle_v23_stairs.env import StairsEnv
from .strategies import STRATEGIES, METRIC_COLUMNS


class StrategyEnv(StairsEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.weights = STRATEGIES[cfg.strategy]
        super().__init__(cfg, render_mode, **kwargs)
        self.order = torch.tensor([0, 2, 3, 1], device=self.device)
        self.triangles = torch.tensor([[1, 3, 2], [0, 3, 2], [0, 1, 3], [0, 1, 2]], device=self.device)
        self.index = torch.arange(self.num_envs, device=self.device)
        self.qualified_air = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self.lift_seen = torch.zeros_like(self.qualified_air)
        self.placed = torch.zeros_like(self.qualified_air)
        self.placement_hold = torch.zeros((self.num_envs, 4), device=self.device)
        self.initial_sole = torch.zeros((self.num_envs, 4, 3), device=self.device)
        self.phase_origin = torch.zeros_like(self.initial_sole)
        self.prior_phase = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.episode_totals = {"episodes": 0, "full_climb": 0, "fall": 0, "any_placement": 0,
            "front_pair": 0, "rear_pair": 0, "all_first": 0, "any_supported_lift": 0}
        self.episode_placed_by_leg = [0, 0, 0, 0]
        self.episode_lift_by_leg = [0, 0, 0, 0]
        self.rows = []
        self.screen_log = {}
        self.acc = {key: torch.zeros(self.num_envs, device=self.device) for key in (
            "front_pair", "rear_pair", "all_first", "advance", "speed", "slip", "yaw",
            "sideways", "horizontal", "depth", "rate", "clearance", "margin")}
        self.counter_file = self.metrics_file.with_suffix(".screen.json")
        self.last_summary_step = -1000
        print("STRATEGY_CONTRACT=" + json.dumps({"strategy": cfg.strategy, "weights": self.weights,
            "actor_observations": cfg.observation_space, "critic_observations": cfg.state_space,
            "reference": "neutral_hip_knee_with_measured_joint_ankle_leveling",
            "sequence": ["front_left", "front_right", "rear_right", "rear_left"],
            "phase_source": "elapsed_time_not_terrain_or_contact",
            "placement_hold_s": cfg.placement_hold_s,
            "hardware_deployable": False}), flush=True)

    def _yaw(self):
        x, y, z, w = self.robot.data.root_quat_w.torch.unbind(1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _phase(self):
        age = self.episode_length_buf.float() * self.step_dt
        phase_id = (age / 8).long().clamp(0, 3)
        fraction = (age / 8 - phase_id).clamp(0, 1)
        return age, phase_id, fraction, self.order[phase_id]

    def _set_commands(self):
        age = self.episode_length_buf.float() * self.step_dt
        ramp = ((age - .8) / 1.5).clamp(0, 1) * ((32. - age) / 1.).clamp(0, 1)
        yaw = self._yaw()
        world_speed = self.cruise * ramp
        self.commands[:, 0] = world_speed * yaw.cos()
        self.commands[:, 1] = -world_speed * yaw.sin()
        yaw_target = math.radians(self.weights["yaw_deg"]) * ((age - 1) / 5).clamp(0, 1)
        self.commands[:, 2] = ((yaw_target - yaw) * .8).clamp(-.45, .45) if self.weights["yaw_weight"] else 0.

    def _reference(self):
        # The former forward-only IK crawl would oppose a sideways strategy.
        # All arms instead have identical neutral hip/knee targets and freedom
        # to lift or move any leg through learned residuals.
        ref = self.neutral.clone()
        q = self.robot.data.root_quat_w.torch
        pitch = torch.asin((2 * (q[:, 3] * q[:, 1] - q[:, 2] * q[:, 0])).clamp(-1, 1))
        actual = self.robot.data.joint_pos.torch[:, self.ids]
        ref[:, 12:16] = pitch[:, None] - actual[:, 4:8] - actual[:, 8:12]
        self.expected_stance[:] = True
        if self.weights["placement_weight"]:
            age, _, _, selected = self._phase()
            self.expected_stance[self.index, selected] = age >= 32.
        self.ref = ref
        return ref

    def _get_observations(self):
        obs = super()._get_observations()
        yaw = self._yaw()
        heading = torch.stack((yaw.sin(), yaw.cos()), 1)
        # Relative reset heading is an IMU/heading-estimator input, not terrain.
        obs["policy"] = torch.cat((obs["policy"], heading), 1)
        obs["critic"] = torch.cat((obs["critic"], heading), 1)
        return obs

    def _support_margin(self, sole, selected):
        mass = self.robot.data.body_mass.torch
        com = (self.robot.data.body_com_pos_w.torch * mass[:, :, None]).sum(1) / mass.sum(1)[:, None]
        com -= self.scene.env_origins
        indices = self.triangles[selected]
        triangle = sole[self.index[:, None], indices, :2]
        edge = torch.roll(triangle, -1, 1) - triangle
        offset = com[:, None, :2] - triangle
        cross = edge[..., 0] * offset[..., 1] - edge[..., 1] * offset[..., 0]
        a, b = triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
        orientation = torch.where(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] >= 0, 1., -1.)
        return (orientation[:, None] * cross / edge.norm(dim=2).clamp(min=.001)).amin(1)

    def _get_rewards(self):
        cfg = self.cfg
        age, phase_id, fraction, selected = self._phase()
        pos = self.robot.data.root_pos_w.torch
        local_pos = pos - self.scene.env_origins
        body_vel = self.robot.data.root_lin_vel_b.torch
        world_vel = self.robot.data.root_lin_vel_w.torch
        local, gap, surface, contact, planted, levels = self._geometry()
        sole = local.mean(2)
        low = gap.amin(2)
        fresh = ~self.history_valid
        self.initial_sole[fresh] = sole[fresh]
        change = (phase_id != self.prior_phase) | fresh
        self.phase_origin[change] = sole[change]
        self.prior_phase.copy_(phase_id)
        slip = ((sole - self.previous_sole)[..., :2] / self.step_dt).norm(dim=2) * self.history_valid[:, None]
        upright = (self.robot.data.projected_gravity_b.torch[:, 2] < -.90) & (~self.fallen)
        support_count = contact.sum(1)
        other_support = support_count[:, None] - contact.long()
        self.air_time += (~contact) * self.step_dt
        self.swing_peak = torch.maximum(self.swing_peak, low * (~contact))
        supported_lift = ((~contact) & (other_support >= 3) & (self.air_time > .08)
                          & (self.swing_peak > .012) & upright[:, None])
        self.qualified_air |= supported_lift
        self.lift_seen |= supported_lift
        # A lift followed by landing back on the floor cannot later qualify
        # a dragged foot as a new airborne tread placement.
        self.qualified_air &= ~(planted & (levels < 1))
        valid_landing = (planted & (levels >= 1) & self.qualified_air &
                         (other_support >= 2) & upright[:, None] & (slip < .015))
        self.placement_hold = torch.where(valid_landing, self.placement_hold + self.step_dt, 0.)
        landed = self.placement_hold >= cfg.placement_hold_s
        new_placement = landed & ~self.placed
        self.placed |= landed
        self.reached = torch.maximum(self.reached, levels * self.placed * planted)
        self.air_time[contact] = 0
        self.swing_peak[contact] = 0
        supported = upright & (support_count >= 2)
        delta = (pos[:, 0] - self.previous_position[:, 0]) * self.history_valid
        advance = ((pos[:, 0] - self.start_position[:, 0]) * supported).clamp(min=0)
        self.acc["advance"] = torch.maximum(self.acc["advance"], advance)
        on_tread = planted & self.placed & (levels >= 1)
        front = on_tread[:, [0, 2]].all(1) & upright
        rear = on_tread[:, [1, 3]].all(1) & upright
        all_first = on_tread.all(1) & upright
        for key, event in (("front_pair", front), ("rear_pair", rear), ("all_first", all_first)):
            self.acc[key] = torch.maximum(self.acc[key], event.float())
        qd = self.robot.data.joint_vel.torch[:, self.ids]
        power = (self.robot.data.applied_torque.torch[:, self.ids] * qd).clamp(min=0).sum(1)
        posture = self.robot.data.projected_gravity_b.torch[:, :2].square().sum(1)
        yaw = self._yaw()
        yaw_target = math.radians(self.weights["yaw_deg"]) * ((age - 1) / 5).clamp(0, 1)
        yaw_score = torch.exp(-((yaw - yaw_target) / .35).square()) * supported
        moving = age < 32
        forward_command = self.commands[:, :2].norm(dim=1)
        tracking = torch.exp(-((world_vel[:, 0] - forward_command) / .025).square()) * supported
        lift_score = (torch.exp(-((low - .045) / .025).square()) * supported_lift).sum(1)
        stop_cost = body_vel.norm(dim=1).square() / .02 ** 2 * (~moving)
        reward = (1.2 * tracking + .5 * lift_score + .25 * (support_count >= 3)
            - .5 * ((slip / .02).square().clamp(max=25) * contact).mean(1)
            - 6 * posture - 2 * ((local_pos[:, 1].abs() - .25).clamp(min=0) / .12).square().clamp(max=9)
            - .3 * ((pos[:, 2] - self.scene.env_origins[:, 2] - surface.mean((1, 2)) - cfg.target_base_height_m) / .06).square().clamp(max=9)
            - .015 * power - .03 * (self.actions - self.last_actions).square().mean(1)
            - .01 * self.filtered.square().mean(1) - .5 * stop_cost.clamp(max=9)
            - .6 * (moving & (world_vel[:, 0] < .003))
            + self.weights["yaw_weight"] * yaw_score * moving) * self.step_dt
        reward += 30 * delta.clamp(-.005, .005) * supported + 4 * new_placement.sum(1)
        reward += 25 * self.success.float() - 2 * self.fallen.float()

        # A lift / advance / lower waypoint is used only to shape the reward.
        # It never writes joint targets or teleports a foot/body.
        current = sole[self.index, selected]
        origin = self.phase_origin[self.index, selected]
        target = origin.clone()
        shift = ((fraction - .40) / .35).clamp(0, 1)
        target[:, 0] = origin[:, 0] + shift * (cfg.stair_start_m + .070 - origin[:, 0]).clamp(min=0)
        lift_ramp = ((fraction - .18) / .22).clamp(0, 1)
        lower = ((fraction - .75) / .18).clamp(0, 1)
        target[:, 2] = origin[:, 2] + lift_ramp * (.065 - origin[:, 2])
        target[:, 2] += lower * (cfg.stair_rise_m - .065)
        waypoint = torch.exp(-((current - target).square() / torch.tensor([.08**2, .04**2, .025**2], device=self.device)).sum(1))
        selected_others = other_support[self.index, selected]
        waypoint *= upright & (selected_others >= 3)
        selected_new = new_placement[self.index, selected]
        retained = (on_tread & self.expected_stance).float().sum(1)
        reward += self.weights["placement_weight"] * (
            waypoint * moving * self.step_dt + 3 * selected_new + .2 * retained * self.step_dt)
        margin = self._support_margin(sole, selected)
        normal_force = self.contacts.data.net_forces_w.torch[:, self.foot_sensors, 2].clamp(min=0)
        selected_force = normal_force[self.index, selected]
        others_force = normal_force.sum(1) - selected_force
        weight = cfg.nominal_mass_kg * 9.81
        unload = (1 - selected_force / (weight * .25)).clamp(0, 1)
        transfer = ((margin + .01) / .03).clamp(0, 1) * unload
        transfer *= upright & (selected_others >= 3) & (others_force > .7 * weight) & (fraction < .75)
        reward += self.weights["transfer_weight"] * transfer * self.step_dt

        self.acc["speed"] += delta / self.step_dt
        self.acc["slip"] += (slip * contact).sum(1) / support_count.clamp(min=1)
        self.acc["yaw"] = torch.maximum(self.acc["yaw"], yaw.abs() * (180 / math.pi) * supported)
        self.acc["sideways"] += body_vel[:, 1].abs() * supported
        self.acc["horizontal"] += body_vel[:, :2].abs().sum(1) * supported
        self.acc["depth"] += self.depth.frame[:, cfg.camera_rows * cfg.camera_cols:].mean(1)
        self.acc["rate"] += self.rate_limited.float().mean(1)
        self.acc["clearance"] = torch.maximum(self.acc["clearance"], (low * (~contact) * (other_support >= 3) * upright[:, None]).amax(1))
        self.acc["margin"] += margin.clamp(-.2, .2)
        self.previous_position.copy_(pos)
        self.previous_sole.copy_(sole)
        self.previous_contact.copy_(contact)
        self.history_valid[:] = True
        return reward

    def _publish(self):
        if not self.rows:
            return
        recent = torch.tensor(self.rows).mean(0).tolist()
        record = {"strategy": self.cfg.strategy, "step": self.common_step_counter,
            "totals": self.episode_totals, "recent_episodes": len(self.rows),
            "recent": dict(zip(METRIC_COLUMNS, recent)),
            "placed_episodes_by_leg": self.episode_placed_by_leg,
            "lift_episodes_by_leg": self.episode_lift_by_leg,
            "leg_order": ["front_left", "rear_left", "front_right", "rear_right"],
            "evidence": "stochastic_training_only", "weights": self.weights}
        temporary = self.counter_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(self.counter_file)
        with self.metrics_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        self.last_summary_step = self.common_step_counter
        self.screen_log = {"Screen/" + key: value for key, value in record["recent"].items()}
        self.extras.setdefault("log", {}).update(self.screen_log)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if hasattr(self, "acc"):
            ids = env_ids[self.episode_length_buf[env_ids] > 0]
            if len(ids):
                age = self.episode_length_buf[ids].float().clamp(min=1)
                a = self.acc
                columns = [self.success[ids], self.fallen[ids], self.placed[ids].any(1),
                    a["front_pair"][ids], a["rear_pair"][ids], a["all_first"][ids],
                    self.placed[ids].float().sum(1), a["advance"][ids], a["speed"][ids] / age,
                    a["slip"][ids] / age, a["yaw"][ids], a["sideways"][ids] / a["horizontal"][ids].clamp(min=.001),
                    self.lift_seen[ids].any(1), a["depth"][ids] / age, a["rate"][ids] / age,
                    age * self.step_dt, a["clearance"][ids], a["margin"][ids] / age]
                rows = torch.stack(columns, 1).float().cpu().tolist()
                self.rows = (self.rows + rows)[-self.cfg.summary_window:]
                self.episode_totals["episodes"] += len(rows)
                for key in self.episode_totals:
                    if key != "episodes":
                        self.episode_totals[key] += sum(int(row[METRIC_COLUMNS.index(key)]) for row in rows)
                for key, tensor in ((self.episode_placed_by_leg, self.placed), (self.episode_lift_by_leg, self.lift_seen)):
                    counts = tensor[ids].sum(0).cpu().tolist()
                    for leg, value in enumerate(counts):
                        key[leg] += int(value)
                if self.common_step_counter - self.last_summary_step >= 64:
                    self._publish()
        # Reset the physics/controller without the old task's episode logger.
        metric_file = self.metrics_file
        self.metrics_file = None
        try:
            AnkleEnv._reset_idx(self, env_ids)
        finally:
            self.metrics_file = metric_file
        if not hasattr(self, "acc"):
            return
        self.filtered[env_ids] = 0
        self.history_valid[env_ids] = False
        self.expected_stance[env_ids] = True
        self.ref[env_ids] = self.neutral[env_ids]
        self.reached[env_ids] = 0
        self.swing_peak[env_ids] = 0
        self.qualified_air[env_ids] = False
        self.lift_seen[env_ids] = False
        self.placed[env_ids] = False
        self.placement_hold[env_ids] = 0
        self.prior_phase[env_ids] = -1
        self.hold[env_ids] = 0
        self.success[env_ids] = False
        self.depth.reset(env_ids)
        self.phase_offset[env_ids] = 0
        count = len(env_ids)
        self.cruise[env_ids] = self.cfg.command_min + (self.cfg.command_max - self.cfg.command_min) * torch.rand(count, device=self.device)
        for tensor in self.acc.values():
            tensor[env_ids] = 0
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        pose[:, 0] += (2 * torch.rand(count, device=self.device) - 1) * .025
        pose[:, 1] += (2 * torch.rand(count, device=self.device) - 1) * .01
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.previous_position[env_ids] = pose[:, :3]
        self.start_position[env_ids] = pose[:, :3]
        self.extras["log"] = dict(self.screen_log)

    def close(self):
        if hasattr(self, "acc"):
            self._publish()
        super().close()
