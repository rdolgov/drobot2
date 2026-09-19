"""Train residual balance around one preplanned FL lift, advance and landing.

This is a known-step curriculum. The optional reference uses time and IMU/joint
measurements; it has no terrain/contact-state feedback. The D405 proxy is in
the actor observations but this task does not prove camera-conditioned planning.
All oracle geometry below is reward, critic or training telemetry only.
"""
import json
import torch
from ankle_v23_experiments.env import StrategyEnv


def smooth(value):
    value = value.clamp(0, 1)
    return value.square() * (3 - 2 * value)


class FootholdEnv(StrategyEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.goal_hold = torch.zeros(self.num_envs, device=self.device)
        self.goal_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.task_rows = []
        self.task_totals = {"episodes": 0, "foothold_success": 0, "full_climb": 0, "fall": 0}
        self.task_sums = {key: torch.zeros(self.num_envs, device=self.device) for key in
                          ("goal_distance", "stance_error", "normal_FL", "normal_RL", "normal_FR", "normal_RR", "root_z", "torque_saturated")}
        self.task_min_distance = torch.full((self.num_envs,), 10., device=self.device)
        self.task_peak = torch.zeros((self.num_envs, 4), device=self.device)
        self.trace_file = self.metrics_file.with_suffix(".trace.jsonl")
        self.task_file = self.metrics_file.with_suffix(".foothold.json")
        self.event_file = self.metrics_file.with_suffix(".foothold_events.jsonl")
        print("FOOTHOLD_CONTRACT=" + json.dumps({
            "guidance": cfg.guidance, "reset_base_height_m": cfg.robot.init_state.pos[2],
            "neutral_floor_height_m": cfg.target_base_height_m,
            "selected_leg": "front_left", "preplanned_displacement_m": [cfg.placement_forward_m, 0, cfg.placement_height_m],
            "success": "qualified airborne FL landing held one second; intermediate only",
            "full_climb_success_unchanged": True, "motor_caps_unchanged": True,
            "evidence": "training_only"}), flush=True)

    def _set_commands(self):
        self.commands[:] = 0.

    def _motion(self):
        age = self.episode_length_buf.float() * self.step_dt
        advance = self.cfg.placement_forward_m * smooth((age - 3.) / 2.)
        lift = self.cfg.swing_height_m * smooth((age - 1.5) / 1.5)
        lift += (self.cfg.placement_height_m - self.cfg.swing_height_m) * smooth((age - 5.) / 1.5)
        return age, advance, lift

    def _reference(self):
        age, advance, lift = self._motion()
        ref = self.neutral.clone()
        if self.cfg.guidance:
            forward = self.cfg.stance_forward_m + advance
            down = self.cfg.stance_down_m - lift
            length = .159896689
            cosine = ((forward.square() + down.square() - 2*length**2)/(2*length**2)).clamp(-.999, .999)
            knee = -torch.acos(cosine)
            hip = torch.atan2(forward, down) - torch.atan2(torch.sin(knee), 1+torch.cos(knee))
            ref[:, 4], ref[:, 8] = hip, knee
        quat = self.robot.data.root_quat_w.torch
        pitch = torch.asin((2*(quat[:, 3]*quat[:, 1]-quat[:, 2]*quat[:, 0])).clamp(-1, 1))
        joints = self.robot.data.joint_pos.torch[:, self.ids]
        ref[:, 12:16] = pitch[:, None] - joints[:, 4:8] - joints[:, 8:12]
        self.expected_stance[:] = True
        self.expected_stance[:, 0] = (age < 1.5) | (age > 6.5)
        self.ref = ref
        return ref

    def _get_dones(self):
        terminal, truncated = super()._get_dones()
        if hasattr(self, "goal_done"):
            terminal |= self.goal_done
        return terminal, truncated

    def _get_rewards(self):
        local, gap, surface, contact, planted, levels = self._geometry()
        sole = local.mean(2)
        slip = ((sole-self.previous_sole)[..., :2]/self.step_dt).norm(dim=2)*self.history_valid[:, None]
        # Preserve the common milestone detector/logging, discard its old task reward.
        super()._get_rewards()
        age, advance, lift = self._motion()
        nominal = torch.tensor([[.08305, .198696689, 0.], [-.08695, .198696689, 0.],
                               [.08695, -.198696689, 0.], [-.08305, -.198696689, 0.]], device=self.device)
        target = nominal[None].repeat(self.num_envs, 1, 1)
        target[:, :, :2] += (self.start_position-self.scene.env_origins)[:, None, :2]
        goal = target[:, 0].clone()
        goal[:, 0] += self.cfg.placement_forward_m
        goal[:, 2] = self.cfg.placement_height_m
        target[:, 0, 0] += advance
        target[:, 0, 2] = lift
        error = sole - target
        scale = torch.tensor([.045, .035, .025], device=self.device)
        waypoint = torch.exp(-(error[:, 0]/scale).square().sum(1))
        stance_error = error[:, 1:].norm(dim=2).mean(1)
        stance_score = torch.exp(-(error[:, 1:]/.025).square().sum(2)).mean(1)
        upright = self.robot.data.projected_gravity_b.torch[:, 2] < -.90
        posture = self.robot.data.projected_gravity_b.torch[:, :2].square().sum(1)
        goal_distance = (sole[:, 0]-goal).norm(dim=1)
        valid = (self.placed[:, 0] & planted[:, 0] & (levels[:, 0] >= 1)
                 & (contact[:, 1:].sum(1) >= 2) & (slip[:, 0] < .015)
                 & (goal_distance < .035) & upright & (~self.fallen))
        self.goal_hold = torch.where(valid, self.goal_hold+self.step_dt, 0.)
        new_goal = (self.goal_hold >= self.cfg.goal_hold_s) & ~self.goal_done
        self.goal_done |= new_goal
        torque = self.robot.data.applied_torque.torch[:, self.ids]
        qd = self.robot.data.joint_vel.torch[:, self.ids]
        power = (torque*qd).clamp(min=0).sum(1)
        pos = self.robot.data.root_pos_w.torch-self.scene.env_origins
        body_velocity = self.robot.data.root_lin_vel_b.torch
        reward = (6*waypoint*upright + 2*stance_score + .5*(contact[:, 1:].sum(1) >= 3)
                  - 8*posture - .5*(self._yaw()/.25).square().clamp(max=9)
                  - .2*((pos[:, 2]-self.cfg.target_base_height_m)/.03).square().clamp(max=9)
                  - .15*((slip/.02).square().clamp(max=25)*contact).mean(1)
                  - .2*(body_velocity.norm(dim=1)/.05).square().clamp(max=9)
                  - .01*power - .02*(self.actions-self.last_actions).square().mean(1)
                  - .01*self.filtered.square().mean(1)) * self.step_dt
        reward += 10*new_goal - 5*self.fallen
        self.task_min_distance = torch.minimum(self.task_min_distance, goal_distance)
        self.task_peak = torch.maximum(self.task_peak, sole[..., 2])
        forces = self.contacts.data.net_forces_w.torch[:, self.foot_sensors, 2]
        for key, value in [("goal_distance", goal_distance), ("stance_error", stance_error),
                           ("root_z", pos[:, 2]), ("torque_saturated", (torque.abs()>.80).float().mean(1)),
                           *[("normal_"+name, forces[:, i]) for i, name in enumerate(("FL", "RL", "FR", "RR"))]]:
            self.task_sums[key] += value
        if new_goal.any():
            ids = new_goal.nonzero().flatten()
            events = {"step": self.common_step_counter, "environments": ids.cpu().tolist(),
                      "root_xyz": pos[ids].cpu().tolist(), "sole_xyz": sole[ids].cpu().tolist(),
                      "levels": levels[ids].cpu().tolist(), "hold_s": self.goal_hold[ids].cpu().tolist(),
                      "kind": "intermediate_foothold_not_full_climb"}
            with self.event_file.open("a") as stream:
                stream.write(json.dumps(events)+"\n")
        if self.common_step_counter % 60 == 0:
            with self.trace_file.open("a") as stream:
                stream.write(json.dumps({"step": self.common_step_counter, "age": age[:4].cpu().tolist(),
                    "root": pos[:4].cpu().tolist(), "sole": sole[:4].cpu().tolist(),
                    "target": target[:4].cpu().tolist(), "forces": forces[:4].cpu().tolist(),
                    "planted": planted[:4].cpu().tolist(), "joints": self.robot.data.joint_pos.torch[:4, self.ids].cpu().tolist(),
                    "torques": torque[:4].cpu().tolist(), "placed": self.placed[:4].cpu().tolist()})+"\n")
        return reward

    def _publish_task(self):
        if not self.task_rows:
            return
        recent = {key: sum(row[key] for row in self.task_rows)/len(self.task_rows) for key in self.task_rows[0]}
        record = {"step": self.common_step_counter, "totals": self.task_totals,
                  "recent_episodes": len(self.task_rows), "recent": recent,
                  "evidence": "stochastic_training_only", "guidance": self.cfg.guidance}
        temporary = self.task_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2)+"\n")
        temporary.replace(self.task_file)
        self.extras.setdefault("log", {}).update({"Foothold/"+key: value for key, value in recent.items()})

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if hasattr(self, "goal_done"):
            ids = env_ids[self.episode_length_buf[env_ids] > 0]
            if len(ids):
                ages = self.episode_length_buf[ids].float().clamp(min=1)
                values = {"foothold_success": self.goal_done[ids].float(), "full_climb": self.success[ids].float(),
                          "fall": self.fallen[ids].float(), "episode_s": ages*self.step_dt,
                          "min_goal_distance_m": self.task_min_distance[ids], "FL_peak_z_m": self.task_peak[ids, 0],
                          **{key: value[ids]/ages for key, value in self.task_sums.items()}}
                numbers = {key: value.cpu().tolist() for key, value in values.items()}
                self.task_rows = (self.task_rows + [{key: value[i] for key, value in numbers.items()} for i in range(len(ids))])[-512:]
                self.task_totals["episodes"] += len(ids)
                for key in ("foothold_success", "full_climb", "fall"):
                    self.task_totals[key] += int(sum(numbers[key]))
                self._publish_task()
        super()._reset_idx(env_ids)
        if not hasattr(self, "goal_done"):
            return
        self.goal_done[env_ids] = False
        self.goal_hold[env_ids] = 0
        self.task_min_distance[env_ids] = 10.
        self.task_peak[env_ids] = 0.
        for value in self.task_sums.values():
            value[env_ids] = 0.
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        pose[:, :2] += (2*torch.rand((len(env_ids), 2), device=self.device)-1)*self.cfg.start_jitter_m
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.previous_position[env_ids] = pose[:, :3]
        self.start_position[env_ids] = pose[:, :3]

    def close(self):
        if hasattr(self, "task_rows"):
            self._publish_task()
        super().close()
