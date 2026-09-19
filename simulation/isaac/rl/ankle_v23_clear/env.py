"""Residual RL around reach-constrained, known-stair crawl references."""
import json
from pathlib import Path
import torch
from ankle_v23_experiments.env import StrategyEnv
from .plan import make_plan


class ClearEnv(StrategyEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        plan = make_plan(cfg.phase_count, self.step_dt, cfg.phase_s,
                         cfg.start_x_m, cfg.stair_rise_m, cfg.stair_tread_m, cfg.stair_start_m)
        self.plan_feet = torch.tensor(plan["feet"], device=self.device)
        self.plan_base = torch.tensor(plan["base"], device=self.device)
        self.plan_stance = torch.tensor(plan["stance"], dtype=torch.bool, device=self.device)
        self.plan_joints = torch.tensor(plan["joints"], device=self.device)
        before = torch.cat((self.plan_joints[:1], self.plan_joints[:-1]), 0)
        after = torch.cat((self.plan_joints[1:], self.plan_joints[-1:]), 0)
        self.plan_velocity = (after-before)/(2*self.step_dt)
        self.plan_pitch = torch.tensor(plan["pitch"], device=self.device)
        compensation = json.loads(Path(__file__).with_name("feedforward.json").read_text())
        self.feedforward = torch.tensor([row["torques_nm"] for row in compensation["rows"]], device=self.device)
        self.feedforward_dt = compensation["dt_s"]
        self.reset_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ground_eligible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.plan_origin = self.start_position.clone()
        self.air_required = torch.ones((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self.curriculum_top = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_feet = torch.tensor(plan["goal_feet"], device=self.device)
        self.goal_legs = plan["moved_legs"]
        self.phase_order = plan["order"]
        self.goal_hold = torch.zeros(self.num_envs, device=self.device)
        self.goal_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_full = torch.zeros_like(self.goal_done)
        self.task_rows = []
        self.task_totals = {"episodes": 0, "plan_success": 0, "full_climb": 0, "fall": 0,
                            "ground_episodes": 0, "ground_plan_success": 0, "curriculum_episodes": 0, "curriculum_plan_success": 0}
        self.task_sums = {key: torch.zeros(self.num_envs, device=self.device) for key in
            ("goal_distance", "tracking_error", "root_tracking", "normal_FL", "normal_RL", "normal_FR", "normal_RR", "root_z", "torque_saturated")}
        self.task_min_distance = torch.full((self.num_envs,), 10., device=self.device)
        self.max_phase = torch.zeros(self.num_envs, device=self.device)
        self.task_file = self.metrics_file.with_suffix(".foothold.json")
        self.trace_file = self.metrics_file.with_suffix(".trace.jsonl")
        self.event_file = self.metrics_file.with_suffix(".plan_events.jsonl")
        self.trajectory = torch.zeros((self.num_envs, int(cfg.episode_length_s)+3, 46), device=self.device)
        self.trajectory_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.trajectory_exports = 0
        self.full_event_count = 0
        self.range_peak = torch.zeros(self.num_envs, device=self.device)
        self.failure_exports = {}
        contract = {"phase_count": cfg.phase_count, "phase_s": cfg.phase_s,
            "plan": plan["phases"], "goal_feet": plan["goal_feet"], "goal_legs": self.goal_legs,
            "teacher_planning_prior": "fixed_known_stair_geometry_not_camera_derived",
            "actor_observations": cfg.observation_space, "critic_observations": cfg.state_space,
            "joint_reference_extra": 16, "motor_limits_unchanged": True,
            "reset": "all_feet_on_floor_before_first_riser", "evidence": "training_only"}
        contract.update(reset="mixed floor and reference-phase curriculum resets",
                        ground_start_fraction=cfg.ground_start_fraction,
                        full_climb_eligibility="floor starts only; curriculum arrivals excluded",
                        compensation="precomputed static joint torques as PD target offsets; no external forces",
                        torque_limits="unchanged; compensation clamped to per-episode effort cap",
                        pitch_reference="planned support slope limited to ten degrees")
        contract["velocity_compensation"] = "position-target lead Kd/Kp * planned joint velocity; candidate damping 0.30; unchanged stiffness 30, rate/position/effort limits"
        contract["privileged_actor"] = cfg.privileged_actor
        contract["privileged_features"] = "body velocity 3, root-reference error 3, sole-reference errors 12, foot contact 4, plan progress 1"
        contract["critic_features"] = "full reference tracking state"
        contract["reward_variant"] = "swing tracking and contact-pattern shaping; interior support reference"
        contract["ankle_control"] = "planned stance ankle angles preserve body attitude stiffness; measured leveling on swing only"
        contract["motion_range_success_guard"] = "episode maximum joint-range overshoot <= 0.02 rad solver tolerance"
        contract["collision_aware_reference"] = "front pair advance before rear pair; sampled nonadjacent URDF box-overlap rejection; self-collision remains enabled"
        self.metrics_file.with_suffix(".plan_contract.json").write_text(json.dumps(contract, indent=2)+"\n")
        print("CRAWLPLAN_CONTRACT="+json.dumps(contract), flush=True)
        physical = {'user_joint_names':self.robot.joint_names,
            'backend_joint_names':self.robot.backend_joint_names,
            'backend_joint_limits':self.robot.root_view.get_dof_limits().numpy()[0].tolist(),
            'user_joint_limits':self.robot.data.joint_pos_limits.torch[0].cpu().tolist(),
            'body_names':self.robot.body_names,
            'body_mass':self.robot.data.body_mass.torch[0].cpu().tolist(),
            'candidate_damping':self.cfg.servo_damping}
        self.metrics_file.with_suffix('.physical_metadata.json').write_text(json.dumps(physical,indent=2)+'\n')

    def _set_commands(self):
        self.commands[:] = 0.

    def _plan_index(self):
        return (self.episode_length_buf+self.reset_step).clamp(max=len(self.plan_feet)-1)

    def _get_dones(self):
        _, _ = super()._get_dones()
        self._update_range_peak()
        reached_top = self.success.clone()
        self.curriculum_top = reached_top & ~self.ground_eligible & (self.range_peak <= .02)
        self.success &= self.ground_eligible & (self.range_peak <= .02)
        truncated = self.episode_length_buf+self.reset_step >= self.max_episode_length-1
        return self.fallen | reached_top, truncated

    def _update_range_peak(self):
        if hasattr(self, 'range_peak'):
            q = self.robot.data.joint_pos.torch[:, self.ids]
            excess = torch.maximum(q-self.limits[:, :, 1]/.95, self.limits[:, :, 0]/.95-q).clamp(min=0).amax(1)
            self.range_peak = torch.maximum(self.range_peak, excess)

    def _apply_action(self):
        self._update_range_peak()
        super()._apply_action()

    def _reference(self):
        index = self._plan_index()
        ref = self.plan_joints[index].clone()
        # Use the same physical drive gains and effort caps. Offsets compensate
        # estimated gravity/contact loads rather than adding forces to the base.
        time_index = (index.float()*self.step_dt).clamp(max=self.cfg.phase_count*self.cfg.phase_s-self.step_dt)/self.feedforward_dt
        low = time_index.long().clamp(max=len(self.feedforward)-2)
        blend = (time_index-low).clamp(0, 1)[:, None]
        effort = (1-blend)*self.feedforward[low]+blend*self.feedforward[low+1]
        effort = torch.maximum(torch.minimum(effort, self.cfg.effort_cap_nm*self.strength), -self.cfg.effort_cap_nm*self.strength)
        velocity = torch.maximum(torch.minimum(self.plan_velocity[index], self.rates*self.rate_strength), -self.rates*self.rate_strength)
        lead = (self.cfg.servo_damping/30.)*velocity
        ref += effort/30.+lead
        quat = self.robot.data.root_quat_w.torch
        pitch = torch.asin((2*(quat[:, 3]*quat[:, 1]-quat[:, 2]*quat[:, 0])).clamp(-1, 1))
        actual = self.robot.data.joint_pos.torch[:, self.ids]
        leveled = pitch[:, None]-actual[:, 4:8]-actual[:, 8:12]+effort[:, 12:16]/30.+lead[:, 12:16]
        ref[:, 12:16] = torch.where(self.plan_stance[index], ref[:, 12:16], leveled)
        self.ref = ref
        self.expected_stance = self.plan_stance[index].clone()
        return ref

    def _get_observations(self):
        obs = super()._get_observations()
        reference_error = self.plan_joints[self._plan_index()]-self.robot.data.joint_pos.torch[:, self.ids]
        obs["policy"] = torch.cat((obs["policy"], reference_error), 1)
        obs["critic"] = torch.cat((obs["critic"], reference_error), 1)
        index = self._plan_index()
        start = self.plan_origin-self.scene.env_origins
        target = self.plan_feet[index].clone()
        target[:, :, :2] += start[:, None, :2]
        base_target = self.plan_base[index].clone()
        base_target[:, :2] += start[:, :2]
        local, _, _, contact, _, _ = self._geometry()
        root = self.robot.data.root_pos_w.torch-self.scene.env_origins
        privileged = torch.cat((self.robot.data.root_lin_vel_b.torch,
            base_target-root, (target-local.mean(2)).flatten(1), contact.float(),
            index[:, None].float()/len(self.plan_feet)), 1)
        obs["critic"] = torch.cat((obs["critic"], privileged), 1)
        if self.cfg.privileged_actor:
            obs["policy"] = torch.cat((obs["policy"], privileged), 1)
        return obs

    def _get_rewards(self):
        local, gap, surface, contact, planted, levels = self._geometry()
        sole = local.mean(2)
        slip = ((sole-self.previous_sole)[..., :2]/self.step_dt).norm(dim=2)*self.history_valid[:, None]
        old_placed = self.placed.clone()
        super()._get_rewards()
        index = self._plan_index()
        start = self.plan_origin-self.scene.env_origins
        target = self.plan_feet[index].clone()
        target[:, :, :2] += start[:, None, :2]
        base_target = self.plan_base[index].clone()
        base_target[:, :2] += start[:, :2]
        goal = self.goal_feet[None].repeat(self.num_envs, 1, 1)
        goal[:, :, :2] += start[:, None, :2]
        errors = sole-target
        pos = self.robot.data.root_pos_w.torch-self.scene.env_origins
        goal_distance = (sole[:, self.goal_legs]-goal[:, self.goal_legs]).norm(dim=2).amax(1)
        upright = self.robot.data.projected_gravity_b.torch[:, 2] < -.90
        goal_levels = (self.goal_feet[:, 2]/self.cfg.stair_rise_m).round()
        good_feet = planted & (levels == goal_levels[None]) & (self.lift_seen | ~self.air_required) & (slip < .015)
        valid = good_feet[:, self.goal_legs].all(1) & (contact.sum(1)>=3) & (goal_distance<.035) & upright & (~self.fallen) & (self.range_peak <= .02)
        self.goal_hold = torch.where(valid, self.goal_hold+self.step_dt, 0.)
        new_goal = (self.goal_hold >= self.cfg.goal_hold_s) & ~self.goal_done
        self.goal_done |= new_goal
        new_full = self.success & ~self.ever_full
        self.ever_full |= self.success
        self.full_event_count += int(new_full.sum())
        # Save existing training trajectories only. No replay rollout is run.
        sample = (self.episode_length_buf % 60 == 1) | new_full | new_goal
        sample_ids = sample.nonzero().flatten()
        if len(sample_ids):
            values = torch.cat((
                (self.episode_length_buf*self.step_dt)[:, None],
                ((self.episode_length_buf+self.reset_step)*self.step_dt)[:, None],
                pos, self.robot.data.root_quat_w.torch, sole.flatten(1), levels,
                self.robot.data.joint_pos.torch[:, self.ids], contact.float(),
                self.ground_eligible.float()[:, None]), 1)
            slots = self.trajectory_count[sample_ids].clamp(max=self.trajectory.shape[1]-1)
            self.trajectory[sample_ids, slots] = values[sample_ids]
            self.trajectory_count[sample_ids] = (slots+1).clamp(max=self.trajectory.shape[1])
        export = new_full | (new_goal & self.ground_eligible & (self.trajectory_exports < 8))
        if export.any():
            for env_id in export.nonzero().flatten().cpu().tolist()[:4]:
                folder = self.metrics_file.parent/"training_trajectories"
                folder.mkdir(exist_ok=True)
                label = "full-climb" if bool(new_full[env_id]) else "ground-stage"
                data = {"kind": label, "step": self.common_step_counter, "environment":env_id,
                    "evidence":"existing_training_episode_not_evaluation", "ground_eligible":bool(self.ground_eligible[env_id]),
                    "reset_phase":float(self.reset_step[env_id]*self.step_dt/self.cfg.phase_s),
                    "privileged_actor":self.cfg.privileged_actor,
                    "peak_joint_range_overshoot_rad":float(self.range_peak[env_id]),
                    "columns":"age, plan_age, root_xyz(3), root_xyzw(4), sole_xyz_FL_RL_FR_RR(12), levels(4), joints(16), contact(4), ground_eligible",
                    "samples":self.trajectory[env_id,:int(self.trajectory_count[env_id])].cpu().tolist()}
                (folder/f"{label}-step{self.common_step_counter}-env{env_id}.json").write_text(json.dumps(data)+"\n")
                self.trajectory_exports += 1
        if self.fallen.any():
            for env_id in self.fallen.nonzero().flatten().cpu().tolist():
                phase = int(index[env_id]*self.step_dt/self.cfg.phase_s)
                if self.failure_exports.get(phase, 0) >= 3:
                    continue
                self.failure_exports[phase] = self.failure_exports.get(phase, 0)+1
                folder = self.metrics_file.parent/'training_failures'
                folder.mkdir(exist_ok=True)
                record = {'step':self.common_step_counter, 'env':env_id, 'phase':phase,
                    'age':float(self.episode_length_buf[env_id]*self.step_dt),
                    'reset_phase':float(self.reset_step[env_id]*self.step_dt/self.cfg.phase_s),
                    'ground_eligible':bool(self.ground_eligible[env_id]),
                    'root':pos[env_id].cpu().tolist(), 'root_target':base_target[env_id].cpu().tolist(),
                    'com':((self.robot.data.body_com_pos_w.torch[env_id]*self.robot.data.body_mass.torch[env_id,:,None]).sum(0)/self.robot.data.body_mass.torch[env_id].sum()-self.scene.env_origins[env_id]).cpu().tolist(),
                    'quaternion_xyzw':self.robot.data.root_quat_w.torch[env_id].cpu().tolist(),
                    'gravity':self.robot.data.projected_gravity_b.torch[env_id].cpu().tolist(),
                    'base_contact':self.contacts.data.net_forces_w.torch[env_id,self.base_sensor].cpu().tolist(),
                    'joint_range_peak':float(self.range_peak[env_id]),
                    'joint_angles':self.robot.data.joint_pos.torch[env_id,self.ids].cpu().tolist(),
                    'joint_targets':self.targets[env_id].cpu().tolist(),
                    'sole':sole[env_id].cpu().tolist(), 'foot_targets':target[env_id].cpu().tolist(),
                    'recent_samples':self.trajectory[env_id,:int(self.trajectory_count[env_id])][-6:].cpu().tolist()}
                (folder/f'phase{phase}-sample{self.failure_exports[phase]}.json').write_text(json.dumps(record,indent=2)+'\n')
        scale = torch.tensor([.04, .03, .025], device=self.device)
        track = torch.exp(-(errors/scale).square().sum(2)).mean(1)
        support = self.expected_stance
        stance_quality = (planted*support).sum(1)/support.sum(1).clamp(min=1)
        root_error = (pos-base_target).norm(dim=1)
        root_track = torch.exp(-((pos-base_target)/.04).square().sum(1))
        expected_gravity = torch.stack((self.plan_pitch[index].sin(), torch.zeros(self.num_envs, device=self.device)), 1)
        posture = (self.robot.data.projected_gravity_b.torch[:, :2]-expected_gravity).square().sum(1)
        torque = self.robot.data.applied_torque.torch[:, self.ids]
        velocity = self.robot.data.joint_vel.torch[:, self.ids]
        power = (torque*velocity).clamp(min=0).sum(1)
        retained = (planted & self.placed & support & (levels>=1)).float().sum(1)
        reward = (8*track*upright + 2*stance_quality + 2*root_track + retained
            - 8*posture - .5*(self._yaw()/.25).square().clamp(max=9)
            - .10*((slip/.02).square().clamp(max=25)*contact).mean(1)
            - .01*power - .02*(self.actions-self.last_actions).square().mean(1)
            - .25*self.filtered.square().mean(1))*self.step_dt
        swing = ~support
        swing_score = (torch.exp(-(errors/scale).square().sum(2))*swing).sum(1)
        contact_match = (contact == support).float().mean(1)
        swing_load = (self.contacts.data.net_forces_w.torch[:, self.foot_sensors, 2].clamp(min=0)/8.).clamp(max=2.)
        reward += (4*swing_score + 2*contact_match - 2*(swing_load*swing).sum(1))*self.step_dt
        q = self.robot.data.joint_pos.torch[:, self.ids]
        soft_excess = torch.maximum(q-self.limits[:, :, 1], self.limits[:, :, 0]-q).clamp(min=0)
        reward -= .5*(soft_excess/.05).square().mean(1)*self.step_dt
        reward += 15*new_goal - 8*self.fallen + 4*(self.placed & ~old_placed).sum(1)
        # Full-goal termination compensates all forfeited positive dense reward.
        remaining = (self.cfg.episode_length_s-(self.episode_length_buf+self.reset_step)*self.step_dt).clamp(min=0)
        reward += new_full*(50+24*remaining)
        reward += self.curriculum_top*(15+24*remaining)
        self.task_min_distance = torch.minimum(self.task_min_distance, goal_distance)
        self.max_phase = torch.maximum(self.max_phase, ((self.episode_length_buf+self.reset_step)*self.step_dt/self.cfg.phase_s).floor())
        forces = self.contacts.data.net_forces_w.torch[:, self.foot_sensors, 2]
        for key, value in [("goal_distance", goal_distance), ("tracking_error", errors.norm(dim=2).mean(1)),
                           ("root_tracking", root_error), ("root_z", pos[:, 2]),
                           ("torque_saturated", (torque.abs()>.80).float().mean(1)),
                           *[("normal_"+name, forces[:, i]) for i, name in enumerate(("FL", "RL", "FR", "RR"))]]:
            self.task_sums[key] += value
        if (new_goal | new_full).any():
            ids = (new_goal | new_full).nonzero().flatten()
            with self.event_file.open("a") as stream:
                stream.write(json.dumps({"step": self.common_step_counter, "environments": ids.cpu().tolist(),
                    "full_climb": new_full[ids].cpu().tolist(), "plan_success": new_goal[ids].cpu().tolist(),
                    "ground_eligible": self.ground_eligible[ids].cpu().tolist(), "reset_phase": (self.reset_step[ids]*self.step_dt/self.cfg.phase_s).cpu().tolist(),
                    "root": pos[ids].cpu().tolist(), "sole": sole[ids].cpu().tolist(), "levels": levels[ids].cpu().tolist(),
                    "hold_s": self.goal_hold[ids].cpu().tolist(), "top_hold_s": self.hold[ids].cpu().tolist()})+"\n")
        if self.common_step_counter % 60 == 0:
            with self.trace_file.open("a") as stream:
                stream.write(json.dumps({"step": self.common_step_counter,
                    "age": (self.episode_length_buf[:4]*self.step_dt).cpu().tolist(),
                    "plan_age": ((self.episode_length_buf[:4]+self.reset_step[:4])*self.step_dt).cpu().tolist(),
                    "ground_eligible": self.ground_eligible[:4].cpu().tolist(),
                    "root": pos[:4].cpu().tolist(), "root_target": base_target[:4].cpu().tolist(),
                    "sole": sole[:4].cpu().tolist(), "target": target[:4].cpu().tolist(),
                    "forces": forces[:4].cpu().tolist(), "planted": planted[:4].cpu().tolist(),
                    "joints": self.robot.data.joint_pos.torch[:4, self.ids].cpu().tolist(),
                    "torques": torque[:4].cpu().tolist(), "placed": self.placed[:4].cpu().tolist()})+"\n")
        return reward

    def _publish_task(self):
        if not self.task_rows:
            return
        recent = {key: sum(row[key] for row in self.task_rows)/len(self.task_rows) for key in self.task_rows[0]}
        record = {"step": self.common_step_counter, "phase_count": self.cfg.phase_count, "totals": self.task_totals,
                  "recent_episodes": len(self.task_rows), "recent": recent,
                  "evidence": "stochastic_training_only", "planning": "known_stair_teacher"}
        record["privileged_actor"] = self.cfg.privileged_actor
        ground = [row for row in self.task_rows if row["ground_eligible"]]
        record["recent_ground_episodes"] = len(ground)
        record["recent_ground"] = {key: sum(row[key] for row in ground)/len(ground) for key in ground[0]} if ground else {}
        temporary = self.task_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2)+"\n")
        temporary.replace(self.task_file)
        self.extras.setdefault("log", {}).update({"Crawlplan/"+key: value for key, value in recent.items()})

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if hasattr(self, "goal_done"):
            ids = env_ids[self.episode_length_buf[env_ids] > 0]
            if len(ids):
                age = self.episode_length_buf[ids].float().clamp(min=1)
                values = {"plan_success": self.goal_done[ids].float(), "full_climb": self.ever_full[ids].float(),
                    "peak_joint_range_overshoot_rad":self.range_peak[ids],
                    "ground_eligible": self.ground_eligible[ids].float(), "reset_phase": self.reset_step[ids]*self.step_dt/self.cfg.phase_s,
                    "fall": self.fallen[ids].float(), "episode_s": age*self.step_dt, "max_phase": self.max_phase[ids],
                    "min_goal_distance_m": self.task_min_distance[ids],
                    **{key: value[ids]/age for key, value in self.task_sums.items()}}
                numbers = {key: value.cpu().tolist() for key, value in values.items()}
                rows = [{key: value[i] for key, value in numbers.items()} for i in range(len(ids))]
                self.task_rows = (self.task_rows+rows)[-512:]
                self.task_totals["episodes"] += len(ids)
                for key in ("plan_success", "full_climb", "fall"):
                    self.task_totals[key] += int(sum(numbers[key]))
                for row in rows:
                    prefix = "ground" if row["ground_eligible"] else "curriculum"
                    self.task_totals[prefix+"_episodes"] += 1
                    self.task_totals[prefix+"_plan_success"] += int(row["plan_success"])
                with self.metrics_file.with_suffix(".plan_episodes.jsonl").open("a") as stream:
                    for row in rows:
                        stream.write(json.dumps({"step": self.common_step_counter, **row})+"\n")
                self._publish_task()
        super()._reset_idx(env_ids)
        if not hasattr(self, "goal_done"):
            return
        self.goal_done[env_ids] = False
        self.range_peak[env_ids] = 0.
        self.trajectory_count[env_ids] = 0
        self.ever_full[env_ids] = False
        self.goal_hold[env_ids] = 0.
        self.task_min_distance[env_ids] = 10.
        self.max_phase[env_ids] = 0.
        for value in self.task_sums.values():
            value[env_ids] = 0.
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        pose[:, :2] += (2*torch.rand((len(env_ids), 2), device=self.device)-1)*self.cfg.start_jitter_m
        self.plan_origin[env_ids] = pose[:, :3]
        phase = torch.randint(1, max(2, self.cfg.phase_count), (len(env_ids),), device=self.device)
        phase.clamp_(max=self.cfg.phase_count-1)
        phase[torch.rand(len(env_ids), device=self.device)<self.cfg.ground_start_fraction] = 0
        reset_step = phase*round(self.cfg.phase_s/self.step_dt)
        self.reset_step[env_ids] = reset_step
        self.ground_eligible[env_ids] = phase == 0
        self.curriculum_top[env_ids] = False
        self.air_required[env_ids] = False
        for p in range(self.cfg.phase_count):
            selected = env_ids[phase == p]
            remaining_legs = sorted(set(self.phase_order[i] for i in range(p, self.cfg.phase_count)))
            for leg in remaining_legs:
                self.air_required[selected, leg] = True
        self.phase_offset[env_ids] = reset_step*self.step_dt*self.cfg.frequency_hz
        pose[:, :3] = self.scene.env_origins[env_ids]+self.plan_base[reset_step]
        pose[:, :2] += (self.plan_origin-self.scene.env_origins)[env_ids, :2]
        pose[:, 2] += .002
        pitch = self.plan_pitch[reset_step]
        pose[:, 3:7] = torch.stack((torch.zeros_like(pitch), (pitch/2).sin(), torch.zeros_like(pitch), (pitch/2).cos()), 1)
        joints = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joints[:, self.ids] = self.plan_joints[reset_step]
        self.robot.write_joint_position_to_sim_index(position=joints, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints), env_ids=env_ids)
        self.targets[env_ids] = self.plan_joints[reset_step]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.previous_position[env_ids] = pose[:, :3]
        self.start_position[env_ids] = pose[:, :3]

    def close(self):
        if hasattr(self, "task_rows"):
            self._publish_task()
        super().close()
