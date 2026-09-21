"""Residual PPO around the real diagonal IK trajectory, with physical rewards.

The phase clock is timed during learning: completion counts still require real
qualified paired landings. Unlike the old training, a failed encoder settle
does not end every exploration attempt before the agent reaches swing.
This is a new controller contract, not weights for the old hardware runtime.
"""
import json
import math
import torch
from ankle_v24_diagonal.env import DiagonalEnv, smooth
from ankle_v2.env_cfg import LEGS


def immediate(t):
    t = t.clamp(0., 1.)
    return t+t*t-t*t*t


class IKDiagonalEnv(DiagonalEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.onset_ankles = torch.zeros((self.num_envs, 4), device=self.device)
        self.previous_ankles = torch.zeros_like(self.onset_ankles)
        self.body_push = torch.zeros(self.num_envs, device=self.device)
        self.pitch_correction = torch.zeros_like(self.onset_ankles)
        masses = self.robot.data.body_mass.torch.clone()
        inertias = self.robot.data.body_inertia.torch.clone()
        scale = cfg.estimated_total_mass_kg/masses.sum(1)
        bodies = torch.arange(len(self.robot.body_names), device=self.device, dtype=torch.int32)
        envs = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self.robot.set_masses_index(masses=masses*scale[:, None], body_ids=bodies, env_ids=envs)
        self.robot.set_inertias_index(inertias=inertias*scale[:, None, None], body_ids=bodies, env_ids=envs)
        self.ik_sums = {key: torch.zeros(self.num_envs, device=self.device) for key in
                        ("samples", "tilt_square", "sync_error", "ankle_onset", "lift_samples")}
        contract = {"schema": "drobot.ankle-v25-ik.v1", "actor_inputs": 82, "critic_inputs": 101,
                    "action": "bounded filtered 16-joint residual around IK; fresh zero-mean Gaussian PPO",
                    "reference": "V5 lift/push plus V6 early ankle lead; vectorized timed training phases",
                    "phase_clock": "timed with overlap; no force/contact information in actor",
                    "success": "inherited strict physical paired-landing counts; phase advance is not success",
                    "mass_kg": cfg.estimated_total_mass_kg,
                    "mass_method": "12-servo 7 lb plus four 55 g servos; uniform density scaling; COM distribution unmeasured",
                    "effort_cap_nm": cfg.effort_cap_nm, "limits_changed": False,
                    "joint_output": "shared interpolation fraction; 60 Hz; 4 degree packet, 200 deg/s sagittal",
                    "reference_stride_m": [cfg.stride_min_m, cfg.stride_max_m],
                    "lift_m": cfg.lift_m, "support_push_m": cfg.support_push_m,
                    "body": "level trunk, low roll/pitch/yaw rates, low vertical/lateral motion",
                    "hardware_qualified": False, "deployed_runtime_compatible": False,
                    "calibration": self.calibration_record}
        if self.metrics_file:
            self.metrics_file.with_suffix(".contract.json").write_text(json.dumps(contract, indent=2)+"\n")
        print("IK_RL_CONTRACT="+json.dumps(contract), flush=True)

    def _advance_scheduler(self):
        active = (self._age() >= self.cfg.startup_s) & (self.cruise > 0.) & ~self.parked
        if self.cfg.evaluation:
            active = (self._age() >= self.cfg.startup_s) & ~self.parked
        self.stage_entries[:, 0] += active & ~self.started
        self.started |= active
        q = self.robot.data.joint_pos.torch[:, self.ids]
        self.onset_ankles = torch.where((self.stage == 0)[:, None], q[:, 12:16], self.onset_ankles)
        self.stage_time += active*self.step_dt
        _, duration = self._stage_progress()
        advance = active & (self.stage_time >= duration)
        finish = advance & (self.stage == 5)
        self.failed_pair_event = finish & ~self.pair_paid
        self.diagonal_sums["failed_pairs"] += self.failed_pair_event
        park = advance & (self.stage == 4) & self._stop_requested()
        self.parked |= park
        self.stage = torch.where(advance & ~park, (self.stage+1) % 6, self.stage)
        self.stage_entries[self.index, self.stage] += advance & ~park
        self.stage_time = torch.where(advance & ~park, 0., self.stage_time)
        self.pair = torch.where(finish, 1-self.pair, self.pair)
        for tensor in (self.landed, self.touchdown_paid, self.pair_paid, self.flight_valid):
            tensor[finish] = False
        for tensor in (self.landed_advance, self.lift_best_score, self.foot_lift_best_score,
                       self.plant_time, self.flight_peak, self.air_time):
            tensor[finish] = 0.
        self.first_placement[:] = False

    def _reference(self):
        u, _ = self._stage_progress()
        selected = self.pairs[self.pair]
        blend, lift_blend = smooth(u), immediate(u)
        onset = self.cfg.support_onset_fraction
        push_blend = torch.where(u <= .1, onset*u/.1,
                                onset+(1.-onset)*immediate((u-.1)/.9))
        early_total = self.cfg.support_push_m*(self.stride/.100)
        early = torch.where(self.stage == 1, early_total*push_blend,
                            torch.where(self.stage >= 2, early_total, 0.))
        # Preserve the final 0.04 placement fraction as stationary settle.
        final_push = smooth((u*.14/.10).clamp(0., 1.))
        push = torch.where(self.stage == 5, early_total+(.5*self.stride-early_total)*final_push, early)
        advance = torch.where(self.stage == 1, early,
                    torch.where(self.stage == 2, early_total+(self.stride-early_total)*blend,
                    torch.where(self.stage >= 3, self.stride, 0.)))
        offset = selected*(-.5*self.stride[:, None]+advance[:, None])-push[:, None]
        lift = torch.where(self.stage == 1, lift_blend,
                torch.where(self.stage == 2, 1., torch.where(self.stage == 3, 1.-blend, 0.)))
        clearance = selected*lift[:, None]*self.cfg.lift_m
        forward = self.front_sign*self.cfg.stance_forward_m+offset
        maximum_down = (self.cfg.maximum_leg_reach_m**2-forward.square()).clamp(min=.01).sqrt()+clearance
        common_down = maximum_down.amin(1).clamp(max=self.cfg.stance_down_m)
        down = common_down[:, None]-clearance
        length = .159896689
        cosine = ((forward.square()+down.square()-2*length**2)/(2*length**2)).clamp(-.999, .999)
        knee = -self.front_sign*torch.acos(cosine)
        hip = torch.atan2(forward, down)-knee*.5
        ref = self.neutral.clone()
        ref[:, 4:8], ref[:, 8:12] = hip, knee
        # Planned support angles prevent lag cancellation of early ankle drive.
        # Body-pitch correction is filtered and bounded separately from IK.
        gravity = self.robot.data.projected_gravity_b.torch
        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
        correction = (pitch[:, None]-.04*self.imu.data.ang_vel_b.torch[:, 1:2]).clamp(-math.radians(8.), math.radians(8.))
        delta = (correction-self.pitch_correction)*self.step_dt/(.10+self.step_dt)
        self.pitch_correction += delta.clamp(-math.radians(30.)*self.step_dt, math.radians(30.)*self.step_dt)
        ref[:, 12:16] = -hip-knee+self.pitch_correction
        self.reference_clipped = (ref < self.limits[:, :, 0]) | (ref > self.limits[:, :, 1])
        self.ref = torch.maximum(torch.minimum(ref, self.limits[:, :, 1]), self.limits[:, :, 0])
        self.ref_forward, self.ref_down, self.ref_clearance = forward, down, clearance
        self.body_push = push
        self.expected_stance = ~(selected & ((self.stage >= 1) & (self.stage <= 3))[:, None])
        self.expected_stance |= (~self.started | self.parked)[:, None]
        return self.ref

    def _pre_physics_step(self, actions):
        self._set_commands()
        self._advance_scheduler()
        self.last_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1., 1.))
        self.filtered += self.step_dt/(self.cfg.filter_time_s+self.step_dt)*(self.actions-self.filtered)
        desired = self._reference()+self.filtered*self.residual_scale
        self.unconstrained_target.copy_(desired)
        desired = self._project_swing_residual(desired)
        # Compensate the hip/knee residual in ankle pitch before ankle residual;
        # a support-pose correction must not inadvertently tilt the sole.
        desired[:, 12:16] = (-desired[:, 4:8]-desired[:, 8:12]+self.pitch_correction
                            +self.filtered[:, 12:16]*self.residual_scale[12:16])
        desired = torch.maximum(torch.minimum(desired, self.limits[:, :, 1]), self.limits[:, :, 0])
        self.requested_target.copy_(desired)
        deliver = torch.ones_like(self.skipped_output) if self.cfg.evaluation else (
            (torch.rand(self.num_envs, device=self.device) >= self.cfg.output_skip_probability) | self.skipped_output)
        delta = desired-self.targets
        caps = torch.minimum(self.rates*self.rate_strength*self.step_dt,
                             torch.full_like(delta, math.radians(4.)))
        fraction = (caps/delta.abs().clamp(min=1.e-8)).amin(1).clamp(max=1.)
        change = delta*fraction[:, None]
        self.previous_target_delta.copy_(self.target_delta)
        self.last_output_velocity.copy_(self.output_velocity)
        self.target_delta = torch.where(deliver[:, None], change, 0.)
        self.output_velocity = self.target_delta/self.step_dt
        self.rate_limited = (delta.abs() > caps+1.e-5) & deliver[:, None]
        self.accel_limited[:] = False
        self.targets += self.target_delta
        self.skipped_output = ~deliver

    def _get_rewards(self):
        reward = super()._get_rewards()
        d = self.last_diagnostics
        contact, low = d["contacts"], d["sole_clearance_m"]
        selected = self.pairs[self.pair]
        moving = self.started & ~self.parked & (self.commands[:, 0] > .005)
        lifting = moving & (self.stage == 1)
        swinging = moving & ((self.stage == 1) | (self.stage == 2))
        support_contact = (contact | selected).all(1)
        height_pair_min = torch.where(selected, low, torch.inf).amin(1).clamp(min=0.)
        height_pair_max = torch.where(selected, low, -torch.inf).amax(1).clamp(min=0.)
        pair_clearance = (height_pair_min/self.cfg.lift_m).clamp(0., 1.)
        sync_error = ((height_pair_max-height_pair_min)/.015).square().clamp(max=4.)
        ankle_delta = self.robot.data.joint_pos.torch[:, self.ids][:, 12:16]-self.onset_ankles
        required = math.radians(.5)
        onset = torch.where(~selected, (ankle_delta/required).clamp(0., 1.), 1.).amin(1)
        progress = (d["body_velocity"][:, 0]/self.commands[:, 0].clamp(min=.012)).clamp(-1., 1.5)
        tilt = self.robot.data.projected_gravity_b.torch[:, :2].square().sum(1)
        # Dense pair-minimum clearance and early ankle movement are shaped
        # without treating static COM-hull containment as a dynamic contact gate.
        extra = (3.*pair_clearance*swinging*support_contact
                 -.8*sync_error*swinging + 1.2*onset*lifting*support_contact
                 +1.5*progress*lifting*support_contact
                 -.5*(tilt/.10**2).clamp(max=9.)
                 -.4*(d["body_velocity"][:, 1]/.04).square().clamp(max=4.)
                 -.20*(self.actions-self.last_actions).square().mean(1))
        reward += extra*self.step_dt
        d.update({"body_push_m": self.body_push.clone(), "ankle_onset_progress": onset.clone(),
                  "pair_clearance_m": height_pair_min.clone(), "pair_height_difference_m": (height_pair_max-height_pair_min).clone(),
                  "tilt_rad": torch.asin(tilt.sqrt().clamp(max=1.)), "timed_clock": torch.ones_like(moving)})
        self.ik_sums["samples"] += moving
        self.ik_sums["tilt_square"] += tilt*moving
        self.ik_sums["sync_error"] += (height_pair_max-height_pair_min)*swinging
        self.ik_sums["ankle_onset"] += onset*lifting
        self.ik_sums["lift_samples"] += lifting
        return reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        log = {}
        if hasattr(self, "ik_sums"):
            s = self.ik_sums
            log = {"Metrics/train_body_tilt_rms_rad": (s["tilt_square"][env_ids]/s["samples"][env_ids].clamp(min=1)).sqrt().mean().item(),
                   "Metrics/train_early_ankle_progress": (s["ankle_onset"][env_ids]/s["lift_samples"][env_ids].clamp(min=1)).mean().item()}
        super()._reset_idx(env_ids)
        if not hasattr(self, "onset_ankles"):
            return
        self.first_placement[env_ids] = False
        if self.cfg.evaluation:
            self.pair[env_ids] = env_ids.to(dtype=torch.long) % 2 if self.num_envs > 1 else self.cfg.evaluation_pair
        self.pitch_correction[env_ids] = 0.
        self.onset_ankles[env_ids] = 0.
        self.body_push[env_ids] = 0.
        for values in self.ik_sums.values():
            values[env_ids] = 0.
        # Reset physically into the first IK placement, including the selected
        # pair's rearward half-stride offset; no closed-chain startup drag.
        selected = self.pairs[self.pair[env_ids]]
        forward = self.front_sign*self.cfg.stance_forward_m-selected*.5*self.stride[env_ids, None]
        length = .159896689
        knee = -self.front_sign*torch.acos(((forward.square()+self.cfg.stance_down_m**2-2*length**2)/(2*length**2)).clamp(-.999,.999))
        hip = torch.atan2(forward, torch.full_like(forward, self.cfg.stance_down_m))-knee*.5
        q = self.neutral[env_ids].clone()
        q[:, 4:8], q[:, 8:12], q[:, 12:16] = hip, knee, -hip-knee
        q = torch.maximum(torch.minimum(q, self.limits[env_ids, :, 1]), self.limits[env_ids, :, 0])
        native = self.robot.data.default_joint_pos.torch[env_ids].clone()
        native[:, self.ids] = q
        self.robot.write_joint_position_to_sim_index(position=native, env_ids=env_ids)
        self.targets[env_ids] = q
        self.ref[env_ids] = q
        self.requested_target[env_ids] = q
        self.robot.set_joint_position_target_index(target=self.targets, joint_ids=self.ids)
        self.extras.setdefault("log", {}).update(log)
