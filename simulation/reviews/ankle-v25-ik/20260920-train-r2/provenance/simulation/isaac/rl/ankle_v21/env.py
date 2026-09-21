"""Slow crawl residual learning with stance-conditioned ankle leveling.

Full-sole contact is a force + corner-height proxy, not a pressure measurement.
Swing feet are explicitly allowed and required to leave the floor.
"""
import json
import math
import torch
from ankle_v2.env import AnkleEnv, rotate
from .env_cfg import PlantedCfg

class PlantedEnv(AnkleEnv):
    cfg: PlantedCfg

    def __init__(self,cfg,render_mode=None,**kwargs):
        super().__init__(cfg,render_mode,**kwargs)
        self.filtered = torch.zeros_like(self.actions)
        self.residual_scale = torch.tensor([v for v in cfg.residual_scales for _ in range(4)],device=self.device)
        self.history_valid = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.ref = self.neutral.clone()
        self.expected_stance = torch.ones((self.num_envs,4),dtype=torch.bool,device=self.device)
        self.ref_clearance = torch.zeros((self.num_envs,4),device=self.device)
        self.phase_bias = torch.tensor([0.,.5,.75,.25],device=self.device)
        self.front_sign = torch.tensor([1.,-1.,1.,-1.],device=self.device)
        self.last_diagnostics = {}
        for key in ('flat_numerator','stance_contacts','tilt_degrees','slip_speed','ankle_motion','stance_expected','stance_observed'):
            self.sums[key] = torch.zeros(self.num_envs,device=self.device)
        # Tread corners use the 94 x 54 mm traction patch, not the larger PLA rim.
        self.corners = torch.tensor([[.031,y,z] for y in (-.047,.047) for z in (-.027,.027)],device=self.device)
        print('V21_CONTRACT='+json.dumps({'mass_kg':cfg.nominal_mass_kg,'actor':82,'critic':90,
              'mode':'filtered_residual_ik_crawl_actual_joint_ankle_leveling',
              'flat_contact_proxy':'net_force_gt_1N_and_corner_span_lt_5mm_and_highest_corner_lt_4mm',
              'ankle_range_unqualified_on_hardware':True}),flush=True)

    def _set_commands(self):
        age = self.episode_length_buf.float()*self.step_dt
        ramp = ((age-.6)/1.2).clamp(0,1)
        stop_at = self.cfg.evaluation_stop_s if self.cfg.evaluation else self.cfg.episode_length_s-3.
        ramp *= ((stop_at+.8-age)/.8).clamp(0,1)
        if self.cfg.evaluation:
            self.commands[:,0] = self.cfg.evaluation_speed*ramp
        else:
            curriculum = .55+.45*min(1.,self.common_step_counter/max(1,self.cfg.curriculum_steps))
            self.commands[:,0] = self.cruise*curriculum*ramp

    def _reference(self):
        age = self.episode_length_buf.float()*self.step_dt
        phase = (age[:,None]*self.cfg.frequency_hz+self.phase_offset[:,None]+self.phase_bias)%1.
        stance = phase < self.cfg.duty_factor
        u = (phase-self.cfg.duty_factor)/(1-self.cfg.duty_factor)
        u = u.clamp(0,1)
        smooth = u**3*(10+u*(-15+6*u))
        sweep = torch.where(stance,.5-phase/self.cfg.duty_factor,-.5+smooth)
        stride = self.commands[:,0]*self.cfg.duty_factor/self.cfg.frequency_hz
        moving_amplitude = (self.commands[:,0]/.025).clamp(0,1)
        lift = self.cfg.lift_m*torch.sin(torch.pi*u).square()*(~stance)*moving_amplitude[:,None]
        forward = self.front_sign*self.cfg.stance_forward_m+sweep*stride[:,None]
        down = self.cfg.stance_down_m-lift
        L = .159896689
        cosine = ((forward.square()+down.square()-2*L*L)/(2*L*L)).clamp(-.999,.999)
        knee = -self.front_sign*torch.acos(cosine)
        hip = torch.atan2(forward,down)-torch.atan2(torch.sin(knee),1+torch.cos(knee))
        ref = self.neutral.clone()
        ref[:,4:8], ref[:,8:12] = hip,knee
        q = self.robot.data.root_quat_w.torch
        pitch = torch.asin((2*(q[:,3]*q[:,1]-q[:,2]*q[:,0])).clamp(-1,1))
        actual = self.robot.data.joint_pos.torch[:,self.ids]
        # Use measured hip/knee angles: the ankle can correct their tracking lag.
        ref[:,12:16] = pitch[:,None]-actual[:,4:8]-actual[:,8:12]
        self.expected_stance = stance | (self.commands[:,0:1]<.005)
        self.ref_clearance = lift
        self.ref = ref
        return ref

    def _pre_physics_step(self,actions):
        self._set_commands()
        self.last_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1,1))
        self.filtered += self.step_dt/(self.cfg.filter_time_s+self.step_dt)*(self.actions-self.filtered)
        desired = self._reference()+self.filtered*self.residual_scale
        if self.cfg.ankle_mode == 'locked_target': desired[:,12:16] = 0.
        elif self.cfg.ankle_mode == 'reference_only': desired[:,12:16] = self.ref[:,12:16]
        desired = torch.maximum(torch.minimum(desired,self.limits[:,:,1]),self.limits[:,:,0])
        delta = desired-self.targets
        step_limit = torch.minimum(self.rates*self.rate_strength*self.step_dt,torch.full_like(delta,math.radians(2)))
        self.rate_limited = delta.abs()>step_limit
        self.target_delta = torch.maximum(torch.minimum(delta,step_limit),-step_limit)
        self.targets += self.target_delta

    def _get_observations(self):
        self._set_commands()
        phase = 2*torch.pi*(self.episode_length_buf.float()*self.step_dt*self.cfg.frequency_hz+self.phase_offset)
        clock = torch.stack((phase.sin(),phase.cos()),dim=1)
        q = self.robot.data.joint_pos.torch[:,self.ids]
        qd = self.robot.data.joint_vel.torch[:,self.ids]
        # Applied targets and filter state make controller memory observable.
        policy = torch.cat((self.commands*10,clock,self.imu.data.ang_vel_b.torch,
             self.robot.data.projected_gravity_b.torch,self.imu.data.lin_acc_b.torch/9.81,
             q-self.neutral,qd*.25,((self.targets-self.neutral)/self.scales).clamp(-1,1),
             self.filtered,self.expected_stance.float()),dim=1)
        critic = torch.cat((policy,self.robot.data.root_lin_vel_b.torch,
                           self.robot.data.root_pos_w.torch[:,2:3],self._contact().float()),dim=1)
        return {'policy':policy,'critic':critic}

    def _get_rewards(self):
        pos = self.robot.data.root_pos_w.torch
        vel = self.robot.data.root_lin_vel_b.torch
        angular = self.robot.data.root_ang_vel_b.torch
        qroot = self.robot.data.root_quat_w.torch
        x,y,z,w = qroot.unbind(dim=1)
        yaw = torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        moving = self.commands[:,0]>.012
        contact = self._contact()
        quat = self.robot.data.body_link_quat_w.torch[:,self.feet]
        foot = self.robot.data.body_link_pos_w.torch[:,self.feet]
        world_corners = foot.unsqueeze(2)+rotate(quat.unsqueeze(2),self.corners)
        heights = world_corners[...,2]-self.scene.env_origins[:,None,2:3]
        low,high = heights.amin(dim=2),heights.amax(dim=2)
        span = high-low
        normal_z = rotate(quat,torch.tensor([-1.,0.,0.],device=self.device))[...,2]
        sole = world_corners.mean(dim=2)
        slip_speed = torch.linalg.vector_norm((sole-self.previous_sole)[...,:2]/self.step_dt,dim=2)
        slip_speed *= self.history_valid[:,None]
        planted = contact & (span<.005) & (high<.004)
        stance_contact = self.expected_stance & contact
        denom = stance_contact.sum(dim=1).clamp(min=1)
        contact_count = contact.sum(dim=1).clamp(min=1)
        flat_score = (torch.exp(-(span/.008).square())*stance_contact).sum(dim=1)/denom
        slip = (slip_speed.square()*contact).sum(dim=1)/contact_count
        tilt = torch.acos(normal_z.clamp(-1,1))
        match = (self.expected_stance==contact).float().mean(dim=1)
        swing = ~self.expected_stance
        clear_score = (torch.exp(-((low-self.ref_clearance)/.012).square())*swing).sum(dim=1)/swing.sum(dim=1).clamp(min=1)
        joints = self.robot.data.joint_pos.torch[:,self.ids]
        qd = self.robot.data.joint_vel.torch[:,self.ids]
        effort = self.robot.data.applied_torque.torch[:,self.ids]
        power = (effort*qd).clamp(min=0).sum(dim=1)
        tracking = torch.exp(-((vel[:,0]-self.commands[:,0])/.035).square())
        posture = self.robot.data.projected_gravity_b.torch[:,:2].square().sum(dim=1)
        lateral = pos[:,1]-self.start_position[:,1]
        progress = (vel[:,0]/self.commands[:,0].clamp(min=.025)).clamp(-1,1.5)*moving
        stance_gap = ((low/.02).clamp(0,2)*self.expected_stance).mean(dim=1)
        reward = (8*tracking+1.5*progress+2*match+2*flat_score+.8*clear_score*moving+1.
            -6*slip.clamp(max=1)-4*stance_gap-5*posture-3*vel[:,1].square()-2*vel[:,2].square()
            -2*yaw.square()-5*lateral.square()-.2*angular.square().sum(dim=1)
            -.2*((pos[:,2]-.330)/.04).square()-.04*power
            -.04*(self.actions-self.last_actions).square().mean(dim=1)-.10*self.filtered.square().mean(dim=1)
            -2*joints[:,:4].square().mean(dim=1)-100*self.fallen.float())*self.step_dt
        valid = self.history_valid
        delta = pos-self.previous_position
        self.sums['move_time'] += moving*self.step_dt
        self.sums['move_distance'] += delta[:,0]*moving*valid
        self.sums['command_integral'] += self.commands[:,0]*moving*self.step_dt
        self.sums['rate_limited'] += self.rate_limited.float().mean(dim=1)
        self.sums['work'] += power*self.step_dt
        self.sums['flight_time'] += (~contact.any(dim=1))*self.step_dt
        self.sums['flat_numerator'] += (planted & self.expected_stance).sum(dim=1)*moving
        self.sums['stance_contacts'] += stance_contact.sum(dim=1)*moving
        self.sums['tilt_degrees'] += (tilt*stance_contact).sum(dim=1)*moving*(180/math.pi)
        self.sums['slip_speed'] += (slip_speed*stance_contact).sum(dim=1)*moving
        self.sums['ankle_motion'] += qd[:,12:16].abs().mean(dim=1)*self.step_dt*moving
        self.sums['stance_expected'] += self.expected_stance.sum(dim=1)*moving
        self.sums['stance_observed'] += stance_contact.sum(dim=1)*moving
        self.contact_counts += (contact & ~self.previous_contact).float()
        self.last_diagnostics = {k:v.detach().clone() for k,v in {
            'moving':moving,'contact':contact,'stance':self.expected_stance,'planted':planted,
            'tilt_deg':tilt*(180/math.pi),'slip_m_s':slip_speed,'corner_span_m':span,
            'corner_low_m':low,'corner_high_m':high,'ankle_rad':joints[:,12:16],
            'ankle_velocity_rad_s':qd[:,12:16],'base_xyz':pos,'base_velocity':vel,
            'yaw_rad':yaw,'roll_pitch_proxy':torch.sqrt(posture),'fallen':self.fallen,
            'command':self.commands[:,0],'rate_limited':self.rate_limited.float().mean(dim=1)}.items()}
        self.previous_position.copy_(pos)
        self.previous_sole.copy_(sole)
        self.previous_contact.copy_(contact)
        self.history_valid[:] = True
        return reward

    def _reset_idx(self,env_ids):
        if env_ids is None: env_ids = torch.arange(self.num_envs,device=self.device)
        log = {}
        if hasattr(self,'filtered') and len(env_ids):
            s = self.sums
            denom = s['stance_contacts'][env_ids].clamp(min=1)
            expected = s['stance_expected'][env_ids].clamp(min=1)
            log = {'Metrics/train_planted_stance_fraction':(s['flat_numerator'][env_ids]/denom).mean().item(),
                'Metrics/train_stance_tilt_deg':(s['tilt_degrees'][env_ids]/denom).mean().item(),
                'Metrics/train_stance_slip_m_s':(s['slip_speed'][env_ids]/denom).mean().item(),
                'Metrics/train_stance_contact_fraction':(s['stance_observed'][env_ids]/expected).mean().item(),
                'Metrics/train_ankle_abs_velocity_rad_s':(s['ankle_motion'][env_ids]/s['move_time'][env_ids].clamp(min=.001)).mean().item()}
            if self.metrics_file and self.common_step_counter-self.last_log_step>=640:
                with self.metrics_file.open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'step':self.common_step_counter,**log})+'\n')
        super()._reset_idx(env_ids)
        if hasattr(self,'filtered'):
            self.filtered[env_ids] = 0
            self.history_valid[env_ids] = False
            self.ref[env_ids] = self.neutral[env_ids]
            self.expected_stance[env_ids] = True
            self.ref_clearance[env_ids] = 0
            self.extras.setdefault('log',{}).update(log)
            if self.cfg.evaluation:
                self.strength[env_ids] = 1
                self.rate_strength[env_ids] = 1
                self.phase_offset[env_ids] = 0
