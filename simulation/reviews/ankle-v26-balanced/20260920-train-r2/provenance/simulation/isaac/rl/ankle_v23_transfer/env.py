"""Scheduled weight transfer and balanced stepping on the unchanged V2.3 foot.

The actor sees proprioception and clock state. Forces and whole-robot COM are
reward/critic information, never privileged controller gates.
"""
import json
import math
import torch
from ankle_v2.env import AnkleEnv, rotate
from ankle_v21.env import PlantedEnv
from .env_cfg import TransferCfg

class TransferEnv(PlantedEnv):
    cfg: TransferCfg

    def __init__(self,cfg,render_mode=None,**kwargs):
        # Reuse the actuator mechanics without constructing V2.1 contact geometry.
        AnkleEnv.__init__(self,cfg,render_mode,**kwargs)
        self.filtered=torch.zeros_like(self.actions)
        self.residual_scale=torch.tensor([v for v in cfg.residual_scales for _ in range(4)],device=self.device)
        self.history_valid=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.ref=self.neutral.clone()
        self.expected_stance=torch.ones((self.num_envs,4),dtype=torch.bool,device=self.device)
        self.ref_clearance=torch.zeros((self.num_envs,4),device=self.device)
        self.ref_forward=torch.zeros_like(self.ref_clearance)
        self.phase_bias=torch.tensor([0.,.5,.75,.25],device=self.device)
        self.front_sign=torch.tensor([1.,-1.,1.,-1.],device=self.device)
        self.corners=torch.tensor([[cfg.contact_x_m,y,z] for y in (-.047,.047) for z in (-.027,.027)],device=self.device)
        self.last_diagnostics={}
        self.swing_peak=torch.zeros_like(self.ref_clearance)
        self.last_touchdown=torch.zeros((self.num_envs,4,3),device=self.device)
        self.touchdown_valid=torch.zeros_like(self.expected_stance)
        self.index=torch.arange(self.num_envs,device=self.device)
        self.triangles=torch.tensor([[1,3,2],[0,3,2],[0,1,3],[0,1,2]],device=self.device)
        self.side_sign=torch.tensor([1.,1.,-1.,-1.],device=self.device)
        self.transfer_targets=torch.tensor([[-.012,-.018],[.026,-.018],[-.012,.018],[.026,.018]],device=self.device)
        self.previous_selected=torch.tensor([3,2,0,1],device=self.device)
        self.step_counts=torch.zeros_like(self.ref_clearance)
        self.swing_samples=torch.zeros_like(self.ref_clearance)
        self.clear_samples=torch.zeros_like(self.ref_clearance)
        self.unload_samples=torch.zeros_like(self.ref_clearance)
        self.unload_success=torch.zeros_like(self.ref_clearance)
        self.supported_air=torch.zeros_like(self.expected_stance)
        self.preparing=torch.zeros_like(self.expected_stance)
        self.body_shift=torch.zeros((self.num_envs,2),device=self.device)
        self.transfer_log_step=-640
        for key in ('flat_numerator','stance_contacts','tilt_degrees','slip_speed','ankle_motion',
                    'stance_expected','stance_observed','swing_expected','swing_clear',
                    'qualified_steps','stride_distance','stance_sweep'):
            self.sums[key]=torch.zeros(self.num_envs,device=self.device)
        print('V23_TRANSFER_CONTRACT='+json.dumps({'model':'V2.3_R3_solid_sole_retained_leg',
            'mass_kg':cfg.nominal_mass_kg,'actor':82,'critic':101,'contact_x_m':cfg.contact_x_m,
            'stance_sweep_m':[.08,.11],'swing_lift_m':cfg.lift_m,'duty':cfg.duty_factor,
            'motor_rate_deg_s':cfg.target_rates_deg_s,'hardware_qualified':False,
            'transfer_phase':[cfg.transfer_start_phase,cfg.duty_factor],
            'body_shift_targets_xy_m':self.transfer_targets.cpu().tolist(),
            'commands_m_s':[cfg.command_min,cfg.command_max],
            'target_base_height_m':cfg.target_base_height_m}),flush=True)

    def _schedule(self):
        age=self.episode_length_buf.float()*self.step_dt
        cruise=torch.full_like(self.cruise,self.cfg.evaluation_speed) if self.cfg.evaluation else self.cruise
        final_sweep=.08+.03*((cruise-self.cfg.command_min)/(self.cfg.command_max-self.cfg.command_min)).clamp(0,1)
        frequency=cruise.clamp(min=self.cfg.command_min)*self.cfg.duty_factor/final_sweep
        progress=1. if self.cfg.evaluation else min(1.,self.common_step_counter/max(1,self.cfg.curriculum_steps))
        ramp=((age-.8)/1.5).clamp(0,1)
        stop_at=self.cfg.evaluation_stop_s if self.cfg.evaluation else self.cfg.episode_length_s-4.
        ramp*=((stop_at+1.-age)/1.).clamp(0,1)
        return age,cruise,frequency,progress,ramp

    def _transfer_phase(self):
        age,_,frequency,_,ramp=self._schedule()
        phase=(age[:,None]*frequency[:,None]+self.phase_offset[:,None]+self.phase_bias)%1.
        selected=phase.argmax(dim=1)
        selected_phase=phase[self.index,selected]
        u=((selected_phase-self.cfg.transfer_start_phase)/(self.cfg.duty_factor-self.cfg.transfer_start_phase)).clamp(0,1)
        blend=u**3*(10+u*(-15+6*u))
        previous=self.previous_selected[selected]
        shift=self.transfer_targets[previous]*(1-blend[:,None])+self.transfer_targets[selected]*blend[:,None]
        active=(self.commands[:,0]>.005)
        shift*=ramp[:,None]*active[:,None]
        return phase,selected,u,shift

    def _support(self,sole,selected):
        # Conservative triangle of support-foot centers, not the larger sole
        # footprint. Signed distance is positive inside all three edges.
        mass=self.robot.data.body_mass.torch
        com=(self.robot.data.body_com_pos_w.torch*mass[:,:,None]).sum(1)/mass.sum(1)[:,None]
        triangle=sole[self.index[:,None],self.triangles[selected],:2]
        edge=torch.roll(triangle,-1,1)-triangle
        offset=com[:,None,:2]-triangle
        cross=edge[...,0]*offset[...,1]-edge[...,1]*offset[...,0]
        a,b=triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]
        orientation=torch.where(a[:,0]*b[:,1]-a[:,1]*b[:,0]>=0,1.,-1.)
        margin=(orientation[:,None]*cross/edge.norm(dim=2).clamp(min=.001)).amin(1)
        force=self.contacts.data.net_forces_w.torch[:,self.foot_sensors,2].clamp(min=0)
        weight=mass.sum(1)*9.81
        return com,margin,force,weight

    def _set_commands(self):
        _,cruise,_,progress,ramp=self._schedule()
        self.commands[:,0]=cruise*(.6+.4*progress)*ramp

    def _reference(self):
        _,_,frequency,progress,ramp=self._schedule()
        phase,selected,_,shift=self._transfer_phase()
        stance=phase<self.cfg.duty_factor
        u=((phase-self.cfg.duty_factor)/(1-self.cfg.duty_factor)).clamp(0,1)
        smooth=u**3*(10+u*(-15+6*u))
        sweep=torch.where(stance,.5-phase/self.cfg.duty_factor,-.5+smooth)
        stride=self.commands[:,0]*self.cfg.duty_factor/frequency
        amplitude=ramp*(self.commands[:,0]>.005)
        lift=self.cfg.lift_m*torch.sin(torch.pi*u).square()*(~stance)*amplitude[:,None]
        forward=self.front_sign*self.cfg.stance_forward_m+sweep*stride[:,None]-shift[:,0:1]
        down=self.cfg.stance_down_m-lift
        L=.159896689
        cosine=((forward.square()+down.square()-2*L*L)/(2*L*L)).clamp(-.999,.999)
        knee=-self.front_sign*torch.acos(cosine)
        hip=torch.atan2(forward,down)-torch.atan2(torch.sin(knee),1+torch.cos(knee))
        ref=self.neutral.clone()
        ref[:,:4]=-self.side_sign*torch.atan2(shift[:,1:2],torch.full_like(shift[:,1:2],self.cfg.target_base_height_m+.05))
        ref[:,4:8],ref[:,8:12]=hip,knee
        q=self.robot.data.root_quat_w.torch
        pitch=torch.asin((2*(q[:,3]*q[:,1]-q[:,2]*q[:,0])).clamp(-1,1))
        actual=self.robot.data.joint_pos.torch[:,self.ids]
        ref[:,12:16]=pitch[:,None]-actual[:,4:8]-actual[:,8:12]
        self.expected_stance=stance|(self.commands[:,0:1]<.005)
        self.ref_clearance=lift
        self.ref_forward=forward
        self.preparing=(phase>=self.cfg.transfer_start_phase)&(phase<self.cfg.duty_factor)&(self.commands[:,0:1]>.005)
        self.body_shift=shift
        self.ref=ref
        return ref

    def _get_observations(self):
        self._set_commands()
        age,_,frequency,_,_=self._schedule()
        phase=2*torch.pi*(age*frequency+self.phase_offset)
        clock=torch.stack((phase.sin(),phase.cos()),dim=1)
        q=self.robot.data.joint_pos.torch[:,self.ids]
        qd=self.robot.data.joint_vel.torch[:,self.ids]
        policy=torch.cat((self.commands*10,clock,self.imu.data.ang_vel_b.torch,
            self.robot.data.projected_gravity_b.torch,self.imu.data.lin_acc_b.torch/9.81,
            q-self.neutral,qd*.25,((self.targets-self.neutral)/self.scales).clamp(-1,1),
            self.filtered,self.expected_stance.float()),dim=1)
        _,selected,_,_=self._transfer_phase()
        quat=self.robot.data.body_link_quat_w.torch[:,self.feet]
        foot=self.robot.data.body_link_pos_w.torch[:,self.feet]
        sole=(foot.unsqueeze(2)+rotate(quat.unsqueeze(2),self.corners)).mean(2)
        com,margin,force,weight=self._support(sole,selected)
        critic=torch.cat((policy,self.robot.data.root_lin_vel_b.torch,
                         self.robot.data.root_pos_w.torch[:,2:3],self._contact().float(),
                         force/weight[:,None],(com[:,:2]-sole.mean(1)[:,:2])*10,
                         margin[:,None]*10,torch.nn.functional.one_hot(selected,4).float()),dim=1)
        return {'policy':policy,'critic':critic}

    def _get_rewards(self):
        pos=self.robot.data.root_pos_w.torch
        vel=self.robot.data.root_lin_vel_b.torch
        angular=self.robot.data.root_ang_vel_b.torch
        qroot=self.robot.data.root_quat_w.torch
        x,y,z,w=qroot.unbind(dim=1)
        yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        moving=self.commands[:,0]>.005
        contact=self._contact()
        quat=self.robot.data.body_link_quat_w.torch[:,self.feet]
        foot=self.robot.data.body_link_pos_w.torch[:,self.feet]
        world_corners=foot.unsqueeze(2)+rotate(quat.unsqueeze(2),self.corners)
        heights=world_corners[...,2]-self.scene.env_origins[:,None,2:3]
        low,high=heights.amin(dim=2),heights.amax(dim=2)
        span=high-low
        normal_z=rotate(quat,torch.tensor([-1.,0.,0.],device=self.device))[...,2]
        sole=world_corners.mean(dim=2)
        slip_speed=torch.linalg.vector_norm((sole-self.previous_sole)[...,:2]/self.step_dt,dim=2)
        slip_speed*=self.history_valid[:,None]
        planted=contact&(span<.005)&(high<.004)
        stance_contact=self.expected_stance&contact
        expected_count=self.expected_stance.sum(dim=1).clamp(min=1)
        contact_count=contact.sum(dim=1).clamp(min=1)
        flat_score=(torch.exp(-(span/.008).square())*stance_contact).sum(dim=1)/expected_count
        slip=((slip_speed/.020).square().clamp(max=25)*contact).sum(dim=1)/contact_count
        tilt=torch.acos(normal_z.clamp(-1,1))
        match=((self.expected_stance==contact)|self.preparing).float().mean(dim=1)
        mid_swing=(~self.expected_stance)&(self.ref_clearance>.008)&moving[:,None]
        swing_count=mid_swing.sum(dim=1).clamp(min=1)
        swing_clear=mid_swing&(~contact)&(low>.008)
        clear_score=(torch.exp(-((low-self.ref_clearance)/.010).square())*(~contact)*mid_swing).sum(dim=1)/swing_count
        scuff=(mid_swing&contact).sum(dim=1)/swing_count
        joints=self.robot.data.joint_pos.torch[:,self.ids]
        qd=self.robot.data.joint_vel.torch[:,self.ids]
        effort=self.robot.data.applied_torque.torch[:,self.ids]
        power=(effort*qd).clamp(min=0).sum(dim=1)
        actual_forward=.159896689*(torch.sin(joints[:,4:8])+torch.sin(joints[:,4:8]+joints[:,8:12]))
        path_score=torch.exp(-((actual_forward-self.ref_forward)/.025).square()).mean(dim=1)
        tracking=torch.exp(-((vel[:,0]-self.commands[:,0])/.012).square())
        posture=self.robot.data.projected_gravity_b.torch[:,:2].square().sum(dim=1)
        lateral=pos[:,1]-self.start_position[:,1]
        progress=(vel[:,0]/self.commands[:,0].clamp(min=.015)).clamp(-1,1.3)*moving
        _,selected,unload_progress,_=self._transfer_phase()
        com,margin,normal_force,weight=self._support(sole,selected)
        selected_force=normal_force[self.index,selected]
        other_force=normal_force.sum(1)-selected_force
        other_count=contact.sum(1)-contact[self.index,selected].long()
        upright=(posture<.15**2)&(~self.fallen)
        supported=(other_count>=3)&(other_force>.65*weight)&upright
        selected_load=selected_force/weight
        load_target=.25*(1-unload_progress)+.02*unload_progress
        unload_score=torch.exp(-((selected_load-load_target).clamp(min=0)/.08).square())
        margin_score=torch.sigmoid((margin-self.cfg.support_margin_m)/.012)
        prep=self.preparing.any(1)&moving
        transfer_score=unload_score*margin_score*supported*(prep|(~self.expected_stance).any(1))
        unloaded=(selected_load<.06)&supported&(margin>=0)&(unload_progress>.5)&prep
        # Dense progress toward swing height is available before contact breaks;
        # only actual clearance with three other supports earns the lift bonus.
        other_per_leg=contact.sum(1)[:,None]-contact.long()
        lift_supported=swing_clear&(other_per_leg>=3)&upright[:,None]&(margin[:,None]>=0)
        dense_lift=(1-((low-self.ref_clearance).abs()/.018).clamp(0,1))*mid_swing
        rear_weight=torch.tensor([1.,1.7,1.,1.7],device=self.device)
        lift_score=(dense_lift*rear_weight).sum(1)/swing_count
        supported_lift_score=(lift_supported*rear_weight).sum(1)/swing_count
        self.air_time+=(~contact)*self.step_dt
        self.swing_peak=torch.maximum(self.swing_peak,low*(~contact))
        self.supported_air|=(~contact)&(self.air_time>.10)&(low>.008)&(other_per_leg>=3)&upright[:,None]
        touchdown=contact&(~self.previous_contact)&self.history_valid[:,None]
        qualified=touchdown&(self.air_time>.10)&(self.swing_peak>.008)&moving[:,None]
        measured=sole[:,:,0]-self.last_touchdown[:,:,0]
        lateral_step=(sole[:,:,1]-self.last_touchdown[:,:,1]).abs()
        valid_step=qualified&self.touchdown_valid&(measured>.02)&(measured<.30)&(lateral_step<.05)
        _,_,frequency,_,_=self._schedule()
        desired_step=self.commands[:,0]/frequency
        # Reward genuine touchdown advances on underused legs. A front-only
        # gait cannot collect the all-four-round bonus.
        support_step=valid_step&self.supported_air&(other_per_leg>=2)&upright[:,None]
        minimum_before=self.step_counts.amin(1)
        lagging=self.step_counts<=minimum_before[:,None]
        touchdown_score=(torch.exp(-((measured-desired_step[:,None])/.045).square())*support_step*rear_weight*(1+lagging.float())).sum(dim=1)
        self.step_counts+=support_step
        balanced_round=self.step_counts.amin(1)-minimum_before
        self.last_touchdown=torch.where(qualified[:,:,None],sole,self.last_touchdown)
        self.touchdown_valid|=qualified
        self.air_time[contact]=0.
        self.swing_peak[contact]=0.
        self.supported_air[contact]=False
        self.swing_samples+=mid_swing
        self.clear_samples+=lift_supported
        self.unload_samples[self.index,selected]+=prep&(unload_progress>.5)
        self.unload_success[self.index,selected]+=unloaded
        stance_fraction=stance_contact.sum(dim=1)/expected_count
        tracking_weight=4. if self.cfg.evaluation else 2+2*min(1.,self.common_step_counter/max(1,self.cfg.curriculum_steps))
        reward=(tracking_weight*tracking+progress
            +2*stance_fraction+flat_score+2*match+2*clear_score*moving
            +4*transfer_score*moving+3*lift_score*moving+4*supported_lift_score*moving
            +path_score*moving+1.-1.5*slip-2*scuff-1.*(posture/.12**2).clamp(max=9)
            -.6*((lateral-self.body_shift[:,1])/.06).square().clamp(max=9)-.6*(yaw/.15).square().clamp(max=9)
            -.5*((vel[:,1].abs()-.08).clamp(min=0)/.04).square().clamp(max=9)-2*vel[:,2].square()-.15*angular.square().sum(dim=1)
            -.4*((pos[:,2]-self.cfg.target_base_height_m)/.035).square()-.035*power
            -.05*(self.actions-self.last_actions).square().mean(dim=1)-.08*self.filtered.square().mean(dim=1)
            -.4*(joints[:,:4]-self.ref[:,:4]).square().mean(dim=1)-100*self.fallen.float())*self.step_dt+.8*touchdown_score+3*balanced_round
        valid=self.history_valid
        delta=pos-self.previous_position
        self.sums['move_time']+=moving*self.step_dt
        self.sums['move_distance']+=delta[:,0]*moving*valid
        self.sums['command_integral']+=self.commands[:,0]*moving*self.step_dt
        self.sums['rate_limited']+=self.rate_limited.float().mean(dim=1)
        self.sums['work']+=power*self.step_dt
        self.sums['flight_time']+=(~contact.any(dim=1))*self.step_dt
        self.sums['flat_numerator']+=(planted&self.expected_stance).sum(dim=1)*moving
        self.sums['stance_contacts']+=stance_contact.sum(dim=1)*moving
        self.sums['tilt_degrees']+=(tilt*stance_contact).sum(dim=1)*moving*(180/math.pi)
        self.sums['slip_speed']+=(slip_speed*stance_contact).sum(dim=1)*moving
        self.sums['ankle_motion']+=qd[:,12:16].abs().mean(dim=1)*self.step_dt*moving
        self.sums['stance_expected']+=self.expected_stance.sum(dim=1)*moving
        self.sums['stance_observed']+=stance_contact.sum(dim=1)*moving
        self.sums['swing_expected']+=mid_swing.sum(dim=1)
        self.sums['swing_clear']+=swing_clear.sum(dim=1)
        self.sums['qualified_steps']+=valid_step.sum(dim=1)
        self.sums['stride_distance']+=(measured*valid_step).sum(dim=1)
        self.sums['stance_sweep']+=self.commands[:,0]*self.cfg.duty_factor/frequency*moving*self.step_dt
        self.contact_counts+=touchdown.float()
        self.last_diagnostics={k:v.detach().clone() for k,v in {
            'moving':moving,'contact':contact,'stance':self.expected_stance,'planted':planted,
            'tilt_deg':tilt*180/math.pi,'slip_m_s':slip_speed,'corner_span_m':span,
            'corner_low_m':low,'corner_high_m':high,'ankle_rad':joints[:,12:16],
            'ankle_velocity_rad_s':qd[:,12:16],'base_xyz':pos,'base_velocity':vel,
            'yaw_rad':yaw,'roll_pitch_proxy':torch.sqrt(posture),'fallen':self.fallen,
            'command':self.commands[:,0],'rate_limited':self.rate_limited.float().mean(dim=1),
            'qualified_touchdown':valid_step,'touchdown_stride_m':measured,
            'mid_swing':mid_swing,'swing_clear':swing_clear,
            'supported_qualified_touchdown':support_step,'supported_swing_clear':lift_supported,
            'normal_force_n':normal_force,'selected_leg':selected,'selected_load_fraction':selected_load,
            'preparing':self.preparing,'late_preparing':prep&(unload_progress>.5),
            'supported_unloaded':unloaded,'support_margin_m':margin,'other_support_count':other_count,
            'com_world_m':com,'reference_body_shift_m':self.body_shift,
            'reference_stance_sweep_m':self.commands[:,0]*self.cfg.duty_factor/frequency}.items()}
        self.previous_position.copy_(pos)
        self.previous_sole.copy_(sole)
        self.previous_contact.copy_(contact)
        self.history_valid[:]=True
        return reward

    def _reset_idx(self,env_ids):
        if env_ids is None: env_ids=torch.arange(self.num_envs,device=self.device)
        log={}
        if hasattr(self,'swing_peak') and len(env_ids):
            s=self.sums
            log={'Metrics/train_qualified_stride_m':(s['stride_distance'][env_ids]/s['qualified_steps'][env_ids].clamp(min=1)).mean().item(),
                 'Metrics/train_qualified_step_count':s['qualified_steps'][env_ids].mean().item(),
                 'Metrics/train_swing_clear_fraction':(s['swing_clear'][env_ids]/s['swing_expected'][env_ids].clamp(min=1)).mean().item(),
                 'Metrics/train_reference_stance_sweep_m':(s['stance_sweep'][env_ids]/s['move_time'][env_ids].clamp(min=.001)).mean().item()}
            log.update({'Metrics/train_min_supported_steps_per_leg':self.step_counts[env_ids].amin(1).mean().item(),
                'Metrics/train_rear_supported_steps':self.step_counts[env_ids][:,[1,3]].mean().item(),
                'Metrics/train_rear_supported_clear_fraction':(self.clear_samples[env_ids][:,[1,3]]/self.swing_samples[env_ids][:,[1,3]].clamp(min=1)).mean().item(),
                'Metrics/train_rear_supported_unload_fraction':(self.unload_success[env_ids][:,[1,3]]/self.unload_samples[env_ids][:,[1,3]].clamp(min=1)).mean().item()})
            if self.metrics_file and self.common_step_counter-self.transfer_log_step>=640:
                with self.metrics_file.open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'step':self.common_step_counter,**log})+'\n')
                self.transfer_log_step=self.common_step_counter
        super()._reset_idx(env_ids)
        if hasattr(self,'swing_peak'):
            self.swing_peak[env_ids]=0.
            self.touchdown_valid[env_ids]=False
            self.last_touchdown[env_ids]=0.
            for tensor in (self.step_counts,self.swing_samples,self.clear_samples,self.unload_samples,self.unload_success,self.body_shift):
                tensor[env_ids]=0.
            self.supported_air[env_ids]=False
            self.preparing[env_ids]=False
            self.extras.setdefault('log',{}).update(log)
