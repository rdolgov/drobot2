"""Damped differential IK for RL foot velocities and swing-foot pitch.

Joint axes come from the retained V2.3 URDF. All commands ultimately become
bounded servo position targets; no body pose or external force is applied.
"""
import math
import torch
from ankle_v2.env import rotate


def initialize(env):
    legs=['front_left','rear_left','front_right','rear_right']
    env.ik_links=torch.tensor([[env.robot.body_names.index(leg+'_'+suffix) for suffix in
                              ['hip_link','proximal_link','distal_link','foot_link']] for leg in legs],device=env.device)
    # The joint axis is invariant under its own rotation, so the child-link
    # quaternion transforms the URDF axis directly into world coordinates.
    axes=torch.zeros((4,4,3),device=env.device)
    axes[:,:,2]=torch.tensor([[1,-1,-1,-1],[1,-1,-1,-1],[1,1,1,1],[1,1,1,1]],device=env.device)
    env.ik_axes=axes
    env.ik_identity=torch.eye(4,device=env.device).reshape(1,1,4,4)


def step(env,actions):
    env._set_commands()
    env.last_actions.copy_(env.actions)
    env.actions.copy_(actions.clamp(-1,1))
    env.filtered += env.step_dt/(env.cfg.filter_time_s+env.step_dt)*(env.actions-env.filtered)
    env._reference()
    joint_pos=env.robot.data.body_link_pos_w.torch[:,env.ik_links]
    joint_quat=env.robot.data.body_link_quat_w.torch[:,env.ik_links]
    axes=rotate(joint_quat,env.ik_axes)
    local,*_=env._geometry()
    sole=local.mean(2)+env.scene.env_origins[:,None,:]
    lever=sole[:,:,None,:]-joint_pos
    linear=torch.cross(axes,lever,dim=-1).transpose(-1,-2)
    pitch_axis=axes[:,:,1]
    angular=(axes*pitch_axis[:,:,None,:]).sum(-1)
    jac=torch.cat((linear,.05*angular[:,:,None,:]),dim=-2)
    normal=rotate(env.robot.data.body_link_quat_w.torch[:,env.feet],
                  torch.tensor([1.,0.,0.],device=env.device))
    down=torch.zeros_like(normal);down[:,:,2]=-1.
    angle=torch.atan2((pitch_axis*torch.cross(normal,down,dim=-1)).sum(-1),(normal*down).sum(-1))
    # Linear pitch commands expose larger rotations under the same
    # initial Beta actions. The filter, angular-rate clamp, joint-target/rate
    # guards, and physical motor/joint limits are retained.
    pitch_action=env.filtered[:,12:16]
    bias=env.guide_pitch_bias[env.reset_variant,env.guide_index]
    bias=torch.where((env.reset_variant != 1)[:,None],bias,0.)
    pitch_action=bias+torch.where(pitch_action>=0.,1.-bias,1.+bias)*pitch_action
    env.requested_pitch_degrees=pitch_action*110.
    tilt=pitch_action*math.radians(110.)
    angular_rate=(2.*(angle+tilt)).clamp(-.7,.7)
    requested=torch.cat((env.filtered[:,:12].reshape(env.num_envs,3,4).transpose(1,2)*.07,
                         (.05*angular_rate)[:,:,None]),dim=2)
    # State-driven reference assistance, not a clock-driven trajectory.
    # Servo setpoints already contain the bounded static PD load preloads.
    # Compare to current targets, not actual q, to avoid integrating the
    # gravity preload again. Physics and all existing motor guards still act.
    active=(env.reset_variant != 1)&~env.guide_done
    reference=env.guide_servo_targets[env.reset_variant,env.guide_index]
    nominal=(1.5*(reference-env.targets)).clamp(-.35,.35)
    nominal=nominal.reshape(env.num_envs,4,4).transpose(1,2)
    feedforward=(jac@nominal[:,:,:,None]).squeeze(-1)
    residual=torch.cat((env.filtered[:,:12].reshape(env.num_envs,3,4).transpose(1,2)*.025,
                        (.05*.35*env.filtered[:,12:16])[:,:,None]),dim=2)
    assisted=feedforward+residual
    cap=torch.tensor([.07,.07,.07,.035],device=env.device)
    assisted=torch.maximum(torch.minimum(assisted,cap),-cap)
    requested=torch.where(active[:,None,None],assisted,requested)
    env.reference_assistance_active=active
    env.nominal_joint_velocity=nominal.transpose(1,2).reshape(env.num_envs,16)*active[:,None]
    env.cartesian_velocity_request=requested.clone()
    env.cartesian_velocity_request[:,:,3]/=.05
    # Solve min ||J v - requested||^2 + damping^2 ||v||^2 with the
    # existing per-joint target-rate and position guards INSIDE the solve.
    # Independent clipping after an unconstrained solve can turn an upward
    # request into a large forward motion at the stair riser.
    hessian=jac.transpose(-1,-2)@jac+.015**2*env.ik_identity
    linear=(jac.transpose(-1,-2)@requested[:,:,:,None]).squeeze(-1)
    unconstrained=torch.linalg.solve(hessian,linear[:,:,:,None]).squeeze(-1)
    targets=env.targets.reshape(env.num_envs,4,4).transpose(1,2)
    bounds=env.robot.data.joint_pos_limits.torch[:,env.ids]*.995
    lower_pos=bounds[:,:,0].reshape(env.num_envs,4,4).transpose(1,2)
    upper_pos=bounds[:,:,1].reshape(env.num_envs,4,4).transpose(1,2)
    rate=(env.rates*env.rate_strength).reshape(env.num_envs,4,4).transpose(1,2)
    rate=torch.minimum(rate,torch.full_like(rate,math.radians(2.)/env.step_dt))
    lower=torch.maximum(-rate,(lower_pos-targets)/env.step_dt)
    upper=torch.minimum(rate,(upper_pos-targets)/env.step_dt)
    velocity=torch.maximum(torch.minimum(unconstrained,upper),lower)
    # Four-variable convex box least squares. A clipped unconstrained start
    # is already optimal when no guard binds. Coordinate minimization then
    # corrects the remaining joints when a guard is active.
    for _ in range(32):
        for joint in range(4):
            residual=linear[:,:,joint]-(hessian[:,:,joint,:]*velocity).sum(-1)
            proposal=velocity[:,:,joint]+residual/hessian[:,:,joint,joint]
            velocity[:,:,joint]=torch.maximum(torch.minimum(proposal,upper[:,:,joint]),lower[:,:,joint])
    limited=(unconstrained<lower)|(unconstrained>upper)
    env.rate_limited=limited.transpose(1,2).reshape(env.num_envs,16)
    desired=targets+velocity*env.step_dt
    desired=torch.maximum(torch.minimum(desired,upper_pos),lower_pos)
    desired=desired.transpose(1,2).reshape(env.num_envs,16)
    env.target_delta=desired-env.targets
    env.targets.copy_(desired)
