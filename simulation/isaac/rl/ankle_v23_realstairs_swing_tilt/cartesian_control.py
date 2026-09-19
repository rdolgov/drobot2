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
    # A quadratic request preserves fine control near zero but lets the
    # policy rotate an unloaded sole far enough for high-clearance IK poses.
    # Actual ankle/hip/knee position and torque limits are still enforced.
    pitch_action=env.filtered[:,12:16]
    tilt=pitch_action*pitch_action.abs()*math.radians(110.)
    angular_rate=(2.*(angle+tilt)).clamp(-.7,.7)
    requested=torch.cat((env.filtered[:,:12].reshape(env.num_envs,3,4).transpose(1,2)*.07,
                         (.05*angular_rate)[:,:,None]),dim=2)
    # Damping avoids extreme velocities near a straight-knee singularity.
    velocity=(jac.transpose(-1,-2)@torch.linalg.solve(
        jac@jac.transpose(-1,-2)+.015**2*env.ik_identity,requested[:,:,:,None])).squeeze(-1)
    velocity=velocity.transpose(1,2).reshape(env.num_envs,16)
    velocity=torch.maximum(torch.minimum(velocity,env.rates*env.rate_strength),-env.rates*env.rate_strength)
    desired=env.targets+velocity*env.step_dt
    bounds=env.robot.data.joint_pos_limits.torch[:,env.ids]*.995
    desired=torch.maximum(torch.minimum(desired,bounds[:,:,1]),bounds[:,:,0])
    delta=desired-env.targets
    limit=torch.minimum(env.rates*env.rate_strength*env.step_dt,torch.full_like(delta,math.radians(2.)))
    env.rate_limited=delta.abs()>limit
    env.target_delta=torch.maximum(torch.minimum(delta,limit),-limit)
    env.targets+=env.target_delta
