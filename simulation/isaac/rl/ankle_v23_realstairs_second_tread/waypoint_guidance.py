"""State-driven soft pose guidance; never writes actuators or simulated state."""
import json
from pathlib import Path
import torch


def initialize(env):
    data = json.loads(Path(__file__).with_name('waypoints.json').read_text())
    for key in ('joints', 'servo_targets', 'feet', 'base', 'quat', 'pitch_bias'):
        setattr(env, 'guide_' + key, torch.tensor(data[key], device=env.device))
    env.guide_counts = torch.tensor(data['counts'], device=env.device)
    env.guide_swing_legs = torch.tensor(data['waypoint_swing_legs'], device=env.device)
    env.guide_required_credits = torch.tensor(data['required_credits'], device=env.device)
    env.guide_front_hold = torch.zeros(env.num_envs, device=env.device)
    env.guide_handoff = torch.zeros(env.num_envs, device=env.device)
    env.front_pair_event_count = 0
    env.rear_plant_event_count = 0
    env.second_tread_event_count = 0
    env.first_tread_gate_index=int(data['first_tread_gate_index'])
    env.second_tread_touchdown_hold=torch.zeros(env.num_envs,device=env.device)
    env.second_tread_touchdown_event_count=0
    env.guide_index = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env.guide_hold = torch.zeros(env.num_envs, device=env.device)
    env.guide_done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.guide_peak = torch.zeros(env.num_envs, device=env.device)
    env.guide_lifetime_peak = torch.zeros((), device=env.device)
    for key in ('guide_fraction', 'guide_joint_rms', 'guide_rear_error', 'guide_potential'):
        env.task_sums[key] = torch.zeros(env.num_envs, device=env.device)


def reset(env, ids):
    env.guide_index[ids] = 0
    env.guide_hold[ids] = 0.
    env.guide_front_hold[ids] = 0.
    env.guide_handoff[ids] = 0.
    env.second_tread_touchdown_hold[ids] = 0.
    env.guide_done[ids] = False
    env.guide_peak[ids] = 0.


def targets(env):
    start = env.plan_origin - env.scene.env_origins
    i = (env.reset_variant, env.guide_index)
    feet = env.guide_feet[i].clone()
    feet[:, :, :2] += start[:, None, :2]
    base = env.guide_base[i].clone()
    base[:, :2] += start[:, :2]
    return env.guide_joints[i], feet, base, env.guide_quat[i]


def errors(env, f):
    q, feet, base, quat = targets(env)
    root = env.robot.data.root_pos_w.torch - env.scene.env_origins
    joint_rms = (q - env.robot.data.joint_pos.torch[:, env.ids]).square().mean(1).sqrt()
    selected = (torch.arange(env.num_envs, device=env.device), env.guide_swing_legs[env.reset_variant, env.guide_index])
    rear_error = (feet[selected] - f['sole'][selected]).norm(dim=1)
    root_error = (base - root).norm(dim=1)
    alignment = (quat * env.robot.data.root_quat_w.torch).sum(1).abs().clamp(0., 1.)
    angle = 2. * torch.acos(alignment)
    return joint_rms, rear_error, root_error, angle


def update(env, f):
    enabled = env.reset_variant != 1
    joint, rear, root, angle = errors(env, f)
    support = f['support'].gather(1, env.guide_swing_legs[env.reset_variant, env.guide_index][:, None]).squeeze(1)
    near = ((joint < .10) & (rear < .025) & (root < .04) & (angle < .14)
            & (support >= .55) & ~env.fallen & (env.range_peak <= .02))
    # Advance only from measured state, one waypoint per 0.1 s of support.
    # No elapsed-time trajectory, action injection, or reference-state reset.
    required = env.guide_required_credits[env.reset_variant, env.guide_index]
    near &= (env.credited_levels >= required).all(1)
    # The second-tread route cannot start until the original measured
    # all-four first-tread one-second hold has occurred in this episode.
    near &= (env.reset_variant!=2)|(env.guide_index<env.first_tread_gate_index)|env.goal_done
    env.guide_hold = torch.where(enabled & ~env.guide_done & near,
                                 env.guide_hold + env.step_dt, 0.)
    advance = (env.guide_hold >= .1) & (env.guide_index < env.guide_counts[env.reset_variant]-1)
    env.guide_index += advance.long()
    env.guide_hold[advance] = 0.
    pair_good = ((env.reset_variant == 0) & f['planted'][:, [0, 2]].all(1)
        & (f['levels'][:, [0, 2]] >= 1).all(1) & (f['foot_slip'][:, [0, 2]] < .025).all(1)
        & ~env.fallen & (env.range_peak <= .02)
        & (env.robot.data.root_lin_vel_w.torch.norm(dim=1) < .08))
    env.guide_front_hold = torch.where(pair_good, env.guide_front_hold + env.step_dt, 0.)
    front_complete = (env.guide_front_hold >= .25) & (env.credited_levels[:, [0, 2]] >= 1).all(1)
    new_pair = front_complete & ~env.guide_done
    if new_pair.any():
        ids = new_pair.nonzero().flatten()
        env.front_pair_event_count += int(len(ids))
        with env.metrics_file.with_suffix('.front_pair_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=env.common_step_counter,count=env.front_pair_event_count,
                environments=ids.cpu().tolist(),sole=f['sole'][ids].cpu().tolist(),
                force=f['force'][ids].cpu().tolist(),gaps=f['gap'][ids].cpu().tolist(),
                evidence='simultaneous front-foot contact hold during changing-policy floor-start training; not whole-robot stage or full climb'))+'\n')
    # Intermediate guided touchdown is distinct from flat foot credit.
    # Its physical contact/geometry is measured; no body state is injected.
    corners=f['local'][:,0];gap=corners[:,:,2]-.36
    inside=(corners[:,:,0].amin(1)>=.594)&(corners[:,:,0].amax(1)<=.836)&(corners[:,:,1].abs().amax(1)<=.66)
    touchdown=(env.reset_variant==2)&env.goal_done&(env.guide_index>=env.guide_counts[2]-1)&inside
    touchdown&=(gap.abs().amin(1)<.008)&(gap.amin(1)>=-.008)&(gap.amax(1)<=.040)
    touchdown&=(f['force'][:,0]>1.)&(f['support'][:,0]>=.65)&(f['foot_slip'][:,0]<.025)
    touchdown&=(env.robot.data.root_lin_vel_w.torch.norm(dim=1)<.08)&~env.fallen&(env.range_peak<=.02)
    env.second_tread_touchdown_hold=torch.where(touchdown,env.second_tread_touchdown_hold+env.step_dt,0.)
    second_complete=(env.second_tread_touchdown_hold>=.25)&(env.reset_variant==2)
    new_touchdown=second_complete&~env.guide_done
    if new_touchdown.any():
        ids=new_touchdown.nonzero().flatten()
        env.second_tread_touchdown_event_count+=int(len(ids))
        with env.metrics_file.with_suffix('.second_touchdown_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=env.common_step_counter,count=env.second_tread_touchdown_event_count,
                environments=ids.cpu().tolist(),force=f['force'][ids].cpu().tolist(),
                gaps=gap[ids].cpu().tolist(),credited_levels=env.credited_levels[ids].cpu().tolist(),
                evidence='actual loaded tilted FL touchdown on tread2 during grounded-curriculum training; NOT flat plant, floor-start transfer or full climb'))+'\n')
    completed = torch.where(env.reset_variant == 0, front_complete, second_complete)
    env.guide_done |= enabled & completed
    # Updated with measured completion before the next observation/action.
    # No body state, forces, or elapsed-time climbing trajectory is applied.
    env.guide_handoff = torch.where(env.guide_done,
        (env.guide_handoff + env.step_dt / 2.).clamp(max=1.), 0.)
    env.guide_index = torch.where(env.guide_done, env.guide_counts[env.reset_variant]-1, env.guide_index)
    fraction = env.guide_index.float() / (env.guide_counts[env.reset_variant]-1).clamp(min=1)
    env.guide_peak = torch.maximum(env.guide_peak, fraction * enabled)
    env.guide_lifetime_peak = torch.maximum(env.guide_lifetime_peak, env.guide_peak.max())


def potential(env, f):
    enabled = env.reset_variant != 1
    joint, rear, root, angle = errors(env, f)
    support = f['support'].gather(1, env.guide_swing_legs[env.reset_variant, env.guide_index][:, None]).squeeze(1)
    similarity = torch.exp(-.25 * ((joint/.16).square() + (rear/.02).square()
                                   + (root/.06).square() + (angle/.20).square()))
    fraction = env.guide_index.float() / (env.guide_counts[env.reset_variant]-1).clamp(min=1)
    phi = 16.*fraction + 2.*similarity*support
    # Keep completion credit in the potential after the real planted hold.
    # Terminal zeroing and gamma are applied by the ordinary reward module.
    phi = torch.where(env.guide_done, 18., phi) * enabled
    return phi, dict(guide_fraction=fraction*enabled, guide_joint_rms=joint*enabled,
                     guide_rear_error=rear*enabled, guide_potential=phi)
