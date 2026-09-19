"""State-driven soft pose guidance; never writes actuators or simulated state."""
import json
from pathlib import Path
import torch


def initialize(env):
    data = json.loads(Path(__file__).with_name('waypoints.json').read_text())
    for key in ('joints', 'servo_targets', 'feet', 'base', 'quat'):
        setattr(env, 'guide_' + key, torch.tensor(data[key], device=env.device))
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
    env.guide_done[ids] = False
    env.guide_peak[ids] = 0.


def targets(env):
    start = env.plan_origin - env.scene.env_origins
    i = env.guide_index
    feet = env.guide_feet[i].clone()
    feet[:, :, :2] += start[:, None, :2]
    base = env.guide_base[i].clone()
    base[:, :2] += start[:, :2]
    return env.guide_joints[i], feet, base, env.guide_quat[i]


def errors(env, f):
    q, feet, base, quat = targets(env)
    root = env.robot.data.root_pos_w.torch - env.scene.env_origins
    joint_rms = (q - env.robot.data.joint_pos.torch[:, env.ids]).square().mean(1).sqrt()
    rear_error = (feet[:, 3] - f['sole'][:, 3]).norm(dim=1)
    root_error = (base - root).norm(dim=1)
    alignment = (quat * env.robot.data.root_quat_w.torch).sum(1).abs().clamp(0., 1.)
    angle = 2. * torch.acos(alignment)
    return joint_rms, rear_error, root_error, angle


def update(env, f):
    enabled = env.reset_variant == 2
    joint, rear, root, angle = errors(env, f)
    support = f['quality'][:, :3].amin(1)
    near = ((joint < .10) & (rear < .025) & (root < .04) & (angle < .14)
            & (support >= .55) & ~env.fallen & (env.range_peak <= .02))
    # Advance only from measured state, one waypoint per 0.1 s of support.
    # No elapsed-time trajectory, action injection, or reference-state reset.
    env.guide_hold = torch.where(enabled & ~env.guide_done & near,
                                 env.guide_hold + env.step_dt, 0.)
    advance = (env.guide_hold >= .1) & (env.guide_index < len(env.guide_joints)-1)
    env.guide_index += advance.long()
    env.guide_hold[advance] = 0.
    env.guide_done |= enabled & env.goal_done
    env.guide_index = torch.where(env.guide_done, len(env.guide_joints)-1, env.guide_index)
    fraction = env.guide_index.float() / (len(env.guide_joints)-1)
    env.guide_peak = torch.maximum(env.guide_peak, fraction * enabled)
    env.guide_lifetime_peak = torch.maximum(env.guide_lifetime_peak, env.guide_peak.max())


def potential(env, f):
    enabled = env.reset_variant == 2
    joint, rear, root, angle = errors(env, f)
    support = f['quality'][:, :3].amin(1)
    similarity = torch.exp(-.25 * ((joint/.16).square() + (rear/.05).square()
                                   + (root/.06).square() + (angle/.20).square()))
    fraction = env.guide_index.float() / (len(env.guide_joints)-1)
    phi = 16.*fraction + 2.*similarity*support
    # Keep completion credit in the potential after the real planted hold.
    # Terminal zeroing and gamma are applied by the ordinary reward module.
    phi = torch.where(env.guide_done, 18., phi) * enabled
    return phi, dict(guide_fraction=fraction*enabled, guide_joint_rms=joint*enabled,
                     guide_rear_error=rear*enabled, guide_potential=phi)
