"""Soft lateral-reflection prior on the actor, not mirrored reward samples.

The robot's mirrored URDF joint axes already use equal logical joint signs.
FL/RL/FR/RR becomes FR/RR/FL/RL. Swapping diagonals shifts the clock by pi.
Payload and actuator asymmetry may remain: use a small consistency loss only,
with real rollout rewards and unchanged physical qualification.
"""
import torch

LEG_MIRROR = [2, 3, 0, 1]
JOINT_MIRROR = [4*g+i for g in range(4) for i in LEG_MIRROR]


def mirror_policy(obs):
    mirrored = obs.clone()
    # Command is polar vx/vy plus axial yaw rate.
    mirrored[..., 1:3] *= -1.
    mirrored[..., 3:5] *= -1.  # sin/cos(pair*pi + within-placement phase)
    mirrored[..., 5] *= -1.   # axial roll rate
    mirrored[..., 7] *= -1.   # axial yaw rate
    mirrored[..., 9] *= -1.   # polar gravity y
    mirrored[..., 12] *= -1.  # polar accelerometer y
    for offset in (14, 30, 46, 62):
        mirrored[..., offset:offset+16] = obs[..., offset:offset+16][..., JOINT_MIRROR]
    mirrored[..., 78:82] = obs[..., 78:82][..., LEG_MIRROR]
    return mirrored


@torch.no_grad()
def symmetric_states(env, obs=None, actions=None):
    augmented = None
    if obs is not None:
        n = obs.batch_size[0]
        augmented = obs.repeat(2)
        augmented['policy'][n:] = mirror_policy(obs['policy'])
        if 'critic' in obs.keys():
            critic = obs['critic'].clone()
            critic[..., :82] = mirror_policy(critic[..., :82])
            critic[..., 83] *= -1.  # body linear velocity y
            critic[..., 86:90] = obs['critic'][..., 86:90][..., LEG_MIRROR]
            critic[..., 90:94] = obs['critic'][..., 90:94][..., LEG_MIRROR]
            critic[..., 95] *= -1.  # relative COM y
            critic[..., 97:101] = obs['critic'][..., 97:101][..., LEG_MIRROR]
            augmented['critic'][n:] = critic
    augmented_actions = None if actions is None else torch.cat((actions, actions[..., JOINT_MIRROR]), dim=0)
    return augmented, augmented_actions
