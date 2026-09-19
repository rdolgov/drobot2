"""Contact-driven full-height stair progress rewards for privileged RL teachers.

This module supplies rewards and observations, never external robot forces.
"""
import torch


def choose_leg(env, ids, sole):
    # Prefer a foot on the lowest not-yet-completed tread, then the nearest
    # forward foot. Selection persists until a qualified landing occurs.
    levels = env.credited_levels[ids]
    priority = levels.float()*10. - sole[ids, :, 0]
    selected = priority.argmin(1)
    env.selected_leg[ids] = selected
    env.next_level[ids] = (levels.gather(1, selected[:, None]).squeeze(1)+1).clamp(max=4)
    env.over_edge[ids] = False
    env.landing_timer[ids] = 0.


def features(env, geometry=None):
    local, gap, surface, contact, planted, levels = geometry or env._geometry()
    sole = local.mean(2)
    ids, leg = env.index, env.selected_leg
    selected = sole[ids, leg]
    corners = local[ids, leg]
    height = env.next_level.float()*env.cfg.stair_rise_m
    edge = env.cfg.stair_start_m+(env.next_level.float()-1)*env.cfg.stair_tread_m
    # The entire 94 x 54 mm traction patch must clear the riser. Its current
    # projected half-width sets the target, instead of an arbitrary fixed y.
    half_x = (corners[:,:,0]-selected[:,None,0]).abs().amax(1)
    low = edge+half_x+.008
    high = edge+env.cfg.stair_tread_m-half_x-.008
    goal_x = torch.maximum(low, torch.minimum(selected[:,0], high))
    goal_y = selected[:,1].clamp(-.55,.55)
    target_z = torch.where(env.over_edge, height, height+.025)
    delta = torch.stack((goal_x-selected[:,0], goal_y-selected[:,1], target_z-selected[:,2]), 1)
    forces = env.contacts.data.net_forces_w.torch[:,env.foot_sensors,2].clamp(min=0.)
    # Smooth reward quality. The binary, strict success checks are unchanged.
    quality = torch.exp(-(gap.abs().amax(2)/.025).square())
    quality *= (forces/3.).clamp(0.,1.)
    quality *= torch.exp(-((surface.amax(2)-surface.amin(2))/.025).square())
    other = env.triangles[leg]
    support = quality[ids[:,None],other].amin(1)
    margin = env._support_margin(sole,leg)
    bottom = corners[:,:,2].amin(1)
    clearance = ((bottom-(height-env.cfg.stair_rise_m)) / (env.cfg.stair_rise_m+.025)).clamp(0.,1.)
    unloaded = (1.-forces[ids,leg]/16.).clamp(0.,1.)
    # Approach the riser only after clearing it; then descend and carry load.
    approach = 1.-(delta[:,0].abs()/.30).clamp(0.,1.)
    landing = torch.exp(-((selected[:,2]-height)/.035).square())*quality[ids,leg]
    foot_score = torch.where(env.over_edge, 4.+4.*landing, 3.*clearance+3.*clearance*approach+unloaded)
    root = env.robot.data.root_pos_w.torch-env.scene.env_origins
    potential = 3.*root[:,0].clamp(-.2,1.8)+(.2+.8*support)*foot_score
    # Losing contact must not improve the instantaneous support potential.
    potential += env.cfg.support_weight*support*(.5+.5*(margin/.06).clamp(-1.,1.))
    obs = torch.cat((torch.nn.functional.one_hot(leg,4).float(),
                     env.credited_levels.float()/4., env.next_level[:,None].float()/4.,
                     env.over_edge[:,None].float(), margin[:,None].clamp(-.2,.2)/.1,
                     support[:,None]),1)
    return dict(potential=potential, obs=obs, delta=delta, sole=sole,
                support=support, margin=margin, clearance=clearance,
                load=forces[ids,leg], gap=gap, planted=planted, levels=levels,
                bottom=bottom, edge=edge, height=height, corners=corners)


def advance(env, slip):
    f = features(env)
    ids, leg = env.index, env.selected_leg
    crossed = (f['corners'][:,:,0].amin(1)>f['edge']+.004) & (f['bottom']>f['height']+.012)
    env.over_edge |= crossed
    good = f['planted'][ids,leg] & (f['levels'][ids,leg]>=env.next_level)
    good &= (slip[ids,leg]<.025) & ~env.fallen & (env.range_peak<=.02)
    env.landing_timer = torch.where(good,env.landing_timer+env.step_dt,0.)
    landed = (env.landing_timer>=.25) & (env.credited_levels[ids,leg]<env.next_level)
    selected_ids = landed.nonzero().flatten()
    if len(selected_ids):
        env.credited_levels[selected_ids,leg[selected_ids]] = env.next_level[selected_ids]
        choose_leg(env,selected_ids,f['sole'])
    return landed, features(env)


def reward(env, slip, power, new_goal, new_full):
    landed, f = advance(env,slip)
    terminal = env.fallen | env.success | env.curriculum_top
    # Discount-consistent shaping; absorbing terminal states have Phi=0.
    next_phi = torch.where(terminal,0.,f['potential'])
    shaping = torch.where(torch.isfinite(env.reward_potential),
                          env.cfg.shaping_gamma*next_phi-env.reward_potential,0.)
    env.reward_potential = f['potential'].detach().clone()
    q = env.robot.data.joint_pos.torch[:,env.ids]
    excess = torch.maximum(q-env.limits[:,:,1],env.limits[:,:,0]-q).clamp(min=0.)
    cost = .03+.003*power+.01*(env.actions-env.last_actions).square().mean(1)
    cost += .25*(excess/.05).square().mean(1)
    # Landing credit is paid once per foot/tread. Hovering and re-touching the
    # same tread cannot collect this bonus. Fall and full-climb criteria stay.
    value = shaping-cost*env.step_dt+8.*landed+20.*new_goal-12.*env.fallen
    value += 120.*new_full+120.*env.curriculum_top
    metrics = dict(selected_load=f['load'], support_quality=f['support'],
                   support_margin_m=f['margin'], selected_clearance=f['clearance'],
                   credited_level_sum=env.credited_levels.float().sum(1),
                   waypoint_distance_m=f['delta'].norm(dim=1), shaping=shaping)
    for key, v in metrics.items():
        env.task_sums[key] += v
    if env.common_step_counter % 60 == 0:
        import json
        with env.metrics_file.with_suffix('.progress.jsonl').open('a') as stream:
            stream.write(json.dumps({'step':env.common_step_counter,
                'selected_leg':env.selected_leg[:8].cpu().tolist(),
                'reset_variant':env.reset_variant[:8].cpu().tolist(),
                'credited_levels':env.credited_levels[:8].cpu().tolist(),
                'over_edge':env.over_edge[:8].cpu().tolist(),
                **{k:v[:8].cpu().tolist() for k,v in metrics.items()}})+'\n')
    return value
