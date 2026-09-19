"""Full-height foot/tread progress with no prescribed order of foot motion."""
import json
import torch
from . import waypoint_guidance


def features(env):
    local,gap,surface,contact,planted,levels=env._geometry()
    sole=local.mean(2)
    next_level=(env.credited_levels+1).clamp(max=4)
    height=next_level.float()*.18
    edge=.34+(next_level.float()-1.)*.25
    lo=local[:,:,:,0].amin(2);hi=local[:,:,:,0].amax(2)
    bottom=local[:,:,:,2].amin(2)
    forces=env.contacts.data.net_forces_w.torch[:,env.foot_sensors,2].clamp(min=0.)
    quality=torch.exp(-(gap.abs().amax(2)/.025).square())*(forces/3.).clamp(0.,1.)
    quality*=torch.exp(-((surface.amax(2)-surface.amin(2))/.025).square())
    support=torch.stack([quality[:,env.triangles[i]].amin(1) for i in range(4)],1)
    margin=torch.stack([env._support_margin(sole,torch.full_like(env.selected_leg,i)) for i in range(4)],1)
    clearance=((bottom-(height-.18))/.195).clamp(0.,1.)
    approach=(1.-((edge+.008-lo).clamp(min=0.)/.30)).clamp(0.,1.)
    unload=(1.-forces/16.).clamp(0.,1.)
    landing=torch.exp(-((sole[:,:,2]-height)/.035).square())*quality
    # Keep crossing potential continuous: a fully cleared/advanced foot is
    # worth 6 before and after the edge latch, then gains 2 for actual contact.
    score=torch.where(env.over_edges,6.+2.*landing,3.*clearance+3.*clearance*approach)
    score*=env.credited_levels.lt(4)
    # No body-height bonus, fixed heading, or requirement to gather all feet
    # on one tread. Strict actual-contact success checks remain separate.
    potential=((.1+.9*support)*score).sum(1)
    # Summed linear unloading cancels when total load stays constant.
    # Reward the readiness of one freely chosen candidate instead. The other
    # three actual contacts gate it; no prescribed foot order or body pose.
    stability=.5+.5*(margin/.06).clamp(-1.,1.)
    readiness=support*(2.*unload+.5*stability)*env.credited_levels.lt(4)
    potential+=readiness.amax(1)
    root=env.robot.data.root_pos_w.torch-env.scene.env_origins
    potential+=3.*root[:,0].clamp(-.2,1.8)
    target_x=(edge+.008-lo).clamp(min=0.)-(hi-edge-.242).clamp(min=0.)
    delta=torch.stack((target_x,sole[:,:,1].clamp(-.55,.55)-sole[:,:,1],
                       torch.where(env.over_edges,height,height+.015)-sole[:,:,2]),2)
    stage=env.credited_levels.clamp(max=3)
    highwater=env.lift_highwater.gather(2,stage[:,:,None]).squeeze(2)
    obs=torch.cat((highwater.float()/6.,env.credited_levels.float()/4.,env.over_edges.float(),next_level.float()/4.,
                   delta.flatten(1),support,margin.clamp(-.2,.2)/.1),1)
    return dict(local=local,sole=sole,gap=gap,planted=planted,levels=levels,lo=lo,bottom=bottom,
                height=height,edge=edge,hi=hi,force=forces,quality=quality,support=support,margin=margin,stage=stage,highwater=highwater,
                clearance=clearance,delta=delta,potential=potential,obs=obs,next_level=next_level,readiness=readiness)


def reward(env,slip,power,new_goal,new_full):
    f=features(env)
    # These are intermediate heights of a foot climbing the same full riser,
    # not reduced stairs or near-goal resets. The history is per foot/tread,
    # so holding, lowering/re-lifting, or revisiting a credited tread cannot
    # pay these bonuses again within an episode.
    tier=((f['bottom']-(f['height']-.18)+1e-7)/.03).floor().long().clamp(0,6)
    supported=(f['support']>=.65)&(f['force']<1.)
    nearby=(f['hi']>=f['edge']-.15)&(f['lo']<=f['edge']+.24)
    inside=f['local'][:,:,:,1].abs().amax(2)<=.66
    eligible=supported&nearby&inside&env.credited_levels.lt(4)
    eligible&=(~env.fallen & (env.range_peak<=.02))[:,None]
    highwater=torch.where(eligible,torch.maximum(f['highwater'],tier),f['highwater'])
    lift_count=(highwater-f['highwater']).sum(1)
    lift_bonus=.25*lift_count
    env.lift_highwater.scatter_(2,f['stage'][:,:,None],highwater[:,:,None])
    env.over_edges |= (f['lo']>f['edge']+.004)&(f['bottom']>f['height']+.008)
    good=f['planted']&(f['levels']>=f['next_level'])&(slip<.025)
    good &= (~env.fallen & (env.range_peak<=.02))[:,None]
    env.landing_timers=torch.where(good,env.landing_timers+env.step_dt,0.)
    landed=(env.landing_timers>=.25)&(env.credited_levels<f['next_level'])
    env.credited_levels=torch.where(landed,f['next_level'],env.credited_levels)
    env.over_edges[landed]=False
    env.landing_timers[landed]=0.
    f=features(env)
    waypoint_guidance.update(env, f)
    guide_phi, guide_metrics = waypoint_guidance.potential(env, f)
    f['potential'] = f['potential'] + guide_phi
    terminal=env.fallen|env.success|env.curriculum_top
    next_phi=torch.where(terminal,0.,f['potential'])
    shaping=torch.where(torch.isfinite(env.reward_potential),env.cfg.shaping_gamma*next_phi-env.reward_potential,0.)
    env.reward_potential=f['potential'].detach().clone()
    q=env.robot.data.joint_pos.torch[:,env.ids]
    excess=torch.maximum(q-env.limits[:,:,1],env.limits[:,:,0]-q).clamp(min=0.)
    cost=.03+.003*power+.01*(env.actions-env.last_actions).square().mean(1)+.25*(excess/.05).square().mean(1)
    value=shaping+lift_bonus-cost*env.step_dt+8.*landed.sum(1)+20.*new_goal-12.*env.fallen+120.*new_full+120.*env.curriculum_top
    # Retain compatible diagnostic keys, now mean over all four free-order feet.
    metrics=dict(lift_milestone_count=lift_count.float(),lift_milestone_bonus=lift_bonus,candidate_readiness=f['readiness'].amax(1),selected_load=f['force'].mean(1),support_quality=f['support'].mean(1),
        support_margin_m=f['margin'].mean(1),selected_clearance=f['clearance'].mean(1),
        credited_level_sum=env.credited_levels.float().sum(1),waypoint_distance_m=f['delta'].norm(dim=2).mean(1),shaping=shaping)
    metrics.update(guide_metrics)
    for k,v in metrics.items():env.task_sums[k]+=v
    if env.common_step_counter%20==0:
        guided = env.reset_variant == 2
        snapshot = dict(step=env.common_step_counter, guided_count=int(guided.sum()), lifetime_peak=float(env.guide_lifetime_peak),
            guided_current_index=env.guide_index[guided].cpu().tolist(), guided_episode_peak=env.guide_peak[guided].cpu().tolist(),
            guided_done=int(env.guide_done.sum()), first_tread_events=env.curriculum_goal_event_count, full_climbs=env.full_event_count)
        env.metrics_file.with_suffix('.guide.json').write_text(json.dumps(snapshot)+'\n')
        with env.metrics_file.with_suffix('.progress.jsonl').open('a') as stream:
            stream.write(json.dumps({'step':env.common_step_counter,'reset_variant':env.reset_variant[:8].cpu().tolist(),
                'credited_levels':env.credited_levels[:8].cpu().tolist(),'over_edge':env.over_edges[:8].cpu().tolist(),
                'lift_highwater':env.lift_highwater[:8].cpu().tolist(),
                **{k:v[:8].cpu().tolist() for k,v in metrics.items()}})+'\n')
    return value
