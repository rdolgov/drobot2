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
    measured_force=env.contacts.data.net_forces_w.torch[:,env.foot_sensors]
    forces=measured_force[:,:,2].clamp(min=0.)
    # A downward or lateral collision is not an unloaded foot. Supporting
    # quality still uses upward force; candidate unloading uses magnitude.
    force_norm=measured_force.norm(dim=2)
    quality=torch.exp(-(gap.abs().amax(2)/.025).square())*(forces/3.).clamp(0.,1.)
    quality*=torch.exp(-((surface.amax(2)-surface.amin(2))/.025).square())
    # Grounded rear curriculum, or after the floor-start front-pair hold: allow a force-bearing toe
    # or corner to support weight transfer. Landing quality stays full-foot.
    # A nearby corner alone is insufficient: require upward measured force,
    # one horizontal surface, bounded penetration, and <=40 mm heel rise.
    bearing_phase=((env.reset_variant==0)&env.guide_done)|(env.reset_variant==2)
    nearest=gap.abs().amin(2)
    same_surface=(surface.amax(2)-surface.amin(2))<.003
    edge_geometry=torch.exp(-(nearest/.004).square())
    edge_geometry*=same_surface&(gap.amin(2)>=-.008)&(gap.amax(2)<=.04)&(nearest<.008)
    bearing_geometry=torch.maximum(planted.float(),edge_geometry*contact)
    edge_quality=edge_geometry*contact*(forces/3.).clamp(0.,1.)
    bearing=torch.where(bearing_phase[:,None],torch.maximum(quality,edge_quality),quality)
    support=torch.stack([bearing[:,env.triangles[i]].amin(1) for i in range(4)],1)
    # Estimate support locations from near-plane corners, rather than the
    # elevated sole center. This is shaping geometry, not measured corner forces.
    corner_weight=torch.exp(-(gap/.004).square())
    bearing_center=(local*corner_weight[:,:,:,None]).sum(2)/corner_weight.sum(2).clamp(min=1e-8)[:,:,None]
    bearing_center=torch.where((corner_weight.sum(2)>1e-8)[:,:,None],bearing_center,sole)
    support_points=torch.where(bearing_phase[:,None,None],bearing_center,sole)
    margin=torch.stack([env._support_margin(support_points,torch.full_like(env.selected_leg,i)) for i in range(4)],1)
    clearance=((bottom-(height-.18))/.195).clamp(0.,1.)
    approach=(1.-((edge+.008-lo).clamp(min=0.)/.30)).clamp(0.,1.)
    unload=(1.-force_norm/16.).clamp(0.,1.)
    landing=torch.exp(-((sole[:,:,2]-height)/.035).square())*quality
    # Keep crossing potential continuous: a fully cleared/advanced foot is
    # worth 6 before and after the edge latch, then gains 2 for actual contact.
    score=torch.where(env.over_edges,6.+2.*landing,3.*clearance+3.*clearance*approach)
    score*=env.credited_levels.lt(4)
    # No body-height bonus, fixed heading, or requirement to gather all feet
    # on one tread. Strict actual-contact success checks remain separate.
    potential=((.1+.9*support)*score).sum(1)
    # Landing geometry must remain informative if the foot missed the
    # 8 mm clearance latch. Reward all corners approaching a horizontal
    # target tread only when the entire patch fits its XY bounds and the
    # other three feet provide measured support. Contact credit is separate.
    contained=(lo>=edge+.004)&(hi<=edge+.246)&(local[:,:,:,1].abs().amax(2)<=.66)
    landing_geometry=torch.exp(-((local[:,:,:,2]-height[:,:,None])/.020).square().mean(2))*contained
    landing_geometry=landing_geometry*support*env.credited_levels.lt(4)
    landing_phi=4.*(env.credited_levels.float()+landing_geometry).sum(1)
    potential+=landing_phi
    # Reward a measured, modest load on the NEXT tread before flat-foot
    # credit. Offline force calculations admit about 2 N on a tilted front
    # foot; demanding 8 N failed. Preserve the flat landing potential above.
    # A riser contact or a foot near the old tread cannot qualify: every
    # traction corner must fit the next tread, whose horizontal surface is
    # independently checked. Other-three measured support is still required.
    next_gap=local[:,:,:,2]-height[:,:,None]
    next_surface=(surface-height[:,:,None]).abs().amax(2)<.003
    touchdown_geometry=torch.exp(-(next_gap.abs().amin(2)/.004).square())
    touchdown_geometry*=contained&next_surface&(next_gap.amin(2)>=-.008)&(next_gap.amax(2)<=.100)&(next_gap.abs().amin(2)<.008)
    prospective_load=touchdown_geometry*contact*(forces/2.).clamp(0.,1.)*support*env.credited_levels.lt(4)
    # Carry the potential into credited progress. gamma*Phi(next)-Phi(prev)
    # gives no repeated positive payment for holding or tapping a contact.
    prospective_phi=2.*(env.credited_levels.float()+prospective_load).sum(1)
    potential+=prospective_phi
    # Summed linear unloading cancels when total load stays constant.
    # Reward the readiness of one freely chosen candidate instead. The other
    # three actual contacts gate it; no prescribed foot order or body pose.
    stability=.5+.5*(margin/.06).clamp(-1.,1.)
    readiness=support*(2.*unload+.5*stability)*env.credited_levels.lt(4)
    potential+=readiness.amax(1)
    # After a real floor-start plant, encourage transfer onto higher feet
    # and unloading a freely chosen lagging foot. This is bounded potential,
    # not a repeatable per-step bonus and not a restriction on foot order.
    # A tilted next-tread contact is useful load support before flat-foot
    # credit. Recognize it only for transfer shaping, after the measured
    # grounded-curriculum first-stage hold. Never alter credited_levels.
    advanced=contained&next_surface&contact&(edge_geometry>0.)&env.credited_levels.lt(4)
    advanced&=((env.reset_variant==2)&env.goal_done&~env.fallen&(env.range_peak<=.02))[:,None]
    working_levels=torch.where(advanced,next_level,env.credited_levels)
    lowest=working_levels.amin(1,keepdim=True)
    lagging=working_levels.eq(lowest)&env.credited_levels.lt(4)
    raised=working_levels.gt(lowest)
    env.tilted_transfer_contacts=advanced
    env.transfer_working_levels=working_levels
    load_scale=.20*9.81*env.robot.data.body_mass.torch.sum(1)
    load_geometry=torch.where(bearing_phase[:,None],bearing_geometry,planted.float())*(levels>=env.credited_levels)
    # Sum measured valid supporting loads before saturation. A feasible
    # tripod can carry 3,17,14 N; requiring each foot to reach 20% of body
    # weight conflicts with the bounded motor allocation at these poses.
    # The candidate readiness still uses the weakest other contact, so
    # total force alone cannot replace the individual support checks.
    raised_force=(forces*load_geometry*raised).sum(1)
    raised_count=raised.sum(1).clamp(min=1)
    # The accepted touchdown can bear2 N; demanding20% body weight on
    # this uncredited contact would oppose the observed feasible maneuver.
    # Preserve the original aggregate scale for every credited support.
    transfer_scale=(torch.where(advanced,2.,load_scale[:,None])*raised).sum(1).clamp(min=1.)
    raised_load=(raised_force/transfer_scale).clamp(0.,1.)
    old_carrying=(forces/load_scale[:,None]).clamp(0.,1.)*load_geometry
    env.old_min_raised_load=torch.where(raised,old_carrying,1.).amin(1)
    env.raised_force_n=raised_force
    env.raised_count=raised.sum(1)
    transfer_active=(env.ground_eligible|(env.reset_variant==2))&raised.any(1)
    transfer_potential=3.*raised_load*(readiness*lagging).amax(1)*transfer_active
    potential+=transfer_potential
    env.load_transfer_potential=transfer_potential
    env.raised_load_fraction=raised_load*transfer_active
    env.lagging_readiness=(readiness*lagging).amax(1)*transfer_active
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
                height=height,edge=edge,hi=hi,force=forces,force_norm=force_norm,quality=quality,support=support,margin=margin,stage=stage,highwater=highwater,
                bearing_phase=bearing_phase,bearing=bearing,bearing_center=bearing_center,edge_geometry=edge_geometry,
                clearance=clearance,delta=delta,potential=potential,obs=obs,next_level=next_level,readiness=readiness,landing_geometry=landing_geometry,landing_phi=landing_phi,prospective_load=prospective_load,prospective_phi=prospective_phi)


def reward(env,slip,power,new_goal,new_full):
    f=features(env)
    # These are intermediate heights of a foot climbing the same full riser,
    # not reduced stairs or near-goal resets. The history is per foot/tread,
    # so holding, lowering/re-lifting, or revisiting a credited tread cannot
    # pay these bonuses again within an episode.
    tier=((f['bottom']-(f['height']-.18)+1e-7)/.03).floor().long().clamp(0,6)
    supported=(f['support']>=.65)&(f['force_norm']<1.)
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
    if landed.any():
        pairs=landed.nonzero()
        ids,legs=pairs[:,0],pairs[:,1]
        env.plant_event_count+=int(len(pairs))
        env.rear_plant_event_count+=int(((legs==1)|(legs==3)).sum())
        env.second_tread_event_count+=int((env.credited_levels[ids,legs]>=2).sum())
        record=dict(step=env.common_step_counter, count=env.plant_event_count,
            evidence='new tread credit during changing-policy training; not whole-robot stage or full-climb success',
            environments=ids.cpu().tolist(), legs=legs.cpu().tolist(),
            reset_variants=env.reset_variant[ids].cpu().tolist(),
            levels=env.credited_levels[ids,legs].cpu().tolist(),
            sole=f['sole'][ids,legs].cpu().tolist(), force=f['force'][ids,legs].cpu().tolist(),
            gaps=f['gap'][ids,legs].cpu().tolist(), guide_index=env.guide_index[ids].cpu().tolist())
        with env.metrics_file.with_suffix('.plant_events.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
    f=features(env)
    f['foot_slip']=slip
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
        guided = env.reset_variant != 1
        ids=f['bearing_phase'].nonzero().flatten()[:16]
        with env.metrics_file.with_suffix('.edge_support.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=env.common_step_counter,
                bearing_phase_count=int(f['bearing_phase'].sum()),environments=ids.cpu().tolist(),
                reset_variant=env.reset_variant[ids].cpu().tolist(),
                floor_pair_count=int(((env.reset_variant==0)&env.guide_done).sum()),
                rear_curriculum_count=int((env.reset_variant==2).sum()),
                force=f['force'][ids].cpu().tolist(),flat_quality=f['quality'][ids].cpu().tolist(),
                bearing_quality=f['bearing'][ids].cpu().tolist(),
                support_quality=f['support'][ids].cpu().tolist(),
                contact_center=f['bearing_center'][ids].cpu().tolist(),
                gap=f['gap'][ids].cpu().tolist(),credited_levels=env.credited_levels[ids].cpu().tolist()))+'\n')
        with env.metrics_file.with_suffix('.load_transfer.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=env.common_step_counter,
                reset_variant=env.reset_variant[:8].cpu().tolist(),
                credited_levels=env.credited_levels[:8].cpu().tolist(),
                forces=f['force'][:8].cpu().tolist(),
                transfer_potential=env.load_transfer_potential[:8].cpu().tolist(),
                raised_load_fraction=env.raised_load_fraction[:8].cpu().tolist(),
                old_min_raised_load=env.old_min_raised_load[:8].cpu().tolist(),
                raised_force_n=env.raised_force_n[:8].cpu().tolist(),
                raised_count=env.raised_count[:8].cpu().tolist(),
                lagging_readiness=env.lagging_readiness[:8].cpu().tolist()))+'\n')
        snapshot = dict(step=env.common_step_counter, front_pair_events=env.front_pair_event_count, plant_events=env.plant_event_count, guided_count=int(guided.sum()), lifetime_peak=float(env.guide_lifetime_peak),
            guided_environment_ids=guided.nonzero().flatten().cpu().tolist(), guided_variants=env.reset_variant[guided].cpu().tolist(),
            guided_current_index=env.guide_index[guided].cpu().tolist(), guided_episode_peak=env.guide_peak[guided].cpu().tolist(),
            selected_foot_force=f['force'].gather(1,env.guide_swing_legs[env.reset_variant,env.guide_index][:,None]).squeeze(1)[guided].cpu().tolist(),
            selected_foot_max_gap=f['gap'].abs().amax(2).gather(1,env.guide_swing_legs[env.reset_variant,env.guide_index][:,None]).squeeze(1)[guided].cpu().tolist(),
            guide_joint_rms=guide_metrics['guide_joint_rms'][guided].cpu().tolist(),
            guide_root_error=waypoint_guidance.errors(env,f)[2][guided].cpu().tolist(),
            guide_orientation_error=waypoint_guidance.errors(env,f)[3][guided].cpu().tolist(),
            guide_support_quality=f['support'].gather(1,env.guide_swing_legs[env.reset_variant,env.guide_index][:,None]).squeeze(1)[guided].cpu().tolist(),
            guide_swing_error=guide_metrics['guide_rear_error'][guided].cpu().tolist(),
            guide_handoff=env.guide_handoff[guided].cpu().tolist(), guided_done=int(env.guide_done.sum()), first_tread_events=env.curriculum_goal_event_count, full_climbs=env.full_event_count)
        front=((env.reset_variant==0)&(env.guide_index>0)).nonzero().flatten()
        if len(front):
            chosen=front[env.guide_index[front].argsort(descending=True)[:4]]
            states=dict(step=env.common_step_counter,environments=chosen.cpu().tolist(),
                index=env.guide_index[chosen].cpu().tolist(),
                origin=(env.plan_origin-env.scene.env_origins)[chosen].cpu().tolist(),
                root=(env.robot.data.root_pos_w.torch-env.scene.env_origins)[chosen].cpu().tolist(),
                quat=env.robot.data.root_quat_w.torch[chosen].cpu().tolist(),
                joints=env.robot.data.joint_pos.torch[chosen][:,env.ids].cpu().tolist(),
                targets=env.targets[chosen].cpu().tolist(),
                sole=f['sole'][chosen].cpu().tolist(),force=f['force'][chosen].cpu().tolist(),
                gap=f['gap'][chosen].cpu().tolist(),
                filtered_actions=env.filtered[chosen].cpu().tolist())
            snapshot['front_states']=states
            with env.metrics_file.with_suffix('.front_states.jsonl').open('a') as stream:
                stream.write(json.dumps(states)+'\n')
        rear_ids=(env.reset_variant==2).nonzero().flatten()
        if len(rear_ids):
            leaders=rear_ids[env.guide_index[rear_ids].argsort(descending=True)[:4]]
            highest=rear_ids[f['bottom'][rear_ids,3].argsort(descending=True)[:4]]
            selected=torch.unique(torch.cat((leaders,highest)))
            state=dict(step=env.common_step_counter,environments=selected.cpu().tolist(),
                index=env.guide_index[selected].cpu().tolist(),age_s=(env.episode_length_buf[selected]*env.step_dt).cpu().tolist(),
                root=(env.robot.data.root_pos_w.torch-env.scene.env_origins)[selected].cpu().tolist(),
                origin=(env.plan_origin-env.scene.env_origins)[selected].cpu().tolist(),
                quat=env.robot.data.root_quat_w.torch[selected].cpu().tolist(),
                joints=env.robot.data.joint_pos.torch[selected][:,env.ids].cpu().tolist(),
                targets=env.targets[selected].cpu().tolist(),strength=env.strength[selected].cpu().tolist(),
                velocity=env.robot.data.joint_vel.torch[selected][:,env.ids].cpu().tolist(),
                torques=env.robot.data.applied_torque.torch[selected][:,env.ids].cpu().tolist(),
                force_xyz=env.contacts.data.net_forces_w.torch[selected][:,env.foot_sensors].cpu().tolist(),
                sole=f['sole'][selected].cpu().tolist(),gap=f['gap'][selected].cpu().tolist(),
                corners=f['local'][selected].cpu().tolist(),filtered_actions=env.filtered[selected].cpu().tolist(),
                support=f['support'][selected].cpu().tolist(),transfer_working_levels=env.transfer_working_levels[selected].cpu().tolist(),tilted_transfer_contacts=env.tilted_transfer_contacts[selected].cpu().tolist(),over_edges=env.over_edges[selected].cpu().tolist(),
                guide_handoff=env.guide_handoff[selected].cpu().tolist(),
                assistance_weight=env.reference_assistance_weight[selected].cpu().tolist(),
                velocity_request_xyz_pitch=env.cartesian_velocity_request[selected].cpu().tolist(),
                credited_levels=env.credited_levels[selected].cpu().tolist(),
                landing_geometry=f['landing_geometry'][selected].cpu().tolist(),landing_phi=f['landing_phi'][selected].cpu().tolist(),
                prospective_load=f['prospective_load'][selected].cpu().tolist(),prospective_phi=f['prospective_phi'][selected].cpu().tolist())
            with env.metrics_file.with_suffix('.rear_states.jsonl').open('a') as stream:
                stream.write(json.dumps(state)+'\n')
        snapshot['prospective_load_mean']=f['prospective_load'].mean(0).cpu().tolist()
        snapshot['prospective_load_max']=f['prospective_load'].amax(0).cpu().tolist()
        snapshot['prospective_potential_mean']=float(f['prospective_phi'].mean())
        snapshot['second_tread_touchdown_events']=env.second_tread_touchdown_event_count
        snapshot['contact_loaded_but_upward_unloaded_feet']=int(((f['force']<1.)&(f['force_norm']>=1.)).sum())
        snapshot['free_swing_guide_blocks']=env.free_swing_guide_blocks
        snapshot['tilted_transfer_contacts']=int(env.tilted_transfer_contacts.sum())
        snapshot['tilted_transfer_potential_max']=float(env.load_transfer_potential.max())
        snapshot['second_tread_events']=env.second_tread_event_count
        snapshot['landing_geometry_mean']=f['landing_geometry'].mean(0).cpu().tolist()
        snapshot['landing_potential_mean']=float(f['landing_phi'].mean())
        env.metrics_file.with_suffix('.guide.json').write_text(json.dumps(snapshot)+'\n')
        with env.metrics_file.with_suffix('.progress.jsonl').open('a') as stream:
            stream.write(json.dumps({'step':env.common_step_counter,'reset_variant':env.reset_variant[:8].cpu().tolist(),
                'credited_levels':env.credited_levels[:8].cpu().tolist(),'over_edge':env.over_edges[:8].cpu().tolist(),
                'lift_highwater':env.lift_highwater[:8].cpu().tolist(),
                **{k:v[:8].cpu().tolist() for k,v in metrics.items()}})+'\n')
    return value
