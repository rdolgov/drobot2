"""Explicitly requested offline replay only. Never opens hardware or Isaac.

Every invocation preserves a new report, including failures. Run with the local
Kit Python executable (not SimulationApp); ORT may live in --dependency-path.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dependency-path', type=Path, default=ROOT/'outputs/v44-study35-offline-deps')
args = parser.parse_args()
sys.path[:0] = [str(args.dependency_path), str(ROOT/'onboard/policy-runtime/src')]
import numpy as np
import torch
from torch import nn
import onnxruntime as ort
from drobot_policy_runtime.policy import OnnxWalkingPolicy, load_policy_metadata, _validate_policy_metadata
from drobot_policy_runtime.runtime import WalkingPolicyLoop, PolicyCommand, AppliedMotorTarget
from drobot_policy_runtime.sources import ImuSample, JointStateSample
from drobot_policy_runtime.v44_study35 import Study35Mapper, validate_metadata, CHECKPOINT_SHA256, HIGH_SPEED
from drobot_policy_runtime._v44_geometry import GEOMETRY
from drobot_policy_runtime._v44_projection import build_domain

STUDY = ROOT/'simulation/reviews/v44-actuator-20260905'
BUNDLE = ROOT/'onboard/models/experimental-v44-study35'
TRACE = STUDY/'front-stance35-v1/treatment/trace.jsonl'
ACTION_TOL, TARGET_TOL = 3.e-6, 2.e-6
REPORT = {'schema':'drobot.v44-study35-offline-check.v1', 'deployment_allowed':False,
          'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
                         'torch':torch.__version__,'onnxruntime':ort.__version__, 'device':'CPU'},
          'checks':{}, 'errors':[]}
OUT = STUDY/'offline-pi-integration'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUT.mkdir(parents=True, exist_ok=False)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def measure(name, actual, expected, tolerance, row=None):
    a,b = np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f'{name}: shape/nonfinite {a.shape} {b.shape}')
    error = float(np.max(np.abs(a-b)))
    record = REPORT['checks'].setdefault(name,{'samples':0,'max_abs_error':0.,'tolerance':tolerance,'passed':True})
    record['samples'] += 1
    if error > record['max_abs_error']:
        record.update(max_abs_error=error, worst_row=row)
    record['passed'] &= error <= tolerance

def reject(name, fn):
    try:
        fn()
    except (ValueError,RuntimeError,FileNotFoundError,AssertionError) as exc:
        REPORT['checks'][name] = {'passed':True,'rejection':str(exc)[:240]}
    else:
        REPORT['checks'][name] = {'passed':False,'reason':'Accepted invalid input'}

def section(name, fn):
    try:
        fn()
    except Exception:
        REPORT['errors'].append({'section':name,'traceback':traceback.format_exc()})
    print(json.dumps({'section_complete':name,'errors':len(REPORT['errors'])}),flush=True)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(50,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,24))
    def forward(self, obs):
        raw=self.mlp(obs).reshape(-1,2,12)
        alpha=nn.functional.softplus(raw[:,0,:])+1.
        beta=nn.functional.softplus(raw[:,1,:])+1.
        return 2.*alpha/(alpha+beta)-1.

def inputs(r):
    front=r['rear_pitch_lift']['front_stance_feedback']
    stop=front['stop_elapsed_s'][0]
    return dict(phase_cycles=r['actual_target_gait_phase_cycles'],age_s=r['pre_step_steps_since_reset']/60.,
                # Recorded yaw includes heading feedback; target authority uses
                # the nominal (zero) yaw request, not that observation channel.
                command=[r['command_applied_vx_vy_yawrate'][0],0.,0.],previous_delivered=front['previous_delivered_target_rad'][0],
                measured_q=r['pre_step_joint_position_rad'],elapsed_s=r['packet_elapsed_budget_s'],
                stop_elapsed_s=stop if stop>0. else None)

class OfflineSink:
    offline_only=True
    def __init__(self):
        self.ack=None
        self.targets=[]
        self.acknowledge=True
    def read_applied_target(self):
        return self.ack
    def write(self, action, target, now):
        self.targets.append((now,target.copy()))
        if self.acknowledge:
            self.ack=AppliedMotorTarget(target.copy(),now,len(self.targets))

class ReplaySources:
    def __init__(self,row):
        self.row=row
        self.now=0.
        self.heading=0.
    def imu(self):
        obs=np.array(self.row['actor_observation_before_normalizer'],dtype=np.float32)
        return ImuSample(obs[5:8],obs[8:11],obs[11:14]*np.float32(9.81),self.heading,self.now)
    def joints(self):
        return JointStateSample(np.array(self.row['pre_step_joint_position_rad'],dtype=np.float32),
                                np.array(self.row['pre_step_joint_velocity_rad_s'],dtype=np.float32),self.now)

def new_loop(source,sink,observer=None):
    return WalkingPolicyLoop(policy,SimpleNamespace(read=source.imu),SimpleNamespace(read=source.joints),sink,
                             command=PolicyCommand(HIGH_SPEED),step_observer=observer)

def actor_check():
    torch.set_num_threads(1)
    actor=Actor().eval()
    # Only our hash-verified local training artifact may be unpickled here.
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    actor.load_state_dict(checkpoint['actor_state_dict'],strict=True)
    with torch.inference_mode():
        for i,r in enumerate(rows):
            obs=np.array(r['actor_observation_before_normalizer'],dtype=np.float32)
            cpu=actor(torch.from_numpy(obs[None,:])).numpy()[0]
            onnx=policy.infer(obs)
            measure('actor.onnx_cpu_checkpoint',onnx,cpu,ACTION_TOL,i)
            measure('actor.onnx_saved_gpu',onnx,r['raw_actor_output'],ACTION_TOL,i)

def mapping_check():
    timings=[]
    active=0
    for i,r in enumerate(rows):
        kw=inputs(r)
        mapped=mapper.map(r['raw_actor_output'],**kw)
        front=r['rear_pitch_lift']['front_stance_feedback']
        rear=r['rear_pitch_lift']['tracking_feedback']
        expected={'raw':front['raw_actor_requested_target_rad'][0],
                  'reference_center':front['reference_center_rad'][0],
                  'front_corrected':front['corrected_target_rad'][0],
                  'rear_corrected':rear['corrected_pre_projection_target_rad'][0],
                  'projected':r['rear_pitch_projection']['projected_requested_target_rad'][0],
                  'requested':r['requested_target_rad'],'delivered':r['delivered_target_rad']}
        for key,value in expected.items():
            measure('mapping.'+key,mapped[key],value,TARGET_TOL,i)
        measure('mapping.contacts',mapped['scheduled_contact'],r['scheduled_contact'],0.,i)
        measure('mapping.lift_m',mapped['common_lift_m'],r['rear_pitch_projection']['common_lift_m'][0],1.e-9,i)
        active += sum(x is not None for x in mapped['domains'])
        t=time.perf_counter()
        mapper.map(policy.infer(np.array(r['actor_observation_before_normalizer'],dtype=np.float32)),**kw)
        timings.append((time.perf_counter()-t)*1000.)
    REPORT['timing_ms']={name:float(np.percentile(timings,p)) for name,p in [('p50',50),('p95',95),('p99',99),('max',100)]}
    REPORT['timing_ms'].update(samples=len(timings),scope='Local Windows CPU ONNX + mapping only, NOT Pi or complete I/O loop')
    REPORT['active_rear_domains']=active

def loop_replay():
    # Recorded exogenous IMU/encoder channels; recursively use our own bus-write
    # acknowledgements and phase integration, NOT golden delivered targets.
    previous_episode=None
    for i,r in enumerate(rows):
        if r['episode_index']!=previous_episode:
            source=ReplaySources(r);sink=OfflineSink();samples=[]
            loop=new_loop(source,sink,samples.append)
            previous_episode=r['episode_index']
        source.row=r
        source.now=r['pre_step_steps_since_reset']/60.
        if r['step_index']>0:
            x,y,z,w=rows[i-1]['root_quaternion_world_xyzw']
            source.heading=math.atan2(2.*(w*z+x*y),1.-2.*(y*y+z*z))
        loop.command=PolicyCommand(r['command_applied_vx_vy_yawrate'][0])
        # This step replay supplies the recorded stop trigger; the scheduler
        # itself is independently exercised by scheduler_check below.
        stop=r['rear_pitch_lift']['front_stance_feedback']['stop_elapsed_s'][0]
        loop._mapping_stop_start_s=source.now-stop if stop>0. else None
        with patch('drobot_policy_runtime.runtime.time.monotonic',lambda:source.now):
            loop.step(0.,source.now)
        s=samples[-1]
        measure('loop.phase_cycles',loop._phase_cycles,r['actual_target_gait_phase_cycles'],1.e-6,i)
        measure('loop.observation',s.observation,r['actor_observation_before_normalizer'],3.e-6,i)
        measure('loop.action',s.action,r['raw_actor_output'],ACTION_TOL,i)
        measure('loop.requested',s.requested_target_rad,r['requested_target_rad'],TARGET_TOL,i)
        measure('loop.delivered',s.rate_limited_target_rad,r['delivered_target_rad'],TARGET_TOL,i)

def rejection_check():
    reject('reject.default_load',lambda:OnnxWalkingPolicy(BUNDLE/'policy.onnx'))
    source=ReplaySources(rows[0])
    reject('reject.nonoffline_sink',lambda:new_loop(source,SimpleNamespace(write=lambda *a:None)))
    for label,mutate in [
        ('missing_marker',lambda m:m.pop('experimental_target_mapping')),
        ('wrong_gain',lambda m:m['experimental_target_mapping'].update(front_stance_gain=.5)),
        ('unknown_schema',lambda m:m['experimental_target_mapping'].update(schema='unknown')),
        ('wrong_checkpoint',lambda m:m.update(checkpoint_sha256='0'*64)),
        ('wrong_clock',lambda m:m['gait_clock'].update(frequency_max_hz=1.)),
        ('wrong_reference',lambda m:m['gait_reference']['joint_position_rad'][10].__setitem__(0,.3)),
    ]:
        altered=copy.deepcopy(policy.metadata);mutate(altered)
        reject('reject.'+label,lambda:_validate_policy_metadata(altered,model_path=BUNDLE/'policy.onnx',metadata_path=BUNDLE/'policy.json'))
    reject('reject.nan_observation',lambda:policy.infer(np.full(50,np.nan)))
    for key,value in [('elapsed_s',-1.),('phase_cycles',1.),('measured_q',[float('nan')]*12),('command',[0.,0.])]:
        kw=inputs(rows[0]);kw[key]=value
        reject('reject.'+key,lambda:mapper.map(rows[0]['raw_actor_output'],**kw))
    for label,command in [('reverse',[-.01,0.,0.]),('turn',[.01,0.,.1]),('lateral',[.01,.01,0.]),('too_fast',[.1,0.,0.])]:
        kw=inputs(rows[0]);kw['command']=command
        reject('reject.'+label,lambda:mapper.map(rows[0]['raw_actor_output'],**kw))
    reject('reject.noncenter_start',lambda:WalkingPolicyLoop(policy,object(),object(),OfflineSink(),initial_target_rad=np.zeros(12)))
    # Temporary assets are never a bus configuration or deployed directory.
    import shutil
    with tempfile.TemporaryDirectory(prefix='drobot-v44-offline-') as td:
        temp=Path(td)
        for path in BUNDLE.iterdir():
            if path.is_file(): shutil.copy2(path,temp/path.name)
        with (temp/'policy.onnx.data').open('r+b') as f:
            f.write(b'INVALID')
        reject('reject.external_weights_hash',lambda:load_policy_metadata(temp/'policy.onnx'))

def scheduler_check():
    source=ReplaySources(rows[0]);sink=OfflineSink();samples=[]
    loop=new_loop(source,sink,samples.append)
    def sleep(dt): source.now += dt
    with patch('drobot_policy_runtime.runtime.time.monotonic',lambda:source.now),patch('drobot_policy_runtime.runtime.time.sleep',sleep):
        loop.run(duration_s=2.,normal_stop_ramp_s=.9,normal_stop_hold_s=3.)
    commands=np.array([s.command.forward_m_s for s in samples])
    indices=np.flatnonzero(commands < HIGH_SPEED-1.e-8)
    assert len(indices)>0 and samples[indices[0]].elapsed_s>=2.
    assert samples[indices[0]].elapsed_s<=2.+.5+1./60.+1.e-8
    assert np.all(np.diff(commands[indices[0]:])<=1.e-10)
    assert commands[-1]==0. and loop.command.forward_m_s==0.
    measure('scheduler.hold_neutral',samples[-1].rate_limited_target_rad,mapper.neutral,1.e-7)
    REPORT['checks']['scheduler.normal_stop']={'passed':True,'samples':len(samples),
        'first_ramp_time_s':samples[indices[0]].elapsed_s,'last_time_s':samples[-1].elapsed_s}
    source=ReplaySources(rows[0]);sink=OfflineSink();sink.acknowledge=False;samples=[]
    loop=new_loop(source,sink,samples.append)
    with patch('drobot_policy_runtime.runtime.time.monotonic',lambda:source.now):
        for now in [0.,.01,.07,.27]:
            source.now=now
            loop.step(0.,now)
    for s in samples:
        measure('clock.unacknowledged_history',s.observation[38:],np.zeros(12),0.)
        measure('clock.unacknowledged_base',s.target_limit_base_rad,mapper.neutral,0.)
        assert np.all(np.abs(s.rate_limited_target_rad-mapper.neutral)<=np.minimum(mapper.rates*s.target_elapsed_s,math.radians(2.))+1.e-7)
    measure('clock.delayed_phase',loop._phase_cycles,.27*.35,1.e-7)
    REPORT['checks']['clock.packet_and_rate_caps']={'passed':True,'samples':len(samples)}
    # A deliberate slow local sink advances fake time; run() must skip missed
    # ticks rather than issue catch-up calls. There is no real sleep or motor.
    source=ReplaySources(rows[0]);sink=OfflineSink();samples=[]
    original=sink.write
    def slow_write(*a): original(*a);source.now+=.055
    sink.write=slow_write
    loop=new_loop(source,sink,samples.append)
    with patch('drobot_policy_runtime.runtime.time.monotonic',lambda:source.now),patch('drobot_policy_runtime.runtime.time.sleep',lambda dt:setattr(source,'now',source.now+dt)):
        loop.run(duration_s=.3)
    assert loop.missed_deadlines>0
    assert np.all(np.diff([s.monotonic_time_s for s in samples])>=.055)
    REPORT['checks']['clock.no_catchup_burst']={'passed':True,'steps':len(samples),'missed_deadlines':loop.missed_deadlines}
    fresh=new_loop(ReplaySources(rows[0]),OfflineSink())
    assert fresh._phase_cycles==0. and fresh._mapping_stop_start_s is None and fresh.step_count==0
    measure('clock.fresh_history',fresh.previous_target_rad,mapper.neutral,0.)

def scheduled_replay_check():
    for episode in (0,1):
        recording=[r for r in rows if r['episode_index']==episode]
        source=ReplaySources(recording[0]);sink=OfflineSink();samples=[]
        original_imu=source.imu
        def read_imu():
            index=round(source.now*60.)
            source.row=recording[index]
            if index:
                x,y,z,w=recording[index-1]['root_quaternion_world_xyzw']
                source.heading=math.atan2(2.*(w*z+x*y),1.-2.*(y*y+z*z))
            return original_imu()
        source.imu=read_imu
        loop=new_loop(source,sink,samples.append)
        with patch('drobot_policy_runtime.runtime.time.monotonic',lambda:source.now),patch('drobot_policy_runtime.runtime.time.sleep',lambda dt:setattr(source,'now',source.now+dt)):
            loop.run(duration_s=25.,normal_stop_ramp_s=.9,normal_stop_hold_s=3.)
        assert len(samples)==len(recording), (len(samples),len(recording))
        for i,(s,r) in enumerate(zip(samples,recording)):
            measure('scheduled.command',s.command.forward_m_s,r['command_applied_vx_vy_yawrate'][0],1.e-8,[episode,i])
            measure('scheduled.observation',s.observation,r['actor_observation_before_normalizer'],3.e-6,[episode,i])
            measure('scheduled.action',s.action,r['raw_actor_output'],ACTION_TOL,[episode,i])
            measure('scheduled.requested',s.requested_target_rad,r['requested_target_rad'],TARGET_TOL,[episode,i])
            measure('scheduled.delivered',s.rate_limited_target_rad,r['delivered_target_rad'],TARGET_TOL,[episode,i])

def projection_check():
    path=ROOT/'simulation/isaac/rl/parallel_walking/rear_pitch_projection.py'
    assert sha(path)=='e65883ead2a51ea5d3a35ea2e07f6d0f1d0dd80f1466796b5278484a60a78051'
    spec=importlib.util.spec_from_file_location('frozen_projection',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frames=[GEOMETRY['frames'][leg] for leg in ('rear_left','rear_right')]
    def rz(angle):
        c,s=torch.cos(angle),torch.sin(angle)
        z,o=torch.zeros_like(angle),torch.ones_like(angle)
        return torch.stack((c,-s,z,s,c,z,z,z,o),dim=-1).reshape(-1,3,3)
    def fault(reason,details):
        raise ValueError(reason)
    env=SimpleNamespace(_sim_step_counter=0,_rear_pitch_mapping_fault=fault,
        _rear_pitch_joint_ids=tuple(torch.tensor(ids,dtype=torch.int64) for ids in ((1,3),(5,7),(9,11))),
        _rear_pitch_abd_rotation=torch.tensor([f[0] for f in frames],dtype=torch.float64),
        _rear_pitch_hip_rotation=torch.tensor([f[1] for f in frames],dtype=torch.float64),
        _rear_pitch_sign=torch.tensor([-1.,1.],dtype=torch.float64),
        _rear_pitch_length=torch.tensor([f[3] for f in frames],dtype=torch.float64),_rear_pitch_rotation_z=rz)
    changed=identities=0
    # Fixed, deterministic cases around recorded contexts exercise both identity
    # and nonidentity projection, not new physics or a continuum certificate.
    contexts=[r for r in rows if max(r['rear_pitch_projection']['common_lift_m'][0])>.001][::90]
    for context_index,r in enumerate(contexts):
        mapped=mapper.map(r['raw_actor_output'],**inputs(r))
        center=mapped['reference_center'];lift=np.array(mapped['common_lift_m'],dtype=np.float64)
        domains={}
        for leg,(ai,hi,ki) in enumerate(((1,5,9),(3,7,11))):
            if lift[leg]>1.e-12:
                domains[leg]=build_domain(frames[leg],float(mapped['rear_corrected'][ai]),float(center[hi]),float(center[ki]),mapper.limits[hi],mapper.limits[ki],lift[leg])
        for trial,(dh,dk) in enumerate(((0.,0.),(.3,.3),(-.3,-.3),(.3,-.3),(-.3,.3))):
            requested=mapped['rear_corrected'].copy()
            requested[[5,7]]+=np.float32(dh);requested[[9,11]]+=np.float32(dk)
            expected=requested.copy()
            for leg,domain in domains.items():
                hi,ki=(5,9) if leg==0 else (7,11)
                pair,identity=domain.project(float(requested[hi]),float(requested[ki]))
                expected[hi],expected[ki]=pair
                identities+=identity;changed+=not identity
                for fraction in (0.,.2,.4,.6,.8,1.):
                    domain.assert_geometry(*pair,lift=lift[leg]*fraction)
            actual,diag=module.project_rear_pitch_targets(env,torch.from_numpy(requested[None]),torch.from_numpy(center[None]),
                torch.from_numpy(mapper.limits[None]),torch.from_numpy(lift[None]))
            measure('projection.scalar_frozen_torch',expected,actual.numpy()[0],TARGET_TOL,[context_index,trial])
            assert int(diag['changed_by_leg'].sum())==sum(not domains[leg].contains(float(requested[5 if leg==0 else 7]),float(requested[9 if leg==0 else 11])) for leg in domains)
    assert changed>0 and identities>0
    REPORT['checks']['projection.branches']={'passed':True,'identity_legs':identities,'changed_legs':changed,'contexts':len(contexts)}
    frame=frames[0]
    reject('reject.geometry_lift',lambda:build_domain(frame,0.,-.52,.52,(-1.,1.),(-1.5,1.5),.006))
    reject('reject.geometry_empty',lambda:build_domain(frame,0.,-.52,.52,(.9,1.),(.9,1.),.005))
    reject('reject.geometry_branch',lambda:build_domain(frame,0.,0.,.52,(-1.,1.),(-1.5,1.5),.005))
    reject('reject.geometry_nan',lambda:build_domain(frame,float('nan'),-.52,.52,(-1.,1.),(-1.5,1.5),.005))
    domain=next(iter(domains.values()))
    reject('reject.geometry_overlift',lambda:domain.assert_geometry(0.,0.,domain.lift+1.e-6))

try:
    manifest=json.loads((BUNDLE/'bundle-manifest.json').read_text())
    for path,digest in manifest['files'].items():
        assert sha(ROOT/path)==digest, f'Bundle/source modified: {path}'
    assert sha(TRACE)=='9ddf99ed1f1c91ca9fd6e3989c50d8f33c4b8ee6b433e7ae57660c3f276bd232'
    checkpoint_path=ROOT/manifest['checkpoint']
    assert sha(checkpoint_path)==CHECKPOINT_SHA256
    REPORT['bundle_manifest_sha256']=sha(BUNDLE/'bundle-manifest.json')
    REPORT['checker_sha256']=sha(__file__)
    REPORT['trace_sha256']=sha(TRACE)
    with TRACE.open() as f:
        header=json.loads(next(f));rows=[json.loads(line) for line in f]
    assert len(rows)==3528
    policy=OnnxWalkingPolicy(BUNDLE/'policy.onnx',allow_experimental_offline=True)
    mapper=policy.experimental_mapper
    for name,fn in [('actor',actor_check),('mapping',mapping_check),('loop_replay',loop_replay),('rejections',rejection_check),('scheduler',scheduler_check),('scheduled_replay',scheduled_replay_check),('projection',projection_check)]:
        section(name,fn)
except Exception:
    REPORT['errors'].append({'section':'setup','traceback':traceback.format_exc()})
REPORT['passed']=not REPORT['errors'] and all(x['passed'] for x in REPORT['checks'].values())
(OUT/'report.json').write_text(json.dumps(REPORT,indent=2)+'\n')
print(json.dumps({'report':str(OUT/'report.json'),'passed':REPORT['passed'],'errors':REPORT['errors'],
                  'failed_checks':{k:v for k,v in REPORT['checks'].items() if not v['passed']}}),flush=True)
sys.exit(0 if REPORT['passed'] else 1)
