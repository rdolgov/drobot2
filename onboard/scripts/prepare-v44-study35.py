"""Package the selected study35 assets locally; never deploy or open hardware."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'onboard/policy-runtime/src'))
from drobot_policy_runtime.contract import ACTION_NAMES
from drobot_policy_runtime.policy import APPLIED_TARGET_OBSERVATION_ORDER
from drobot_policy_runtime.v44_study35 import (CHECKPOINT_SHA256,HIGH_SHA256,LOW_SHA256,LOW_SPEED,HIGH_SPEED,PROFILE,SCHEMA,validate_metadata)

STUDY = ROOT/'simulation/reviews/v44-actuator-20260905'
OUT = ROOT/'onboard/models/experimental-v44-study35'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def pinned(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Selected source changed: {path}')
    return Path(path)


def prepare():
    manifest_path = pinned(STUDY/'front-stance35-v1/prepared.json','fa184f94e519452a96cf9432cdc310024dcb6e725d5111a4a32c08b7bee4c9f7')
    source = json.loads(manifest_path.read_text())
    checkpoint = pinned(source['roles']['checkpoint']['path'],CHECKPOINT_SHA256)
    # Bind the selected physical/controller configuration, not the subsequently
    # edited experiment environment. In particular it evaluated heading hold
    # gain .75 and maximum correction .12; disabling that is not study35.
    selected_env = pinned(checkpoint.parent/'params/env.yaml',
        'ff61d9e550323c26516db168465fe2a3d7d57a09bcf153b5b725853cec531fce')
    refs = {}
    for name,digest in [('high',HIGH_SHA256),('low',LOW_SHA256)]:
        path = pinned(source['roles'][name+'_reference']['path'],digest)
        refs[name] = (path,json.loads(path.read_text()))
        if tuple(refs[name][1]['joint_order']) != ACTION_NAMES:
            raise ValueError('Reference order mismatch')
    video_dir = STUDY/'front-stance35-high-video-v1'
    receipt = json.loads((video_dir/'prepared.json').read_text())
    if receipt['binding']['sha256'] != 'd65c778871d6c38537eeaf33e892303bc6d84ddeb0f9ee64371a2a24b880ac39':
        raise ValueError('Unexpected selected video binding')
    pinned(video_dir/'review/model_396.pt',CHECKPOINT_SHA256)
    if json.loads((video_dir/'completion.json').read_text())['failure'] is not None:
        raise ValueError('Selected recording did not complete')
    OUT.mkdir(exist_ok=True)
    # Copied assets are NOT treated as verified inference until the separate
    # offline checker compares them with the bound checkpoint and saved actions.
    for name in ('policy.onnx','policy.onnx.data'):
        shutil.copy2(video_dir/'review/exported'/name,OUT/name)
    for name,(path,_) in refs.items():
        shutil.copy2(path,OUT/(name+'.json'))
    trace = pinned(STUDY/'front-stance35-v1/treatment/trace.jsonl',
        '9ddf99ed1f1c91ca9fd6e3989c50d8f33c4b8ee6b433e7ae57660c3f276bd232')
    with trace.open() as f:
        header = json.loads(f.readline())
    metadata = {
        'training_profile':PROFILE, 'checkpoint_sha256':CHECKPOINT_SHA256,
        'onnx_sha256':sha(OUT/'policy.onnx'),
        'onnx_external_data_sha256':{'policy.onnx.data':sha(OUT/'policy.onnx.data')},
        'observation_count':50,'action_count':12,'action_order':list(ACTION_NAMES),
        'observation_order':list(APPLIED_TARGET_OBSERVATION_ORDER),
        'observation_scales':{'command_forward_lateral_yaw':[10.,10.,10.]},
        'action_history_mode':'applied_target',
        'heading_hold':{'enabled':True,'mode':'relative_yaw_feedback','kp_s':.75,'max_correction_rad_s':.12},
        'gait_clock':{'mode':'speed_scaled','period_s':1/.35,'standstill_deadband_m_s':.004,
            'speed_min_m_s':LOW_SPEED,'speed_max_m_s':HIGH_SPEED,'frequency_min_hz':.3,'frequency_max_hz':.35,'stride_scale_min':1.},
        'joint_target_contract':{
            'neutral_joint_position_rad':header['actor_input_output_contract']['default_joint_position_rad'],
            'action_scale_rad':[.12]*4+[.3]*4+[.4]*4,
            'target_velocity_limit_rad_s_by_action':list(header['controller_measurement_contract']['nominal_target_rate_limit_rad_s_by_joint'].values()),
            'max_target_step_rad':header['controller_measurement_contract']['target_packet_step_limit_rad']},
        'action_contract':{'mode':'gait_residual','residual_scale_by_action':[.06]*4+[.5]*8,'zero_command_residual_scale':0.},
        'gait_reference':{'start_ramp_s':1.5,'joint_position_rad':refs['high'][1]['joint_position_rad'],
            'low_speed_joint_position_rad':refs['low'][1]['joint_position_rad'],
            'blend_speed_min_m_s':LOW_SPEED,'blend_speed_max_m_s':HIGH_SPEED},
        'run_stop_contract':{'recommended_stop_ramp_s':.9,'recommended_stopped_hold_s':3.,'ramp_shape':'cubic_smoothstep',
            'phase_alignment':{'enabled':True,'ramp_start_phase_cycles':.05,'phase_tolerance_cycles':.03,'maximum_wait_s':.5}},
        'experimental_target_mapping':{'schema':SCHEMA,'offline_only':True,'production_deployment_allowed':False,
            'front_stance_gain':.25,'rear_swing_gain':.25,'rear_projection':'common-F5','rear_lift_m':.005,'plant_response_filter_on_pi':False},
        'qualification':{'status':'UNQUALIFIED_OFFLINE_CANDIDATE','hardware_verified':False,'deployment_allowed':False,
            'evaluated_forward_command_m_s':HIGH_SPEED,'nominal_actual_speed_m_s':.015411198422089947,'canonical_gates_passed':28,'canonical_gates_total':30},
    }
    validate_metadata(metadata)
    (OUT/'policy.json').write_text(json.dumps(metadata,indent=2)+'\n')
    code = ROOT/'onboard/policy-runtime/src/drobot_policy_runtime'
    paths = [OUT/n for n in ('policy.onnx','policy.onnx.data','policy.json','high.json','low.json')]
    paths += [code/n for n in ('v44_study35.py','_v44_projection.py','_v44_geometry.py','policy.py','runtime.py','contract.py','sources.py','__init__.py')]
    paths += [Path(__file__).resolve()]
    evidence = {'schema':'drobot.v44-study35-offline-bundle.v1','deployment_allowed':False,
        'source_study_manifest_sha256':sha(manifest_path),'checkpoint':str(checkpoint.relative_to(ROOT)).replace('\\','/'),
        'checkpoint_sha256':CHECKPOINT_SHA256,
        'selected_env_sha256':sha(selected_env),'canonical_trace_sha256':sha(trace),
        'files':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths},
        'note':'Preparation is not verification; see offline check report. Original study qualification/export guards are unchanged.'}
    (OUT/'bundle-manifest.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'bundle':str(OUT),'manifest_sha256':sha(OUT/'bundle-manifest.json'),'deployment_allowed':False}))


if __name__ == '__main__':
    prepare()
