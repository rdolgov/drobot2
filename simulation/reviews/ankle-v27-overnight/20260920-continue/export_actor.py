"""Serialize the selected local checkpoint's Gaussian mean actor; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
weights = checkpoint['actor_state_dict']
keys = ['obs_normalizer._mean','obs_normalizer._std']
keys += [f'mlp.{i}.{part}' for i in (0,2,4,6) for part in ('weight','bias')]
arrays = {key:weights[key].detach().cpu().numpy() for key in keys}
if arrays['mlp.6.weight'].shape != (16,128):
    raise ValueError('This export is only for the 82-input, 16-mean Gaussian actor')
if any(not np.isfinite(value).all() for value in arrays.values()):
    raise ValueError('Checkpoint contains nonfinite actor data')
np.savez(args.output, **arrays)
legs = ('front_left','rear_left','front_right','rear_right')
metadata = {
    'schema':'drobot.diagonal-ik-gaussian.actor.v1',
    'source_checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    'weights_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
    'iteration':checkpoint['iter'], 'input_size':82, 'output_size':16,
    'layer_sizes':[82,256,256,128,16], 'hidden_activation':'ELU',
    'inference':'(observation-mean)/(std+0.01), three ELU hidden layers, linear Gaussian mean, clamp to [-1,1]',
    'action_names':[f'{leg}_{kind}' for kind in ('hip_abduction','hip_flexion','knee','ankle_pitch') for leg in legs],
    'servo_ids':[1,7,4,10,2,8,5,11,3,9,6,12,13,15,14,16],
    'observation_order':['commands*10[3]','diagonal stage clock sin/cos[2]','body angular velocity[3]',
        'projected gravity[3]','body linear acceleration/9.81[3]','joint position-neutral[16]',
        'joint velocity*0.25[16]','normalized completed motor target[16]','filtered action[16]','expected stance[4]'],
    'controller':'ankle_v25_ik.IKDiagonalEnv inherited by ankle_v26_feasible; full timing/reference/projection contract required',
    'control_hz':60, 'deployed_v23_runtime_compatible':False,
    'controller_port_implemented':False, 'hardware_qualified':False, 'deployed':False,
    'export_validation':'Stored array dimensions and finiteness only; no inference parity or hardware check performed'}
args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(args.output)
