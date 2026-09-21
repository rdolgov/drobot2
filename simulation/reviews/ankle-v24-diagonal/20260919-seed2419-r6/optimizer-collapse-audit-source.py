import json,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
folder=Path('logs/rsl_rl/drobot_ankle_v24_diagonal/2026-09-19_21-28-02_diagonal-2419-r6')
generator=torch.Generator().manual_seed(2419)
probe=torch.cat([torch.zeros(1,82),torch.randn(256,82,generator=generator).clamp(-3,3)])
records=[]
for number in [0,750,1500,2000,2250,2500,2750,2999]:
 path=folder/f'model_{number}.pt';d=torch.load(path,map_location='cpu',weights_only=True);s=d['actor_state_dict'];x=probe.clone()
 for layer in [0,2,4,6]:
  x=F.linear(x,s[f'mlp.{layer}.weight'],s[f'mlp.{layer}.bias'])
  if layer!=6:x=F.elu(x)
 raw=x.reshape(-1,2,16);parameters=F.softplus(raw)+1.;a,b=parameters.unbind(1);mean=2*a/(a+b)-1;std=2*(a*b/((a+b)**2*(a+b+1))).sqrt()
 records.append({'iteration':number,'optimizer_learning_rates':[g['lr'] for g in d['optimizer_state_dict']['param_groups']], 'raw_head_min':float(raw.min()),'raw_head_max':float(raw.max()),'concentration_exact_floor_fraction':float((parameters==1).float().mean()),'maximum_absolute_mean_action':float(mean.abs().max()),'mean_action_std':float(std.mean()),'maximum_softplus_derivative':float(raw.sigmoid().max()),'actor_parameters_finite':all(bool(torch.isfinite(v).all()) for v in s.values())})
try:
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 events=EventAccumulator(str(folder),size_guidance={'scalars':0}).Reload();tags=events.Tags().get('scalars',[])
 matching={tag:[{'iteration':v.step,'value':v.value} for v in events.Scalars(tag) if v.step%100==0 or v.step>=2950] for tag in tags if 'learning_rate' in tag.lower()}
except Exception as error:matching={'read_error':str(error)}
result={'scope':'Read-only stored checkpoints and TensorBoard scalar history. Neural probes use zero and 256 seeded clipped-normal normalized inputs, not recorded observations. Final diagnostic trace separately establishes zero mean actions on its actual observations.','configured_initial_learning_rate':.0003,'installed_adaptive_cap':.01,'head_transform':'softplus(raw) + 1, float32','records':records,'learning_rate_history':matching}
output=Path('simulation/reviews/ankle-v24-diagonal/20260919-seed2419-r6/optimizer-collapse-audit.json');output.write_text(json.dumps(result,indent=2)+'\n')
for r in records:print(json.dumps(r))
print('LR history tags:',list(matching))
