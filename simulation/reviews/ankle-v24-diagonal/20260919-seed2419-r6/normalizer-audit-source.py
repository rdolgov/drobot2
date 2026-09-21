import json,hashlib
from pathlib import Path
import torch
paths=[Path('simulation/isaac/models/ankle-v23-transfer/seed2311-model_2999.pt'),Path('logs/rsl_rl/drobot_ankle_v24_diagonal/2026-09-19_21-28-02_diagonal-2419-r6/model_0.pt'),Path('logs/rsl_rl/drobot_ankle_v24_diagonal/2026-09-19_21-28-02_diagonal-2419-r6/model_2000.pt')]
records=[];initial=None
for p in paths:
 d=torch.load(p,map_location='cpu',weights_only=True);s=d['actor_state_dict']
 if initial is None:initial=s
 normal={k:(int(v.item()) if v.numel()==1 else {'minimum':float(v.min()),'maximum':float(v.max()),'finite':bool(torch.isfinite(v).all())}) for k,v in s.items() if 'normalizer' in k}
 delta=max(float((v-initial[k]).abs().max()) for k,v in s.items() if k.startswith('mlp.') and k.endswith('weight'))
 records.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'iteration':d['iter'],'normalizer':normal,'maximum_actor_weight_change_from_warm':delta})
result={'scope':'Read-only checkpoint tensors and installed normalization/load source; no policy execution or training','findings':'Actor weights and normalization statistics update; normalization until defaults to None. Actor loading includes prior normalizer statistics, while critic, optimizer and iteration are fresh. No frozen-normalizer or actor-loading failure identified.','records':records}
Path('simulation/reviews/ankle-v24-diagonal/20260919-seed2419-r6/normalizer-audit.json').write_text(json.dumps(result,indent=2)+'\n')
for r in records:print(r['iteration'],r['normalizer'].get('obs_normalizer.count'),r['maximum_actor_weight_change_from_warm'])
