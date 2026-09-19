"""The constant reset pose is an observation, not a motion trajectory."""
import json
from pathlib import Path
import numpy as np
def make_plan(phases,dt=1/60,phase_s=12.,start_x=.2,rise=.18,tread=.25,edge=.34):
    if phases!=1 or dt!=1/60 or phase_s!=12. or start_x!=.2 or rise!=.18 or tread!=.25 or edge!=.34:
        raise ValueError('This curriculum uses one unscripted stage on four 180 mm stairs')
    folder=Path(__file__).resolve().parent
    meta=json.loads((folder/'reference.json').read_text())
    with np.load(folder/'reference.npz') as a:
        result={k:np.concatenate((a[k],np.repeat(a[k][-1:],242,axis=0)),axis=0) for k in a.files}
    result.update(goal_feet=meta['goal_feet'],moved_legs=[3],phases=meta['phases'],order=[3])
    return result
