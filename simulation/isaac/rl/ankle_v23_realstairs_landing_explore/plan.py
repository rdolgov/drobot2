"""A reset library observed by the policy; no prescribed motion between poses."""
import json
from pathlib import Path
import numpy as np

def make_plan(phases,dt=1/60,phase_s=12.,start_x=.2,rise=.18,tread=.25,edge=.34):
    if (phases,dt,phase_s,start_x,rise,tread,edge)!=(1,1/60,12.,.2,.18,.25,.34):
        raise ValueError('Mixed reset curriculum requires the original four 180 mm stairs')
    folder=Path(__file__).resolve().parent
    meta=json.loads((folder/'reference.json').read_text())
    with np.load(folder/'reference.npz') as data:
        result={key:data[key] for key in data.files}
    result.update(goal_feet=meta['goal_feet'],moved_legs=[3],phases=meta['phases'],order=[3])
    return result
