"""Static reset library; no prescribed motion trajectory."""
import json
from pathlib import Path
import numpy as np
def make_plan(phases,dt=1/60,phase_s=12.,start_x=.2,rise=.18,tread=.25,edge=.34):
    if phases!=1 or dt!=1/20 or rise!=.18 or tread!=.25 or edge!=.34:
        raise ValueError('Full-height 180 mm stair task only')
    folder=Path(__file__).resolve().parent
    meta=json.loads((folder/'reference.json').read_text())
    with np.load(folder/'reference.npz') as a:
        result={k:a[k] for k in a.files}
    result.update(goal_feet=meta['goal_feet'],moved_legs=[0,1,2,3],phases=meta['phases'],order=[2])
    return result
