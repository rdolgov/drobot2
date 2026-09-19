"""Load the full-height first-foothold reference; no dynamics are pre-solved."""
import json
from pathlib import Path
import numpy as np

def make_plan(phases, dt=1/60, phase_s=36., start_x=.2, rise=.18, tread=.25, edge=.34):
    if phases!=4 or abs(dt-1/60)>1e-8 or phase_s!=36. or start_x!=.2 or rise!=.18 or tread!=.25 or edge!=.34:
        raise ValueError('This reference is the four-phase FR up, RL floor advance, FL up, RR floor shuffle stage on four 180 mm stairs; recalculate for another stage')
    folder=Path(__file__).resolve().parent
    metadata=json.loads((folder/'reference.json').read_text())
    end=round(phases*phase_s/dt)
    hold=round(4/dt)+2
    with np.load(folder/'reference.npz') as archive:
        result={key:np.concatenate((archive[key][:end],np.repeat(archive[key][end-1:end],hold,axis=0)),axis=0)
                for key in ('feet','base','stance','joints','pitch','rpy','quat')}
    result.update(goal_feet=result['feet'][-1].tolist(),moved_legs=[0,1,2,3],phases=metadata['phases'],order=[2,1,0,3])
    return result
