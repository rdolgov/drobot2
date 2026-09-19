"""Load a precomputed geometric/static reference; no dynamics are pre-solved."""
import json
from pathlib import Path
import numpy as np


def make_plan(phases, dt=1/60, phase_s=12., start_x=.2, rise=.02, tread=.25, edge=.34):
    if (abs(dt-1/60)>1e-8 or phase_s!=12. or start_x!=.2 or rise!=.02 or tread!=.25 or edge!=.34):
        raise ValueError('Recalculate the static reference for changed geometry or timing')
    folder=Path(__file__).resolve().parent
    metadata=json.loads((folder/'reference.json').read_text())
    end=round(phases*phase_s/dt)
    hold=round(4/dt)+2
    with np.load(folder/'reference.npz') as archive:
        result={key:np.concatenate((archive[key][:end],np.repeat(archive[key][end-1:end],hold,axis=0)),axis=0)
                for key in ('feet','base','stance','joints','pitch')}
    result.update(goal_feet=result['feet'][-1].tolist(), moved_legs=sorted(set(metadata['order'][:phases])),
                  phases=metadata['phases'][:phases],order=metadata['order'][:phases])
    return result
