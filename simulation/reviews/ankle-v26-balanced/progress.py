"""Read the running job's saved metrics without touching the simulator."""
import json
import argparse
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument('--run', default='20260920-train-r2')
args = parser.parse_args()
run = Path(__file__).resolve().parent/args.run
state = json.loads((run/'status.json').read_text())
contract = json.loads((run/'contract.json').read_text())
last_iteration = state.get('start_iteration', 0)+contract['arguments']['iterations']-1
updates = [json.loads(x) for x in (run/'optimizer.jsonl').read_text().splitlines() if x.endswith('}')]
result = {'status': state['status'], 'start_iteration': state.get('start_iteration'),
          'last_optimizer_update': updates[-1]}
if len(updates) > 2:
    recent = updates[-20:]
    seconds_per_iteration = (recent[-1]['unix']-recent[0]['unix'])/(recent[-1]['iteration']-recent[0]['iteration'])
    result.update(seconds_per_iteration=seconds_per_iteration,
                  estimated_remaining_minutes=max(0,last_iteration-updates[-1]['iteration'])*seconds_per_iteration/60)
metrics = run/'metrics.jsonl'
if metrics.exists():
    rows = [json.loads(x) for x in metrics.read_text().splitlines() if x.endswith('}')]
    base = [r for r in rows if 'Metrics/train_forward_speed_m_s' in r]
    diagonal = [r for r in rows if 'Metrics/train_paired_landings' in r]
    if base:
        result['latest_episode_batch'] = {k:v for k,v in base[-1].items() if k in
          ('step','episode_count','Metrics/train_forward_speed_m_s','Metrics/train_fall_fraction','Metrics/train_episode_seconds')}
    if diagonal:
        result['latest_step_batch'] = {k:v for k,v in diagonal[-1].items() if any(s in k for s in ('paired_landings','qualified_touchdowns','lift_clear','step'))}
print(json.dumps(result, indent=2))
