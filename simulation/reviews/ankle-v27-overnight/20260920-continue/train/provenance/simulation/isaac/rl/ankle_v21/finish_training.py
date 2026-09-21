"""Wait for the two authorized jobs, select checkpoints, then record walks.

Persistent orchestration only; does not start duplicate PPO jobs. Run with
the CAD Python. Isaac workers are started hidden through the Windows launcher.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'simulation/reviews/ankle-v21-planted'
STATE = OUT/'workflow.json'
status = {'schema':'drobot.ankle-v21-workflow.v1','status':'waiting_for_training',
          'seeds':[2101,2102],'iterations_per_seed':3000,'environments_per_seed':256,
          'steps_per_iteration':64,'planned_transitions':98304000,
          'created_unix':time.time(),'completed_recordings':[]}

def save():
    status['updated_unix'] = time.time()
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(status,indent=2)+'\n')
    tmp.replace(STATE)

def run_worker(args,folder):
    folder.mkdir(parents=True,exist_ok=True)
    report = folder/'report.json'
    if report.exists() and json.loads(report.read_text()).get('status')=='completed': return
    command = ['C:/isaacsim/python.bat',str(Path(__file__).with_name('evaluate_record.py')),*args,'--output',str(folder)]
    (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (folder/'worker.log').open('w',encoding='utf-8') as log:
        result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONIOENCODING':'utf-8'},creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or not report.exists() or json.loads(report.read_text()).get('status')!='completed':
        raise RuntimeError(f'Worker did not complete: {folder}; exit {result.returncode}')

try:
    save()
    while True:
        jobs = []
        for seed in status['seeds']:
            path = OUT/f'20260912-seed{seed}/train.metrics.completion.json'
            jobs.append(json.loads(path.read_text()) if path.exists() else {'status':'starting'})
        status['training_jobs'] = [{'seed':s,'status':j.get('status'),'checkpoint':j.get('checkpoints')} for s,j in zip(status['seeds'],jobs)]
        save()
        if any(j.get('status')=='failed' for j in jobs): raise RuntimeError('A training job failed; inspect its completion record and log')
        if all(j.get('status')=='completed' for j in jobs): break
        time.sleep(30)
    checkpoints = []
    for job in jobs:
        final = Path(job['checkpoints'][0])
        checkpoints.extend(str(final.parent/f'model_{i}.pt') for i in (1500,2250,2999))
    status.update(status='selecting_checkpoints',candidate_checkpoints=checkpoints)
    save()
    selection = OUT/'selection'
    run_worker(['--checkpoints',*checkpoints,'--seed','2201'],selection)
    trials = json.loads((selection/'report.json').read_text())['trials']
    grouped = {}
    for trial in trials: grouped.setdefault(trial['checkpoint'],[]).append(trial)
    ranking = []
    for checkpoint,cases in grouped.items():
        accepted = all(t['mean']['fall_fraction']<=.10 and t['mean']['forward_m_s']>=.5*t['speed_command_m_s']
                       and t['mean']['planted_stance_fraction']>=.70 and t['mean']['stop_speed_m_s']<=.015 for t in cases)
        ranking.append({'checkpoint':checkpoint,'accepted_selection_gate':accepted,'mean_score':sum(t['score'] for t in cases)/len(cases)})
    ranking.sort(key=lambda r:(r['accepted_selection_gate'],r['mean_score']),reverse=True)
    (OUT/'checkpoint-ranking.json').write_text(json.dumps(ranking,indent=2)+'\n')
    best = ranking[0]
    status.update(status='recording',ranking=ranking,best=best)
    save()
    # Record each training seed's best checkpoint. Retain failures in both clips.
    for seed in status['seeds']:
        selected = next(r for r in ranking if f'printable-{seed}' in r['checkpoint'])
        folder = OUT/f'walk-seed{seed}'
        run_worker(['--checkpoints',selected['checkpoint'],'--record','--speed','.03','--seed','2301'],folder)
        record = json.loads((folder/'report.json').read_text())
        status['completed_recordings'].append({'seed':seed,'selected':selected,'report':str(folder/'report.json'),'video':record['trials'][0]['video']})
        save()
    archive = ROOT/'simulation/isaac/models/ankle-v21-planted'
    archive.mkdir(parents=True,exist_ok=True)
    for seed in status['seeds']:
        selected = next(r for r in ranking if f'printable-{seed}' in r['checkpoint'])
        path = Path(selected['checkpoint'])
        dest = archive/f'seed{seed}-{path.name}'
        dest.write_bytes(path.read_bytes())
        selected['archive'] = str(dest)
        selected['sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
    status.update(status='completed',finished_unix=time.time(),result_is_hardware_qualified=False)
    save()
except BaseException as exc:
    import traceback
    status.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    save()
    raise
