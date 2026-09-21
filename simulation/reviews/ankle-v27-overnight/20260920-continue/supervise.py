"""Finish the authorized overnight experiment; never launch another trainer."""
from pathlib import Path
import hashlib
import json
import msvcrt
import os
import shutil
import subprocess
import sys
import time
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RUNNER = ROOT/'simulation/isaac/rl/ankle_v26_feasible/run.py'
PRIOR = ROOT/'simulation/reviews/ankle-v26-balanced'
STATE = OUT/'workflow.json'
PYTHON = ROOT/'.venv/Scripts/python.exe'
LOCK = (OUT/'workflow.lock').open('a+b')
if LOCK.tell() == 0:
    LOCK.write(b'0'); LOCK.flush()
LOCK.seek(0)
try:
    msvcrt.locking(LOCK.fileno(), msvcrt.LK_NBLCK, 1)
except OSError:
    raise SystemExit('An overnight supervisor already holds this experiment lock')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    for attempt in range(10):
        try:
            temp.replace(path)
            return
        except PermissionError:
            time.sleep(.1)
    raise RuntimeError(f'Cannot update {path}')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

state = read(STATE) if STATE.exists() else {'schema':'drobot.overnight-diagonal.v1',
    'created_unix':time.time(), 'hardware_deployed':False, 'result_is_hardware_qualified':False}
state.update(supervisor_pid=os.getpid(), status='waiting_for_training')

def save(**updates):
    state.update(**updates, updated_unix=time.time())
    write(STATE, state)

def check_sources():
    contract = read(OUT/'train/contract.json')
    changed = [name for name, sha in contract['source_hashes'].items() if digest(ROOT/name) != sha]
    if changed:
        raise RuntimeError(f'Frozen training sources changed; review before evaluating: {changed}')
    asset = ROOT/'simulation/exports/isaac/ankle_v23_stride/active/robot.usdc'
    if digest(asset) != contract['asset_sha256']:
        raise RuntimeError('Robot asset changed since training began')

def worker(name, checkpoint, *, record=False, pair=0):
    folder = OUT/name
    report = folder/'report.json'
    if report.exists() and read(report).get('status') == 'completed':
        return folder
    if folder.exists():
        raise RuntimeError(f'Incomplete worker output exists; inspect before resuming: {folder}')
    check_sources()
    command = ['C:/isaacsim/python.bat', str(RUNNER), 'evaluate', '--output', str(folder),
               '--checkpoint', str(checkpoint), '--seed', '2522' if record else '2523']
    if record:
        command += ['--record', '--pair', str(pair)]
    save(status='recording' if record else 'evaluating', active_worker=name,
         active_checkpoint=str(checkpoint))
    write(OUT/(name+'-command.json'), command)
    with (OUT/(name+'.log')).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONIOENCODING':'utf-8'}, creationflags=subprocess.CREATE_NO_WINDOW)
        save(active_worker_wrapper_pid=process.pid)
        code = process.wait(timeout=1800)
    if code or not report.exists() or read(report).get('status') != 'completed':
        raise RuntimeError(f'{name} failed; inspect saved worker output (exit {code})')
    with (OUT/(name+'-analysis.log')).open('w', encoding='utf-8') as log:
        subprocess.run([str(PYTHON), str(PRIOR/'analyze.py'), str(folder)], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW)
    return folder

def candidate(folder, checkpoint, label):
    report = read(folder/'report.json')
    trials = report['trials']
    if len(trials) != 4:
        raise RuntimeError('Selection requires all four trials')
    analysis = read(folder/'analysis.json')
    clear = [t['advance_samples_both_clear_fraction'] for t in analysis['trials']]
    mean = lambda values: sum(values)/len(values)
    minimum_steps = [min(t['full_steps_by_leg_FL_RL_FR_RR']) for t in trials]
    falls = sum(t['outcome']['failed'] for t in trials)
    qualified = sum(t['qualified'] for t in trials)
    pair_events = sum(sum(t['full_steps_by_leg_FL_RL_FR_RR'])/2 for t in trials)
    # The ranking is for review, never permission to run real hardware.
    # Both starts/strengths and completed steps take priority over translation.
    score = [qualified, -falls, min(minimum_steps), mean(minimum_steps), min(clear),
             min(mean(clear[::2]), mean(clear[1::2])), pair_events, mean(clear),
             -mean([abs(t['lateral_m']) for t in trials]),
             -mean([t['tilt_rms_deg'] for t in trials]), mean([t['forward_m'] for t in trials])]
    return {'label':label, 'checkpoint':str(checkpoint), 'sha256':digest(checkpoint),
            'report':str(folder/'report.json'), 'analysis':str(folder/'analysis.json'),
            'qualified_trials':qualified, 'falls':falls, 'complete_pair_events':pair_events,
            'both_clear_fractions':clear, 'ranking_score':score, 'trials':trials}

try:
    save()
    deadline = time.time()+4*3600
    while True:
        training = read(OUT/'train/status.json') if (OUT/'train/status.json').exists() else {}
        save(training=training)
        if training.get('status') == 'completed':
            if not training.get('training_returned') or training.get('last_iteration') != 2549:
                raise RuntimeError('Training did not reach its planned endpoint')
            break
        if training.get('status') in ('failed','stopped_early'):
            raise RuntimeError('Training stopped; preserve evidence and inspect before resuming')
        launch = read(OUT/'train-launch.json')
        if launch.get('status') == 'exited':
            raise RuntimeError('Training launcher exited without a completed training record')
        if time.time() > deadline:
            raise RuntimeError('Training exceeded four hours; inspect the existing job, do not duplicate it')
        time.sleep(30)
    check_sources()
    checkpoints = [OUT/f'train/checkpoints/model_{i}.pt' for i in (600,1000,1400,1800,2200,2549)]
    baseline = PRIOR/'20260920-train-r2/checkpoints/model_549.pt'
    ranking = [candidate(PRIOR/'20260920-eval-r2-549', baseline, 'user-approved starting candidate')]
    write(OUT/'ranking.json', ranking)
    for checkpoint in checkpoints:
        folder = worker('eval-'+checkpoint.stem, checkpoint)
        ranking.append(candidate(folder, checkpoint, checkpoint.stem))
        ranking.sort(key=lambda row:row['ranking_score'], reverse=True)
        write(OUT/'ranking.json', ranking)
        save(evaluated_candidates=len(ranking), best_so_far=ranking[0])
    best = ranking[0]
    selected = OUT/'selected'
    selected.mkdir(exist_ok=True)
    shutil.copy2(best['checkpoint'], selected/'model.pt')
    write(selected/'selection.json', {'selected':best, 'starting_checkpoint':str(baseline),
        'hardware_qualified':False, 'deployed':False,
        'selection_purpose':'review candidate; matching controller and supported real trial still required'})
    videos = []
    for pair in (0,1):
        folder = worker(f'video-pair-{pair}', Path(best['checkpoint']), record=True, pair=pair)
        videos.append({'pair':pair, 'path':str(folder/'walk.mp4'), 'report':str(folder/'report.json')})
    subprocess.run(['C:/isaacsim/python.bat', str(OUT/'export_actor.py'), str(selected/'model.pt'),
                    str(selected/'actor.npz')], cwd=ROOT, check=True,
                    creationflags=subprocess.CREATE_NO_WINDOW)
    subprocess.run([str(PYTHON),str(OUT/'compose_video.py')], cwd=ROOT,check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)
    lines = ['# Overnight diagonal training results', '',
        f"Selected for review: **{best['label']}**. Qualified trials: {best['qualified_trials']}/4.",
        'The user-approved checkpoint is retained. No Pi deployment or physical motion occurred.', '',
        '[Comparison video](comparison.mp4) · [First diagonal](video-pair-0/walk.mp4) · [Other starting diagonal](video-pair-1/walk.mp4)', '',
        '| Candidate | Qualified trials | Falls | Complete pair events | Both-clear fractions, four trials |',
        '|---|---:|---:|---:|---|']
    for row in ranking:
        fractions=', '.join(f'{v:.1%}' for v in row['both_clear_fractions'])
        lines.append(f"| {row['label']} | {row['qualified_trials']}/4 | {row['falls']} | {row['complete_pair_events']:g} | {fractions} |")
    lines += ['', 'All selection trials use the first 32-second episode, both starting diagonals,',
        'and nominal/85% motors. Standing, dragging and body translation do not count as full steps.',
        'Complete walking needs at least two full steps per leg, acceptable drift and body tilt,',
        'and no falls. Ranking never changes these criteria.', '',
        'The exported actor is the Gaussian mean network, not the deployed V23 Beta policy.',
        'See [hardware handoff](hardware-handoff.md) before attempting a physical trial.', '']
    (OUT/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    save(status='completed', finished_unix=time.time(), selected=best, videos=videos,
         comparison_video=str(OUT/'comparison.mp4'), requires_agent_review=True,
         result_is_hardware_qualified=False)
except BaseException as error:
    save(status='failed', error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    LOCK.close()
