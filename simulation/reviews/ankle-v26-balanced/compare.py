"""Produce a compact table from completed saved reports, without running models."""
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
results = []
for run in [root/'20260920-baseline', *sorted(root.glob('20260920-eval-*'))]:
    if not (run/'report.json').exists():
        continue
    report = json.loads((run/'report.json').read_text())
    analysis = json.loads((run/'analysis.json').read_text()) if (run/'analysis.json').exists() else None
    ts = report['trials']
    result = {'run': run.name, 'checkpoint': report['checkpoint'],
        'qualified_trials': sum(t['qualified'] for t in ts), 'trials': len(ts),
        'falls': sum(t['outcome']['failed'] for t in ts),
        'forward_cm': [round(t['forward_m']*100, 2) for t in ts],
        'lateral_cm': [round(t['lateral_m']*100, 2) for t in ts],
        'body_tilt_rms_deg': [round(t['tilt_rms_deg'], 2) for t in ts],
        'full_steps_FL_RL_FR_RR': [t['full_steps_by_leg_FL_RL_FR_RR'] for t in ts],
        'touchdowns_FL_RL_FR_RR': [t['qualified_touchdowns_by_leg'] for t in ts],
        'peak_clearance_mm_FL_RL_FR_RR': [[round(v*1000,1) for v in t['peak_swing_clearance_m_by_leg']] for t in ts]}
    if analysis:
        result['both_clear_percent'] = [round(t['advance_samples_both_clear_fraction']*100,1) for t in analysis['trials']]
    results.append(result)
(root/'comparison.json').write_text(json.dumps({'evaluations': results, 'hardware_qualified': False}, indent=2)+'\n')
for r in results:
    print(r['run'], 'qualified', r['qualified_trials'], '/', r['trials'], 'falls', r['falls'])
    print('  forward cm:', r['forward_cm'], 'lateral cm:', r['lateral_cm'])
    print('  tilt RMS deg:', r['body_tilt_rms_deg'], 'both clear %:', r.get('both_clear_percent'))
    print('  full steps:', r['full_steps_FL_RL_FR_RR'])
    print('  touchdowns:', r['touchdowns_FL_RL_FR_RR'])
    print('  peak clearance mm:', r['peak_clearance_mm_FL_RL_FR_RR'])
