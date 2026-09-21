"""Summarize saved evaluation evidence; no simulator or model execution."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
run = args.run
report = json.loads((run/'report.json').read_text())
rows = [json.loads(line) for line in (run/'trace.jsonl').read_text().splitlines()]
events = [json.loads(line) for line in (run/'touchdown-events.jsonl').read_text().splitlines()]
pairs = ((0, 3), (1, 2))
results = []
for trial in report['trials']:
    e = trial['environment']
    motion = [r for r in rows if r['active'][e] and not r['parked'][e] and r['time_s'] > 2.5]
    advance = [r for r in motion if r['stage'][e] == 2]
    lift = [r for r in motion if r['stage'][e] == 1]
    selected_landings = [r for r in events if r['environment'] == e and r['flight_was_selected']
                        and r['stage'] >= 3]
    reasons = Counter()
    legs = []
    for landing in selected_landings:
        failed = []
        if landing['flight_forward_m'] < landing['required_stride_m']:
            failed.append('short_forward_travel')
        if landing['flight_forward_m'] >= .18:
            failed.append('excess_forward_travel')
        if landing['flight_sideways_m'] >= .035:
            failed.append('excess_sideways_travel')
        if landing['flight_air_time_s'] < .10:
            failed.append('short_air_time')
        if landing['flight_peak_m'] < .008:
            failed.append('insufficient_clearance')
        if not landing['flight_support_good']:
            failed.append('lost_qualified_support')
        reasons.update(failed or ['all_individual_flight_criteria_met'])
    for leg, name in enumerate(('FL', 'RL', 'FR', 'RR')):
        landings = [r for r in selected_landings if r['leg'] == leg]
        substantial = [r for r in landings if r['flight_peak_m'] >= .008 and r['flight_air_time_s'] >= .10]
        legs.append({'leg': name, 'selected_touchdowns_in_lower_or_later': len(landings),
                     'substantial_flights': len(substantial),
                     'best_airborne_landing_forward_mm': max((r['flight_forward_m']*1000 for r in substantial), default=None),
                     'mean_airborne_landing_forward_mm': mean(r['flight_forward_m']*1000 for r in substantial) if substantial else None})
    results.append({**trial,
        'advance_samples_both_clear_fraction': mean(all(r['sole_clearance_m'][e][l] > .008 and not r['contacts'][e][l]
            for l in pairs[r['pair'][e]]) for r in advance) if advance else 0.,
        'support_ankle_lift_progress_mean': mean(r['ankle_onset_progress'][e] for r in lift) if lift else 0.,
        'selected_landing_rejections_nonexclusive': dict(reasons), 'landing_evidence_by_leg': legs})
summary = {'run': str(run), 'checkpoint': report['checkpoint'], 'trials': results,
           'trace_sample_hz': 10, 'touchdown_event_sample_hz': 60,
           'hardware_qualified': False,
           'interpretation': 'Forward body translation alone is not completed stepping; original success criteria unchanged.'}
(run/'analysis.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
