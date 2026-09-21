"""Write review documentation from saved evidence and an explicit selection."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
selection = json.loads((root/'selection.json').read_text())
comparison = json.loads((root/'comparison.json').read_text())
lines = ['# Continued diagonal walking model review', '', selection['summary'], '',
         '**Simulation review only. No model was deployed and no robot motion was commanded.**', '',
         '## Model selection', '',
         f"- Review candidate: `{selection['checkpoint']}`.",
         f"- SHA256: `{selection['checkpoint_sha256']}`.",
         f"- Decision: {selection['decision']}",
         '- [Updated comparison video](continued-diagonal-comparison.mp4)',
         '- [Selection metadata](selection.json)',
         '- [Machine-readable comparisons](comparison.json)',
         '- [Training and reward changes](../../docs/rl-ankle-v26-balanced.md)', '',
         '## Independent 32-second evaluations', '',
         'Each table lists nominal motors starting FL+RR, nominal starting FR+RL,',
         '85% motors starting FL+RR, and 85% starting FR+RL. Movement stops after',
         '24 seconds, followed by a hold. Only the first episode is measured.', '',
         'Both-clear is the fraction of saved advance-stage samples with both',
         'selected feet above 8 mm and out of contact. This is a 10 Hz estimate.',
         'Touchdown measurements are recorded at 60 Hz. Body translation alone',
         'does not count as a completed step.', '']
for result in comparison['evaluations']:
    lines += [f"### {result['run']}", '',
              f"Qualified trials: **{result['qualified_trials']}/4**. Falls: {result['falls']}.", '',
              '| Trial | Forward cm | Lateral cm | Body tilt RMS deg | Both clear % | Full steps FL/RL/FR/RR |',
              '|---|---:|---:|---:|---:|---|']
    for i in range(4):
        counts = '/'.join(str(int(v)) for v in result['full_steps_FL_RL_FR_RR'][i])
        label = ('100%, FL+RR','100%, FR+RL','85%, FL+RR','85%, FR+RL')[i]
        lines.append(f"| {label} | {result['forward_cm'][i]} | {result['lateral_cm'][i]} | {result['body_tilt_rms_deg'][i]} | {result['both_clear_percent'][i]} | {counts} |")
    lines += ['', f"[Full report]({result['run']}/report.json) · [Landing analysis]({result['run']}/analysis.json)", '']
lines += ['## Training records', '']
for run in sorted(root.glob('20260920-train-*')):
    state = json.loads((run/'status.json').read_text())
    lines += [f"- `{run.name}`: {state['status']}; last logged update {state.get('last_iteration')}; "
              f"[status]({run.name}/status.json), [source and checkpoint provenance]({run.name}/contract.json)."]
lines += ['', '## Limits of the result', '',
          'The strict physical landing and walking criteria are unchanged. A full step',
          'requires actual supported flight, sufficient foot travel, a flat quiet plant,',
          'and both selected feet completing the landing. The actor uses proprioception,',
          'not simulator foot positions or contact forces. Estimated mass distribution',
          'and loaded servo response remain unmeasured assumptions.', '',
          'Rendered one-environment trials have separate reports; their exact motion can',
          'differ from the four-environment batch because of numerical scene differences.',
          'Videos retain those measurements, play at real speed and exclude reset episodes.', '',
          'Only the requested training, simulation evaluations and recording were run.',
          'No unrelated automated tests, UI checks or hardware checks were performed.', '']
(root/'README.md').write_text('\n'.join(lines), encoding='utf-8')
print(root/'README.md')
