# Continued diagonal walking model review

The final candidate improves forward body travel in 3 of 4 trials and completes one physical diagonal-pair step in one reduced-strength trial. Body tilt remains 1.12–1.36 degrees RMS with no falls. It still fails all four complete-walking trials, with inconsistent rear-left lift, reduced simultaneous clearance and excessive sideways drift in two conditions.

**Simulation review only. No model was deployed and no robot motion was commanded.**

## Model selection

- Review candidate: `simulation\reviews\ankle-v26-balanced\20260920-train-r2\checkpoints\model_549.pt`.
- SHA256: `e130f5a1a463a293e6a0696fb94354828ceaa1b34bed74ce12797d18807bbe8c`.
- Decision: Keep as an experimental continuation checkpoint, not a replacement for the previous gait. Preserve the previous checkpoint and show both in the review video. Further work must address unloading and lifting the weaker diagonal before more forward-travel reward.
- [Updated comparison video](continued-diagonal-comparison.mp4)
- [Selection metadata](selection.json)
- [Machine-readable comparisons](comparison.json)
- [Training and reward changes](../../docs/rl-ankle-v26-balanced.md)

## Independent 32-second evaluations

Each table lists nominal motors starting FL+RR, nominal starting FR+RL,
85% motors starting FL+RR, and 85% starting FR+RL. Movement stops after
24 seconds, followed by a hold. Only the first episode is measured.

Both-clear is the fraction of saved advance-stage samples with both
selected feet above 8 mm and out of contact. This is a 10 Hz estimate.
Touchdown measurements are recorded at 60 Hz. Body translation alone
does not count as a completed step.

### 20260920-baseline

Qualified trials: **0/4**. Falls: 0.

| Trial | Forward cm | Lateral cm | Body tilt RMS deg | Both clear % | Full steps FL/RL/FR/RR |
|---|---:|---:|---:|---:|---|
| 100%, FL+RR | 9.0 | -6.69 | 1.07 | 53.2 | 0/0/0/0 |
| 100%, FR+RL | 21.45 | -4.44 | 1.11 | 0 | 0/0/0/0 |
| 85%, FL+RR | 5.24 | -2.84 | 1.56 | 23.4 | 0/0/0/0 |
| 85%, FR+RL | 26.04 | -0.43 | 1.05 | 0 | 0/0/0/0 |

[Full report](20260920-baseline/report.json) · [Landing analysis](20260920-baseline/analysis.json)

### 20260920-eval-400

Qualified trials: **0/4**. Falls: 0.

| Trial | Forward cm | Lateral cm | Body tilt RMS deg | Both clear % | Full steps FL/RL/FR/RR |
|---|---:|---:|---:|---:|---|
| 100%, FL+RR | 9.77 | 7.83 | 1.16 | 6.4 | 0/0/0/0 |
| 100%, FR+RL | 20.86 | 2.65 | 1.02 | 0 | 0/0/0/0 |
| 85%, FL+RR | 15.19 | 1.99 | 1.21 | 0 | 0/0/0/0 |
| 85%, FR+RL | 22.98 | 1.2 | 1.35 | 0 | 0/0/0/0 |

[Full report](20260920-eval-400/report.json) · [Landing analysis](20260920-eval-400/analysis.json)

### 20260920-eval-r1-500

Qualified trials: **0/4**. Falls: 0.

| Trial | Forward cm | Lateral cm | Body tilt RMS deg | Both clear % | Full steps FL/RL/FR/RR |
|---|---:|---:|---:|---:|---|
| 100%, FL+RR | 15.06 | 1.98 | 0.78 | 6.4 | 0/0/0/0 |
| 100%, FR+RL | 17.13 | 6.36 | 1.3 | 0 | 0/0/0/0 |
| 85%, FL+RR | 13.5 | -0.64 | 0.77 | 4.3 | 0/0/0/0 |
| 85%, FR+RL | 19.64 | 3.66 | 1.29 | 0 | 0/0/0/0 |

[Full report](20260920-eval-r1-500/report.json) · [Landing analysis](20260920-eval-r1-500/analysis.json)

### 20260920-eval-r2-400

Qualified trials: **0/4**. Falls: 0.

| Trial | Forward cm | Lateral cm | Body tilt RMS deg | Both clear % | Full steps FL/RL/FR/RR |
|---|---:|---:|---:|---:|---|
| 100%, FL+RR | 19.14 | -2.02 | 1.79 | 4.3 | 0/0/0/0 |
| 100%, FR+RL | 28.46 | 2.85 | 1.33 | 0 | 0/0/0/0 |
| 85%, FL+RR | 5.7 | -1.02 | 2.6 | 0 | 0/0/0/0 |
| 85%, FR+RL | 8.99 | 0.28 | 2.97 | 0 | 0/0/0/0 |

[Full report](20260920-eval-r2-400/report.json) · [Landing analysis](20260920-eval-r2-400/analysis.json)

### 20260920-eval-r2-500

Qualified trials: **0/4**. Falls: 0.

| Trial | Forward cm | Lateral cm | Body tilt RMS deg | Both clear % | Full steps FL/RL/FR/RR |
|---|---:|---:|---:|---:|---|
| 100%, FL+RR | 19.87 | 7.17 | 1.09 | 27.7 | 0/0/0/0 |
| 100%, FR+RL | 21.75 | 13.08 | 1.37 | 0 | 0/0/0/0 |
| 85%, FL+RR | 27.62 | 6.02 | 0.89 | 2.1 | 0/0/0/0 |
| 85%, FR+RL | 31.61 | 8.54 | 1.48 | 0 | 0/0/0/0 |

[Full report](20260920-eval-r2-500/report.json) · [Landing analysis](20260920-eval-r2-500/analysis.json)

### 20260920-eval-r2-549

Qualified trials: **0/4**. Falls: 0.

| Trial | Forward cm | Lateral cm | Body tilt RMS deg | Both clear % | Full steps FL/RL/FR/RR |
|---|---:|---:|---:|---:|---|
| 100%, FL+RR | 28.73 | -1.78 | 1.17 | 8.5 | 0/0/0/0 |
| 100%, FR+RL | 19.45 | 9.87 | 1.36 | 4.3 | 0/0/0/0 |
| 85%, FL+RR | 26.57 | 1.46 | 1.3 | 6.4 | 1/0/0/1 |
| 85%, FR+RL | 36.41 | 6.52 | 1.12 | 0 | 0/0/0/0 |

[Full report](20260920-eval-r2-549/report.json) · [Landing analysis](20260920-eval-r2-549/analysis.json)

## Training records

- `20260920-train-r1`: stopped_early; last logged update 508; [status](20260920-train-r1/status.json), [source and checkpoint provenance](20260920-train-r1/contract.json).
- `20260920-train-r2`: completed; last logged update 549; [status](20260920-train-r2/status.json), [source and checkpoint provenance](20260920-train-r2/contract.json).

## Limits of the result

The strict physical landing and walking criteria are unchanged. A full step
requires actual supported flight, sufficient foot travel, a flat quiet plant,
and both selected feet completing the landing. The actor uses proprioception,
not simulator foot positions or contact forces. Estimated mass distribution
and loaded servo response remain unmeasured assumptions.

Rendered one-environment trials have separate reports; their exact motion can
differ from the four-environment batch because of numerical scene differences.
Videos retain those measurements, play at real speed and exclude reset episodes.

Only the requested training, simulation evaluations and recording were run.
No unrelated automated tests, UI checks or hardware checks were performed.
