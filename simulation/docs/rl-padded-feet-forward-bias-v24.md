# V24 padded-feet forward-bias residual crawl

## Why this version exists

The first real V23 trial moved backward after adhesive Velcro-like pads were
installed under the rectangular shoes. They are not foam: the important model
changes are potentially lower sliding friction and a modestly softer interface
than bare printed plastic. The rear battery may also be keeping too much load
behind the body center. V24 treats this as a contact-and-load-transfer mismatch,
not merely a request for a larger speed reward.

## Simulation changes

- Continue from selected V23 checkpoint 1500 and retain its 50-value observation,
  residual action, sequential RR/FR/RL/FL crawl, 60 Hz loop, and target limiter.
- Lower tread static/dynamic friction from `1.05/0.85` to `0.75/0.55` to
  conservatively represent a Velcro-like pad that may slide more than the bare
  printed tread.
- Reduce compliant-contact stiffness from `12000` to `7000` and raise damping
  from `45` to `85` to represent the modest compliance of the Velcro-like pad
  rather than rigid printed tread.
- Keep the measured 523 g external rear payload and its rear-center uncertainty.
- Randomize simulated servo effort and target-rate capacity from 70-100% to
  represent bounded low-charge voltage sag. Low charge does not change balance
  or battery mass, and hardware below its safe voltage cutoff must not be used.
- Increase analytic forward weight transfer from 8 mm to 18 mm.
- Target a mild 3 degree nose-down body pitch instead of an exactly level body.
- Add an explicit normalized penalty for any body-forward velocity opposite the
  requested command, alongside stronger instant and sustained forward progress.
- Add a command-relative overspeed penalty and tighten velocity tracking at the
  low end so the UI speed remains meaningful instead of merely setting cadence.
- Strengthen scheduled stance, three-foot support, and excess-airborne-foot
  terms after the first continuation moved forward but used two-foot support too
  often at 0.015 m/s.
- Add a one-sided, normalized rearward-pitch cost after the speed/support
  correction remained about 0.6 degrees nose-up despite the 3 degree nose-down
  target. This makes the requested forward bias materially visible to PPO.
- Preserve the V23 straightness and anti-jerk objectives.

## First-continuation screen

The first 1,000-update continuation ran from V23 iteration 1500 through V24
iteration 2499. A deterministic 20-second, one-episode screen used seed 4401,
91.0% effective effort, and 74.2% effective target rate. V23 was replayed with
its original 8 mm reference shift; V24 used 18 mm. Selected results were:

| Policy | Command | Distance | Backward steps | Lateral | 3/4-foot support | Joint RMS accel. |
|---|---:|---:|---:|---:|---:|---:|
| V23 model 1500 | 0.005 m/s | 0.0883 m | 27.6% | 0.0046 m | 100.0% | 5.56 rad/s^2 |
| V24 model 2350 | 0.005 m/s | 0.1787 m | 12.6% | 0.0183 m | 97.3% | 6.48 rad/s^2 |
| V24 model 2450 | 0.005 m/s | 0.1736 m | 14.2% | 0.0176 m | 98.0% | 6.18 rad/s^2 |
| V23 model 1500 | 0.015 m/s | 0.3012 m | 12.8% | 0.0084 m | 94.8% | 7.62 rad/s^2 |
| V24 model 2350 | 0.015 m/s | 0.4331 m | 6.4% | 0.0075 m | 81.1% | 8.22 rad/s^2 |
| V24 model 2450 | 0.015 m/s | 0.4137 m | 5.8% | 0.0006 m | 80.3% | 8.37 rad/s^2 |

This run reduced reverse motion but overshot both requested speeds and spent too
much time with two feet airborne at 0.015 m/s. It was therefore not exported.
A corrective continuation from model 2450 adds a normalized overspeed cost,
tightens the low-speed tracking width, and increases support rewards/penalties.

## Selected checkpoint

Model 3248 is selected after the speed/support correction and a final
pitch-focused continuation. The final deterministic gate used seed 4401 and
included an episode with 91.0% effective effort and 74.2% target rate:

| Command | Trials | Mean speed | Backward steps | Lateral | Mean pitch | 3/4-foot support | Joint RMS accel. | Falls/stalls |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.005 m/s | 2 x 30 s | 0.0065 m/s | 20.1% | 0.0038 m | -0.0087 rad | 98.6% | 6.56 rad/s^2 | 0 / 0 |
| 0.015 m/s | 2 x 30 s | 0.0129 m/s | 10.6% | 0.0366 m | -0.0083 rad | 95.6% | 7.96 rad/s^2 | 0 / 0 |
| 0.030 m/s | 1 x 20 s | 0.0174 m/s | 10.3% | 0.0193 m | -0.0089 rad | 87.2% | 8.59 rad/s^2 | 0 / 0 |

The final pitch is still slightly nose-up, but improves from model 2949's
approximately `-0.0104 rad` at low speed while retaining the 18 mm forward
support shift. Further pitch pressure was not accepted merely to hit the full
3 degree target because support, straightness, and smoothness remain the safer
selection priorities. At 0.030 m/s under target-rate sag, the policy stays
upright but does not reach the requested speed. The deployed range is therefore
limited to `0.005-0.030 m/s`, with `0.005 m/s` recommended; speed should only be
raised after reviewing real voltage sag and telemetry.

Release artifacts:

- checkpoint: `simulation/isaac/models/parallel-walking-v24-padded-feet-forward-bias/model_3248.pt`;
- checkpoint SHA-256: `e9c521fbd9f63ea0c9329bc3487a44be5f9dbc58530e9f7749eeba37635b37d2`;
- ONNX and metadata: `onboard/models/parallel-walking-v24-padded-feet-forward-bias/model_3248.onnx` and `model_3248.json`;
- ONNX SHA-256: `9f447870002dda069a017f22b9a06009031f6883aa2a82ef8d03079237a326f6`;
- 20-second 0.015 m/s review: `simulation/reviews/parallel-walking-v24-padded-feet-forward-bias-model3248-20s.mp4`;
- review contact sheet: `simulation/reviews/v24-padded-feet-forward-bias-contact-sheet.jpg`.

## Selection gate

Candidate evaluation reports actual signed speed, backward-step fraction, mean
forward pitch, lateral displacement, final heading error, support fraction,
target-limiter backlog, and body/joint acceleration. A checkpoint is rejected if
it falls, stalls, spends material time moving backward, loses the sequential
support pattern, or becomes jerkier merely to obtain forward displacement.

Real hardware remains the final authority because the pad material, floor, and
servo asymmetries are only approximated. The first test must use 0.005 m/s on a
power supply, with the automatic telemetry recorder, a spotter, and emergency
stop ready.
