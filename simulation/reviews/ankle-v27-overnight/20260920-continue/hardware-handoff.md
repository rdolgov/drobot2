# September 21 physical-trial handoff

The user approved the latest full simulation recording and asked to continue
training overnight, with a possible real trial tomorrow. Start from the
approved V26 feasible checkpoint 549, SHA256
`e130f5a1a463a293e6a0696fb94354828ceaa1b34bed74ce12797d18807bbe8c`.
Keep it as a fallback regardless of the overnight selection.

## Selected checkpoint and completed evaluation

The 2,000-update continuation completed at iteration **2549**, with 65,536,000
new simulated transitions. The unchanged ranking selected `model_2549` from
six new checkpoints and the saved approved baseline. The selected checkpoint
is available at `selected/model.pt`; its SHA256 is
`867c3a4a839a806fcda8ba2a86df3a3103c016e9c9e108d43e020d1100908b92`.
The source is `train/checkpoints/model_2549.pt`.

All four 32-second evaluation cases passed the original walking criteria,
with no falls and **five complete steps on every leg in every case**:

| Starting diagonal | Motor strength | Forward | Signed lateral drift | Body tilt RMS / peak | Both swing feet clear during sampled advance |
|---|---:|---:|---:|---:|---:|
| FL + RR | 100% | 49.14 cm | +2.97 cm | 2.37 / 4.96 degrees | 100% |
| FR + RL | 100% | 49.45 cm | -4.70 cm | 2.49 / 6.47 degrees | 100% |
| FL + RR | 85% | 51.30 cm | -1.36 cm | 2.53 / 4.85 degrees | 100% |
| FR + RL | 85% | 51.61 cm | -4.99 cm | 2.56 / 6.29 degrees | 100% |

Both-feet clearance is measured at 10 Hz during the advance stage; it is not
a claim that both feet remain airborne through lift, lowering or landing.
Per-leg peak swing clearance ranged from 38.2 to 48.7 mm. Full step counts use
the stricter flight, forward travel, support and settled-landing criteria.
The fourth case passed the 5 cm drift limit by only about **0.13 mm**; lateral
drift remains a concern, and four simulated cases do not establish hardware
reliability. Body tilt is higher than the approved baseline's 1.1-1.4 degrees
RMS, while complete stepping improved from no trial passing to all four.

Evidence: `eval-model_2549/report.json`, `eval-model_2549/analysis.json`,
`ranking.json`, and `selected/selection.json`. The separate rendered trials
use a different seed and must retain their own reports; the four-case numbers
above must not be presented as measurements from the videos.

## Recorded trials and export

Both normal-speed recordings are complete. Each contains the first 32-second
trial and a 2-second final-frame hold, with seed 2522 and nominal motors.

| Recording | Forward | Signed drift | Tilt RMS / peak | Complete steps FL / RL / FR / RR | Qualified |
|---|---:|---:|---:|---|---|
| `video-pair-0/walk.mp4`, FL + RR first | 48.65 cm | +2.12 cm | 2.35 / 4.95 degrees | 5 / 4 / 4 / 5 | Yes |
| `video-pair-1/walk.mp4`, FR + RL first | 48.90 cm | -5.72 cm | 2.58 / 7.22 degrees | 5 / 5 / 5 / 5 | No: drift exceeds 5 cm |

Neither recording fell, and both swing feet were clear at all sampled advance
points in both recordings. The second recording is retained in full, including
its failed drift criterion. The comparison presents the approved starting
model, then the selected model, using the same 2.5-20.5 second interval at
normal speed. Saved frames from both full recordings and the comparison were
visually reviewed; this frame review is not a complete video playback audit.

Export: `selected/actor.npz`, SHA256
`d166ccee1f68008f2e8aae31d73717e43d7b8e22cc2a117d82f6cbf6d2a91a44`.
`selected/actor.json` records the architecture, normalizer, action/servo order,
source checkpoint and limitations. The exporter checked stored array shapes
and finiteness only; no NumPy-versus-training inference parity was run.
The source/asset identities are in `train/contract.json`, with saved source
copies under `train/provenance/`. These define the controller integration.

The local phone review retains its URL and all prior video routes:
<http://192.168.1.54:8765/7c89c5858ecf176b03a14b81/>.
The three new routes are `overnight.mp4`, `overnight-pair-0.mp4`, and
`overnight-pair-1.mp4`. Keep the PC awake and the phone on the same network.

## Completed overnight workflow

- 2,000 further PPO updates on 512 simulated robots, preserving actor, critic,
  observation normalizers and optimizer state. The learning rate is reduced
  to 0.00005. The environment, rewards, torque limits and calibration remain
  unchanged. Completed final iteration: 2549.
- Independent evaluations at checkpoints 600, 1000, 1400, 1800, 2200 and 2549,
  compared with the approved checkpoint's saved four-condition evaluation.
- `selected/model.pt`, a Gaussian mean actor export `selected/actor.npz` with
  identity metadata, and normal-speed videos of both initial diagonals.
- The original walking qualification criteria, retained even if every model
  fails. The selected file is a review candidate, not hardware approval.

## Controller compatibility work required before deployment

The installed V23 NumPy runtime uses a 32-output Beta actor, a different
reference/phase clock and different motor-output limiting. These new weights
use a 16-output Gaussian mean. Both observation vectors happen to contain 82
values; this does not make their semantics interchangeable. Do not put these
weights into the old model slot or bypass its identity checks.

Port the exact timed diagonal controller from the training-source snapshot:

- Joint-major order, with legs FL/RL/FR/RR. Servo IDs are
  `1,7,4,10,2,8,5,11,3,9,6,12,13,15,14,16`. Preserve calibrated directions,
  centers and travel, including the user's reversed servos 6 and 14.
- Three ELU hidden layers, normalized inputs `(x-mean)/(std+0.01)`, linear
  16-action mean and action clamping. No Beta conversion or random sampling
  on hardware. The exported actor has not yet had an inference-parity check.
- 60 Hz policy output; 100 ms action filtering; residual scales
  0.14/0.22/0.35/0.15 rad by joint group. Copy the reference's swing projection
  and calibration limits exactly, not just its final joint target formula.
- Timed transfer/lift/advance/lower/settle/push stages with fractions
  0.18/0.18/0.22/0.18/0.10/0.14 and the pair-aware phase clock. The training
  controller includes immediate support-ankle drive during lift and bounded
  filtered pitch compensation. Its stage clock is different from the older
  deployment's foot phases and from the hardcoded gait's encoder gates.
- 80–100 mm intended stride, 35 mm lift, 80 mm fore/aft and 288 mm down stance,
  the training neutral used for observation subtraction, and initial-pair
  half-stride pose. Neutral is the asset/config default pose; do not substitute
  the IK reset pose for the observation normalization neutral.
- Actual completed output history, shared interpolation across all 16 joints,
  4-degree packet cap and 60/200/200/200 degrees/s group limits. No burst of
  accumulated travel after a missed output frame. Match deadline and freshness
  handling before enabling this experimental controller.
- Finish the current landing and hold when stopping normally; use the existing
  hardware emergency stop path for an immediate stop. Simulation contact
  forces and sole positions must not become required actor inputs.

No Pi connection, restart, upload or movement is part of the overnight job.
The matching Pi controller and UI integration are not implemented by this
training workflow. The next session must complete and review that integration
before describing the exported candidate as ready to run.

## Proposed first real trial

When the user returns, establish current support/clearance and stop readiness;
old readiness confirmations are not evidence about tomorrow's setup. Start
with a short supported motion check using the matching controller, then a
short floor trial if tracking and stop behavior are satisfactory. Record joint
targets/positions, applied-output timing, IMU tilt and video. Inspect rear-left
lift, ankle timing and lateral drift specifically. These physical actions and
their checks have not been performed overnight.
