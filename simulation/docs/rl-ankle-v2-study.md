# RL study for mechanical V2 pitch ankles

Original proposal: 2026-09-11. On 2026-09-12, the user authorized training and
an isolated 12/16-joint PPO pilot was implemented and completed for all three
designs (500 iterations each, seed 1901). See the
[training record](rl-ankle-v2-training-20260912.md) for actual execution and
limitations. The broader multi-seed study below remains a proposal. Mechanical
V2 is independent of the existing V43/V44 gait-version numbers.

PPO can learn ankle coordination alongside hip and knee commands. The present
task is to establish whether the added control authority outweighs the weight
and actuator limits. There is no evidence yet that this design runs faster.

The [mechanical investigation and three URDFs](../../cad/docs/ankle-v2.md)
provide local concept models. The machine-readable
[study contract](../isaac/rl/ankle_v2/study.json) specifies the comparison,
joint order, observations, budgets and proposed acceptance criteria. It is a
proposal with `runnable: false`, not a Hydra config. The new
[pilot training package](../isaac/rl/ankle_v2/README.md) registers separate
tasks and launches a limited implementation of this comparison.

## Comparison that answers the question

| Variant | Movable joints | Nominal mass | What it tells us |
| --- | ---: | ---: | --- |
| Rigid rectangular feet | 12 | 3.18 kg | Common freshly trained baseline |
| V2 hardware with mechanically locked ankles | 12 | 3.50 kg | Cost of added mass and geometry |
| V2 active pitch ankles | 16 | 3.50 kg | Benefit of actively rotating the same feet |

An active policy beating only the weighted locked model is insufficient: the
complete design must also beat the lighter baseline. All three have the same
foot area, ground clearance, base mass assumptions and common actuator limits.
The new URDFs calibrate baseline mass differently from the old runtime, so do
not compare their scores directly against historical V43/V44 numbers.

Use five paired training seeds and separate held-out evaluation seeds from
the contract. Train all variants from scratch with the same PPO settings,
hidden-layer sizes, command distribution, reward terms, domain randomization
and environment-step budget. A 128-environment, 64-step, 500-iteration pilot
is 4,096,000 environment transitions per variant per seed; it is a proposed
starting budget, not a convergence claim. If no variant learns a usable gait,
the result is inconclusive. Extend matched budgets or repair the experiment
before judging the morphology.

Start with stand, step and stop at 0-0.14 m/s; introduce the higher command
bands only when the common curriculum supports it. Use the same curriculum
rules for every variant. Evaluate actual forward displacement after startup
at 0.05, 0.10, 0.14, 0.18, 0.22 and 0.30 m/s. These are **commands to explore**,
not predictions that the hardware can attain them.

## Action and observation contract

Resolve joints by their names rather than assuming simulator DOF order. Keep
the existing grouping for the first 12 outputs, then append the four ankles:

1. Hip abduction: front-left, rear-left, front-right, rear-right.
2. Hip flexion: front-left, rear-left, front-right, rear-right.
3. Knee: front-left, rear-left, front-right, rear-right.
4. Ankle pitch: front-left, rear-left, front-right, rear-right (active only).

The actor remains proprioceptive: command (3), gait sin/cos (2), IMU angular
velocity/projected gravity/linear acceleration (9), joint errors (N), joint
velocities (N), and previous **controller-applied** normalized targets (N).
This gives 50 observations for N=12 and 62 for N=16. True base velocity,
height and four contact values add 8 privileged critic observations: 58/70.
If V44's additional 32 critic-context features are retained, totals become
90/102. Model any new actuator memory with explicit observation history or a
recurrent policy; changing the instantaneous vector size alone is not enough.

Use bounded position targets with per-joint elapsed-time rate limits, packet
caps, actual target history and graceful-stop handling. Start with 60 deg/s
abduction and 100 deg/s sagittal/ankle target limits for the common comparison.
The active URDF's ankle range is +/-35 degrees. These conservative screening
assumptions do not establish a motor's measured loaded operating envelope.

An optional analytic warm-start for the ankle is foot-level compensation.
The frame calculation gives positive ankle rotation about body -Y at neutral,
so positive means toe-up. With standard positive body pitch about +Y and
desired toe-up angle `alpha`, the sagittal approximation is:

```text
q_ankle_ref = alpha + body_pitch - q_hip_flexion - q_knee
```

At zero body pitch, this reduces to cancelling the hip/knee sum. Clip and
rate-limit it; use full foot-frame rotations with body roll or leg abduction.
This is an initialization idea, not implemented behavior. If comparing warm
starts, give each baseline equivalent training effort and report that study
separately from the from-scratch morphology comparison.

## Actuation and reward design

The local servo ledger gives approximately 0.981 N m rated torque, 2.942 N m
stall torque and 4.712 rad/s no-load velocity for the existing variant. For a
conservative first study, use 90% rated torque as a sustained cap and also
limit torque with speed:

```text
abs(tau) <= min(0.8825985,
                0.90 * 2.941995 * max(0, 1 - abs(qdot)/4.712389))
```

This is a screening approximation, not an identified motor model. URDF limits
alone cannot implement it. Do not grant stall torque at no-load speed or hide
the four extra motors in an unchanged power budget. Count all active joints
in positive mechanical work, RMS torque, peak torque and saturation metrics;
report mechanical cost of transport as a proxy, not battery consumption.
Motor braking/regeneration, gearbox loss and thermal behavior need additional
models before predicting endurance.

Reward sustained command tracking, uprightness, heading, lateral corridor,
four-leg participation, low slip, smooth targets and reliable stops. Foot
tilt, corner clearance and ground contact must use the **new foot bodies**.
Allow useful late-stance toe-down rotation and swing-foot orientation while
retaining flat quiet feet at a stop. A full-time flat-sole penalty would remove
the new joint's intended push-off benefit. Penalize collisions and persistent
dragging; avoid letting the policy earn speed by pitching onto a shoe corner,
sliding the chassis, hopping in place or exploiting a contact artifact.

Randomize load, friction, joint zero, rate/effort strength, contact compliance,
control latency and correlated supply sag. Model the added 16-motor electrical
load rather than assuming independent unlimited actuators. The approximately
0.3 s phase-equivalent lag in the recorded robot is not a 0.3 s communication
delay; use the [recorded-response interpretation](../../debug/walk-trials/20260905T193441588801Z-2a2a9832/actuator-constraints/README.md)
when developing an identified response model.

Accurate actuators and latency matter for RL transfer: the primary
[Sim-to-Real quadruped study](https://research.google/pubs/sim-to-real-learning-agile-locomotion-for-quadruped-robots/)
combines actuator modeling, latency and randomization to transfer learned
trotting/galloping. That supports the method; it does not establish transfer
for these concept models.

## Integration scope identified in the original proposal

Read-only source inspection found that this is more than setting
`action_space = 16`:

| Existing owner | Required work in a separate task |
| --- | --- |
| `commanded_walking_env.py` | Remove assumptions tied to `EXPECTED_ACTION_ORDER`, native 12-DOF effort handling and fixed 12-value tensor slices; use named joint/body mappings. |
| `commanded_walking_env_cfg.py` | New robot asset, 16 actions, 62 actor inputs and the chosen critic/history contract; distinct ankle actuator group. |
| `_author_rectangular_shoes` and rear-payload authoring | Bypass for these already populated assets; calling them would duplicate or overwrite model geometry/mass. |
| Contact/sole calculations | Bind four `*_foot_link` bodies; recompute corner locations and orientation from those bodies rather than distal links. |
| Symmetry and reference mappings | Extend permutation/sign maps and handle ankle references explicitly; do not append four zeros to an exported V43/V44 policy. |
| PPO task registration and evaluation | Add separate task IDs/importer and collect equal-budget, held-out results for all variants. |
| ONNX export and hardware contract | Introduce versioned 16-output metadata and later motor IDs/calibration; the current runtime accepts 12 outputs. |

Preserve foot bodies when importing fixed ankle models so all variants can
use equivalent contact sensors. Use a floating base, common contact materials,
and deliberate adjacent-link collision filtering. The concept cradle/servo
placement still needs fit work; disabling broad self-collision is not evidence
that it fits. The current working tree already contains ongoing V44 changes;
this investigation added isolated assets/docs instead of editing that task.

## Decision rule

Measure 30 seconds of moving performance followed by a 5-second stop. Report
actual speed, falls, drift, heading, duty factors, corner clearance, foot slip,
joint tracking/rate/effort saturation, mechanical energy and all failed seeds.
Select checkpoints using held-out evaluations, not final iteration number.

The proposed engineering success gate is at least **15% greater maximum
sustained accepted speed** than the rigid baseline, also beating weighted
locked, across paired seeds without worse falls/stops. The JSON records
proposed speed-tracking, corridor, heading and stop thresholds. Report seed
dispersion and the robustness cases; do not turn the 15% margin into a claim
about statistical significance.

Faster movement is not automatically running. Log all-feet-airborne intervals
with force and corner-clearance evidence, duty factor and synchronized video
before calling it aerial running. Do not reward flight so strongly that it
substitutes bouncing for useful forward motion.

If active and weighted locked are similarly slow, the new joint is not being
used effectively or actuator limits dominate. If active beats locked but
loses to rigid, the added mass has erased its benefit. A lighter passive
spring/damper ankle, smaller compliant feet, or a motor mounted higher on the
shank with a transmission are then useful separate design comparisons. A
powered-off geared ST3215 is not equivalent to a freely compliant ankle.

## Original proposal execution status (2026-09-11)

No tests, validators, builds, viewers, simulations, training jobs or hardware
connections were run. No model weights or speed results were produced. Before
training, the separate importer/task above must be implemented and its named
checks explicitly requested. The existing training commands do not yet run
this study, so no launch command is presented as if it were available.
