# V44 early-contact rear-right landing policy

This package contains the bounded PPO support residual accepted for the first
tread's rear-right landing. It is an intermediate stair-climbing result, not a
full-staircase policy.

## Physical and sensing contract

- stair rise: `0.180 m`
- stair tread depth: `0.250 m`
- applied joint-effort cap: `0.8825985 N m`, from the real leg test
- rear-right physical lift gate: at least `0.190 m`
- policy observation: 95 camera-blind values comprising IMU/proprioception,
  joint state, contact/load, composite COM/support state, phase, previous
  action, and analytic fixed-stair geometry
- policy action: a compact nine-value support residual composed with frozen
  V17 and V35 rear-right swing policies
- external camera: recording only; RGB pixels never enter training or inference

V44 adds minimum support-margin targeting, asymmetric front-foot support reach
corrections, pitch regulation, compliant touchdown load correction, and a
post-clearance early-contact latch. The front-left `30 mm` and front-right
`90 mm` reach corrections are controller references, not robot CAD changes.

## Contents

- `drobot_stairs_ppo_final.zip`: 512-step PPO model, SHA-256
  `b05c348a427be61aaeab6b48cec550ab0e64a2a424b9e6ac2d1f00e094549a62`
- `drobot_stairs_ppo_final.zip.contract.json`: model/runtime contract
- `quadruped_stairs_v44_early_contact_rear_right_landing.yaml`: exact task
  configuration
- `training_report.json`: seed-869 bounded training report
- `evaluation_report_seed870.json`: fresh trained-policy evaluation
- `search_acceptance_seed862.json`: deterministic controller acceptance search
- `recording_report_phase_seed870.json`: successful 331-frame external-camera
  phase-local landing recording report
- `full_sequence_recording_attempt_seed870.json`: strict four-attempt
  reset-to-contact recording failure retained to document prefix instability
- `v45_rear_left_transfer_attempt_seed870.json`: four-attempt next-transfer
  failure report retained as scoped negative evidence

See `RESULTS.md` for measured results and reproduction commands.
