# V45 rear-left dynamic-transfer diagnostic

This package contains a bounded `4,096`-step PPO experiment trained from the
exact accepted V44 rear-right landing boundary. It is retained as negative
evidence: the policy did **not** complete the rear-right-to-rear-left transfer
and is not a stair-climbing or hardware-deployment checkpoint.

## Physical and sensing contract

- stair rise: `0.180 m`
- stair tread depth: `0.250 m`
- applied joint-effort cap: `0.8825985 N m`, from the real one-leg test
- requested rear-left clearance after transfer: `0.190 m`
- policy observation: 95 camera-blind values from IMU/proprioception, joint
  state, contact/load, composite COM/support state, phase, previous action,
  and analytic fixed-stair geometry
- policy action: all 12 normalized joint residuals at the transfer boundary
- external camera: recording only; RGB pixels never enter training or inference

## Contents

- `drobot_stairs_ppo_final.zip`: seed-875 PPO model, SHA-256
  `c97cb2d5c4f1f7cb1cf70c011026b36033c4e71317c82e416f6d4529a48ab5c0`
- `drobot_stairs_ppo_final.zip.contract.json`: model/runtime contract
- `quadruped_stairs_v45_rear_left_transfer.yaml`: exact task configuration
- `phase_snapshot_seed870.json`: exact transfer boundary, SHA-256
  `587ffc5e447e8f36f877490dae7525848529480304565cc7cc1c04e7a1143f85`
- `training_report.json`, `monitor.csv`, and `progress_watchdog.json`: bounded
  training evidence
- `evaluation_report_seed876.json`: deterministic replay and recording report;
  `status: PASS` means the evaluator ran, while `task_success: false` records
  the failed transfer
- `constant_support_action_search_seed874.json`: 27-action authority audit
- `rear_right_outward_offset_search_seed870.json`: 5-30 mm foothold search
- `zero_action_transfer_probe_seed870.json`: exact-boundary analytic probe

The recorded failure is
[`reviews/ppo-stairs-v45-rear-left-dynamic-transfer-eval-seed876.mp4`](../../../reviews/ppo-stairs-v45-rear-left-dynamic-transfer-eval-seed876.mp4).
See `RESULTS.md` for exact commands, measurements, and the next recommended
controller change.
