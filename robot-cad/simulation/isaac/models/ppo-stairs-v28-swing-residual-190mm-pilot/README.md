# V28 post-transfer swing-residual pilot

This package is an **experimental small-run artifact**, not a promoted stair
policy. It tests whether PPO can correct only the three front-left swing joints
after the verified V10 front-right placement and inter-leg transfer.

## Fixed physical contract

- Stair tread depth: **0.25 m**
- Stair rise: **0.18 m**
- Joint effort cap: **0.8825985 N m**
- Front-left frozen base policy: V17 190 mm single-foot lift
- PPO residual mask: front-left hip abduction, hip flexion, and knee only
- Residual scale: 0.20
- Policy inputs: IMU, joints, contacts/load, composite-COM error, previous
  action, and the known analytic stair profile
- RGB camera pixels: recording only; not policy input

## Small-run evidence

- Training: 2,048 PPO steps, seed 481, live precursor replay on every reset.
- Recent training episodes: 4/6 passed at 216.4-220.0 mm; the final two failed
  at 171.0 and 169.9 mm.
- Independent checkpoint-512 evaluation: 2/5. The 512-step mean action was
  still exactly zero, so this is the frozen V17 baseline rather than learned
  improvement.
- Recorded final-policy seed 510: pass at 219.8 mm lift, 16.2 mm maximum
  support slip, 11.0 degrees maximum body tilt, and no failure flags.
- Same-state authority search: a +0.20 knee correction added only 5.5 mm in
  the hard reset and caused excessive slip when not phase gated.
- Higher 225-265 mm analytic apex targets increased saturation and reduced
  measured clearance. They were rejected.

The final model is included so the pilot can be reproduced and inspected, but
it should not replace the V17 isolated-lift model or V27 front-pair packet.

## Files

- `drobot_stairs_ppo_final.zip`: final 2,048-step residual policy
- `drobot_stairs_ppo_final.zip.contract.json`: model/environment contract
- `training_report.json`: complete training and recent-episode metrics
- `evaluation_report_seed490_checkpoint512.json`: independent 5-episode check
- `recording_report_seed510.json`: metrics for the published representative run
- `swing_bias_search_seed500.json`: bounded swing-action authority sweep
- `swing_clearance_search_seed500.json`: phase-gated clearance sweep

## Evaluate the final pilot

From `robot-cad`:

```powershell
& 'C:\isaacsim\python.bat' simulation/isaac/rl/stairs/evaluate_stairs_ppo.py `
  --config simulation/isaac/rl/stairs/quadruped_stairs_v24_front_pair_conservative_support.yaml `
  --model simulation/isaac/models/ppo-stairs-v28-swing-residual-190mm-pilot/drobot_stairs_ppo_final.zip `
  --episodes 5 --seed 520 --device cpu --active-steps 1 `
  --placement-level left-supported-190mm-lift `
  --maximum-lateral-deviation-m 0.20 `
  --leg-model front_right=simulation/isaac/models/ppo-stairs-v10-180mm-25cm-front-right-placement-small/drobot_stairs_ppo_final.zip `
  --leg-model front_left=simulation/isaac/models/ppo-stairs-v28-swing-residual-190mm-pilot/drobot_stairs_ppo_final.zip `
  --leg-base-model front_left=simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
  --leg-base-swing-only front_left `
  --leg-residual-scale front_left=0.20 `
  --leg-residual-swing-only front_left
```

Model SHA-256:
`fc12fdb75515872750db5561d7d3e2a34f3a1d3fbbea4b20a670611aca9cd5a3`
