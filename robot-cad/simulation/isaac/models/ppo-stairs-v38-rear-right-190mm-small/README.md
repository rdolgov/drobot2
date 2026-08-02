# V38 positive-margin rear-right 190 mm lift

This package is a bounded `512`-step PPO milestone for the mixed-height first
stair pose. The verified V10 and V17 policies put the two front feet on the
first tread, an analytic composite-COM controller transfers support away from
rear-right, and this compact nine-action PPO policy corrects only the three
support legs while the frozen V17/V35 composition raises rear-right.

The isolated task terminates after rear-right sustains a physical lift of at
least `0.190 m`. It does not lower that foot onto the tread, move rear-left, or
complete a stair.

## Physical and sensing contract

- Stair rise: exactly `0.180 m`
- Stair tread depth: exactly `0.250 m`
- Applied joint-effort cap: `0.8825985 N m`, measured in the real leg test
- Rear-right lift gate: physical foot-tip rise `>= 0.190 m` for `0.50 s`
- Rear transfer gate: support margin `>= 0.015 m`, body tilt `<= 12 deg`, all
  three support contacts retained, and rear-right unloaded to `<= 1 N`
- Policy input: IMU/proprioception, joint state, contact/load, computed
  composite COM/support state, phase, previous action, and analytic stair
  geometry
- Camera: external recording only; RGB pixels never enter policy inference

## Measured result

- Training: `512` PPO steps, seed `847`, `77.78 s`, CPU
- Independent deterministic evaluation: seed `848`, `1/1`, no failure reason
- Physical rear-right foot-tip rise: `0.217319 m`
- Rear transfer completion margin: `0.038519 m`
- Maximum body tilt: `11.2504 deg`
- Maximum measured support slip: `0.057582 m`
- Requested PD effort at or above 95% of the cap: `42.29%` of samples
- Model-contract verification: PASS
- Loaded PPO-algorithm verification: PASS

The `57.6 mm` simulated support slip and frequent effort saturation are
important limitations. This result proves clearance and a no-fall hold in the
current contact model; it is not hardware-readiness evidence. Real rubber-pad
friction and compliance should be measured before adding RGB vision or
attempting a physical stair.

## Contents

- `drobot_stairs_ppo_final.zip`: compact nine-action rear-right support policy
- `drobot_stairs_ppo_final.zip.contract.json`: schema-2 environment and PPO
  integrity contract
- `quadruped_stairs_v38_positive_margin_rear_transfer.yaml`: exact accepted
  task configuration
- `training_report.json`: bounded seed-847 training provenance
- `evaluation_report_seed848.json`: independent deterministic acceptance
- `recording_report_seed848.json`: full-resolution external-camera provenance
- `RESULTS.md`: outcome, rejected variants, and next gate

The compact repository video is
[`reviews/ppo-stairs-v38-rear-right-190mm-lift-seed848.mp4`](../../../../reviews/ppo-stairs-v38-rear-right-190mm-lift-seed848.mp4).
The same evidence and review notes are hosted privately at
<https://drobot-design-review.romka.chatgpt.site>.

## Reproduce training

Run from `robot-cad`:

```powershell
& simulation\isaac\rl\stairs\train_stairs_v38_rear_right_lift_small.ps1 `
  -OutputDir simulation\isaac\output\rl\ppo-stairs-v38-rear-right-190mm-512-seed847 `
  -Seed 847
```

## Reproduce independent evaluation

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation\isaac\models\ppo-stairs-v38-rear-right-190mm-small\quadruped_stairs_v38_positive_margin_rear_transfer.yaml `
  --model simulation\isaac\models\ppo-stairs-v38-rear-right-190mm-small\drobot_stairs_ppo_final.zip `
  --episodes 1 --seed 848 --device cpu --active-steps 1 `
  --placement-level left-center-tread-load --episode-seconds 65 `
  --maximum-lateral-deviation-m 0.30 `
  --leg-model front_right=simulation\isaac\models\ppo-stairs-v10-180mm-25cm-front-right-placement-small\drobot_stairs_ppo_final.zip `
  --leg-model front_left=simulation\isaac\models\ppo-stairs-v17-single-foot-190mm-small\drobot_stairs_ppo_final.zip `
  --leg-model rear_right=simulation\isaac\models\ppo-stairs-v38-rear-right-190mm-small\drobot_stairs_ppo_final.zip `
  --leg-residual-support-only rear_right --leg-compact-action rear_right `
  --leg-residual-scale rear_right=1.0 `
  --leg-base-model rear_right=simulation\isaac\models\ppo-stairs-v17-single-foot-190mm-small\drobot_stairs_ppo_final.zip `
  --leg-base-swing-only rear_right `
  --leg-base-residual-model rear_right=simulation\isaac\models\ppo-stairs-v35-rear-right-190mm-lift-small\drobot_stairs_ppo_final.zip `
  --leg-base-residual-scale rear_right=0.5 `
  --leg-base-residual-swing-only rear_right `
  --leg-base-residual-compact-action rear_right
```

## Integrity

- Model SHA-256: `e0a61ca2e5daca79528aa30a3b07c28abd176c852130a2d66f004be04d30789d`
- Config SHA-256: `8f34e7c5c22486d4cdcc74e795ab117e0d0336a326ef7416b5fd2212f3269f80`
- Full-resolution video SHA-256: `e2e70210fb0339645716dab58ea315bf1a43c598b5a6e8f600b5431333f36737`
- Compact repository video SHA-256: `b113e33b472cd11ff3c5dd694ba811fd9a7e6aa06dd4c394e8fd3f812707470c`
- Review still SHA-256: `a2b2c12bb3e779694affab583d26d88fbcfc0c8daf616b4cf186c20e2d4ec193`
