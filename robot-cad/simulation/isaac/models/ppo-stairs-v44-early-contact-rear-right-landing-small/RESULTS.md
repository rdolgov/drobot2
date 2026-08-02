# V44 measured results

## Outcome

The seed-870 fresh evaluation completed all `45/45` required rear-right
contact-hold frames. It raised the physical foot tip `217.990 mm`, maintained
all three support contacts, kept a `39.443 mm` minimum support margin, limited
support slip to `14.000 mm`, and limited rear-right tread load to `10.238 N`.
The first accepted contact occurred at `10.681 deg` body tilt and `6.789 N`.
The rollout's reported maximum tilt was `12.034 deg`; the strict live accepted
contact window remained inside the `12 deg` completion gate and no failure was
raised.

The 512-step seed-869 PPO job completed in `77.254 s`. Its bounded horizon ends
before the rear-right contact phase (normally near step 616), so the training
report contains zero completed episodes and is not by itself landing proof.
The separate fresh evaluation is the acceptance evidence.

The deterministic seed-862 search also completed `45/45` hold frames with
`218.873 mm` lift, `39.660 mm` minimum support margin, `18.485 N` maximum tread
load, and `13.809 mm` support slip.

## Small training

From the repository root:

```powershell
& simulation\isaac\rl\stairs\train_stairs_v44_early_contact_rear_right_landing_small.ps1 `
  -OutputDir simulation\isaac\output\rl\ppo-stairs-v44-early-contact-rear-right-landing-512-seed869 `
  -Seed 869
```

The script initializes the compact nine-action residual from V38, composes the
frozen V17 swing base and `0.5 * V35` compact swing residual, trains for exactly
512 PPO steps, and writes its generated model and report below the ignored
`simulation/isaac/output/rl/` directory.

## Fresh evaluation and recording

The accepted composition uses V10 for front-right, V17 for front-left, and the
V44 model for the rear-right support residual. Rear-right swing is V17 plus
`0.5 * V35`. The checked-in JSON reports retain absolute source paths, hashes,
Isaac Sim version, policy dimensions, and all measured gates.

The accepted external-camera recording replays candidate 91 from the same
cached rear-right phase used by fresh evaluation and encodes only after the
physical landing completes:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\search_rear_right_landing.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v44_early_contact_rear_right_landing.yaml `
  --seed 870 --candidate-start 91 --candidate-limit 1 `
  --maximum-target-steps 680 `
  --support-residual-model simulation\isaac\models\ppo-stairs-v44-early-contact-rear-right-landing-small\drobot_stairs_ppo_final.zip `
  --record-video reviews\ppo-stairs-v44-rear-right-landing-phase-seed870.mp4 `
  --record-thumbnail reviews\ppo-stairs-v44-rear-right-landing-phase-seed870.png `
  --record-fps 30 --record-width 640 --record-height 360 `
  --report simulation\isaac\models\ppo-stairs-v44-early-contact-rear-right-landing-small\recording_report_phase_seed870.json
```

The resulting 331-frame, 30 fps, `640 x 360` phase-local MP4 completes `45/45`
hold frames at step 662. It is intentionally not presented as a
reset-to-contact replay.
The strict full-sequence recorder rejected four attempts (`body_tipped`, two
`body_transfer_failed`, then `body_tipped`); that report is retained as
`full_sequence_recording_attempt_seed870.json`.

## Limitations and next gate

This result proves three first-tread placements through rear-right. It does not
prove rear-left placement, body transfer onto the tread, a second stair, or a
complete staircase. A V45 probe attempted to expose the rear-left transfer
state four times at seed 870; each prefix terminated with `body_tipped` at the
rear-right-to-rear-left boundary. The next training target is attitude and COM
regulation through that transition, before adding rear-left swing or vision.
