# Front-right 190 mm direct lift curriculum

## Outcome

This experiment answers the isolated mechanical/control question before adding
another foot placement: can the real-test-constrained robot raise one foot at
least `190 mm` and remain balanced? The answer in the fixed Isaac Sim scene is
**yes**.

The task starts from four-foot support beside the known staircase, shifts the
body over the other three feet, and lifts only `front_right`. The staircase is
still the exact full-size geometry used by the stair work: `180 mm` rise and
`250 mm` tread depth. It does not first place `front_left`; removing that
two-foot sequence eliminates the lateral drift that stopped v12/v13's earlier
post-transfer experiment.

The `8,192`-step PPO run completed `28/28` training episodes successfully. A
mastery gate required two physical successes before each progression from
`15`, `20`, `35`, `60`, `100`, `140`, to `190 mm`; the final level accumulated
10 successes. Strict deterministic evaluation then passed `3/3` episodes:

| Metric | Strict deterministic result |
| --- | ---: |
| Front-right maximum lift | `204.61`, `204.84`, `205.05 mm` |
| Required lift and hold | `190 mm` for `0.50 s` |
| Minimum support contact fraction | `1.00` in all episodes |
| Maximum support-foot slip | `3.42 mm` |
| Maximum body tilt | `2.39 deg` |
| Minimum base clearance | `362.57 mm` |
| Successes | **3/3** |

The reviewed deterministic recording is
`reviews/ppo-stairs-v13-front-right-190mm-lift-success.mp4`. Its selected
episode reaches `204.61 mm`, holds for `0.50 s`, keeps all three support feet in
contact, and records `3.39 mm` maximum support slip.

## Policy and sensing contract

This is a placement-reference-plus-PPO-residual policy. RGB camera pixels are
**not** policy input. It uses simulated IMU signals, joint position/velocity,
previous action, foot progress/contact-derived placement state, and an analytic
height profile for the already-known fixed staircase. The analytic terrain
profile must be replaced by a hardware-reproducible estimator before unknown
stairs are attempted.

No higher-friction foot material was enabled. The authored world material was
kept, and measured support slip stayed far below the `25 mm` measurable-slip
gate. The result therefore supports balance/sequence work before traction
changes.

## Reproduction

```powershell
& C:\isaacsim\python.bat simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v13_front_right_stabilized_lift.yaml `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v13-direct-front-right-190mm-8192-seed195 `
  --total-timesteps 8192 --seed 195 --device cpu

& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v13_front_right_stabilized_lift.yaml `
  --model simulation\isaac\models\ppo-stairs-v13-front-right-190mm-lift-small\drobot_stairs_ppo_final.zip `
  --episodes 3 --seed 195 --device cpu `
  --report simulation\isaac\output\rl\ppo-stairs-v13-direct-evaluation.json
```

## Limitation and next experiment

This proves one-foot lift-and-hold in a fixed reset, not foot placement, stair
ascent, robustness to geometry error, or hardware readiness. The next training
stage should keep this learned right-foot lift frozen and add a short forward
placement onto the first `250 mm` tread, followed by a controlled return to
four-foot support. Add vision only after that fixed-geometry placement is
repeatable; revisit traction only if contact logs show support slip approaching
the measurable threshold.
