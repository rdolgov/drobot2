# V91 front-left 190 mm raise-and-hold

V91 is a deliberately isolated stair prerequisite. The robot starts from
four-foot support beside the exact `180 mm` rise, `250 mm` tread staircase,
raises only its front-left foot at least `190 mm`, and must hold the clearance
for `0.50 s` without losing a support foot or exceeding the balance and slip
limits.

The policy resumed the verified V80 checkpoint for 1,024 additional PPO steps
at seed 1043. The real-test joint directions and limits remain active and the
applied effort is capped at `0.8825985 N m`. All three completed training
episodes passed, with `203.058-207.323 mm` lift, at most `3.064 deg` body tilt,
and at most `3.509 mm` support slip.

Fresh deterministic seed-1044 evaluation passed `5/5` episodes:

| Metric | Result |
| --- | ---: |
| front-left lift | `205.004-208.036 mm` |
| required clearance hold | `0.50 s` in all episodes |
| maximum body tilt | `2.333 deg` |
| maximum support-foot slip | `3.308 mm` |
| minimum support-foot normal load | `10.324 N` |

The 166-frame H.264 evidence video is
`reviews/ppo-stairs-v91-front-left-190mm-hold-seed1044.mp4`. Its selected
episode reached `207.755 mm` lift, `2.133 deg` tilt, and `3.208 mm` support
slip.

The policy is camera-blind. Its observation contains IMU, joint/proprioceptive
state, previous action, support contact/load, and analytic terrain geometry.
The RGB camera is used only to record review evidence. This result proves the
isolated lift-and-balance prerequisite; it does not prove weight transfer from
a foot already landed on the stair or a complete stair climb.

## Reproduce

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\train_stairs_v91_front_left_190mm_hold_ppo.py

& C:\isaacsim\python.bat simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation/isaac/rl/stairs/quadruped_stairs_v15_front_left_stabilized_lift.yaml `
  --model simulation/isaac/output/rl/ppo-stairs-v91-front-left-190mm-hold-1024-seed1043/drobot_stairs_ppo_final.zip `
  --episodes 5 --seed 1044 --device cpu --active-steps 1 `
  --placement-level front-left-stabilized-190mm-lift-hold `
  --maximum-lateral-deviation-m 0.20
```

SHA-256 values:

- model: `07c19d7a82e0809fe0535cd09ef6d550bcbae03628d07eaf6594aa659d599eb9`
- training report: `91e8d5898b737c1e32e390fc7e96c3b705cb74bfde21d1ac3bdbcd1a3202f20c`
- evaluation report: `c09c97a77e566408ca5fd8433b0f35dbb0bd2c8c7bd517d3ad74d623f835155f`
- recording report: `12fca8e7609514674f75b09081cd9586abb0608ad876493d650d02920bed78d8`
- MP4: `16a19cc7b31bed04d28f0fe72fa76ad4dc8b6e4edc1b35f0a92694642a94e05a`
