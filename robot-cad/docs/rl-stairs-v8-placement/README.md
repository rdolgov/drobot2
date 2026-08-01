# Force-verified single-tread placement PPO v8

## Decision

The simplified fixed-geometry placement gate passes. A `2,048`-step PPO smoke
policy raises the front-left foot `205.2-207.4 mm`, advances it over the exact
`180 mm` riser, lowers it onto the known `250 mm` tread, and holds measured
tread load for `0.75 s`. Deterministic evaluation passed `5/5` episodes.

This authorizes the next body-transfer and ordered-leg stage. It does not
authorize vision work, hardware deployment, or a claim that the robot climbs a
complete stair. The experimental front-left to front-right v9 integration
placed only the first foot before the body tipped; that result remains a
failure.

## What changed

- `quadruped_stairs_v8_single_tread_placement.yaml` adds an explicit five-phase
  reference: diagonal weight shift, vertical lift, horizontal advance, lower,
  and force-held landing.
- The policy retains the shared 48-value proprioceptive prefix and appends the
  fixed terrain, navigation, foot-progress, target-leg, phase, Cartesian-error,
  contact-load, support-contact, support-margin, and slip state. RGB and range
  pixels are not inputs.
- Distal-link PhysX contact tensor views are registered before physics starts.
  Success requires at least `1 N` on the target tread and at least `1 N` on
  every support foot, plus the reviewed tread window, height tolerance,
  upright gate, and continuous hold.
- Contact logs distinguish support loss from traction. The passing evaluation
  kept all three supports loaded and limited support slip to `3.393 mm`, below
  the `25 mm` measurable-slip threshold.
- `quadruped_stairs_v9_front_pair_placement.yaml` is an experimental ordered
  front-pair task. It carries the first placed foot as support while targeting
  front-right. The latest diagnostic still tipped and is not a release model.
- Placement media use a close external review camera. The camera remains
  evaluation-only.

## Source of truth

| File | Purpose |
| --- | --- |
| `simulation/isaac/rl/stairs/quadruped_stairs_v8_single_tread_placement.yaml` | Exact geometry, five-phase reference, force/slip thresholds, reward, hardware cap, and PPO settings |
| `simulation/isaac/rl/stairs/quadruped_stairs_v9_front_pair_placement.yaml` | Experimental front-left then front-right integration |
| `simulation/isaac/rl/stairs/_stair_rl_contract.py` | Pure placement curriculum, phase reference, observation, contact gate, and reward terms |
| `simulation/isaac/rl/stairs/_quadruped_stairs_env.py` | IK reference, pre-play contact views, force/load/slip metrics, ordered-leg state, and episode gates |
| `simulation/isaac/rl/_quadruped_rl_env.py` | Pre-physics-play extension hook used to register contact tensors |
| `simulation/isaac/rl/stairs/train_stairs_ppo.py` | Balance-prefix initialization and placement-aware PPO contract sizing |
| `simulation/isaac/rl/stairs/evaluate_stairs_ppo.py` | Deterministic contract verification and close placement screenshot view |
| `simulation/isaac/rl/stairs/record_stairs_ppo.py` | Exact H.264 placement recording and close review view |
| `tests/test_quadruped_stairs_rl_contract.py` | Pure geometry, phase, observation, force gate, and v8/v9 configuration checks |

The physical profile remains the one-leg real-test record: `0.8825985 N m`
per-joint effort cap, the measured joint limits, and no interpretation of the
speed register as a rad/s override.

## Reproduce the passing smoke run

From `robot-cad`:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v8_single_tread_placement.yaml `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v8-placement-2048-small `
  --initialize-from-balance simulation\isaac\models\ppo-foot-lift-v2-balance-190mm-small\drobot_foot_lift_ppo_final.zip `
  --smoke-test `
  --total-timesteps 2048 `
  --seed 142 `
  --device cpu
```

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v8_single_tread_placement.yaml `
  --model simulation\isaac\models\ppo-stairs-v8-180mm-25cm-single-foot-placement-small\drobot_stairs_ppo_final.zip `
  --episodes 5 `
  --active-steps 1 `
  --seed 170 `
  --device cpu
```

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\record_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v8_single_tread_placement.yaml `
  --model simulation\isaac\models\ppo-stairs-v8-180mm-25cm-single-foot-placement-small\drobot_stairs_ppo_final.zip `
  --active-steps 1 `
  --seed 170 `
  --fps 30 `
  --width 960 `
  --height 540 `
  --video reviews\ppo-stairs-v8-180mm-25cm-single-foot-placement.mp4 `
  --thumbnail reviews\ppo-stairs-v8-180mm-25cm-single-foot-placement.png
```

Run the pure checks:

```powershell
& .venv\Scripts\python.exe -m pytest `
  tests\test_quadruped_stairs_rl_contract.py `
  tests\test_quadruped_rl_contract.py `
  tests\test_isaac_documentation.py `
  -q `
  --basetemp .pytest-tmp-stairs-v8-placement
```

## Verified result

| Gate | Result |
| --- | ---: |
| Deterministic placements | `5 / 5` |
| Swing-foot clearance | `205.2-207.4 mm` |
| Tread normal load | `8.90-9.50 N` |
| Force-backed hold | `0.75 s` each |
| Minimum support-contact fraction | `100%` |
| Maximum support slip | `3.393 mm` |
| Maximum body tilt | `2.955 deg` |
| Measurable slip (`>25 mm`) | `no` |

The review artifacts are:

- `reviews/ppo-stairs-v8-180mm-25cm-single-foot-placement.mp4`
- `reviews/ppo-stairs-v8-180mm-25cm-single-foot-placement.png`
- `reviews/ppo-stairs-v8-180mm-25cm-single-foot-placement-results.json`
- `simulation/isaac/models/ppo-stairs-v8-180mm-25cm-single-foot-placement-small/`

The v9 diagnostic placed front-left with `9.436 N`, then failed during the
front-right transfer with `258.3 mm` support slip and `39.88 deg` tilt. Its
concise evidence is
`reviews/ppo-stairs-v9-180mm-25cm-front-pair-results.json`.

## Next gate

Add a body-transfer phase after the first landing that keeps the placed foot
world-anchored while the torso moves inside the new mixed-height support
polygon. Gate that phase on all support loads, bounded slip, and tilt before
starting the next leg. Only after an ordered fixed-geometry sequence succeeds
should the analytic terrain profile be replaced by depth or vision.

## Limitations

- `2,048` PPO steps validate the learning pipeline and a trained residual; they
  are not converged training.
- The tread geometry is known analytically. No camera or ToF input is used.
- The contact is the URDF's virtual `12.5 mm` fork-tip sphere, not a validated
  printed foot or rubber sole.
- Contact friction is simulated. Hardware still needs tethered,
  current-limited, thermal, voltage-sag, and emergency-stop validation.
- V8 places one foot only. V9 proves that simply chaining mirrored references
  is insufficient for the front pair.
