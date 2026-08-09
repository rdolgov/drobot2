# Force-verified per-leg single-tread placement PPO v8-v10

## Decision

The explicit per-leg fixed-geometry curriculum now passes for both front feet.
The v8 `2,048`-step PPO raises and force-places front-left `5/5`; the mirrored
v10 `2,048`-step PPO raises front-right `201.9-204.4 mm` and force-places it
`5/5` on the same exact `180 mm` rise and `250 mm` tread.

The v9 integration now also passes a separate all-feet-loaded body-transfer
gate between those skills. Under the conservative `200 mm` lateral corridor,
the composed controller raises front-right `162.8 mm` before the corridor
termination. A `230 mm` diagnostic reaches `180.9 mm`, but front-left slip
crosses the measurable `25 mm` threshold. This remains a partial front-pair
result, not a stair-climb claim and not authorization for vision or hardware
deployment.

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
- `quadruped_stairs_v10_front_right_single_tread_placement.yaml` mirrors the
  complete force-backed curriculum for front-right without changing geometry,
  observations, effort cap, or success criteria.
- `quadruped_stairs_v9_front_pair_placement.yaml` adds an incenter-targeted
  body transfer. It anchors all loaded feet, gates the handoff on force,
  support margin, pose error, base speed, body rate, and uprightness, then
  composes the independently verified left/right policies with zero residual
  action during transfer.
- Reach-limited references are clipped explicitly and counted. Support slip is
  now reported per leg, which identified the already-placed front-left foot as
  the only support crossing the `25 mm` slip threshold.
- A rubber-pad sensitivity (`1.20/1.00` static/dynamic friction versus the
  authored `0.90/0.75`) changed that slip by less than `2%` and reduced
  front-right lift. The remaining slip is controller-induced dragging, not
  evidence that higher friction alone solves the sequence.
- Placement media use a close external review camera. The camera remains
  evaluation-only.

## Source of truth

| File | Purpose |
| --- | --- |
| `simulation/isaac/rl/stairs/quadruped_stairs_v8_single_tread_placement.yaml` | Exact geometry, five-phase reference, force/slip thresholds, reward, hardware cap, and PPO settings |
| `simulation/isaac/rl/stairs/quadruped_stairs_v9_front_pair_placement.yaml` | Experimental front-left then front-right integration |
| `simulation/isaac/rl/stairs/quadruped_stairs_v10_front_right_single_tread_placement.yaml` | Mirrored force-backed front-right curriculum |
| `simulation/isaac/rl/stairs/_stair_rl_contract.py` | Pure placement curriculum, phase reference, observation, contact gate, and reward terms |
| `simulation/isaac/rl/stairs/_quadruped_stairs_env.py` | IK reference, pre-play contact views, force/load/slip metrics, ordered-leg state, and episode gates |
| `simulation/isaac/rl/_quadruped_rl_env.py` | Pre-physics-play extension hook used to register contact tensors |
| `simulation/isaac/rl/stairs/train_stairs_ppo.py` | Balance-prefix initialization and placement-aware PPO contract sizing |
| `simulation/isaac/rl/stairs/evaluate_stairs_ppo.py` | Deterministic contract verification, optional hash-verified per-leg model composition, and close placement screenshot view |
| `simulation/isaac/rl/stairs/record_stairs_ppo.py` | Exact H.264 placement recording and close review view |
| `tests/test_quadruped_stairs_rl_contract.py` | Pure geometry, phase, observation, force gate, and v8-v10 configuration checks |

The physical profile remains the one-leg real-test record: `0.8825985 N m`
per-joint effort cap, the measured joint limits, and no interpretation of the
speed register as a rad/s override.

## Reproduce the passing smoke run

From the repository root:

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

Use the same commands with
`quadruped_stairs_v10_front_right_single_tread_placement.yaml`, seed `192` for
training, seed `193` for deterministic evaluation/recording, and the release
model at
`simulation\isaac\models\ppo-stairs-v10-180mm-25cm-front-right-placement-small\drobot_stairs_ppo_final.zip`.

Compose the verified front policies around the v9 transfer with:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\train_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v9_front_pair_placement.yaml `
  --output-dir simulation\isaac\output\rl\ppo-stairs-v9-front-pair-composed-init `
  --initialize-from-balance simulation\isaac\models\ppo-foot-lift-v2-balance-190mm-small\drobot_foot_lift_ppo_final.zip `
  --initialize-only `
  --seed 194 `
  --device cpu

& C:\isaacsim\python.bat `
  simulation\isaac\rl\stairs\evaluate_stairs_ppo.py `
  --config simulation\isaac\rl\stairs\quadruped_stairs_v9_front_pair_placement.yaml `
  --model simulation\isaac\output\rl\ppo-stairs-v9-front-pair-composed-init\drobot_stairs_ppo_initialized.zip `
  --leg-model front_left=simulation\isaac\models\ppo-stairs-v8-180mm-25cm-single-foot-placement-small\drobot_stairs_ppo_final.zip `
  --leg-model front_right=simulation\isaac\models\ppo-stairs-v10-180mm-25cm-front-right-placement-small\drobot_stairs_ppo_final.zip `
  --episodes 1 `
  --active-steps 1 `
  --seed 195 `
  --device cpu
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

The mirrored front-right artifacts are:

- `reviews/ppo-stairs-v10-180mm-25cm-front-right-placement.mp4`
- `reviews/ppo-stairs-v10-180mm-25cm-front-right-placement.png`
- `reviews/ppo-stairs-v10-180mm-25cm-front-right-placement-results.json`
- `simulation/isaac/models/ppo-stairs-v10-180mm-25cm-front-right-placement-small/`

The v10 trained checkpoint passed `5/5` with `201.9-204.4 mm` clearance,
`8.75-9.95 N` tread load, `3.478 mm` maximum support slip, `100%` support
contact, and `3.326 deg` maximum tilt.

The v9 transfer itself now passes with `88.6 mm` positive support margin,
`21.3 N` retained on front-left, all support contacts, `12.1 mm/s` base speed,
and `4.17 deg` maximum tilt at handoff. The conservative composition then
reaches `162.8 mm` front-right lift with `24.34 mm` support slip before the
`200 mm` corridor gate. Comparative transfer, slip, and friction evidence is
in `reviews/ppo-stairs-v9-180mm-25cm-front-pair-results.json`.

## Next gate

Train the post-transfer controller to unload front-right without dragging the
already placed front-left foot laterally. Keep the strict `200 mm` corridor and
`25 mm` slip threshold; require the composed right foot to clear at least
`190 mm`, force-load the tread, and hold before adding either rear leg. Only
after the ordered fixed-geometry sequence succeeds should the analytic terrain
profile be replaced by depth or vision.

## Limitations

- `2,048` PPO steps validate the learning pipeline and a trained residual; they
  are not converged training.
- The tread geometry is known analytically. No camera or ToF input is used.
- The contact is the URDF's virtual `12.5 mm` fork-tip sphere, not a validated
  printed foot or rubber sole.
- Contact friction is simulated. Hardware still needs tethered,
  current-limited, thermal, voltage-sag, and emergency-stop validation.
- V8 and v10 place one front foot each. V9 proves the body-transfer handoff but
  does not yet force-place both front feet in one episode.
