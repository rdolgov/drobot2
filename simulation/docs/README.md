# Simulation and learning documentation index

Run documented commands from the repository root. A smoke test proves that a
pipeline executes; it does not prove that a policy converged or transfers to
hardware.

| Topic | Owning document |
| --- | --- |
| Executable kinematics and dynamics manual; reusable NumPy model | [Model manual and setup](../mathematical_model/README.md), [Jupyter notebook](../mathematical_model/Drobot2_Model_Manual.ipynb) |
| Isaac import, sensors, standing, gait, and runtime checks | [`../isaac/README.md`](../isaac/README.md) |
| Flat-ground reinforcement-learning task | [`rl-training.md`](rl-training.md) |
| Mechanical V2 pitch-ankle RL comparison and first training pilot | [Study design](rl-ankle-v2-study.md), [September 12 training record](rl-ankle-v2-training-20260912.md) |
| Printable V2.1 ankle, completed PPO results and walk recordings | [Results, videos and training workflow](rl-ankle-v21-planted.md) |
| V2.3 R3 solid-sole ankle with longer strides | [Completed results and recordings; selection criteria not met](rl-ankle-v23-stride.md) |
| V2.3 rear-foot unloading and weight transfer | [Completed recordings; rear-right steps learned, four-leg criteria not met](rl-ankle-v23-transfer.md) |
| V2.3 active ankle stair climbing with RealSense D405 depth | [Completed first stage: no successful climbs](rl-ankle-v23-stairs-d405.md) |
| V2.3 stair curriculum and planned foot placements | [Full floor-start climbs: three teacher and one depth-student training successes on 20 mm risers](rl-ankle-v23-curriculum.md) |
| V2.3 original full-size stair objective | [Active continuation: four 180 mm risers and 250 mm treads](rl-ankle-v23-fullheight.md) |
| V2.3 turning, sideways and staged foot-placement PPO experiments | [Seven reward strategies: one held placement, no full climbs](rl-ankle-v23-strategy-screen.md) |
| External rear-battery smooth walking (V20) | [`rl-external-rear-payload-walking-v20.md`](rl-external-rear-payload-walking-v20.md) |
| Low-speed residual crawl (V22) | [`rl-low-speed-residual-crawl-v22.md`](rl-low-speed-residual-crawl-v22.md) |
| Higher-speed straight residual crawl (V23) | [`rl-higher-speed-straight-crawl-v23.md`](rl-higher-speed-straight-crawl-v23.md) |
| Padded-feet forward-bias residual crawl (V24) | [`rl-padded-feet-forward-bias-v24.md`](rl-padded-feet-forward-bias-v24.md) |
| Robust straight low-stance residual crawl design (V25) | [`rl-robust-straight-low-stance-v25.md`](rl-robust-straight-low-stance-v25.md) |
| Forward-biased cycle-gated straight crawl (V28) | [`rl-forward-biased-cycle-gated-v28.md`](rl-forward-biased-cycle-gated-v28.md) |
| Schedule-matched support straight crawl (V29, rejected V30 bootstrap) | [`rl-schedule-matched-support-v29.md`](rl-schedule-matched-support-v29.md) |
| Symmetry-gated robust straight crawl research profile (V30, nominal candidate rejected) | [`rl-symmetry-gated-robust-straight-v30.md`](rl-symmetry-gated-robust-straight-v30.md) |
| Fast run and graceful stop research policy (V31, rejected on hardware) | [`rl-fast-run-stop-v31.md`](rl-fast-run-stop-v31.md) |
| Hardware-matched run and graceful stop continuation (V32) | [`rl-hardware-matched-run-stop-v32.md`](rl-hardware-matched-run-stop-v32.md) |
| Integrated-target-velocity run and graceful stop (V34) | [`rl-integrated-target-velocity-run-stop-v34.md`](rl-integrated-target-velocity-run-stop-v34.md) |
| Foot-warp trot with residual stabilization and phase-aligned stop (V43 guarded candidate) | [`rl-foot-warp-trot-run-stop-v43.md`](rl-foot-warp-trot-run-stop-v43.md) |
| Flat-contact hardware-matched corrective profile (V44; trained/evaluated, unqualified, not deployed) | [`rl-flat-contact-hardware-matched-v44.md`](rl-flat-contact-hardware-matched-v44.md), [training record](rl-v44-training-20260905.md) |
| Active V44 actuator/contact continuation (no deployment) | [Hypothesis and training record](rl-v44-actuator-calibration-20260905.md) |
| Original stair task | [`rl-stairs/README.md`](rl-stairs/README.md) |
| Corrected close-start stair task | [`rl-stairs-v2/README.md`](rl-stairs-v2/README.md) |
| Hardware-informed stair task | [`rl-stairs-v3/README.md`](rl-stairs-v3/README.md) |
| Residual shallow-step task | [`rl-stairs-v5/README.md`](rl-stairs-v5/README.md) |
| Full-size 180 mm stair task | [`rl-stairs-v6-180mm/README.md`](rl-stairs-v6-180mm/README.md) |
| VL53L5CX stair perception | [`rl-stairs-v7-vl53l5cx/README.md`](rl-stairs-v7-vl53l5cx/README.md) |
| Fixed-tread placement | [`rl-stairs-v8-placement/README.md`](rl-stairs-v8-placement/README.md) |
| Pure parallel stair PPO | [`rl-stairs-pure-parallel/README.md`](rl-stairs-pure-parallel/README.md) |
| Foot-lift PPO | [`rl-foot-lift-v1/README.md`](rl-foot-lift-v1/README.md) |
| Scripted stair feasibility | [`stair-feasibility/README.md`](stair-feasibility/README.md) |
| Later stair transfer audits | [`rl-stairs-v12-lift-hold/README.md`](rl-stairs-v12-lift-hold/README.md), [`rl-stairs-v13-direct-lift/README.md`](rl-stairs-v13-direct-lift/README.md), [`rl-stairs-v19-v21-transfer-audit/README.md`](rl-stairs-v19-v21-transfer-audit/README.md), and [`rl-stairs-v22-live-transfer/README.md`](rl-stairs-v22-live-transfer/README.md) |

Mechanical sources remain under [`../../cad/`](../../cad/README.md),
hardware commissioning under [`../../hardware/`](../../hardware/README.md), and
electrical design under [`../../electrical/`](../../electrical/README.md).
