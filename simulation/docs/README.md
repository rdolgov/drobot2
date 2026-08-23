# Simulation and learning documentation index

Run documented commands from the repository root. A smoke test proves that a
pipeline executes; it does not prove that a policy converged or transfers to
hardware.

| Topic | Owning document |
| --- | --- |
| Isaac import, sensors, standing, gait, and runtime checks | [`../isaac/README.md`](../isaac/README.md) |
| Flat-ground reinforcement-learning task | [`rl-training.md`](rl-training.md) |
| External rear-battery smooth walking (V20) | [`rl-external-rear-payload-walking-v20.md`](rl-external-rear-payload-walking-v20.md) |
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
