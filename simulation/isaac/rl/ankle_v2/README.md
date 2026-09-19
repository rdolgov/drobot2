# Mechanical V2 ankle PPO pilot

The isolated Isaac Lab tasks `Drobot-Ankle-V2-Active`,
`Drobot-Ankle-V2-Baseline`, and `Drobot-Ankle-V2-Locked` train directly from
the three standalone URDF concepts under `cad/exports/urdf/`.

Run from repository root with the installed Isaac Sim Python:

```powershell
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v2/import_assets.py
& ./simulation/isaac/rl/ankle_v2/run_pilot.ps1 -Iterations 500 -NumEnvs 128 -Seed 1901
```

The importer retains physical foot bodies, omits the massless optical frame,
and filters directly connected collision neighbors. It does not add the
legacy runtime's feet, payload or mass correction a second time. The PPO
tasks use explicit joint-name mapping, 50/62 actor inputs, 58/70 critic inputs,
a 60 Hz controller and coupled effort/speed limits updated at 120 Hz.

`run_pilot.ps1` creates an isolated review directory with source snapshots,
commands, metrics and logs. `train.py` supervises the actual Isaac worker,
survives Kit fast shutdown, and records completion only after the final
iteration, training-return message and final checkpoint exist. Repeating the
same command/metrics path joins an existing job or reuses a matching completed
job. This serialization currently uses Windows file locking.

Read completed training output without policy execution:

```powershell
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v2/summarize_training.py `
  simulation/reviews/ankle-v2-training/<batch-directory>
```

All three first-seed jobs completed 500 iterations each on September 12.
The [archived policy weights](../../models/ankle-v2-pilot/20260912-071916-seed1901/README.md)
and the first authorized run are recorded in
[the September 12 training record](../../../docs/rl-ankle-v2-training-20260912.md).
The active ankle showed no speed advantage in this pilot's training telemetry.
`study.json` retains the broader multi-seed proposal, not an executable Hydra
configuration. The [study design](../../../docs/rl-ankle-v2-study.md) and
[mechanical investigation](../../../../cad/docs/ankle-v2.md) describe its context.
The pilot does not implement the full randomization/evaluation study, produce
print-ready brackets, or support 16-motor hardware deployment.
