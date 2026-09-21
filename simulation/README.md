# Drobot simulation and learning

This top-level area owns Isaac Sim integration, validation runners,
reinforcement-learning tasks, trained policies, simulator exports, evaluation
media, and their tests and documentation.

The [September 21 diagonal walking result](docs/rl-ankle-v27-overnight.md)
links the selected checkpoint 2549, actor export, fallback, frozen sources,
evaluation reports and videos. It remains a simulation candidate requiring
matching controller integration before a physical trial.

## Layout

```text
simulation/
|-- isaac/                 Isaac runners, RL environments, and trained models
|-- exports/isaac/         Versioned USD/USDA/USDC simulator assets
|-- reviews/               Evaluation videos, images, and reports
|-- docs/                  Simulation and learning documentation
|-- tests/                 Isaac-free contract and documentation tests
|-- conversations/         Supplemental simulation engineering records
`-- scripts/               Simulation environment setup helpers
```

CAD-derived URDF inputs remain under `cad/exports/urdf/`. Simulation
outputs consume those inputs and are written under `simulation/exports/isaac/`.
Hardware profiles are read from `hardware/`.

## Setup and tests

Run commands from the repository root:

```powershell
py -m venv cad\.venv
cad\.venv\Scripts\python.exe -m pip install -e ".\cad[dev]"
.\simulation\scripts\setup_isaac_rl.ps1

cad\.venv\Scripts\python.exe -m pytest -q simulation\tests
```

Scripts that import Isaac Sim must use `C:\isaacsim\python.bat`. See the
[Isaac workflow](isaac/README.md) and the [documentation index](docs/README.md)
for exact training, evaluation, and recording commands.
