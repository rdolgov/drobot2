# Repository layout and ownership

Drobot2 is a monorepo with one top-level area per engineering concern:

| Area | Owns | Does not own |
| --- | --- | --- |
| `cad/` | Parametric parts and assemblies, mechanical specifications, vendor geometry, manufacturing exports, CAD reviews, and CAD-derived URDF | Simulation runtimes, training, hardware-control applications, and electrical plans |
| `simulation/` | Isaac integration, worlds, runtime validation, reinforcement learning, trained models, simulator exports, and evaluation media | Editable mechanical geometry and physical-device control |
| `hardware/` | Physical robot commissioning, calibration profiles, transport, dashboards, and bounded test applications | CAD source, simulator training, and power-system design |
| `onboard/` | Raspberry Pi ROS 2 deployment, LAN dashboard hosting, ROS command/status interfaces, and boot-service integration | Calibration ownership, gait equations, CAD, and simulation |
| `electrical/` | Power distribution, wiring, protection, schematics, BOM, and power budget | Mechanical geometry and control software |

The main cross-area handoff is the CAD-derived robot description:

```text
cad sources -> cad/exports/urdf -> simulation import/runtime
        |                         |
        |                         `-> hardware configuration reference
        |                                  |
        |                                  `-> onboard ROS 2 runtime
        `-> electrical and hardware fit constraints
```

Keep documentation beside its owner. Put a document here only when it defines
a repository-wide convention or coordinates multiple top-level areas. Commands
are run from the repository root unless a package README explicitly says to
enter that package directory.
