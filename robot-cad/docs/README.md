# Project documentation index

This index assigns one Markdown owner to each design and simulation topic.
Keep the owning document current in the same commit as its source, generated
artifact, or validation change.

| Topic | Owning document | Required update trigger |
| --- | --- | --- |
| CAD generation and visual review | [`cad-workflow.md`](cad-workflow.md) | CAD workflow, tooling, snapshot, or Viewer changes |
| Imported design history | [`migration.md`](migration.md) | Source migration or provenance changes |
| Coordinate conventions | [`../specs/coordinate-system.md`](../specs/coordinate-system.md) | Datum, axis, handedness, or unit changes |
| Body, battery, wiring, camera, and IMU mechanics | [`../specs/quadruped-body.yaml`](../specs/quadruped-body.yaml) and root [`README.md`](../README.md) | Enclosure, mounting, clearance, or physical component changes |
| URDF physics and sensor contract | [`../specs/quadruped-urdf-ledger.md`](../specs/quadruped-urdf-ledger.md) | Link, joint, mass, inertia, collision, sensor, or actuator changes |
| Isaac import, camera, IMU, standing, and gait checks | [`../simulation/isaac/README.md`](../simulation/isaac/README.md) | Simulator scripts, APIs, worlds, validation, or acceptance changes |
| Reinforcement-learning walking task | [`rl-training.md`](rl-training.md) | Observation, action, reward, reset, PPO, dependency, training, or evaluation changes |
| Separate reinforcement-learning stair task | [`rl-stairs/README.md`](rl-stairs/README.md) | Stair geometry, terrain observation, curriculum, reward, PPO, transfer, training, evaluation, or recording changes |
| Corrected close-start stair task, progress watchdog, and perception plan | [`rl-stairs-v2/README.md`](rl-stairs-v2/README.md) and [`rl-stairs-v2/perception-plan.md`](rl-stairs-v2/perception-plan.md) | V2 reset, navigation/terrain observation, sensor choice, physical-height or foot-placement reward, mastery curriculum, early-abort threshold, or progress-report changes |
| Hardware-informed stair task | [`rl-stairs-v3/README.md`](rl-stairs-v3/README.md) | V3 one-leg hardware profile, runtime joint-limit or effort-cap override, expanded stair action range, smoke training, evaluation, or recording changes |
| Real-stair scripted kinematic, support, collision, and torque feasibility | [`stair-feasibility/README.md`](stair-feasibility/README.md) | Scripted stair geometry, IK trajectory, contact/support gates, actuator limits, physical-feasibility result, or decision to authorize stair RL |
| Three-motor one-leg ST3215 hardware testbed | [`../hardware/one-leg-testbed/README.md`](../hardware/one-leg-testbed/README.md) | macOS USB setup, motor ID assignment, position control, neutral calibration, telemetry, or hardware-test safety changes |
| Battery, fusing, power distribution, Raspberry Pi power, and four-leg servo bus | [`../electrical/README.md`](../electrical/README.md) | Battery, fuse, wire, connector, controller, compute-power, servo-bus topology, BOM, or power-budget changes |
| Purchasable/reference geometry provenance | [`../vendor/README.md`](../vendor/README.md) | New or replaced vendor assets |

Every topic update should state:

1. what changed and why;
2. editable source-of-truth inputs;
3. exact commands needed to reproduce it;
4. validation that was actually run and its result;
5. generated or logged outputs;
6. assumptions, approximations, and work still required.

A smoke test proves that a pipeline executes. It does not prove that a
walking policy converged, that simulation transfers to hardware, or that a
printed robot is safe.

Supplemental, review-oriented engineering records are indexed in
[`../conversations/README.md`](../conversations/README.md). Topic-owning
documents in this index remain authoritative.
