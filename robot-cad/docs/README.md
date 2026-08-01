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
| Hardware-informed residual stair task and shallow-step success | [`rl-stairs-v5/README.md`](rl-stairs-v5/README.md) | V4/v5 transfer correction, frozen-base residual control, staged stair heights, physical fork-tip shaping, success distillation/replay, release model, evaluation, or recording changes |
| Hardware-profiled full-size stair task (`180 mm` rise, `250 mm` tread) | [`rl-stairs-v6-180mm/README.md`](rl-stairs-v6-180mm/README.md) | V6 fixed-depth height curriculum, strict four-foot success, foot-sequence observations, 180 mm training/evaluation, release model, recording, or hosted review changes |
| Force-verified fixed-tread placement and ordered-leg follow-up (`180 mm` rise, `250 mm` tread) | [`rl-stairs-v8-placement/README.md`](rl-stairs-v8-placement/README.md) | V8 placement phases, force/contact/slip gates, balance transfer, single-foot model/evaluation/recording, v9 ordered-leg integration, or body-transfer changes |
| Cheap multi-zone ToF stair perception (`VL53L5CX`, `180 mm` rise, `250 mm` tread) | [`rl-stairs-v7-vl53l5cx/README.md`](rl-stairs-v7-vl53l5cx/README.md) | V7 ray geometry, sensor cadence/noise/latency, hardware-reproducible terrain observation, sensor-policy transfer, training/evaluation, model, recording, or hosted review changes |
| Simplified `190 mm` supported and unsupported single-foot-lift PPO skills | [`rl-foot-lift-v1/README.md`](rl-foot-lift-v1/README.md) | Foot-lift IK reference, weight transfer, PPO balance residual, support-triangle margin, observation/reward/gates, hardware effort profile, training/evaluation, model, recording, or hosted review changes |
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
