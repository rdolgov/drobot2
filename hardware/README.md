# Drobot hardware control

This area owns physical robot configuration, commissioning, and guarded local
control software.

## Layout

- [`robot-runtime/`](robot-runtime/README.md) — shared real-robot manifest,
  calibrated servo profiles, neutral centers, and operational helper scripts.
- [`test-apps/`](test-apps/README.md) — one-leg and four-leg commissioning
  applications. These consume `robot-runtime/`; they do not own robot state.

The Raspberry Pi package under [`../onboard/`](../onboard/README.md) reuses
these transport, dashboard, gait, profile, and calibration sources through a
ROS 2 node; it does not maintain a second hardware configuration.

Mechanical design lives in `cad/`, simulation validation in `simulation/`,
and power-system planning in `electrical/`.
