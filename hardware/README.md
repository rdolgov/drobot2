# Drobot hardware control

This area owns physical robot configuration, commissioning, and guarded local
control software.

## Layout

- [`robot-runtime/`](robot-runtime/README.md) — shared real-robot manifest,
  calibrated servo profiles, neutral centers, and operational helper scripts.
- [`test-apps/`](test-apps/README.md) — one-leg and four-leg commissioning
  applications. These consume `robot-runtime/`; they do not own robot state.

Mechanical design lives in `cad/`, simulation validation in `simulation/`,
and power-system planning in `electrical/`.
