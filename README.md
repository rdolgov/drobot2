# drobot2

Monorepo for the Drobot2 robot. Each major engineering area lives in its own
top-level folder so CAD, simulation, control, firmware, and related work can
evolve together without becoming separate repositories.

## Project areas

- `robot-cad/` — Build123d models, mechanical specifications, validation,
  generated-artifact workflows, and CAD review tooling.
- `sim/` — reserved for future simulation work.

Generated outputs and local virtual environments are intentionally ignored.
See `robot-cad/README.md` for CAD setup, generation, validation, and preview
instructions.
