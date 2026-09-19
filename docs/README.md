# Drobot2 documentation

This directory owns documentation that applies across two or more top-level
project areas. Theme-specific design notes, validation evidence, and operating
instructions stay with the package that owns them.

## Shared documents

- [Executable Drobot2 modeling manual](../simulation/mathematical_model/Drobot2_Model_Manual.ipynb) — step-by-step mathematics and Python implementations, ending in a reusable kinematics and dynamics model; [setup and API](../simulation/mathematical_model/README.md).
- [Drobot2 mathematical model](../outputs/robot-math/index.html) — illustrated transforms, forward/inverse kinematics, Jacobians, dynamics, contacts and ankle extension; [scope and sources](../outputs/robot-math/README.md).
- [Repository layout and ownership](repository-layout.md)
- [V2 pitch-ankle running investigation](../cad/docs/ankle-v2.md) and [proposed RL comparison](../simulation/docs/rl-ankle-v2-study.md)

## Package documentation

- [CAD and mechanical design](../cad/docs/README.md)
- [Simulation and reinforcement learning](../simulation/docs/README.md)
- [Hardware control](../hardware/README.md)
- [Raspberry Pi ROS 2 onboard control](../onboard/README.md)
- [Electrical design](../electrical/README.md)
