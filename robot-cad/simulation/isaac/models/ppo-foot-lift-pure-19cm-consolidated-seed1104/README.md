# Pure-PPO 19 cm supported foot lift, consolidated

This package contains the selected model-418 checkpoint and its resolved RSL-RL
and Isaac Lab configuration. It was trained with 128 GPU-parallel simulations,
direct 12-joint PPO actions, and no gait phase, selected leg, inverse kinematics,
reference trajectory, or action replay.

## Result

- Stochastic consolidation training: 375/5,580 completed episodes (6.7204%).
- Deterministic unseen seed 1105: 151/1,384 (10.9104%).
- Deterministic unseen seed 1106: 142/1,388 (10.2305%).
- Pooled deterministic evaluation: 293/2,772 (10.5700%).
- Strict gate: any foot at least 190 mm above its reset height for eight control
  steps, at least three other feet in force-verified support, and upright body.
- SHA-256 of `model_418.pt`:
  `062e82d8d4eda0cdcc396f4c6a2b3138d46698150915d098f0e79c88379adba8`.

The short visible seed-1115 review is intentionally labeled as a failed mean
sample (0/12 episodes). It demonstrates the actual current behavior and should
not be mistaken for a reliable pass. RGB is review-only.

## Policy contract

The 70-value actor input is IMU angular velocity and projected gravity, 12 joint
positions, 12 joint velocities, previous 12-value action, four foot load/contact
channels, and 24 values compressed from the VL53L5CX 8-by-8 depth grid. The
actuator effort limit is 0.8825985 N*m. The environment retains the exact stair
geometry used for transfer: 180 mm rise and 250 mm tread depth.

The checkpoint proves intermittent supported reach. It does not prove tread
placement, body transfer, a complete stair step, or hardware transfer.
