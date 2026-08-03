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

## Fully folded sideways transfer audit

The retained model was tested without a prescribed leg from the 0.30 m,
100%-folded reset at 90 degrees to the stair. The same simulated VL53L5CX was
aimed toward the stair, hip-abduction action authority was bounded at 0.42 rad,
and all joints remained capped at 0.8825985 N*m. A 60-step settling window
prevents spawn motion from counting. Success requires one foot above its local
terrain while at least three other feet are force-loaded and the body remains
upright.

- At the strict 50 mm/four-step gate, retained model 418 scored 19/428 on
  seed 1219 and 7/293 on seed 1222: pooled 26/721 (3.605%).
- At the 100 mm/six-step gate, retained model 418 scored 2/796 on seed 1215
  and 0/702 on seed 1224: pooled 2/1,498 (0.134%).
- Five 128-environment PPO pilots added 1,122,304 transitions. An ordinary
  50 mm continuation reached 35/1,828 stochastic successes but its best
  screened checkpoint scored 13/395 versus source 19/428 on held-out seed
  1219. A low-noise continuation with a corrected +200 completion incentive
  peaked at 16/357 during training, but model 426 scored only 3/351 versus
  source 7/293 on seed 1222. Both were rejected.
- The corrected 100 mm continuation produced 3/840 stochastic events at model
  436, but neither source (0/702) nor candidate (0/610) reproduced a success
  on seed 1224. It was not promoted into the 140 or 190 mm stages.
- The ordinary seed-1225 third-person clip scored 0/5. Its SHA-256 is
  `83806ba701333cd87d566ad619c64cba24af338f27c5f2e07ef502f9bb0c95eb`.

This verifies sparse 50 mm force-backed unloading from the fully folded
sideways stance, not a repeatable lift. Ten centimeters remains unconfirmed,
and no foot has yet been placed on the 250 mm tread in this curriculum.
