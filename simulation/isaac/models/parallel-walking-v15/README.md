# Parallel walking V15 selected checkpoint

`model_125.pt` is the selected stable RSL-RL checkpoint for
`Drobot-Commanded-Walk-Forward-Direct`. It was initialized from the tracked
pure-RL SB3 walking actor, then adapted for 125 PPO iterations with 128 parallel
Isaac Lab environments and the measured 0.8825985 N*m effort limit.

- task: `drobot_commanded_walk_forward_v15_rl_transfer_direct`
- training seed: `3102`
- control rate: `60 Hz`
- command: `0.15 m/s` forward
- deterministic evaluation: `10/10` episodes without a fall
- mean forward displacement: `0.384889 m` per eight-second episode
- best observed evaluation episode: `0.604290 m`
- mean base height: `0.369075 m`
- SHA-256: `995dc00e4603da91386f80976fcec3da9df0b50cff73798a056c86deade0a887`

This is meaningful stable forward motion, but it has not yet reached the full
`1.2 m` target for 0.15 m/s tracking. It is a training/evaluation checkpoint,
not a hardware-deployment release.

A 30-second third-person preview is tracked at
`reviews/parallel-walking-v15-model125-30s.mp4`.
