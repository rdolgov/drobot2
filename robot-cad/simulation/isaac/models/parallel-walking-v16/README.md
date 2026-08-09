# Parallel walking V16 selected checkpoint

`model_250.pt` is the selected sustained RSL-RL checkpoint for
`Drobot-Commanded-Walk-Forward-Direct`. It uses a bounded Beta policy head and was selected
from a 1,000-iteration pure-PPO continuation trained with 128 parallel Isaac Lab environments.

- experiment: `drobot_commanded_walk_forward_v16_sustained_beta_direct`
- source run: `2026-08-06_09-02-02_manual-headless`
- source iteration: `250`
- training seed: `3302`
- control rate: `60 Hz`
- command: `0.15 m/s` forward
- actor: 48 -> 256 -> 256 -> 24 Beta parameters
- policy parameters: 84,504
- deterministic evaluation: three uninterrupted 30-second trials, `0/3` falls
- mean forward displacement: `4.442677 m`
- mean final five-second speed: `0.141725 m/s`
- mean absolute lateral displacement: `1.604028 m`
- five-second stall-window fraction: `0.0`
- mean action saturation fraction: `0.004336`
- checkpoint SHA-256: `6361c7aab7d92fa191c7a79a5e48e427fbd38002343459737dc46a176b179a9b`

The release video is `reviews/parallel-walking-v16-model250-sustained-30s.mp4`, verified as
1,800 frames, 60 fps, 1,280 x 720, and exactly 30 seconds. Its SHA-256 is
`fe0c1f657fee668952c8d267071c2b18d7786383cf2582c0e71ce3217f4b4d71`.

This checkpoint fixes V15's post-ten-second stop, but it drifts right by about 1.6 m over a
30-second, 4.4 m forward trial. It is a simulation training/evaluation artifact, not a
hardware-deployment release.
