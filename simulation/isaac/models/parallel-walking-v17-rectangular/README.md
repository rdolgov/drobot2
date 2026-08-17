# Parallel walking V17 rectangular-shoe policy

`model_800.pt` is the selected 60 Hz forward policy for the 2026-08-13 flat
rectangular shoe. It was initialized from V16 `model_250.pt`, transferred to
the new shoe physics, then continued with smoothness and sustained-progress
rewards. Checkpoint selection used three uninterrupted 30-second deterministic
episodes rather than PPO return.

## Selection result

- falls: 0/3
- mean forward distance: 0.3486 m in 30 s
- mean absolute lateral displacement: 0.0249 m
- mean final five-second speed: 0.0059 m/s
- mean stalled-window fraction: 0.9615
- action saturation: approximately 0.0009

This is a conservative, smooth, straight, fall-free simulation policy. It does
not yet achieve the 0.15 m/s command and should not be described as a mature
continuous gait. Faster candidates were rejected because they fell or drifted
diagonally.

The actor contract is unchanged: 48 observations and 12 bounded actions. SHA-256:
`EED2DBE61A28E7632A4CDD5F2B097C729CE9FD334FFEB2BD9EECB0F08ACB6B6A`.

See `simulation/docs/rl-rectangular-shoe-walking-v17.md` for the shoe model,
reward history, exact training runs, rejected-policy results, and limitations.
