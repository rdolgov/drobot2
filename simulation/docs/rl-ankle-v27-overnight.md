# Overnight diagonal walking model 2549

The September 20-21 continuation completed 2,000 PPO updates on 512 simulated
robots, preserving the approved model's actor, critic, normalizers and optimizer.
The fixed learning rate was 0.00005. The final checkpoint was selected from
600, 1000, 1400, 1800, 2200 and 2549 using physical stepping before translation.
Physics, calibration limits and complete-walking criteria were unchanged.

## Model files

- [Selected checkpoint](../reviews/ankle-v27-overnight/20260920-continue/selected/model.pt):
  SHA256 `867c3a4a839a806fcda8ba2a86df3a3103c016e9c9e108d43e020d1100908b92`.
- [Gaussian actor export](../reviews/ankle-v27-overnight/20260920-continue/selected/actor.npz)
  and [metadata](../reviews/ankle-v27-overnight/20260920-continue/selected/actor.json):
  actor SHA256 `d166ccee1f68008f2e8aae31d73717e43d7b8e22cc2a117d82f6cbf6d2a91a44`.
- [Approved fallback checkpoint 549](../reviews/ankle-v26-balanced/20260920-train-r2/checkpoints/model_549.pt):
  SHA256 `e130f5a1a463a293e6a0696fb94354828ceaa1b34bed74ce12797d18807bbe8c`.

These checkpoint and NPZ files are committed directly to Git. Videos, PNGs
and USD assets use the repository's Git LFS setup; use `git lfs pull` after
cloning to retrieve those files.

## Findings and recordings

The four 32-second selection cases (both starting diagonals, nominal and
85% motor strength) all passed: 49.1-51.6 cm forward, 1.36-4.99 cm lateral
drift, 2.37-2.56 degrees RMS tilt, and five complete steps on every leg.
There were no falls. Both swing feet were clear at all sampled advance points.

The separate full videos retain their own measurements. The first passed
with 48.65 cm forward and 2.12 cm drift. The second failed the unchanged
5 cm drift limit with 48.90 cm forward and 5.72 cm drift. Neither fell.
The weakest drift pass in the selection batch has only a 0.13 mm margin.

- [Complete results and candidate ranking](../reviews/ankle-v27-overnight/20260920-continue/README.md)
- [Approved model, then overnight model](../reviews/ankle-v27-overnight/20260920-continue/comparison.mp4)
- [Full recording: FL + RR first](../reviews/ankle-v27-overnight/20260920-continue/video-pair-0/walk.mp4)
- [Full recording: FR + RL first](../reviews/ankle-v27-overnight/20260920-continue/video-pair-1/walk.mp4)
- [Hardware handoff and compatibility requirements](../reviews/ankle-v27-overnight/20260920-continue/hardware-handoff.md)

## Reproduction and deployment status

The environment and runner are in `simulation/isaac/rl/ankle_v26_feasible/`,
inheriting the committed V26 balanced and V25/V24 diagonal environments.
The overnight review includes its plan, optimizer history, checkpoint reports,
landing evidence, video provenance and the frozen training-source snapshot.
The recorded source hashes describe original bytes; the provenance snapshot
preserves those bytes independently of normal Git line-ending conversion.
Historical run records retain their original machine paths and timestamps.
Use the repository-relative model links above on another machine.

To reproduce the authorized continuation with a fresh output directory:

```powershell
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v26_feasible/run.py train --output simulation/reviews/ankle-v27-reproduction --checkpoint simulation/reviews/ankle-v26-balanced/20260920-train-r2/checkpoints/model_549.pt --iterations 2000 --num-envs 512 --seed 2721 --resume-optimizer --learning-rate 0.00005
```

This command is documentation; it was not rerun when publishing the files.
No tests, builds, inference checks, simulations, UI checks or hardware checks
were run for the GitHub publication request.

The exported 82-input/16-output Gaussian actor requires the matching timed
diagonal controller. The deployed V23 Beta runtime is incompatible. Controller
integration and inference parity are still outstanding; these files are not
approval for a physical trial. No Pi connection or deployment was performed.
