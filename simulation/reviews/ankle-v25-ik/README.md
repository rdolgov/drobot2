# IK-guided diagonal RL review

Two residual PPO candidates were trained from the diagonal IK reference. Training produced checkpoints, but no evaluated checkpoint achieved reliable long paired strides. The final candidate keeps the trunk flatter while rear-foot flight, forward placement and paired landings remain inconsistent. No candidate has been deployed to the Pi.

Each four-trial evaluation uses 32 seconds, both starting diagonals, and nominal/85% motor strength. The final portion asks the robot to stop. No post-reset episodes count.

| Candidate | Forward range | RMS tilt range | Qualified trials | Completed pair events across trials |
|---|---:|---:|---:|---:|
| [IK reference](20260920-reference-full/report.json) | 11.8–24.9 cm | 1.95–2.99° | 0/4 | 2 |
| [Initial RL, checkpoint 100](20260920-eval-100/report.json) | 22.2–51.1 cm | 1.87–2.35° | 0/4 | 0 |
| [Initial RL, checkpoint 300](20260920-eval-300/report.json) | 0.2–39.3 cm | 1.61–2.28° | 0/4 | 1 |
| [Pair-focused RL, checkpoint 100](20260920-pair-eval-100/report.json) | 7.4–37.2 cm | 1.86–3.49° | 0/4 | 0 |
| [Pair-focused RL, checkpoint 200](20260920-pair-eval-200/report.json) | 0.7–17.6 cm | 0.93–1.56° | 0/4 | 0 |
| [Pair-focused RL, checkpoint 299](20260920-pair-eval-299/report.json) | 5.2–26.0 cm | 1.05–1.56° | 0/4 | 0 |

A paired event requires real flight and long forward placement of both selected feet followed by a quiet plant. Phase advancement and forward sliding are not qualifying steps. A few events do not establish reliable walking.

## Per-trial evidence

| Candidate / start | Strength | Forward | Lateral | Motion body height | Full steps FL/RL/FR/RR | Peak swing clearance FL/RL/FR/RR |
|---|---:|---:|---:|---:|---|---|
| IK reference / FL+RR | 100% | 13.9 cm | 1.2 cm | 36.5 cm | 0/1/1/0 | 51.4/10.7/35.9/0.0 mm |
| IK reference / FR+RL | 100% | 24.9 cm | -0.1 cm | 36.6 cm | 0/1/1/0 | 47.4/12.9/45.3/0.8 mm |
| IK reference / FL+RR | 85% | 11.8 cm | 0.2 cm | 36.5 cm | 0/0/0/0 | 52.0/1.0/27.0/0.0 mm |
| IK reference / FR+RL | 85% | 23.9 cm | 0.5 cm | 36.6 cm | 0/0/0/0 | 45.4/1.0/38.2/0.0 mm |
| Initial RL, checkpoint 100 / FL+RR | 100% | 37.5 cm | -3.3 cm | 36.7 cm | 0/0/0/0 | 43.6/4.5/25.5/0.1 mm |
| Initial RL, checkpoint 100 / FR+RL | 100% | 41.4 cm | -5.3 cm | 36.6 cm | 0/0/0/0 | 41.0/7.8/56.2/2.5 mm |
| Initial RL, checkpoint 100 / FL+RR | 85% | 22.2 cm | -2.5 cm | 36.5 cm | 0/0/0/0 | 44.4/0.0/19.5/0.8 mm |
| Initial RL, checkpoint 100 / FR+RL | 85% | 51.1 cm | -6.4 cm | 36.3 cm | 0/0/0/0 | 25.4/0.3/46.3/0.0 mm |
| Initial RL, checkpoint 300 / FL+RR | 100% | 1.0 cm | -1.5 cm | 36.6 cm | 0/0/0/0 | 40.3/18.1/13.1/13.1 mm |
| Initial RL, checkpoint 300 / FR+RL | 100% | 34.6 cm | -5.2 cm | 36.8 cm | 0/1/1/0 | 10.2/11.3/47.5/1.8 mm |
| Initial RL, checkpoint 300 / FL+RR | 85% | 0.2 cm | -0.1 cm | 36.6 cm | 0/0/0/0 | 38.2/2.0/9.2/12.0 mm |
| Initial RL, checkpoint 300 / FR+RL | 85% | 39.3 cm | 0.6 cm | 36.3 cm | 0/0/0/0 | 26.5/5.0/30.0/0.0 mm |
| Pair-focused RL, checkpoint 100 / FL+RR | 100% | 37.2 cm | -0.9 cm | 36.7 cm | 0/0/0/0 | 41.0/3.3/17.2/0.0 mm |
| Pair-focused RL, checkpoint 100 / FR+RL | 100% | 7.4 cm | 2.6 cm | 35.8 cm | 0/0/0/0 | 4.3/0.0/28.7/0.1 mm |
| Pair-focused RL, checkpoint 100 / FL+RR | 85% | 34.3 cm | 0.6 cm | 36.7 cm | 0/0/0/0 | 41.0/0.0/16.0/0.0 mm |
| Pair-focused RL, checkpoint 100 / FR+RL | 85% | 25.3 cm | 1.7 cm | 36.2 cm | 0/0/0/0 | 6.9/0.1/32.9/0.6 mm |
| Pair-focused RL, checkpoint 200 / FL+RR | 100% | 11.0 cm | 0.3 cm | 36.6 cm | 0/0/0/0 | 32.0/0.8/16.7/31.8 mm |
| Pair-focused RL, checkpoint 200 / FR+RL | 100% | 17.6 cm | -0.3 cm | 36.2 cm | 0/0/0/0 | 4.0/1.5/30.1/1.2 mm |
| Pair-focused RL, checkpoint 200 / FL+RR | 85% | 0.7 cm | 2.8 cm | 36.6 cm | 0/0/0/0 | 32.0/0.0/20.3/18.2 mm |
| Pair-focused RL, checkpoint 200 / FR+RL | 85% | 0.8 cm | 1.7 cm | 36.6 cm | 0/0/0/0 | 31.7/0.4/20.0/18.5 mm |
| Pair-focused RL, checkpoint 299 / FL+RR | 100% | 9.0 cm | -6.7 cm | 36.8 cm | 0/0/0/0 | 26.7/10.9/19.4/38.7 mm |
| Pair-focused RL, checkpoint 299 / FR+RL | 100% | 21.5 cm | -4.4 cm | 36.5 cm | 0/0/0/0 | 5.4/13.7/36.5/19.5 mm |
| Pair-focused RL, checkpoint 299 / FL+RR | 85% | 5.2 cm | -2.8 cm | 36.5 cm | 0/0/0/0 | 21.4/0.0/15.3/41.7 mm |
| Pair-focused RL, checkpoint 299 / FR+RL | 85% | 26.0 cm | -0.4 cm | 36.3 cm | 0/0/0/0 | 0.3/0.0/34.4/1.0 mm |

## Training and model assumptions

- Initial run: fresh Gaussian residual PPO; saved checkpoint 300 retained; stopped after iteration 313 rather than completing its planned 500 iterations.
- Second run: separate fresh actor, stronger real-pair clearance and level-body rewards, forward-reward gating, and larger bounded corrections. Its status and completion files are the authority for completion.
- 16-servo model with estimated 3.395 kg mass. Mass distribution and loaded actuator dynamics have not been measured on the updated physical robot.
- Original 0.883 Nm working torque cap, torque-speed envelope, calibration and joint limits retained. Neither run uses a stair policy.
- Videos use actual simulation frames, with phase/contact/clearance/displacement annotations. Frames after an episode reset are excluded.
- Source and calibration snapshots, checkpoint hashes, exact arguments, optimizer logs and evaluation traces are retained in each run directory.

See [the controller/reward description](../../docs/rl-ankle-v25-ik.md) and comparison.json for exact values.

## Videos and checkpoints

[36-second comparison](diagonal-rl-comparison.mp4) shows the same 2.5–14.5 second interval of the IK reference, initial RL checkpoint 100, and the final pair-focused checkpoint. Each segment plays at real speed. These clips illustrate the tradeoff between forward motion and body stability; neither is presented as qualified walking.

Full single-environment videos and their own measured reports:

- [IK reference](20260920-reference-video/walk.mp4)
- [Initial RL checkpoint 100](20260920-progress-video/walk.mp4)
- [Final pair-focused checkpoint 299](20260920-pair-final-video/walk.mp4)

Recordings use one nominal-strength robot with FL+RR first. The four-environment evaluations above include both first diagonals and weaker motors; results can differ from the recording because of scene size and simulation numerics.

Saved trained weights:

- [Initial RL checkpoint 100](20260920-train-r1/checkpoints/model_100.pt), used in the forward-progress clip.
- [Initial RL checkpoint 300](20260920-train-r1/checkpoints/model_300.pt), retained from the capped first run.
- [Final pair-focused checkpoint 299](20260920-train-r2/checkpoints/model_299.pt), from the completed second run.

These are simulator checkpoints with their matching controller packages and frozen provenance. They are not directly compatible with the currently deployed Pi policy runtime.
