# Continued diagonal walking training

The September 20 request, “Looks good, keep improving model,” continues the
training, simulation evaluation and video workflow. This experiment continues
the V25 pair-focused actor rather than starting another actor from zero.
No hardware motion or deployment is part of this experiment.

## Baseline and reason for the change

The instrumented V26 baseline loads V25 `model_299.pt` and reproduces the four
previous 32-second evaluations exactly. The new diagnostics do not change
physics, observations, reference motion or delivered actions. It records real
touchdowns at 60 Hz and the full diagnostic trace at 10 Hz.

The baseline stays upright with 1.05–1.56 degrees RMS body tilt, but has zero
qualified touchdowns or completed paired steps. Forward travel is short at
every selected landing in the lower/plant/push stages: 22, 24, 21 and 23
events respectively across the four trials. Several of these are small
contact bounces, so the report also separates substantial flights (100 ms and
8 mm clearance). Even those flights usually land after only 1–6 cm of actual
forward foot travel, below the 6.5 cm requirement at the evaluated 10 cm
stride. The weak diagonal depends on starting pair and motor strength.

## V26 learning changes

- Continue the actor, critic and observation normalizers from V25 pair
  checkpoint 299. Use a fresh optimizer, fixed learning rate 0.00005 and
  400 further PPO iterations on 512 environments. Iterations run 300–699.
- Keep full stride from the first training reset. Restarting the simulator
  must not restart the short-stride curriculum of the earlier experiment.
- Add bounded progress credit for actual forward foot travel while airborne
  above 8 mm, with the opposite pair supporting weight. Each foot can earn
  0.35 per placement; the weaker foot's progress in a selected pair can earn
  another 2.0. Only increases in the placement's best progress pay. Dragging
  on the floor and repeated forward/backward oscillations cannot repeatedly
  collect this credit. These are learning signals, not successful-step counts.
- Add a small (0.05) actor consistency loss under left/right reflection.
  Mirror FL/RL/FR/RR to FR/RR/FL/RL, shift the diagonal clock by pi, and reflect
  polar/axial body measurements correctly. The mirrored robot joint axes
  already use equal logical signs. Do not mirror PPO rewards/returns or
  fabricate rollout transitions; physical asymmetries still receive actual
  environment rewards.
- Record the evidence for each physical touchdown before flight counters
  reset: air time, clearance, forward/sideways travel, selected leg, support
  history and required stride.

The initial continuation was stopped after update 508, with checkpoint 500
retained, because the independent checkpoint-400 evaluation regressed paired
clearance. It is recorded as stopped early, not completed. Two isolated
front-right touchdowns qualified, but simultaneous clearance fell from
53.2/0/23.4/0% to 6.4/0/0/0% across the four trials. This is not a better gait.

The separate `ankle_v26_feasible` candidate restarts from the V25 baseline.
It adds a cost for sagittal swing corrections discarded by the existing
projection: 1.5 times the average squared residual-normalized hip/knee
difference, integrated over time. Ankle compensation and stance corrections
are excluded. This addresses large left-side knee corrections that were
saturating against the projected trajectory without producing useful lift.
The actor's soft reflection loss increases to 0.25, with learning rate
0.0001 and 250 additional iterations (300–549). Controller and physical
limits remain unchanged. The initial candidate's sources are preserved.

All previous level-body, low-sway, smooth-target, sole-contact and early
support-ankle rewards remain. Torque/speed limits, 3.39514659 kg estimated
mass, residual limits, calibration, actor/critic dimensions, reference timing
and strict complete-step criteria are unchanged. Body mass distribution and
loaded servo response remain unmeasured simulation assumptions.

## Evaluation and reproducibility

Evaluate deterministic actions over the first 32-second episode only, asking
the reference to stop at 24 seconds. Evaluate both initial diagonals at 100%
and 85% motor strength. Preserve failures and do not include reset episodes
as successful walking. Compare translation, lateral drift, body tilt, both
feet clearing together, individual touchdown evidence and full paired steps.

Artifacts live in `simulation/reviews/ankle-v26-balanced/`. Every simulator run
captures source files and SHA256 hashes, its loaded checkpoint hash and the
unchanged USD hash. The original V25 runs and video remain intact. Results
and video selection are written to this directory after evaluation.

Training command used:

```powershell
& C:/isaacsim/python.bat simulation/isaac/rl/ankle_v26_balanced/run.py train --output simulation/reviews/ankle-v26-balanced/20260920-train-r1 --checkpoint simulation/reviews/ankle-v25-ik/20260920-train-r2/checkpoints/model_299.pt --iterations 400 --num-envs 512 --seed 2620
```

The authorized checks for this task are the training run, simulation
evaluations and video recording. No unrelated test suite, hardware check or
UI verification is run.

## Completed continuation and outcome

The feasible-correction run completed all 250 additional PPO updates on
512 environments (8,192,000 transitions), saving `model_549.pt` with SHA256
`e130f5a1a463a293e6a0696fb94354828ceaa1b34bed74ce12797d18807bbe8c`.
Checkpoints 400, 500 and 549 were independently evaluated. Its final four
trials moved forward 28.73, 19.45, 26.57 and 36.41 cm, compared with the
baseline's 9.00, 21.45, 5.24 and 26.04 cm. RMS body tilt was 1.12–1.36 degrees
and there were no falls. One FL+RR pair completed a strict physical landing
in the reduced-strength trial; all other full-step counts remained zero.

This is a mixed result, not a qualified replacement. Both-feet-clear advance
samples fell to 8.5/4.3/6.4/0%, and sideways drift exceeded 5 cm in two trials.
Rear-left clearance remained inconsistent. All four complete-walking trials
failed qualification. The previous checkpoint remains available and nothing
was deployed to the Pi. The next bottleneck is reliable unloading and lift of
the weaker diagonal, rather than further reward for body translation alone.

The new 36-second video compares 18-second excerpts from the old and new
one-environment recordings, using the same seed, starting pair and playback
speed. The new full recording moved 20.16 cm forward and 12.02 cm sideways,
with 1.32-degree RMS tilt and no full paired steps. Those video-run results
are kept separate from the four-environment evaluation above.

See [the complete review](../reviews/ankle-v26-balanced/README.md), including
every evaluated checkpoint, saved landing events, source provenance and the
video selection record.
