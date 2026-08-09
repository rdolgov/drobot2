# Pure parallel stair PPO

## Scope and result status

This task is a separate, pure-reward PPO experiment for the full-size stair:
`180 mm` rise and `250 mm` tread. It intentionally removes the scripted gait
phase, prescribed leg order, inverse-kinematics reference, and action replay
used by earlier stair experiments. PPO controls all 12 joints directly in 128
GPU-parallel Isaac Lab environments.

The first 80-iteration run is a pipeline and exploration result, not a
converged stair policy. It processed 245,760 transitions in 109.09 seconds and
did not produce a repeatable climb. One training sample reached 0.4588 m of
forward progress, just beyond the first riser, while the largest observed base
height gain was 0.1133 m. Those maxima occurred at different iterations and do
not establish a successful step. The deterministic video likewise shows the
short-run policy leaning into the first riser without climbing it.

Follow-up training first extended the original reward chain by 1,228,800
transitions; it converged mainly on surviving in front of the riser. The reward
was then corrected to measure the physical fork-tip contact point rather than
the distal-link origin. A 308-iteration run processed 946,176 transitions, and
a 400-iteration lift-hold continuation processed another 1,228,800. Across all
pure-parallel runs in this experiment, PPO processed 3,649,536 transitions.

The lift-hold continuation produced two logged events with two simultaneous
force/height-verified tread contacts (iterations 399 and 606). The full-climb
success rate remained 0%, and a fresh deterministic playback of iteration 600
approached and lifted at the first riser but did not climb. These are useful
exploration and reward-shaping results, not a stair-placement pass.

The second round added more hip authority, a symmetric any-foot-to-any-tread
placement potential, target clamping at the robot's real soft joint limits,
and separate pure-PPO reset tasks for a nearly fully folded 300 mm stance and
a true 90-degree sideways stance. These are reset conditions only: there is
still no scripted action, gait phase, leg order, IK reference, or trajectory.

The folded, forward, and sideways comparisons processed 122,880, 122,880, and
61,440 valid transitions respectively. The folded policy mainly learned to
survive in the crouch. The sideways policy learned some lateral displacement
but no tread contact. The forward hip-authority policy found two simultaneous
tread contacts and was continued. Including that continuation and a lower
entropy consolidation, this round processed 1,499,136 valid transitions.
Training maxima were 0.4344 m progress, 0.1127 m base-height gain, 0.6520 m
fork-tip clearance, and two simultaneous tread contacts. Full-climb success
remained 0%. The selected six-second deterministic iteration-260 playback did
not climb the first step.

A third round simplified the objective before returning to tread placement.
The direct first-tread curriculum first processed 168,960 transitions without
a tread contact, while frequently producing 190 mm or greater unsupported foot
clearance. Training then used a symmetric `100 -> 140 -> 190 mm` supported-lift
curriculum: any foot may lift, but at least three feet must remain in
force-verified support and the body must remain upright. The three stages
processed 307,200, 307,200, and 460,800 transitions respectively.

The final 190 mm stage produced intermittent strict successes. Its best logged
reset batch was 50% successful at iteration 149, and maximum observed foot
clearance was 0.3126 m. The success did not converge: later batches returned to
0%, and the selected deterministic iteration-220 review shows autonomous lift
attempts and resets rather than a clean sustained success. This establishes
that the simulated mechanism and policy can sometimes satisfy the 190 mm
supported-lift gate; it does not yet establish robustness or stair climbing.

The selected 19 cm lift checkpoint was then transferred back to the first-tread
task for another 368,640 transitions. The first-step reward retained supported
lifting while adding symmetric tread placement, support retention, and base
elevation. At iterations 301 and 302, PPO produced force-verified first-tread
contacts and short supported holds; the best logged reset batch averaged 0.333
tread contacts and 0.0778 s of hold. First-step success remained 0% because the
best base-height gain was only 0.0108 m versus the required 0.06 m. Iteration
300 is packaged as a contact checkpoint, not a completed-step policy.

A fourth round introduced a narrow, surface-centered landing potential while
keeping the broad discovery potential and symmetric any-foot objective. A
supported-landing curriculum produced 20% and 14.3% success batches, and a
low-entropy replay reproduced a 20% batch. Transferring that checkpoint to a
20 mm body-rise stage processed 245,760 more transitions. The best reset batch
averaged 0.50 force-verified tread contacts, maximum supported tread hold was
0.1333 s, and maximum body gain was 0.0116 m. These maxima were not combined in
one episode, so the 20 mm stage and full stair climb remain at 0% success. The
six-second deterministic review is an honest attempt/reset video, not a pass.

A fifth round inserted a 10 mm body-rise bridge and processed 1,846,272 more
transitions across 128- and 512-environment runs. It tested contact-retention
reward, a narrow-placement-by-base-gain transfer term, lower forward-progress
weight, wider reset distance, and higher Gaussian action exploration. An exact
landing replay again produced a 20% supported-landing batch at iteration 349.
The 10 mm transfer later produced a 0.20 contact batch with 0.060 s hold and,
in a different reset batch, 0.0161 m body gain. It never combined the hold and
height gate, so 10 mm success remained 0% and the 20/40/60 mm stages were not
attempted. The reward now includes an explicit completion bonus on the step
that satisfies support, hold, upright, and stage-height conditions; this fixes
successful termination previously discarding future survival reward, but a
completion event must still be sampled before PPO can reinforce it.

A follow-up 307,200-transition landing run tested rewarding downward fork-tip
motion inside an 80 mm band above the first tread. It produced no tread contact,
so that touchdown-velocity term was rejected rather than retained in the task.
The DirectRLEnv audit also confirmed that dones are computed before rewards;
the completion bonus now uses the environment's authoritative success flag on
the terminal transition instead of predicting the next hold count.

A sixth round tested whether longer 64-step on-policy batches would retain a
complete rare landing sequence. Starting from the earlier landing checkpoint,
128 environments processed 491,520 transitions. Fork-tip clearance reached
0.3347 m and base-height gain reached 0.0169 m, but the run produced no tread
contact, so rollout length by itself did not solve placement. Compute then
returned to the simpler 190 mm supported-lift task for 138,240 transitions.
That continuation produced repeated strict success batches, peaking at 50%,
0.2400 m clearance, and 0.1333 s supported hold. Success is still intermittent,
and the new deterministic six-second review still collapses and resets; it is
not a foot-lift pass or a stair-climb pass.

A further 307,200-transition continuation reproduced the stochastic lift more
strongly: two reset batches reached 100% strict success, maximum clearance was
0.3891 m, and the supported hold reached the complete 0.2667 s gate. The final
batch was still 0%, and deterministic iteration-359 playback on unseen seed
1103 folded and reset. The useful conclusion is unchanged: reachable motion
and successful sampled actions exist, but PPO has not yet moved that behavior
into a reliable mean policy.

Those historical reset-batch percentages were subsequently found to describe
only the environments that happened to reset on the logged step, sometimes a
single robot. They must not be interpreted as whole-population success rates.
The logger now accumulates every completed episode and prints authoritative
totals when a bounded run closes.

Using the corrected metric, a 184,320-transition low-entropy consolidation of
the 190 mm supported-lift policy completed 5,580 episodes with 375 successes
(6.7204%) during stochastic PPO. Its deterministic mean policy was then tested
on two unseen 128-environment seeds with sensor noise: seed 1105 achieved
151/1,384 (10.9104%), and seed 1106 achieved 142/1,388 (10.2305%). The pooled
result is 293/2,772 (10.5700%). This is a reproducible supported foot-lift
result, but it is not yet a stair climb.

Transferring that mean policy to exact first-tread landing produced 24/5,495
(0.4368%) stochastic training successes. Deterministic evaluation achieved
2/1,357 (0.1474%) exact supported landings. A following 10 mm body-rise stage
produced 0/5,443 successes. Thirty-iteration adaptations from the same landing
checkpoint also produced 0/506 successes from the nearly fully folded 300 mm
start and 0/2,888 from a true 90-degree sideways start. Sideways exploration
did generate up to 286.5 mm foot clearance, but did not convert it into valid
support on the 250 mm tread. The forward stance remains the best measured
starting point; the immediate bottleneck is support-preserving tread contact,
not raw leg reach.

A final low-entropy exact-landing consolidation processed another 184,320
transitions. It achieved 27/5,481 stochastic training successes (0.4926%), but
its unseen deterministic mean produced 2/1,336 (0.1497%), effectively unchanged
from model 477's 0.1474%. Because the landing prerequisite did not materially
improve, the 10 mm body-rise stage was not retried. More continuation under the
same reward is unlikely to be the highest-value next experiment.

The next curriculum corrected an important weakness in that landing gate: the
older success condition allowed one tread foot plus only two of the remaining
three feet in ground support. A new phase-free gate requires a tread landing to
persist for three policy steps while all four feet are force-supported. It also
rewards tread-centered approach and downward motion only when the other three
feet remain loaded. These signals use simulator ground truth for reward and
metrics only; they add nothing to the 70-value actor observation.

A direct jump to a centered four-support landing was too sparse, producing
0/9,042 successes in 307,200 transitions. An intermediate broad-contact stage,
initialized from landing model 477, processed 245,760 transitions in 128
parallel environments and achieved 23/7,283 four-support landing episodes
(0.3158%). Its deterministic mean policy then achieved 3/1,245 (0.2410%) on
unseen noisy-sensor seed 1120. This is a small but genuine mean-policy result:
the held-out rate is close to the stochastic training rate. A subsequent
184,320-transition transfer to the stricter centered-contact gate achieved
0/5,433, so that centered checkpoint was rejected. Model 556 from the broad
four-support stage remained the starting point for a gradual width curriculum.

The first width stage removed the outer 20 mm from each end of the 250 mm tread,
requiring contact inside a centered `+/-105 mm` band while retaining all four
supports. It produced 7/7,322 stochastic successes (0.0956%) over 245,760
transitions. Despite that lower exploratory rate, model 635 preserved the mean
policy result on unseen noisy-sensor seed 1123: 3/1,248 (0.2404%), effectively
identical to model 556's 3/1,245 broad-band result. A further `+/-90 mm` stage
collapsed to 1/6,983 stochastic (0.0143%) and 0/1,153 deterministic, so model
714 was rejected. Finally, a contact-gated 10 mm body-rise transfer from model
635 processed 184,320 transitions and achieved 0/5,379. Model 635 is the new
selected curriculum checkpoint: it improves the placement constraint without
losing measured mean-policy repeatability, but it still does not raise the body,
complete a step, or climb the staircase.

## Editable sources

- `simulation/isaac/rl/parallel_stairs/pure_stairs_env.py`: vectorized
  observation, direct action, reward, failure gates, and reset behavior.
- `simulation/isaac/rl/parallel_stairs/pure_stairs_env_cfg.py`: robot,
  actuator, sensor, physics, and scene configuration.
- `simulation/isaac/rl/parallel_stairs/exact_stairs_terrain.py`: exact stair
  mesh generator.
- `simulation/isaac/rl/parallel_stairs/agents/rsl_rl_ppo_cfg.py`: PPO and
  256-by-256 actor/critic configuration.
- `simulation/exports/isaac/quadruped_robot_floating.usdc`: robot asset loaded by the
  task.

Isaac Lab was installed separately at
`C:\Users\roman\Documents\dev\IsaacLab`, `develop` commit
`90ee100616d9b77eb8e28f171252dc58e39181d7` (version 3.0.0), and linked to the
local Isaac Sim installation.

## Policy contract

The actor receives 70 values at 30 Hz:

| Input | Values | Real source |
| --- | ---: | --- |
| Body angular velocity | 3 | IMU gyroscope |
| Projected gravity | 3 | IMU orientation/gravity estimate |
| Joint position error | 12 | Servo feedback |
| Joint velocity | 12 | Servo feedback/estimate |
| Previous action | 12 | Controller state |
| Foot load/contact | 4 | Four foot load/contact channels |
| Compressed depth | 24 | VL53L5CX 8-by-8 depth grid |

The depth model runs at 15 Hz with one sensor-frame of latency, near-field
`+/-15 mm` error, 5% far-field proportional error, and 5% dropout. The 8-by-8
grid is compressed into three lateral lanes for each of eight rows.

RGB is not a policy input. It is enabled only for the review recording. Stair
coordinates, simulator body pose, terrain height, and other privileged ground
truth are excluded from the observation. Ground truth is used only to compute
reward, failure termination, and evaluation metrics.

The action is a normalized 12-vector mapped to joint-position targets around
the nominal stance. The actuator configuration retains the real-test
`0.8825985 N*m` effort cap on every joint. The hip-authority variants use
0.30 rad abduction, 0.90 rad hip-flexion, and 1.20 rad knee action scales, then
clamp every target to the URDF-derived soft limits.

## Reward and reset

The reward combines forward base displacement, upward base displacement,
incremental maximum-foot clearance, a persistent hold reward that saturates at
`190 mm` clearance, a symmetric distance potential from every physical fork
tip to every tread, contacts supported on higher treads, an alive term,
uprightness, action-rate, effort, and body-rate penalties. There is
no gait clock, commanded foot, gait phase, reference trajectory, or scripted
action. Episodes terminate for insufficient base height, excessive tilt,
excessive lateral displacement, walking backward out of the approach, or the
12-second time limit.

The foot-lift precursor adds no phase or leg identity. Its continuous reward
combines symmetric maximum clearance with retained support, and its staged
success gates require 100 mm for six steps, 140 mm for seven steps, and finally
190 mm for eight steps. Base contact is penalized; it becomes a fall failure
only when accompanied by meaningful height loss or tilt, allowing low postures
without treating every incidental sensor impulse as a terminal fall.

The terrain uses four solid steps, a 0.45 m approach measured from the robot
spawn origin, and a 0.75 m top platform. Static/dynamic friction are 1.10/0.90.

## Reproduction

From the repository root in PowerShell:

```powershell
& C:\Users\roman\Documents\dev\IsaacLab\isaaclab.bat -p simulation/isaac/rl/parallel_stairs/zero_agent_pure_parallel_stairs.py --task Drobot-Pure-Stairs-Direct --num_envs 128 --device cuda

& C:\Users\roman\Documents\dev\IsaacLab\isaaclab.bat -p simulation/isaac/rl/parallel_stairs/train_pure_parallel_stairs.py --rl_library rsl_rl --task Drobot-Pure-Stairs-Direct --num_envs 128 --seed 1055 --device cuda --max_iterations 80 --run_name pure128-seed1055

& C:\Users\roman\Documents\dev\IsaacLab\isaaclab.bat -p simulation/isaac/rl/parallel_stairs/train_pure_parallel_stairs.py --rl_library rsl_rl --task Drobot-Pure-Stairs-Direct --num_envs 128 --seed 1055 --device cuda --resume --load_run 2026-08-02_21-25-21_pure128-seed1055 --checkpoint model_79.pt --max_iterations 400 --run_name pure128-resume400-seed1055

& C:\Users\roman\Documents\dev\IsaacLab\isaaclab.bat -p simulation/isaac/rl/parallel_stairs/train_pure_parallel_stairs.py --rl_library rsl_rl --task Drobot-Pure-Stairs-Low-Hip-Direct --num_envs 128 --seed 1062 --device cuda --max_iterations 40 --run_name compare-low-fold-hip-seed1062

& C:\Users\roman\Documents\dev\IsaacLab\isaaclab.bat -p simulation/isaac/rl/parallel_stairs/train_pure_parallel_stairs.py --rl_library rsl_rl --task Drobot-Pure-Stairs-Sideways-Hip-Direct --num_envs 128 --seed 1064 --device cuda --max_iterations 20 --run_name compare-sideways-hip-fixed-seed1064

& C:\Users\roman\Documents\dev\IsaacLab\isaaclab.bat -p simulation/isaac/rl/parallel_stairs/train_pure_parallel_stairs.py --rl_library rsl_rl --task Drobot-Pure-Stairs-Hip-Direct --num_envs 128 --seed 1066 --device cuda --resume --load_run 2026-08-02_22-42-05_forward-hip-long-seed1065 --checkpoint model_100.pt --max_iterations 200 --run_name forward-hip-consolidate-seed1066
```

To record a deterministic 12-second review run, supply the selected checkpoint
as an absolute path:

```powershell
& C:\Users\roman\Documents\dev\IsaacLab\isaaclab.bat -p simulation/isaac/rl/parallel_stairs/play_pure_parallel_stairs.py --rl_library rsl_rl --task Drobot-Pure-Stairs-Direct --num_envs 1 --seed 1056 --device cuda --checkpoint C:\absolute\path\to\model.pt --video --video_length 360
```

## Validation and limitations

- A 128-environment robot smoke ran for 300 steps at an aggregate 761.2
  environment-steps/second including simulator startup.
- A two-iteration PPO smoke survived beyond the earlier one-frame termination
  defect and reached about 1,985 environment-steps/second.
- The first bounded training run averaged roughly 2,200-2,500
  environment-steps/second after startup.
- The follow-up chain ran at roughly 2,001-2,635 environment-steps/second and
  reached two valid simultaneous tread contacts during stochastic exploration.
- Deterministic playback loaded the saved 70-input, 256-by-256 actor and wrote
  an 8-second, 240-frame MP4. It did not climb the first riser.
- The second-round deterministic playback wrote a six-second, 180-frame MP4
  from iteration 260. RGB was used only to record the review; the actor still
  consumed IMU, depth, joint, previous-action, and foot-load values only.
- The supported-lift stages processed 1,075,200 transitions. The final 190 mm
  stage reached intermittent success. Historical reset-batch percentages are
  retained above only as experiment history and are not population estimates.
- The consolidated model-418 mean policy achieved 293/2,772 (10.5700%) strict
  supported lifts across two unseen deterministic, noisy-sensor evaluations.
- The transferred model-477 mean achieved 2/1,357 (0.1474%) exact supported
  first-tread landings; 10 mm body rise, low-fold landing, and sideways landing
  remained at 0% in their bounded runs.
- A low-entropy landing consolidation reached 27/5,481 stochastic (0.4926%)
  and 2/1,336 deterministic (0.1497%), which was not a material mean-policy
  improvement.
- The broad four-support model-556 mean achieved 3/1,245 (0.2410%) held-out
  landings after 23/7,283 (0.3158%) during stochastic PPO. The stricter centered
  transfer remained 0/5,433 and was rejected.
- Width105 model 635 retained 3/1,248 (0.2404%) held-out four-support landings
  after narrowing the accepted contact band by 40 mm total. Width90 fell to
  0/1,153 deterministic, and the contact-gated 10 mm body-rise bridge remained
  0/5,379.
- The single-robot review clip is explicitly a failed deterministic sample
  (seed 1115, 0/12), consistent with a policy that succeeds only about one in
  ten episodes. It is not presented as a pass video.
- No simulation result here proves hardware transfer, robust first-step
  acquisition, or a full climb. Continue training and evaluate several unseen
  seeds before considering mechanical changes or a larger neural network.
