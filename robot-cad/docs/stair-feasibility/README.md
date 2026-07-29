# Real-stair kinematic and torque feasibility

## Decision

The 2026-07-27 scripted Isaac experiment failed at every requested riser:
`100`, `140`, `180`, and `196 mm`. Do not resume stair reinforcement
learning yet.

All four ideal front-left-foot targets fit within the URDF joint limits, but
the floating robot could not clear the riser edge or establish tread contact
under the `0.980665 N m` ST3215 rated-torque cap. The `100 mm` trial achieved
only `16.0 mm` of foot lift. The taller trials lost support, tipped into the
block, and generated non-foot collisions. The immediate problem is dynamic
support, contact geometry, and sustainable actuator authority rather than
policy perception or PPO exploration.

This is a feasibility result for one scripted front-foot placement. It is not
a whole-stair-climb test and it does not prove hardware safety.

## Experiment contract

`Drobot-Real-Stair-Feasibility-v1` is intentionally separate from the RL task.
It creates one static block with a `280 mm` tread and tests riser heights of
`100`, `140`, `180`, and `196 mm`. The robot starts close to the block, shifts
its weight away from the front-left leg, and commands a smooth Cartesian
inverse-kinematics trajectory:

1. settle on four feet;
2. shift the body backward and to the right;
3. lift and move the front-left foot past the riser edge;
4. lower the foot onto the tread;
5. hold the landing pose.

No learned policy or RL model is loaded. Physics runs at `120 Hz`, the
controller updates at `60 Hz`, and every drive remains capped at the continuous
rated torque. The experiment records:

- achieved foot lift and edge clearance;
- tread-contact hold time and riser strikes;
- support-foot contact, slip, and support-triangle margin;
- body tilt and base drop;
- joint tracking error and requested PD saturation;
- projected joint constraint load and non-foot block collisions.

The full pass gate requires edge clearance, stable tread contact, adequate
three-foot support, bounded slip/tilt/drop, acceptable joint tracking, no
non-foot collision, and no sustained torque saturation.

## Source of truth

| File | Purpose |
| --- | --- |
| `simulation/isaac/experiments/stair_feasibility/real_stair_feasibility.yaml` | Geometry, timing, friction, torque cap, and pass/fail thresholds |
| `simulation/isaac/experiments/stair_feasibility/_contract.py` | Pure IK, support-margin, configuration, and trial-gate calculations |
| `simulation/isaac/experiments/stair_feasibility/run_real_stair_feasibility.py` | Isaac stage construction, scripted motion, measurement, screenshots, and report |
| `simulation/isaac/experiments/stair_feasibility/_manual_control.py` | Pure selectable-leg manual target state and safety checks |
| `simulation/isaac/experiments/stair_feasibility/manual_180mm_stair.py` | Interactive physics-enabled 180 mm challenge and session reporting |
| `simulation/isaac/experiments/stair_feasibility/manual_180mm_motor_angles.py` | Numbered direct motor-angle launcher |
| `simulation/isaac/experiments/stair_feasibility/probe_motor_physics.py` | Isolated zero-gravity motor-drive response probe |
| `tests/test_real_stair_feasibility_contract.py` | Deterministic configuration, IK, and gate tests |
| `tests/test_manual_stair_control.py` | Deterministic key mapping, leg isolation, reset, and joint-margin tests |

The measured run used configuration SHA-256
`8fbe89903843c171df5a802e7344b2c2079422810aa0eee4d61cc3fe18fba3bf`
and floating robot USDC SHA-256
`71b639bd877913bffeac47a1cfcb6f3dcabbbd1e25c6fba90b8b87e7ea96c6b8`.

## Reproduce

Run the simulator experiment from the repository root:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\experiments\stair_feasibility\run_real_stair_feasibility.py `
  --config simulation\isaac\experiments\stair_feasibility\real_stair_feasibility.yaml `
  --output-dir simulation\isaac\output\stair-feasibility-v1 `
  --headless
```

Run the pure contract checks without launching Isaac:

```powershell
& .venv\Scripts\python.exe -m pytest `
  tests\test_real_stair_feasibility_contract.py -q `
  --basetemp .pytest-tmp-real-stair
```

To isolate one height while tuning the scripted controller, add
`--heights-mm 100` or another configured height. Add `--no-screenshots` only
for a quick execution smoke test.

The authoritative result is `report.json`. Isaac's Windows launcher may not
reliably propagate the experiment's nonzero feasibility-failure exit code, so
automation must read the report's top-level `status` and
`curriculum_authorized` fields.

## Interactive 180 mm challenge

The scripted result may reflect a poor controller rather than an impossible
mechanism. The separate interactive runner lets a person command all four legs
while preserving the same floating-base dynamics:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\experiments\stair_feasibility\manual_180mm_stair.py
```

Click once inside the Isaac viewport so it receives keyboard input.

| Key | Action |
| --- | --- |
| `1` | Select front-left leg |
| `2` | Select front-right leg |
| `3` | Select rear-left leg |
| `4` | Select rear-right leg |
| hold `W` / `S` | Move selected foot target forward / backward |
| hold `E` / `D` | Move selected foot target up / down |
| hold `Q` / `A` | Increase / decrease selected hip-abduction target |
| `R` | Reset the robot and all four targets |
| `Space` | Pause / resume physics |
| `C` | Print current target, foot height/load, base height, and tilt |
| `X` or `Esc` | Save the report and exit |

The keys change joint position targets through inverse kinematics; they never
teleport a leg. Every non-selected leg remains a live torque-limited
articulation holding its last target. Gravity, self-collision, ground and stair
contacts, friction, the URDF hard limits, velocity limits, and the rated
`0.980665 N m` effort cap remain active. The controller rejects a requested
target if it would exceed the two-link reach or the configured hard-limit
margin.

The default report is:

`simulation/isaac/output/stair-feasibility-manual-180mm/session.json`

It records maximum lift and stair load for every foot, maximum body tilt,
minimum base height, tracking error, reset count, torque profile, and final
targets. A top-level `PASS` only means the interactive application ran
successfully; inspect `tread_contact_detected_by_leg` and the measured
stability fields to judge the attempt.

For a short diagnostic comparison only:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\experiments\stair_feasibility\manual_180mm_stair.py `
  --torque-profile stall
```

That raises the simulated effort cap to `2.941995 N m`. It must not be treated
as continuous hardware capability. If manual control succeeds only at stall
torque, the result supports an actuator/load-margin diagnosis rather than a
passing design.

## Direct numbered motor-angle control

Use the separate launcher when you want to bypass foot-space commands and set
one motor target angle directly:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\experiments\stair_feasibility\manual_180mm_motor_angles.py
```

Type the motor number, press `Enter`, and hold `Up` or `Down` to change its
target angle. `Backspace` edits the number, `Z` sets the selected target to
zero, `R` restores all 12 standing targets, and `C` prints the current state.

| Number | Motor |
| ---: | --- |
| 1 | front-left hip abduction |
| 2 | front-left hip flexion |
| 3 | front-left knee |
| 4 | front-right hip abduction |
| 5 | front-right hip flexion |
| 6 | front-right knee |
| 7 | rear-left hip abduction |
| 8 | rear-left hip flexion |
| 9 | rear-left knee |
| 10 | rear-right hip abduction |
| 11 | rear-right hip flexion |
| 12 | rear-right knee |

The control panel always shows `SELECTED MOTOR #N`, its full joint name, target
angle, measured angle, and the **live** gravity magnitude. The scene starts at
`9.81 m/s2`, but if gravity is disabled in Isaac the panel changes to
`GRAVITY OFF (0.00 m/s2)`.

With a floating base and zero gravity, world-space leg motion can reverse
because the body counter-rotates in response to an internal joint motion.
Interpret the display as follows:

- target angle moves back: controller or input changed the command;
- target stays but measured angle moves back: the capped drive is losing to
  contact, coupling, or another load;
- target and measured angles stay together but the leg moves in the viewport:
  the floating base moved or rotated around the joint.

The direct-angle session report is written to
`simulation/isaac/output/stair-feasibility-motor-angles-180mm/session.json`.
It records the live gravity range, selected motor number, all final targets,
tracking error, body tilt, foot lift, and stair contact.

## Isolated zero-gravity motor check

Use the focused probe to distinguish a drive problem from floor contact,
another leg holding the body, or floating-base reaction:

```powershell
& C:\isaacsim\python.bat `
  simulation\isaac\experiments\stair_feasibility\probe_motor_physics.py `
  --base-mode floating `
  --joint front_left_hip_abduction `
  --target-deg 20 `
  --duration-s 1.5
```

This run has no ground or stair and explicitly sets gravity to zero. It retains
the robot's hard joint limits, self-collision, no-load speed, position-drive
gains, and rated `0.980665 N m` effort cap. The JSON report records the
starting angle, response samples, final tracking error, applied gains, and
base pose. The default report path is
`simulation/isaac/output/motor-physics-audit/zero-gravity-fixed-probe.json`;
use `--report` to keep fixed- and floating-base trials separately.

## Measured result

| Riser | IK target in limits | Achieved / required lift | Support contact | Minimum support margin | Maximum tilt | Non-foot collision | Result |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `100 mm` | yes | `16.0 / 115.0 mm` | `68.0%` | `-92.7 mm` | `20.0 deg` | `0.0 N` | FAIL |
| `140 mm` | yes | `20.8 / 155.0 mm` | `25.6%` | `-263.8 mm` | `62.2 deg` | `64.8 N` | FAIL |
| `180 mm` | yes | `26.2 / 195.0 mm` | `21.1%` | `-355.5 mm` | `89.2 deg` | `66.6 N` | FAIL |
| `196 mm` | yes | `36.7 / 211.0 mm` | `38.8%` | `-352.2 mm` | `93.9 deg` | `1104.8 N` | FAIL |

No trial reached the edge-measurement region or held tread contact. The
required lift includes the riser height plus the configured `15 mm` clearance.
The `196 mm` trial also registered a direct riser strike. Requested PD output
was saturated for `12.5-27.9%` of samples, and multiple joints reached at least
`95%` of the rated cap in every trial.

The projected-load ratio is useful for locating overloaded phases, but values
above one are derived from simulator constraint forces. They are not measured
motor current or literal commanded servo torque. The controller's requested
PD demand and the capped drive behavior remain the appropriate evidence that
the current motion lacks actuator authority.

Outputs are local, repeatable run products:

- `simulation/isaac/output/stair-feasibility-v1/report.json`
- `simulation/isaac/output/stair-feasibility-v1/screenshots/riser-100mm.png`
- `simulation/isaac/output/stair-feasibility-v1/screenshots/riser-140mm.png`
- `simulation/isaac/output/stair-feasibility-v1/screenshots/riser-180mm.png`
- `simulation/isaac/output/stair-feasibility-v1/screenshots/riser-196mm.png`

## What to revise before RL

1. **Foot/contact model.** Replace the virtual `12.5 mm` spherical fork-tip
   contact with the intended flat, grippy physical foot. Validate its material,
   contact patch, collision geometry, and real support polygon.
2. **Weight transfer and scripted control.** First make the same motion work
   with a fixed or externally supported base, then with three-foot support.
   Tune body shift and foot timing from measured normal loads rather than a
   fixed offset.
3. **Actuator and mass budget.** Perform static torque, current, voltage-sag,
   and thermal sizing for the worst hip and knee poses. A short stall-torque
   comparison can diagnose margin, but must not count as a sustainable pass.
4. **Joint and leg geometry.** The requested single-foot poses already fit the
   present hard limits on paper, so lengthening legs or expanding limits is not
   the first change. Validate full-range self-collision, cables, and servo
   horns before changing geometry.
5. **Repeat the physical gate.** Require stable rated-torque placement at
   `100`, `140`, `180`, and `196 mm`, then test front-foot pairs, body transfer,
   rear feet, and a complete step.

Only after the `180-196 mm` mechanical/control gate passes should PPO restart
with a curriculum from `40 mm` toward the real-stair range. A depth sensor can
then provide transferable stair geometry, but perception cannot compensate for
a foot trajectory the mechanism cannot execute while supported.

## Assumptions and limitations

- The distal contact is a virtual sphere, not the final printed foot.
- Friction values (`0.90` static, `0.75` dynamic) are provisional.
- The controller is scripted Cartesian IK with position drives, not an
  optimized whole-body controller.
- Reported PD demand is a proxy, not measured current or temperature.
- The test commands only the front-left foot on one block.
- A later simulator pass still requires tethered, current-limited hardware
  validation with an emergency stop.
