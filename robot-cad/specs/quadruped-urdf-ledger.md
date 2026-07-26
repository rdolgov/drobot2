# Quadruped URDF design ledger

Status: provisional physics model for the current four-leg ST3215 CAD
assembly. This ledger is part of the model: its assumptions must remain
visible when interpreting NVIDIA Isaac Sim results.

## Authority, units, and frames

- Robot name: `st3215_quadruped`.
- Editable source of truth:
  `robot_cad/urdf/quadruped_robot.py::gen_urdf()`.
- Generated artifact: `exports/urdf/quadruped_robot.urdf`; do not hand-edit it.
- Primary consumer: NVIDIA Isaac Sim 6.0.1. Secondary consumer: CAD Viewer.
- URDF units: metres, kilograms, seconds, radians, newtons, and newton-metres.
- CAD/STL units: millimetres; all STL meshes use scale `0.001 0.001 0.001`.
- Body convention: right-handed `+X` forward, `+Y` left, `+Z` up.
- `base_link` origin: centre of the body footprint at its exterior bottom.
- URDF RPY convention: `R = Rz(yaw) * Ry(pitch) * Rx(roll)`.
- Moving-link frames are located at their incoming servo-output axis.
- Geometry authority, in descending order:
  1. `robot_cad/assembly/quadruped_robot.py`;
  2. `robot_cad/assembly/robot_leg.py`;
  3. named CAD component generators and project YAML specifications;
  4. Feetech/Waveshare ST3215 data;
  5. assumptions explicitly identified below.

The immutable servo model is
`vendor/servos/waveshare_feetech_st3215_servo.step`, SHA-256
`29954eb73bd22b3f9536de2c1d8f96843b5c5b32288a8f4cb09709b8b892e39b`.
It is the 12 V Feetech ST-3215-C018 sold by Waveshare as the 30 kgf-cm
ST3215, not the HS or 7.4 V variant. The imported STEP contains one invalid
solid, so physics uses a box envelope and the verified 55 g mass.

Vendor references:
[Feetech product](https://www.feetechrc.com/525603.html),
[Feetech datasheet](https://www.feetechrc.com/Data/feetechrc/upload/file/20260622/6391772519923113075854851.pdf),
[Waveshare product](https://www.waveshare.com/product/modules/motors-servos/motors-servos/st3215-servo.htm),
and [Waveshare wiki](https://www.waveshare.com/wiki/ST3215_Servo).

## Topology

The URDF contains 13 physical links and 12 revolute joints:

```text
base_link
|- front_left_hip_link
|  `- front_left_proximal_link
|     `- front_left_distal_link
|- rear_left_hip_link
|  `- rear_left_proximal_link
|     `- rear_left_distal_link
|- front_right_hip_link
|  `- front_right_proximal_link
|     `- front_right_distal_link
`- rear_right_hip_link
   `- rear_right_proximal_link
      `- rear_right_distal_link
```

The body tub, lid, tray, four fixed body-side mounts, battery, and electronics
are collapsed into `base_link`. Each servo case belongs to the moving child
that contains its motor bay: hip servo in the hip link, hip-flexion servo in
the proximal link, and knee servo in the distal link.

For each full leg prefix (`front_left`, `rear_left`, `front_right`,
`rear_right`), the joint names are:

- `{prefix}_hip_abduction`: `base_link` to `{prefix}_hip_link`;
- `{prefix}_hip_flexion`: hip link to `{prefix}_proximal_link`;
- `{prefix}_knee`: proximal link to `{prefix}_distal_link`.

## Exact zero-pose transforms

Root transforms from `base_link`:

| Leg | xyz (m) | rpy (rad) | local +Z in body |
|---|---:|---:|---:|
| front left | `(0.060, 0.170084989, 0.050)` | `(1.570796327, 0, 1.570796327)` | `+X` |
| rear left | `(-0.060, 0.170084989, 0.050)` | `(1.570796327, 0, 1.570796327)` | `+X` |
| front right | `(0.060, -0.170084989, 0.050)` | `(1.570796327, 0, -1.570796327)` | `-X` |
| rear right | `(-0.060, -0.170084989, 0.050)` | `(1.570796327, 0, -1.570796327)` | `-X` |

Transforms inside every leg:

| Transform | xyz (m) | rpy (rad) |
|---|---:|---:|
| hip link to hip-flexion joint | `(0.028611700, -0.095696689, -0.001950000)` | `(-1.570796327, 0, -1.570796327)` |
| proximal link to knee | `(0.159896689, 0, 0)` | `(0, 0, 0)` |
| distal link to fork-tip proxy | `(0.159896689, 0, 0)` | `(0, 0, 0)` |

All abduction axes are joint-frame `(0, 0, 1)`. Flexion and knee axes are
`(0, 0, -1)` on the left and `(0, 0, 1)` on the right. This sign convention
makes positive pitch have the same robot-forward meaning on both sides. At
zero flexion and knee, each arm's local `+X` points downward in the body frame.

## Geometry and collision ledger

| Owner | Geometry | Origin xyz / rpy | Size or source |
|---|---|---|---|
| base | body tub mesh | `(0,0,0) / (0,0,0)` | `exports/stl/quadruped_body_base.stl` |
| base | lid mesh | `(0,0,0.096) / (0,0,0)` | `exports/stl/quadruped_body_lid.stl` |
| base | tray mesh | `(0,0,0.056) / (0,0,0)` | `exports/stl/quadruped_electronics_tray.stl` |
| base | four mount meshes | centres `(+-0.060,+-0.117,0.050)`; `rpy=(1.570796,0,+-1.570796)` | `exports/stl/st3215_hip_body_mount.stl` |
| base | collision box | `(0,0,0.050) / (0,0,0)` | `0.220 x 0.170 x 0.100 m` |
| every hip | printable mesh | `(0.0286117,-0.0426117,-0.00195) / (-1.570796,0,0)` | `exports/stl/st3215_hip.stl` |
| every arm | printable mesh | `(0.0948117,-0.012,0) / (0,0,0)` | `exports/stl/upper_arm.stl` |
| every moving link | exact ST3215 visual mesh | `(0.0255,0,0.007475) / (1.570796,0,0)` | `exports/stl/st3215_servo_visual.stl` |
| every moving link | ST3215 collision box | `(0.0125,0,-0.001825) / (1.570796,0,0)` | source-axis size `0.045223408 x 0.0378 x 0.024723408 m` |
| every distal link | fork-tip sphere | `(0.159896689,0,0) / (0,0,0)` | radius `0.0125 m` |

Visuals use the exact printable meshes. Physics uses primitive boxes for
stable dynamic contact. The magenta distal spheres are explicitly named
`simulation_only_fork_tip_contact_proxy`. The current CAD has no foot or
ankle; these spheres only permit a preliminary contact experiment and cannot
establish that the printable hardware can walk.

## Mass and inertia model

Input assumptions:

| Component | Assumption |
|---|---|
| printable CAD | uniform, fully solid PLA at `1240 kg/m^3` |
| each ST3215 | verified `0.055 kg`; uniform box-envelope inertia |
| battery | provisional `0.450 kg`, centred low in the battery bay |
| electronics and wiring | provisional `0.150 kg`, centred on the tray |
| omitted | bolts, horns, nuts, loose wiring, and future feet |

Solid-PLA component estimates are: base 0.610 kg, lid 0.178 kg, tray
0.084 kg, each body mount 0.144 kg, each hip print 0.115 kg, and each arm
print 0.160 kg. Real Bambu shell/infill mass must replace these conservative
solid estimates after slicing or weighing.

Inertias are about the listed COM, expressed in link axes, with inertial
`rpy=(0,0,0)`. The six values are `(ixx, ixy, ixz, iyy, iyz, izz)` in
`kg*m^2`.

| Link class | Mass (kg) | COM xyz (m) | Inertia |
|---|---:|---:|---|
| base | `2.049119` | `(0,0,0.046485537)` | `(0.0132456004, 0.0000110328, 0, 0.0097442040, 0, 0.0196972212)` |
| each hip | `0.169697` | `(0.023744808,-0.031777881,-0.001911095)` | `(0.0001832517,0.0000248550,0.0000002392,0.0000718098,-0.0000002529,0.0002314034)` |
| each proximal/distal arm | `0.215137` | `(0.073011680,-0.000021551,-0.000924466)` | `(0.0000747088,0.0000005653,-0.0000069897,0.0005598933,0.0000000356,0.0005075793)` |

The rounded total model mass is `4.449003 kg`.

## Joint and actuator ledger

| Joint class | Assumed position range |
|---|---:|
| hip abduction | `-25 to +25 deg` |
| hip flexion | `-60 to +60 deg` |
| knee | `-90 to +90 deg` |

Only the knee range was already specified. Hip ranges remain conservative
simulation assumptions until cable routing and a CAD collision sweep are done.

Verified ST3215 parameters used by the model:

- mass: `0.055 kg`;
- rated torque: `0.980665 N*m`;
- stall torque: `2.941995 N*m`;
- maximum no-load speed: `4.712389 rad/s`;
- torque constant: `1.0787315 N*m/A`;
- rated current: `0.9 A`, or `10.8 A` for 12 servos;
- stall current: `2.7 A`, or `32.4 A` for 12 simultaneously stalled servos.

The URDF exposes stall torque and no-load speed as hard joint limits. Isaac
must separately run a rated-torque case capped at `0.980665 N*m`. A stall-limit
run is an upper-envelope experiment, not evidence of continuous capability.
Vendor data is insufficient for an exact PD gain, torque-speed, thermal,
backlash, bus-voltage, or gearbox-compliance model; controller assumptions
must be logged.

## Isaac articulation handoff

The generated handoff artifacts are:

- `exports/isaac/quadruped_robot_fixed.usdc`: self-contained fixed-base
  commissioning articulation;
- `exports/isaac/quadruped_robot_floating.usdc`: self-contained floating-base
  dynamics articulation;
- `exports/isaac/quadruped_robot_manual_world.usda`: portable world referencing
  the floating USDC beside it, with Earth gravity, floor contact, the
  conservative standing targets, and the rated ST3215 effort cap.

Both USDC assets contain one articulation root, 13 rigid bodies, 12 revolute
joints, and 12 angular position drives. The manual articulation path is
`/World/Robot`. The generated world is intended for Isaac Sim's standard
**Physics > Articulation Inspector**; its launcher sets the initial stand only
once and does not continuously overwrite manual joint commands.

## Known limitations

1. Battery and electronics are placeholders; the actual parts are not selected.
2. Fully solid PLA is a mass upper bound, not a sliced Bambu estimate.
3. There are no physical feet, compliant pads, or measured friction.
4. Servo wires, bolts, horns, and self-collision have not been clearance-swept.
5. Rigid links omit PLA flex, anisotropy, creep, fork fatigue, fastener
   pull-out, gearbox compliance, backlash, play, and wear.
6. Self-collision is initially disabled because exact servo/fork overlaps are
   already known in the source assembly.
7. There is no electrical or thermal model. Mechanically possible motion may
   still demand unsafe current or duty cycle.

## Validation and Isaac acceptance criteria

Generation:

- the URDF generator and validator pass;
- exactly 13 links, 12 revolute joints, one root, no cycles, and no missing
  mesh references;
- every physical link has inertial, visual, and collision elements;
- total mass is within numerical tolerance of `4.449003 kg`;
- fixed-base joint sweeps confirm axes, mirrored signs, and continuous poses.

Gravity and standing:

- explicit gravity `(0,0,-9.81) m/s^2`, at least 120 physics steps/s;
- free drop contacts the ground and settles without NaNs, tunnelling,
  disconnection, or explosive energy;
- rated-torque standing runs at least 10 seconds after settling, keeps the body
  bottom above 0.20 m, absolute roll/pitch below 15 deg, and RMS joint error
  below 0.15 rad;
- repeat at stall limit only as a labelled upper-envelope comparison.

Motion:

- start with a four-beat crawl, roughly 0.75 duty factor, 20-30 mm stride,
  10-15 mm lift, and 2-3 s period;
- under the rated cap, complete four cycles without falling, ground-striking
  the body, violating limits, or producing NaNs;
- require forward progress over 0.02 m, lateral drift under 0.05 m, and
  roll/pitch under 25 deg;
- log base pose, velocity, joint targets/positions/errors, actuator effort,
  saturation time, and contacts.

A visible gait animation alone is not a pass. Even a passing proxy-foot result
remains preliminary until physical feet, measured printed masses, the selected
battery, measured friction, collision sweeps, and electrical/thermal limits
are added.
