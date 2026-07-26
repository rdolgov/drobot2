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

The URDF contains 15 physical links, one massless optical frame, 12 revolute
joints, two fixed camera joints, and one fixed IMU joint:

```text
base_link
|- imu_link
|- camera_link
|  `- camera_optical_frame
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

The body tub, lid, tray, four fixed body-side mounts, battery, and remaining
electronics are collapsed into `base_link`. The selected 2.5 g BNO085 board
is kept as `imu_link`, with its frame at the measured sensing-package centre.
Each servo case belongs to the moving child
that contains its motor bay: hip servo in the hip link, hip-flexion servo in
the proximal link, and knee servo in the distal link. `camera_link` owns the
unchanged LeKiwi printable mount, the Arducam reference mesh, their approximate
mass properties, and two simple collision boxes. `camera_optical_frame` is
frame-only and uses the standard `+Z` forward, `+X` right, `+Y` down optical
convention.

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
| IMU link | exact BNO085 board mesh | `(0,0,-0.002160) / (0,0,0)` | `exports/stl/adafruit_bno085_stemma_qt.stl` |
| IMU link | board collision box | `(-0.000003,-0.000805,0.000105) / (0,0,0)` | `0.0254 x 0.02286 x 0.00453 m` |
| base | four mount meshes | centres `(+-0.060,+-0.117,0.050)`; `rpy=(1.570796,0,+-1.570796)` | `exports/stl/st3215_hip_body_mount.stl` |
| base | collision box | `(0,0,0.050) / (0,0,0)` | `0.220 x 0.170 x 0.100 m` |
| camera link | LeKiwi mount mesh | `(0,0,0) / (0,0,0)` | `vendor/references/lekiwi/base_camera_mount.stl` |
| camera link | Arducam mesh | `(0,0,0.023) / (0,1.570796,0)` | `vendor/references/lekiwi/arducam_5mp_camera_model.stl` |
| camera link | mount collision box | `(0.003330537,0,0.022592295) / (0,0,0)` | `0.016661073 x 0.048 x 0.045184589 m` |
| camera link | camera collision box | `(0.01075,0,0.023) / (0,0,0)` | `0.0215 x 0.038 x 0.038 m` |
| every hip | printable mesh | `(0.0286117,-0.0426117,-0.00195) / (-1.570796,0,0)` | `exports/stl/st3215_hip.stl` |
| every arm | printable mesh | `(0.0948117,-0.012,0) / (0,0,0)` | `exports/stl/upper_arm.stl` |
| every hip | guarded printable collision box | `(0.0266617,-0.0460425,-0.00195) / (0,0,0)` | `0.0713 x 0.127308 x 0.04805 m` |
| every arm | guarded printable collision box | `(0.096254,0,-0.00195) / (0,0,0)` | `0.155285 x 0.035229 x 0.0713 m` |
| every moving link | exact ST3215 visual mesh | `(0.0255,0,0.007475) / (1.570796,0,0)` | `exports/stl/st3215_servo_visual.stl` |
| every moving link | ST3215 collision box | `(0.0125,0,-0.001825) / (1.570796,0,0)` | source-axis size `0.045223408 x 0.0378 x 0.024723408 m` |
| every distal link | fork-tip sphere | `(0.159896689,0,0) / (0,0,0)` | radius `0.0125 m` |

Visuals use the exact printable meshes. Physics uses primitive boxes for
stable dynamic contact. The magenta distal spheres are explicitly named
`simulation_only_fork_tip_contact_proxy`. The current CAD has no foot or
ankle; these spheres only permit a preliminary contact experiment and cannot
establish that the printable hardware can walk.

Printable hip and arm boxes include a `2 mm` guard on every side so PhysX
contact begins before rendered PLA surfaces visibly overlap. Isaac
self-collision is enabled. The 12 directly connected joint-neighbor pairs are
explicitly filtered with `UsdPhysics.FilteredPairsAPI` because their servo and
fork geometry intentionally overlaps at the pivot. All inter-leg pairs,
non-adjacent same-leg pairs, and body-versus-non-adjacent-leg pairs remain
collision-enabled.

## Mass and inertia model

Input assumptions:

| Component | Assumption |
|---|---|
| printable CAD | uniform, fully solid PLA at `1240 kg/m^3` |
| each ST3215 | verified `0.055 kg`; uniform box-envelope inertia |
| battery | provisional `0.450 kg`, centred low in the battery bay |
| electronics and wiring | provisional `0.150 kg`, centred on the tray |
| Adafruit BNO085 board | listed product weight `0.0025 kg`; separate from the provisional electronics payload |
| Arducam camera | provisional listed product weight `0.060 kg`, including its attached cable |
| LeKiwi camera mount | `0.014635989 kg` from upstream STL signed volume at solid-PLA density |
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
| fixed camera assembly | `0.074635989` | `(0.009261819,-0.000021765,0.022460669)` in `camera_link` | `(0.000019829381,-0.000000009911,-0.000000245577,0.000013126569,-0.000000003592,0.000013357698)` |
| fixed BNO085 board | `0.0025` | `(-0.000003,-0.000805,0.000105)` in `imu_link` | uniform-envelope estimate `(0.000000113145938,0,0,0.000000138683521,0,0.000000243279083)` |
| each hip | `0.169697` | `(0.023744808,-0.031777881,-0.001911095)` | `(0.0001832517,0.0000248550,0.0000002392,0.0000718098,-0.0000002529,0.0002314034)` |
| each proximal/distal arm | `0.215137` | `(0.073011680,-0.000021551,-0.000924466)` | `(0.0000747088,0.0000005653,-0.0000069897,0.0005598933,0.0000000356,0.0005075793)` |

The rounded total model mass is `4.526139 kg`.

## Camera frames and sensor profile

The fixed transform from `base_link` to `camera_link` is
`xyz=(0.090,0,0.100) m`, `rpy=(0,0,0)`. The optical frame is
`xyz=(0.0245,0,0.023) m`,
`rpy=(-1.570796327,0,-1.570796327)`, placing the lens centre at
`(0.1145,0,0.123) m` in `base_link` and pointing it along robot `+X`.

Isaac authors `/World/Robot/Geometry/base_link/lekiwi_camera` as a child of
the imported base rigid body. USD cameras look along local `-Z`, so the
authored quaternion is `wxyz=(0.5,0.5,-0.5,-0.5)`, equivalent to
`rotateXYZ=(90,0,-90) deg`. A quaternion `xformOp:orient` is required because
PhysX/Fabric preserves it when the parent rigid body moves.

The lightweight simulation profile is:

- resolution: `480 x 640` pixels in `(height,width)` order;
- tick rate: `30 Hz`;
- pinhole horizontal field of view: `95 deg`;
- focal length: `1.686049360 mm`;
- aperture: `3.68 x 2.76 mm`;
- clipping range: `0.05 to 100 m`;
- available validated annotators: `rgb` and `distance_to_image_plane`.

## IMU frame and training observation profile

`imu_link` is fixed to `base_link` at
`xyz=(0,0,0.065160) m`, `rpy=(0,0,0)`. Its origin is the sensing-package
centre from Adafruit's official product-4754 STEP, not the PCB centre. The
installed frame therefore uses the body convention directly: `+X` forward,
`+Y` left, and `+Z` up. The board bottom is at `z=0.063 m`; its Qwiic
connectors face robot front/rear.

Isaac authors `/World/Robot/Geometry/base_link/body_imu` as an
`IsaacImuSensor` child of the imported base rigid body. Isaac Sim 6 uses
`isaacsim.sensors.experimental.physics.IMUSensor`; it reads every physics
step. Acceleration, angular velocity, and orientation each use a three-sample
rolling average in the asset. Training code packs the following body-frame
observation:

- angular velocity in `rad/s` (3 values);
- projected unit gravity in the IMU/body frame (3 values);
- linear acceleration divided by `9.81 m/s^2` (3 values).

This nine-value sensor block is intended to be concatenated with commands,
joint positions/velocities, and previous actions. The BNO085 game rotation
vector should be used on hardware so servo-current magnetic fields do not
enter the walking state estimate. Bias, latency, quantization, dropout, and
mount vibration are domain-randomization inputs, not yet physical truth.

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
  dynamics articulation with guarded inter-leg collision;
- `exports/isaac/quadruped_robot_manual_world.usda`: portable world referencing
  the floating USDC beside it, with Earth gravity, floor contact, the
  conservative standing targets, and the rated ST3215 effort cap.

Both USDC assets contain one articulation root, 13 rigid bodies, 12 revolute
joints, 12 angular position drives, one RTX camera, and one body IMU sensor.
Fixed camera and IMU links are merged into `base_link` at import, preserving
the articulation body count. The
manual articulation path is `/World/Robot`. The generated world is intended for Isaac Sim's standard
**Physics > Articulation Inspector**; its launcher sets the initial stand only
once and does not continuously overwrite manual joint commands.

## Known limitations

1. Battery and electronics are placeholders; the actual parts are not selected.
2. Fully solid PLA is a mass upper bound, not a sliced Bambu estimate.
3. There are no physical feet, compliant pads, or measured friction.
4. Servo wires, bolts, and horns have not been clearance-swept.
5. Rigid links omit PLA flex, anisotropy, creep, fork fatigue, fastener
   pull-out, gearbox compliance, backlash, play, and wear.
6. Self-collision uses conservative guarded boxes rather than exact convex CAD
   hulls. Directly connected joint neighbors are filtered because exact
   servo/fork overlaps are already known in the source assembly.
7. There is no electrical or thermal model. Mechanically possible motion may
   still demand unsafe current or duty cycle.
8. The 95-degree camera field of view is a product-profile approximation, not
   a lens calibration. Distortion, rolling shutter, exposure, microphone,
   USB latency, and sensor noise are not modeled.
9. The simulated IMU is clean physics-engine data with a short rolling
   average. Real BNO085 bias, vibration, latency, saturation, mounting
   compliance, clock drift, calibration quality, and packet loss still need
   measurement and domain randomization.

## Validation and Isaac acceptance criteria

Generation:

- the URDF generator and validator pass;
- exactly 16 URDF links, 12 revolute joints, three fixed joints, one root, no
  cycles, and no missing mesh references;
- every physical link has inertial, visual, and collision elements;
- total mass is within numerical tolerance of `4.526139 kg`;
- fixed-base joint sweeps confirm axes, mirrored signs, and continuous poses.

Camera:

- one mounted USD camera survives fixed and floating import;
- the camera remains a child of `base_link` and points along robot `+X` with
  image-up along robot `+Z` after PhysX/Fabric starts;
- a 640 x 480 RGB frame and depth frame are both retrievable through
  `isaacsim.sensors.experimental.rtx.CameraSensor`;
- a forward calibration target is visible at the expected depth.

IMU:

- one `IsaacImuSensor` survives fixed and floating import at the documented
  body-centred path;
- runtime reads are valid, quaternion norm is one, and a stationary
  gravity-included reading is approximately `+9.81 m/s^2` along local `+Z`;
- the nine-value walking observation is finite and uses the documented field
  order and normalization.

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
