# Stair perception and staged learning plan

## Decision

Use a forward-and-downward depth sensor to create a small robot-frame terrain
profile. Keep the body IMU and joint feedback in the policy. Do not train PPO
directly from full RGB frames for the next experiment.

For the inexpensive prototype, use an **ST VL53L5CX 8 x 8 multi-zone ToF
sensor on the Pololu 3417 carrier**. The 2026-07-31 observed one-piece price
was `$19.95`; the carrier is approximately `13 x 18 x 3 mm`, weighs `0.5 g`
without headers, accepts `2.5-5.5 V`, and connects over I2C. This is now the
modeled v7 policy input. A wider stereo depth camera remains an upgrade path,
not a requirement for the first sensor/learning test.

The selected physical Arducam is a monocular RGB camera. Isaac can synthesize
depth from its camera prim, but that does not make metric depth available from
the physical camera. Monocular depth estimation remains a possible later
experiment, but it adds scale, lighting, latency, and sim-to-real uncertainty
before the locomotor has demonstrated that it can place a foot on one step.

## Existing camera blind region

The current optical axis points horizontally along robot `+X`. Its simulated
profile is 95 degrees horizontal by approximately 78.6 degrees vertical, and
the lens is 0.123 m above `base_link`. With the body near its 0.373 m target
clearance, the lower image ray meets level ground roughly 0.61 m ahead. The v2
first riser begins only 0.31-0.37 m ahead of the reset base, so it can be below
the camera's useful view just when precise foot placement is needed.

The earlier `25 deg` pitch estimate applies to the wider D405 field of view.
The VL53L5CX detection volume is only about `45 deg` horizontally and
vertically (`65 deg` diagonal), so v7 models a `40 deg` downward pitch. Its
zone-center rays span approximately `20.3-59.7 deg` downward. At the current
optical origin this covers the first risers without pointing the top zones
above the horizon. Final placement still needs a CAD adapter and a real
minimum-range/occlusion bench check.

## Required sensing

| Signal | Status | Purpose |
| --- | --- | --- |
| BNO085 orientation, angular velocity, acceleration | already selected; required | body attitude and disturbance response |
| Servo joint position and estimated velocity | already in policy; required | leg pose and low-level feedback |
| Forward/down metric depth at 15-30 Hz | add; required for the transferable stair policy | stair edge, rise, tread, and lateral alignment |
| Per-foot contact or load | optional but strongly recommended | confirm swing unload, landing, slip, and support |
| Monocular RGB | already selected; optional for stairs | teleoperation, recording, and later semantic perception |

The depth sensor should have a practical near range no worse than 0.15-0.20 m,
cover both front feet, and measure at least 0.15-1.0 m ahead at 15 Hz or more.
An indoor short-range ToF depth camera is the simplest first choice. Stereo can
also work if its near range, texture dependence, compute cost, and mounting
baseline fit the robot. A single horizontal 2D lidar or a single-point range
sensor is not sufficient because it does not describe both riser height and
tread geometry; a small multi-zone ToF array is a lower-bandwidth alternative
to a depth camera.

## Current candidate screening

The selected cheap prototype is the **ST VL53L5CX on Pololu carrier 3417**.
It provides 64 range zones, a `65 deg` diagonal detection volume, and a
`20 mm-4 m` published range. Crucially, the `60 Hz` headline applies to
`4 x 4` mode; full `8 x 8` operation is limited to `15 Hz`. V7 therefore holds
one 8 x 8 sample for eight `120 Hz` control frames and adds one sample frame
(`66.7 ms`) of latency. The ST accuracy figure near `20-200 mm` is represented
as bounded `+/-15 mm` noise; longer ranges use bounded `+/-5%` noise. The
additional `5%` per-zone dropout is a conservative simulation assumption, not
a manufacturer specification.

A **RealSense D405**, mounted on an adapter and pitched down about 25 degrees,
is the wider-field upgrade. Its published ideal range is `0.07-0.50 m`, depth
field of view is `87 x 58 deg`, depth rate is up to `90 Hz`, and streaming
depth/IR power is `1.55 W`. It covers the close front-foot workspace with much
more spatial detail, but costs, weighs, and consumes substantially more than
the multi-zone ToF prototype.

An **OAK-D Lite** offers onboard stereo/vision compute, but its published ideal
depth range begins around 0.8 m; approximately 0.2 m minimum depth requires
400p extended-disparity mode. Its 2.5-3 W base streaming load and longer-range
bias make it less attractive for this first close-stair experiment.

Manufacturer references:

- [Pololu VL53L5CX carrier 3417](https://www.pololu.com/product/3417)
- [RealSense D405 specifications](https://www.realsenseai.com/products/d405/)
- [ST VL53L5CX product and datasheet](https://www.st.com/en/imaging-and-photonics-solutions/vl53l5cx.html)
- [ST VL53L5CX Linux driver](https://www.st.com/en/embedded-software/stsw-img025.html)
- [Luxonis OAK-D Lite specifications](https://docs.luxonis.com/hardware/products/OAK-D%20Lite)

Do not purchase or finalize the adapter from these paper specifications alone.
First render the proposed sensor frustum in Isaac, then bench-test minimum
range, invalid pixels, dark stair material, motion blur, latency, USB/CSI
support, power, and Raspberry Pi throughput. Compute and power hardware are
still provisional in the mechanical specification.

## Policy interface

V7 exposes a compact representation that can be reproduced directly from the
real sensor without reconstructing privileged terrain height:

```text
8 x 8 range frame (15 Hz)
  -> apply range limits and reject invalid zones
  -> median columns into left [0,1,2], center [3,4], right [5,6,7]
  -> retain all 8 vertical rows (24 normalized depth values)
  -> one-frame delay, then hold each vector for 8 control frames at 120 Hz
```

The v7 policy receives those 24 depth values, IMU, joint state, previous
action, navigation/goal state, and the existing foot-sequence observations.
It does not receive RGB pixels or analytic stair-height samples. Navigation,
goal, and foot-progress inputs remain simulator conveniences and still need a
hardware estimator or removal before deployment. Extrinsic randomization is
not yet implemented.

## Training sequence

The failed 50k run shows that approach distance and terrain awareness alone do
not teach the prerequisite foot-placement motion. Train it in stages:

Training is currently blocked by the separate
[real-stair feasibility result](../stair-feasibility/README.md). Scripted
front-foot placement failed even at `100 mm` under the rated-torque cap.
Complete the foot-contact, weight-transfer, and actuator revisions and pass
the physical gate before starting this sequence.

1. **Front-foot lift:** start stationary at a randomized distance from one
   20-40 mm riser. Reward each front toe for unloading, clearing the edge, and
   moving toward a valid landing region.
2. **Front-foot placement:** reward stable contact on the tread and penalize
   riser strikes, scraping, crossing feet, and support-foot slip.
3. **Body transfer:** after both front feet are placed, reward body height and
   forward center-of-mass transfer while remaining upright.
4. **Rear-foot placement:** apply the same clearance/contact objectives to the
   rear legs.
5. **Whole-step completion:** require a stable hold with all feet and the body
   on the higher support surface.
6. **Approach and continuation:** only after fixed-start mastery, randomize
   approach distance, stair rise/depth, friction, heading, and multiple steps.

Foot pose and contact may be privileged **reward** signals during simulation;
they do not have to be policy observations at inference. A scripted or
inverse-kinematics step-up reference can seed the behavior before PPO
fine-tuning, which is more sample-efficient than waiting for random joint
actions to discover coordinated foot lifting.

## Progress gates for the next run

Use subgoal gates rather than one late body-height gate:

- by 25k steps: at least 5% of episodes lift each commanded front toe at least
  50 mm while the opposite support feet remain loaded;
- by 50k: at least 3% achieve a stable front-foot tread contact;
- by 100k: at least 2% raise the body at least 20 mm and complete the first
  step;
- abort immediately on repeated PPO KL divergence or if any earlier gate fails.

These are experiment gates, not claims of convergence or hardware safety.
After simulation success, evaluate randomized unseen stairs and then use a
tethered, current-limited physical test with an emergency stop.
