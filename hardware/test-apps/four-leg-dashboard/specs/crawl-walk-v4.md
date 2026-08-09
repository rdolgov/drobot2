# Crawl walk V4: wide mirrored stance

## Decision

V4 replaces the V3 common-direction walking posture. The assembled robot fell
backward on V3's first floor move. Reproducing that geometry in Isaac also
produced a backward collapse. V4 uses separate mechanisms for the two support
dimensions:

- front upper legs/knees project forward and rear upper legs/knees project
  backward for fore/aft support;
- hip abduction moves every leg outward for left/right support; and
- front and rear foot targets are splayed about 113 mm beyond their hip pivots.

The dashboard remains a **supported commissioning tool**. V4 stayed upright in
Isaac, but did not pass every torque, tracking, slip, and foot-unloading gate.

## Geometry

Each sagittal leg is a two-link chain with equal link length
`L = 0.159896689 m`. For down target `d`, fore/aft target `x`, and knee bend
`q`:

```text
r = sqrt(d^2 + x^2)
q = acos((r^2 - 2 L^2) / (2 L^2))
```

V4 uses `d = 0.272960722 m` and `|x| = 0.113064033 m`. The exact neutral
targets sent to the assembled robot are front hip `+45 degrees`, rear hip
`-45 degrees`, front knees at relative `-45 degrees`, and rear knees at relative
`+45 degrees`. The rear sign was corrected after a supported hardware
observation showed that rear `-90 degrees` lifted the lower legs. The front
knee magnitude was reduced after `-90 degrees` produced excessive downward
bend, and the rear magnitude was matched at `45 degrees`. Walking knee changes
are applied around these hardware-derived standing baselines.
This is a deliberately low, long supported-test posture.

The nominal hip-abduction magnitude is 15 degrees outward. The assembled left
axes use logical `-15 degrees` on Legs 1 and 3, while the right axes use logical
`+15 degrees` on Legs 2 and 4. At this leg length it adds roughly 71 mm of
lateral reach per side, increasing left/right contact separation by about
141 mm. Electrical direction signs remain in the individual servo profiles and
are applied after these logical joint targets.

## Why not an 80-degree crouch

An 80-degree centered knee bend reduces vertical leg reach to about 245 mm.
That creates travel but increases sustained joint load and reduces collapse
margin. Rated-torque Isaac checks failed before gait motion:

| Candidate | Nominal height | Result |
| --- | ---: | --- |
| 60-degree mirrored, 12-degree outward | 275 mm | collapsed during stance settle; base z 126 mm |
| 80-degree mirrored, 12-degree outward | 243 mm | collapsed during stance settle; base z 108 mm |

The knee motor cannot safely provide a Cartesian walking stroke by itself.
Hip flexion and knee must move together so the foot follows the intended
forward/down path instead of an arc that can kick the body.

## Tracked supported-test profile

| Parameter | Value |
| --- | ---: |
| Cycle period | 20 s |
| Cycles | 1 |
| Foot stride | 35 mm |
| Foot lift | 16 mm |
| Stance down target | 272.961 mm |
| Front/rear foot splay | 113.064 mm |
| Front/rear hip flexion | +45 / -45 degrees |
| Front / rear knees | -45 / +45 degrees relative |
| Hip abduction, left / right | -15 / +15 degrees outward |
| Open-loop forward transfer | 16 mm |
| Open-loop lateral transfer | 12 mm |
| First swing leg | front-left |
| Servo torque setting | 30% |
| Command ramp | 45 degrees/s |

Joint targets use the ST3215 signed extended-position coordinate across the
encoder seam. The current Leg 1 knee center (`1005`) maps the front stance
target `-45 degrees` to raw `493`, without crossing the seam. A manual
`-90 degrees` target maps to raw `-19`; the transport sign-encodes that target,
matching the official SDK behavior, rather than rejecting it or changing it to
the potentially long-way absolute target `4077`. Feedback supplied as either
signed `-19` or single-turn `4077` is normalized to the nearest calibrated
angle, preserving the `-90 degrees` reading across power cycles.

During a swing the selected foot advances 35 mm. Its diagonal support partner
pushes rearward 17.5 mm and both adjacent support feet push rearward 8.75 mm.
All four hip-flexion commands therefore participate in every step.

## Isaac result

An intermediate 30 mm / 8-degree posture stayed upright and advanced, unlike
V3:

| Metric | V4 result |
| --- | ---: |
| Settled body tilt | 1.01 degrees |
| Maximum body tilt | 6.31 degrees |
| Minimum base height | 342.9 mm |
| Forward displacement | +26.4 mm |
| Lateral drift | 2.6 mm |
| Expected support contact | 99.89% |
| Maximum support slip | 19.0 mm |
| Maximum joint tracking error | 0.267 rad |
| Contact-verified steps | 0 / 4 |

That intermediate posture and balance direction were materially better, but
the run remains a
**FAIL** because tracking error exceeds 0.15 rad, support slip exceeds 15 mm,
and none of the provisional fork-tip contacts remained unloaded for the
required interval. Isaac contact uses simple fork-tip spheres rather than the
real foot/floor interface, so the result is a conservative boundary rather
than proof of real failure or success. The exact 45-degree / 15-degree posture
was intentionally not run to completion after the user requested direct
supported hardware testing instead of further simulation.

## Supported hardware test ladder

1. Rigidly support or tether the chassis, keep every foot clear, and keep the
   physical power cutoff in hand.
2. Start the dashboard, connect IDs 1-12, and require nominal voltage,
   temperature, current, and `0 / 12` armed.
3. Press **SET WIDE WALK STANCE** only. Confirm both front hip-flexion readings
   converge to `+45 degrees`, both rear readings to `-45 degrees`, front knees
   to `-45 degrees`, rear knees to `+45 degrees`, and all four abduction joints
   to left `-15 degrees` / right `+15 degrees` outward.
4. Hold for 10 seconds while watching total/per-leg current, voltage sag,
   temperature, mechanical sag, noise, and any backward pitch. Press
   **STOP + DISARM** immediately if anything is abnormal.
5. If the feet are going to take weight, lower the external support gradually;
   do not drop the full body mass onto the servos. Keep the tether able to catch
   a fall. Record whether the stance holds and the loaded current values.
6. Disarm, re-support the body, and only then use **TEST COORDINATED MOTION**.
   Verify the front-left, rear-right, front-right, rear-left swing order and
   stop on sag, pitch, collision, cable pull, or supply warning.
7. Do not run untethered repeated cycles until one real cycle holds body height,
   moves all four feet as intended, and produces acceptable telemetry.

## Commands

```powershell
cd C:\Users\roman\Documents\dev\drobot2\hardware\test-apps
.\install-test-apps.ps1
.\start-four-leg-web.ps1 -Port COM4
```

The exact Isaac command and generated report are tracked in the dashboard
README and `validation/isaac-coordinated-support-push.json`.
