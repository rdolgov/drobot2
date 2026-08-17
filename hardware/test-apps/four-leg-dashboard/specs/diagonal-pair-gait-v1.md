# Diagonal-pair flat-support gait V1

## Purpose

This is a separate two-beat test route for the four-leg dashboard. It leaves
the existing one-leg-at-a-time distributed crawl unchanged. The UI button
**TEST DIAGONAL PAIRS** calls `POST /api/diagonal-pair-forward`; the existing
button continues to call `POST /api/crawl-forward`.

The pair order is:

1. front-left plus rear-right; and
2. front-right plus rear-left.

Only the selected diagonal is airborne. The opposite diagonal uses the same
exact flat-sole inverse-kinematics branch as the rectangular-shoe crawl.

## Periodic foot sequence

One 4-second cycle contains two 2-second pair placements. Each placement uses:

| Phase | Step fraction | Time | Action |
| --- | ---: | ---: | --- |
| Lift pair | 0.25 | 0.50 s | Raise both selected shoes to full lift |
| Swing pair | 0.25 | 0.50 s | Advance both selected shoes by 96 mm |
| Lower pair | 0.25 | 0.50 s | Lower both shoes back to the flat branch |
| Firm plant | 0.10 | 0.20 s | Hold all four flat targets |
| All-feet push | 0.12 | 0.24 s | Push all targets rearward by 48 mm |
| Pair settle | 0.03 | 0.06 s | Hold the completed placement |

Each diagonal begins at either minus one-half stride or zero offset. Its swing
adds one full stride. The four-foot phase then subtracts one-half stride from
every target. After both pairs have moved, all joint targets exactly match the
start of the cycle, so no reset motion is required.

## Shoe and joint geometry

The mode reuses the tracked rectangular-shoe geometry:

- proximal axis spacing: `0.159896689 m`;
- effective knee-to-ground contact length: `0.190896689 m`;
- sole size: `100 x 60 mm`;
- stance fore/aft offset: `0.080 m`;
- exact flat-sole stance depth: `0.329341447 m`;
- hip abduction: `0 degrees`; and
- contact-centre lift: `0.035 m`.

Every planted leg satisfies `hip flexion + knee = 0 degrees`. The two unloaded
legs use general two-link IK during lift, swing, and lowering, then rejoin the
flat branch before the firm-plant phase.

## Active parameters and calculated envelope

| Parameter | Value |
| --- | ---: |
| Hardware cycles | Continuous until STOP + DISARM |
| Cycle time | 4.0 s |
| Total gait time | Unbounded until explicitly stopped |
| Finite fallback cycles | 2 |
| Pair placements per cycle | 2 |
| Pair stride | 0.096 m |
| Four-foot push after each pair | 0.048 m |
| Lift | 0.035 m |
| Maximum absolute hip-flexion target | 67.25 degrees |
| Maximum absolute knee target | 73.11 degrees |
| Peak requested joint rate | 144.1 degrees/s |
| Horizontal-swing long-edge clearance | 19.43-29.90 mm |

Offline source sampling found finite targets and zero periodic endpoint error.
The target envelope is inside the tracked hip-flexion range of plus/minus 90
degrees, knee range of plus/minus 120 degrees, and 270-degree-per-second command
cap.

## Stability boundary

The supporting diagonal provides two contact areas whose centers form a line,
not a three-point support polygon. Open-loop balance therefore depends on the
real center of mass remaining close to that diagonal and on both shoes carrying
load. Unequal shoe height, chassis roll, floor variation, servo lag, or wiring
drag can unload one support and produce a rapid tip.

This route has not been simulated or moved on hardware. Restart the dashboard
after updating so the new Python route and browser button are loaded.
