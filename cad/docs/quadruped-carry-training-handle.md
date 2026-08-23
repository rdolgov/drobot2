# Quadruped carry and training handle

This one-piece U handle bolts to the upper central M3 grids on both long sides
of the robot body. It provides a central carry grip and two rounded slots for
looping resistance bands during supported gait training.

## Body interface

- Each side plate uses six existing M3 holes: `X=-10, 0, +10 mm` at
  `Z=70 and 80 mm`.
- The narrow 36 x 28 x 6 mm plates span `X=-18..+18 mm`, sit above the side
  wire ports, and do not cover the 32 x 20 mm openings centered at `Z=50 mm`.
- The adjacent 76 mm leg plates end at `X=-22 mm` and begin at `X=+22 mm`,
  leaving 4 mm clearance on each side of the handle plate.
- Use wide washers and nyloc nuts inside the body. Select bolt length after
  checking the printed 6 mm plate and 3.2 mm body wall.

## Handle

- The rounded 194 mm crossbar spans the body width.
- Its lower surface is 40 mm above the installed 100 mm body lid.
- Two 24 x 9 mm rounded slots accept looped resistance bands while preserving
  a 100 mm unobstructed central hand-grip region.
- Rounded post-to-crossbar gussets reinforce both upper corners.

Generate the STEP, STL, 3MF, and installed preview from `cad/`:

```powershell
.\scripts\generate_quadruped_carry_training_handle.ps1
```

This is a first-pass printed lifting component, not a certified lifting point.
Use PETG, ASA, polycarbonate, or nylon rather than brittle PLA, print on its
side so the U outline lies in the layer plane, and proof-load it close to the
floor before carrying the robot.
