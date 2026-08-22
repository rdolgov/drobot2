# Raspberry Pi 5 enclosure

This dedicated three-piece enclosure mounts a Raspberry Pi 5 to the robot body
floor without depending on the removable lid. The exact catalog Pi model is
used in the fit preview; the printable pieces are the base, lid, and BNO085
protection roof.

## Mechanical interfaces

- The base uses four M3 clearance holes at `(±40, ±50) mm`, all members of the
  robot body's 10 mm-pitch floor grid.
- The Pi uses its standard 58 x 49 mm four-hole pattern on 6 mm standoffs with
  2.2 mm pilot holes for M2.5 self-tapping screws.
- Four M3 screws secure the lid to blind pilot towers in the base.
- The lid carries the exact Adafruit BNO085 four-hole M2 pattern on 4 mm
  standoffs. The separate roof is open on all four sides for wire access.

The internal cavity is 96 x 66 x 31 mm. Broad windows on all four walls leave
corner posts and lower sills while exposing the Pi connectors and providing
airflow. Six rounded lid slots add vertical ventilation around the central IMU.

Generate all STEP, STL, 3MF, and fit-preview files from `cad/` with:

```powershell
.\scripts\generate_raspberry_pi_5_enclosure.ps1
```

Recommended hardware is four M3 body screws with washers/nuts or standoffs,
four short M2.5 Pi screws, four M3 lid screws, and four M2 nylon through-bolts
with nuts for the IMU and its roof. Confirm screw lengths against the actual
robot floor and electronics before installation.
