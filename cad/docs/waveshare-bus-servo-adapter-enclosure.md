# Waveshare Bus Servo Adapter (A) enclosure

This two-piece printed enclosure protects the project's selected Waveshare Bus
Servo Adapter (A). The board is a USB/UART half-duplex serial-servo adapter,
not a CAN controller. The official Waveshare STEP geometry and its 37 x 28 mm
mounting pattern control the fit.

## Access and mounting

- A broad short-side opening exposes the USB-C connector and 9-12.6 V external
  power terminal without removing the cover.
- A rounded lid opening above the terminal allows screwdriver access to its
  clamp screws.
- The opposite side opening exposes both three-pin servo connectors.
- A smaller end window exposes the UART/header area.
- Four M3 clearance holes at `(±20, ±30) mm` mount the enclosure to the robot
  body's 10 mm floor grid.
- Four 5 mm standoffs use 2.0 mm pilot holes for the board's M2 hardware.
- Four M3 screws secure the removable lid to blind pilot bosses.

The cavity is 48 x 40 x 21 mm and the rounded main footprint is 53 x 45 mm,
excluding mounting and lid ears.

Generate the STEP, STL, 3MF, and exploded fit-preview artifacts from `cad/`:

```powershell
.\scripts\generate_waveshare_bus_servo_adapter_enclosure.ps1
```

