# Quadruped electrical system

This folder is the electrical source of truth for the current twelve-servo
quadruped concept. It covers one 3S battery, four three-servo legs, the
Waveshare Bus Servo Adapter (A), a Raspberry Pi, the USB camera, and the
body-mounted BNO085.

The baseline is intentionally sized for a Raspberry Pi 5. A Pi 4 can use the
same power system with extra margin.

> Status: engineering proposal, not hardware-validated. Wire temperatures,
> voltage drop, fuse behavior, converter temperature, data reliability, and
> walking runtime must be measured on the finished robot before unattended or
> high-load operation.

## Documentation record

- **What changed:** established a complete one-battery architecture for four
  fused servo legs, the Raspberry Pi, and the Waveshare serial-bus adapter.
- **Why:** the earlier four-circuit plan did not include separate fused power
  for compute and controller electronics.
- **Editable sources:** this file, [BOM.md](BOM.md),
  [SCHEMATICS.md](SCHEMATICS.md), and
  [power-budget.csv](power-budget.csv).
- **Reproduction:** from the repository root, run
  `Import-Csv electrical\power-budget.csv | Format-Table -AutoSize` in
  PowerShell to review the retained estimates. There is no generated
  electrical artifact.
- **Validation performed:** manufacturer ratings and connector identities
  were checked against the sources listed below; Markdown/local-link and CSV
  structure were checked. No wire, converter, fuse, battery, data-bus, or
  walking test has been performed.
- **Outputs:** documentation and a CSV planning model only; no CAD geometry,
  PCB, or manufactured harness was generated.
- **Known limitations:** the exact Raspberry Pi model and any additional real
  CAN interface are not selected, component fit on the electronics tray is
  unverified, and every current/runtime figure remains an estimate.

## Important controller naming

The selected Waveshare Bus Servo Adapter (A) is **not CAN bus**. It is a
USB/UART adapter for the ST3215 single-wire, half-duplex TTL serial bus.
Waveshare specifies a 9–12.6 V external input and USB or UART host control.

If a real CAN HAT or CAN controller is added later:

- power it from the Raspberry Pi rail exactly as its manual specifies, never
  directly from the 3S battery unless it explicitly accepts 9–12.6 V;
- keep `CANH` and `CANL` completely separate from the ST3215 `DATA` wire;
- reserve 2 W inside the Raspberry Pi power budget until the exact board is
  selected;
- share system ground unless the selected CAN interface is galvanically
  isolated and its manual says otherwise.

## Baseline architecture

Use one 3S 11.1 V nominal, 12.6 V fully charged battery and one covered
six-circuit ATO/ATC fuse block with an integrated negative bus.

The recommended branch schedule is:

| Circuit | Fuse | Wire | Load |
| --- | ---: | --- | --- |
| 1 | 5 A | 22 AWG to Mini-SPOX pigtail | Front-left leg, 3× ST3215 |
| 2 | 5 A | 22 AWG to Mini-SPOX pigtail | Front-right leg, 3× ST3215 |
| 3 | 5 A | 22 AWG to Mini-SPOX pigtail | Rear-left leg, 3× ST3215 |
| 4 | 5 A | 22 AWG to Mini-SPOX pigtail | Rear-right leg, 3× ST3215 |
| 5 | 5 A | 18 AWG | 3S-to-5 V/5 A Raspberry Pi converter |
| 6 | 1 A | 22 AWG | Waveshare adapter 9–12.6 V input |

Install a separate **30 A main fuse** in 12 AWG wire as close as practical to
the battery-positive connector; target no more than 100 mm (4 in) of
unprotected positive wire. The 30 A fuse protects the main harness. The six
smaller fuses protect the individual branch wires. A fuse does not regulate or
smooth current.

Do not install a fuse in battery negative. Battery negative goes directly to
the fuse block's negative-bus input stud.

See [SCHEMATICS.md](SCHEMATICS.md) for the complete power and data drawings and
[BOM.md](BOM.md) for purchasable parts.

## Why power is split but communication is shared

Each leg still uses the familiar three-wire daisy chain:

`body pigtail → servo 1 → servo 2 → servo 3`

The three conductors are `DATA`, battery positive, and ground. Power enters
each leg through its own 5 A branch fuse, while all four `DATA` conductors meet
the adapter `DATA` conductor in one passive five-port junction.

The adapter's servo-output `VCC` conductor must be disconnected and insulated.
The adapter is powered separately through its green screw terminal on the 1 A
branch. This prevents all twelve servo currents from flowing through the
adapter power path. The official adapter schematic labels that path `MAX:6A`.

The four leg grounds, adapter ground, and Raspberry Pi converter input ground
all terminate on the same negative bus. This common ground provides the
reference for the single-wire data signal.

## Raspberry Pi and controller power

The Raspberry Pi must not receive raw 3S battery voltage.

Use a converter designed for a Pi 5 that accepts the 3S pack range and provides
5 V at 5 A through USB-C. The baseline BOM uses the Pichondria converter:

- input: 6–20 V DC or 2S–4S lithium battery;
- output: 5 V, 5 A;
- use a short, 5 A-rated USB-C cable;
- allow airflow around the converter.

The converter vendor instructs Pi 5 users to set:

```text
usb_max_current_enable=1
```

in `/boot/firmware/config.txt` and `PSU_MAX_CURRENT=5000` in the EEPROM
configuration. Apply those overrides **only** after confirming that the exact
converter, wiring, and USB-C cable are capable of 5 A. A Pi 5 can boot from a
good 5 V/3 A supply, but Raspberry Pi limits downstream USB power when a 5 A
supply is not detected.

The Pi power allowance includes:

- the Raspberry Pi;
- the active cooler/fan;
- the Waveshare adapter's USB logic connection;
- the selected Arducam USB camera;
- the BNO085;
- up to 2 W for a future CAN HAT or other small interface.

The Waveshare adapter also needs its 9–12.6 V screw-terminal input from circuit
6. Set its jumper to the documented USB-control position when connected to the
Pi by USB.

## Servo wiring limits

Waveshare lists the 12 V ST3215 at 200 mA no-load and 2.7 A locked-rotor
current. One leg therefore has:

- 0.6 A total if all three motors are simultaneously spinning with no load;
- 8.1 A theoretical total if all three are locked;
- an expected normal average that must remain well below the locked-rotor
  case.

The Molex Mini-SPOX terminal used by the servo accepts 22–28 AWG wire and the
connector family is rated up to 3 A with 22 AWG under its specified test
conditions. The first connector in a three-servo leg carries the combined leg
current, so it is the limiting connection even though an automotive 5 A fuse
tolerates short motor transients.

Start with the 5 A branch fuses, but verify that each leg remains below 3 A
average and that the first connector does not heat appreciably. If a leg
sustains more than 3 A or the connector rises more than approximately 10 °C
above ambient, add another power-injection point or individual servo power
feeds. Do not simply install a larger fuse.

For the first build, use premade Waveshare 5264-3PIN cables and cut one end to
make the four body pigtails. This is more reliable than learning on the small
Mini-SPOX contacts. The exact parts for later custom crimping are:

- housing: Molex `50375033` / `0050375033`;
- terminal: Molex `08701039` / `0008701039`;
- wire: 22–28 AWG, maximum insulation diameter 1.90 mm.

Never trust wire color or connector orientation alone. Before plugging in a
servo, use continuity mode to verify `DATA`, `VCC`, and `GND` end to end.

## Rough power and runtime estimate

The assumed battery is 3S, 5.2 Ah:

```text
nominal energy = 11.1 V × 5.2 Ah = 57.7 Wh
planning energy = 57.7 Wh × 0.80 = 46.2 Wh
```

The 80% factor reserves energy for voltage sag, converter loss, and avoiding a
deep LiPo discharge. It is not a measured battery test.

| Operating case | Estimated battery power | Estimated runtime |
| --- | ---: | ---: |
| Pi, camera, adapter; servos torque-disabled | 15 W | about 185 min |
| All servos moving unloaded plus electronics | 44 W | about 63 min |
| Light standing/slow motions | 50 W | about 55 min |
| Normal walking planning case | 100 W | about 28 min |
| Aggressive gait/obstacle work | 175 W | about 16 min |
| All twelve servos locked | over 400 W theoretical | fault; do not operate |

A reasonable first expectation is **20–40 minutes of walking from one pack**.
The actual number can be substantially different because gait, robot mass,
foot impacts, servo acceleration, cooling, terrain, and battery sag dominate
the result. The calculations are retained in
[power-budget.csv](power-budget.csv).

At 5.2 Ah and 80C, the pack's advertised discharge figure is
`5.2 × 80 = 416 A`. That is a rating calculation, not a prediction of actual
short-circuit current, but it explains why the 30 A main fuse must remain close
to the battery.

## Battery charging and low-voltage handling

Charge the 3S LiPo only with a balance charger that supports 3S LiPo chemistry.
Connect both the XT60 main lead and the four-wire 3S JST-XH balance lead to the
charger. For a 5.2 Ah pack, a conservative 1C charge rate is 5.2 A; a lower
setting such as 4 A is acceptable and easier on a 50 W charger.

During charging:

- remove the pack from the robot;
- charge on a nonflammable surface in an attended location;
- select `LiPo BALANCE`, 3S, and verify the charger detects three cells;
- never charge a swollen, punctured, hot, or mechanically damaged pack;
- use the charger's storage program before leaving a pack unused for more
  than a few days.

Provide a software warning around 10.5 V under light load and stop testing if
any individual cell approaches the battery manufacturer's minimum. Pack
voltage alone cannot reveal an imbalanced cell, so periodically inspect
per-cell voltage with the balance charger.

## Build sequence

1. Build only the battery connector, 30 A main fuse, and fuse block.
2. With the battery disconnected, verify that positive and negative buses are
   not shorted.
3. Power the empty fuse block and verify battery voltage and polarity.
4. Install the Raspberry Pi converter branch by itself. Verify converter
   polarity and output before connecting the Pi.
5. Install and test the adapter's 1 A branch and USB data connection.
6. Build one leg pigtail and one three-servo chain. Verify every Mini-SPOX pin
   by continuity, then test that leg with the robot mechanically unloaded.
7. Repeat for the other three legs.
8. Add the five-wire passive data junction only after every leg has unique
   servo IDs.
9. Perform the validation below before fitting the lid or walking.

## Required validation

Record the results next to this document before calling the design validated:

- battery polarity and full-charge voltage;
- converter output at Pi idle and CPU/USB load;
- `vcgencmd get_throttled` result during a 10-minute Pi load test;
- idle, standing, normal-walking, and peak battery current;
- current of each leg during the same gait;
- voltage at the first and last servo of every leg;
- first-connector and wire temperature after 10 minutes;
- fuse nuisance trips or discoloration;
- servo packet errors at the selected baud rate;
- battery cell voltages immediately after the walking test;
- measured runtime to the conservative stop threshold.

The five-port `DATA` junction creates a small star topology. It is expected to
work at robot-scale cable lengths but is not yet validated at the ST3215
default 1 Mbps. If packet errors appear, shorten the four stubs, route `DATA`
with a nearby ground, reduce baud rate if supported by the complete software
stack, or change the body wiring to a linear trunk.

## Multiple-battery variants

The baseline uses one pack. Do not parallel packs merely by joining their
positive leads.

For two or four packs, each pack needs its own main fuse and disconnect. Keep
the different battery positives isolated and assign each battery specific leg
branches. All grounds may be joined for the shared TTL data reference. Use
packs of the same chemistry and similar state of charge, and confirm that
removing any one pack cannot back-feed it through the adapter or Raspberry Pi.

Complete and measure the one-battery architecture before changing to multiple
packs; otherwise current-sharing and runtime problems become much harder to
diagnose.

## Sources

Sources and shopping links were checked on 2026-07-29:

- [Waveshare ST3215 product specifications](https://www.waveshare.com/st3215-servo.htm)
- [Waveshare Bus Servo Adapter (A) documentation](https://www.waveshare.com/wiki/Bus_Servo_Adapter_%28A%29)
- [Waveshare Bus Servo Adapter (A) schematic](https://files.waveshare.com/wiki/Bus-Servo-Adapter-%28A%29/Bus%20Servo%20Adapter%20%28A%29-Sch.pdf)
- [Molex Mini-SPOX product specification](https://www.molex.com/content/dam/molex/molex-dot-com/products/automated/en-us/productspecificationpdf/526/5264/52641001-PS-000.pdf)
- [Raspberry Pi power-supply guidance](https://www.raspberrypi.com/documentation/computers/getting-started.html#power-supply)
- [Raspberry Pi 5 USB Power Delivery note](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-009856-WP-1-USB%20Power%20delivery%20on%20Raspberry%20Pi%205.pdf)
- [Pichondria Pi 5 converter tutorial](https://pichondria.com/usb-pd-2-0-3-0-to-5v-5a-converter-for-raspberrypi-5-tutorial/)
