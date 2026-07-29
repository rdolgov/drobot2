# Electrical bill of materials

This BOM implements the one-battery architecture in [README.md](README.md).
Links were checked on 2026-07-29; price, stock, seller, and included accessories
can change.

## Core power system

| Qty | Part | Recommended item / buy link | Notes |
| ---: | --- | --- | --- |
| 1 | 3S battery | [Zeee 3S 11.1 V 5200 mAh 80C XT60](https://zeeebattery.com/products/zeee-3s-lipo-battery-5200mah-11-1v-80c-xt60-1) | Existing or equivalent pack; confirm physical size before purchase |
| 1 | Balance charger | [HOTA D6 Pro overview](https://hotachargers.com/) or [Amazon ASIN B0827S7NYV](https://www.amazon.com/dp/B0827S7NYV) | Buy an authentic unit; AC-powered, dual-channel, 1S–6S LiPo balance and storage charging; useful if two packs are owned |
| 1 | Six-circuit fuse block with negative bus | [Blue Sea Systems 5025](https://www.bluesea.com/products/5025/ST_Blade_Fuse_Block_-_6_Circuits_with_Negative_Bus_and_Cover) or [Amazon ASIN B000THQ0CQ](https://www.amazon.com/dp/B000THQ0CQ) | Six positive fuse positions and six negative returns; 100 A block rating, 30 A per circuit |
| 1 | Main inline fuse holder | [Littelfuse FHAS100-BP](https://www.littelfuse.com/products/fuse-blocks-fuseholders-and-fuse-accessories/automotive-and-commercial-vehicle-fuse-holders/ato-mini-sealed-fuse-holders/fhas100-bp.aspx) | Sealed ATO holder, 12 AWG leads, 30 A maximum |
| 2 | 30 A ATO/ATC fuses | [Littelfuse ATO fuses at Mouser](https://www.mouser.com/c/circuit-protection/fuses/automotive-fuses/?series=ATO) | One installed at battery positive, one spare |
| 6 | 5 A ATO/ATC fuses | [Littelfuse ATO fuses at Mouser](https://www.mouser.com/c/circuit-protection/fuses/automotive-fuses/?series=ATO) | Four legs, Pi converter, and one spare |
| 2 | 1 A ATO/ATC fuses | [Littelfuse ATO fuses at Mouser](https://www.mouser.com/c/circuit-protection/fuses/automotive-fuses/?series=ATO) | Adapter branch and one spare |
| 1 pair | Battery connector | [Genuine Amass XT60H pair](https://www.stevensaero.com/product/xt60-connector-pair/) | Match battery polarity; accepts 12 AWG |
| 1 m each | 12 AWG red and black stranded copper wire | Local automotive, marine, or RC supplier | Battery, main fuse, and fuse-block studs; keep short |
| 2 m each | 18 AWG red and black stranded copper wire | Local automotive or marine supplier | Pi converter input |
| As needed | Heat-shrink ring/fork terminals | Match 12, 18, and 22 AWG plus Blue Sea #10-32 studs/#8-32 screws | Use a real ratcheting crimper and perform a pull test |

The Blue Sea 5025 is the clean recommendation because its fifth and sixth
circuits account for the Raspberry Pi and servo adapter. A cheaper six-way
fuse block can work, but its real terminal material, current rating, and screw
retention should be inspected rather than trusted from the listing alone. The
5025's published weight is approximately 0.25 kg, so its tray fit and mass
penalty must be reviewed before purchase; it is electrically convenient, not
yet mechanically approved for this robot.

## Raspberry Pi and controller

| Qty | Part | Recommended item / buy link | Notes |
| ---: | --- | --- | --- |
| 1 | Raspberry Pi 5-capable battery converter | [Pichondria 6–20 V / 2S–4S to 5 V 5 A converter](https://www.tindie.com/products/regaldreamtech/usb-pd-2030-to-5v-5a-converter-board-for-rpi5/) | Connect its DC solder pads to circuit 5; follow its Pi configuration tutorial |
| 1 | Short USB-C to USB-C power cable | 5 A, e-marked, 100 W or higher cable from a reputable vendor | Short/thick cable minimizes the 5 V drop |
| 1 | Waveshare Bus Servo Adapter (A) | [Waveshare product/documentation](https://www.waveshare.com/wiki/Bus_Servo_Adapter_%28A%29) | Existing selected controller; USB/UART TTL, not CAN |
| 1 | Short USB-A to USB-C data cable | [Adafruit 6 in USB A-to-C data cable](https://www.adafruit.com/product/4472) | Pi host to adapter; must carry data, not only power |
| 1 | Raspberry Pi 5 active cooler | [Raspberry Pi accessories](https://www.raspberrypi.com/products/) | Include in Pi power budget; choose after the Pi is confirmed |

If the robot actually uses a Pi 4, the same converter is acceptable. Raspberry
Pi recommends 5 V/3 A for Pi 4 and 5 V/5 A for full Pi 5 peripheral power.

## Servo power and data harness

| Qty | Part | Recommended item / buy link | Notes |
| ---: | --- | --- | --- |
| 2 packs | 300 mm 5264-3PIN servo cables, six per pack | [Waveshare premade cables](https://www.waveshare.com/sr-cable-5264-3pin.htm) | Eight leg jumpers, one adapter data pigtail, three spares |
| 1 pack | 900 mm 5264-3PIN servo cables, six per pack | [Waveshare premade cables](https://www.waveshare.com/sr-cable-5264-3pin.htm) | Cut to make four body-to-leg pigtails; two spares |
| 1 | Five-port data junction | [WAGO 221-415](https://www.wago.com/us/wire-splicing-connectors/compact-splicing-connector/p/221-415) or [Amazon ASIN B06XH47DC2](https://www.amazon.com/dp/B06XH47DC2) | Adapter `DATA` plus four leg `DATA` wires; all five ports are common |
| As needed | 22 AWG flexible stranded wire, red/black/white | Electronics or RC supplier | Mini-SPOX maximum supported wire size is 22 AWG |
| As needed | Braided sleeve and strain relief | Electronics supplier | Protect all four moving leg harnesses from pinching and abrasion |
| 1 | Multimeter with continuity mode | Existing tool | Mandatory for connector polarity checks |

For later fully custom cables:

| Qty | Part | Buy link | Notes |
| ---: | --- | --- | --- |
| 10 | Molex Mini-SPOX 3-position housing `50375033` | [DigiKey Mini-SPOX family](https://www.digikey.com/en/product-highlight/m/molex/mini-spox-wire-to-board-connector-system) | Buy spares |
| 50 | Molex terminal `08701039` | [DigiKey Mini-SPOX family](https://www.digikey.com/en/product-highlight/m/molex/mini-spox-wire-to-board-connector-system) | 22–28 AWG, 1.90 mm maximum insulation diameter |
| 1 | Correct open-barrel crimp tool | [Official Molex `638281900`](https://www.molex.com/en-us/products/part-detail/638281900) | The official tool is expensive; premade cables are preferred for the first robot |

Do not substitute a connector because it merely looks like JST-XH. The servo
connector is 2.50 mm Molex Mini-SPOX/5264.

## Optional safety and measurement items

| Qty | Item | Purpose |
| ---: | --- | --- |
| 1 | LiPo charging container or quality charging bag | Reduces exposure to nearby combustible material; it does not make charging unattended-safe |
| 1 | 3S cell checker/low-voltage alarm | Quick per-cell check through the balance connector |
| 1 | DC clamp meter or inline RC watt meter rated above 30 A | Measure real gait current and refine the runtime model |
| 1 | Easily reachable DC-rated disconnect | Provides a fast shutdown method; the battery XT60 remains the final physical disconnect |
| 1 | Spare set of 1 A, 5 A, and 30 A fuses | Replace only after finding the cause of a blown fuse |

## If the existing four-way ELECTOP block was already purchased

The [ELECTOP four-way positive fuse block](https://www.amazon.com/dp/B0CN92862F)
can still feed the four legs. The
[Blue Sea 2314 covered MiniBus](https://www.amazon.com/dp/B000OTJ89Q) can still
serve as the common negative bus.

However, those two parts do not provide fused positive outputs for the Pi and
adapter. Add:

- one covered positive bus, such as the
  [Blue Sea 2304/2314 100 A MiniBus](https://www.bluesea.com/products/2304/Common_100A_Mini_BusBar_-_5_Gang);
- one 5 A inline ATO holder from that positive bus to the Pi converter;
- one 1 A inline ATO holder from that positive bus to the adapter;
- a cover for every exposed positive bus.

The main-feed sequence becomes:

`battery + → 30 A main fuse → disconnect → covered positive bus`

The positive bus then feeds the ELECTOP input, the 5 A Pi inline fuse, and the
1 A adapter inline fuse. This is electrically workable but larger and usually
costs more than replacing the distribution parts with one six-circuit 5025.
