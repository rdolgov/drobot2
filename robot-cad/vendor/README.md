# Vendor and immutable reference CAD

These files are inputs to custom-part generators. Do not edit or transform them
in place; apply placement transforms only in Python assembly code.

| File | Manufacturer / origin | Exact part or reference | Source | Retrieved | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `servos/waveshare_feetech_st3215_servo.step` | Waveshare Feetech | ST3215 Servo | [step.parts catalog](https://www.step.parts/parts/waveshare_feetech_st3215_servo) | 2026-07-25 | `29954eb73bd22b3f9536de2c1d8f96843b5c5b32288a8f4cb09709b8b892e39b` |
| `references/so101/Upper_arm_SO101.step` | SO-101 source project | Original upper-arm B-rep reference | Migrated from local `text-to-cad` Git LFS object | 2026-07-25 | `efa19a6dd2ccb459248500c76629cfa840630e7e15d9e146394d31da1525dd61` |
| `references/lekiwi/base_camera_mount.stl` | SIGRobotics-UIUC LeKiwi | Arducam 5 MP wide-angle base camera mount | [LeKiwi source](https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/3DPrintMeshes/base_camera_mount.stl) | 2026-07-26 | `631ed680e524b90dcb59dd315830ed1bac842dfd5e77c412ef4c52ff4da2e854` |
| `references/lekiwi/arducam_5mp_camera_model.stl` | SIGRobotics-UIUC LeKiwi | Arducam camera assembly reference mesh | [LeKiwi source](https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/URDF/meshes/Camera-Model-v3-1.stl) | 2026-07-26 | `f5ff94543c19e91a6d5ffb540e2d25347d813d191793405ff2e4178230dec21e` |
| `sensors/adafruit_bno085_stemma_qt.step` | Adafruit Industries | Product 4754 BNO085 STEMMA QT breakout | [Official Adafruit CAD](https://github.com/adafruit/Adafruit_CAD_Parts/tree/main/4754%20BNO085%20STEMMA%20QT) | 2026-07-26 | `115ae06a3215c4eb6bb0eb5f4aeeaa39f13fe0a76d8be2ef1416dc8a85b804d1` |

The ST3215 catalog file is the exact geometry used to derive the keyed cavity.
step.parts searches for `STS3212`, `ST3212`, and `Feetech STS3212` returned no
matches. The catalog separately lists ST3215, ST3215-HS, and ST3235; those
variants must not be interchanged without revalidation.

step.parts searches for `Arducam 5MP wide angle USB camera`, `Arducam`,
`USB camera`, and `wide angle camera` returned no exact purchasable camera
match on 2026-07-26.  The camera and unchanged printable mount references
therefore come from the upstream Apache-2.0 LeKiwi project.  The body-side
interface is modeled parametrically from the mount's measured three-hole M3
row at 20 mm pitch; the vendor meshes remain immutable.

step.parts searches for `BNO085`, `BMI088`, `ICM-42688-P`, and "IMU breakout"
returned no exact matches on 2026-07-26.  The BNO085 reference
therefore comes from Adafruit's official product-4754 CAD repository.  Its
measured 20.32 x 17.78 mm four-hole pattern and 2.5 mm board holes are used
without altering the downloaded STEP geometry.

Git LFS stores STEP and other large CAD binaries. Run `git lfs pull` after
cloning before generating geometry.
