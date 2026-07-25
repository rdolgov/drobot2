# Vendor and immutable reference CAD

These files are inputs to custom-part generators. Do not edit or transform them
in place; apply placement transforms only in Python assembly code.

| File | Manufacturer / origin | Exact part or reference | Source | Retrieved | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `servos/waveshare_feetech_st3215_servo.step` | Waveshare Feetech | ST3215 Servo | [step.parts catalog](https://www.step.parts/parts/waveshare_feetech_st3215_servo) | 2026-07-25 | `29954eb73bd22b3f9536de2c1d8f96843b5c5b32288a8f4cb09709b8b892e39b` |
| `references/so101/Upper_arm_SO101.step` | SO-101 source project | Original upper-arm B-rep reference | Migrated from local `text-to-cad` Git LFS object | 2026-07-25 | `efa19a6dd2ccb459248500c76629cfa840630e7e15d9e146394d31da1525dd61` |

The ST3215 catalog file is the exact geometry used to derive the keyed cavity.
step.parts searches for `STS3212`, `ST3212`, and `Feetech STS3212` returned no
matches. The catalog separately lists ST3215, ST3215-HS, and ST3235; those
variants must not be interchanged without revalidation.

Git LFS stores STEP and other large CAD binaries. Run `git lfs pull` after
cloning before generating geometry.
