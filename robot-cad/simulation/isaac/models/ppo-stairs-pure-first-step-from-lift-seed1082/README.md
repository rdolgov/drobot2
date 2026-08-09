# Pure-PPO first-step transfer from the 19 cm lift skill

This package preserves iteration 300 of a 120-iteration, 128-environment PPO
transfer from the supported 19 cm foot-lift checkpoint to the exact first stair:
`180 mm` rise and `250 mm` tread depth.

The actor still receives only 70 deployable IMU, VL53L5CX depth, joint,
previous-action, and foot-load/contact values. The transfer adds no scripted
leg choice, gait phase, inverse kinematics, reference trajectory, or privileged
actor input. It retains the supported-lift reward while adding symmetric
any-foot first-tread placement, retained support, and base-elevation reward.

The run processed 368,640 transitions. At iterations 301 and 302, some reset
episodes achieved force-verified first-tread contact and short supported holds.
The best logged reset batch averaged 0.333 tread contacts and 0.0778 seconds of
tread hold. First-step success remained 0% because maximum logged base gain was
only 0.0108 m, below the required 0.06 m. This is a contact checkpoint, not a
completed-step policy.

