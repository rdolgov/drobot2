# V28 representative post-transfer swing pilot

The video and trajectory in this directory show one deterministic seed-510
success from the 2,048-step V28 experimental swing-residual policy:

- front-left lift: 219.8 mm
- maximum support slip: 16.2 mm
- maximum body tilt: 11.0 degrees
- exact stair geometry: 250 mm tread, 180 mm rise
- real-test effort cap: 0.8825985 N m

This is not a robustness claim. The independent checkpoint-512 evaluation was
2/5 and the final two training episodes failed below 190 mm. The video is
published so the successful motion can be inspected while those limitations
remain explicit.

The policy is camera-blind. RGB is used only to record this review video; the
policy consumes IMU, joint, contact/load, composite-COM, previous-action, and
known analytic stair-profile inputs.

Video SHA-256:
`baec1813a9fb4973d0c0a8f564b13d4a06db591b4bd05b53ab2bc344ba4937f8`
