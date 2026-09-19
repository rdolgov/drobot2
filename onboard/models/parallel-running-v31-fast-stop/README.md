# V31 fast run and graceful stop

`model_1650.onnx` is the selected V31 research policy for Raspberry Pi
commissioning. It uses the 50-observation, 12-action deployment contract at
60 Hz, an integrated speed-scaled diagonal-trot clock, relative-yaw heading
hold, and a trained transition to a four-foot stopped stance.

The dashboard deliberately caps the first hardware trials at `0.10 m/s`, even
though the training range extends to `0.20 m/s`. Start supported or tethered at
`0.05 m/s` for two or three seconds, with a physical cutoff ready. A normal
timer completion ramps the command to zero over `0.60 s`, continues inference
for a `1.0 s` stopped hold, and then leaves torque holding the policy's stopped
stance. The explicit Stop button and hardware faults retain the immediate
disarm path.

V31 passed its behavioral simulation gates, including abrupt and ramped stop
cases, but failed the conservative sustained-effort occupancy gate. Treat it as
a research candidate until real voltage/current/temperature evidence supports
a higher commissioning speed.

The dashboard keeps V30 available in the policy selector for rollback:

```text
/home/rd/drobot2/onboard/models/parallel-walking-v30-symmetry-gated-robust-straight-crawl/model_5000.onnx
```

Training, evaluation, hashes, and known limitations are documented in
`simulation/docs/rl-fast-run-stop-v31.md`.
