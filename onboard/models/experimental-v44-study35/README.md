# V44 study35 — offline candidate, NOT deployable

Selected study34 checkpoint396 plus study35 target mapping. The name refers to
an experiment inside V44, not the previous production model V35. No new training
was performed to produce this bundle.

- `policy.onnx` and `policy.onnx.data` must stay together; weights are external.
- `policy.json` binds observation/action semantics, reference tables, heading
  feedback, target mapping, clock, start and stop behavior.
- `high.json` / `low.json` are the unmodified selected reference assets.
- `bundle-manifest.json` binds assets and required runtime source versions.
- Normal web/hardware loading is deliberately rejected. Offline loading requires
  `allow_experimental_offline=True` and a sink explicitly declaring
  `offline_only=True`. Neither is a hardware deployment switch.

[Integration, verification reports and exact offline commands](../../docs/v44-study35-offline-integration.md).

Both full 3,528-row canonical episode replays pass, including inference, causal
feedback, F5 projection, actual 5 mm rear lift and scheduled return to center.
This verifies software correspondence, not real-robot stability or Pi timing.
The nominal experiment still passed only 28/30 gait qualification gates and
achieved about 1.54 cm/s. Do not present it as a successful running model.

The mapper uses nominal forward/stop commands only, up to the evaluated high
command of 0.02227272727272727 m/s. Lower values are permitted for deceleration;
their acceptance does not establish a qualified low-speed operating range.
Start targets must be calibrated center. Existing 2-degree packet and per-joint
elapsed-time rate limits remain; no simulated plant filter is double-applied.
The original hardware-tested fallback has not been replaced.
