# Hardware test applications

The hardware test applications are peer packages:

- [`one-leg-testbed/`](one-leg-testbed/README.md) — guarded ST3215 setup,
  telemetry, bounded movement, and single-leg browser control.
- [`four-leg-dashboard/`](four-leg-dashboard/README.md) — twelve-servo
  commissioning dashboard and deterministic crawl controls.

The current walking controller design, analytic leg IK, rejected alternatives,
Isaac results, and supported hardware test ladder are documented in the
[`crawl walk V2 spec`](four-leg-dashboard/specs/crawl-walk-v2.md).

Both applications read shared physical-robot state from
[`../robot-runtime/`](../robot-runtime/README.md).

Install both into the one-leg testbed's shared virtual environment from the
repository root:

```powershell
.\hardware\test-apps\install-test-apps.ps1
```
