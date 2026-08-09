# Hardware test applications

The hardware test applications are peer packages:

- [`one-leg-testbed/`](one-leg-testbed/README.md) — guarded ST3215 setup,
  telemetry, bounded movement, and single-leg browser control.
- [`four-leg-dashboard/`](four-leg-dashboard/README.md) — twelve-servo
  commissioning dashboard and deterministic crawl controls.

Both applications read shared physical-robot state from
[`../robot-runtime/`](../robot-runtime/README.md).

Install both into the one-leg testbed's shared virtual environment from the
repository root:

```powershell
.\hardware\test-apps\install-test-apps.ps1
```
