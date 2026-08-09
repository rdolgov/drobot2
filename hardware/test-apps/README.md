# Hardware test applications

The hardware test applications are peer packages:

- [`one-leg-testbed/`](one-leg-testbed/README.md) — guarded ST3215 setup,
  telemetry, bounded movement, and single-leg browser control.
- [`four-leg-dashboard/`](four-leg-dashboard/README.md) — twelve-servo
  commissioning dashboard and deterministic crawl controls.

The current wide mirrored 45-degree commissioning gait, hip-abduction stance,
Isaac boundary, and supported hardware test ladder are documented in the
[`crawl walk V4 spec`](four-leg-dashboard/specs/crawl-walk-v4.md). V2 and V3
remain alongside it as historical design evidence.

Both applications read shared physical-robot state from
[`../robot-runtime/`](../robot-runtime/README.md).

Install both into the one-leg testbed's shared virtual environment from the
repository root:

```powershell
.\hardware\test-apps\install-test-apps.ps1
```
