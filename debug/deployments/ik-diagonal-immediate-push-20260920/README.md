# Diagonal IK V5 — installed

Installed on `pi5-dog.local` via `rd@192.168.1.58` at
`2026-09-21T00:53:05Z`. The source installer reported `installed: true` and
`service_restarted: true`; systemd reported `active`, `running` and
`ExecMainStatus=0`. Controller and UI sources are V5 / client 17.

The operator explicitly confirmed "Supported and disarmed—activate V5"
before the torque-releasing restart. No walk or motion check was started.
Rollback backup:
`/home/rd/drobot2/debug/deployment-backups/ik-diagonal-20260921T005305472624Z`.
See [installation receipt](applied.json); [staging receipt](staged.json)
preserves the earlier upload-only state.

## Changes

- Move the full existing 20 mm early body advance into the opposite pair's
  lift, instead of splitting it into 10 mm lift and 10 mm swing.
- Both lift and push use the same immediate ramp, with continuous positions
  and a stopped endpoint. Existing shared command interpolation and packet
  limits coordinate all four legs in the sync-write packet.
- Select support ankle feedforward before lift, allowing the old measured-leg
  correction to settle before propulsion begins.
- Carry raised feet with the body during lift; complete the original 100 mm
  world stride in swing while retaining the ankle angle limit.
- Client 17 marks `IMMEDIATE ANKLE PUSH V5`; logs expose ankle correction
  separately from desired, delivered and measured joint positions.

See [recorded evidence and controller changes](../../../hardware/test-apps/four-leg-dashboard/specs/ankle-ik-diagonal-v5-immediate-push.md).
Source inspection, analysis of two existing V4 recordings and independent
scalar geometry calculations were performed. No tests, compilation,
simulation, UI validation, hardware checks or walk were run. Physical timing
and forward travel are not verified. Calibration, directions, joint limits,
motor firmware settings and learned model weights are preserved.

## Activation command (completed successfully)

```bash
python3 /home/rd/apply-ik-diagonal-v2.py --repo /home/rd/drobot2 --archive /home/rd/ik-diagonal-immediate-push-v5.zip --sha256 411e785fcf44b143d17687e9ba9f55befe3ac8cc1993c1fbda2bba8ba1d1fb2c --apply
```

The installer created the rollback backup, replaced five sources, and
restarted the previously active dashboard. No walk was started.
Reload <http://192.168.1.58:8080/> with Ctrl+F5.

Rollback command, not executed:

```bash
python3 /home/rd/apply-ik-diagonal-v2.py --repo /home/rd/drobot2 --restore /home/rd/drobot2/debug/deployment-backups/ik-diagonal-20260921T005305472624Z --apply
```
