# Diagonal IK V4 — installed

Installed on `pi5-dog.local` via `rd@192.168.1.58` at
`2026-09-21T00:40:01Z` after the operator replied "Ready" to the supported
and disarmed restart prompt. The installer reported `installed: true` and
`service_restarted: true`; systemd reported `active`, `running`, and
`ExecMainStatus=0`. No walk or hardware motion check was started.

Rollback backup:
`/home/rd/drobot2/debug/deployment-backups/ik-diagonal-20260921T004001177589Z`.
See [installation receipt](applied.json); [staging receipt](staged.json)
retains the earlier upload-only state.

## Changes installed

- 4-second diagonal reference cycle instead of 8 seconds; encoder waits and
  command limits can add time, so actual speed is not guaranteed to double.
- Start opposite-pair lift and the planted-ankle push on the same phase and
  smooth progress: 10 mm advance during lift, then 10 mm during swing.
- Shared motor-command interpolation fraction and the existing sync-write
  packet keep the lifting pair and support ankles together when limited.
- Hold reference progress if prior commands have not been delivered within
  0.3 degrees, with a two-second delivery timeout and existing encoder gates.
- Permit 200 degrees/s for diagonal walking, with a 4-degree packet ceiling
  and effective 60 Hz rate limit of 3.333 degrees per packet. Stance setup
  keeps 100 degrees/s and a 2-degree packet ceiling; abduction keeps 60 degrees/s.
- Account for raised-foot lift in diagonal reach bounds to avoid an unwanted
  body dip. Recompute leveling reference immediately on a support-role change.
- Client 16 marks the button `2× PACE` and reports synchronized motion and
  delivery waits. Recorded samples include command fraction and delivery wait.

Firmware power/speed/acceleration settings, calibration, servo directions,
joint travel, tilt guards and endpoint tracking requirements are preserved.
No learned model is replaced. No walk is sent by installation.

Only source inspection and independent scalar geometry/timing calculations
were performed. Tests, compilation, simulation, UI validation and hardware
checks were not run. See [controller description](../../../hardware/test-apps/four-leg-dashboard/specs/ankle-ik-diagonal-v4-sync-2x.md).

## Installation command

This command completed successfully:

```bash
python3 /home/rd/apply-ik-diagonal-v2.py --repo /home/rd/drobot2 --archive /home/rd/ik-diagonal-sync-2x-v4.zip --sha256 65bb831e5ddc9aa79d6dd4f30f208c684ade90087d777c4c6bfb9376e55ac69c --apply
```

The installer created the rollback backup, replaced the five files, and
restarted the active dashboard. Restart releases torque. Calibration files
and model weights were preserved.

Rollback command, prepared but not executed:

```bash
python3 /home/rd/apply-ik-diagonal-v2.py --repo /home/rd/drobot2 --restore /home/rd/drobot2/debug/deployment-backups/ik-diagonal-20260921T004001177589Z --apply
```

Reload <http://pi5-dog.local:8080/> or
<http://192.168.1.58:8080/>. The diagonal start still requires disarmed motors
and prepares its own stance. Starting the next trial remains an operator action.
