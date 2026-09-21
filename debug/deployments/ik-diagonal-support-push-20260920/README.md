# Diagonal IK V3 support push — installed on Pi

Installed on `pi5-dog.local` (`192.168.1.58`) at 2026-09-21 00:22 UTC after
the operator confirmed the robot was supported and disarmed. All five runtime
files were replaced after the expected source/prerequisite hashes matched.
The dashboard service restarted; systemd reported `active/running`, with
`ExecMainStatus=0`. See [deployment record](applied.json). The earlier
[staging record](staged.json) is retained as history.

Rollback snapshot:
`/home/rd/drobot2/debug/deployment-backups/ik-diagonal-20260921T002202194594Z`.
Calibration and learned model files were preserved. No walk was commanded.

## Prepared behavior

- Keep the 100 mm stride, 35 mm lift, alternating diagonal pairs and two cycles.
- Request 5 mm forward body advance during lift and 15 mm during swing using
  the planted hips, knees and ankles; hold that advance through landing.
- Complete the remaining 30 mm with all feet planted. No backward reset of
  the early push is added at handover.
- Use planned support hip/knee angles with IMU pitch compensation for the
  propulsive ankles, so tracking lag does not cancel their feedforward.
- Keep measured-angle leveling on swing feet, joint and motion limits,
  encoder settling gates, tilt limits, calibration and servo directions.

Client 15 labels the button `DIAGONAL 100 MM + ANKLE PUSH / 2 CYCLES`, with the planned
stance push and scheduled ankle-drive corners. Trials retain actual,
desired and delivered joint positions plus the new push fields.

See the [controller description](../../../hardware/test-apps/four-leg-dashboard/specs/ankle-ik-diagonal-v3-support-push.md).
No tests, compilers, simulation, UI validation, hardware checks or walk were
run. Independent geometry arithmetic and source inspection are the only
engineering review performed; real forward travel and balance are unverified.

## Applied command

The following command completed successfully on the Pi:

```bash
python3 /home/rd/apply-ik-diagonal-v2.py --repo /home/rd/drobot2 --archive /home/rd/ik-diagonal-support-push-v3.zip --sha256 16f29740933498da733156c6b538fa225dd53b71643364b7a4c0124037a472f8 --apply
```

The existing source-only installer supports this five-file bundle. It creates
a timestamped rollback backup and restarts the dashboard if it was active.
It does not start walking. Its expected hashes include the previously
installed static UI help update and the newer Pi calibrations for legs 1,
3 and 4, which are preserved in place. Any intervening source change aborts
installation before the service is stopped.

After installation, reload <http://pi5-dog.local:8080/> or
<http://192.168.1.58:8080/>. Diagonal start still requires disarmed motors
and prepares its own stance. Actual walking remains an operator action.

To restore the preceding source snapshot while supported and disarmed:

```bash
python3 /home/rd/apply-ik-diagonal-v2.py --repo /home/rd/drobot2 --restore /home/rd/drobot2/debug/deployment-backups/ik-diagonal-20260921T002202194594Z --apply
```
