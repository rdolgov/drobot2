# Diagonal IK V6 — uploaded, activation pending

Uploaded to `rd@192.168.1.58:/home/rd/ik-diagonal-measured-onset-v6.zip`.
The source-only installer reported `file_hashes_match: true` and
`installed: false` after comparing archive, replacement and prerequisite
hashes. The active controller and UI remain V5 / client 17.

V6 gives the planted-ankle push an early lead: 6 mm of scheduled forward
advance accompanies 3.815 mm of lift. The clock then holds that pose until
both planted ankle encoders show at least 0.5 degrees of movement in the
nominal push direction from their actual pre-lift positions. It resumes
without requiring a zero-speed settle. The remaining push and lift finish
together; total early advance stays 20 mm. Client 18 displays `ANKLE LEAD V6`
and an explicit low-lift waiting status. Calibration, directions, joint
limits, servo firmware settings, model weights and IMU guards are preserved.

See [V5 evidence and V6 control description](../../../hardware/test-apps/four-leg-dashboard/specs/ankle-ik-diagonal-v6-measured-onset.md).
The latest V5 recording confirms simultaneous commands but initial measured
ankle lag, plus larger IMU-driven ankle movement after lift. This update adds
an encoder-onset condition; it does not prove foot contact or body advance.

Source inspection and analysis of the existing recording were performed.
Tests, compilation, simulation, UI validation and hardware/motion checks were
not run. No walk was commanded. The operator has been asked for current
supported/disarmed readiness because the latest recorded walk ended holding
torque and activation requires a torque-releasing dashboard restart.

## Activation command (not executed)

```bash
python3 /home/rd/apply-ik-diagonal-v2.py --repo /home/rd/drobot2 --archive /home/rd/ik-diagonal-measured-onset-v6.zip --sha256 2082bf93a5e51e7185cdbc6bc108d8bad07c2cf3135aeb0802ed3554f3a44290 --apply
```

The installer creates a timestamped rollback backup and replaces five runtime
source/configuration/UI files before restarting the previously active service.
It never starts a walk. After activation, Ctrl+F5 reloads
<http://192.168.1.58:8080/>.
