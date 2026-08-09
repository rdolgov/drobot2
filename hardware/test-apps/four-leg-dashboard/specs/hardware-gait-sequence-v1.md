# Hardware gait sequence V1

This is a deterministic joint-space hardware experiment built from the
confirmed wide stance. It does not use AI, foot-space inverse kinematics, or a
simulator-generated trajectory during execution.

## Starting stance

| Joint group | Target |
| --- | ---: |
| Left hip abduction, Legs 1 and 3 | -15 degrees |
| Right hip abduction, Legs 2 and 4 | +15 degrees |
| Front hip flexion, Legs 1 and 2 | +45 degrees |
| Rear hip flexion, Legs 3 and 4 | -45 degrees |
| Front knees, Legs 1 and 2 | -45 degrees |
| Rear knees, Legs 3 and 4 | +45 degrees |

## Sequence

In this sequence, **hip** means the hip-flexion joint, not hip abduction.
Targets accumulate until the return-to-stance transition.

| Transition | Changed target | Other joints |
| --- | --- | --- |
| 1 | Leg 1 hip flexion to +90 degrees | Hold stance |
| 2 | Leg 2 hip flexion to +70 degrees | Leg 1 remains at +90 degrees |
| 3 | Leg 4 knee to +20 degrees | Prior targets remain active |
| 4 | Return all joints to stance | Smooth simultaneous return |

The four transitions run twice. With the tracked `period_s = 20`, each
transition receives 2.5 seconds and uses smoothstep interpolation. The hardware
profiles use servo speed 700 and a 90% torque cap. The controller ramp is 90
degrees/s. Configured joint limits, tightened telemetry warnings, the 1.5-second
browser heartbeat, and **STOP + DISARM** remain active.

Live telemetry warnings remain visible but do not interrupt an active command,
block gait start, or disarm motors. The tracked display thresholds are 11.0 V
low voltage, 12.6 V high voltage, 0.3 V voltage spread, 55 C temperature, and
2500 mA diagnostic current per leg. Actual telemetry read or motion exceptions,
the browser heartbeat watchdog, and explicit stop/disarm commands still disarm.

This is a supported commissioning sequence, not a validated walking gait. The
current implementation intentionally records the requested joint behavior
before adding foot-clearance or body-translation logic.
