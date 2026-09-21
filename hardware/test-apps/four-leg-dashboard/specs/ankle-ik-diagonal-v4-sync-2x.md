# Diagonal IK V4: twice the reference pace, synchronized lift and push

Requested September 20, 2026. Source changes are prepared without automated
tests, builds, simulation, browser validation or hardware checks. See the
release directory for deployment status:
`debug/deployments/ik-diagonal-sync-2x-20260920/`.

## Timing and trajectory

The diagonal reference cycle is 4 seconds instead of 8. Each pair's placement
takes 2 reference seconds: transfer 0.36, lift 0.36, swing 0.44, lower 0.36,
plant 0.20, final push 0.20 and settle 0.08. Existing encoder gates add time;
this is twice the requested trajectory pace, not a measured doubling of
actual travel speed or trial duration. The 100 mm stride, 35 mm lift, 20 mm
early body advance and two-cycle limit remain.

`support_lift_fraction = 0.50` assigns 10 mm to lift (previously 5 mm) and
10 mm to swing. Opposite feet lift by `35 mm * smooth(progress)` while the
planted pair advances the body reference by `10 mm * smooth(progress)`.
The same phase and quintic progress start and stop both motions. Zero initial
velocity is retained, so synchronization does not add an abrupt ankle kick.
The 20 mm advance is held through landing; the remaining 30 mm follows with
four scheduled supports. No backward reset is introduced at handover.

Diagonal reach bounds now account for the raised foot's shorter vertical
reach before choosing a common body height. Previously the bound treated
every ankle as if it were planted, introducing an unnecessary body dip as a
lifted rear foot moved aft. Independent endpoint arithmetic for the revised
lift gives front x=20 mm/down=253 mm/ankle=32.9565 degrees and rear
x=-140 mm/down=253 mm/ankle=3.6720 degrees. The retained rear ground endpoint
has reach 315.981 mm, below the 317 mm limit. The front lifted ankle has only
about 0.29 degrees of nominal margin to its 33.25-degree cap, so body-tilt
leveling can still clip and reports that condition. These are calculations,
not dynamic validation.

## Coordinated motor output

Diagonal walking allows a sagittal/ankle target rate of 200 degrees/s instead
of 100. Its packet ceiling is 4 degrees, with the 60 Hz rate ceiling limiting
the actual step to at most 3.333 degrees (or a lower shared bus limit).
Abduction retains 60 degrees/s. Initial stance preparation and stopped holds
retain 100 degrees/s and at most 2 degrees per packet. Single-leg crawl,
manual motor control and RL retain their previous effective rate limits.
Servo firmware speed, acceleration, torque limit, directions, calibration,
travel bounds and IMU/encoder fault thresholds are unchanged.

The diagonal output worker computes one fraction from the most restrictive
joint's allowed step. Every changing joint uses that fraction instead of
being clipped independently. The hardware transport sends the resulting
pose in its existing broadcast sync-write packet. Encoder quantization and
loaded servo tracking can still produce physical timing differences.

If a previously requested pose is more than 0.3 degrees ahead of any
delivered command, the reference clock holds that frame while all joints
catch up. Fresh IMU compensation alone does not count as an undelivered prior
packet. A continuous delivery hold faults after 2 wall-clock seconds. Existing
endpoint gates independently retain measured position/velocity checks and
their timeout. Missed bus ticks never cause catch-up bursts.

Ankle leveling recomputes the selected reference when support roles change,
without waiting for its normal 50 ms cache interval. Its filter and slew
bounds remain. Support ankles retain planned-angle feedforward plus IMU
compensation; swinging ankles retain measured-angle leveling.

## Dashboard and records

Client 16 labels the diagonal button `2× PACE`, shows the 4-second cycle,
synchronized lift/push and any delivery wait. Recordings use controller
`ankle_ik_v4_synchronized_push` and label
`IK diagonal 2x / synchronized ankle push`. Additive `command_fraction` and
`delivery_wait_s` fields help distinguish scheduled pace from command-limit
holds. They describe controller output, not measured foot contact or body
translation. Start still requires disarmed motors and prepares its own stance.

Deployment replaces five runtime files with a timestamped rollback backup.
The controller restart releases torque, so activation uses an operator's
current supported/disarmed confirmation. No walk is started by installation.
