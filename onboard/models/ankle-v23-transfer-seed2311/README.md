# Experimental V2.3 walking policy

Selected flat-ground checkpoint: seed 2311, iteration 2999, trained September
13, 2026. Later local ankle experiments target stairs and use other actor
inputs. This package contains the selected flat-ground actor's frozen arrays
and observation normalization statistics, extracted without inference.

**Not a qualified walking gait.** The existing simulation selection failed
speed, drift and stepping criteria; the rear-left foot had no qualifying
steps. See [training results](../../../simulation/docs/rl-ankle-v23-transfer.md).
The NumPy hardware port has not undergone numerical parity or hardware testing.

The actor uses 82 observations, 16 actions, an ELU MLP (256/256/128), and the
deterministic mean of a Beta distribution. Runtime implementation:
[`ankle_transfer.py`](../../policy-runtime/src/drobot_policy_runtime/ankle_transfer.py).
Actor ordering is joint-kind first, then FL/RL/FR/RR; servo order is
`1,7,4,10,2,8,5,11,3,9,6,12,13,15,14,16`. The original 50-input / 12-output
ONNX runtime cannot run these weights.

The port implements the trained transfer reference, measured hip/knee ankle
leveling, action filter, normalization, phase and start/stop ramps. Hardware
history uses the last host-completed bus target; finite bus timing can differ
from simulation. Body pitch is derived from projected gravity X. No heading
correction is added. Runtime joint envelopes use 95% of the URDF limits
(ankles ±33.25°), independently constrained by calibrated hardware bounds.

Default trial: 5 seconds, 0.018 m/s; allowed speeds 0.015–0.021 m/s, maximum
15 seconds. PREPARE RL STANCE first moves all 16 motors slowly to the training
pose. It does not start inference. Explicit Stop disarms; normal completion
ramps to zero over one second and holds the final stance. The package does
not execute automatically when installed or when the dashboard starts.

Repackage with the training Python:

```powershell
& C:/isaacsim/kit/python/python.exe onboard/scripts/package-ankle-transfer.py
```

This only extracts arrays from the pinned checkpoint. No tests, builds,
simulations, inference, browser validation, or hardware motion checks were
run during packaging/deployment, per the user's verification preference.
