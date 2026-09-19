"""Extract frozen actor arrays for the Pi; no inference, training or compilation.

Run with the training Python (torch/numpy required). Source checkpoint identity
is pinned to the selected flat-ground V2.3 transfer actor, seed 2311 / 2999.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "simulation/isaac/models/ankle-v23-transfer/seed2311-model_2999.pt"
digest = hashlib.sha256(source.read_bytes()).hexdigest()
if digest != "d0b1e4e31fe39919dcedefd2e816dc48b34aeadfd366ccbaf9b9d4c32f7d069d":
    raise ValueError("This packager only supports the selected seed-2311 checkpoint")
checkpoint = torch.load(source, map_location="cpu", weights_only=True)
actor = checkpoint["actor_state_dict"]
output = ROOT / "onboard/models/ankle-v23-transfer-seed2311"
output.mkdir(parents=True, exist_ok=True)
weights = output / "policy.npz"
np.savez(weights, **{key: tensor.detach().cpu().numpy() for key, tensor in actor.items()})
legs = ("front_left", "rear_left", "front_right", "rear_right")
payload = {
    "schema": "drobot.ankle-v23-transfer.numpy.v1",
    "source_checkpoint": str(source.relative_to(ROOT)).replace("\\", "/"),
    "source_checkpoint_sha256": digest,
    "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    "iteration": int(checkpoint["iter"]), "seed": 2311,
    "observation_size": 82, "action_size": 16,
    "action_names": [f"{leg}_{kind}" for kind in
                     ("hip_abduction", "hip_flexion", "knee", "ankle_pitch") for leg in legs],
    "servo_ids": [1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12, 13, 15, 14, 16],
    "observation_order": [
        "command_xyz_times_10[3]", "clock_sin_cos[2]", "imu_angular_velocity_body[3]",
        "projected_gravity_body[3]", "imu_linear_acceleration_body_over_9_81[3]",
        "joint_position_minus_neutral[16]", "joint_velocity_times_0_25[16]",
        "previous_applied_target_normalized[16]", "filtered_action[16]", "expected_stance[4]",
    ],
    "actor": {"hidden_dims": [256, 256, 128], "activation": "elu",
              "distribution": "beta_deterministic_mean", "normalization_epsilon": .01},
    "forward_command_range_m_s": {"min": .015, "max": .021, "recommended": .018},
    "gait_period_s": .095 / (.018 * .82),
    "run_stop_contract": {"recommended_stop_ramp_s": 1., "recommended_stopped_hold_s": 1., "ramp_shape": "linear"},
    "experimental_notice": "Experimental 16-motor V2.3 walking policy. Simulation selection failed: no qualifying rear-left steps, with speed and drift errors. This hardware port has not been tested. Start with a supported short trial.",
    "verification": {"numerical_parity": "not_run", "hardware": "not_run"},
    "training_report": "simulation/docs/rl-ankle-v23-transfer.md",
}
weights.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Packaged {weights.relative_to(ROOT)} ({weights.stat().st_size} bytes); no inference run")
