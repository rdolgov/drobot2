"""Export the selected bounded-Beta walking actor without Isaac dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


class DeterministicWalkingActor(nn.Module):
    """Reproduce the RSL-RL MLP and deterministic Beta-distribution mean."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(50, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 24),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(observation).reshape(-1, 2, 12)
        alpha = torch.nn.functional.softplus(raw[:, 0, :]) + 1.0
        beta = torch.nn.functional.softplus(raw[:, 1, :]) + 1.0
        return 2.0 * alpha / (alpha + beta) - 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-task", default="")
    parser.add_argument("--training-profile", default="")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = DeterministicWalkingActor()
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 50), dtype=torch.float32)
    torch.onnx.export(
        actor,
        example,
        output_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    metadata = {
        "format": "drobot-walking-policy-v1",
        "source_checkpoint": checkpoint_path.name,
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_iteration": int(checkpoint["iter"]),
        "onnx_sha256": sha256(output_path),
        "control_hz": 60,
        "gait_period_s": 0.8,
        "observation_count": 50,
        "action_count": 12,
        "observation_order": [
            "command_forward_m_s[1]",
            "command_lateral_m_s[1]",
            "command_yaw_rad_s[1]",
            "gait_clock_sin[1]",
            "gait_clock_cos[1]",
            "imu_angular_velocity_body_rad_s[3]",
            "projected_gravity_body[3]",
            "imu_linear_acceleration_body_over_9_81[3]",
            "joint_position_error_rad[12]",
            "joint_velocity_over_4_5836625[12]",
            "previous_normalized_action[12]",
        ],
        "action_order": [
            "front_left_hip_abduction",
            "rear_left_hip_abduction",
            "front_right_hip_abduction",
            "rear_right_hip_abduction",
            "front_left_hip_flexion",
            "rear_left_hip_flexion",
            "front_right_hip_flexion",
            "rear_right_hip_flexion",
            "front_left_knee",
            "rear_left_knee",
            "front_right_knee",
            "rear_right_knee",
        ],
    }
    if args.training_task:
        metadata["training_task"] = args.training_task
    if args.training_profile:
        metadata["training_profile"] = args.training_profile
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
