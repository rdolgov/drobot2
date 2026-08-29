"""Export the selected bounded-Beta walking actor without Isaac dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac._quadruped_runtime import distributed_push_crawl_by_name

ACTION_ORDER = (
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
)


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
    parser.add_argument(
        "--gait-clock-mode",
        choices=("fixed", "speed_scaled"),
        default="fixed",
    )
    parser.add_argument("--gait-period-s", type=float, default=0.8)
    parser.add_argument("--gait-standstill-deadband-m-s", type=float, default=0.0)
    parser.add_argument("--gait-speed-min-m-s", type=float, default=0.04)
    parser.add_argument("--gait-speed-max-m-s", type=float, default=0.10)
    parser.add_argument("--gait-frequency-min-hz", type=float, default=1.25)
    parser.add_argument("--gait-frequency-max-hz", type=float, default=1.25)
    parser.add_argument("--gait-stride-scale-min", type=float, default=1.0)
    parser.add_argument(
        "--gait-pattern",
        choices=("diagonal_trot", "sequential_crawl", "distributed_support_crawl"),
        default="diagonal_trot",
    )
    parser.add_argument("--gait-duty-factor", type=float, default=0.65)
    parser.add_argument(
        "--action-mode", choices=("direct", "gait_residual"), default="direct"
    )
    parser.add_argument("--residual-action-scale", type=float, default=1.0)
    parser.add_argument("--reference-sample-count", type=int, default=2048)
    parser.add_argument("--reference-start-ramp-s", type=float, default=0.0)
    parser.add_argument("--reference-stride-m", type=float, default=0.050)
    parser.add_argument("--reference-lift-m", type=float, default=0.016)
    parser.add_argument("--reference-weight-shift-forward-m", type=float, default=0.006)
    parser.add_argument(
        "--gait-phase-offsets",
        type=float,
        nargs=4,
        metavar=("FL", "RL", "FR", "RR"),
        default=(0.0, 0.5, 0.5, 0.0),
    )
    parser.add_argument("--target-velocity-limit-rad-s", type=float, default=4.5836625)
    parser.add_argument("--max-target-step-deg", type=float, default=2.0)
    parser.add_argument("--startup-ramp-rate-deg-s", type=float, default=45.0)
    parser.add_argument("--startup-settle-s", type=float, default=0.5)
    parser.add_argument("--startup-position-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--forward-speed-min-m-s", type=float, default=0.04)
    parser.add_argument("--forward-speed-max-m-s", type=float, default=0.10)
    parser.add_argument("--recommended-forward-speed-m-s", type=float, default=0.04)
    args = parser.parse_args()

    if args.gait_period_s <= 0.0:
        parser.error("--gait-period-s must be positive")
    if not 0.0 < args.gait_duty_factor < 1.0:
        parser.error("--gait-duty-factor must be between zero and one")
    if args.target_velocity_limit_rad_s <= 0.0 or args.max_target_step_deg <= 0.0:
        parser.error("target velocity and step limits must be positive")
    if not 0.0 <= args.gait_stride_scale_min <= 1.0:
        parser.error("--gait-stride-scale-min must be in [0, 1]")
    if not 0.0 < args.residual_action_scale <= 1.0:
        parser.error("--residual-action-scale must be in (0, 1]")
    if args.reference_sample_count <= 0:
        parser.error("--reference-sample-count must be positive")
    if args.action_mode == "gait_residual" and args.gait_pattern != "distributed_support_crawl":
        parser.error("gait-residual export requires distributed_support_crawl")

    reference_joint_position_rad: list[list[float]] = []
    if args.action_mode == "gait_residual":
        for sample_index in range(args.reference_sample_count):
            pose, _state = distributed_push_crawl_by_name(
                sample_index / args.reference_sample_count,
                period_s=1.0,
                stride_m=args.reference_stride_m,
                lift_m=args.reference_lift_m,
                support_extension_m=0.0,
                weight_shift_forward_m=args.reference_weight_shift_forward_m,
                weight_shift_lateral_m=0.0,
                down_m=0.329341447,
                fore_aft_m=0.080,
                abduction_deg=0.0,
            )
            reference_joint_position_rad.append(
                [float(pose[name]) for name in ACTION_ORDER]
            )
    if not (
        0.0
        <= args.forward_speed_min_m_s
        <= args.recommended_forward_speed_m_s
        <= args.forward_speed_max_m_s
    ):
        parser.error(
            "forward speed values must satisfy 0 <= min <= recommended <= max"
        )
    if args.gait_clock_mode == "speed_scaled":
        if not (
            0.0
            <= args.gait_standstill_deadband_m_s
            <= args.gait_speed_min_m_s
            < args.gait_speed_max_m_s
        ):
            parser.error(
                "speed-scaled gait values must satisfy "
                "0 <= deadband <= speed min < speed max"
            )
        if not (
            0.0
            < args.gait_frequency_min_hz
            <= args.gait_frequency_max_hz
        ):
            parser.error(
                "speed-scaled gait frequencies must satisfy 0 < min <= max"
            )

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
        "format": "drobot-walking-policy-v2",
        "source_checkpoint": checkpoint_path.name,
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_iteration": int(checkpoint["iter"]),
        "onnx_sha256": sha256(output_path),
        "control_hz": 60,
        "gait_period_s": args.gait_period_s,
        "gait_clock": {
            "mode": args.gait_clock_mode,
            "period_s": args.gait_period_s,
            "standstill_deadband_m_s": args.gait_standstill_deadband_m_s,
            "speed_min_m_s": args.gait_speed_min_m_s,
            "speed_max_m_s": args.gait_speed_max_m_s,
            "frequency_min_hz": args.gait_frequency_min_hz,
            "frequency_max_hz": args.gait_frequency_max_hz,
            "stride_scale_min": args.gait_stride_scale_min,
        },
        "gait_contract": {
            "pattern": args.gait_pattern,
            "duty_factor": args.gait_duty_factor,
            "phase_offsets_fl_rl_fr_rr": args.gait_phase_offsets,
        },
        "forward_command_range_m_s": {
            "min": args.forward_speed_min_m_s,
            "max": args.forward_speed_max_m_s,
            "recommended": args.recommended_forward_speed_m_s,
        },
        "action_contract": {
            "mode": args.action_mode,
            "residual_scale": args.residual_action_scale,
        },
        "observation_count": 50,
        "action_count": 12,
        "joint_target_contract": {
            "neutral_joint_position_rad": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.5239596454,
                -0.5239596454,
                0.5239596454,
                -0.5239596454,
                -0.5239596454,
                0.5239596454,
                -0.5239596454,
                0.5239596454,
            ],
            "action_scale_rad": [
                0.12,
                0.12,
                0.12,
                0.12,
                0.30,
                0.30,
                0.30,
                0.30,
                0.40,
                0.40,
                0.40,
                0.40,
            ],
            "target_velocity_limit_rad_s": args.target_velocity_limit_rad_s,
            "max_target_step_rad": args.max_target_step_deg * 3.141592653589793 / 180.0,
        },
        "startup": {
            "mode": "prepared_neutral",
            "ramp_rate_deg_s": args.startup_ramp_rate_deg_s,
            "settle_s": args.startup_settle_s,
            "position_tolerance_deg": args.startup_position_tolerance_deg,
        },
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
        "action_order": list(ACTION_ORDER),
    }
    if reference_joint_position_rad:
        metadata["gait_reference"] = {
            "mode": "distributed_push",
            "start_ramp_s": args.reference_start_ramp_s,
            "sample_count": args.reference_sample_count,
            "stride_m": args.reference_stride_m,
            "lift_m": args.reference_lift_m,
            "weight_shift_forward_m": args.reference_weight_shift_forward_m,
            "joint_position_rad": reference_joint_position_rad,
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
