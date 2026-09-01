"""Export the selected bounded-Beta walking actor without Isaac dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

V28_TRAINING_PROFILE = (
    "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload"
)
V28_TRAINING_TASK = (
    "Drobot-Commanded-Walk-Forward-Biased-Cycle-Gated-Four-Leg-"
    "Straight-Crawl-External-Rear-Payload-Direct"
)
V29_TRAINING_PROFILE = (
    "schedule-matched-support-straight-crawl-external-rear-payload"
)
V29_TRAINING_TASK = (
    "Drobot-Commanded-Walk-Schedule-Matched-Support-Straight-Crawl-"
    "External-Rear-Payload-Direct"
)
V30_TRAINING_PROFILE = (
    "symmetry-gated-robust-straight-crawl-external-rear-payload"
)
V30_TRAINING_TASK = (
    "Drobot-Commanded-Walk-Symmetry-Gated-Robust-Straight-Crawl-"
    "External-Rear-Payload-Direct"
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
    parser.add_argument(
        "--residual-action-scales",
        type=float,
        nargs=12,
        metavar="SCALE",
        default=None,
        help=(
            "Optional per-action residual scales in --action-order order; the "
            "scalar residual scale remains the backward-compatible fallback."
        ),
    )
    parser.add_argument("--reference-sample-count", type=int, default=2048)
    parser.add_argument("--reference-start-ramp-s", type=float, default=0.0)
    parser.add_argument("--reference-stride-m", type=float, default=0.050)
    parser.add_argument("--reference-lift-m", type=float, default=0.016)
    parser.add_argument("--reference-weight-shift-forward-m", type=float, default=0.006)
    parser.add_argument(
        "--reference-rear-weight-shift-forward-m",
        type=float,
        default=None,
    )
    parser.add_argument("--reference-weight-shift-lateral-m", type=float, default=0.0)
    parser.add_argument(
        "--reference-translate-lateral-weight-shift",
        action="store_true",
    )
    parser.add_argument("--reference-stance-fore-aft-m", type=float, default=0.080)
    parser.add_argument("--reference-stance-down-m", type=float, default=0.329341447)
    parser.add_argument(
        "--reference-stance-center-offset-m",
        type=float,
        default=0.0,
        help=(
            "Independent fore/aft translation of the complete stance reference. "
            "Positive values move every foot forward relative to the body."
        ),
    )
    parser.add_argument("--reference-smooth-support-push", action="store_true")
    parser.add_argument(
        "--reference-contact-transition-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of each analytic airborne interval reserved at both ends "
            "for gradual unloading and touchdown contact."
        ),
    )
    parser.add_argument(
        "--reference-distributed-push-phase-fractions",
        type=float,
        nargs=8,
        metavar="FRACTION",
        default=None,
        help=(
            "Optional weight-transfer/lift/swing/lower/firm-plant/weight-return/"
            "all-feet-push/settle fractions for the distributed crawl."
        ),
    )
    parser.add_argument("--reference-forward-body-pitch-rad", type=float, default=0.0)
    parser.add_argument("--neutral-sagittal-angle-rad", type=float, default=0.5239596454)
    parser.add_argument("--neutral-front-hip-rad", type=float, default=None)
    parser.add_argument("--neutral-rear-hip-rad", type=float, default=None)
    parser.add_argument("--heading-hold-enabled", action="store_true")
    parser.add_argument("--heading-hold-kp-s", type=float, default=0.0)
    parser.add_argument("--heading-hold-max-correction-rad-s", type=float, default=0.0)
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
    residual_action_scales = (
        tuple(args.residual_action_scales)
        if args.residual_action_scales is not None
        else (args.residual_action_scale,) * len(ACTION_ORDER)
    )
    if any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in residual_action_scales
    ):
        parser.error("--residual-action-scales values must be finite and in (0, 1]")
    if args.reference_sample_count <= 0:
        parser.error("--reference-sample-count must be positive")
    if args.reference_distributed_push_phase_fractions is not None:
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in args.reference_distributed_push_phase_fractions
        ):
            parser.error("reference phase fractions must be finite and positive")
        if not math.isclose(
            sum(args.reference_distributed_push_phase_fractions),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            parser.error("reference phase fractions must total one")
    if not 0.0 <= args.reference_contact_transition_fraction < 0.5:
        parser.error("reference contact transition fraction must be in [0, 0.5)")
    if (
        args.reference_stance_fore_aft_m <= 0.0
        or args.reference_stance_down_m <= 0.0
    ):
        parser.error("reference stance dimensions must be positive")
    if (
        not math.isfinite(args.reference_stance_center_offset_m)
        or abs(args.reference_stance_center_offset_m) > 0.030
    ):
        parser.error("reference stance center offset must be finite and within +/-0.030 m")
    if not 0.0 <= args.neutral_sagittal_angle_rad < 0.5 * math.pi:
        parser.error("--neutral-sagittal-angle-rad must be in [0, pi/2)")
    neutral_front_hip_rad = (
        args.neutral_front_hip_rad
        if args.neutral_front_hip_rad is not None
        else args.neutral_sagittal_angle_rad
    )
    neutral_rear_hip_rad = (
        args.neutral_rear_hip_rad
        if args.neutral_rear_hip_rad is not None
        else -args.neutral_sagittal_angle_rad
    )
    if any(
        not math.isfinite(value) or abs(value) >= 0.5 * math.pi
        for value in (neutral_front_hip_rad, neutral_rear_hip_rad)
    ):
        parser.error("explicit neutral hip angles must be finite and within +/-pi/2")
    if args.heading_hold_enabled and (
        args.heading_hold_kp_s <= 0.0
        or args.heading_hold_max_correction_rad_s <= 0.0
    ):
        parser.error("enabled heading hold requires positive gain and correction limit")
    if (
        args.action_mode == "gait_residual"
        and args.gait_pattern != "distributed_support_crawl"
    ):
        parser.error("gait-residual export requires distributed_support_crawl")

    if args.training_profile in (
        V28_TRAINING_PROFILE,
        V29_TRAINING_PROFILE,
        V30_TRAINING_PROFILE,
    ):
        is_v29 = args.training_profile == V29_TRAINING_PROFILE
        is_v30 = args.training_profile == V30_TRAINING_PROFILE
        is_schedule_matched = is_v29 or is_v30
        profile_label = "V30" if is_v30 else "V29" if is_v29 else "V28"
        # Export metadata is also executable controller configuration.  Refuse
        # an export whose analytic gait differs silently from training; the
        # generic CLI defaults describe a much faster legacy trot. V29 also
        # changes the phase layout and contact-transition contract, which must
        # never be lost between Isaac and the Raspberry Pi runtime.
        mismatches: list[str] = []

        def require_equal(name: str, actual: object, expected: object) -> None:
            if actual != expected:
                mismatches.append(f"{name}={actual!r} (expected {expected!r})")

        def require_close(name: str, actual: float, expected: float) -> None:
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-7):
                mismatches.append(f"{name}={actual!r} (expected {expected!r})")

        expected_training_task = (
            V30_TRAINING_TASK
            if is_v30
            else V29_TRAINING_TASK
            if is_v29
            else V28_TRAINING_TASK
        )
        require_equal("training_task", args.training_task, expected_training_task)
        require_equal("gait_clock_mode", args.gait_clock_mode, "speed_scaled")
        require_equal("gait_pattern", args.gait_pattern, "distributed_support_crawl")
        require_equal("action_mode", args.action_mode, "gait_residual")
        require_equal("reference_smooth_support_push", args.reference_smooth_support_push, True)
        require_equal(
            "reference_translate_lateral_weight_shift",
            args.reference_translate_lateral_weight_shift,
            True,
        )
        require_equal("heading_hold_enabled", args.heading_hold_enabled, True)
        require_close("gait_standstill_deadband_m_s", args.gait_standstill_deadband_m_s, 0.002)
        require_close("gait_speed_min_m_s", args.gait_speed_min_m_s, 0.005)
        require_close(
            "gait_speed_max_m_s",
            args.gait_speed_max_m_s,
            0.039 if is_v30 else 0.037 if is_v29 else 0.045,
        )
        require_close("gait_frequency_min_hz", args.gait_frequency_min_hz, 0.12)
        require_close(
            "gait_frequency_max_hz",
            args.gait_frequency_max_hz,
            0.85 if is_v30 else 0.80,
        )
        require_close("gait_duty_factor", args.gait_duty_factor, 0.8625)
        require_close("gait_stride_scale_min", args.gait_stride_scale_min, 1.0)
        require_close("residual_action_scale", args.residual_action_scale, 0.05)
        expected_residual_scales = (
            (0.10,) * 4
            + ((0.12,) * 4 if is_schedule_matched else (0.04,) * 4)
            + (0.15,) * 4
        )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7)
            for actual, expected in zip(residual_action_scales, expected_residual_scales)
        ):
            mismatches.append(
                "residual_action_scales do not match the "
                f"{profile_label} "
                f"0.10/{0.12 if is_schedule_matched else 0.04:.2f}/0.15 contract"
            )
        require_close("reference_start_ramp_s", args.reference_start_ramp_s, 1.5)
        require_close("reference_stride_m", args.reference_stride_m, 0.046)
        require_close("reference_lift_m", args.reference_lift_m, 0.024)
        require_close(
            "reference_weight_shift_forward_m",
            args.reference_weight_shift_forward_m,
            0.015 if is_schedule_matched else 0.008,
        )
        if args.reference_rear_weight_shift_forward_m is None:
            expected_rear_shift = 0.025 if is_v30 else 0.020 if is_v29 else 0.010
            mismatches.append(
                "reference_rear_weight_shift_forward_m is missing "
                f"(expected {expected_rear_shift:.3f})"
            )
        else:
            require_close(
                "reference_rear_weight_shift_forward_m",
                args.reference_rear_weight_shift_forward_m,
                0.025 if is_v30 else 0.020 if is_v29 else 0.010,
            )
        require_close(
            "reference_weight_shift_lateral_m",
            args.reference_weight_shift_lateral_m,
            -0.012 if is_v30 else 0.006,
        )
        require_close("reference_stance_fore_aft_m", args.reference_stance_fore_aft_m, 0.092)
        require_close("reference_stance_down_m", args.reference_stance_down_m, 0.3216749408355507)
        require_close(
            "reference_stance_center_offset_m",
            args.reference_stance_center_offset_m,
            0.0,
        )
        require_close(
            "reference_contact_transition_fraction",
            args.reference_contact_transition_fraction,
            0.08 if is_schedule_matched else 0.0,
        )
        if is_schedule_matched:
            expected_phase_fractions = (
                0.20,
                0.15,
                0.20,
                0.20,
                0.10,
                0.08,
                0.02,
                0.05,
            )
            if args.reference_distributed_push_phase_fractions is None or any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7)
                for actual, expected in zip(
                    args.reference_distributed_push_phase_fractions,
                    expected_phase_fractions,
                )
            ):
                mismatches.append(
                    "reference_distributed_push_phase_fractions do not match "
                    f"the {profile_label} schedule"
                )
        require_close("reference_forward_body_pitch_rad", args.reference_forward_body_pitch_rad, math.radians(2.0))
        require_close("neutral_front_hip_rad", neutral_front_hip_rad, 0.6713465566913456)
        require_close("neutral_rear_hip_rad", neutral_rear_hip_rad, -0.5561956608266132)
        require_close("heading_hold_kp_s", args.heading_hold_kp_s, 1.5)
        require_close("heading_hold_max_correction_rad_s", args.heading_hold_max_correction_rad_s, 0.20)
        expected_phases = (0.07, 0.32, 0.57, 0.82)
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7)
            for actual, expected in zip(args.gait_phase_offsets, expected_phases)
        ):
            mismatches.append(
                f"gait_phase_offsets do not match the {profile_label} crawl order"
            )
        require_close("target_velocity_limit_rad_s", args.target_velocity_limit_rad_s, math.radians(240.0))
        require_close("max_target_step_deg", args.max_target_step_deg, 4.0)
        require_close("forward_speed_min_m_s", args.forward_speed_min_m_s, 0.005)
        require_close(
            "forward_speed_max_m_s",
            args.forward_speed_max_m_s,
            0.039 if is_v30 else 0.037 if is_v29 else 0.045,
        )
        require_close("recommended_forward_speed_m_s", args.recommended_forward_speed_m_s, 0.005)
        if mismatches:
            parser.error(
                f"{profile_label} export contract mismatch; pass the complete "
                "trained profile:\n  - "
                + "\n  - ".join(mismatches)
            )

    reference_joint_position_rad: list[list[float]] = []
    stance_forward_bias_m = 0.0
    if args.action_mode == "gait_residual":
        for sample_index in range(args.reference_sample_count):
            pose, _state = distributed_push_crawl_by_name(
                sample_index / args.reference_sample_count,
                period_s=1.0,
                stride_m=args.reference_stride_m,
                lift_m=args.reference_lift_m,
                support_extension_m=0.0,
                weight_shift_forward_m=args.reference_weight_shift_forward_m,
                weight_shift_lateral_m=args.reference_weight_shift_lateral_m,
                rear_weight_shift_forward_m=(
                    args.reference_rear_weight_shift_forward_m
                ),
                translate_lateral_weight_shift=(
                    args.reference_translate_lateral_weight_shift
                ),
                forward_body_pitch_rad=args.reference_forward_body_pitch_rad,
                down_m=args.reference_stance_down_m,
                fore_aft_m=args.reference_stance_fore_aft_m,
                abduction_deg=0.0,
                smooth_support_push=args.reference_smooth_support_push,
                phase_fractions=(
                    args.reference_distributed_push_phase_fractions
                ),
                stance_center_offset_m=args.reference_stance_center_offset_m,
                contact_transition_fraction=(
                    args.reference_contact_transition_fraction
                ),
            )
            reference_joint_position_rad.append(
                [float(pose[name]) for name in ACTION_ORDER]
            )
            stance_forward_bias_m = float(_state["stance_forward_bias_m"])
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
        "heading_hold": {
            "enabled": args.heading_hold_enabled,
            "mode": (
                "relative_yaw_feedback" if args.heading_hold_enabled else "disabled"
            ),
            "kp_s": args.heading_hold_kp_s,
            "max_correction_rad_s": args.heading_hold_max_correction_rad_s,
        },
        "forward_command_range_m_s": {
            "min": args.forward_speed_min_m_s,
            "max": args.forward_speed_max_m_s,
            "recommended": args.recommended_forward_speed_m_s,
        },
        "action_contract": {
            "mode": args.action_mode,
            # Older runtimes understand only the scalar field.  The smallest
            # configured value is a fail-safe fallback for a non-uniform policy;
            # vector-aware runtimes use the ordered field below.
            "residual_scale": min(residual_action_scales),
            "residual_scale_by_action": list(residual_action_scales),
        },
        "observation_count": 50,
        "action_count": 12,
        "joint_target_contract": {
            "neutral_joint_position_rad": [
                0.0,
                0.0,
                0.0,
                0.0,
                neutral_front_hip_rad,
                neutral_rear_hip_rad,
                neutral_front_hip_rad,
                neutral_rear_hip_rad,
                -neutral_front_hip_rad,
                -neutral_rear_hip_rad,
                -neutral_front_hip_rad,
                -neutral_rear_hip_rad,
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
            "mode": (
                "smooth_distributed_push"
                if args.reference_smooth_support_push
                else "distributed_push"
            ),
            "smooth_support_push": args.reference_smooth_support_push,
            "distributed_push_phase_fractions": (
                list(args.reference_distributed_push_phase_fractions)
                if args.reference_distributed_push_phase_fractions is not None
                else None
            ),
            "contact_transition_fraction": (
                args.reference_contact_transition_fraction
            ),
            "start_ramp_s": args.reference_start_ramp_s,
            "sample_count": args.reference_sample_count,
            "stride_m": args.reference_stride_m,
            "lift_m": args.reference_lift_m,
            "weight_shift_forward_m": args.reference_weight_shift_forward_m,
            "rear_weight_shift_forward_m": (
                args.reference_rear_weight_shift_forward_m
            ),
            "weight_shift_lateral_m": args.reference_weight_shift_lateral_m,
            "translate_lateral_weight_shift": (
                args.reference_translate_lateral_weight_shift
            ),
            "stance_fore_aft_m": args.reference_stance_fore_aft_m,
            "stance_down_m": args.reference_stance_down_m,
            "stance_center_offset_m": args.reference_stance_center_offset_m,
            "forward_body_pitch_rad": args.reference_forward_body_pitch_rad,
            "stance_forward_bias_m": stance_forward_bias_m,
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
