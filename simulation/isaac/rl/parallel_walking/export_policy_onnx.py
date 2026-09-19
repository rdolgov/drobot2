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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulation.isaac._quadruped_runtime import distributed_push_crawl_by_name
from v44_training_contract import (
    PROFILE as V44_TRAINING_PROFILE,
    TASK as V44_TRAINING_TASK,
    validate_export_arguments as validate_v44_export_arguments,
)

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
V31_TRAINING_PROFILE = "fast-run-stop-external-rear-payload"
V31_TRAINING_TASK = (
    "Drobot-Commanded-Run-Fast-Stop-External-Rear-Payload-Direct"
)
V32_TRAINING_PROFILE = "hardware-matched-run-stop-external-rear-payload"
V32_TRAINING_TASK = (
    "Drobot-Commanded-Run-Hardware-Matched-Stop-External-Rear-Payload-Direct"
)
V33_TRAINING_PROFILE = "applied-target-run-stop-external-rear-payload"
V33_TRAINING_TASK = (
    "Drobot-Commanded-Run-Applied-Target-History-Stop-External-Rear-"
    "Payload-Direct"
)
V34_TRAINING_PROFILE = "integrated-target-velocity-run-stop-external-rear-payload"
V34_TRAINING_TASK = (
    "Drobot-Commanded-Run-Integrated-Target-Velocity-Stop-External-Rear-Payload"
)
V35_TRAINING_PROFILE = "command-scaled-high-speed-run-stop-external-rear-payload"
V35_TRAINING_TASK = (
    "Drobot-Commanded-Run-Command-Scaled-High-Speed-Stop-External-Rear-Payload"
)
V36_TRAINING_PROFILE = "flat-sole-high-speed-run-stop-external-rear-payload"
V36_TRAINING_TASK = (
    "Drobot-Commanded-Run-Flat-Sole-High-Speed-Stop-External-Rear-Payload"
)
V38_TRAINING_PROFILE = (
    "straight-low-abduction-long-run-stop-external-rear-payload"
)
V38_TRAINING_TASK = (
    "Drobot-Commanded-Run-Straight-Low-Abduction-Long-Stop-External-Rear-Payload"
)
V43_TRAINING_PROFILE = (
    "foot-warp-trot-residual-run-stop-external-rear-payload"
)
V43_TRAINING_TASK = (
    "Drobot-Commanded-Run-Foot-Warp-Trot-Residual-Stop-External-Rear-Payload"
)
V43_REFERENCE_HIGH_SHA256 = (
    "1ff1494a172124f1c8943e3bb99b8be75b41442f13187e6c8db3c93c8ebf5f69"
)
V43_REFERENCE_LOW_SHA256 = (
    "7ac46755e3554f2e5781f5c5a6fd2af5bbfbe96880ec374a5683eeef01274fea"
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


def load_phase_reference(
    path: Path,
    parser: argparse.ArgumentParser,
) -> tuple[list[list[float]], dict[str, object]]:
    """Load and strictly validate a generated joint-space phase reference."""

    resolved = path.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"could not load phase reference {path}: {exc}")
    if not isinstance(payload, dict):
        parser.error(f"phase reference {path} must contain a JSON object")
    if payload.get("schema") != "drobot.phase-joint-reference.v1":
        parser.error(
            f"phase reference {path} has unsupported schema "
            f"{payload.get('schema')!r}"
        )
    if tuple(payload.get("joint_order", ())) != ACTION_ORDER:
        parser.error(f"phase reference {path} joint order does not match the actor")
    rows = payload.get("joint_position_rad")
    if not isinstance(rows, list) or not rows:
        parser.error(f"phase reference {path} must contain joint-position rows")
    normalized_rows: list[list[float]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(ACTION_ORDER):
            parser.error(
                f"phase reference {path} row {index} must contain 12 values"
            )
        try:
            normalized_row = [float(value) for value in row]
        except (TypeError, ValueError):
            parser.error(
                f"phase reference {path} row {index} contains a non-number"
            )
        if not all(math.isfinite(value) for value in normalized_row):
            parser.error(
                f"phase reference {path} row {index} contains a non-finite value"
            )
        normalized_rows.append(normalized_row)
    if payload.get("sample_count") != len(normalized_rows):
        parser.error(
            f"phase reference {path} sample_count does not match its rows"
        )
    return normalized_rows, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-task", default="")
    parser.add_argument("--training-profile", default="")
    parser.add_argument(
        "--training-provenance", type=Path, default=None,
        help="V44 hash-bound training sidecar; defaults to CHECKPOINT.training.json.",
    )
    parser.add_argument(
        "--gait-clock-mode",
        choices=("fixed", "speed_scaled"),
        default="fixed",
    )
    parser.add_argument(
        "--integrated-gait-phase",
        action="store_true",
        help="Declare that phase advances by integrating the current cadence.",
    )
    parser.add_argument(
        "--gait-park-at-standstill",
        action="store_true",
        help="Declare that the gait clock freezes when the speed reaches zero.",
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
        "--action-mode",
        choices=("direct", "gait_residual", "integrated_target_velocity"),
        default="direct",
    )
    parser.add_argument(
        "--integrated-stop-action-scale",
        type=float,
        default=0.0,
        help=(
            "Fraction of integrated velocity output retained as a stabilizing "
            "residual while zero-command neutral homing remains active."
        ),
    )
    parser.add_argument(
        "--integrated-stop-target-residual-scale",
        type=float,
        default=0.0,
        help=(
            "Fraction of each joint position envelope available as a policy "
            "posture residual around neutral at zero command."
        ),
    )
    parser.add_argument(
        "--action-history-mode",
        choices=("requested_action", "applied_target"),
        default="requested_action",
        help=(
            "Meaning of the final 12 policy observations: the previous raw "
            "network request or the previous controller-applied target in "
            "normalized action coordinates."
        ),
    )
    parser.add_argument("--residual-action-scale", type=float, default=1.0)
    parser.add_argument(
        "--gait-residual-stop-action-scale",
        type=float,
        default=1.0,
        help=(
            "Fraction of gait-residual actor authority retained after the "
            "command enters the standstill deadband."
        ),
    )
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
    parser.add_argument(
        "--reference-phase-table-path",
        type=Path,
        default=None,
        help="Use an exact generated joint-space phase table instead of analytic crawl.",
    )
    parser.add_argument(
        "--reference-low-phase-table-path",
        type=Path,
        default=None,
        help="Optional low-speed phase table blended with the primary table.",
    )
    parser.add_argument(
        "--reference-blend-speed-min-m-s",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--reference-blend-speed-max-m-s",
        type=float,
        default=None,
    )
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
    parser.add_argument(
        "--target-velocity-limits-rad-s",
        type=float,
        nargs=12,
        metavar="RAD_S",
        default=None,
        help=(
            "Optional per-action target velocity limits in --action-order "
            "order; the scalar limit remains the backward-compatible fallback."
        ),
    )
    parser.add_argument("--max-target-step-deg", type=float, default=2.0)
    parser.add_argument("--startup-ramp-rate-deg-s", type=float, default=45.0)
    parser.add_argument("--startup-settle-s", type=float, default=0.5)
    parser.add_argument("--startup-position-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--forward-speed-min-m-s", type=float, default=0.04)
    parser.add_argument("--forward-speed-max-m-s", type=float, default=0.10)
    parser.add_argument("--recommended-forward-speed-m-s", type=float, default=0.04)
    parser.add_argument(
        "--command-observation-scales",
        type=float,
        nargs=3,
        metavar=("FORWARD", "LATERAL", "YAW"),
        default=(1.0, 1.0, 1.0),
        help="Multipliers applied to the three command observations.",
    )
    args = parser.parse_args()

    if args.gait_period_s <= 0.0:
        parser.error("--gait-period-s must be positive")
    if not 0.0 < args.gait_duty_factor < 1.0:
        parser.error("--gait-duty-factor must be between zero and one")
    if args.target_velocity_limit_rad_s <= 0.0 or args.max_target_step_deg <= 0.0:
        parser.error("target velocity and step limits must be positive")
    target_velocity_limits = (
        tuple(args.target_velocity_limits_rad_s)
        if args.target_velocity_limits_rad_s is not None
        else (args.target_velocity_limit_rad_s,) * len(ACTION_ORDER)
    )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in target_velocity_limits
    ):
        parser.error(
            "--target-velocity-limits-rad-s values must be finite and positive"
        )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in args.command_observation_scales
    ):
        parser.error("command observation scales must be finite and positive")
    if not 0.0 <= args.gait_stride_scale_min <= 1.0:
        parser.error("--gait-stride-scale-min must be in [0, 1]")
    if not 0.0 < args.residual_action_scale <= 1.0:
        parser.error("--residual-action-scale must be in (0, 1]")
    if not 0.0 <= args.gait_residual_stop_action_scale <= 1.0:
        parser.error("--gait-residual-stop-action-scale must be in [0, 1]")
    if not 0.0 <= args.integrated_stop_action_scale <= 1.0:
        parser.error("--integrated-stop-action-scale must be in [0, 1]")
    if not 0.0 <= args.integrated_stop_target_residual_scale <= 1.0:
        parser.error(
            "--integrated-stop-target-residual-scale must be in [0, 1]"
        )
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
    reference_phase_rows: list[list[float]] = []
    reference_phase_payload: dict[str, object] | None = None
    reference_low_phase_rows: list[list[float]] = []
    reference_low_phase_payload: dict[str, object] | None = None
    if args.reference_phase_table_path is not None:
        reference_phase_rows, reference_phase_payload = load_phase_reference(
            args.reference_phase_table_path,
            parser,
        )
    if args.reference_low_phase_table_path is not None:
        if not reference_phase_rows:
            parser.error(
                "--reference-low-phase-table-path requires "
                "--reference-phase-table-path"
            )
        (
            reference_low_phase_rows,
            reference_low_phase_payload,
        ) = load_phase_reference(args.reference_low_phase_table_path, parser)
        if len(reference_low_phase_rows) != len(reference_phase_rows):
            parser.error("low- and high-speed phase tables must have equal length")
    blend_bounds_present = (
        args.reference_blend_speed_min_m_s is not None,
        args.reference_blend_speed_max_m_s is not None,
    )
    if blend_bounds_present[0] != blend_bounds_present[1]:
        parser.error("both reference blend-speed bounds are required")
    if reference_low_phase_rows:
        if not all(blend_bounds_present):
            parser.error("a low-speed phase table requires blend-speed bounds")
        if not (
            math.isfinite(args.reference_blend_speed_min_m_s)
            and math.isfinite(args.reference_blend_speed_max_m_s)
            and args.reference_blend_speed_min_m_s
            < args.reference_blend_speed_max_m_s
        ):
            parser.error(
                "reference blend speeds must be finite and maximum must exceed minimum"
            )
    elif any(blend_bounds_present):
        parser.error("reference blend-speed bounds require a low-speed phase table")
    if reference_phase_rows and args.action_mode != "gait_residual":
        parser.error("phase-table references require --action-mode gait_residual")
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
        and not reference_phase_rows
    ):
        parser.error(
            "gait-residual export requires distributed_support_crawl or an exact "
            "phase table"
        )
    if (
        args.action_history_mode == "applied_target"
        and args.action_mode
        not in ("direct", "gait_residual", "integrated_target_velocity")
    ):
        parser.error(
            "applied-target action history requires direct, gait-residual, "
            "or integrated-target-velocity actions"
        )

    if args.training_profile in (
        V31_TRAINING_PROFILE,
        V32_TRAINING_PROFILE,
        V33_TRAINING_PROFILE,
        V34_TRAINING_PROFILE,
        V35_TRAINING_PROFILE,
        V36_TRAINING_PROFILE,
        V38_TRAINING_PROFILE,
    ):
        is_v38 = args.training_profile == V38_TRAINING_PROFILE
        is_v36 = args.training_profile == V36_TRAINING_PROFILE
        is_v35 = args.training_profile == V35_TRAINING_PROFILE
        is_v34 = args.training_profile == V34_TRAINING_PROFILE
        is_v33 = args.training_profile == V33_TRAINING_PROFILE
        is_v32 = args.training_profile == V32_TRAINING_PROFILE
        is_hardware_matched = not (
            args.training_profile == V31_TRAINING_PROFILE
        )
        profile_label = (
            "V38"
            if is_v38
            else "V36"
            if is_v36
            else "V35"
            if is_v35
            else "V34"
            if is_v34
            else "V33"
            if is_v33
            else "V32"
            if is_v32
            else "V31"
        )
        mismatches: list[str] = []

        def require_equal(name: str, actual: object, expected: object) -> None:
            if actual != expected:
                mismatches.append(f"{name}={actual!r} (expected {expected!r})")

        def require_close(name: str, actual: float, expected: float) -> None:
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-7):
                mismatches.append(f"{name}={actual!r} (expected {expected!r})")

        require_equal(
            "training_task",
            args.training_task,
            V38_TRAINING_TASK
            if is_v38
            else V36_TRAINING_TASK
            if is_v36
            else V35_TRAINING_TASK
            if is_v35
            else V34_TRAINING_TASK
            if is_v34
            else V33_TRAINING_TASK
            if is_v33
            else V32_TRAINING_TASK
            if is_v32
            else V31_TRAINING_TASK,
        )
        require_equal("gait_clock_mode", args.gait_clock_mode, "speed_scaled")
        require_equal("integrated_gait_phase", args.integrated_gait_phase, True)
        require_equal("gait_park_at_standstill", args.gait_park_at_standstill, True)
        require_equal("gait_pattern", args.gait_pattern, "diagonal_trot")
        require_equal(
            "action_mode",
            args.action_mode,
            (
                "integrated_target_velocity"
                if (is_v34 or is_v35 or is_v36 or is_v38)
                else "direct"
            ),
        )
        require_close(
            "integrated_stop_action_scale",
            args.integrated_stop_action_scale,
            0.0,
        )
        require_close(
            "integrated_stop_target_residual_scale",
            args.integrated_stop_target_residual_scale,
            0.0,
        )
        require_equal(
            "action_history_mode",
            args.action_history_mode,
            (
                "applied_target"
                if (is_v33 or is_v34 or is_v35 or is_v36 or is_v38)
                else "requested_action"
            ),
        )
        require_equal("heading_hold_enabled", args.heading_hold_enabled, True)
        require_close("gait_standstill_deadband_m_s", args.gait_standstill_deadband_m_s, 0.004)
        is_flat_sole_high_speed = is_v36 or is_v38
        is_high_speed = is_v35 or is_flat_sole_high_speed
        expected_gait_speed_min = 0.080 if is_high_speed else 0.060 if is_v34 else 0.040 if is_hardware_matched else 0.060
        expected_gait_speed_max = 0.140 if is_high_speed else 0.120 if is_hardware_matched else 0.200
        expected_frequency_min = 1.20 if is_flat_sole_high_speed else 1.00 if is_v35 else 0.90 if is_v34 else 0.040 / (0.065 * 0.75) if is_hardware_matched else 0.90
        expected_frequency_max = 1.65 if is_flat_sole_high_speed else 1.45 if is_v35 else 1.40 if is_v34 else 0.120 / 0.065 if is_hardware_matched else 2.40
        expected_duty_factor = 0.62 if is_flat_sole_high_speed else 0.64 if is_v35 else 0.68 if is_hardware_matched else 0.60
        expected_stride_scale_min = 0.80 if is_high_speed else 0.85 if is_v34 else 0.75 if is_hardware_matched else 0.70
        require_close(
            "gait_speed_min_m_s", args.gait_speed_min_m_s, expected_gait_speed_min
        )
        require_close(
            "gait_speed_max_m_s", args.gait_speed_max_m_s, expected_gait_speed_max
        )
        require_close(
            "gait_frequency_min_hz", args.gait_frequency_min_hz, expected_frequency_min
        )
        require_close(
            "gait_frequency_max_hz", args.gait_frequency_max_hz, expected_frequency_max
        )
        require_close(
            "gait_duty_factor", args.gait_duty_factor, expected_duty_factor
        )
        require_close(
            "gait_stride_scale_min", args.gait_stride_scale_min, expected_stride_scale_min
        )
        expected_phases = (0.0, 0.5, 0.5, 0.0)
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7)
            for actual, expected in zip(args.gait_phase_offsets, expected_phases)
        ):
            mismatches.append("gait_phase_offsets do not match the V31 diagonal pairs")
        require_close("heading_hold_kp_s", args.heading_hold_kp_s, 0.75 if is_hardware_matched else 1.5)
        require_close(
            "heading_hold_max_correction_rad_s",
            args.heading_hold_max_correction_rad_s,
            0.12 if is_hardware_matched else 0.20,
        )
        require_close(
            "target_velocity_limit_rad_s",
            args.target_velocity_limit_rad_s,
            math.radians(120.0 if is_hardware_matched else 240.0),
        )
        expected_target_velocity_limits = (
            (math.radians(60.0),) * 4 + (math.radians(120.0),) * 8
            if is_v38
            else (
                math.radians(120.0 if is_hardware_matched else 240.0),
            )
            * len(ACTION_ORDER)
        )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7)
            for actual, expected in zip(
                target_velocity_limits,
                expected_target_velocity_limits,
                strict=True,
            )
        ):
            mismatches.append(
                "target_velocity_limits_rad_s do not match the versioned policy contract"
            )
        require_close("max_target_step_deg", args.max_target_step_deg, 2.0 if is_hardware_matched else 4.0)
        expected_forward_min = 0.080 if is_high_speed else 0.060 if is_v34 else 0.040 if is_hardware_matched else 0.050
        expected_forward_max = 0.140 if is_high_speed else 0.120 if is_hardware_matched else 0.200
        expected_recommended_speed = 0.120 if is_flat_sole_high_speed else 0.080 if (is_v34 or is_v35) else 0.060 if is_hardware_matched else 0.080
        require_close("forward_speed_min_m_s", args.forward_speed_min_m_s, expected_forward_min)
        require_close("forward_speed_max_m_s", args.forward_speed_max_m_s, expected_forward_max)
        require_close("recommended_forward_speed_m_s", args.recommended_forward_speed_m_s, expected_recommended_speed)
        expected_command_scales = (10.0, 10.0, 10.0) if is_high_speed else (1.0, 1.0, 1.0)
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7)
            for actual, expected in zip(
                args.command_observation_scales,
                expected_command_scales,
                strict=True,
            )
        ):
            mismatches.append(
                "command_observation_scales do not match the versioned policy contract"
            )
        if mismatches:
            parser.error(
                f"{profile_label} export contract mismatch; pass the complete trained profile:\n  - "
                + "\n  - ".join(mismatches)
            )

    if args.training_profile == V43_TRAINING_PROFILE:
        mismatches: list[str] = []

        def require_v43_equal(
            name: str,
            actual: object,
            expected: object,
        ) -> None:
            if actual != expected:
                mismatches.append(f"{name}={actual!r} (expected {expected!r})")

        def require_v43_close(name: str, actual: float, expected: float) -> None:
            try:
                is_match = math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1.0e-7
                )
            except (TypeError, ValueError):
                is_match = False
            if not is_match:
                mismatches.append(f"{name}={actual!r} (expected {expected!r})")

        require_v43_equal("training_task", args.training_task, V43_TRAINING_TASK)
        require_v43_equal("gait_clock_mode", args.gait_clock_mode, "speed_scaled")
        require_v43_equal("integrated_gait_phase", args.integrated_gait_phase, True)
        require_v43_equal("gait_park_at_standstill", args.gait_park_at_standstill, True)
        require_v43_equal("gait_pattern", args.gait_pattern, "diagonal_trot")
        require_v43_equal("action_mode", args.action_mode, "gait_residual")
        require_v43_equal(
            "action_history_mode", args.action_history_mode, "applied_target"
        )
        require_v43_equal("heading_hold_enabled", args.heading_hold_enabled, True)
        require_v43_close(
            "gait_standstill_deadband_m_s",
            args.gait_standstill_deadband_m_s,
            0.004,
        )
        require_v43_close("gait_speed_min_m_s", args.gait_speed_min_m_s, 0.10)
        require_v43_close("gait_speed_max_m_s", args.gait_speed_max_m_s, 0.14)
        require_v43_close("gait_frequency_min_hz", args.gait_frequency_min_hz, 1.15)
        require_v43_close("gait_frequency_max_hz", args.gait_frequency_max_hz, 1.60)
        require_v43_close("gait_stride_scale_min", args.gait_stride_scale_min, 1.0)
        require_v43_close("gait_duty_factor", args.gait_duty_factor, 0.46)
        expected_phases = (0.0, 0.5, 0.5, 0.0)
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7)
            for actual, expected in zip(
                args.gait_phase_offsets,
                expected_phases,
                strict=True,
            )
        ):
            mismatches.append("gait_phase_offsets do not match the V43 diagonal pairs")
        require_v43_close("residual_action_scale", args.residual_action_scale, 0.04)
        require_v43_close(
            "gait_residual_stop_action_scale",
            args.gait_residual_stop_action_scale,
            0.0,
        )
        expected_residual_scales = (0.04,) * 4 + (0.12,) * 8
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7)
            for actual, expected in zip(
                residual_action_scales,
                expected_residual_scales,
                strict=True,
            )
        ):
            mismatches.append(
                "residual_action_scales do not match the V43 0.04/0.12 contract"
            )
        require_v43_close("reference_start_ramp_s", args.reference_start_ramp_s, 1.20)
        require_v43_close(
            "reference_blend_speed_min_m_s",
            args.reference_blend_speed_min_m_s,
            0.10,
        )
        require_v43_close(
            "reference_blend_speed_max_m_s",
            args.reference_blend_speed_max_m_s,
            0.14,
        )
        if len(reference_phase_rows) != 256:
            mismatches.append(
                f"reference_phase_table has {len(reference_phase_rows)} rows "
                "(expected 256)"
            )
        if len(reference_low_phase_rows) != 256:
            mismatches.append(
                f"reference_low_phase_table has {len(reference_low_phase_rows)} rows "
                "(expected 256)"
            )
        if args.reference_phase_table_path is None or (
            sha256(args.reference_phase_table_path.resolve())
            != V43_REFERENCE_HIGH_SHA256
        ):
            mismatches.append("reference_phase_table does not match tested V43 high table")
        if args.reference_low_phase_table_path is None or (
            sha256(args.reference_low_phase_table_path.resolve())
            != V43_REFERENCE_LOW_SHA256
        ):
            mismatches.append(
                "reference_low_phase_table does not match tested V43 low table"
            )
        require_v43_close("heading_hold_kp_s", args.heading_hold_kp_s, 0.75)
        require_v43_close(
            "heading_hold_max_correction_rad_s",
            args.heading_hold_max_correction_rad_s,
            0.12,
        )
        require_v43_close(
            "target_velocity_limit_rad_s",
            args.target_velocity_limit_rad_s,
            math.radians(180.0),
        )
        expected_target_velocity_limits = (
            (math.radians(60.0),) * 4 + (math.radians(180.0),) * 8
        )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7)
            for actual, expected in zip(
                target_velocity_limits,
                expected_target_velocity_limits,
                strict=True,
            )
        ):
            mismatches.append(
                "target_velocity_limits_rad_s do not match the V43 contract"
            )
        require_v43_close("max_target_step_deg", args.max_target_step_deg, 2.0)
        require_v43_close("forward_speed_min_m_s", args.forward_speed_min_m_s, 0.10)
        require_v43_close("forward_speed_max_m_s", args.forward_speed_max_m_s, 0.14)
        require_v43_close(
            "recommended_forward_speed_m_s",
            args.recommended_forward_speed_m_s,
            0.12,
        )
        expected_command_scales = (10.0, 10.0, 10.0)
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7)
            for actual, expected in zip(
                args.command_observation_scales,
                expected_command_scales,
                strict=True,
            )
        ):
            mismatches.append(
                "command_observation_scales do not match the V43 contract"
            )
        if mismatches:
            parser.error(
                "V43 export contract mismatch; pass the complete trained profile:\n  - "
                + "\n  - ".join(mismatches)
            )

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
        require_close(
            "gait_residual_stop_action_scale",
            args.gait_residual_stop_action_scale,
            1.0,
        )
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

    reference_joint_position_rad: list[list[float]] = list(reference_phase_rows)
    stance_forward_bias_m = 0.0
    if args.action_mode == "gait_residual" and not reference_joint_position_rad:
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

    v44_provenance = None
    if args.training_profile == V44_TRAINING_PROFILE or args.training_task == V44_TRAINING_TASK:
        try:
            v44_provenance = validate_v44_export_arguments(
                args, residual_action_scales, target_velocity_limits
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"V44 export contract/provenance mismatch: {exc}")

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
            "integrated_phase": args.integrated_gait_phase,
            "park_at_standstill": args.gait_park_at_standstill,
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
            "output_semantics": (
                "fraction_of_elapsed_time_target_velocity_limit"
                if args.action_mode == "integrated_target_velocity"
                else "normalized_joint_position_residual_about_gait_reference"
                if args.action_mode == "gait_residual"
                else "normalized_joint_position"
            ),
            "zero_command_target_behavior": (
                "slew_to_neutral"
                if args.action_mode in (
                    "gait_residual",
                    "integrated_target_velocity",
                )
                else "policy_controlled"
            ),
            "zero_command_policy_velocity_scale": (
                args.integrated_stop_action_scale
                if args.action_mode == "integrated_target_velocity"
                else 0.0
            ),
            "zero_command_policy_target_scale": (
                args.integrated_stop_target_residual_scale
                if args.action_mode == "integrated_target_velocity"
                else 0.0
            ),
            # Older runtimes understand only the scalar field.  The smallest
            # configured value is a fail-safe fallback for a non-uniform policy;
            # vector-aware runtimes use the ordered field below.
            "residual_scale": min(residual_action_scales),
            "residual_scale_by_action": list(residual_action_scales),
            "zero_command_residual_scale": (
                args.gait_residual_stop_action_scale
                if args.action_mode == "gait_residual"
                else 0.0
            ),
        },
        "observation_count": 50,
        "action_count": 12,
        "observation_scales": {
            "command_forward_lateral_yaw": list(
                args.command_observation_scales
            ),
        },
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
            # An older scalar-only runtime falls back to the safest joint rate.
            # Vector-aware runtimes preserve sagittal speed while limiting only
            # the four abduction axes.
            "target_velocity_limit_rad_s": min(target_velocity_limits),
            "target_velocity_limit_rad_s_by_action": list(
                target_velocity_limits
            ),
            "max_target_step_rad": args.max_target_step_deg * 3.141592653589793 / 180.0,
        },
        "startup": {
            "mode": "prepared_neutral",
            "ramp_rate_deg_s": args.startup_ramp_rate_deg_s,
            "settle_s": args.startup_settle_s,
            "position_tolerance_deg": args.startup_position_tolerance_deg,
        },
        "action_history_mode": args.action_history_mode,
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
            (
                "previous_applied_normalized_target[12]"
                if args.action_history_mode == "applied_target"
                else "previous_normalized_action[12]"
            ),
        ],
        "action_order": list(ACTION_ORDER),
    }
    if args.training_profile in (
        V31_TRAINING_PROFILE,
        V32_TRAINING_PROFILE,
        V33_TRAINING_PROFILE,
        V34_TRAINING_PROFILE,
        V35_TRAINING_PROFILE,
        V36_TRAINING_PROFILE,
        V38_TRAINING_PROFILE,
        V43_TRAINING_PROFILE,
        V44_TRAINING_PROFILE,
    ):
        is_v43 = args.training_profile in (V43_TRAINING_PROFILE, V44_TRAINING_PROFILE)
        is_v38 = args.training_profile == V38_TRAINING_PROFILE
        metadata["run_stop_contract"] = {
            "trained_run_duration_s": (
                [10.0, 12.0]
                if is_v43
                else [8.0, 10.0]
                if is_v38
                else [4.5, 6.5]
            ),
            "trained_graceful_ramp_s": (
                [0.55, 1.00]
                if is_v43
                else [0.45, 1.00]
                if is_v38
                else [0.35, 0.90]
            ),
            "trained_abrupt_stop_fraction": (
                0.15 if is_v43 else 0.20 if is_v38 else 0.35
            ),
            "recommended_stop_ramp_s": 0.90 if is_v43 else 0.60,
            "recommended_stopped_hold_s": 2.0 if is_v43 else 1.0,
            "ramp_shape": "cubic_smoothstep" if is_v43 else "linear",
            "phase_alignment": {
                "enabled": is_v43,
                "ramp_start_phase_cycles": 0.05 if is_v43 else 0.0,
                "phase_tolerance_cycles": 0.03 if is_v43 else 0.0,
                "maximum_wait_s": 0.50 if is_v43 else 0.0,
            },
            "stopped_stance": {
                "all_four_feet": True,
                "fore_aft_m": 0.080,
                "down_m": 0.329341447,
                "stance_center_offset_m": 0.0,
                "target_body_pitch_rad": 0.0,
                "target_base_height_m": 0.375,
            },
        }
    if reference_joint_position_rad:
        if reference_phase_rows:
            metadata["gait_reference"] = {
                "mode": "phase_table",
                "start_ramp_s": args.reference_start_ramp_s,
                "sample_count": len(reference_phase_rows),
                "joint_position_rad": reference_phase_rows,
                "low_speed_joint_position_rad": reference_low_phase_rows,
                "blend_speed_min_m_s": args.reference_blend_speed_min_m_s,
                "blend_speed_max_m_s": args.reference_blend_speed_max_m_s,
                "source": {
                    "high_filename": args.reference_phase_table_path.name,
                    "high_sha256": sha256(
                        args.reference_phase_table_path.resolve()
                    ),
                    "high_generation": reference_phase_payload.get("generation"),
                    "low_filename": (
                        args.reference_low_phase_table_path.name
                        if args.reference_low_phase_table_path is not None
                        else None
                    ),
                    "low_sha256": (
                        sha256(args.reference_low_phase_table_path.resolve())
                        if args.reference_low_phase_table_path is not None
                        else None
                    ),
                    "low_generation": (
                        reference_low_phase_payload.get("generation")
                        if reference_low_phase_payload is not None
                        else None
                    ),
                },
            }
        else:
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
    if v44_provenance is not None:
        metadata["training_provenance"] = v44_provenance
        metadata["qualification_status"] = "unqualified_corrective_stage_candidate"
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
