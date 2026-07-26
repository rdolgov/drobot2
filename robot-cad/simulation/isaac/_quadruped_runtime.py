"""Pure-Python kinematics and constants shared by the Isaac Sim runners.

The URDF is the source of truth for link frames, inertias, and hard joint
limits.  This module only carries the controller-side assumptions needed to
construct conservative standing and crawl targets.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping

LEGS = ("front_left", "front_right", "rear_left", "rear_right")
JOINT_KINDS = ("hip_abduction", "hip_flexion", "knee")
EXPECTED_DOF_NAMES = frozenset(f"{leg}_{joint_kind}" for leg in LEGS for joint_kind in JOINT_KINDS)

# Verified Feetech ST-3215-C018 / Waveshare ST3215, 12 V variant.
RATED_TORQUE_NM = 0.980665
STALL_TORQUE_NM = 2.941995
MAX_NO_LOAD_VELOCITY_RAD_S = 4.712389
TORQUE_PROFILES_NM = {
    "rated": RATED_TORQUE_NM,
    "stall": STALL_TORQUE_NM,
}

# Both printable upper-arm links use the same fork-axis spacing.
LINK_LENGTH_M = 0.159896689

DEFAULT_STANCE_DOWN_M = 0.270
DEFAULT_STANCE_FORE_AFT_M = 0.050
DEFAULT_ABDUCTION_DEG = 6.0

CRAWL_DUTY_FACTOR = 0.75
CRAWL_PHASE_OFFSETS = {
    # Four-beat lateral sequence: rear-right, front-right, rear-left,
    # front-left.  Exactly one leg swings at a time away from boundaries.
    "rear_right": 0.75,
    "front_right": 0.50,
    "rear_left": 0.25,
    "front_left": 0.00,
}

QUASISTATIC_SWING_ORDER = (
    "rear_right",
    "front_right",
    "rear_left",
    "front_left",
)
QUASISTATIC_STEP_FRACTION = 0.22
QUASISTATIC_ADVANCE_FRACTION = 0.10
QUASISTATIC_SETTLE_FRACTION = 0.02
QUASISTATIC_STEP_PHASES = (
    ("weight_transfer", 0.20),
    ("lift", 0.16),
    ("swing", 0.28),
    ("lower", 0.16),
    ("touchdown", 0.10),
    ("weight_return", 0.10),
)


def add_robot_reference(stage_utils, usd_path: str, path: str) -> dict:
    """Reference either a packaged variant asset or monolithic Isaac asset."""

    from pxr import Usd

    absolute_usd = os.path.abspath(usd_path)
    asset_stage = Usd.Stage.Open(absolute_usd)
    if asset_stage is None:
        raise RuntimeError(f"Could not open robot USD asset: {absolute_usd}")
    default_prim = asset_stage.GetDefaultPrim()
    variant_sets = (
        list(default_prim.GetVariantSets().GetNames())
        if default_prim and default_prim.IsValid()
        else []
    )
    physics_variants = []
    if "Physics" in variant_sets:
        physics_variants = list(
            default_prim.GetVariantSet("Physics").GetVariantNames()
        )
    asset_stage = None

    if "physx" in physics_variants:
        stage_utils.add_reference_to_stage(
            usd_path=absolute_usd,
            path=path,
            variants=[("Physics", "physx")],
        )
        selected_physics = "physx"
    else:
        stage_utils.add_reference_to_stage(
            usd_path=absolute_usd,
            path=path,
        )
        selected_physics = None
    return {
        "asset": absolute_usd,
        "variant_sets": variant_sets,
        "physics_variants": physics_variants,
        "selected_physics_variant": selected_physics,
    }


def torque_cap_nm(profile: str, explicit_nm: float | None) -> tuple[str, float]:
    """Resolve a named ST3215 torque profile or a positive explicit cap."""
    if explicit_nm is not None:
        if explicit_nm <= 0.0:
            raise ValueError("Explicit effort limit must be positive")
        return "custom", float(explicit_nm)
    try:
        return profile, TORQUE_PROFILES_NM[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown torque profile: {profile!r}") from exc


def _front_sign(leg: str) -> float:
    if leg.startswith("front_"):
        return 1.0
    if leg.startswith("rear_"):
        return -1.0
    raise ValueError(f"Unknown leg name: {leg!r}")


def _side_sign(leg: str) -> float:
    if leg.endswith("_left"):
        return 1.0
    if leg.endswith("_right"):
        return -1.0
    raise ValueError(f"Unknown leg name: {leg!r}")


def smoothstep(value: float) -> float:
    value = min(max(float(value), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def leg_ik(leg: str, down_m: float, world_forward_m: float) -> tuple[float, float]:
    """Solve one leg in the URDF joint frame.

    At zero, each printable arm points down along its local +X.  The generated
    URDF mirrors the physical joint axes: left flexion/knee axes are local -Z
    and right axes are local +Z.  Consequently, the same positive command
    rotates either side toward world +X and both sides share one sagittal IK.
    """
    if down_m <= 0.0:
        raise ValueError("Leg down target must be positive")
    front_sign = _front_sign(leg)
    sagittal_y_m = world_forward_m

    distance_sq = down_m * down_m + sagittal_y_m * sagittal_y_m
    cosine_knee = (distance_sq - 2.0 * LINK_LENGTH_M * LINK_LENGTH_M) / (
        2.0 * LINK_LENGTH_M * LINK_LENGTH_M
    )
    if not -1.0 <= cosine_knee <= 1.0:
        raise ValueError(
            "Unreachable leg target "
            f"{leg}: down={down_m:.6f} m, forward={world_forward_m:.6f} m, "
            f"cos(knee)={cosine_knee:.6f}"
        )

    knee = front_sign * math.acos(min(max(cosine_knee, -1.0), 1.0))
    hip_flexion = math.atan2(sagittal_y_m, down_m) - math.atan2(
        LINK_LENGTH_M * math.sin(knee),
        LINK_LENGTH_M + LINK_LENGTH_M * math.cos(knee),
    )
    return hip_flexion, knee


def pose_by_name(
    *,
    down_by_leg_m: Mapping[str, float],
    forward_by_leg_m: Mapping[str, float],
    abduction_deg: float = DEFAULT_ABDUCTION_DEG,
    abduction_by_leg_deg: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build a complete 12-joint pose keyed by the exact URDF joint names."""
    result: dict[str, float] = {}
    for leg in LEGS:
        leg_abduction_deg = (
            float(abduction_by_leg_deg[leg])
            if abduction_by_leg_deg is not None
            else abduction_deg
        )
        hip_flexion, knee = leg_ik(
            leg,
            float(down_by_leg_m[leg]),
            float(forward_by_leg_m[leg]),
        )
        result[f"{leg}_hip_abduction"] = math.radians(leg_abduction_deg)
        result[f"{leg}_hip_flexion"] = hip_flexion
        result[f"{leg}_knee"] = knee
    return result


def stance_by_name(
    *,
    down_m: float = DEFAULT_STANCE_DOWN_M,
    fore_aft_m: float = DEFAULT_STANCE_FORE_AFT_M,
    abduction_deg: float = DEFAULT_ABDUCTION_DEG,
) -> dict[str, float]:
    """Return a symmetric, flexed four-point stance."""
    return pose_by_name(
        down_by_leg_m={leg: down_m for leg in LEGS},
        forward_by_leg_m={leg: _front_sign(leg) * fore_aft_m for leg in LEGS},
        abduction_deg=abduction_deg,
    )


def crawl_by_name(
    gait_time_s: float,
    *,
    period_s: float,
    stride_m: float,
    lift_m: float,
    down_m: float = DEFAULT_STANCE_DOWN_M,
    fore_aft_m: float = DEFAULT_STANCE_FORE_AFT_M,
    abduction_deg: float = DEFAULT_ABDUCTION_DEG,
    duty_factor: float = CRAWL_DUTY_FACTOR,
) -> dict[str, float]:
    """Return a slow four-beat crawl target with one leg swinging at a time."""
    if period_s <= 0.0:
        raise ValueError("Crawl period must be positive")
    if stride_m <= 0.0 or lift_m <= 0.0:
        raise ValueError("Crawl stride and lift must be positive")
    if not 0.5 < duty_factor < 1.0:
        raise ValueError("Crawl duty factor must be between 0.5 and 1.0")

    cycle = gait_time_s / period_s
    half_stride = stride_m / 2.0
    down_by_leg: dict[str, float] = {}
    forward_by_leg: dict[str, float] = {}
    for leg in LEGS:
        phase = (cycle + CRAWL_PHASE_OFFSETS[leg]) % 1.0
        nominal_forward = _front_sign(leg) * fore_aft_m
        if phase < duty_factor:
            stance_u = phase / duty_factor
            stride_offset = half_stride - stride_m * stance_u
            leg_down = down_m
        else:
            swing_u = (phase - duty_factor) / (1.0 - duty_factor)
            stride_offset = -half_stride + stride_m * smoothstep(swing_u)
            leg_down = down_m - lift_m * math.sin(math.pi * swing_u)
        forward_by_leg[leg] = nominal_forward + stride_offset
        down_by_leg[leg] = leg_down

    return pose_by_name(
        down_by_leg_m=down_by_leg,
        forward_by_leg_m=forward_by_leg,
        abduction_deg=abduction_deg,
    )


def quasistatic_crawl_by_name(
    gait_time_s: float,
    *,
    period_s: float,
    stride_m: float,
    lift_m: float,
    weight_shift_forward_m: float,
    weight_shift_lateral_m: float,
    down_m: float = DEFAULT_STANCE_DOWN_M,
    fore_aft_m: float = DEFAULT_STANCE_FORE_AFT_M,
    abduction_deg: float = DEFAULT_ABDUCTION_DEG,
) -> tuple[dict[str, float], dict[str, object]]:
    """Return a contact-verifiable crawl target with explicit weight transfer.

    Each foot is advanced while the body is commanded toward the diagonally
    opposite support corner.  The four placed feet then drive backward
    together, which is the propulsion phase that should move the floating
    body forward if contact friction is sufficient.
    """
    if period_s <= 0.0:
        raise ValueError("Quasi-static crawl period must be positive")
    if stride_m <= 0.0 or lift_m <= 0.0:
        raise ValueError("Quasi-static stride and lift must be positive")
    if weight_shift_forward_m < 0.0 or weight_shift_lateral_m < 0.0:
        raise ValueError("Weight-transfer magnitudes must be non-negative")
    if not math.isclose(
        len(QUASISTATIC_SWING_ORDER) * QUASISTATIC_STEP_FRACTION
        + QUASISTATIC_ADVANCE_FRACTION
        + QUASISTATIC_SETTLE_FRACTION,
        1.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("Quasi-static phase fractions do not total one cycle")

    cycle_phase = (gait_time_s / period_s) % 1.0
    step_region = len(QUASISTATIC_SWING_ORDER) * QUASISTATIC_STEP_FRACTION
    offsets = {leg: 0.0 for leg in LEGS}
    down_by_leg = {leg: down_m for leg in LEGS}
    body_shift_forward_m = 0.0
    body_shift_lateral_m = 0.0
    swing_leg: str | None = None
    phase_name = "cycle_settle"
    phase_progress = 0.0
    expected_support_legs = list(LEGS)

    if cycle_phase < step_region:
        step_index = min(
            int(cycle_phase / QUASISTATIC_STEP_FRACTION),
            len(QUASISTATIC_SWING_ORDER) - 1,
        )
        swing_leg = QUASISTATIC_SWING_ORDER[step_index]
        for completed_leg in QUASISTATIC_SWING_ORDER[:step_index]:
            offsets[completed_leg] = stride_m

        step_start = step_index * QUASISTATIC_STEP_FRACTION
        step_u = (cycle_phase - step_start) / QUASISTATIC_STEP_FRACTION
        phase_start = 0.0
        phase_u = 0.0
        for candidate_name, candidate_fraction in QUASISTATIC_STEP_PHASES:
            phase_end = phase_start + candidate_fraction
            if step_u < phase_end or candidate_name == QUASISTATIC_STEP_PHASES[-1][0]:
                phase_name = candidate_name
                phase_u = (step_u - phase_start) / candidate_fraction
                phase_u = min(max(phase_u, 0.0), 1.0)
                break
            phase_start = phase_end
        phase_progress = phase_u

        if phase_name == "weight_transfer":
            transfer = smoothstep(phase_u)
        elif phase_name == "weight_return":
            transfer = 1.0 - smoothstep(phase_u)
        else:
            transfer = 1.0

        body_shift_forward_m = (
            -_front_sign(swing_leg) * weight_shift_forward_m * transfer
        )
        body_shift_lateral_m = (
            -_side_sign(swing_leg) * weight_shift_lateral_m * transfer
        )

        if phase_name == "lift":
            down_by_leg[swing_leg] = down_m - lift_m * smoothstep(phase_u)
        elif phase_name == "swing":
            down_by_leg[swing_leg] = down_m - lift_m
            offsets[swing_leg] = stride_m * smoothstep(phase_u)
        elif phase_name == "lower":
            down_by_leg[swing_leg] = down_m - lift_m * (1.0 - smoothstep(phase_u))
            offsets[swing_leg] = stride_m
        elif phase_name in ("touchdown", "weight_return"):
            offsets[swing_leg] = stride_m

        if phase_name in ("lift", "swing", "lower"):
            expected_support_legs = [leg for leg in LEGS if leg != swing_leg]
    elif cycle_phase < step_region + QUASISTATIC_ADVANCE_FRACTION:
        phase_name = "all_feet_advance"
        phase_progress = (
            cycle_phase - step_region
        ) / QUASISTATIC_ADVANCE_FRACTION
        remaining = 1.0 - smoothstep(phase_progress)
        offsets = {leg: stride_m * remaining for leg in LEGS}
    else:
        phase_progress = (
            cycle_phase - step_region - QUASISTATIC_ADVANCE_FRACTION
        ) / QUASISTATIC_SETTLE_FRACTION

    forward_by_leg = {
        leg: _front_sign(leg) * fore_aft_m
        + offsets[leg]
        - body_shift_forward_m
        for leg in LEGS
    }
    nominal_abduction = math.radians(abduction_deg)
    foot_delta_lateral_m = -body_shift_lateral_m
    abduction_by_leg: dict[str, float] = {}
    for leg in LEGS:
        nominal_leg_down = down_by_leg[leg]
        vertical = nominal_leg_down * math.cos(nominal_abduction)
        outward = nominal_leg_down * math.sin(nominal_abduction)
        shifted_outward = outward + _side_sign(leg) * foot_delta_lateral_m
        down_by_leg[leg] = math.hypot(vertical, shifted_outward)
        abduction_by_leg[leg] = math.degrees(
            math.atan2(shifted_outward, vertical)
        )
    pose = pose_by_name(
        down_by_leg_m=down_by_leg,
        forward_by_leg_m=forward_by_leg,
        abduction_by_leg_deg=abduction_by_leg,
    )
    state: dict[str, object] = {
        "cycle_phase": cycle_phase,
        "phase": phase_name,
        "phase_progress": min(max(phase_progress, 0.0), 1.0),
        "swing_leg": swing_leg,
        "expected_support_legs": expected_support_legs,
        "foot_forward_offsets_m": offsets,
        "body_shift_forward_m": body_shift_forward_m,
        "body_shift_lateral_m": body_shift_lateral_m,
    }
    return pose, state


def targets_for_order(
    dof_names: Iterable[str],
    pose: Mapping[str, float],
) -> list[float]:
    names = list(dof_names)
    missing = [name for name in names if name not in pose]
    if missing:
        raise KeyError(f"Pose does not define these joints: {missing}")
    return [float(pose[name]) for name in names]


def body_tilt_deg(quaternion_wxyz: Iterable[float]) -> float:
    """Return unsigned body tilt away from world +Z, ignoring yaw."""
    values = [float(value) for value in quaternion_wxyz]
    if len(values) != 4:
        raise ValueError("Quaternion must contain four WXYZ values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    _, x, y, _ = [value / norm for value in values]
    world_up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(min(max(world_up_z, -1.0), 1.0)))
