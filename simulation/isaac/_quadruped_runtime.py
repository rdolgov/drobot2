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
EXPECTED_DOF_NAMES = frozenset(
    f"{leg}_{joint_kind}" for leg in LEGS for joint_kind in JOINT_KINDS
)

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
RECTANGULAR_SHOE_CONTACT_EXTENSION_M = 0.031
RECTANGULAR_SHOE_HALF_FORE_AFT_M = 0.050
RECTANGULAR_SHOE_EFFECTIVE_DISTAL_LENGTH_M = (
    LINK_LENGTH_M + RECTANGULAR_SHOE_CONTACT_EXTENSION_M
)

DEFAULT_STANCE_DOWN_M = 0.329341447
DEFAULT_STANCE_FORE_AFT_M = 0.080
DEFAULT_ABDUCTION_DEG = 0.0

CRAWL_DUTY_FACTOR = 0.75
CRAWL_PHASE_OFFSETS = {
    # Four-beat lateral sequence: rear-right, front-right, rear-left,
    # front-left.  Exactly one leg swings at a time away from boundaries.
    "rear_right": 0.75,
    "front_right": 0.50,
    "rear_left": 0.25,
    "front_left": 0.00,
}

DISTRIBUTED_PUSH_PHASES = (
    ("weight_transfer", 0.10),
    ("lift", 0.23),
    ("swing", 0.22),
    ("lower", 0.23),
    ("firm_plant", 0.06),
    ("weight_return", 0.06),
    ("all_feet_push", 0.08),
    ("step_settle", 0.02),
)
DISTRIBUTED_PUSH_SWING_ORDER = (
    "rear_right",
    "front_right",
    "rear_left",
    "front_left",
)
COORDINATED_PUSH_PHASES = (
    ("lift", 0.22),
    ("swing_push", 0.36),
    ("lower", 0.22),
    ("touchdown", 0.10),
    ("step_settle", 0.10),
)
COORDINATED_PUSH_SWING_ORDER = (
    "front_left",
    "rear_right",
    "front_right",
    "rear_left",
)

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
        physics_variants = list(default_prim.GetVariantSet("Physics").GetVariantNames())
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


def _is_diagonal(first: str, second: str) -> bool:
    return _front_sign(first) != _front_sign(second) and _side_sign(
        first
    ) != _side_sign(second)


def smoothstep(value: float) -> float:
    value = min(max(float(value), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def leg_ik(
    leg: str,
    down_m: float,
    world_forward_m: float,
    *,
    same_bend_direction: bool = False,
    knees_outward: bool = False,
    distal_length_m: float = LINK_LENGTH_M,
) -> tuple[float, float]:
    """Solve one leg in the URDF joint frame.

    At zero, each printable arm points down along its local +X.  The generated
    URDF mirrors the physical joint axes left-to-right.  The front and rear
    pairs select opposite IK branches so the feet and knees open away from the
    center of the chassis, preserving the four-point support polygon.
    """
    if down_m <= 0.0:
        raise ValueError("Leg down target must be positive")
    front_sign = _front_sign(leg)
    sagittal_y_m = world_forward_m

    if distal_length_m <= 0.0:
        raise ValueError("Distal contact length must be positive")
    distance_sq = down_m * down_m + sagittal_y_m * sagittal_y_m
    cosine_knee = (
        distance_sq - LINK_LENGTH_M**2 - distal_length_m**2
    ) / (2.0 * LINK_LENGTH_M * distal_length_m)
    if not -1.0 <= cosine_knee <= 1.0:
        raise ValueError(
            "Unreachable leg target "
            f"{leg}: down={down_m:.6f} m, forward={world_forward_m:.6f} m, "
            f"cos(knee)={cosine_knee:.6f}"
        )

    if same_bend_direction and knees_outward:
        raise ValueError("IK branch cannot be both same-direction and knees-outward")
    if knees_outward:
        knee_sign = -front_sign
    else:
        knee_sign = 1.0 if same_bend_direction else front_sign
    knee = knee_sign * math.acos(min(max(cosine_knee, -1.0), 1.0))
    hip_flexion = math.atan2(sagittal_y_m, down_m) - math.atan2(
        distal_length_m * math.sin(knee),
        LINK_LENGTH_M + distal_length_m * math.cos(knee),
    )
    return hip_flexion, knee


def flat_sole_down_m(world_forward_m: float) -> float:
    """Return contact depth with the rectangular sole horizontal."""
    world_forward_m = float(world_forward_m)
    if abs(world_forward_m) >= LINK_LENGTH_M:
        raise ValueError(
            "Rectangular-shoe flat-sole target exceeds proximal reach: "
            f"forward={world_forward_m:.6f} m"
        )
    return (
        math.sqrt(LINK_LENGTH_M**2 - world_forward_m**2)
        + RECTANGULAR_SHOE_EFFECTIVE_DISTAL_LENGTH_M
    )


def flat_sole_leg_ik(leg: str, world_forward_m: float) -> tuple[float, float]:
    """Point the rectangular shoe normal down in the sagittal plane."""
    flat_sole_down_m(world_forward_m)
    hip_flexion = math.asin(float(world_forward_m) / LINK_LENGTH_M)
    knee = -hip_flexion
    if leg.startswith("front_") and knee > 0.0:
        raise ValueError(f"Front rectangular-shoe stance crossed the chassis: {leg}")
    if leg.startswith("rear_") and knee < 0.0:
        raise ValueError(f"Rear rectangular-shoe stance crossed the chassis: {leg}")
    return hip_flexion, knee


def pose_by_name(
    *,
    down_by_leg_m: Mapping[str, float],
    forward_by_leg_m: Mapping[str, float],
    abduction_deg: float = DEFAULT_ABDUCTION_DEG,
    abduction_by_leg_deg: Mapping[str, float] | None = None,
    same_bend_direction: bool = False,
    knees_outward: bool = False,
    distal_length_m: float = LINK_LENGTH_M,
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
            same_bend_direction=same_bend_direction,
            knees_outward=knees_outward,
            distal_length_m=distal_length_m,
        )
        result[f"{leg}_hip_abduction"] = (
            -_side_sign(leg) * math.radians(leg_abduction_deg)
        )
        result[f"{leg}_hip_flexion"] = hip_flexion
        result[f"{leg}_knee"] = knee
    return result


def _command_knees_downward(pose: Mapping[str, float]) -> dict[str, float]:
    """Map front/rear knees to their mirrored physical downward directions."""
    result: dict[str, float] = {}
    for name, value in pose.items():
        if not name.endswith("_knee"):
            result[name] = float(value)
            continue
        bend = abs(float(value))
        result[name] = bend if name.startswith("rear_") else -bend
    return result


def coordinated_push_stance_by_name(
    *,
    down_m: float,
    abduction_deg: float = 0.0,
) -> dict[str, float]:
    """Return the common-direction, approximately 45-degree ready stance."""
    return pose_by_name(
        down_by_leg_m={leg: down_m for leg in LEGS},
        forward_by_leg_m={leg: 0.0 for leg in LEGS},
        abduction_deg=abduction_deg,
        same_bend_direction=True,
    )


def outward_bent_crawl_stance_by_name(
    *,
    down_m: float,
    fore_aft_m: float,
    abduction_deg: float,
) -> dict[str, float]:
    """Return the wide stance with front/rear knee pivots opening outward."""
    return _command_knees_downward(
        pose_by_name(
            down_by_leg_m={leg: down_m for leg in LEGS},
            forward_by_leg_m={
                leg: _front_sign(leg) * fore_aft_m for leg in LEGS
            },
            abduction_deg=abduction_deg,
            knees_outward=True,
        )
    )


def coordinated_push_crawl_by_name(
    gait_time_s: float,
    *,
    period_s: float,
    stride_m: float,
    lift_m: float,
    weight_shift_forward_m: float,
    weight_shift_lateral_m: float,
    down_m: float,
    fore_aft_m: float,
    abduction_deg: float,
) -> tuple[dict[str, float], dict[str, object]]:
    """Move one swing foot and all three support feet on every step."""
    if period_s <= 0.0:
        raise ValueError("Coordinated-push period must be positive")
    if stride_m <= 0.0 or lift_m <= 0.0:
        raise ValueError("Coordinated-push stride and lift must be positive")
    if weight_shift_forward_m < 0.0 or weight_shift_lateral_m < 0.0:
        raise ValueError("Coordinated-push weight shifts must be non-negative")
    if not math.isclose(sum(value for _name, value in COORDINATED_PUSH_PHASES), 1.0):
        raise AssertionError("Coordinated-push phase fractions must total one step")

    cycle_phase = (max(0.0, gait_time_s) / period_s) % 1.0
    step_count = len(COORDINATED_PUSH_SWING_ORDER)
    step_fraction = 1.0 / step_count
    step_index = min(int(cycle_phase / step_fraction), step_count - 1)
    step_u = (cycle_phase - step_index * step_fraction) / step_fraction
    swing_leg = COORDINATED_PUSH_SWING_ORDER[step_index]

    phase_name = COORDINATED_PUSH_PHASES[-1][0]
    phase_u = 1.0
    phase_start = 0.0
    for candidate_name, candidate_fraction in COORDINATED_PUSH_PHASES:
        phase_end = phase_start + candidate_fraction
        if step_u < phase_end or candidate_name == COORDINATED_PUSH_PHASES[-1][0]:
            phase_name = candidate_name
            phase_u = min(
                max((step_u - phase_start) / candidate_fraction, 0.0),
                1.0,
            )
            break
        phase_start = phase_end

    def step_delta(leg: str, selected: str) -> float:
        if leg == selected:
            return stride_m
        if _is_diagonal(leg, selected):
            return -stride_m / 2.0
        return -stride_m / 4.0

    histories = {leg: [0.0] for leg in LEGS}
    cumulative = {leg: 0.0 for leg in LEGS}
    for selected in COORDINATED_PUSH_SWING_ORDER:
        for leg in LEGS:
            cumulative[leg] += step_delta(leg, selected)
            histories[leg].append(cumulative[leg])
    offsets = {
        leg: -(min(history) + max(history)) / 2.0 for leg, history in histories.items()
    }

    for completed_swing in COORDINATED_PUSH_SWING_ORDER[:step_index]:
        for leg in LEGS:
            offsets[leg] += step_delta(leg, completed_swing)

    motion_progress = smoothstep(phase_u) if phase_name == "swing_push" else 0.0
    if phase_name in ("lower", "touchdown", "step_settle"):
        motion_progress = 1.0
    for leg in LEGS:
        offsets[leg] += motion_progress * step_delta(leg, swing_leg)

    down_by_leg = {leg: down_m for leg in LEGS}
    if phase_name == "lift":
        down_by_leg[swing_leg] = down_m - lift_m * smoothstep(phase_u)
    elif phase_name == "swing_push":
        down_by_leg[swing_leg] = down_m - lift_m
    elif phase_name == "lower":
        down_by_leg[swing_leg] = down_m - lift_m * (1.0 - smoothstep(phase_u))

    active_support = phase_name in ("lift", "swing_push", "lower")
    transfer = 1.0 if active_support else 0.0
    body_shift_forward_m = -_front_sign(swing_leg) * weight_shift_forward_m * transfer
    body_shift_lateral_m = -_side_sign(swing_leg) * weight_shift_lateral_m * transfer
    forward_by_leg = {
        leg: (
            _front_sign(leg) * fore_aft_m
            + offsets[leg]
            - body_shift_forward_m
        )
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
        abduction_by_leg[leg] = math.degrees(math.atan2(shifted_outward, vertical))

    pose = _command_knees_downward(
        pose_by_name(
            down_by_leg_m=down_by_leg,
            forward_by_leg_m=forward_by_leg,
            abduction_by_leg_deg=abduction_by_leg,
            knees_outward=True,
        )
    )
    push_partner = next(
        leg for leg in LEGS if leg != swing_leg and _is_diagonal(leg, swing_leg)
    )
    return pose, {
        "cycle_phase": cycle_phase,
        "phase": phase_name,
        "phase_progress": phase_u,
        "swing_leg": swing_leg,
        "push_partner": push_partner,
        "expected_support_legs": [leg for leg in LEGS if leg != swing_leg],
        "foot_offsets_m": dict(offsets),
    }


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


def distributed_push_crawl_by_name(
    gait_time_s: float,
    *,
    period_s: float,
    stride_m: float,
    lift_m: float,
    support_extension_m: float,
    weight_shift_forward_m: float,
    weight_shift_lateral_m: float,
    down_m: float = DEFAULT_STANCE_DOWN_M,
    fore_aft_m: float = DEFAULT_STANCE_FORE_AFT_M,
    abduction_deg: float = DEFAULT_ABDUCTION_DEG,
) -> tuple[dict[str, float], dict[str, object]]:
    """Return a four-beat crawl with every planted rectangular sole flat."""
    if period_s <= 0.0:
        raise ValueError("Distributed-push period must be positive")
    if stride_m <= 0.0 or lift_m <= 0.0:
        raise ValueError("Distributed-push stride and lift must be positive")
    if not math.isclose(support_extension_m, 0.0, abs_tol=1e-12):
        raise ValueError(
            "Rectangular flat-sole crawl requires zero support extension"
        )
    if weight_shift_forward_m < 0.0 or weight_shift_lateral_m < 0.0:
        raise ValueError("Distributed-push weight shifts must be non-negative")
    if not math.isclose(weight_shift_lateral_m, 0.0, abs_tol=1e-12):
        raise ValueError("Flat-sole crawl requires zero lateral weight shift")
    if not math.isclose(abduction_deg, 0.0, abs_tol=1e-12):
        raise ValueError("Flat-sole crawl requires zero hip abduction")
    nominal_flat_down_m = flat_sole_down_m(fore_aft_m)
    if not math.isclose(down_m, nominal_flat_down_m, abs_tol=0.002):
        raise ValueError(
            "Flat-sole stance depth does not match its fore/aft offset: "
            f"configured={down_m:.4f} m, expected={nominal_flat_down_m:.4f} m"
        )
    if not math.isclose(sum(value for _name, value in DISTRIBUTED_PUSH_PHASES), 1.0):
        raise AssertionError("Distributed-push phase fractions do not total one step")

    cycle_phase = (max(0.0, gait_time_s) / period_s) % 1.0
    step_count = len(DISTRIBUTED_PUSH_SWING_ORDER)
    step_fraction = 1.0 / step_count
    step_index = min(int(cycle_phase / step_fraction), step_count - 1)
    step_u = (cycle_phase - step_index * step_fraction) / step_fraction
    swing_leg = DISTRIBUTED_PUSH_SWING_ORDER[step_index]

    phase_name = DISTRIBUTED_PUSH_PHASES[-1][0]
    phase_u = 1.0
    phase_start = 0.0
    for candidate_name, candidate_fraction in DISTRIBUTED_PUSH_PHASES:
        phase_end = phase_start + candidate_fraction
        if step_u < phase_end or candidate_name == DISTRIBUTED_PUSH_PHASES[-1][0]:
            phase_name = candidate_name
            phase_u = min(
                max((step_u - phase_start) / candidate_fraction, 0.0),
                1.0,
            )
            break
        phase_start = phase_end

    half_stride = stride_m / 2.0
    push_increment = stride_m / step_count
    offsets = {
        leg: half_stride - (step_count - index) * push_increment
        for index, leg in enumerate(DISTRIBUTED_PUSH_SWING_ORDER)
    }
    for completed_index in range(step_index):
        completed_swing = DISTRIBUTED_PUSH_SWING_ORDER[completed_index]
        offsets[completed_swing] += stride_m
        for leg in LEGS:
            offsets[leg] -= push_increment

    if phase_name == "swing":
        offsets[swing_leg] += stride_m * smoothstep(phase_u)
    elif phase_name in (
        "lower",
        "firm_plant",
        "weight_return",
        "all_feet_push",
        "step_settle",
    ):
        offsets[swing_leg] += stride_m
    if phase_name == "all_feet_push":
        push = push_increment * smoothstep(phase_u)
        offsets = {leg: offset - push for leg, offset in offsets.items()}
    elif phase_name == "step_settle":
        offsets = {leg: offset - push_increment for leg, offset in offsets.items()}

    if phase_name == "weight_transfer":
        transfer = smoothstep(phase_u)
    elif phase_name == "weight_return":
        transfer = 1.0 - smoothstep(phase_u)
    elif phase_name in ("all_feet_push", "step_settle"):
        transfer = 0.0
    else:
        transfer = 1.0
    body_shift_forward_m = -_front_sign(swing_leg) * weight_shift_forward_m * transfer
    body_shift_lateral_m = -_side_sign(swing_leg) * weight_shift_lateral_m * transfer

    forward_by_leg = {
        leg: _front_sign(leg) * fore_aft_m + offsets[leg] - body_shift_forward_m
        for leg in LEGS
    }
    flat_down_by_leg = {
        leg: flat_sole_down_m(forward_by_leg[leg]) for leg in LEGS
    }
    down_by_leg = dict(flat_down_by_leg)
    swing_is_airborne = phase_name in (
        "lift",
        "swing",
        "lower",
    )
    if phase_name == "lift":
        down_by_leg[swing_leg] -= lift_m * smoothstep(phase_u)
    elif phase_name == "swing":
        down_by_leg[swing_leg] -= lift_m
    elif phase_name == "lower":
        down_by_leg[swing_leg] -= lift_m * (1.0 - smoothstep(phase_u))
    nominal_abduction = math.radians(abduction_deg)
    foot_delta_lateral_m = -body_shift_lateral_m
    abduction_by_leg: dict[str, float] = {}
    for leg in LEGS:
        nominal_leg_down = down_by_leg[leg]
        vertical = nominal_leg_down * math.cos(nominal_abduction)
        outward = nominal_leg_down * math.sin(nominal_abduction)
        shifted_outward = outward + _side_sign(leg) * foot_delta_lateral_m
        down_by_leg[leg] = math.hypot(vertical, shifted_outward)
        abduction_by_leg[leg] = math.degrees(math.atan2(shifted_outward, vertical))

    pose: dict[str, float] = {}
    sole_pitch_by_leg_deg: dict[str, float] = {}
    shoe_edge_clearance_by_leg_m: dict[str, float] = {}
    for leg in LEGS:
        if leg == swing_leg and swing_is_airborne:
            hip_flexion, knee = leg_ik(
                leg,
                down_by_leg[leg],
                forward_by_leg[leg],
                knees_outward=True,
                distal_length_m=RECTANGULAR_SHOE_EFFECTIVE_DISTAL_LENGTH_M,
            )
        else:
            hip_flexion, knee = flat_sole_leg_ik(leg, forward_by_leg[leg])
        pose[f"{leg}_hip_abduction"] = (
            -_side_sign(leg) * math.radians(abduction_by_leg[leg])
        )
        pose[f"{leg}_hip_flexion"] = hip_flexion
        pose[f"{leg}_knee"] = knee
        sole_pitch_rad = hip_flexion + knee
        sole_pitch_by_leg_deg[leg] = math.degrees(sole_pitch_rad)
        contact_center_lift_m = max(
            flat_down_by_leg[leg] - down_by_leg[leg],
            0.0,
        )
        shoe_edge_clearance_by_leg_m[leg] = max(
            contact_center_lift_m
            - RECTANGULAR_SHOE_HALF_FORE_AFT_M * abs(math.sin(sole_pitch_rad)),
            0.0,
        )
    pose = _command_knees_downward(pose)
    expected_support_legs = (
        [leg for leg in LEGS if leg != swing_leg]
        if swing_is_airborne
        else list(LEGS)
    )
    return pose, {
        "cycle_phase": cycle_phase,
        "phase": phase_name,
        "phase_progress": phase_u,
        "swing_leg": swing_leg,
        "expected_support_legs": expected_support_legs,
        "foot_offsets_m": dict(offsets),
        "planted_forward_m": dict(forward_by_leg),
        "flat_sole_support": True,
        "flat_sole_nominal_support": True,
        "all_planted_soles_flat": True,
        "support_extension_holds_contact_x": False,
        "flat_support_down_m": dict(flat_down_by_leg),
        "sole_pitch_deg": sole_pitch_by_leg_deg,
        "shoe_edge_clearance_m": shoe_edge_clearance_by_leg_m,
        "nominal_stance_down_m": down_m,
        "support_extension_m": 0.0,
        "support_extension_progress": 0.0,
        "active_support_extension_m": 0.0,
    }


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
        phase_progress = (cycle_phase - step_region) / QUASISTATIC_ADVANCE_FRACTION
        remaining = 1.0 - smoothstep(phase_progress)
        offsets = {leg: stride_m * remaining for leg in LEGS}
    else:
        phase_progress = (
            cycle_phase - step_region - QUASISTATIC_ADVANCE_FRACTION
        ) / QUASISTATIC_SETTLE_FRACTION

    forward_by_leg = {
        leg: _front_sign(leg) * fore_aft_m + offsets[leg] - body_shift_forward_m
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
        abduction_by_leg[leg] = math.degrees(math.atan2(shifted_outward, vertical))
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
