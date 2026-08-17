"""Deterministic foot-space crawl targets for the four-leg dashboard.

The equations intentionally mirror ``simulation.isaac._quadruped_runtime`` so
the exact browser-driven gait can be checked against the Isaac robot.  Values
are returned in calibrated motor degrees and require no simulator at runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

# Both printed arm segments reach 159.896689 mm between servo axes.  The new
# rectangular shoe has its PLA face 30 mm beyond the distal fork axis and uses
# a 1 mm bonded tread.  Its 100 mm fore/aft length matters whenever an airborne
# shoe pitches, because a corner sits below the contact-face centre.
PROXIMAL_LINK_LENGTH_M = 0.159896689
DISTAL_FORK_AXIS_LENGTH_M = 0.159896689
RECTANGULAR_SHOE_CONTACT_EXTENSION_M = 0.031
RECTANGULAR_SHOE_HALF_FORE_AFT_M = 0.050
DISTAL_CONTACT_LENGTH_M = (
    DISTAL_FORK_AXIS_LENGTH_M + RECTANGULAR_SHOE_CONTACT_EXTENSION_M
)
LEG_CORNERS = ("front_left", "front_right", "rear_left", "rear_right")
JOINT_NAMES = ("hip_abduction", "hip_flexion", "knee")
SWING_ORDER = ("rear_right", "front_right", "rear_left", "front_left")
STEP_FRACTION = 0.22
ADVANCE_FRACTION = 0.10
SETTLE_FRACTION = 0.02
STEP_PHASES = (
    ("weight_transfer", 0.20),
    ("lift", 0.16),
    ("swing", 0.28),
    ("lower", 0.16),
    ("touchdown", 0.10),
    ("weight_return", 0.10),
)
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
DIAGONAL_PAIR_PHASES = (
    ("lift_pair", 0.25),
    ("swing_pair", 0.25),
    ("lower_pair", 0.25),
    ("firm_plant", 0.10),
    ("all_feet_push", 0.12),
    ("pair_settle", 0.03),
)
DIAGONAL_PAIR_SWING_ORDER = (
    ("front_left", "rear_right"),
    ("front_right", "rear_left"),
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
HARDWARE_SEQUENCE_REPETITIONS = 2
HARDWARE_SEQUENCE_PHASES = (
    "leg_1_hip_flexion_to_90",
    "leg_2_hip_flexion_to_70",
    "leg_4_knee_to_20",
    "return_to_stance",
)


def _front_sign(corner: str) -> float:
    return 1.0 if corner.startswith("front_") else -1.0


def _side_sign(corner: str) -> float:
    return 1.0 if corner.endswith("_left") else -1.0


def _smoothstep(value: float) -> float:
    value = min(max(float(value), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _is_diagonal(first: str, second: str) -> bool:
    return _front_sign(first) != _front_sign(second) and _side_sign(
        first
    ) != _side_sign(second)


def _leg_ik(
    corner: str,
    down_m: float,
    forward_m: float,
    *,
    same_bend_direction: bool = False,
    knees_outward: bool = False,
) -> tuple[float, float]:
    distance_sq = down_m * down_m + forward_m * forward_m
    cosine_knee = (
        distance_sq
        - PROXIMAL_LINK_LENGTH_M**2
        - DISTAL_CONTACT_LENGTH_M**2
    ) / (2.0 * PROXIMAL_LINK_LENGTH_M * DISTAL_CONTACT_LENGTH_M)
    if not -1.0 <= cosine_knee <= 1.0:
        raise ValueError(
            f"Unreachable crawl target for {corner}: "
            f"down={down_m:.4f} m, forward={forward_m:.4f} m"
        )
    if same_bend_direction and knees_outward:
        raise ValueError("IK branch cannot be both same-direction and knees-outward")
    if knees_outward:
        knee_sign = -_front_sign(corner)
    else:
        knee_sign = 1.0 if same_bend_direction else _front_sign(corner)
    knee = knee_sign * math.acos(cosine_knee)
    hip_flexion = math.atan2(forward_m, down_m) - math.atan2(
        DISTAL_CONTACT_LENGTH_M * math.sin(knee),
        PROXIMAL_LINK_LENGTH_M + DISTAL_CONTACT_LENGTH_M * math.cos(knee),
    )
    return hip_flexion, knee


def _flat_sole_down_m(forward_m: float) -> float:
    """Return contact depth when the rectangular sole is horizontal.

    A flat sole requires the distal link to point vertically down.  With only
    hip flexion and knee pitch, horizontal contact position then determines
    the vertical contact position rather than leaving both independently
    commandable.
    """
    forward_m = float(forward_m)
    if abs(forward_m) >= PROXIMAL_LINK_LENGTH_M:
        raise ValueError(
            "Rectangular-shoe flat-sole target exceeds proximal reach: "
            f"forward={forward_m:.4f} m"
        )
    proximal_down_m = math.sqrt(PROXIMAL_LINK_LENGTH_M**2 - forward_m**2)
    return proximal_down_m + DISTAL_CONTACT_LENGTH_M


def _flat_sole_leg_ik(corner: str, forward_m: float) -> tuple[float, float]:
    """Point the rectangular shoe's contact-face normal down."""
    _flat_sole_down_m(forward_m)
    hip_flexion = math.asin(float(forward_m) / PROXIMAL_LINK_LENGTH_M)
    knee = -hip_flexion
    if corner.startswith("front_") and knee > 0.0:
        raise ValueError(f"Front rectangular-shoe stance crossed the chassis: {corner}")
    if corner.startswith("rear_") and knee < 0.0:
        raise ValueError(f"Rear rectangular-shoe stance crossed the chassis: {corner}")
    return hip_flexion, knee


def _pose_degrees(
    down_by_corner_m: Mapping[str, float],
    forward_by_corner_m: Mapping[str, float],
    abduction_by_corner_deg: Mapping[str, float],
    *,
    same_bend_direction: bool = False,
    knees_outward: bool = False,
) -> dict[tuple[str, str], float]:
    pose: dict[tuple[str, str], float] = {}
    for corner in LEG_CORNERS:
        hip_flexion, knee = _leg_ik(
            corner,
            float(down_by_corner_m[corner]),
            float(forward_by_corner_m[corner]),
            same_bend_direction=same_bend_direction,
            knees_outward=knees_outward,
        )
        # The assembled left hip axes use the opposite logical sign from the
        # right axes: outward is negative on Legs 1/3 and positive on 2/4.
        pose[(corner, "hip_abduction")] = (
            -_side_sign(corner) * float(abduction_by_corner_deg[corner])
        )
        pose[(corner, "hip_flexion")] = math.degrees(hip_flexion)
        pose[(corner, "knee")] = math.degrees(knee)
    return pose


def _command_knees_downward(
    pose: Mapping[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """Map front/rear knees to their mirrored physical downward directions."""
    return {
        key: (
            abs(float(value))
            if key[1] == "knee" and key[0].startswith("rear_")
            else -abs(float(value))
            if key[1] == "knee"
            else float(value)
        )
        for key, value in pose.items()
    }


def coordinated_push_stance_degrees(
    *,
    down_m: float,
    abduction_deg: float,
) -> dict[tuple[str, str], float]:
    """Return the common-direction, approximately 45-degree ready stance."""
    return _pose_degrees(
        {corner: down_m for corner in LEG_CORNERS},
        {corner: 0.0 for corner in LEG_CORNERS},
        {corner: abduction_deg for corner in LEG_CORNERS},
        same_bend_direction=True,
    )


def outward_bent_crawl_stance_degrees(
    *,
    down_m: float,
    fore_aft_m: float,
    abduction_deg: float,
) -> dict[tuple[str, str], float]:
    """Return a wide stance with front and rear knee pivots opening outward."""
    return _command_knees_downward(
        _pose_degrees(
            {corner: down_m for corner in LEG_CORNERS},
            {
                corner: _front_sign(corner) * fore_aft_m
                for corner in LEG_CORNERS
            },
            {corner: abduction_deg for corner in LEG_CORNERS},
            knees_outward=True,
        )
    )


def hardware_joint_sequence_degrees(
    gait_time_s: float,
    *,
    period_s: float,
    down_m: float,
    fore_aft_m: float,
    abduction_deg: float,
) -> tuple[dict[tuple[str, str], float], dict[str, object]]:
    """Run the first hardware-derived gait sequence twice from wide stance."""
    if period_s <= 0.0:
        raise ValueError("period_s must be positive")

    stance = outward_bent_crawl_stance_degrees(
        down_m=down_m,
        fore_aft_m=fore_aft_m,
        abduction_deg=abduction_deg,
    )
    leg_1 = dict(stance)
    leg_1[("front_left", "hip_flexion")] = 90.0
    leg_2 = dict(leg_1)
    leg_2[("front_right", "hip_flexion")] = 70.0
    leg_4 = dict(leg_2)
    leg_4[("rear_right", "knee")] = 20.0

    starts = (stance, leg_1, leg_2, leg_4)
    ends = (leg_1, leg_2, leg_4, stance)
    phase_count = len(HARDWARE_SEQUENCE_PHASES)
    total_segments = phase_count * HARDWARE_SEQUENCE_REPETITIONS
    sequence_phase = (max(0.0, float(gait_time_s)) % period_s) / period_s
    segment_position = sequence_phase * total_segments
    segment_index = min(int(segment_position), total_segments - 1)
    phase_index = segment_index % phase_count
    repetition = segment_index // phase_count
    phase_progress = segment_position - segment_index
    blend = _smoothstep(phase_progress)
    start = starts[phase_index]
    end = ends[phase_index]
    pose = {
        key: float(start[key]) + (float(end[key]) - float(start[key])) * blend
        for key in stance
    }
    return pose, {
        "cycle_phase": sequence_phase,
        "phase": (
            f"pass_{repetition + 1}_"
            f"{HARDWARE_SEQUENCE_PHASES[phase_index]}"
        ),
        "phase_progress": phase_progress,
        "sequence_pass": repetition + 1,
        "swing_corner": None,
        "push_partner": None,
    }


def coordinated_push_crawl_degrees(
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
) -> tuple[dict[tuple[str, str], float], dict[str, object]]:
    """Move one swing foot and all three support feet on every step.

    Each planted leg moves rearward while the selected foot moves forward.
    The diagonal support leg contributes half of the push and each adjacent
    support leg contributes one quarter.  Across four steps, every foot moves
    forward by one stride while airborne and rearward by one stride while
    planted, making the targets continuous and periodic.
    """
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
    swing_corner = COORDINATED_PUSH_SWING_ORDER[step_index]

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

    def step_delta(corner: str, selected: str) -> float:
        if corner == selected:
            return stride_m
        if _is_diagonal(corner, selected):
            return -stride_m / 2.0
        return -stride_m / 4.0

    # Center every leg's complete trajectory around zero.  With the selected
    # front-left-first order this keeps both rear feet planted for the first
    # lift, avoiding the unsupported rear edge created by V3's first move.
    histories = {corner: [0.0] for corner in LEG_CORNERS}
    cumulative = {corner: 0.0 for corner in LEG_CORNERS}
    for selected in COORDINATED_PUSH_SWING_ORDER:
        for corner in LEG_CORNERS:
            cumulative[corner] += step_delta(corner, selected)
            histories[corner].append(cumulative[corner])
    offsets = {
        corner: -(min(history) + max(history)) / 2.0
        for corner, history in histories.items()
    }

    for completed_swing in COORDINATED_PUSH_SWING_ORDER[:step_index]:
        for corner in LEG_CORNERS:
            offsets[corner] += step_delta(corner, completed_swing)

    motion_progress = _smoothstep(phase_u) if phase_name == "swing_push" else 0.0
    if phase_name in ("lower", "touchdown", "step_settle"):
        motion_progress = 1.0
    for corner in LEG_CORNERS:
        offsets[corner] += motion_progress * step_delta(corner, swing_corner)

    down_by_corner = {corner: down_m for corner in LEG_CORNERS}
    if phase_name == "lift":
        down_by_corner[swing_corner] = down_m - lift_m * _smoothstep(phase_u)
    elif phase_name == "swing_push":
        down_by_corner[swing_corner] = down_m - lift_m
    elif phase_name == "lower":
        down_by_corner[swing_corner] = down_m - lift_m * (1.0 - _smoothstep(phase_u))

    active_support = phase_name in ("lift", "swing_push", "lower")
    transfer = 1.0 if active_support else 0.0
    body_shift_forward_m = (
        -_front_sign(swing_corner) * weight_shift_forward_m * transfer
    )
    body_shift_lateral_m = -_side_sign(swing_corner) * weight_shift_lateral_m * transfer
    forward_by_corner = {
        corner: (
            _front_sign(corner) * fore_aft_m
            + offsets[corner]
            - body_shift_forward_m
        )
        for corner in LEG_CORNERS
    }

    nominal_abduction = math.radians(abduction_deg)
    foot_delta_lateral_m = -body_shift_lateral_m
    abduction_by_corner: dict[str, float] = {}
    for corner in LEG_CORNERS:
        nominal_leg_down = down_by_corner[corner]
        vertical = nominal_leg_down * math.cos(nominal_abduction)
        outward = nominal_leg_down * math.sin(nominal_abduction)
        shifted_outward = outward + _side_sign(corner) * foot_delta_lateral_m
        down_by_corner[corner] = math.hypot(vertical, shifted_outward)
        abduction_by_corner[corner] = math.degrees(
            math.atan2(shifted_outward, vertical)
        )

    pose = _command_knees_downward(
        _pose_degrees(
            down_by_corner,
            forward_by_corner,
            abduction_by_corner,
            knees_outward=True,
        )
    )
    push_partner = next(
        corner
        for corner in LEG_CORNERS
        if corner != swing_corner and _is_diagonal(corner, swing_corner)
    )
    return pose, {
        "cycle_phase": cycle_phase,
        "phase": phase_name,
        "phase_progress": phase_u,
        "swing_corner": swing_corner,
        "push_partner": push_partner,
        "expected_support_corners": [
            corner for corner in LEG_CORNERS if corner != swing_corner
        ],
        "foot_offsets_m": dict(offsets),
    }


def distributed_push_crawl_degrees(
    gait_time_s: float,
    *,
    period_s: float,
    stride_m: float,
    lift_m: float,
    support_extension_m: float,
    weight_shift_forward_m: float,
    weight_shift_lateral_m: float,
    down_m: float,
    fore_aft_m: float,
    abduction_deg: float,
) -> tuple[dict[tuple[str, str], float], dict[str, object]]:
    """Return a rectangular-shoe crawl with every planted sole flat.

    The three support legs use the exact flat-sole branch throughout weight
    transfer, swing, touchdown, and propulsion.  Only the unloaded swing leg
    uses general two-link IK, because two pitch joints cannot independently
    command its X, Z, and sole pitch.  It rejoins the flat branch before load.
    """
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
    nominal_flat_down_m = _flat_sole_down_m(fore_aft_m)
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
    swing_corner = DISTRIBUTED_PUSH_SWING_ORDER[step_index]

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
        corner: half_stride - (step_count - index) * push_increment
        for index, corner in enumerate(DISTRIBUTED_PUSH_SWING_ORDER)
    }
    for completed_index in range(step_index):
        completed_swing = DISTRIBUTED_PUSH_SWING_ORDER[completed_index]
        offsets[completed_swing] += stride_m
        for corner in LEG_CORNERS:
            offsets[corner] -= push_increment

    if phase_name == "swing":
        offsets[swing_corner] += stride_m * _smoothstep(phase_u)
    elif phase_name in (
        "lower",
        "firm_plant",
        "weight_return",
        "all_feet_push",
        "step_settle",
    ):
        offsets[swing_corner] += stride_m
    if phase_name == "all_feet_push":
        push = push_increment * _smoothstep(phase_u)
        offsets = {corner: offset - push for corner, offset in offsets.items()}
    elif phase_name == "step_settle":
        offsets = {
            corner: offset - push_increment for corner, offset in offsets.items()
        }

    if phase_name == "weight_transfer":
        transfer = _smoothstep(phase_u)
    elif phase_name == "weight_return":
        transfer = 1.0 - _smoothstep(phase_u)
    elif phase_name in ("all_feet_push", "step_settle"):
        transfer = 0.0
    else:
        transfer = 1.0
    body_shift_forward_m = (
        -_front_sign(swing_corner) * weight_shift_forward_m * transfer
    )
    body_shift_lateral_m = -_side_sign(swing_corner) * weight_shift_lateral_m * transfer

    forward_by_corner = {
        corner: _front_sign(corner) * fore_aft_m
        + offsets[corner]
        - body_shift_forward_m
        for corner in LEG_CORNERS
    }
    flat_down_by_corner = {
        corner: _flat_sole_down_m(forward_by_corner[corner])
        for corner in LEG_CORNERS
    }
    down_by_corner = dict(flat_down_by_corner)
    swing_is_airborne = phase_name in (
        "lift",
        "swing",
        "lower",
    )
    if phase_name == "lift":
        down_by_corner[swing_corner] -= lift_m * _smoothstep(phase_u)
    elif phase_name == "swing":
        down_by_corner[swing_corner] -= lift_m
    elif phase_name == "lower":
        down_by_corner[swing_corner] -= lift_m * (1.0 - _smoothstep(phase_u))
    nominal_abduction = math.radians(abduction_deg)
    foot_delta_lateral_m = -body_shift_lateral_m
    abduction_by_corner: dict[str, float] = {}
    for corner in LEG_CORNERS:
        nominal_leg_down = down_by_corner[corner]
        vertical = nominal_leg_down * math.cos(nominal_abduction)
        outward = nominal_leg_down * math.sin(nominal_abduction)
        shifted_outward = outward + _side_sign(corner) * foot_delta_lateral_m
        down_by_corner[corner] = math.hypot(vertical, shifted_outward)
        abduction_by_corner[corner] = math.degrees(
            math.atan2(shifted_outward, vertical)
        )

    pose: dict[tuple[str, str], float] = {}
    sole_pitch_by_corner_deg: dict[str, float] = {}
    shoe_edge_clearance_by_corner_m: dict[str, float] = {}
    for corner in LEG_CORNERS:
        if corner == swing_corner and swing_is_airborne:
            hip_flexion, knee = _leg_ik(
                corner,
                down_by_corner[corner],
                forward_by_corner[corner],
                knees_outward=True,
            )
        else:
            hip_flexion, knee = _flat_sole_leg_ik(
                corner,
                forward_by_corner[corner],
            )
        pose[(corner, "hip_abduction")] = (
            -_side_sign(corner) * abduction_by_corner[corner]
        )
        pose[(corner, "hip_flexion")] = math.degrees(hip_flexion)
        pose[(corner, "knee")] = math.degrees(knee)
        sole_pitch_rad = hip_flexion + knee
        sole_pitch_by_corner_deg[corner] = math.degrees(sole_pitch_rad)
        contact_center_lift_m = max(
            flat_down_by_corner[corner] - down_by_corner[corner],
            0.0,
        )
        shoe_edge_clearance_by_corner_m[corner] = max(
            contact_center_lift_m
            - RECTANGULAR_SHOE_HALF_FORE_AFT_M * abs(math.sin(sole_pitch_rad)),
            0.0,
        )
    pose = _command_knees_downward(pose)
    return pose, {
        "cycle_phase": cycle_phase,
        "phase": phase_name,
        "phase_progress": phase_u,
        "swing_corner": swing_corner,
        "expected_support_corners": (
            [corner for corner in LEG_CORNERS if corner != swing_corner]
            if swing_is_airborne
            else list(LEG_CORNERS)
        ),
        "foot_offsets_m": dict(offsets),
        "planted_forward_m": dict(forward_by_corner),
        "flat_sole_support": True,
        "flat_sole_nominal_support": True,
        "all_planted_soles_flat": True,
        "support_extension_holds_contact_x": False,
        "flat_support_down_m": dict(flat_down_by_corner),
        "sole_pitch_deg": sole_pitch_by_corner_deg,
        "shoe_edge_clearance_m": shoe_edge_clearance_by_corner_m,
        "nominal_stance_down_m": down_m,
        "support_extension_m": 0.0,
        "support_extension_progress": 0.0,
        "active_support_extension_m": 0.0,
    }


def diagonal_pair_gait_degrees(
    gait_time_s: float,
    *,
    period_s: float,
    stride_m: float,
    lift_m: float,
    support_extension_m: float,
    down_m: float,
    fore_aft_m: float,
    abduction_deg: float,
) -> tuple[dict[tuple[str, str], float], dict[str, object]]:
    """Return a two-beat gait that swings one diagonal leg pair at a time.

    Front-left moves with rear-right, followed by front-right with rear-left.
    The two planted diagonal shoes remain on the exact flat-sole branch.  Once
    the moving pair is planted, all four targets push rearward by half a stride
    so the two-pair cycle ends at its starting joint targets.
    """
    if period_s <= 0.0:
        raise ValueError("Diagonal-pair period must be positive")
    if stride_m <= 0.0 or lift_m <= 0.0:
        raise ValueError("Diagonal-pair stride and lift must be positive")
    if not math.isclose(support_extension_m, 0.0, abs_tol=1e-12):
        raise ValueError("Diagonal-pair flat support requires zero extension")
    if not math.isclose(abduction_deg, 0.0, abs_tol=1e-12):
        raise ValueError("Diagonal-pair flat support requires zero hip abduction")
    nominal_flat_down_m = _flat_sole_down_m(fore_aft_m)
    if not math.isclose(down_m, nominal_flat_down_m, abs_tol=0.002):
        raise ValueError(
            "Flat-sole stance depth does not match its fore/aft offset: "
            f"configured={down_m:.4f} m, expected={nominal_flat_down_m:.4f} m"
        )
    if not math.isclose(sum(value for _name, value in DIAGONAL_PAIR_PHASES), 1.0):
        raise AssertionError("Diagonal-pair phase fractions do not total one step")

    cycle_phase = (max(0.0, gait_time_s) / period_s) % 1.0
    pair_count = len(DIAGONAL_PAIR_SWING_ORDER)
    pair_fraction = 1.0 / pair_count
    pair_index = min(int(cycle_phase / pair_fraction), pair_count - 1)
    pair_u = (cycle_phase - pair_index * pair_fraction) / pair_fraction
    swing_pair = DIAGONAL_PAIR_SWING_ORDER[pair_index]

    phase_name = DIAGONAL_PAIR_PHASES[-1][0]
    phase_u = 1.0
    phase_start = 0.0
    for candidate_name, candidate_fraction in DIAGONAL_PAIR_PHASES:
        phase_end = phase_start + candidate_fraction
        if pair_u < phase_end or candidate_name == DIAGONAL_PAIR_PHASES[-1][0]:
            phase_name = candidate_name
            phase_u = min(
                max((pair_u - phase_start) / candidate_fraction, 0.0),
                1.0,
            )
            break
        phase_start = phase_end

    half_stride = stride_m / 2.0
    push_increment = stride_m / pair_count
    offsets: dict[str, float] = {}
    for index, pair in enumerate(DIAGONAL_PAIR_SWING_ORDER):
        pair_offset = half_stride - (pair_count - index) * push_increment
        for corner in pair:
            offsets[corner] = pair_offset

    for completed_index in range(pair_index):
        for corner in DIAGONAL_PAIR_SWING_ORDER[completed_index]:
            offsets[corner] += stride_m
        for corner in LEG_CORNERS:
            offsets[corner] -= push_increment

    if phase_name == "swing_pair":
        advance = stride_m * _smoothstep(phase_u)
        for corner in swing_pair:
            offsets[corner] += advance
    elif phase_name in (
        "lower_pair",
        "firm_plant",
        "all_feet_push",
        "pair_settle",
    ):
        for corner in swing_pair:
            offsets[corner] += stride_m

    if phase_name == "all_feet_push":
        push = push_increment * _smoothstep(phase_u)
        offsets = {corner: offset - push for corner, offset in offsets.items()}
    elif phase_name == "pair_settle":
        offsets = {
            corner: offset - push_increment for corner, offset in offsets.items()
        }

    forward_by_corner = {
        corner: _front_sign(corner) * fore_aft_m + offsets[corner]
        for corner in LEG_CORNERS
    }
    flat_down_by_corner = {
        corner: _flat_sole_down_m(forward_by_corner[corner])
        for corner in LEG_CORNERS
    }
    down_by_corner = dict(flat_down_by_corner)
    pair_is_airborne = phase_name in ("lift_pair", "swing_pair", "lower_pair")
    if phase_name == "lift_pair":
        lift = lift_m * _smoothstep(phase_u)
    elif phase_name == "swing_pair":
        lift = lift_m
    elif phase_name == "lower_pair":
        lift = lift_m * (1.0 - _smoothstep(phase_u))
    else:
        lift = 0.0
    if pair_is_airborne:
        for corner in swing_pair:
            down_by_corner[corner] -= lift

    pose: dict[tuple[str, str], float] = {}
    sole_pitch_by_corner_deg: dict[str, float] = {}
    shoe_edge_clearance_by_corner_m: dict[str, float] = {}
    for corner in LEG_CORNERS:
        if corner in swing_pair and pair_is_airborne:
            hip_flexion, knee = _leg_ik(
                corner,
                down_by_corner[corner],
                forward_by_corner[corner],
                knees_outward=True,
            )
        else:
            hip_flexion, knee = _flat_sole_leg_ik(
                corner,
                forward_by_corner[corner],
            )
        pose[(corner, "hip_abduction")] = 0.0
        pose[(corner, "hip_flexion")] = math.degrees(hip_flexion)
        pose[(corner, "knee")] = math.degrees(knee)
        sole_pitch_rad = hip_flexion + knee
        sole_pitch_by_corner_deg[corner] = math.degrees(sole_pitch_rad)
        contact_center_lift_m = max(
            flat_down_by_corner[corner] - down_by_corner[corner],
            0.0,
        )
        shoe_edge_clearance_by_corner_m[corner] = max(
            contact_center_lift_m
            - RECTANGULAR_SHOE_HALF_FORE_AFT_M * abs(math.sin(sole_pitch_rad)),
            0.0,
        )

    pose = _command_knees_downward(pose)
    expected_support = (
        [corner for corner in LEG_CORNERS if corner not in swing_pair]
        if pair_is_airborne
        else list(LEG_CORNERS)
    )
    return pose, {
        "cycle_phase": cycle_phase,
        "phase": phase_name,
        "phase_progress": phase_u,
        "swing_corner": None,
        "swing_pair": list(swing_pair),
        "expected_support_corners": expected_support,
        "foot_offsets_m": dict(offsets),
        "planted_forward_m": dict(forward_by_corner),
        "flat_sole_support": True,
        "flat_sole_nominal_support": True,
        "all_planted_soles_flat": True,
        "support_extension_holds_contact_x": False,
        "flat_support_down_m": dict(flat_down_by_corner),
        "sole_pitch_deg": sole_pitch_by_corner_deg,
        "shoe_edge_clearance_m": shoe_edge_clearance_by_corner_m,
        "nominal_stance_down_m": down_m,
        "support_extension_m": 0.0,
        "support_extension_progress": 0.0,
        "active_support_extension_m": 0.0,
    }


def quasistatic_crawl_degrees(
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
) -> tuple[dict[tuple[str, str], float], dict[str, object]]:
    """Return one continuous, deterministic four-beat crawl pose."""

    if period_s <= 0.0:
        raise ValueError("Crawl period must be positive")
    if stride_m <= 0.0 or lift_m <= 0.0:
        raise ValueError("Crawl stride and lift must be positive")
    if weight_shift_forward_m < 0.0 or weight_shift_lateral_m < 0.0:
        raise ValueError("Crawl weight shifts must be non-negative")

    cycle_phase = (gait_time_s / period_s) % 1.0
    step_region = len(SWING_ORDER) * STEP_FRACTION
    offsets = {corner: 0.0 for corner in LEG_CORNERS}
    down_by_corner = {corner: down_m for corner in LEG_CORNERS}
    body_shift_forward_m = 0.0
    body_shift_lateral_m = 0.0
    swing_corner: str | None = None
    phase_name = "cycle_settle"
    phase_progress = 0.0

    if cycle_phase < step_region:
        step_index = min(
            int(cycle_phase / STEP_FRACTION),
            len(SWING_ORDER) - 1,
        )
        swing_corner = SWING_ORDER[step_index]
        for completed_corner in SWING_ORDER[:step_index]:
            offsets[completed_corner] = stride_m

        step_start = step_index * STEP_FRACTION
        step_u = (cycle_phase - step_start) / STEP_FRACTION
        phase_start = 0.0
        phase_u = 0.0
        for candidate_name, candidate_fraction in STEP_PHASES:
            phase_end = phase_start + candidate_fraction
            if step_u < phase_end or candidate_name == STEP_PHASES[-1][0]:
                phase_name = candidate_name
                phase_u = min(
                    max((step_u - phase_start) / candidate_fraction, 0.0),
                    1.0,
                )
                break
            phase_start = phase_end
        phase_progress = phase_u

        if phase_name == "weight_transfer":
            transfer = _smoothstep(phase_u)
        elif phase_name == "weight_return":
            transfer = 1.0 - _smoothstep(phase_u)
        else:
            transfer = 1.0
        body_shift_forward_m = (
            -_front_sign(swing_corner) * weight_shift_forward_m * transfer
        )
        body_shift_lateral_m = (
            -_side_sign(swing_corner) * weight_shift_lateral_m * transfer
        )

        if phase_name == "lift":
            down_by_corner[swing_corner] = down_m - lift_m * _smoothstep(phase_u)
        elif phase_name == "swing":
            down_by_corner[swing_corner] = down_m - lift_m
            offsets[swing_corner] = stride_m * _smoothstep(phase_u)
        elif phase_name == "lower":
            down_by_corner[swing_corner] = down_m - lift_m * (
                1.0 - _smoothstep(phase_u)
            )
            offsets[swing_corner] = stride_m
        elif phase_name in ("touchdown", "weight_return"):
            offsets[swing_corner] = stride_m
    elif cycle_phase < step_region + ADVANCE_FRACTION:
        phase_name = "all_feet_advance"
        phase_progress = (cycle_phase - step_region) / ADVANCE_FRACTION
        remaining = 1.0 - _smoothstep(phase_progress)
        offsets = {corner: stride_m * remaining for corner in LEG_CORNERS}
    else:
        phase_progress = (
            cycle_phase - step_region - ADVANCE_FRACTION
        ) / SETTLE_FRACTION

    forward_by_corner = {
        corner: _front_sign(corner) * fore_aft_m
        + offsets[corner]
        - body_shift_forward_m
        for corner in LEG_CORNERS
    }
    nominal_abduction = math.radians(abduction_deg)
    foot_delta_lateral_m = -body_shift_lateral_m
    abduction_by_corner: dict[str, float] = {}
    for corner in LEG_CORNERS:
        nominal_leg_down = down_by_corner[corner]
        vertical = nominal_leg_down * math.cos(nominal_abduction)
        outward = nominal_leg_down * math.sin(nominal_abduction)
        shifted_outward = outward + _side_sign(corner) * foot_delta_lateral_m
        down_by_corner[corner] = math.hypot(vertical, shifted_outward)
        abduction_by_corner[corner] = math.degrees(
            math.atan2(shifted_outward, vertical)
        )

    pose = _pose_degrees(
        down_by_corner,
        forward_by_corner,
        abduction_by_corner,
    )
    return pose, {
        "cycle_phase": cycle_phase,
        "phase": phase_name,
        "phase_progress": min(max(phase_progress, 0.0), 1.0),
        "swing_corner": swing_corner,
    }
