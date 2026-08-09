"""Deterministic quasi-static crawl targets for the four-leg dashboard.

The equations intentionally mirror ``simulation.isaac._quadruped_runtime`` so
the exact browser-driven gait can be checked against the Isaac robot.  Values
are returned in calibrated motor degrees and require no simulator at runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

LINK_LENGTH_M = 0.159896689
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


def _front_sign(corner: str) -> float:
    return 1.0 if corner.startswith("front_") else -1.0


def _side_sign(corner: str) -> float:
    return 1.0 if corner.endswith("_left") else -1.0


def _smoothstep(value: float) -> float:
    value = min(max(float(value), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _leg_ik(corner: str, down_m: float, forward_m: float) -> tuple[float, float]:
    distance_sq = down_m * down_m + forward_m * forward_m
    cosine_knee = (distance_sq - 2.0 * LINK_LENGTH_M**2) / (2.0 * LINK_LENGTH_M**2)
    if not -1.0 <= cosine_knee <= 1.0:
        raise ValueError(
            f"Unreachable crawl target for {corner}: "
            f"down={down_m:.4f} m, forward={forward_m:.4f} m"
        )
    knee = _front_sign(corner) * math.acos(cosine_knee)
    hip_flexion = math.atan2(forward_m, down_m) - math.atan2(
        LINK_LENGTH_M * math.sin(knee),
        LINK_LENGTH_M + LINK_LENGTH_M * math.cos(knee),
    )
    return hip_flexion, knee


def _pose_degrees(
    down_by_corner_m: Mapping[str, float],
    forward_by_corner_m: Mapping[str, float],
    abduction_by_corner_deg: Mapping[str, float],
) -> dict[tuple[str, str], float]:
    pose: dict[tuple[str, str], float] = {}
    for corner in LEG_CORNERS:
        hip_flexion, knee = _leg_ik(
            corner,
            float(down_by_corner_m[corner]),
            float(forward_by_corner_m[corner]),
        )
        pose[(corner, "hip_abduction")] = float(abduction_by_corner_deg[corner])
        pose[(corner, "hip_flexion")] = math.degrees(hip_flexion)
        pose[(corner, "knee")] = math.degrees(knee)
    return pose


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
