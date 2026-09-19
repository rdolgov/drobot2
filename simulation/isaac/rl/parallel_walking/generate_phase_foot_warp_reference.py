"""Warp a proven periodic joint cycle in sagittal foot space.

The source trajectory remains the motion prior.  Unlike multiplying joint
angles about neutral, this tool changes only fore/aft foot travel and optional
swing clearance while preserving the source foot height during stance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PROXIMAL_LINK_LENGTH_M = 0.159896689
DISTAL_CONTACT_LENGTH_M = 0.159896689 + 0.031
LEG_SPECS = (
    ("front_left", 0, 4, 8, 0.0, 1.0),
    ("rear_left", 1, 5, 9, 0.5, -1.0),
    ("front_right", 2, 6, 10, 0.5, 1.0),
    ("rear_right", 3, 7, 11, 0.0, -1.0),
)


def _circular_sample(rows: list[list[float]], phase: float, column: int) -> float:
    sample = (phase % 1.0) * len(rows)
    lower = math.floor(sample)
    fraction = sample - lower
    a = rows[lower % len(rows)][column]
    b = rows[(lower + 1) % len(rows)][column]
    return a + fraction * (b - a)


def _source_leg_phase(
    target_leg_phase: float,
    source_duty_factor: float,
    target_duty_factor: float,
) -> float:
    if target_leg_phase < target_duty_factor:
        return target_leg_phase * source_duty_factor / target_duty_factor
    return source_duty_factor + (
        (target_leg_phase - target_duty_factor)
        * (1.0 - source_duty_factor)
        / (1.0 - target_duty_factor)
    )


def _forward_down(hip: float, knee: float) -> tuple[float, float]:
    forward = (
        PROXIMAL_LINK_LENGTH_M * math.sin(hip)
        + DISTAL_CONTACT_LENGTH_M * math.sin(hip + knee)
    )
    down = (
        PROXIMAL_LINK_LENGTH_M * math.cos(hip)
        + DISTAL_CONTACT_LENGTH_M * math.cos(hip + knee)
    )
    return forward, down


def _inverse_sagittal(
    forward: float,
    down: float,
    front_sign: float,
) -> tuple[float, float]:
    cosine_knee = (
        down * down
        + forward * forward
        - PROXIMAL_LINK_LENGTH_M**2
        - DISTAL_CONTACT_LENGTH_M**2
    ) / (2.0 * PROXIMAL_LINK_LENGTH_M * DISTAL_CONTACT_LENGTH_M)
    cosine_knee = min(1.0, max(-1.0, cosine_knee))
    knee = -front_sign * math.acos(cosine_knee)
    hip = math.atan2(forward, down) - math.atan2(
        DISTAL_CONTACT_LENGTH_M * math.sin(knee),
        PROXIMAL_LINK_LENGTH_M + DISTAL_CONTACT_LENGTH_M * math.cos(knee),
    )
    return hip, knee


def _circular_smooth(
    rows: list[list[float]],
    window: int,
) -> list[list[float]]:
    if window <= 1:
        return rows
    radius = window // 2
    offsets = range(-radius, radius + 1)
    return [
        [
            sum(rows[(row_index + offset) % len(rows)][column] for offset in offsets)
            / len(offsets)
            for column in range(len(rows[0]))
        ]
        for row_index in range(len(rows))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-duty-factor", type=float, default=0.62)
    parser.add_argument("--target-duty-factor", type=float, required=True)
    parser.add_argument("--fore-aft-scale", type=float, required=True)
    parser.add_argument(
        "--stance-crouch-m",
        type=float,
        default=0.0,
        help=(
            "Shorten every leg by this amount. This creates reach margin for "
            "the longer stride and corresponds to a lower body stance."
        ),
    )
    parser.add_argument("--extra-swing-lift-m", type=float, default=0.0)
    parser.add_argument(
        "--reach-margin-m",
        type=float,
        default=0.0005,
        help="Keep the requested foot radius this far inside full extension.",
    )
    parser.add_argument("--smooth-bins", type=int, default=3)
    parser.add_argument("--diagnostic-frequency-hz", type=float, default=1.60)
    args = parser.parse_args()

    if not 0.40 <= args.source_duty_factor < 1.0:
        parser.error("--source-duty-factor must be within [0.40, 1.0)")
    if not 0.40 <= args.target_duty_factor < 1.0:
        parser.error("--target-duty-factor must be within [0.40, 1.0)")
    if not 0.75 <= args.fore_aft_scale <= 1.75:
        parser.error("--fore-aft-scale must be within [0.75, 1.75]")
    if not 0.0 <= args.stance_crouch_m <= 0.060:
        parser.error("--stance-crouch-m must be within [0, 0.060]")
    if not 0.0 <= args.extra_swing_lift_m <= 0.040:
        parser.error("--extra-swing-lift-m must be within [0, 0.040]")
    if not 0.0 <= args.reach_margin_m <= 0.020:
        parser.error("--reach-margin-m must be within [0, 0.020]")
    if args.smooth_bins < 1 or args.smooth_bins % 2 == 0:
        parser.error("--smooth-bins must be a positive odd integer")
    if args.diagnostic_frequency_hz <= 0.0:
        parser.error("--diagnostic-frequency-hz must be positive")

    source_bytes = args.source.read_bytes()
    payload = json.loads(source_bytes)
    if payload.get("schema") != "drobot.phase-joint-reference.v1":
        parser.error("source is not a drobot.phase-joint-reference.v1 file")
    rows = payload.get("joint_position_rad")
    if not isinstance(rows, list) or len(rows) < 32:
        parser.error("source must contain at least 32 joint-position rows")
    if any(not isinstance(row, list) or len(row) != 12 for row in rows):
        parser.error("every source row must contain 12 joint positions")

    # Use the midpoint of each source foot's complete fore/aft envelope.  This
    # preserves its learned front/rear stance bias while increasing travel.
    forward_centers: dict[str, float] = {}
    for leg_name, _abduction, hip, knee, _phase_offset, _front_sign in LEG_SPECS:
        positions = [
            _forward_down(float(row[hip]), float(row[knee]))[0] for row in rows
        ]
        forward_centers[leg_name] = 0.5 * (min(positions) + max(positions))

    reach_projection_count = 0
    maximum_reach_projection_m = 0.0
    output_rows: list[list[float]] = []
    for sample_index in range(len(rows)):
        global_phase = sample_index / len(rows)
        output_row = [0.0] * 12
        for (
            leg_name,
            abduction_index,
            hip_index,
            knee_index,
            phase_offset,
            front_sign,
        ) in LEG_SPECS:
            target_leg_phase = (global_phase + phase_offset) % 1.0
            source_leg_phase = _source_leg_phase(
                target_leg_phase,
                args.source_duty_factor,
                args.target_duty_factor,
            )
            source_global_phase = (source_leg_phase - phase_offset) % 1.0
            abduction = _circular_sample(rows, source_global_phase, abduction_index)
            source_hip = _circular_sample(rows, source_global_phase, hip_index)
            source_knee = _circular_sample(rows, source_global_phase, knee_index)
            forward, down = _forward_down(source_hip, source_knee)
            center = forward_centers[leg_name]
            forward = center + args.fore_aft_scale * (forward - center)
            down -= args.stance_crouch_m
            if target_leg_phase >= args.target_duty_factor:
                swing_u = (
                    target_leg_phase - args.target_duty_factor
                ) / (1.0 - args.target_duty_factor)
                # sin^2 has zero height and zero slope at both contacts, so the
                # extra clearance does not introduce a touchdown velocity step.
                down -= args.extra_swing_lift_m * math.sin(math.pi * swing_u) ** 2
            maximum_radius = (
                PROXIMAL_LINK_LENGTH_M
                + DISTAL_CONTACT_LENGTH_M
                - args.reach_margin_m
            )
            if down > maximum_radius:
                maximum_reach_projection_m = max(
                    maximum_reach_projection_m,
                    down - maximum_radius,
                )
                down = maximum_radius
                reach_projection_count += 1
            maximum_abs_forward = math.sqrt(
                max(0.0, maximum_radius**2 - down**2)
            )
            projected_forward = min(
                maximum_abs_forward,
                max(-maximum_abs_forward, forward),
            )
            if not math.isclose(projected_forward, forward, abs_tol=1.0e-12):
                maximum_reach_projection_m = max(
                    maximum_reach_projection_m,
                    abs(projected_forward - forward),
                )
                forward = projected_forward
                reach_projection_count += 1
            hip, knee = _inverse_sagittal(forward, down, front_sign)
            output_row[abduction_index] = abduction
            output_row[hip_index] = hip
            output_row[knee_index] = knee
        output_rows.append(output_row)

    output_rows = _circular_smooth(output_rows, args.smooth_bins)
    maximum_velocity_deg_s = []
    for joint_index in range(12):
        maximum_delta = max(
            abs(
                output_rows[(row_index + 1) % len(output_rows)][joint_index]
                - output_rows[row_index][joint_index]
            )
            for row_index in range(len(output_rows))
        )
        maximum_velocity_deg_s.append(
            math.degrees(maximum_delta * len(output_rows) * args.diagnostic_frequency_hz)
        )

    payload["joint_position_rad"] = output_rows
    payload["sample_count"] = len(output_rows)
    payload["source_checkpoint"] = str(args.source)
    payload["source_checkpoint_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    payload["generation"] = {
        "method": "phase-remapped-sagittal-foot-warp.v1",
        "source_duty_factor": args.source_duty_factor,
        "target_duty_factor": args.target_duty_factor,
        "fore_aft_scale": args.fore_aft_scale,
        "stance_crouch_m": args.stance_crouch_m,
        "extra_swing_lift_m": args.extra_swing_lift_m,
        "reach_margin_m": args.reach_margin_m,
        "reach_projection_count": reach_projection_count,
        "maximum_reach_projection_m": maximum_reach_projection_m,
        "circular_smoothing_bins": args.smooth_bins,
        "forward_center_m_by_leg": forward_centers,
        "diagnostic_frequency_hz": args.diagnostic_frequency_hz,
        "maximum_joint_target_velocity_deg_s": maximum_velocity_deg_s,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(
        "Maximum target velocity at "
        f"{args.diagnostic_frequency_hz:.3f} Hz: "
        f"{max(maximum_velocity_deg_s):.1f} deg/s"
    )


if __name__ == "__main__":
    main()
