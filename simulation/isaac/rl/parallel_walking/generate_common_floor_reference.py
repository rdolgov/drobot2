"""Generate an UNQUALIFIED common-floor V44 ablation, never replace V44 v1.

This is analytical geometry only: no physics, inference or hardware is run.
The rigid shoe cannot stay exactly flat while its two sagittal joints maintain
one floor height throughout a fore/aft stroke. This prior instead permits a
bounded sole angle and gives every support shoe the same lowest-edge depth.

The 94 mm tread extends 1 mm below the 100 mm structural sole. In the bounded
angle range here the tread, not the structural sole, contains the lowest edge.
Both surfaces are checked explicitly. Smooth absolute value avoids a joint
velocity kink when the lowest edge changes from toe to heel; it conservatively
adds at most 0.235 mm of body-relative corner clearance. World attitude, loads,
servo response and contact still require simulation and hardware evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PROXIMAL_M = 0.159896689
STRUCTURAL_CONTACT_M = PROXIMAL_M + 0.031
TREAD_PROJECTION_M = 0.001
TREAD_CONTACT_M = STRUCTURAL_CONTACT_M + TREAD_PROJECTION_M
STRUCTURAL_HALF_LENGTH_M = 0.050
STRUCTURAL_HALF_WIDTH_M = 0.030
TREAD_HALF_LENGTH_M = 0.047
TREAD_HALF_WIDTH_M = 0.027
COMMON_LOWEST_EDGE_DEPTH_M = 0.330
STANCE_FORWARD_M = 0.080
STRIDE_M = 0.055
DUTY = 0.55
SWING_CLEARANCE_M = 0.010
SMOOTH_ABS_EPSILON = 0.005
MAX_STANCE_SOLE_DEG = 12.0
MAX_TARGET_RATE_DEG_S = 90.0
PROFILE = "common-floor-v1-hardware-matched-run-stop-external-rear-payload"
ACTION_ORDER = tuple(
    f"{leg}_{joint}"
    for joint in ("hip_abduction", "hip_flexion", "knee")
    for leg in ("front_left", "rear_left", "front_right", "rear_right")
)
LEGS = (
    ("front_left", 0, 4, 8, 0.0, 1.0),
    ("rear_left", 1, 5, 9, 0.5, -1.0),
    ("front_right", 2, 6, 10, 0.5, 1.0),
    ("rear_right", 3, 7, 11, 0.0, -1.0),
)


def smoothstep(u: float) -> float:
    """Quintic with zero first/second derivatives at either endpoint."""
    return u**3 * (10.0 + u * (-15.0 + 6.0 * u))


def inverse_sagittal(forward: float, down: float, sign: float) -> tuple[float, float]:
    cosine = (
        forward**2 + down**2 - PROXIMAL_M**2 - TREAD_CONTACT_M**2
    ) / (2.0 * PROXIMAL_M * TREAD_CONTACT_M)
    if not math.isfinite(cosine) or not -1.0 - 1e-10 <= cosine <= 1.0 + 1e-10:
        raise ValueError("Common-floor reference is unreachable; do not project it")
    knee = -sign * math.acos(max(-1.0, min(1.0, cosine)))
    hip = math.atan2(forward, down) - math.atan2(
        TREAD_CONTACT_M * math.sin(knee),
        PROXIMAL_M + TREAD_CONTACT_M * math.cos(knee),
    )
    return hip, knee


def pose_for_floor(forward: float, clearance: float, sign: float) -> tuple[float, float]:
    """Solve a smooth conservative tread-edge depth relative to the hip frame."""
    if sign * forward <= 0.0:
        raise ValueError("Common-floor prior must retain the opposed IK branches")
    target = COMMON_LOWEST_EDGE_DEPTH_M - clearance
    reach_squared = (PROXIMAL_M + TREAD_CONTACT_M) ** 2 - forward**2
    if reach_squared <= 0.0:
        raise ValueError("No reachable sagittal depth bracket")
    low, high = 0.150, math.sqrt(reach_squared) - 1e-10

    def smooth_edge_depth(down: float) -> float:
        hip, knee = inverse_sagittal(forward, down, sign)
        return down + TREAD_HALF_LENGTH_M * math.sqrt(
            math.sin(hip + knee) ** 2 + SMOOTH_ABS_EPSILON**2
        )

    if smooth_edge_depth(low) >= target or smooth_edge_depth(high) <= target:
        raise ValueError("No bracket for the common-floor shoe edge")
    for _ in range(60):
        middle = (low + high) / 2.0
        if smooth_edge_depth(middle) > target:
            high = middle
        else:
            low = middle
    down = (low + high) / 2.0
    if abs(smooth_edge_depth(down) - target) > 1e-9:
        raise ValueError("Common-floor edge solve did not converge")
    return inverse_sagittal(forward, down, sign)


def _validate_stride(stride_m: float) -> None:
    if not math.isfinite(stride_m) or stride_m <= 0.0:
        raise ValueError("Stride must be finite and positive")
    if stride_m >= 2.0 * STANCE_FORWARD_M:
        raise ValueError("Stride must retain the opposed IK branches throughout the sweep")


def phase_geometry(
    phase: float, sign: float, swing_clearance_m: float = SWING_CLEARANCE_M,
    stride_m: float = STRIDE_M,
) -> tuple[float, float, bool, float]:
    _validate_stride(stride_m)
    stance = phase < DUTY
    u = phase / DUTY if stance else (phase - DUTY) / (1.0 - DUTY)
    sweep = 0.5 - smoothstep(u) if stance else smoothstep(u) - 0.5
    forward = sign * STANCE_FORWARD_M + stride_m * sweep
    envelope = smoothstep(min(1.0, u / 0.5)) * smoothstep(min(1.0, (1.0 - u) / 0.5))
    clearance = 0.0 if stance else swing_clearance_m * envelope
    hip, knee = pose_for_floor(forward, clearance, sign)
    return hip, knee, stance, clearance


def actual_corner_depths(hip: float, knee: float) -> tuple[float, ...]:
    """All eight corners, zero abduction/level body; lateral corners coincide.

    The two +/- lateral corners at each longitudinal end have the same depth
    here, but both are included to keep the structural/tread geometry explicit.
    """
    sole = hip + knee
    upper_down = PROXIMAL_M * math.cos(hip)
    return tuple(
        upper_down + length * math.cos(sole) + edge_sign * half * math.sin(sole)
        for length, half in (
            (STRUCTURAL_CONTACT_M, STRUCTURAL_HALF_LENGTH_M),
            (TREAD_CONTACT_M, TREAD_HALF_LENGTH_M),
        )
        for edge_sign in (-1.0, 1.0)
        for _lateral_sign in (-1.0, 1.0)
    )


def build_rows(
    count: int, swing_clearance_m: float = SWING_CLEARANCE_M,
    stride_m: float = STRIDE_M,
) -> list[list[float]]:
    _validate_stride(stride_m)
    rows = []
    for index in range(count):
        row = [0.0] * 12
        for _leg, abd, hip_idx, knee_idx, offset, sign in LEGS:
            hip, knee, stance, _clearance = phase_geometry(
                (index / count + offset) % 1.0, sign, swing_clearance_m, stride_m
            )
            if abs(hip) > math.radians(57.0) or abs(knee) > math.radians(85.5):
                raise ValueError("Common-floor reference exceeds 95% soft joint limits")
            if stance and abs(hip + knee) > math.radians(MAX_STANCE_SOLE_DEG):
                raise ValueError("Common-floor stance sole exceeds its explicit 12 degree budget")
            row[abd], row[hip_idx], row[knee_idx] = 0.0, hip, knee
        rows.append(row)
    return rows


def summarize(
    rows: list[list[float]], frequency: float,
    swing_clearance_m: float = SWING_CLEARANCE_M,
) -> dict:
    count = len(rows)
    rates = [
        max(abs(rows[(i + 1) % count][j] - rows[i][j]) for i in range(count))
        * count * frequency * 180.0 / math.pi
        for j in range(12)
    ]
    if max(rates) > MAX_TARGET_RATE_DEG_S:
        raise ValueError("Common-floor prior exceeds the 90 degree/s rate budget")
    legs = {}
    maximum_support_depth_difference = 0.0
    for leg, _abd, hip_idx, knee_idx, offset, _sign in LEGS:
        stance_angles, all_angles, gaps, stance_depths, core_heights = [], [], [], [], []
        for i, row in enumerate(rows):
            phase = (i / count + offset) % 1.0
            stance = phase < DUTY
            u = max(0.0, (phase - DUTY) / (1.0 - DUTY))
            envelope = smoothstep(min(1.0, u / 0.5)) * smoothstep(min(1.0, (1.0 - u) / 0.5))
            target_clearance = 0.0 if stance else swing_clearance_m * envelope
            angle = math.degrees(row[hip_idx] + row[knee_idx])
            lowest_depth = max(actual_corner_depths(row[hip_idx], row[knee_idx]))
            height = COMMON_LOWEST_EDGE_DEPTH_M - lowest_depth
            gap = height - target_clearance
            if gap < -1e-8 or gap > TREAD_HALF_LENGTH_M * SMOOTH_ABS_EPSILON + 1e-8:
                raise ValueError("Actual structural/tread corners violate conservative edge clearance")
            all_angles.append(angle)
            gaps.append(gap)
            if stance:
                stance_angles.append(angle)
                stance_depths.append(lowest_depth)
            elif envelope >= 0.99:
                core_heights.append(height)
        legs[leg] = {
            "hip_range_deg": [math.degrees(min(r[hip_idx] for r in rows)), math.degrees(max(r[hip_idx] for r in rows))],
            "knee_range_deg": [math.degrees(min(r[knee_idx] for r in rows)), math.degrees(max(r[knee_idx] for r in rows))],
            "stance_sole_range_deg": [min(stance_angles), max(stance_angles)],
            "whole_cycle_sole_range_deg": [min(all_angles), max(all_angles)],
            "additional_actual_corner_gap_m": [min(gaps), max(gaps)],
            "stance_lowest_corner_depth_m": [min(stance_depths), max(stance_depths)],
            "core_swing_actual_corner_height_m": [min(core_heights), max(core_heights)],
        }
    for i, row in enumerate(rows):
        depths = [
            max(actual_corner_depths(row[hip_idx], row[knee_idx]))
            for _leg, _abd, hip_idx, knee_idx, offset, _sign in LEGS
            if (i / count + offset) % 1.0 < DUTY
        ]
        if depths:
            maximum_support_depth_difference = max(maximum_support_depth_difference, max(depths) - min(depths))
    return {
        "method": "offline_analytical_joint_and_all_eight_corner_geometry",
        "samples_per_cycle": count,
        "maximum_joint_target_velocity_deg_s": rates,
        "maximum_scheduled_support_depth_difference_m": maximum_support_depth_difference,
        "by_leg": legs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument(
        "--stride-m", type=float, default=STRIDE_M,
        help="Fixed fore/aft stroke; default 0.055 m; command speed derives from stroke/cadence/duty",
    )
    parser.add_argument(
        "--swing-clearance-m", type=float, default=SWING_CLEARANCE_M,
        help="Body-relative lowest-corner swing lift; default 0.010 m",
    )
    parser.add_argument(
        "--low-frequency-hz", type=float, default=0.30,
        help="Low reference cadence; command speed derives from fixed stroke/duty",
    )
    parser.add_argument(
        "--high-frequency-hz", type=float, default=0.40,
        help="High reference cadence; command speed derives from fixed stroke/duty",
    )
    args = parser.parse_args()
    if args.samples < 256:
        parser.error("Use at least 256 samples; dense analytical metrics use at least 4096")
    if not math.isfinite(args.swing_clearance_m) or args.swing_clearance_m <= 0.0:
        parser.error("--swing-clearance-m must be finite and positive")
    try:
        _validate_stride(args.stride_m)
    except ValueError as exc:
        parser.error(f"--stride-m: {exc}")
    if not (
        math.isfinite(args.low_frequency_hz)
        and math.isfinite(args.high_frequency_hz)
        and 0.0 < args.low_frequency_hz <= args.high_frequency_hz
    ):
        parser.error("Cadences must be finite, positive and ordered low <= high")
    outputs = tuple(
        (name, args.stride_m * frequency / DUTY, frequency)
        for name, frequency in (
            ("low.json", args.low_frequency_hz),
            ("high.json", args.high_frequency_hz),
        )
    )
    paths = [args.output_dir / name for name, _speed, _frequency in outputs]
    if any(path.exists() for path in paths):
        parser.error("Candidate output already exists; choose a new directory to preserve provenance")
    rows = build_rows(args.samples, args.swing_clearance_m, args.stride_m)
    dense_rows = build_rows(max(4096, args.samples), args.swing_clearance_m, args.stride_m)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payloads = []
    for _name, speed, frequency in outputs:
        if abs(args.stride_m * frequency / DUTY - speed) > 1e-12:
            raise ValueError("Cadence/stride must match the nominal no-slip command")
        payloads.append({
            "schema": "drobot.phase-joint-reference.v1",
            "sample_count": args.samples,
            "joint_order": ACTION_ORDER,
            "joint_position_rad": rows,
            "forward_speed_m_s": speed,
            "generation": {
                "qualification": "UNQUALIFIED_ABLATION",
                "method": "common-floor-smooth-tread-edge.v1",
                "training_profile": PROFILE,
                "generator_sha256": source_hash,
                "speed_m_s": speed,
                "frequency_hz": frequency,
                "duty_factor": DUTY,
                "phase_offsets_fl_rl_fr_rr": [0.0, 0.5, 0.5, 0.0],
                "stance_fore_aft_m": STANCE_FORWARD_M,
                "stride_m": args.stride_m,
                "common_lowest_edge_depth_m": COMMON_LOWEST_EDGE_DEPTH_M,
                "sole_corner_clearance_m": args.swing_clearance_m,
                "clearance_envelope": "quintic-symmetric.v1",
                "clearance_transition_fraction": 0.5,
                "smooth_absolute_epsilon": SMOOTH_ABS_EPSILON,
                "maximum_smoothing_corner_gap_m": TREAD_HALF_LENGTH_M * SMOOTH_ABS_EPSILON,
                "maximum_stance_sole_angle_deg": MAX_STANCE_SOLE_DEG,
                "target_velocity_limit_deg_s": MAX_TARGET_RATE_DEG_S,
                "packet_step_limit_deg": 2.0,
                "geometry": {
                    "proximal_m": PROXIMAL_M,
                    "structural_contact_m": STRUCTURAL_CONTACT_M,
                    "tread_contact_m": TREAD_CONTACT_M,
                    "structural_sole_fore_aft_m": 2.0 * STRUCTURAL_HALF_LENGTH_M,
                    "structural_sole_lateral_m": 2.0 * STRUCTURAL_HALF_WIDTH_M,
                    "tread_fore_aft_m": 2.0 * TREAD_HALF_LENGTH_M,
                    "tread_lateral_m": 2.0 * TREAD_HALF_WIDTH_M,
                    "tread_projection_m": TREAD_PROJECTION_M,
                },
                "exported_table_metrics": summarize(rows, frequency, args.swing_clearance_m),
                "dense_analytical_metrics": summarize(dense_rows, frequency, args.swing_clearance_m),
                "limitations": [
                    "UNQUALIFIED: no physics, policy inference, contact stability or hardware evidence.",
                    "All-eight-corner calculation assumes zero abduction and a level body on a plane.",
                    "Support soles deliberately tilt within 12 degrees; a rigid foot has no ankle.",
                    "Smooth edge swapping permits up to 0.235 mm extra clearance, not exact all-foot contact.",
                    "The tread contact-center defines fore/aft reach; structural center differs by <=1 mm.",
                    "Low and high have identical fixed-stroke rows; speed changes through cadence only.",
                    "Dense rates are finite differences of analytical targets, not measured actuator capability.",
                    "Body tilt, roll, compliance, payload loading and target tracking can still erase clearance.",
                    "Stopping/centering and startup ramps are controller behavior, not validated by this generator.",
                ],
            },
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in zip(paths, payloads):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        dense = payload["generation"]["dense_analytical_metrics"]
        settings = payload["generation"]
        print(
            f"Wrote {path}; UNQUALIFIED_ABLATION; analytical lift "
            f"{settings['sole_corner_clearance_m'] * 1000.0:.1f} mm, "
            f"cadence {settings['frequency_hz']:.3f} Hz, "
            f"nominal no-slip speed {settings['speed_m_s']:.4f} m/s, "
            f"peak target rate {max(dense['maximum_joint_target_velocity_deg_s']):.3f} deg/s"
        )


if __name__ == "__main__":
    main()
