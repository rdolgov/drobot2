"""Create NEW, unqualified sequential-crawl references; no simulation is run.

All scheduled stance centers move backward at the same constant speed. A
quintic Hermite swing return matches that speed and zero acceleration at both
ends. This avoids incompatible phase-shifted stance-center speeds. Common-floor
shoe geometry is shared with the separately preserved V44 analytical generator.

Scheduled support is NOT demonstrated physical contact, load sharing, or static
stability. The rigid shoe rolls between edges; center-velocity agreement does
not prove exact no-slip agreement of real contact patches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import generate_common_floor_reference as geometry

LEGS = ("front_left", "rear_left", "front_right", "rear_right")
ACTION_ORDER = tuple(f"{leg}_{joint}" for joint in ("hip_abduction", "hip_flexion", "knee") for leg in LEGS)
SIGNS = (1.0, -1.0, 1.0, -1.0)
DEFAULT_OFFSETS = (0.0, 0.25, 0.5, 0.75)
STRUCTURAL_GAP_LIMIT_M = geometry.TREAD_HALF_LENGTH_M * geometry.SMOOTH_ABS_EPSILON


def finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def schedule(duty: float, offsets: tuple[float, ...]) -> dict:
    """Check all open intervals between contact events, not just sample rows."""
    if not .75 <= duty < 1.0:
        raise ValueError("Sequential crawl requires duty factor in [0.75, 1)")
    if len(offsets) != 4 or any(not math.isfinite(offset) or not 0.0 <= offset < 1.0 for offset in offsets):
        raise ValueError("Supply FL, RL, FR, RR phase offsets in [0, 1)")
    liftoffs = [(duty - offset) % 1.0 for offset in offsets]
    touchdowns = [(1.0 - offset) % 1.0 for offset in offsets]
    boundaries = sorted({0.0, 1.0, *liftoffs, *touchdowns})
    intervals = []
    fractions = {3: 0.0, 4: 0.0}
    for start, end in zip(boundaries, boundaries[1:]):
        phase = (start + end) / 2.0
        support = [leg for leg, offset in zip(LEGS, offsets) if (phase + offset) % 1.0 < duty]
        if len(support) < 3:
            raise ValueError("Phase offsets create overlapping swings; fewer than three scheduled supports")
        fractions[len(support)] += end - start
        intervals.append({"phase_start": start, "phase_end": end, "scheduled_support": support,
                          "scheduled_swing": [leg for leg in LEGS if leg not in support]})
    return {
        "proof": "Contact masks are constant between listed phase events; every open event interval has >=3 scheduled supports.",
        "minimum_scheduled_support_count": min(len(item["scheduled_support"]) for item in intervals),
        "three_foot_support_cycle_fraction": fractions[3],
        "four_foot_support_cycle_fraction": fractions[4],
        "liftoff_order_from_phase_zero": [LEGS[index] for index in sorted(range(4), key=lambda index: liftoffs[index])],
        "liftoff_phase_by_leg": dict(zip(LEGS, liftoffs)),
        "touchdown_phase_by_leg": dict(zip(LEGS, touchdowns)),
        "intervals": intervals,
    }


def leg_pose(phase: float, sign: float, duty: float, stride: float, lift: float) -> tuple[float, float, bool, float, float]:
    stance = phase < duty
    if stance:
        # d(forward)/dt = -stride*cadence/duty for EVERY scheduled support.
        displacement = stride * (.5 - phase / duty)
        clearance = 0.0
    else:
        u = (phase - duty) / (1.0 - duty)
        endpoint_tangent = -stride * (1.0 - duty) / duty
        displacement = -stride / 2.0 + endpoint_tangent * u + (stride - endpoint_tangent) * geometry.smoothstep(u)
        envelope = geometry.smoothstep(min(1.0, u / .5)) * geometry.smoothstep(min(1.0, (1.0 - u) / .5))
        clearance = lift * envelope
    forward = sign * geometry.STANCE_FORWARD_M + displacement
    hip, knee = geometry.pose_for_floor(forward, clearance, sign)
    if abs(hip) > math.radians(57.0) or abs(knee) > math.radians(85.5):
        raise ValueError("Crawl reference exceeds the existing 95% soft joint limits")
    if stance and abs(hip + knee) > math.radians(12.0):
        raise ValueError("Crawl stance sole exceeds the explicit 12 degree geometry allowance")
    actual_depth = max(geometry.actual_corner_depths(hip, knee))
    gap = geometry.COMMON_LOWEST_EDGE_DEPTH_M - actual_depth - clearance
    if not -1e-8 <= gap <= STRUCTURAL_GAP_LIMIT_M + 1e-8:
        raise ValueError("Actual structural/tread corner clearance violates the conservative floor solve")
    return hip, knee, stance, clearance, forward


def make_rows(count: int, duty: float, offsets: tuple[float, ...], stride: float, lift: float) -> list[list[float]]:
    rows = []
    for index in range(count):
        row = [0.0] * 12
        support_count = 0
        for leg_index, (offset, sign) in enumerate(zip(offsets, SIGNS)):
            hip, knee, stance, _clearance, _forward = leg_pose((index / count + offset) % 1.0, sign, duty, stride, lift)
            row[4 + leg_index], row[8 + leg_index] = hip, knee
            support_count += int(stance)
        if support_count < 3:
            raise ValueError("Sampled contact mask violates three-foot support")
        rows.append(row)
    return rows


def metrics(rows: list[list[float]], duty: float, offsets: tuple[float, ...], frequency: float) -> dict:
    count = len(rows)
    rates = [max(abs(rows[(i + 1) % count][j] - rows[i][j]) for i in range(count)) * count * frequency * 180.0 / math.pi for j in range(12)]
    limits = [60.0] * 4 + [90.0] * 8
    if any(rate > limit for rate, limit in zip(rates, limits)):
        raise ValueError("Crawl prior exceeds existing 60 degree/s abduction or 90 degree/s sagittal cap")
    support_depth_spread = 0.0
    legs = {}
    for leg_index, leg in enumerate(LEGS):
        hip_id, knee_id = 4 + leg_index, 8 + leg_index
        stance_angles, swing_angles, stance_heights, core_heights = [], [], [], []
        for i, row in enumerate(rows):
            phase = (i / count + offsets[leg_index]) % 1.0
            angle = math.degrees(row[hip_id] + row[knee_id])
            height = geometry.COMMON_LOWEST_EDGE_DEPTH_M - max(geometry.actual_corner_depths(row[hip_id], row[knee_id]))
            if phase < duty:
                stance_angles.append(angle)
                stance_heights.append(height)
            else:
                swing_angles.append(angle)
                u = (phase - duty) / (1.0 - duty)
                envelope = geometry.smoothstep(min(1.0, u / .5)) * geometry.smoothstep(min(1.0, (1.0 - u) / .5))
                if envelope >= .99:
                    core_heights.append(height)
        legs[leg] = {
            "hip_range_deg": [math.degrees(min(row[hip_id] for row in rows)), math.degrees(max(row[hip_id] for row in rows))],
            "knee_range_deg": [math.degrees(min(row[knee_id] for row in rows)), math.degrees(max(row[knee_id] for row in rows))],
            "stance_sole_range_deg": [min(stance_angles), max(stance_angles)],
            "swing_sole_range_deg": [min(swing_angles), max(swing_angles)],
            "stance_actual_corner_height_m": [min(stance_heights), max(stance_heights)],
            "core_swing_actual_corner_height_m": [min(core_heights), max(core_heights)],
        }
    for i, row in enumerate(rows):
        support_depths = [max(geometry.actual_corner_depths(row[4+j], row[8+j])) for j in range(4) if (i / count + offsets[j]) % 1.0 < duty]
        support_depth_spread = max(support_depth_spread, max(support_depths) - min(support_depths))
    return {
        "method": "offline_analytical_geometry_and_finite_difference_rates",
        "samples_per_cycle": count,
        "maximum_joint_target_velocity_deg_s": rates,
        "maximum_scheduled_support_depth_difference_m": support_depth_spread,
        "nominal_60hz_target_step_upper_bound_deg": [rate / 60.0 for rate in rates],
        "by_leg": legs,
    }


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duty-factor", type=float, default=.80)
    parser.add_argument("--phase-offsets", type=float, nargs=4, metavar=("FL", "RL", "FR", "RR"), default=DEFAULT_OFFSETS)
    parser.add_argument("--stride-m", type=float, default=.055)
    parser.add_argument("--swing-clearance-m", type=float, default=.020)
    parser.add_argument("--low-frequency-hz", type=float, default=.10)
    parser.add_argument("--high-frequency-hz", type=float, default=.12)
    parser.add_argument("--samples", type=int, default=1024)
    args = parser.parse_args()
    if args.samples < 512:
        parser.error("Use at least 512 samples; dense geometry calculations use >=4096")
    try:
        for name in ("stride_m", "swing_clearance_m", "low_frequency_hz", "high_frequency_hz"):
            finite_positive(getattr(args, name), name)
        if args.low_frequency_hz > args.high_frequency_hz:
            raise ValueError("Low cadence must not exceed high cadence")
        offsets = tuple(args.phase_offsets)
        schedule_record = schedule(args.duty_factor, offsets)
        paths = [args.output_dir / name for name in ("low.json", "high.json")]
        if any(path.exists() for path in paths):
            raise ValueError("Output exists; choose a NEW directory instead of overwriting a reference")
        rows = make_rows(args.samples, args.duty_factor, offsets, args.stride_m, args.swing_clearance_m)
        dense = make_rows(max(4096, args.samples), args.duty_factor, offsets, args.stride_m, args.swing_clearance_m)
        records = []
        for frequency in (args.low_frequency_hz, args.high_frequency_hz):
            speed = args.stride_m * frequency / args.duty_factor
            if speed <= .004:
                raise ValueError("Candidate speed is at/below existing 0.004 m/s standstill deadband; redesign the cadence/stroke")
            records.append({
                "schema": "drobot.phase-joint-reference.v1", "sample_count": args.samples,
                "joint_order": ACTION_ORDER, "joint_position_rad": rows,
                "forward_speed_m_s": speed,
                "generation": {
                    "qualification": "UNQUALIFIED_SEQUENTIAL_CRAWL_ABLATION",
                    "method": "common-floor-constant-stance-velocity-crawl.v1",
                    "training_profile": "common-floor-sequential-crawl-contact-curriculum",
                    "generator_sha256": digest(Path(__file__)),
                    "geometry_helper_path": Path(geometry.__file__).name,
                    "geometry_helper_sha256": digest(Path(geometry.__file__)),
                    "speed_m_s": speed, "frequency_hz": frequency, "gait_period_s": 1.0 / frequency,
                    "duty_factor": args.duty_factor, "phase_offsets_fl_rl_fr_rr": offsets,
                    "stride_m": args.stride_m, "stance_fore_aft_m": geometry.STANCE_FORWARD_M,
                    "common_lowest_edge_depth_m": geometry.COMMON_LOWEST_EDGE_DEPTH_M,
                    "sole_corner_clearance_m": args.swing_clearance_m,
                    "clearance_envelope": "quintic-symmetric.v1", "clearance_transition_fraction": .5,
                    "smooth_absolute_epsilon": geometry.SMOOTH_ABS_EPSILON,
                    "maximum_smoothing_corner_gap_m": STRUCTURAL_GAP_LIMIT_M,
                    "maximum_stance_sole_angle_deg": 12.0,
                    "target_velocity_limit_deg_s_by_action": [60.0] * 4 + [90.0] * 8,
                    "packet_step_limit_deg": 2.0,
                    "stance_center_backward_velocity_m_s": -speed,
                    "forward_push_derivation": "For every stance leg, x_body=sign*0.080+stride*(0.5-phase/duty), so dx_body/dt=-stride*cadence/duty. A stationary contact center would therefore imply forward body speed +stride*cadence/duty. Rigid-shoe rolling/sliding can change actual speed.",
                    "swing_return": "Hermite quintic: -stride/2+tangent*u+(stride-tangent)*smoothstep5(u), tangent=-stride*(1-duty)/duty. Position, velocity and acceleration match constant-speed stance at both joins.",
                    "contact_schedule": schedule_record,
                    "exported_table_metrics": metrics(rows, args.duty_factor, offsets, frequency),
                    "dense_analytical_metrics": metrics(dense, args.duty_factor, offsets, frequency),
                    "limitations": [
                        "No dynamics, balance, policy inference or real hardware validation has been run.",
                        "Three scheduled supports do not establish physical contact or COM projection inside the support polygon.",
                        "No payload, mass, COM, torque cap, drive gain or motion safety limit is changed by a reference.",
                        "Level-body/zero-abduction corner geometry allows <=0.235 mm smoothing clearance; loaded posture can erase lift.",
                        "Rigid-shoe contact-edge rolling means matched stance-center velocities do not prove exact no-slip contact.",
                        "A packet hold can still activate the unchanged two-degree per-packet cap on recovery.",
                        "Startup/stop blending is not covered by periodic geometric rate bounds.",
                        "This intentionally slow contact curriculum is not a running gait or a claimed hardware speed.",
                    ],
                },
            })
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, record in zip(paths, records):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        generation = record["generation"]
        print(f"Wrote {path}; ANALYTICAL_UNQUALIFIED; {generation['frequency_hz']:.3f} Hz, "
              f"{generation['speed_m_s']:.6f} m/s, minimum scheduled support3, "
              f"peak target rate {max(generation['dense_analytical_metrics']['maximum_joint_target_velocity_deg_s']):.3f} deg/s")


if __name__ == "__main__":
    main()
