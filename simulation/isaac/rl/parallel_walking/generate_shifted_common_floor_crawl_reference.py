"""Generate a NEW analytical lateral-weight-transfer crawl, never overwrite.

No simulator, policy or hardware is exercised. Unlike simply adding abduction
to a sagittal table, this solves all eight structural/tread corners in 3D,
including the abduction-to-flexion offset and mirrored abduction axes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import generate_common_floor_reference as geometry
import generate_common_floor_crawl_reference as crawl

LEGS = crawl.LEGS
SIDES = (1.0, 1.0, -1.0, -1.0)
FRONTS = crawl.SIGNS
OFFSETS = crawl.DEFAULT_OFFSETS
HIP_OUTWARD_M = .0286117
HIP_DROP_M = .095696689
HIP_FORWARD_OFFSET_M = -.00195  # multiply by side sign
LATERAL_ABS_EPSILON = .0005
SURFACE_SMOOTH_MAX_EPSILON_M = .000002
SMOOTH_GAP_BOUND_M = (
    geometry.STRUCTURAL_HALF_LENGTH_M * geometry.SMOOTH_ABS_EPSILON
    + geometry.STRUCTURAL_HALF_WIDTH_M * LATERAL_ABS_EPSILON
    + SURFACE_SMOOTH_MAX_EPSILON_M / 2.0
)
SURFACES = (
    (geometry.STRUCTURAL_CONTACT_M, geometry.STRUCTURAL_HALF_LENGTH_M, geometry.STRUCTURAL_HALF_WIDTH_M),
    (geometry.TREAD_CONTACT_M, geometry.TREAD_HALF_LENGTH_M, geometry.TREAD_HALF_WIDTH_M),
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_URDF = REPOSITORY_ROOT / "cad" / "exports" / "urdf" / "quadruped_robot.urdf"
RUNTIME_KINEMATICS = REPOSITORY_ROOT / "simulation" / "isaac" / "_quadruped_runtime.py"


def body_lateral_shift(phase: float, amplitude: float, duty: float) -> float:
    """Move in all-support intervals; hold away from the side about to swing."""
    transfer_width = duty - .75
    if not 0.0 < transfer_width < .25:
        raise ValueError("This four-beat schedule needs duty strictly between .75 and 1")
    phase %= 1.0
    if phase < transfer_width:
        return amplitude * (2 * geometry.smoothstep(phase / transfer_width) - 1)
    if phase < .5:
        return amplitude
    if phase < .5 + transfer_width:
        return amplitude * (1 - 2 * geometry.smoothstep((phase - .5) / transfer_width))
    return -amplitude


def body_shift_derivative(phase: float, amplitude: float, duty: float) -> float:
    """Analytical d(body Y)/d(global cycle phase), metres per cycle."""
    width = duty - .75
    phase %= 1.0
    if phase < width:
        sign, u = 1.0, phase / width
    elif .5 <= phase < .5 + width:
        sign, u = -1.0, (phase-.5) / width
    else:
        return 0.0
    return sign * 2 * amplitude / width * 30 * u*u * (1-u)*(1-u)


def trajectory(phase: float, front: float, duty: float, stride: float, lift: float):
    stance = phase < duty
    if stance:
        displacement = stride * (.5 - phase / duty)
        envelope = 0.0
    else:
        u = (phase - duty) / (1.0 - duty)
        tangent = -stride * (1.0 - duty) / duty
        displacement = -stride / 2 + tangent * u + (stride - tangent) * geometry.smoothstep(u)
        envelope = geometry.smoothstep(min(1., u / .5)) * geometry.smoothstep(min(1., (1. - u) / .5))
    return front * geometry.STANCE_FORWARD_M + displacement, lift * envelope, stance, envelope


def abduction_for_center(down: float, body_y: float, side: float) -> float:
    """Exact target-center Y translation; final joint sign is mirrored by side.

    At zero abduction the flexion point relative to the abduction origin is
    (-side*.00195, side*.0286117, -.095696689). The joint rotates about
    side*X, so outward Y = outward*cos(a)+(drop+down)*sin(a).
    Requesting body +Y translates every tread center by -Y, not an outward
    stance-width change. This is the exact-offset branch in the existing
    _quadruped_runtime.py, extended here to also solve the vertical floor.
    """
    vertical = HIP_DROP_M + down
    radius = math.hypot(HIP_OUTWARD_M, vertical)
    desired_outward = HIP_OUTWARD_M - side * body_y
    if abs(desired_outward) >= radius:
        raise ValueError("Lateral translation is outside abduction reach")
    return math.asin(desired_outward / radius) - math.atan2(HIP_OUTWARD_M, vertical)


def corner_positions(abduction: float, hip: float, knee: float, side: float):
    """All eight exact corner XYZ positions relative to the abduction origin.

    Body axes are +X forward, +Y left, +Z up. The shoe pitch is h+k. First
    construct its sagittal corners, then rotate the entire offset chain by
    Rx(side*a). Root-joint XYZ is intentionally omitted (constant per leg).
    """
    sole = hip + knee
    ca, sa = math.cos(abduction), math.sin(abduction)
    corners = []
    for distal, half_length, half_width in SURFACES:
        for end in (-half_length, half_length):
            for lateral in (-half_width, half_width):
                x = side * HIP_FORWARD_OFFSET_M + geometry.PROXIMAL_M * math.sin(hip) + distal * math.sin(sole) + end * math.cos(sole)
                unrotated_down = HIP_DROP_M + geometry.PROXIMAL_M * math.cos(hip) + distal * math.cos(sole) - end * math.sin(sole)
                unrotated_y = side * HIP_OUTWARD_M + lateral
                y = unrotated_y * ca + side * unrotated_down * sa
                z = side * unrotated_y * sa - unrotated_down * ca
                corners.append((x, y, z))
    return tuple(corners)


def smooth_corner_depth(abduction: float, hip: float, knee: float) -> float:
    """Conservative C2 smooth maximum of both surfaces' lowest-corner depth."""
    sole = hip + knee
    ca, sa = math.cos(abduction), math.sin(abduction)
    longitudinal = math.sqrt((math.sin(sole) * ca)**2 + geometry.SMOOTH_ABS_EPSILON**2)
    lateral = math.sqrt(sa**2 + LATERAL_ABS_EPSILON**2)
    depths = [
        (HIP_DROP_M + geometry.PROXIMAL_M * math.cos(hip) + distal * math.cos(sole)) * ca
        - HIP_OUTWARD_M * sa + half_length * longitudinal + half_width * lateral
        for distal, half_length, half_width in SURFACES
    ]
    first, second = depths
    return (first + second + math.sqrt((first - second)**2 + SURFACE_SMOOTH_MAX_EPSILON_M**2)) / 2.0


def solve_pose(forward: float, clearance: float, body_y: float, side: float, front: float):
    target_depth = HIP_DROP_M + geometry.COMMON_LOWEST_EDGE_DEPTH_M - clearance
    if front * forward <= 0:
        raise ValueError("Requested trajectory crossed its opposed IK branch")
    reach_squared = (geometry.PROXIMAL_M + geometry.TREAD_CONTACT_M)**2 - forward**2
    if reach_squared <= 0:
        raise ValueError("Unreachable sagittal forward target")

    def pose(down):
        hip, knee = geometry.inverse_sagittal(forward, down, front)
        return abduction_for_center(down, body_y, side), hip, knee

    low, high = .150, math.sqrt(reach_squared) - 1e-10
    if smooth_corner_depth(*pose(low)) >= target_depth or smooth_corner_depth(*pose(high)) <= target_depth:
        raise ValueError("Cannot bracket a reachable common-floor 3D corner pose")
    for _ in range(60):
        middle = (low + high) / 2
        if smooth_corner_depth(*pose(middle)) > target_depth:
            high = middle
        else:
            low = middle
    result = pose((low + high) / 2)
    if abs(smooth_corner_depth(*result) - target_depth) > 1e-9:
        raise ValueError("3D common-floor solve did not converge")
    return result


def make_rows(count: int, duty: float, stride: float, lift: float, shift: float):
    rows = []
    for i in range(count):
        phase = i / count
        body_y = body_lateral_shift(phase, shift, duty)
        row = [0.0] * 12
        for j, (offset, front, side) in enumerate(zip(OFFSETS, FRONTS, SIDES)):
            forward, clearance, stance, _ = trajectory((phase + offset) % 1, front, duty, stride, lift)
            a, h, k = solve_pose(forward, clearance, body_y, side, front)
            if any(abs(angle) > math.radians(limit) for angle, limit in zip((a, h, k), (23.75, 57., 85.5))):
                raise ValueError("Existing 95% soft joint limit exceeded")
            tilt = math.acos(max(-1., min(1., math.cos(h + k) * math.cos(a))))
            if stance and tilt > math.radians(12.):
                raise ValueError(f"Full 3D stance sole tilt exceeds 12 degrees: {math.degrees(tilt):.6f}")
            depth = -min(point[2] for point in corner_positions(a, h, k, side))
            gap = HIP_DROP_M + geometry.COMMON_LOWEST_EDGE_DEPTH_M - depth - clearance
            if not -1e-8 <= gap <= SMOOTH_GAP_BOUND_M + 1e-8:
                raise ValueError("Exact 3D corner geometry violates smooth floor gap bound")
            row[j], row[4+j], row[8+j] = a, h, k
        rows.append(row)
    return rows


def metrics(rows, frequency, duty, stride, lift, shift):
    count = len(rows)
    rates = [max(abs(rows[(i+1) % count][j] - rows[i][j]) for i in range(count)) * count * frequency * 180 / math.pi for j in range(12)]
    if any(rate > limit for rate, limit in zip(rates, [60.] * 4 + [90.] * 8)):
        raise ValueError(f"Unchanged target velocity cap exceeded; peaks by action: {rates}")
    if max(rates) / 60. > 2.:
        raise ValueError("Nominal 60 Hz reference exceeds existing two-degree packet cap")
    by_leg = {}
    depth_spread = 0.
    lateral_error = 0.
    for j, leg in enumerate(LEGS):
        stance_tilts, swing_tilts, stance_heights, core_heights = [], [], [], []
        for i, row in enumerate(rows):
            phase = i / count
            a, h, k = row[j], row[4+j], row[8+j]
            corners = corner_positions(a, h, k, SIDES[j])
            height = HIP_DROP_M + geometry.COMMON_LOWEST_EDGE_DEPTH_M + min(c[2] for c in corners)
            tilt = math.degrees(math.acos(max(-1., min(1., math.cos(h+k)*math.cos(a)))))
            _, _, stance, envelope = trajectory((phase+OFFSETS[j]) % 1, FRONTS[j], duty, stride, lift)
            if stance:
                stance_tilts.append(tilt); stance_heights.append(height)
            else:
                swing_tilts.append(tilt)
                if envelope >= .99:
                    core_heights.append(height)
            tread_center_y = sum(c[1] for c in corners[4:]) / 4
            expected_y = SIDES[j] * HIP_OUTWARD_M - body_lateral_shift(phase, shift, duty)
            lateral_error = max(lateral_error, abs(tread_center_y - expected_y))
        by_leg[leg] = {
            "joint_angle_ranges_deg": [[math.degrees(min(r[joint] for r in rows)), math.degrees(max(r[joint] for r in rows))] for joint in (j, 4+j, 8+j)],
            "stance_sole_tilt_range_deg": [min(stance_tilts), max(stance_tilts)],
            "swing_sole_tilt_range_deg": [min(swing_tilts), max(swing_tilts)],
            "stance_lowest_corner_height_m": [min(stance_heights), max(stance_heights)],
            "core_99pct_swing_lowest_corner_height_m": [min(core_heights), max(core_heights)],
        }
    for i, row in enumerate(rows):
        depths = [-min(c[2] for c in corner_positions(row[j], row[4+j], row[8+j], SIDES[j])) for j in range(4) if (i/count+OFFSETS[j]) % 1 < duty]
        depth_spread = max(depth_spread, max(depths) - min(depths))
    return {
        "method": "offline_full_3D_geometry_and_finite_difference_rates_NO_dynamics",
        "samples_per_cycle": count, "maximum_joint_target_velocity_deg_s": rates,
        "nominal_60hz_target_step_upper_bound_deg": [r/60 for r in rates],
        "maximum_scheduled_support_corner_depth_difference_m": depth_spread,
        "maximum_tread_center_lateral_translation_error_m": lateral_error,
        "by_leg": by_leg,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--body-lateral-shift-m", type=float, default=.030)
    parser.add_argument("--duty-factor", type=float, default=.80)
    parser.add_argument("--stride-m", type=float, default=.055)
    parser.add_argument("--swing-clearance-m", type=float, default=.020)
    parser.add_argument("--low-frequency-hz", type=float, default=.10)
    parser.add_argument("--high-frequency-hz", type=float, default=.12)
    parser.add_argument("--samples", type=int, default=1024)
    args = parser.parse_args()
    try:
        for key in ("body_lateral_shift_m", "duty_factor", "stride_m", "swing_clearance_m", "low_frequency_hz", "high_frequency_hz"):
            crawl.finite_positive(getattr(args, key), key)
        if args.samples < 512 or args.low_frequency_hz > args.high_frequency_hz:
            raise ValueError("Require >=512 rows and low cadence <= high cadence")
        schedule = crawl.schedule(args.duty_factor, OFFSETS)
        paths = [args.output_dir / name for name in ("low.json", "high.json")]
        if any(path.exists() for path in paths):
            raise ValueError("Never overwrite references; choose a NEW output directory")
        rows = make_rows(args.samples, args.duty_factor, args.stride_m, args.swing_clearance_m, args.body_lateral_shift_m)
        dense = make_rows(max(4096, args.samples), args.duty_factor, args.stride_m, args.swing_clearance_m, args.body_lateral_shift_m)
        records = []
        for frequency in (args.low_frequency_hz, args.high_frequency_hz):
            speed = args.stride_m * frequency / args.duty_factor
            if speed <= .004:
                raise ValueError("Derived speed is below existing standstill deadband")
            records.append({
                "schema": "drobot.phase-joint-reference.v1", "sample_count": args.samples,
                "joint_order": crawl.ACTION_ORDER, "joint_position_rad": rows, "forward_speed_m_s": speed,
                "body_lateral_shift_m": [body_lateral_shift(i/args.samples, args.body_lateral_shift_m, args.duty_factor) for i in range(args.samples)],
                "body_lateral_shift_derivative_m_per_cycle": [body_shift_derivative(i/args.samples, args.body_lateral_shift_m, args.duty_factor) for i in range(args.samples)],
                "generation": {
                    "qualification": "UNQUALIFIED_LATERAL_TRANSFER_CRAWL_ABLATION",
                    "method": "shifted-common-floor-crawl-full-3d.v1",
                    "generator_sha256": crawl.digest(Path(__file__)),
                    "geometry_helper_sha256": crawl.digest(Path(geometry.__file__)),
                    "crawl_helper_sha256": crawl.digest(Path(crawl.__file__)),
                    "canonical_urdf_path": CANONICAL_URDF.relative_to(REPOSITORY_ROOT).as_posix(),
                    "canonical_urdf_sha256": crawl.digest(CANONICAL_URDF),
                    "runtime_kinematics_path": RUNTIME_KINEMATICS.relative_to(REPOSITORY_ROOT).as_posix(),
                    "runtime_kinematics_sha256": crawl.digest(RUNTIME_KINEMATICS),
                    "frequency_hz": frequency, "gait_period_s": 1/frequency, "speed_m_s": speed,
                    "duty_factor": args.duty_factor, "phase_offsets_fl_rl_fr_rr": OFFSETS,
                    "stride_m": args.stride_m, "stance_fore_aft_m": geometry.STANCE_FORWARD_M,
                    "common_lowest_edge_depth_m": geometry.COMMON_LOWEST_EDGE_DEPTH_M,
                    "sole_corner_clearance_m": args.swing_clearance_m,
                    "body_lateral_shift_amplitude_m": args.body_lateral_shift_m,
                    "lateral_reference_contract": {
                        "schema": "drobot.phase-body-lateral-reference.v1",
                        "phase_alignment": "Exactly the same phase i/sample_count as joint rows. Low/high vectors are identical; cadence supplies units per second.",
                        "positive_direction": "body translation along episode-left (+Y at initial heading), NOT foot offset",
                        "derivative_units": "metres per unit global cycle phase; multiply by actual phase rate for m/s",
                        "phase_zero_shift_m": -args.body_lateral_shift_m,
                        "cycle_mean_shift_m": 0.,
                        "periodic_peak_lateral_velocity_m_s": 2*args.body_lateral_shift_m/(args.duty_factor-.75)*1.875*frequency,
                        "periodic_peak_lateral_acceleration_m_s2": 2*args.body_lateral_shift_m/(args.duty_factor-.75)**2*(10/math.sqrt(3))*frequency**2,
                        "startup_stop": "Periodic waveform only. Controller startup amplitude ramp and post-run recentering must be handled explicitly; these arrays do not silently redefine a stop target.",
                    },
                    "clearance_envelope": "quintic-symmetric.v1", "clearance_transition_fraction": .5,
                    "body_shift_schedule": {
                        "transfer_negative_to_positive_phase": [0., args.duty_factor-.75],
                        "transfer_positive_to_negative_phase": [.5, args.duty_factor-.25],
                        "interpolation": "quintic smoothstep, zero endpoint velocity/acceleration",
                        "positive_body_Y_held_for": ["rear_right", "front_right"],
                        "negative_body_Y_held_for": ["rear_left", "front_left"],
                    },
                    "contact_schedule": schedule,
                    "kinematics": {
                        "body_axes": "+X forward,+Y left,+Z up",
                        "hip_outward_m": HIP_OUTWARD_M, "hip_drop_m": HIP_DROP_M,
                        "final_abduction_axis_by_leg": ["+X", "+X", "-X", "-X"],
                        "positive_body_shift_joint_sign_by_leg": [-1, -1, 1, 1],
                        "corner_transform": "Construct sagittal corners including hip offset, then Rx(side*abduction). Solve both structural and tread lower-corner depths while every tread center Y=side*hip_outward-body_shift_Y.",
                        "source_anchors": [
                            "simulation/isaac/_quadruped_runtime.py:29-37,809-832 exact lateral translation",
                            "cad/drobot_cad/urdf/quadruped_robot.py:125-126,215-234 hip frame/axis constants (read-only)",
                            "cad/exports/urdf/quadruped_robot.urdf:288-310,424-446,560-582,696-718 canonical joint origins/axes (read-only)",
                            "simulation/isaac/rl/parallel_walking/commanded_walking_env.py:_initialize_sole_diagnostics all eight shoe corners",
                        ],
                        "smooth_abs_pitch_epsilon": geometry.SMOOTH_ABS_EPSILON,
                        "smooth_abs_abduction_epsilon": LATERAL_ABS_EPSILON,
                        "smooth_surface_max_epsilon_m": SURFACE_SMOOTH_MAX_EPSILON_M,
                        "maximum_conservative_corner_gap_m": SMOOTH_GAP_BOUND_M,
                    },
                    "maximum_stance_sole_tilt_deg": 12.,
                    "target_velocity_limit_deg_s_by_action": [60.] * 4 + [90.] * 8,
                    "packet_step_limit_deg": 2.,
                    "exported_table_metrics": metrics(rows, frequency, args.duty_factor, args.stride_m, args.swing_clearance_m, args.body_lateral_shift_m),
                    "dense_analytical_metrics": metrics(dense, frequency, args.duty_factor, args.stride_m, args.swing_clearance_m, args.body_lateral_shift_m),
                    "forward_push_derivation": "All planted tread centers retain dx_body/dt=-stride*frequency/duty. During all-support transfers their Y changes together by -d(body_shift_Y)/dt; body+Y targets are opposite-side support shifts, not commanded net lateral travel.",
                    "limitations": [
                        "Analytical preparation only: no dynamics, policy, hardware or whole-robot balance validation.",
                        "The measured rear-support boundary also has a forward normal; this lateral-only ablation may remain outside the conservative center-point triangle.",
                        "No guarantee the physical COM follows the requested body shift, or that scheduled support becomes measured load support.",
                        "Root body is assumed level. Shoe finite contact patch/rolling, friction and loaded servo tracking remain dynamic concerns.",
                        "Joint-center Y matching does not prove actual pressure-center no-slip motion.",
                        "The unchanged two-degree cap may still clip packets after sample holds; startup and stop blending are not bounded by periodic rates.",
                        "All-support transfers occupy only duty-.75 of a cycle: lateral body velocity/acceleration can be appreciable even at this low forward speed.",
                        "Payload, mass, COM, actuator gains, torque caps and safety limits are unchanged.",
                    ],
                },
            })
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, record in zip(paths, records):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, indent=2, allow_nan=False); stream.write("\n")
        g = record["generation"]
        print(f"Wrote {path}; ANALYTICAL_UNQUALIFIED; body sway +/-{args.body_lateral_shift_m*1000:.1f}mm, {g['frequency_hz']:.3f}Hz, max rate {max(g['dense_analytical_metrics']['maximum_joint_target_velocity_deg_s']):.3f}deg/s")


if __name__ == "__main__":
    main()
