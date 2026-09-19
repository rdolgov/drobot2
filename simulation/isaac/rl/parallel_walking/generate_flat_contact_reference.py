"""Generate the NEW V44 motion prior; never modify V43's recorded tables.

This is an analytical reference, not a learned policy or evidence of stability.
The rigid foot has no ankle: flat support entails a changing contact-center
height. Runtime body pose, contact and joint response must still be learned.
Only sagittal/body-relative corner clearance is constructed here. Simulation
rewards use world-frame corners and actual link states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PROXIMAL_M = 0.159896689
DISTAL_CONTACT_M = PROXIMAL_M + 0.031
SOLE_HALF_LENGTH_M = 0.050
STANCE_FORWARD_M = 0.080
DUTY = 0.55
CORNER_CLEARANCE_M = 0.010
PROFILE = "flat-contact-hardware-matched-run-stop-external-rear-payload"
ACTION_ORDER = tuple(f"{leg}_{joint}" for joint in ("hip_abduction", "hip_flexion", "knee")
                     for leg in ("front_left", "rear_left", "front_right", "rear_right"))
LEGS = ((0, 4, 8, 0.0, 1.0), (1, 5, 9, 0.5, -1.0),
        (2, 6, 10, 0.5, 1.0), (3, 7, 11, 0.0, -1.0))


def smoothstep(u: float) -> float:
    """Quintic curve with zero first and second derivatives at its ends."""
    return u**3 * (10 + u * (-15 + 6 * u))


def inverse_sagittal(forward: float, down: float, front_sign: float) -> tuple[float, float]:
    cosine = (forward**2 + down**2 - PROXIMAL_M**2 - DISTAL_CONTACT_M**2) / (
        2 * PROXIMAL_M * DISTAL_CONTACT_M
    )
    if not -1.0 - 1e-10 <= cosine <= 1.0 + 1e-10:
        raise ValueError("Reference is unreachable; do not project or silently clip it")
    knee = -front_sign * math.acos(max(-1.0, min(1.0, cosine)))
    hip = math.atan2(forward, down) - math.atan2(
        DISTAL_CONTACT_M * math.sin(knee), PROXIMAL_M + DISTAL_CONTACT_M * math.cos(knee)
    )
    return hip, knee


def foot_pose(forward: float, clearance: float, front_sign: float) -> tuple[float, float]:
    """Flat at stance; lift the low edge, not only the center, during swing.

    Solve down + L/2*abs(sin(hip+knee)) = flat_down - clearance.
    flat_down is a body-relative support locus, NOT a measured ground height.
    Abduction is zero in this prior, so lateral corners share the same height.
    """
    if abs(forward) >= PROXIMAL_M or front_sign * forward <= 0:
        raise ValueError("The selected flat-sole IK branch needs signed reach margin")
    flat_down = math.sqrt(PROXIMAL_M**2 - forward**2) + DISTAL_CONTACT_M
    if clearance <= 1e-12:
        hip = math.asin(forward / PROXIMAL_M)
        return hip, -hip
    low, high = 0.150, flat_down
    target = flat_down - clearance

    def lowest_edge_down(down: float) -> float:
        hip, knee = inverse_sagittal(forward, down, front_sign)
        return down + SOLE_HALF_LENGTH_M * abs(math.sin(hip + knee))

    if lowest_edge_down(low) >= target or lowest_edge_down(high) < target:
        raise ValueError("No bracket for swing corner clearance")
    for _ in range(60):
        middle = (low + high) / 2
        if lowest_edge_down(middle) > target:
            high = middle
        else:
            low = middle
    hip, knee = inverse_sagittal(forward, (low + high) / 2, front_sign)
    if abs(lowest_edge_down((low + high) / 2) - target) > 1e-8:
        raise ValueError("Corner clearance solution did not converge")
    return hip, knee


def make_reference(speed: float, frequency: float, count: int = 512) -> dict:
    # Ideal no-slip advance from stance distance/time only, not a claimed speed.
    stride = speed * DUTY / frequency
    rows = []
    for i in range(count):
        row = [0.0] * 12
        for abd, hip_idx, knee_idx, offset, sign in LEGS:
            phase = (i / count + offset) % 1.0
            stance = phase < DUTY
            u = phase / DUTY if stance else (phase - DUTY) / (1 - DUTY)
            sweep = 0.5 - smoothstep(u) if stance else smoothstep(u) - 0.5
            forward = sign * STANCE_FORWARD_M + stride * sweep
            # Same C2 lift envelope as the world-corner reward. Unlike sin^2,
            # its acceleration joins the stationary stance continuously.
            lift_envelope = smoothstep(min(1.0, u / 0.5)) * smoothstep(min(1.0, (1.0 - u) / 0.5))
            clearance = 0.0 if stance else CORNER_CLEARANCE_M * lift_envelope
            hip, knee = foot_pose(forward, clearance, sign)
            # Match the simulator's 95% soft limits; reject bad geometry rather
            # than hide it with a post-hoc projection or angle smoothing.
            if abs(hip) > math.radians(57) or abs(knee) > math.radians(85.5):
                raise ValueError(f"V44 soft joint limit exceeded at phase row {i}")
            row[abd], row[hip_idx], row[knee_idx] = 0.0, hip, knee
        rows.append(row)
    peak_rates = [max(abs(rows[(i + 1) % count][j] - rows[i][j]) for i in range(count))
                  * count * frequency * 180 / math.pi for j in range(12)]
    if max(peak_rates) > 90.0:
        raise ValueError("The prior exceeds the V44 90 deg/s envelope; redesign, do not clip")
    return {
        "schema": "drobot.phase-joint-reference.v1", "sample_count": count,
        "joint_order": ACTION_ORDER, "joint_position_rad": rows,
        "forward_speed_m_s": speed,
        "generation": {
            "method": "flat-sole-corner-clearance.v1", "training_profile": PROFILE,
            "duty_factor": DUTY, "phase_offsets_fl_rl_fr_rr": [0.0, 0.5, 0.5, 0.0],
            "speed_m_s": speed, "frequency_hz": frequency,
            "stance_fore_aft_m": STANCE_FORWARD_M,
            "sole_corner_clearance_m": CORNER_CLEARANCE_M,
            "clearance_envelope": "quintic-symmetric.v1",
            "clearance_transition_fraction": 0.5,
            "target_velocity_limit_deg_s": 90.0, "packet_step_limit_deg": 2.0,
            "stride_m": stride, "maximum_joint_target_velocity_deg_s": peak_rates,
            "geometry": {"proximal_m": PROXIMAL_M, "distal_contact_m": DISTAL_CONTACT_M,
                         "sole_fore_aft_m": 2 * SOLE_HALF_LENGTH_M},
            "limitations": [
                "No policy inference, dynamics, floor contact or hardware validation implied.",
                "Support soles flat relative to body; support depth varies with sweep.",
                "Diagonal support heights can differ: body pitch and small residuals must reconcile contact.",
                "Corner lift is relative to the local flat support locus, not verified world-floor clearance.",
                "Interpolated tables, body tilt, compliance and loaded tracking can reduce clearance.",
                "Nominal low-speed corrective stage, not a validated running or flight-phase gait.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("references"))
    args = parser.parse_args()
    outputs = [("v44_flat_contact_low.json", .03, .35), ("v44_flat_contact_high.json", .05, .45)]
    # Generate both in memory and reject existing targets before writing either.
    destinations = [args.output_dir / name for name, _, _ in outputs]
    if any(path.exists() for path in destinations):
        parser.error("V44 reference exists; use a new output directory to preserve provenance")
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payloads = [make_reference(speed, frequency) for _, speed, frequency in outputs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in zip(destinations, payloads):
        payload["generation"]["generator_sha256"] = source_sha256
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        print(f"Wrote {path.name}; analytical candidate only, not evaluated")


if __name__ == "__main__":
    main()
