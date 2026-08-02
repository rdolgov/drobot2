"""Search staged four-foot COM preload before rear-left stair-foot unload.

V46 tipped while commanding the complete rear-right-to-rear-left transfer
from one stale reference origin.  This V47 experiment restores the exact V46
physical boundary, searches small analytic COM increments with all four feet
loaded, and re-anchors every accepted increment at the measured articulation
and composite COM.  No camera pixels or learned actions enter this search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from collections.abc import Mapping
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def comma_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError("candidate values must be finite")
    return result


def target_deltas(value: str) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for item in value.split(","):
        if not item.strip():
            continue
        fields = item.split(":")
        if len(fields) != 2:
            raise ValueError("target deltas must use forward:lateral pairs")
        forward_m, lateral_m = (float(field.strip()) for field in fields)
        if (
            not math.isfinite(forward_m)
            or not math.isfinite(lateral_m)
            or abs(forward_m) > 0.05
            or abs(lateral_m) > 0.05
        ):
            raise ValueError("each preload delta must be within +/-50 mm")
        result.append((forward_m, lateral_m))
    if not result:
        raise ValueError("at least one target delta is required")
    return tuple(result)


def json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def encoded_json(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(json_compatible(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_stairs_v46_rear_right_sidestep.yaml"),
)
parser.add_argument(
    "--phase-snapshot",
    default=(
        "simulation/isaac/output/rl/"
        "rear-left-transfer-snapshot-v46-seed937.json"
    ),
)
parser.add_argument("--seed", type=int, default=945)
parser.add_argument(
    "--target-deltas-m",
    default="0.005:-0.005",
    help="Comma-separated forward:lateral increments from each settled state.",
)
parser.add_argument("--durations-seconds", default="5.0")
parser.add_argument(
    "--pitch-feedback-modes",
    default="off",
    help="Comma-separated transfer pitch modes: off, front_only, or all.",
)
parser.add_argument("--load-sharing-gains-m", default="0.060")
parser.add_argument("--load-sharing-maximum-correction-m", type=float, default=0.020)
parser.add_argument("--load-sharing-smoothing-factor", type=float, default=0.50)
parser.add_argument("--maximum-stages", type=int, default=6)
parser.add_argument("--settle-hold-seconds", type=float, default=0.40)
parser.add_argument("--minimum-stage-margin-progress-m", type=float, default=0.002)
parser.add_argument("--target-support-margin-m", type=float, default=0.015)
parser.add_argument("--maximum-target-error-m", type=float, default=0.015)
parser.add_argument("--maximum-base-speed-m-s", type=float, default=0.030)
parser.add_argument("--maximum-body-rate-rad-s", type=float, default=0.250)
parser.add_argument("--maximum-body-tilt-deg", type=float, default=12.0)
parser.add_argument("--minimum-all-foot-load-n", type=float, default=1.0)
parser.add_argument("--maximum-all-foot-slip-m", type=float, default=0.022)
parser.add_argument(
    "--report",
    default=(
        "simulation/isaac/output/rl/"
        "rear-left-progressive-preload-v47-seed945.json"
    ),
)
parser.add_argument(
    "--save-transfer-snapshot",
    default=(
        "simulation/isaac/output/rl/"
        "rear-left-transfer-snapshot-v47-seed945.json"
    ),
)
args, _ = parser.parse_known_args()

deltas = target_deltas(args.target_deltas_m)
durations = comma_floats(args.durations_seconds)
load_sharing_gains = comma_floats(args.load_sharing_gains_m)
pitch_modes = tuple(
    item.strip().lower()
    for item in args.pitch_feedback_modes.split(",")
    if item.strip()
)
if not pitch_modes or any(
    value not in {"off", "front_only", "all"} for value in pitch_modes
):
    parser.error("pitch feedback modes must be off, front_only, or all")
if any(value <= 0.0 or value > 0.20 for value in load_sharing_gains):
    parser.error("load-sharing gains must be within (0, 0.20]")
if not 0.0 < args.load_sharing_maximum_correction_m <= 0.05:
    parser.error("load-sharing maximum correction must be within (0, 0.05]")
if not 0.0 < args.load_sharing_smoothing_factor <= 1.0:
    parser.error("load-sharing smoothing factor must be within (0, 1]")
if any(value <= 0.0 or value > 10.0 for value in durations):
    parser.error("durations must be within (0, 10] seconds")
if args.maximum_stages < 1 or args.maximum_stages > 12:
    parser.error("--maximum-stages must be within [1, 12]")
if args.settle_hold_seconds <= 0.0:
    parser.error("--settle-hold-seconds must be positive")
if args.minimum_stage_margin_progress_m < 0.0:
    parser.error("--minimum-stage-margin-progress-m cannot be negative")
if args.target_support_margin_m <= 0.0:
    parser.error("--target-support-margin-m must be positive")

config_path = project_path(args.config)
snapshot_path = project_path(args.phase_snapshot)
report_path = project_path(args.report)
saved_snapshot_path = project_path(args.save_transfer_snapshot)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
task_config = deepcopy(config["task"])
source_task_id = str(task_config["id"])
task_config["id"] = "Drobot-Quadruped-Stairs-v47-Progressive-Rear-Left-Preload"
preload_load_config = task_config["placement_reference"]["inter_leg_transfer"][
    "com_regulation"
].setdefault("four_foot_preload_load_sharing", {})
preload_load_config.update(
    {
        "enabled": True,
        "next_swing_legs": ["rear_left"],
        "proportional_gain_m": load_sharing_gains[0],
        "maximum_correction_m": args.load_sharing_maximum_correction_m,
        "smoothing_factor": args.load_sharing_smoothing_factor,
        "minimum_total_load_n": 1.0,
    }
)
with snapshot_path.open("r", encoding="utf-8") as stream:
    snapshot_wrapper = json.load(stream)
source_snapshot = deepcopy(snapshot_wrapper["snapshot"])

expected_contract = {
    "stair_rise_m": 0.18,
    "stair_tread_depth_m": 0.25,
    "effort_cap_nm": 0.8825985,
    "target_leg": "rear_left",
    "phase_snapshot_mode": "inter_leg_transfer",
}
actual_contract = {
    "stair_rise_m": float(task_config["staircase"]["rise_m"]),
    "stair_tread_depth_m": float(task_config["staircase"]["tread_depth_m"]),
    "effort_cap_nm": float(task_config["robot_hardware_profile"]["effort_cap_nm"]),
    "target_leg": snapshot_wrapper.get("target_leg"),
    "phase_snapshot_mode": snapshot_wrapper.get("phase_snapshot_mode"),
}
for field, expected in expected_contract.items():
    actual = actual_contract[field]
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9):
            parser.error(f"{field} changed: {actual!r} != {expected!r}")
        if not math.isclose(
            float(snapshot_wrapper[field]), expected, rel_tol=0.0, abs_tol=1e-9
        ):
            parser.error(f"phase snapshot {field} changed")
    elif actual != expected:
        parser.error(f"{field} changed: {actual!r} != {expected!r}")
if tuple(snapshot_wrapper["placement_sequence_legs"]) != tuple(
    task_config["placement_reference"]["sequence_legs"]
):
    parser.error("phase snapshot placement sequence changed")

world_path = project_path(task_config["world"])

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from _quadruped_runtime import LEGS  # noqa: E402
from _quadruped_stairs_env import (  # noqa: E402
    QuadrupedStairsEnv,
    _support_triangle_signed_margin_m,
)
from _stair_rl_contract import (  # noqa: E402
    reanchor_inter_leg_transfer_snapshot,
)

report: dict[str, object] = {
    "status": "ERROR",
    "strict_pass": False,
    "task_id": task_config["id"],
    "source_task_id": source_task_id,
    "scope": "Exact-V46-snapshot progressive four-foot rear-left COM preload",
    "config": str(config_path),
    "phase_snapshot": str(snapshot_path),
    "seed": args.seed,
    **expected_contract,
    "camera_policy_input": False,
    "learned_action_input": False,
    "candidate_target_deltas_m": [list(value) for value in deltas],
    "candidate_durations_seconds": list(durations),
    "candidate_pitch_feedback_modes": list(pitch_modes),
    "candidate_load_sharing_gains_m": list(load_sharing_gains),
    "load_sharing_maximum_correction_m": (
        args.load_sharing_maximum_correction_m
    ),
    "load_sharing_smoothing_factor": args.load_sharing_smoothing_factor,
    "maximum_stages": args.maximum_stages,
}
raw_env: QuadrupedStairsEnv | None = None
exit_code = 1
try:
    raw_env = QuadrupedStairsEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
    )
    raw_env.set_evaluation_level(1)
    raw_env.set_placement_level("left-center-tread-load")
    rear_override = raw_env.inter_leg_transfer_config[
        "override_by_next_swing_leg"
    ]["rear_left"]
    configured_maximum_seconds = float(rear_override["maximum_seconds"])
    # The search owns only the four-foot preload. An intentionally impossible
    # support-margin gate prevents rear-left unload during every candidate.
    rear_override["minimum_support_margin_m"] = 0.50
    rear_override["minimum_upright_cosine"] = math.cos(
        math.radians(args.maximum_body_tilt_deg)
    )

    original_start_balance = np.asarray(
        source_snapshot["placement_transfer_start_balance_position_m"],
        dtype=np.float64,
    )
    original_target_balance = np.asarray(
        source_snapshot["placement_transfer_target_balance_position_m"],
        dtype=np.float64,
    )
    baseline_snapshot = deepcopy(source_snapshot)
    accepted_stages: list[dict[str, object]] = []
    stage_searches: list[dict[str, object]] = []
    achieved_margin_m = -math.inf
    final_balance_m = original_start_balance.copy()
    final_snapshot: dict[str, object] | None = None
    settle_hold_steps = max(
        1, int(math.ceil(args.settle_hold_seconds * raw_env.control_hz))
    )
    zero_action = np.zeros(raw_env.action_space.shape, dtype=np.float32)

    for stage_index in range(args.maximum_stages):
        stage_candidates: list[dict[str, object]] = []
        captured_by_id: dict[str, dict[str, object]] = {}
        for pitch_mode, load_sharing_gain_m, duration_seconds, delta_xy in product(
            pitch_modes, load_sharing_gains, durations, deltas
        ):
            raw_env.pitch_feedback_config["enabled"] = pitch_mode != "off"
            raw_env.pitch_feedback_enabled = pitch_mode != "off"
            raw_env.pitch_feedback_config.setdefault(
                "inter_leg_transfer_front_only_by_swing_leg", {}
            )["rear_left"] = pitch_mode == "front_only"
            raw_env.four_foot_preload_load_sharing_config[
                "proportional_gain_m"
            ] = load_sharing_gain_m
            rear_override["duration_seconds"] = float(duration_seconds)
            rear_override["maximum_seconds"] = max(
                configured_maximum_seconds,
                float(duration_seconds) + args.settle_hold_seconds + 3.0,
            )
            candidate_id = (
                f"stage{stage_index + 1}-pitch{pitch_mode}-"
                f"loadgain{load_sharing_gain_m:.3f}-"
                f"duration{duration_seconds:.1f}-"
                f"forward{delta_xy[0]:+.3f}-lateral{delta_xy[1]:+.3f}"
            )
            candidate_snapshot = reanchor_inter_leg_transfer_snapshot(
                baseline_snapshot,
                balance_position_m=baseline_snapshot[
                    "placement_transfer_start_balance_position_m"
                ],
                target_delta_xy_m=delta_xy,
                reference_by_leg=baseline_snapshot[
                    "placement_transfer_reference_by_leg"
                ],
            )
            _, reset_info = raw_env.restore_placement_phase_snapshot(
                candidate_snapshot,
                seed=args.seed + stage_index,
            )
            initial_margin_m = _support_triangle_signed_margin_m(
                raw_env.latest_placement_com_position_m[:2],
                raw_env._sample_foot_tips()[  # noqa: SLF001
                    list(raw_env.placement_support_leg_indices), :2
                ],
            )
            initial_balance_m = np.asarray(
                reset_info.get(
                    "placement_balance_position_m",
                    raw_env.latest_placement_com_position_m,
                ),
                dtype=np.float64,
            )
            initial_foot_tips = raw_env._sample_foot_tips().copy()  # noqa: SLF001
            maximum_steps = int(
                math.ceil(
                    (duration_seconds + args.settle_hold_seconds + 2.0)
                    * raw_env.control_hz
                )
            )
            settled_steps = 0
            steps = 0
            terminated = False
            truncated = False
            maximum_tilt_deg = 0.0
            maximum_all_foot_slip_m = 0.0
            initial_total_loads = (
                raw_env.latest_ground_normal_loads_n
                + np.sum(raw_env.latest_step_normal_loads_n, axis=1)
            )
            completed_indices = [
                LEGS.index(leg)
                for leg in raw_env.completed_placement_legs
            ]
            minimum_all_foot_load_n = float(np.min(initial_total_loads))
            minimum_completed_tread_load_n = float(
                np.min(raw_env.latest_step_normal_loads_n[completed_indices, 0])
            )
            final_margin_m = initial_margin_m
            final_target_error_m = math.inf
            final_balance_m = initial_balance_m.copy()
            last_info = dict(reset_info)
            for step_index in range(1, maximum_steps + 1):
                steps = step_index
                _, _, terminated, truncated, info = raw_env.step(zero_action)
                last_info = dict(info)
                final_margin_m = float(
                    info.get("placement_support_margin_m", -math.inf)
                )
                final_target_error_m = float(
                    info.get("placement_transfer_base_target_error_m", math.inf)
                )
                final_balance_m = np.asarray(
                    info.get(
                        "placement_balance_position_m",
                        raw_env.latest_placement_com_position_m,
                    ),
                    dtype=np.float64,
                )
                tilt_deg = math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                info.get("placement_upright_cosine", 1.0),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                maximum_tilt_deg = max(maximum_tilt_deg, tilt_deg)
                foot_tips = raw_env._sample_foot_tips()  # noqa: SLF001
                maximum_all_foot_slip_m = max(
                    maximum_all_foot_slip_m,
                    float(
                        np.max(
                            np.linalg.norm(
                                foot_tips[:, :2] - initial_foot_tips[:, :2],
                                axis=1,
                            )
                        )
                    ),
                )
                total_loads = raw_env.latest_ground_normal_loads_n + np.sum(
                    raw_env.latest_step_normal_loads_n,
                    axis=1,
                )
                minimum_all_foot_load_n = min(
                    minimum_all_foot_load_n,
                    float(np.min(total_loads)),
                )
                completed_indices = [
                    LEGS.index(leg)
                    for leg in raw_env.completed_placement_legs
                ]
                completed_tread_load = float(
                    np.min(raw_env.latest_step_normal_loads_n[completed_indices, 0])
                )
                minimum_completed_tread_load_n = min(
                    minimum_completed_tread_load_n,
                    completed_tread_load,
                )
                ready = bool(
                    float(info.get("placement_transfer_fraction", 0.0))
                    >= 1.0 - 1e-6
                    and final_target_error_m <= args.maximum_target_error_m
                    and float(
                        info.get("placement_transfer_base_speed_m_s", math.inf)
                    )
                    <= args.maximum_base_speed_m_s
                    and float(
                        info.get("placement_transfer_body_rate_rad_s", math.inf)
                    )
                    <= args.maximum_body_rate_rad_s
                    and tilt_deg <= args.maximum_body_tilt_deg
                    and float(np.min(total_loads)) >= args.minimum_all_foot_load_n
                    and completed_tread_load >= args.minimum_all_foot_load_n
                    and maximum_all_foot_slip_m <= args.maximum_all_foot_slip_m
                    and raw_env.placement_transfer_unload_start_step is None
                )
                settled_steps = settled_steps + 1 if ready else 0
                if settled_steps >= settle_hold_steps:
                    captured = raw_env.capture_placement_phase_snapshot()
                    references = raw_env._reference_parameters_from_joint_positions(  # noqa: SLF001
                        np.asarray(captured["joint_positions_rad"], dtype=np.float32)
                    )
                    captured_by_id[candidate_id] = (
                        reanchor_inter_leg_transfer_snapshot(
                            captured,
                            balance_position_m=final_balance_m,
                            target_delta_xy_m=(0.0, 0.0),
                            reference_by_leg=references,
                        )
                    )
                    break
                if terminated or truncated:
                    break

            margin_progress_m = final_margin_m - initial_margin_m
            final_total_loads = (
                raw_env.latest_ground_normal_loads_n
                + np.sum(raw_env.latest_step_normal_loads_n, axis=1)
            )
            final_completed_tread_loads = raw_env.latest_step_normal_loads_n[
                completed_indices, 0
            ]
            settled = candidate_id in captured_by_id
            accepted = bool(
                settled
                and not terminated
                and not truncated
                and not last_info.get("failure_reasons")
                and (
                    margin_progress_m >= args.minimum_stage_margin_progress_m
                    or final_margin_m >= args.target_support_margin_m
                )
            )
            result = {
                "id": candidate_id,
                "stage": stage_index + 1,
                "duration_seconds": duration_seconds,
                "pitch_feedback_mode": pitch_mode,
                "load_sharing_gain_m": load_sharing_gain_m,
                "target_delta_xy_m": list(delta_xy),
                "steps": steps,
                "settled": settled,
                "accepted": accepted,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "failure_reasons": list(last_info.get("failure_reasons", ())),
                "initial_support_margin_m": initial_margin_m,
                "final_support_margin_m": final_margin_m,
                "support_margin_progress_m": margin_progress_m,
                "initial_balance_position_m": initial_balance_m.tolist(),
                "final_balance_position_m": final_balance_m.tolist(),
                "measured_balance_delta_xy_m": (
                    final_balance_m[:2] - initial_balance_m[:2]
                ).tolist(),
                "final_target_error_m": final_target_error_m,
                "maximum_body_tilt_deg": maximum_tilt_deg,
                "maximum_all_foot_slip_m": maximum_all_foot_slip_m,
                "minimum_all_foot_load_n": minimum_all_foot_load_n,
                "minimum_completed_tread_load_n": minimum_completed_tread_load_n,
                "final_all_foot_loads_n": final_total_loads.tolist(),
                "final_completed_tread_loads_n": (
                    final_completed_tread_loads.tolist()
                ),
                "final_base_speed_m_s": float(
                    last_info.get("placement_transfer_base_speed_m_s", math.inf)
                ),
                "final_body_rate_rad_s": float(
                    last_info.get("placement_transfer_body_rate_rad_s", math.inf)
                ),
                "final_transfer_fraction": float(
                    last_info.get("placement_transfer_fraction", 0.0)
                ),
                "final_pre_unload_gate_failures": list(
                    last_info.get("placement_pre_unload_gate_failures", ())
                ),
            }
            stage_candidates.append(result)
            print(
                "DROBOT_PROGRESSIVE_PRELOAD_CANDIDATE="
                + json.dumps(
                    {
                        "id": candidate_id,
                        "accepted": accepted,
                        "margin_mm": round(1000.0 * final_margin_m, 2),
                        "progress_mm": round(1000.0 * margin_progress_m, 2),
                        "target_error_mm": round(1000.0 * final_target_error_m, 2),
                        "tilt_deg": round(maximum_tilt_deg, 2),
                        "failures": result["failure_reasons"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        stage_candidates.sort(
            key=lambda item: (
                bool(item["accepted"]),
                float(item["final_support_margin_m"]),
                -float(item["final_target_error_m"]),
                -float(item["maximum_body_tilt_deg"]),
            ),
            reverse=True,
        )
        best = stage_candidates[0]
        stage_searches.append(
            {
                "stage": stage_index + 1,
                "accepted_count": sum(
                    bool(item["accepted"]) for item in stage_candidates
                ),
                "best": best,
                "ranked_candidates": stage_candidates,
            }
        )
        if not best["accepted"]:
            break
        accepted_stages.append(best)
        baseline_snapshot = captured_by_id[str(best["id"])]
        final_snapshot = deepcopy(baseline_snapshot)
        achieved_margin_m = float(best["final_support_margin_m"])
        final_balance_m = np.asarray(
            best["final_balance_position_m"], dtype=np.float64
        )
        print(
            "DROBOT_PROGRESSIVE_PRELOAD_STAGE_ACCEPTED="
            + json.dumps(
                {
                    "stage": stage_index + 1,
                    "candidate": best["id"],
                    "support_margin_mm": round(1000.0 * achieved_margin_m, 2),
                    "balance_position_m": final_balance_m.tolist(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if achieved_margin_m >= args.target_support_margin_m:
            break

    strict_pass = bool(
        final_snapshot is not None
        and achieved_margin_m >= args.target_support_margin_m
    )
    saved_snapshot = None
    if final_snapshot is not None:
        snapshot_payload = {
            "schema_version": 1,
            "task_id": task_config["id"],
            "source_task_id": source_task_id,
            "target_leg": "rear_left",
            "phase_snapshot_mode": "inter_leg_transfer",
            "placement_sequence_legs": list(
                task_config["placement_reference"]["sequence_legs"]
            ),
            "stair_rise_m": expected_contract["stair_rise_m"],
            "stair_tread_depth_m": expected_contract["stair_tread_depth_m"],
            "effort_cap_nm": expected_contract["effort_cap_nm"],
            "seed": args.seed,
            "target_support_margin_reached": strict_pass,
            "achieved_support_margin_m": achieved_margin_m,
            "accepted_preload_stages": accepted_stages,
            "snapshot": final_snapshot,
        }
        snapshot_bytes = encoded_json(snapshot_payload)
        saved_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        saved_snapshot_path.write_bytes(snapshot_bytes)
        saved_snapshot = {
            "path": str(saved_snapshot_path),
            "sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        }

    report.update(
        {
            "status": "PASS" if strict_pass else "NO_ACCEPTED_PATH",
            "strict_pass": strict_pass,
            "original_start_balance_position_m": original_start_balance.tolist(),
            "original_target_balance_position_m": original_target_balance.tolist(),
            "original_target_delta_xy_m": (
                original_target_balance[:2] - original_start_balance[:2]
            ).tolist(),
            "accepted_stage_count": len(accepted_stages),
            "accepted_stages": accepted_stages,
            "achieved_support_margin_m": (
                achieved_margin_m if math.isfinite(achieved_margin_m) else None
            ),
            "final_balance_position_m": final_balance_m.tolist(),
            "stage_searches": stage_searches,
            "saved_transfer_snapshot": saved_snapshot,
        }
    )
    exit_code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    if raw_env is not None:
        raw_env.close()
    report_bytes = encoded_json(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(report_bytes)
    print(
        "DROBOT_PROGRESSIVE_PRELOAD_SEARCH="
        + json.dumps(
            {
                "status": report["status"],
                "strict_pass": report.get("strict_pass", False),
                "accepted_stage_count": report.get("accepted_stage_count", 0),
                "achieved_support_margin_m": report.get(
                    "achieved_support_margin_m"
                ),
                "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    simulation_app.close()

sys.exit(exit_code)
