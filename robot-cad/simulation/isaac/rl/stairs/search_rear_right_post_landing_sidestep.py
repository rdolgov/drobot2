"""Search a measured rear-right sidestep after its accepted first landing.

The search restores the exact V45 rear-left transfer boundary, rewinds only
the placement state machine to rear-right, and runs a second force-backed
placement with zero learned residual action. Candidates pass only when the
physical foot contact moves outward, re-lands on the tread, and the other
three feet retain a positive composite-COM support margin throughout. The
following transfer remains responsible for moving COM into the new polygon.
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
        raise ValueError("candidate list must contain finite numbers")
    return result


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


def write_json(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(json_compatible(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_stairs_v46_rear_right_sidestep.yaml"),
)
parser.add_argument(
    "--phase-snapshot",
    default=(
        "simulation/isaac/models/"
        "ppo-stairs-v45-rear-left-dynamic-transfer-4096/"
        "phase_snapshot_seed870.json"
    ),
)
parser.add_argument("--seed", type=int, default=877)
parser.add_argument(
    "--outward-offsets-m",
    default=None,
    help="Optional comma-separated override for config candidates.",
)
parser.add_argument(
    "--relative-apex-lifts-m",
    default=None,
    help="Optional comma-separated re-lifts above the first landing.",
)
parser.add_argument(
    "--forward-offsets-m",
    default=None,
    help="Optional comma-separated tread-depth corrections.",
)
parser.add_argument(
    "--minimum-physical-outward-displacement-m",
    type=float,
    default=None,
    help=(
        "Optional acceptance override; a small negative value permits a "
        "force-backed inward settle instead of requiring a sidestep."
    ),
)
parser.add_argument(
    "--minimum-rear-right-tread-load-n",
    type=float,
    default=None,
)
parser.add_argument(
    "--maximum-forward-foot-drift-m",
    type=float,
    default=None,
    help="Optional acceptance override for an intentional rear-edge step.",
)
parser.add_argument("--foot-pad-thickness-m", type=float)
parser.add_argument("--foot-pad-width-m", type=float)
parser.add_argument("--foot-pad-length-m", type=float)
parser.add_argument(
    "--foot-pad-shape",
    choices=("box", "sphere"),
    default="box",
)
parser.add_argument("--foot-pad-radius-m", type=float)
parser.add_argument(
    "--foot-pad-contact-plane-offset-m",
    type=float,
    default=0.0125,
)
parser.add_argument(
    "--report",
    default=(
        "simulation/isaac/output/rl/"
        "rear-right-post-landing-sidestep-v46-seed877.json"
    ),
)
parser.add_argument(
    "--save-transfer-snapshot",
    default=(
        "simulation/isaac/output/rl/"
        "rear-left-transfer-snapshot-v46-seed877.json"
    ),
)
args, _ = parser.parse_known_args()

config_path = project_path(args.config)
phase_snapshot_path = project_path(args.phase_snapshot)
report_path = project_path(args.report)
saved_snapshot_path = project_path(args.save_transfer_snapshot)

with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
task_config = dict(config["task"])
reposition_config = dict(task_config["placement_reference"]["post_landing_reposition"])
acceptance = dict(reposition_config["acceptance"])
if args.minimum_physical_outward_displacement_m is not None:
    acceptance["minimum_physical_outward_displacement_m"] = (
        args.minimum_physical_outward_displacement_m
    )
if args.minimum_rear_right_tread_load_n is not None:
    acceptance["minimum_rear_right_tread_load_n"] = (
        args.minimum_rear_right_tread_load_n
    )
if args.maximum_forward_foot_drift_m is not None:
    acceptance["maximum_forward_foot_drift_m"] = (
        args.maximum_forward_foot_drift_m
    )
outward_offsets_m = (
    comma_floats(args.outward_offsets_m)
    if args.outward_offsets_m
    else tuple(float(value) for value in reposition_config["outward_offsets_m"])
)
relative_apex_lifts_m = (
    comma_floats(args.relative_apex_lifts_m)
    if args.relative_apex_lifts_m
    else tuple(float(value) for value in reposition_config["relative_apex_lifts_m"])
)
forward_offsets_m = (
    comma_floats(args.forward_offsets_m)
    if args.forward_offsets_m
    else tuple(float(value) for value in reposition_config["forward_offsets_m"])
)
if any(value < -0.10 or value > 0.15 for value in outward_offsets_m):
    parser.error("outward offsets must be within [-0.10, 0.15] m")
if any(value <= 0.0 or value > 0.10 for value in relative_apex_lifts_m):
    parser.error("relative apex lifts must be within (0, 0.10] m")
if any(value < -0.05 or value > 0.05 for value in forward_offsets_m):
    parser.error("forward offsets must be within [-0.05, 0.05] m")
if not -0.10 <= float(
    acceptance["minimum_physical_outward_displacement_m"]
) <= 0.15:
    parser.error(
        "minimum physical outward displacement must be within [-0.10, 0.15] m"
    )
if not 1.0 <= float(acceptance["minimum_rear_right_tread_load_n"]) <= 50.0:
    parser.error("minimum rear-right tread load must be within [1, 50] N")
if not 0.0 < float(acceptance["maximum_forward_foot_drift_m"]) <= 0.10:
    parser.error("maximum forward foot drift must be within (0, 0.10] m")
foot_pad_dimensions = (
    args.foot_pad_thickness_m,
    args.foot_pad_width_m,
    args.foot_pad_length_m,
)
foot_pad_requested = any(value is not None for value in foot_pad_dimensions) or (
    args.foot_pad_radius_m is not None
)
if (
    foot_pad_requested
    and args.foot_pad_shape == "box"
    and any(value is None for value in foot_pad_dimensions)
):
    parser.error("foot-pad thickness, width, and length must be supplied together")
if (
    foot_pad_requested
    and args.foot_pad_shape == "sphere"
    and args.foot_pad_radius_m is None
):
    parser.error("--foot-pad-radius-m is required for a sphere foot pad")
if foot_pad_requested:
    foot_contact_patch: dict[str, object] = {
        "enabled": True,
        "id": "simulation-rubber-pad-v50-rear-right-foothold",
        "shape": args.foot_pad_shape,
        "legs": ["rear_right"],
        "contact_plane_offset_m": args.foot_pad_contact_plane_offset_m,
        "contact_offset_m": 0.002,
        "rest_offset_m": 0.0,
    }
    if args.foot_pad_shape == "box":
        foot_contact_patch.update(
            {
                "thickness_m": args.foot_pad_thickness_m,
                "width_m": args.foot_pad_width_m,
                "length_m": args.foot_pad_length_m,
            }
        )
    else:
        foot_contact_patch["radius_m"] = args.foot_pad_radius_m
    task_config["foot_contact_patch"] = foot_contact_patch

with phase_snapshot_path.open("r", encoding="utf-8") as stream:
    snapshot_wrapper = json.load(stream)
source_snapshot = dict(snapshot_wrapper["snapshot"])

expected_contract = {
    "stair_rise_m": 0.18,
    "stair_tread_depth_m": 0.25,
    "effort_cap_nm": 0.8825985,
}
actual_contract = {
    "stair_rise_m": float(task_config["staircase"]["rise_m"]),
    "stair_tread_depth_m": float(task_config["staircase"]["tread_depth_m"]),
    "effort_cap_nm": float(task_config["robot_hardware_profile"]["effort_cap_nm"]),
}
force_backed_settle = (
    args.minimum_physical_outward_displacement_m is not None
    or args.minimum_rear_right_tread_load_n is not None
)
inward_rear_edge_requested = any(
    value < 0.0 for value in (*outward_offsets_m, *forward_offsets_m)
)
if foot_pad_requested:
    task_config["id"] = (
        "Drobot-Quadruped-Stairs-v50-Rubber-Pad-Rear-Right-Foothold"
    )
elif inward_rear_edge_requested:
    task_config["id"] = (
        "Drobot-Quadruped-Stairs-v52-Inward-Rear-Edge-Reposition"
    )
elif force_backed_settle:
    task_config["id"] = (
        "Drobot-Quadruped-Stairs-v48-Force-Backed-Rear-Right-Foothold"
    )
for key, expected in expected_contract.items():
    if not math.isclose(actual_contract[key], expected, rel_tol=0.0, abs_tol=1e-9):
        parser.error(f"{key} changed: {actual_contract[key]} != {expected}")
    if not math.isclose(float(snapshot_wrapper[key]), expected, rel_tol=0.0, abs_tol=1e-9):
        parser.error(f"phase snapshot {key} changed")
if snapshot_wrapper.get("phase_snapshot_mode") != "inter_leg_transfer":
    parser.error("phase snapshot must be an inter-leg-transfer boundary")
if snapshot_wrapper.get("target_leg") != "rear_left":
    parser.error("phase snapshot must target rear_left")
if tuple(snapshot_wrapper["placement_sequence_legs"]) != tuple(
    task_config["placement_reference"]["sequence_legs"]
):
    parser.error("phase snapshot placement sequence changed")

world_path = project_path(task_config["world"])

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from _quadruped_stairs_env import (  # noqa: E402
    LEGS,
    QuadrupedStairsEnv,
    _support_triangle_signed_margin_m,
)
from _stair_rl_contract import post_landing_reposition_snapshot  # noqa: E402

report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "source_task_id": snapshot_wrapper.get("source_task_id"),
    "scope": (
        "Exact-snapshot rubber-pad rear-right foothold settle"
        if foot_pad_requested
        else (
            "Exact-snapshot inward rear-edge rear-right reposition"
            if inward_rear_edge_requested
            else (
            "Exact-snapshot force-backed rear-right foothold settle"
            if force_backed_settle
            else "Exact-snapshot rear-right post-landing sidestep"
            )
        )
    ),
    "config": str(config_path),
    "world": str(world_path),
    "phase_snapshot": str(phase_snapshot_path),
    "phase_snapshot_sha256": hashlib.sha256(phase_snapshot_path.read_bytes()).hexdigest(),
    "seed": args.seed,
    **actual_contract,
    "camera_policy_input": False,
    "external_camera_used": False,
    "candidate_count": (
        len(outward_offsets_m)
        * len(relative_apex_lifts_m)
        * len(forward_offsets_m)
    ),
    "outward_offsets_m": list(outward_offsets_m),
    "relative_apex_lifts_m": list(relative_apex_lifts_m),
    "forward_offsets_m": list(forward_offsets_m),
    "acceptance": acceptance,
    "foot_contact_patch": task_config.get(
        "foot_contact_patch",
        {"enabled": False},
    ),
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
    raw_env.set_placement_level(str(source_snapshot["placement_curriculum_level"]))
    rewound_snapshot = post_landing_reposition_snapshot(
        source_snapshot,
        leg=str(reposition_config["leg"]),
    )
    rear_right_index = LEGS.index("rear_right")
    next_support_indices = [
        LEGS.index(leg) for leg in ("front_left", "front_right", "rear_right")
    ]
    baseline_lift_m = float(rewound_snapshot["placement_leg_baseline_lift_offset_m"])
    level_template = deepcopy(dict(reposition_config["level"]))
    timing_template = deepcopy(dict(reposition_config["timing"]))
    maximum_steps = int(reposition_config["maximum_steps"])
    minimum_relative_reclearance_m = float(
        reposition_config["minimum_relative_reclearance_m"]
    )
    if maximum_steps < 1:
        raise ValueError("post-landing reposition maximum_steps must be positive")
    if not 0.0 < minimum_relative_reclearance_m <= 0.025:
        raise ValueError(
            "post-landing minimum_relative_reclearance_m must be within "
            "(0, 0.025]"
        )

    # The accepted first landing already contains support-reach and forward
    # carry corrections. Do not apply those one-shot changes a second time.
    raw_env.com_regulation_config.setdefault(
        "support_extension_m_by_swing_leg", {}
    )["rear_right"] = {}
    raw_env.inter_leg_transfer_config.setdefault(
        "post_clearance_body_shift_by_leg", {}
    )["rear_right"] = {}
    raw_env.inter_leg_transfer_config.setdefault(
        "post_clearance_swing_base_delta_end_scale_by_leg", {}
    )["rear_right"] = {"forward": 1.0, "lateral": 1.0, "vertical": 1.0}
    # Keep snapshot restore compatible with the source boundary while still
    # rejecting the low side of a 1-5 N contact oscillation.  A stronger final
    # foothold requirement remains an independent acceptance gate below; using
    # it as the global contact threshold would also require every source
    # support foot to carry that load before the candidate can even start.
    raw_env.placement_reference_config["contact_on_threshold_n"] = max(
        float(raw_env.placement_reference_config["contact_on_threshold_n"]),
        min(float(acceptance["minimum_rear_right_tread_load_n"]), 2.0),
    )

    candidates: list[dict[str, object]] = []
    accepted_snapshots: dict[int, dict[str, object]] = {}
    for candidate_index, (
        outward_offset_m,
        relative_apex_lift_m,
        forward_offset_m,
    ) in enumerate(
        product(outward_offsets_m, relative_apex_lifts_m, forward_offsets_m)
    ):
        level = deepcopy(level_template)
        level["apex_lift_m"] = baseline_lift_m + relative_apex_lift_m
        level["landing_lift_m"] = baseline_lift_m
        level["swing_forward_offset_m"] = forward_offset_m
        level["landing_forward_offset_m"] = forward_offset_m
        raw_env.placement_reference_config.setdefault(
            "level_override_by_leg", {}
        )["rear_right"] = level
        raw_env.placement_reference_config.setdefault(
            "timing_override_by_leg", {}
        )["rear_right"] = deepcopy(timing_template)
        raw_env.inter_leg_transfer_config.setdefault(
            "swing_outward_offset_m_by_leg", {}
        )["rear_right"] = outward_offset_m

        raw_env.restore_placement_phase_snapshot(
            deepcopy(rewound_snapshot),
            seed=args.seed + candidate_index,
        )
        initial_foot_tips = np.asarray(raw_env.latest_foot_tips_m, dtype=np.float64).copy()
        initial_rr = initial_foot_tips[rear_right_index].copy()
        initial_physical_lift_m = float(
            initial_rr[2] - raw_env.initial_foot_bottom_z_m[rear_right_index]
        )
        required_reclearance_lift_m = (
            initial_physical_lift_m + minimum_relative_reclearance_m
        )
        # The first placement already cleared the full 190 mm riser. This
        # second, lateral-only move begins on the tread and uses a measured
        # release threshold relative to that landing, not the ground plane.
        level["minimum_lift_m"] = required_reclearance_lift_m
        raw_env.placement_reference_config["level_override_by_leg"][
            "rear_right"
        ] = level
        raw_env.advance_clearance_gate_minimum_m = required_reclearance_lift_m
        zero_action = np.zeros(12, dtype=np.float32)
        minimum_margin_m = math.inf
        minimum_support_contact_fraction = 1.0
        maximum_tilt_deg = 0.0
        maximum_physical_lift_m = 0.0
        maximum_tread_load_n = 0.0
        completed = False
        last_info: dict[str, object] = {}
        steps = 0
        for step in range(1, maximum_steps + 1):
            steps = step
            _, _, terminated, truncated, info = raw_env.step(zero_action)
            last_info = dict(info)
            upright = float(info.get("placement_upright_cosine", 1.0))
            maximum_tilt_deg = max(
                maximum_tilt_deg,
                math.degrees(math.acos(float(np.clip(upright, -1.0, 1.0)))),
            )
            minimum_margin_m = min(
                minimum_margin_m,
                float(info.get("placement_support_margin_m", math.inf)),
            )
            minimum_support_contact_fraction = min(
                minimum_support_contact_fraction,
                float(info.get("placement_support_contact_fraction", 1.0)),
            )
            maximum_physical_lift_m = max(
                maximum_physical_lift_m,
                float(info.get("placement_swing_lift_m", 0.0)),
            )
            maximum_tread_load_n = max(
                maximum_tread_load_n,
                float(info.get("swing_tread_normal_load_n", 0.0)),
            )
            if info.get("placement_leg_completed_event") == "rear_right":
                if not raw_env.placement_transfer_active:
                    raise RuntimeError(
                        "Second rear-right landing did not start rear-left transfer"
                    )
                completed = True
                accepted_snapshots[candidate_index] = deepcopy(
                    raw_env.capture_placement_phase_snapshot()
                )
                break
            if terminated or truncated:
                break

        final_foot_tips = np.asarray(raw_env.latest_foot_tips_m, dtype=np.float64).copy()
        final_rr = final_foot_tips[rear_right_index]
        final_com = np.asarray(
            raw_env.latest_placement_com_position_m, dtype=np.float64
        ).copy()
        next_support_margin_m = _support_triangle_signed_margin_m(
            final_com[:2],
            final_foot_tips[next_support_indices, :2],
        )
        final_tread_load_n = float(
            np.sum(
                np.asarray(raw_env.latest_step_top_normal_loads_n)[
                    rear_right_index
                ]
            )
        )
        final_step_layer_load_n = float(
            np.sum(np.asarray(raw_env.latest_step_normal_loads_n)[rear_right_index])
        )
        physical_outward_m = float(initial_rr[1] - final_rr[1])
        forward_drift_m = abs(float(final_rr[0] - initial_rr[0]))
        maximum_support_slip_m = float(raw_env.maximum_support_slip_m)
        measured_relative_reclearance_m = (
            maximum_physical_lift_m - initial_physical_lift_m
        )
        gate_results = {
            "completed_second_landing": completed,
            "physical_outward_displacement": physical_outward_m
            >= float(acceptance["minimum_physical_outward_displacement_m"]),
            "forward_foot_drift": forward_drift_m
            <= float(acceptance["maximum_forward_foot_drift_m"]),
            "rear_right_tread_load": final_tread_load_n
            >= float(acceptance["minimum_rear_right_tread_load_n"]),
            "body_tilt": maximum_tilt_deg
            <= float(acceptance["maximum_body_tilt_deg"]),
            "support_slip": maximum_support_slip_m
            <= float(acceptance["maximum_support_slip_m"]),
            "support_contact": minimum_support_contact_fraction >= 1.0,
            "replacement_support_margin": minimum_margin_m
            >= float(acceptance["minimum_replacement_support_margin_m"]),
            "relative_reclearance": measured_relative_reclearance_m
            >= minimum_relative_reclearance_m,
        }
        accepted = all(gate_results.values())
        result = {
            "candidate_index": candidate_index,
            "outward_offset_m": outward_offset_m,
            "relative_apex_lift_m": relative_apex_lift_m,
            "forward_offset_m": forward_offset_m,
            "absolute_apex_lift_m": baseline_lift_m + relative_apex_lift_m,
            "steps": steps,
            "completed": completed,
            "accepted": accepted,
            "acceptance_gates": gate_results,
            "initial_rear_right_xyz_m": initial_rr.tolist(),
            "final_rear_right_xyz_m": final_rr.tolist(),
            "physical_outward_displacement_m": physical_outward_m,
            "forward_foot_drift_m": forward_drift_m,
            "final_com_xyz_m": final_com.tolist(),
            "next_support_margin_m": next_support_margin_m,
            "minimum_replacement_support_margin_m": minimum_margin_m,
            "minimum_support_contact_fraction": minimum_support_contact_fraction,
            "maximum_body_tilt_deg": maximum_tilt_deg,
            "maximum_physical_lift_m": maximum_physical_lift_m,
            "initial_physical_lift_m": initial_physical_lift_m,
            "required_reclearance_lift_m": required_reclearance_lift_m,
            "measured_relative_reclearance_m": measured_relative_reclearance_m,
            "maximum_tread_load_n": maximum_tread_load_n,
            "final_rear_right_tread_load_n": final_tread_load_n,
            "final_rear_right_step_layer_load_n": final_step_layer_load_n,
            "maximum_support_slip_m": maximum_support_slip_m,
            "failure_reasons": list(last_info.get("failure_reasons", ())),
            "terminated": bool(last_info.get("terminated", False)),
            "truncated": bool(last_info.get("truncated", False)),
        }
        candidates.append(result)
        print(
            "DROBOT_REAR_RIGHT_SIDESTEP_CANDIDATE="
            + json.dumps(result, sort_keys=True),
            flush=True,
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            bool(item["accepted"]),
            bool(item["completed"]),
            float(item["next_support_margin_m"]),
            float(item["physical_outward_displacement_m"]),
            -float(item["maximum_body_tilt_deg"]),
            -float(item["forward_foot_drift_m"]),
        ),
        reverse=True,
    )
    accepted_ranked = [item for item in ranked if bool(item["accepted"])]
    saved_snapshot: dict[str, object] | None = None
    if accepted_ranked:
        best_index = int(accepted_ranked[0]["candidate_index"])
        payload = {
            "schema_version": 1,
            "source_task_id": task_config["id"],
            "source_phase_snapshot": str(phase_snapshot_path),
            "source_phase_snapshot_sha256": report["phase_snapshot_sha256"],
            "target_leg": "rear_left",
            "phase_snapshot_mode": "inter_leg_transfer",
            **actual_contract,
            "placement_sequence_legs": list(
                task_config["placement_reference"]["sequence_legs"]
            ),
            "seed": args.seed,
            "candidate": accepted_ranked[0],
            "snapshot": accepted_snapshots[best_index],
        }
        snapshot_sha256 = write_json(saved_snapshot_path, payload)
        saved_snapshot = {
            "path": str(saved_snapshot_path),
            "sha256": snapshot_sha256,
        }
    report.update(
        {
            "status": "PASS",
            "task_success": bool(accepted_ranked),
            "accepted_candidates": len(accepted_ranked),
            "best": ranked[0],
            "ranked_candidates": ranked,
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
    report_sha256 = write_json(report_path, report)
    print(
        "DROBOT_REAR_RIGHT_SIDESTEP_SEARCH="
        + json.dumps(
            {
                "status": report["status"],
                "task_success": report.get("task_success", False),
                "report": str(report_path),
                "report_sha256": report_sha256,
                "best": report.get("best"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    simulation_app.close()

sys.exit(exit_code)
