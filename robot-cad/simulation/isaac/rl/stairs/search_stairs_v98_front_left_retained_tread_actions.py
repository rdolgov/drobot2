"""Search constant hip/knee offsets at the V94 front-left transfer boundary.

This is a controller-authority audit before spending another PPO budget. Every
candidate restores the same verified mid-transfer snapshot. Stage one searches
front-right hip-flexion/knee offsets that retain the already-placed tread foot.
Stage two combines the best retained-tread pairs with front-left hip/knee
offsets that unload the next swing foot. RGB is not a policy or search input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import torch._dynamo  # noqa: F401
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _stair_rl_contract import (  # noqa: E402
    config_for_first_tread_experiment,
    reanchor_inter_leg_transfer_snapshot,
)

FRONT_RIGHT_JOINTS = ("front_right_hip_flexion", "front_right_knee")
FRONT_LEFT_JOINTS = ("front_left_hip_flexion", "front_left_knee")


def _project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _comma_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError("action values must be finite")
    if any(abs(item) > 1.0 for item in result):
        raise ValueError("action values must be within [-1, 1]")
    return result


def _candidate_score(candidate: Mapping[str, object]) -> tuple[float, ...]:
    return (
        float(candidate["qualified_hold_steps"]),
        float(candidate["retained_tread_steps"]),
        float(candidate["minimum_completed_tread_load_n"]),
        -float(candidate["minimum_swing_load_n"]),
        -float(candidate["maximum_support_slip_m"]),
        float(candidate["minimum_upright_cosine"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            SCRIPT_DIR / "quadruped_stairs_v14_front_pair_right_then_left.yaml"
        ),
    )
    parser.add_argument(
        "--first-tread-profile",
        default="front-pair-preposition-load-advance-forward-floor",
    )
    parser.add_argument("--placement-level", default="left-quarter-tread-load")
    parser.add_argument(
        "--phase-snapshot",
        default=(
            "simulation/isaac/models/"
            "ppo-stairs-v97-front-left-retained-tread-unload-8192-seed1051/"
            "phase_snapshot_seed1047.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=1052)
    parser.add_argument("--candidate-seconds", type=float, default=2.5)
    parser.add_argument("--stable-hold-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-completed-tread-load-n", type=float, default=5.0)
    parser.add_argument("--maximum-swing-load-n", type=float, default=20.0)
    parser.add_argument("--minimum-upright-cosine", type=float, default=0.975)
    parser.add_argument("--maximum-support-slip-m", type=float, default=0.025)
    parser.add_argument(
        "--front-right-values",
        default="-0.6,-0.3,0.0,0.3,0.6",
    )
    parser.add_argument("--front-left-values", default="-0.4,0.0,0.4")
    parser.add_argument("--stage-two-front-right-count", type=int, default=3)
    parser.add_argument(
        "--snapshot",
        default=(
            "simulation/isaac/output/rl/"
            "front-left-retained-tread-snapshot-v98-seed1052.json"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "simulation/isaac/output/rl/"
            "front-left-retained-tread-action-search-v98-seed1052.json"
        ),
    )
    args, _ = parser.parse_known_args()

    front_right_values = _comma_floats(args.front_right_values)
    front_left_values = _comma_floats(args.front_left_values)
    if args.candidate_seconds <= 0.0:
        parser.error("--candidate-seconds must be positive")
    if args.stable_hold_seconds <= 0.0:
        parser.error("--stable-hold-seconds must be positive")
    if args.stage_two_front_right_count < 1:
        parser.error("--stage-two-front-right-count must be positive")

    config_path = _project_path(args.config)
    phase_snapshot_path = _project_path(args.phase_snapshot)
    snapshot_path = _project_path(args.snapshot)
    report_path = _project_path(args.report)
    with config_path.open("r", encoding="utf-8") as stream:
        loaded_config = yaml.safe_load(stream)
    task_config = config_for_first_tread_experiment(
        loaded_config["task"],
        args.first_tread_profile,
    )
    transfer_override = task_config["placement_reference"][
        "inter_leg_transfer"
    ].setdefault("override_by_next_swing_leg", {}).setdefault(
        "front_left",
        {},
    )
    transfer_override.update(
        {
            "residual_action_scale": 1.0,
            "swing_unload_lift_m": 0.0,
            "require_swing_unload": True,
            "maximum_swing_unloaded_load_n": 1.0,
        }
    )
    with phase_snapshot_path.open("r", encoding="utf-8") as stream:
        phase_snapshot_payload = json.load(stream)
    if phase_snapshot_payload.get("target_leg") != "front_left":
        raise ValueError("phase snapshot target leg must be front_left")
    if not np.isclose(
        float(phase_snapshot_payload["stair_tread_depth_m"]),
        0.25,
        atol=1e-9,
    ):
        raise ValueError("phase snapshot must use the 250 mm tread")
    phase_snapshot = dict(phase_snapshot_payload["snapshot"])
    world_path = _project_path(str(task_config["world"]))

    from isaacsim import SimulationApp  # noqa: PLC0415

    simulation_app = SimulationApp({"headless": True})
    raw_env = None
    report: dict[str, object] = {
        "status": "FAIL",
        "scope": "V98 exact-snapshot front-right retention/front-left unload action search",
        "task_id": task_config["id"],
        "config": str(config_path),
        "phase_snapshot": str(phase_snapshot_path),
        "phase_snapshot_sha256": hashlib.sha256(
            phase_snapshot_path.read_bytes()
        ).hexdigest(),
        "seed": args.seed,
        "stair_rise_m": float(task_config["staircase"]["rise_m"]),
        "stair_tread_depth_m": float(task_config["staircase"]["tread_depth_m"]),
        "effort_cap_nm": float(
            task_config["robot_hardware_profile"]["effort_cap_nm"]
        ),
        "rgb_camera_policy_input": False,
        "candidate_seconds": args.candidate_seconds,
        "gate": {
            "stable_hold_seconds": args.stable_hold_seconds,
            "minimum_completed_tread_load_n": args.minimum_completed_tread_load_n,
            "maximum_swing_load_n": args.maximum_swing_load_n,
            "minimum_upright_cosine": args.minimum_upright_cosine,
            "maximum_support_slip_m": args.maximum_support_slip_m,
        },
    }
    exit_code = 1
    try:
        from _placement_phase_training import (  # noqa: PLC0415
            PlacementPhaseTrainingEnv,
        )
        from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: PLC0415

        raw_env = QuadrupedStairsEnv(
            simulation_app,
            world_path=str(world_path),
            task_config=task_config,
        )
        raw_env.set_evaluation_level(1)
        raw_env.set_placement_level(args.placement_level, activate_immediately=True)
        phase_env = PlacementPhaseTrainingEnv(
            raw_env,
            target_leg="front_left",
            precursor_policies={},
            target_residual_mask=np.ones(12, dtype=bool),
            compact_residual_action=True,
            train_transfer=True,
            initial_phase_snapshot=phase_snapshot,
            initial_phase_snapshot_mode="inter_leg_transfer",
            transfer_unload_thresholds_n=(args.maximum_swing_load_n, 1.0),
            transfer_unload_successes_per_level=1,
            transfer_upright_cosines=(args.minimum_upright_cosine, 0.9781476),
        )
        joint_index = {name: index for index, name in enumerate(raw_env.dof_names)}
        maximum_steps = int(math.ceil(args.candidate_seconds * raw_env.control_hz))
        hold_steps = int(math.ceil(args.stable_hold_seconds * raw_env.control_hz))

        def evaluate(
            *,
            candidate_id: str,
            action_by_joint: Mapping[str, float],
            stage: int,
        ) -> dict[str, object]:
            action = np.zeros(12, dtype=np.float32)
            for name, value in action_by_joint.items():
                action[joint_index[name]] = float(value)
            _, reset_info = phase_env.reset(seed=args.seed)
            minimum_swing_load = math.inf
            minimum_tread_load = math.inf
            maximum_tread_load = 0.0
            minimum_upright = 1.0
            maximum_slip = 0.0
            retained_tread_steps = 0
            qualified_window: deque[dict[str, object]] = deque(maxlen=hold_steps)
            best_snapshot_candidate: dict[str, object] | None = None
            last_info = dict(reset_info)
            terminated = False
            truncated = False
            for step_index in range(1, maximum_steps + 1):
                _, _, terminated, truncated, info = phase_env.step(action)
                last_info = dict(info)
                swing_load = float(
                    info.get("placement_transfer_swing_total_load_n", math.inf)
                )
                tread_load = float(
                    info.get(
                        "placement_transfer_completed_tread_min_load_n",
                        0.0,
                    )
                )
                upright = float(info.get("placement_upright_cosine", 0.0))
                slip = float(info.get("maximum_support_slip_m", math.inf))
                margin = float(info.get("placement_support_margin_m", -math.inf))
                minimum_swing_load = min(minimum_swing_load, swing_load)
                minimum_tread_load = min(minimum_tread_load, tread_load)
                maximum_tread_load = max(maximum_tread_load, tread_load)
                minimum_upright = min(minimum_upright, upright)
                maximum_slip = max(maximum_slip, slip)
                if tread_load >= args.minimum_completed_tread_load_n:
                    retained_tread_steps += 1
                qualified = bool(
                    tread_load >= args.minimum_completed_tread_load_n
                    and swing_load <= args.maximum_swing_load_n
                    and upright >= args.minimum_upright_cosine
                    and slip <= args.maximum_support_slip_m
                    and margin >= 0.015
                )
                if not qualified:
                    qualified_window.clear()
                else:
                    captured = raw_env.capture_placement_phase_snapshot()
                    references = raw_env._reference_parameters_from_joint_positions(  # noqa: SLF001
                        np.asarray(captured["joint_positions_rad"], dtype=np.float32)
                    )
                    balance_position = np.asarray(
                        info.get(
                            "placement_balance_position_m",
                            raw_env.latest_placement_com_position_m,
                        ),
                        dtype=np.float64,
                    )
                    stationary = reanchor_inter_leg_transfer_snapshot(
                        captured,
                        balance_position_m=balance_position,
                        target_delta_xy_m=(0.0, 0.0),
                        reference_by_leg=references,
                    )
                    qualified_window.append(
                        {
                            "step": step_index,
                            "swing_load_n": swing_load,
                            "completed_tread_load_n": tread_load,
                            "upright_cosine": upright,
                            "support_slip_m": slip,
                            "support_margin_m": margin,
                            "snapshot": stationary,
                        }
                    )
                    if len(qualified_window) == hold_steps:
                        window_best = min(
                            qualified_window,
                            key=lambda item: float(item["swing_load_n"]),
                        )
                        if best_snapshot_candidate is None or float(
                            window_best["swing_load_n"]
                        ) < float(best_snapshot_candidate["swing_load_n"]):
                            best_snapshot_candidate = deepcopy(window_best)
                if terminated or truncated:
                    break
            qualified_hold_steps = hold_steps if best_snapshot_candidate else 0
            result = {
                "id": candidate_id,
                "stage": stage,
                "action_by_joint": dict(action_by_joint),
                "steps": step_index,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "failure_reasons": list(last_info.get("failure_reasons", ())),
                "minimum_swing_load_n": minimum_swing_load,
                "final_swing_load_n": float(
                    last_info.get("placement_transfer_swing_total_load_n", math.inf)
                ),
                "minimum_completed_tread_load_n": minimum_tread_load,
                "maximum_completed_tread_load_n": maximum_tread_load,
                "final_completed_tread_load_n": float(
                    last_info.get(
                        "placement_transfer_completed_tread_min_load_n",
                        0.0,
                    )
                ),
                "retained_tread_steps": retained_tread_steps,
                "qualified_hold_steps": qualified_hold_steps,
                "minimum_upright_cosine": minimum_upright,
                "maximum_support_slip_m": maximum_slip,
                "final_support_margin_m": float(
                    last_info.get("placement_support_margin_m", -math.inf)
                ),
                "final_gate_failures": list(
                    last_info.get("placement_transfer_gate_failures", ())
                ),
            }
            if best_snapshot_candidate is not None:
                result["best_qualified_state"] = best_snapshot_candidate
            print(
                "DROBOT_V98_CANDIDATE="
                + json.dumps(
                    {
                        "id": candidate_id,
                        "stage": stage,
                        "tread_max_n": round(maximum_tread_load, 3),
                        "tread_final_n": round(
                            float(result["final_completed_tread_load_n"]),
                            3,
                        ),
                        "swing_min_n": round(minimum_swing_load, 3),
                        "retained_steps": retained_tread_steps,
                        "qualified": bool(best_snapshot_candidate),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return result

        stage_one: list[dict[str, object]] = []
        for hip_value, knee_value in product(front_right_values, repeat=2):
            stage_one.append(
                evaluate(
                    candidate_id=f"fr-h{hip_value:+.1f}-k{knee_value:+.1f}",
                    action_by_joint={
                        FRONT_RIGHT_JOINTS[0]: hip_value,
                        FRONT_RIGHT_JOINTS[1]: knee_value,
                    },
                    stage=1,
                )
            )
        stage_one.sort(key=_candidate_score, reverse=True)
        selected_front_right = stage_one[: args.stage_two_front_right_count]

        stage_two: list[dict[str, object]] = []
        for front_right in selected_front_right:
            base_actions = dict(front_right["action_by_joint"])
            for hip_value, knee_value in product(front_left_values, repeat=2):
                actions = {
                    **base_actions,
                    FRONT_LEFT_JOINTS[0]: hip_value,
                    FRONT_LEFT_JOINTS[1]: knee_value,
                }
                stage_two.append(
                    evaluate(
                        candidate_id=(
                            f"{front_right['id']}-fl-h{hip_value:+.1f}-k{knee_value:+.1f}"
                        ),
                        action_by_joint=actions,
                        stage=2,
                    )
                )
        all_candidates = [*stage_one, *stage_two]
        all_candidates.sort(key=_candidate_score, reverse=True)
        best = all_candidates[0]
        snapshot_record = best.get("best_qualified_state")
        snapshot_report: dict[str, object] | None = None
        if isinstance(snapshot_record, dict):
            snapshot = snapshot_record.pop("snapshot")
            output_payload = {
                "schema_version": 1,
                "source_task_id": task_config["id"],
                "target_leg": "front_left",
                "phase_snapshot_mode": "inter_leg_transfer",
                "stair_rise_m": float(task_config["staircase"]["rise_m"]),
                "stair_tread_depth_m": float(
                    task_config["staircase"]["tread_depth_m"]
                ),
                "effort_cap_nm": float(
                    task_config["robot_hardware_profile"]["effort_cap_nm"]
                ),
                "placement_sequence_legs": list(
                    task_config["placement_reference"]["sequence_legs"]
                ),
                "seed": args.seed,
                "source_phase_snapshot_sha256": report[
                    "phase_snapshot_sha256"
                ],
                "search_candidate": best["id"],
                "search_action_by_joint": best["action_by_joint"],
                "capture_metrics": snapshot_record,
                "snapshot": _json_compatible(snapshot),
            }
            snapshot_bytes = (
                json.dumps(output_payload, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(snapshot_bytes)
            snapshot_report = {
                "path": str(snapshot_path),
                "sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
            }
        report.update(
            {
                "status": "PASS",
                "stage_one_candidate_count": len(stage_one),
                "stage_two_candidate_count": len(stage_two),
                "qualified_candidate_count": sum(
                    bool(item.get("best_qualified_state"))
                    for item in all_candidates
                ),
                "task_success": snapshot_report is not None,
                "best": best,
                "snapshot": snapshot_report,
                "ranked_candidates": all_candidates,
            }
        )
        exit_code = 0
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        if raw_env is not None:
            raw_env.close()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(_json_compatible(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        simulation_app.close()
    print("DROBOT_STAIRS_V98_SEARCH=" + json.dumps(report), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
