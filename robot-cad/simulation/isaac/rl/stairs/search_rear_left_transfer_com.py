"""Search a positive-margin COM target for the rear-right to rear-left handoff.

The verified front-foot policies and accepted V44 rear-right landing are
replayed once. Each candidate restores the same physical transfer-start
snapshot, shifts only the retained COM target, and runs the analytic transfer
with zero residual action. This keeps target geometry separate from PPO and
makes rear-left training start from an evidence-backed support target.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from copy import deepcopy
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_stairs_v45_rear_left_transfer.yaml"),
)
parser.add_argument("--seed", type=int, default=871)
parser.add_argument(
    "--forward-deltas-m",
    default="0.000,0.020",
    help="Comma-separated additions to the cached COM/base target.",
)
parser.add_argument(
    "--lateral-deltas-m",
    default="-0.040,-0.020,0.000,0.020,0.040,0.060",
    help="Comma-separated additions to the cached COM/base target.",
)
parser.add_argument("--minimum-support-margin-m", type=float, default=0.015)
parser.add_argument("--maximum-body-tilt-deg", type=float, default=12.0)
parser.add_argument(
    "--report",
    default=(
        "simulation/isaac/output/rl/"
        "rear-left-transfer-com-search-v45-seed871.json"
    ),
)
parser.add_argument(
    "--front-right-model",
    default=(
        "simulation/isaac/models/"
        "ppo-stairs-v10-180mm-25cm-front-right-placement-small/"
        "drobot_stairs_ppo_final.zip"
    ),
)
parser.add_argument(
    "--front-left-model",
    default=(
        "simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/"
        "drobot_stairs_ppo_final.zip"
    ),
)
parser.add_argument(
    "--rear-right-swing-base-model",
    default=(
        "simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/"
        "drobot_stairs_ppo_final.zip"
    ),
)
parser.add_argument(
    "--rear-right-swing-residual-model",
    default=(
        "simulation/isaac/models/ppo-stairs-v35-rear-right-190mm-lift-small/"
        "drobot_stairs_ppo_final.zip"
    ),
)
parser.add_argument(
    "--rear-right-support-model",
    default=(
        "simulation/isaac/models/"
        "ppo-stairs-v44-early-contact-rear-right-landing-small/"
        "drobot_stairs_ppo_final.zip"
    ),
)
args, _ = parser.parse_known_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def comma_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError("candidate deltas must contain finite numbers")
    return result


if args.minimum_support_margin_m <= 0.0:
    parser.error("--minimum-support-margin-m must be positive")
if not 0.0 < args.maximum_body_tilt_deg < 45.0:
    parser.error("--maximum-body-tilt-deg must be within (0, 45)")

forward_deltas_m = comma_floats(args.forward_deltas_m)
lateral_deltas_m = comma_floats(args.lateral_deltas_m)
config_path = project_path(args.config)
report_path = project_path(args.report)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
task_config = dict(config["task"])
world_path = project_path(task_config["world"])

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from _placement_phase_training import (  # noqa: E402
    FrozenBaseResidualPolicy,
    PlacementPhaseTrainingEnv,
)
from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: E402
from _stair_rl_contract import placement_policy_action_mask  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "config": str(config_path),
    "world": str(world_path),
    "seed": args.seed,
    "stair_rise_m": float(task_config["staircase"]["rise_m"]),
    "stair_tread_depth_m": float(task_config["staircase"]["tread_depth_m"]),
    "effort_cap_nm": float(task_config["robot_hardware_profile"]["effort_cap_nm"]),
    "minimum_support_margin_m": args.minimum_support_margin_m,
    "maximum_body_tilt_deg": args.maximum_body_tilt_deg,
    "candidate_count": len(forward_deltas_m) * len(lateral_deltas_m),
    "models": {
        "front_right": str(project_path(args.front_right_model)),
        "front_left": str(project_path(args.front_left_model)),
        "rear_right_swing_base": str(
            project_path(args.rear_right_swing_base_model)
        ),
        "rear_right_swing_residual": str(
            project_path(args.rear_right_swing_residual_model)
        ),
        "rear_right_support": str(
            project_path(args.rear_right_support_model)
        ),
    },
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
    rear_right_swing_mask = placement_policy_action_mask(
        raw_env.dof_names,
        target_leg="rear_right",
        mode="swing_only",
    )
    rear_right_support_mask = placement_policy_action_mask(
        raw_env.dof_names,
        target_leg="rear_right",
        mode="support_only",
    )
    rear_right_swing_policy = FrozenBaseResidualPolicy(
        base_policy=PPO.load(
            str(project_path(args.rear_right_swing_base_model)), device="cpu"
        ),
        residual_policy=PPO.load(
            str(project_path(args.rear_right_swing_residual_model)),
            device="cpu",
        ),
        action_space=raw_env.action_space,
        residual_scale=0.5,
        base_mask=rear_right_swing_mask,
        residual_mask=rear_right_swing_mask,
        compact_residual_action=True,
    )
    rear_right_policy = FrozenBaseResidualPolicy(
        base_policy=rear_right_swing_policy,
        residual_policy=PPO.load(
            str(project_path(args.rear_right_support_model)), device="cpu"
        ),
        action_space=raw_env.action_space,
        residual_scale=1.0,
        residual_mask=rear_right_support_mask,
        compact_residual_action=True,
    )
    precursor_policies = {
        "front_right": PPO.load(
            str(project_path(args.front_right_model)), device="cpu"
        ),
        "front_left": PPO.load(
            str(project_path(args.front_left_model)), device="cpu"
        ),
        "rear_right": rear_right_policy,
    }
    support_mask = placement_policy_action_mask(
        raw_env.dof_names,
        target_leg="rear_left",
        mode="support_only",
    )
    phase_env = PlacementPhaseTrainingEnv(
        raw_env,
        target_leg="rear_left",
        precursor_policies=precursor_policies,
        target_residual_mask=support_mask,
        compact_residual_action=True,
        train_transfer=True,
        maximum_reset_attempts=4,
        maximum_precursor_steps=3600,
    )
    zero_action = np.zeros(phase_env.action_space.shape, dtype=np.float32)
    print("DROBOT_REAR_LEFT_TRANSFER_SEARCH_PHASE=precursor_replay", flush=True)
    _, initial_info = phase_env.reset(seed=args.seed)
    if phase_env.phase_snapshot is None or not raw_env.placement_transfer_active:
        raise RuntimeError("rear-left transfer-start snapshot was not captured")
    baseline_snapshot = deepcopy(phase_env.phase_snapshot)
    baseline_target_balance = np.asarray(
        baseline_snapshot["placement_transfer_target_balance_position_m"],
        dtype=np.float64,
    )
    baseline_target_base = np.asarray(
        baseline_snapshot["placement_transfer_target_base_position_m"],
        dtype=np.float64,
    )

    rear_override = raw_env.inter_leg_transfer_config[
        "override_by_next_swing_leg"
    ]["rear_left"]
    rear_override["minimum_support_margin_m"] = float(
        args.minimum_support_margin_m
    )
    rear_override["minimum_upright_cosine"] = float(
        math.cos(math.radians(args.maximum_body_tilt_deg))
    )
    maximum_seconds = float(rear_override["maximum_seconds"])
    maximum_steps = int(math.ceil((maximum_seconds + 0.5) * raw_env.control_hz))

    candidates: list[dict[str, object]] = []
    for forward_delta_m in forward_deltas_m:
        for lateral_delta_m in lateral_deltas_m:
            candidate_snapshot = deepcopy(baseline_snapshot)
            target_delta = np.asarray(
                [forward_delta_m, lateral_delta_m, 0.0], dtype=np.float64
            )
            candidate_snapshot["placement_transfer_target_balance_position_m"] = (
                baseline_target_balance + target_delta
            )
            candidate_snapshot["placement_transfer_target_base_position_m"] = (
                baseline_target_base + target_delta
            )
            phase_env.phase_snapshot = candidate_snapshot
            observation, _ = phase_env.reset(seed=args.seed)
            del observation

            completed = False
            terminated = False
            truncated = False
            last_info: dict[str, object] = dict(initial_info)
            minimum_margin_after_unload_m = float("inf")
            maximum_tilt_deg = 0.0
            maximum_target_error_m = 0.0
            minimum_margin_state: dict[str, object] | None = None
            steps_taken = 0
            for _step in range(1, maximum_steps + 1):
                steps_taken += 1
                _, _, terminated, truncated, info = phase_env.step(zero_action)
                last_info = dict(info)
                margin_m = float(info.get("placement_support_margin_m", 0.0))
                maximum_tilt_deg = max(
                    maximum_tilt_deg, float(raw_env.maximum_tilt_deg)
                )
                maximum_target_error_m = max(
                    maximum_target_error_m,
                    float(
                        info.get(
                            "placement_transfer_base_target_error_m", 0.0
                        )
                    ),
                )
                if raw_env.placement_transfer_unload_start_step is not None:
                    if margin_m < minimum_margin_after_unload_m:
                        minimum_margin_after_unload_m = margin_m
                        minimum_margin_state = {
                            "balance_position_m": raw_env.latest_placement_com_position_m.tolist(),
                            "support_foot_positions_m": raw_env._sample_foot_tips()[  # noqa: SLF001
                                list(raw_env.placement_support_leg_indices)
                            ].tolist(),
                        }
                if bool(info.get("phase_training_transfer_completed")):
                    completed = True
                    break
                if terminated or truncated:
                    break

            completion_metrics = dict(
                last_info.get("last_completed_inter_leg_transfer_metrics", {})
            )
            if completion_metrics.get("transition") != "rear_right->rear_left":
                completion_metrics = {}
            completion_margin_m = float(
                completion_metrics.get("support_margin_m", -math.inf)
            )
            completion_tilt_deg = float(
                completion_metrics.get("body_tilt_deg", math.inf)
            )
            if not math.isfinite(minimum_margin_after_unload_m):
                minimum_margin_after_unload_m = -math.inf
            strict_pass = bool(
                completed
                and minimum_margin_after_unload_m
                >= args.minimum_support_margin_m
                and completion_margin_m >= args.minimum_support_margin_m
                and completion_tilt_deg <= args.maximum_body_tilt_deg
                and not last_info.get("failure_reasons")
            )
            reported_minimum_margin_after_unload_m = (
                minimum_margin_after_unload_m
                if math.isfinite(minimum_margin_after_unload_m)
                else None
            )
            reported_completion_margin_m = (
                completion_margin_m if math.isfinite(completion_margin_m) else None
            )
            reported_completion_tilt_deg = (
                completion_tilt_deg if math.isfinite(completion_tilt_deg) else None
            )
            result = {
                "id": (
                    f"forward{forward_delta_m:+.3f}-"
                    f"lateral{lateral_delta_m:+.3f}"
                ),
                "target_forward_delta_m": forward_delta_m,
                "target_lateral_delta_m": lateral_delta_m,
                "target_balance_position_m": (
                    baseline_target_balance + target_delta
                ).tolist(),
                "steps": steps_taken,
                "completed": completed,
                "strict_pass": strict_pass,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "failure_reasons": list(last_info.get("failure_reasons", ())),
                "minimum_margin_after_unload_m": (
                    reported_minimum_margin_after_unload_m
                ),
                "completion_support_margin_m": reported_completion_margin_m,
                "completion_body_tilt_deg": reported_completion_tilt_deg,
                "maximum_body_tilt_deg": maximum_tilt_deg,
                "maximum_balance_target_error_m": maximum_target_error_m,
                "maximum_reference_reach_excess_m": float(
                    raw_env.maximum_placement_reference_reach_excess_m
                ),
                "reference_reach_clip_count": int(
                    raw_env.placement_reference_reach_clip_count
                ),
                "minimum_margin_state": minimum_margin_state,
                "completion_metrics": completion_metrics,
                "final_gate_failures": list(
                    last_info.get("placement_transfer_gate_failures", ())
                ),
                "final_pre_unload_gate_failures": list(
                    last_info.get("placement_pre_unload_gate_failures", ())
                ),
            }
            candidates.append(result)
            print(
                "DROBOT_REAR_LEFT_TRANSFER_CANDIDATE="
                + json.dumps(
                    {
                        "id": result["id"],
                        "completed": completed,
                        "strict_pass": strict_pass,
                        "post_unload_margin_mm": (
                            round(1000.0 * minimum_margin_after_unload_m, 1)
                            if reported_minimum_margin_after_unload_m is not None
                            else None
                        ),
                        "completion_margin_mm": (
                            round(1000.0 * completion_margin_m, 1)
                            if reported_completion_margin_m is not None
                            else None
                        ),
                        "completion_tilt_deg": (
                            round(completion_tilt_deg, 2)
                            if reported_completion_tilt_deg is not None
                            else None
                        ),
                        "failures": result["failure_reasons"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    candidates.sort(
        key=lambda item: (
            bool(item["strict_pass"]),
            bool(item["completed"]),
            float(item["minimum_margin_after_unload_m"] or -math.inf),
            float(item["completion_support_margin_m"] or -math.inf),
            -float(item["completion_body_tilt_deg"] or math.inf),
        ),
        reverse=True,
    )
    report.update(
        {
            "status": "PASS",
            "strict_pass_count": sum(
                bool(item["strict_pass"]) for item in candidates
            ),
            "baseline_target_balance_position_m": baseline_target_balance.tolist(),
            "support_action_indices": np.flatnonzero(support_mask).tolist(),
            "support_dof_names": [
                raw_env.dof_names[index] for index in np.flatnonzero(support_mask)
            ],
            "best": candidates[0],
            "ranked_candidates": candidates,
        }
    )
    exit_code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    if raw_env is not None:
        raw_env.close()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_REAR_LEFT_TRANSFER_SEARCH=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
