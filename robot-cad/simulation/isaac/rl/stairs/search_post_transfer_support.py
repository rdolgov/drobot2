"""Search bounded constant support actions at a cached stair handoff."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
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
    default=str(SCRIPT_DIR / "quadruped_stairs_v36_transfer_support_residual.yaml"),
)
parser.add_argument("--seed", type=int, default=840)
parser.add_argument("--candidates", type=int, default=64)
parser.add_argument("--action-sigma", type=float, default=0.045)
parser.add_argument(
    "--report",
    default=(
        "simulation/isaac/output/rl/"
        "ppo-stairs-v36-post-transfer-action-search-seed840.json"
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
args, _ = parser.parse_known_args()
if args.candidates < 1:
    parser.error("--candidates must be positive")
if args.action_sigma <= 0.0:
    parser.error("--action-sigma must be positive")


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


config_path = project_path(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
task_config = dict(config["task"])
world_path = project_path(task_config["world"])
report_path = project_path(args.report)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from _placement_phase_training import PlacementPhaseTrainingEnv  # noqa: E402
from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: E402
from _stair_rl_contract import placement_policy_action_mask  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "config": str(config_path),
    "world": str(world_path),
    "seed": args.seed,
    "candidate_count": args.candidates,
    "action_sigma": args.action_sigma,
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
    precursor_policies = {
        "front_right": PPO.load(
            str(project_path(args.front_right_model)),
            device="cpu",
        ),
        "front_left": PPO.load(
            str(project_path(args.front_left_model)),
            device="cpu",
        ),
    }
    support_mask = placement_policy_action_mask(
        raw_env.dof_names,
        target_leg="rear_right",
        mode="support_only",
    )
    wrapped = PlacementPhaseTrainingEnv(
        raw_env,
        target_leg="rear_right",
        precursor_policies=precursor_policies,
        target_residual_mask=support_mask,
        compact_residual_action=True,
        train_transfer=True,
        transfer_post_hold_seconds=float(
            task_config["placement_reference"]["inter_leg_transfer"][
                "policy_post_hold_seconds"
            ]
        ),
        train_post_transfer_hold_only=True,
        maximum_reset_attempts=1,
        maximum_precursor_steps=3600,
    )
    zero_action = np.zeros(wrapped.action_space.shape, dtype=np.float32)
    print("DROBOT_POST_TRANSFER_SEARCH_PHASE=precursor_replay", flush=True)
    wrapped.reset(seed=args.seed)
    print("DROBOT_POST_TRANSFER_SEARCH_PHASE=analytic_transfer", flush=True)
    while wrapped.phase_snapshot_mode != "post_transfer_hold":
        _, _, terminated, truncated, info = wrapped.step(zero_action)
        if terminated or truncated:
            raise RuntimeError(
                "Analytic transfer failed before post-transfer snapshot: "
                f"{info.get('failure_reasons', ())}"
            )
    print("DROBOT_POST_TRANSFER_SEARCH_PHASE=candidate_replay", flush=True)

    rng = np.random.default_rng(args.seed + 1000)
    candidates = [zero_action]
    for axis in range(zero_action.size):
        for value in (-0.06, -0.03, 0.03, 0.06):
            action = zero_action.copy()
            action[axis] = value
            candidates.append(action)
    while len(candidates) < args.candidates:
        candidates.append(
            np.clip(
                rng.normal(0.0, args.action_sigma, size=zero_action.size),
                -0.12,
                0.12,
            ).astype(np.float32)
        )
    candidates = candidates[: args.candidates]

    results: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        wrapped.reset(seed=args.seed + index + 1)
        reward_sum = 0.0
        minimum_margin_m = float("inf")
        minimum_upright = 1.0
        result_info: dict[str, object] = {}
        for _step in range(1, wrapped.transfer_post_hold_steps + 1):
            _, reward, terminated, truncated, info = wrapped.step(candidate)
            result_info = dict(info)
            reward_sum += float(reward)
            minimum_margin_m = min(
                minimum_margin_m,
                float(info.get("placement_support_margin_m", float("inf"))),
            )
            minimum_upright = min(
                minimum_upright,
                float(info.get("placement_upright_cosine", 1.0)),
            )
            if terminated or truncated:
                break
        results.append(
            {
                "index": index,
                "action": candidate.tolist(),
                "steps": _step,
                "hold_completed": bool(
                    result_info.get(
                        "phase_training_transfer_post_hold_completed",
                        False,
                    )
                ),
                "reward_sum": reward_sum,
                "minimum_support_margin_m": minimum_margin_m,
                "minimum_upright_cosine": minimum_upright,
                "failure_reasons": list(
                    result_info.get("failure_reasons", ())
                ),
            }
        )
    ranked = sorted(
        results,
        key=lambda item: (
            bool(item["hold_completed"]),
            int(item["steps"]),
            float(item["minimum_upright_cosine"]),
            float(item["reward_sum"]),
        ),
        reverse=True,
    )
    report.update(
        {
            "status": "PASS",
            "support_action_indices": np.flatnonzero(support_mask).tolist(),
            "support_dof_names": [
                raw_env.dof_names[index] for index in np.flatnonzero(support_mask)
            ],
            "hold_steps": wrapped.transfer_post_hold_steps,
            "successful_candidates": sum(
                bool(item["hold_completed"]) for item in results
            ),
            "best": ranked[0],
            "ranked_results": ranked,
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
    print("DROBOT_POST_TRANSFER_SEARCH=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
