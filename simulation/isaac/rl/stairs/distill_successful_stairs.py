"""Distill successful stochastic stair PPO episodes into its policy mean."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch._dynamo  # noqa: F401
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _run_support import (  # noqa: E402
    file_hash_records,
    model_manifest_path,
    read_model_manifest,
    sha256_file,
    validate_model_manifest,
    write_model_manifest,
)
from _stair_rl_contract import config_for_height_stage  # noqa: E402

parser = argparse.ArgumentParser(
    description=(
        "Collect physically successful stochastic stair episodes and fit "
        "the PPO action mean to their residual actions."
    )
)
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_stairs_v5.yaml"),
)
parser.add_argument("--height-stage", default=None)
parser.add_argument("--model", required=True)
parser.add_argument("--output-model", required=True)
parser.add_argument("--report", required=True)
parser.add_argument("--active-steps", type=int, default=1)
parser.add_argument("--target-successes", type=int, default=3)
parser.add_argument("--max-episodes", type=int, default=80)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--learning-rate", type=float, default=0.0001)
parser.add_argument(
    "--terrain-trigger-index",
    type=int,
    default=3,
    help=(
        "Zero demonstration residuals until this zero-based terrain sample "
        "becomes positive; default 3 is the +0.16 m lookahead."
    ),
)
parser.add_argument("--seed", type=int, default=314)
parser.add_argument("--device", default="cpu")
parser.add_argument(
    "--allow-unverified-model",
    action="store_true",
    help=(
        "Deliberately collect from a policy whose manifest predates a "
        "compatible runtime-only config change; the output is rebound to "
        "the current verified environment."
    ),
)
args, _ = parser.parse_known_args()


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if min(
    args.active_steps,
    args.target_successes,
    args.max_episodes,
    args.epochs,
    args.batch_size,
) <= 0:
    parser.error("episode, success, epoch, step, and batch counts must be positive")
if args.learning_rate <= 0.0:
    parser.error("--learning-rate must be positive")

config_path = _resolve_project_path(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
if int(config.get("schema_version", 0)) != 1:
    parser.error(f"Unsupported stairs config schema: {config.get('schema_version')}")
try:
    config = config_for_height_stage(config, args.height_stage)
except ValueError as exc:
    parser.error(str(exc))
task_config = dict(config["task"])
world_path = _resolve_project_path(task_config["world"])
world_dependency_paths = tuple(
    _resolve_project_path(value)
    for value in task_config.get("world_dependencies", ())
)
model_path = _resolve_project_path(args.model)
output_model_path = _resolve_project_path(args.output_model)
report_path = _resolve_project_path(args.report)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "height_stage": args.height_stage,
    "source_model": str(model_path),
    "output_model": str(output_model_path),
    "target_successes": args.target_successes,
    "max_episodes": args.max_episodes,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "seed": args.seed,
}
env: QuadrupedStairsEnv | None = None
start_time = time.perf_counter()
exit_code = 1

try:
    env = QuadrupedStairsEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
    )
    env.set_evaluation_level(args.active_steps)
    verification = validate_model_manifest(
        model_path=model_path,
        config_path=config_path,
        world_path=world_path,
        world_dependencies=world_dependency_paths,
        environment_contract=env.contract,
        allow_unverified=args.allow_unverified_model,
    )
    model = PPO.load(str(model_path), device=args.device)
    model.set_random_seed(args.seed)
    successful_observations: list[np.ndarray] = []
    successful_actions: list[np.ndarray] = []
    episode_records: list[dict[str, object]] = []

    for episode_index in range(args.max_episodes):
        observation, _ = env.reset(seed=args.seed + episode_index)
        trajectory_observations: list[np.ndarray] = []
        trajectory_actions: list[np.ndarray] = []
        episode_metrics: dict[str, object] | None = None
        while True:
            action, _ = model.predict(observation, deterministic=False)
            trajectory_observations.append(observation.copy())
            trajectory_actions.append(
                np.asarray(action, dtype=np.float32).reshape(12).copy()
            )
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                episode_metrics = dict(info["episode_metrics"])
                break
        succeeded = bool(episode_metrics["stairs_completed"])
        episode_records.append(
            {
                "episode": episode_index + 1,
                "succeeded": succeeded,
                "length_steps": int(episode_metrics["length_steps"]),
                "forward_displacement_m": float(
                    episode_metrics["forward_displacement_m"]
                ),
                "maximum_base_elevation_gain_m": float(
                    episode_metrics["maximum_base_elevation_gain_m"]
                ),
                "highest_foot_step_by_leg": dict(
                    episode_metrics["highest_foot_step_by_leg"]
                ),
                "failure_reasons": list(episode_metrics["failure_reasons"]),
            }
        )
        if succeeded:
            successful_observations.extend(trajectory_observations)
            successful_actions.extend(trajectory_actions)
            if sum(record["succeeded"] for record in episode_records) >= (
                args.target_successes
            ):
                break

    success_count = sum(record["succeeded"] for record in episode_records)
    if success_count < args.target_successes:
        raise RuntimeError(
            f"Collected {success_count}/{args.target_successes} successful episodes"
        )

    observations = torch.as_tensor(
        np.asarray(successful_observations, dtype=np.float32),
        device=model.device,
    )
    action_targets = np.asarray(successful_actions, dtype=np.float32)
    terrain_column = 48 + args.terrain_trigger_index
    if terrain_column < 48 or terrain_column >= 56:
        raise ValueError("--terrain-trigger-index must select one of 8 samples")
    pre_stair_mask = (
        np.asarray(successful_observations, dtype=np.float32)[
            :, terrain_column
        ]
        <= 0.0
    )
    action_targets[pre_stair_mask] = 0.0
    actions = torch.as_tensor(
        action_targets,
        device=model.device,
    )
    actor_parameters = [
        *model.policy.mlp_extractor.policy_net.parameters(),
        *model.policy.action_net.parameters(),
    ]
    optimizer = torch.optim.Adam(actor_parameters, lr=args.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    losses: list[float] = []
    for _ in range(args.epochs):
        order = torch.randperm(observations.shape[0], generator=generator)
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, observations.shape[0], args.batch_size):
            indices = order[start : start + args.batch_size].to(model.device)
            batch_observations = observations[indices]
            batch_actions = actions[indices]
            features = model.policy.extract_features(batch_observations)
            latent = model.policy.mlp_extractor.forward_actor(features)
            mean_actions = model.policy.action_net(latent)
            loss = torch.mean(torch.square(mean_actions - batch_actions))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batch_count += 1
        losses.append(epoch_loss / batch_count)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_model_path.with_suffix("")))
    if not output_model_path.is_file():
        raise RuntimeError(f"Did not save distilled model: {output_model_path}")
    source_manifest = read_model_manifest(model_path)
    source_manifest.update(
        {
            "task_id": env.contract["task_id"],
            "model": str(output_model_path),
            "model_sha256": sha256_file(output_model_path),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "world": str(world_path),
            "world_sha256": sha256_file(world_path),
            "world_dependencies": file_hash_records(
                world_dependency_paths
            ),
            "environment_contract": env.contract,
            "distilled_from": {
                "model": str(model_path),
                "model_sha256": sha256_file(model_path),
                "successful_episodes": success_count,
                "demonstration_steps": len(successful_observations),
                "behavior_cloning_epochs": args.epochs,
                "final_loss": losses[-1],
                "optimizer_state_reusable": False,
            },
        }
    )
    output_manifest_path = model_manifest_path(output_model_path)
    write_model_manifest(output_manifest_path, source_manifest)
    report.update(
        {
            "status": "PASS",
            "source_contract_verification": verification,
            "episodes_attempted": len(episode_records),
            "successful_episodes": success_count,
            "demonstration_steps": len(successful_observations),
            "zero_residual_approach_steps": int(np.sum(pre_stair_mask)),
            "terrain_trigger_observation_column": terrain_column,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_history": losses,
            "episode_records": episode_records,
            "output_model_sha256": sha256_file(output_model_path),
            "output_manifest": str(output_manifest_path),
            "scope": (
                "Behavior cloning of successful stochastic residual-policy "
                "episodes; deterministic evaluation is still required."
            ),
        }
    )
    exit_code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    report["elapsed_seconds"] = time.perf_counter() - start_time
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if env is not None:
        env.close()
    simulation_app.close()

print("DROBOT_STAIRS_DISTILL_RESULT=" + json.dumps(report, sort_keys=True))
sys.exit(exit_code)
