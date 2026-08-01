"""Train PPO residuals around the Drobot 190 mm front-foot reference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch._dynamo  # noqa: F401
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
STAIRS_DIR = RL_DIR / "stairs"
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(STAIRS_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _foot_lift_contract import FOOT_LIFT_OBSERVATION_SIZE  # noqa: E402
from _policy_transfer import transfer_policy_state  # noqa: E402
from _run_support import (  # noqa: E402
    build_model_manifest,
    expected_ppo_algorithm_contract,
    file_hash_records,
    model_manifest_path,
    sha256_file,
    validate_ppo_algorithm_contract,
    write_model_manifest,
)

parser = argparse.ArgumentParser(
    description="Train the Drobot 190 mm single-foot-lift PPO residual policy."
)
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_foot_lift_v1.yaml"),
)
parser.add_argument("--world", default=None)
parser.add_argument(
    "--output-dir",
    default="simulation/isaac/output/rl/ppo-foot-lift-v1-190mm",
)
parser.add_argument("--total-timesteps", type=int, default=None)
parser.add_argument("--seed", type=int, default=190)
parser.add_argument("--device", default="cpu")
parser.add_argument(
    "--smoke-test",
    action="store_true",
    help="Run 512 PPO steps to validate the complete training pipeline.",
)
parser.add_argument("--gui", action="store_true")
args, _ = parser.parse_known_args()


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if args.total_timesteps is not None and args.total_timesteps <= 0:
    parser.error("--total-timesteps must be positive")
config_path = _resolve_project_path(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
if int(config.get("schema_version", 0)) != 1:
    parser.error(f"Unsupported foot-lift config schema: {config.get('schema_version')}")
task_config = dict(config["task"])
ppo_config = dict(config["ppo"])
world_path = _resolve_project_path(args.world or task_config["world"])
world_dependency_paths = tuple(
    _resolve_project_path(value) for value in task_config.get("world_dependencies", ())
)
output_dir = _resolve_project_path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
report_path = output_dir / "training_report.json"

if args.smoke_test:
    total_timesteps = args.total_timesteps or 512
    rollout_steps = min(128, total_timesteps)
    batch_size = min(64, rollout_steps)
    epochs = min(2, int(ppo_config["epochs"]))
    checkpoint_frequency = max(128, rollout_steps)
    training_mode = "smoke"
else:
    total_timesteps = args.total_timesteps or int(ppo_config["total_timesteps"])
    rollout_steps = int(ppo_config["rollout_steps"])
    batch_size = int(ppo_config["batch_size"])
    epochs = int(ppo_config["epochs"])
    checkpoint_frequency = int(ppo_config["checkpoint_frequency_steps"])
    training_mode = "full"
if rollout_steps < 2 or batch_size < 2 or rollout_steps % batch_size:
    parser.error("rollout_steps must be divisible by batch_size and both >= 2")
algorithm_contract = expected_ppo_algorithm_contract(
    ppo_config,
    training_mode=training_mode,
    rollout_steps=rollout_steps,
    batch_size=batch_size,
    epochs=epochs,
    observation_size=FOOT_LIFT_OBSERVATION_SIZE,
    action_size=12,
)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": not args.gui,
        "width": 1280,
        "height": 720,
    }
)

import stable_baselines3  # noqa: E402
from _quadruped_foot_lift_env import QuadrupedFootLiftEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "config": str(config_path),
    "config_sha256": sha256_file(config_path),
    "world": str(world_path),
    "world_sha256": sha256_file(world_path) if world_path.is_file() else None,
    "world_dependencies": [str(path) for path in world_dependency_paths],
    "output_dir": str(output_dir),
    "smoke_test": args.smoke_test,
    "training_mode": training_mode,
    "requested_total_timesteps": total_timesteps,
    "seed": args.seed,
    "device_request": args.device,
    "isaac_sim_version": "6.0.1",
    "torch_version": torch.__version__,
    "stable_baselines3_version": stable_baselines3.__version__,
    "cuda_available": torch.cuda.is_available(),
}
exit_code = 1
raw_env: QuadrupedFootLiftEnv | None = None
monitored_env = None
start_time = time.perf_counter()

try:
    if not world_path.is_file():
        raise FileNotFoundError(world_path)
    file_hash_records(world_dependency_paths)
    raw_env = QuadrupedFootLiftEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
        render_mode="human" if args.gui else None,
    )
    monitored_env = Monitor(raw_env, filename=str(output_dir / "monitor.csv"))
    policy_kwargs = {
        "activation_fn": torch.nn.ELU,
        "net_arch": list(ppo_config["policy_hidden_layers"]),
        "log_std_init": float(ppo_config["initial_log_std"]),
    }
    model = PPO(
        "MlpPolicy",
        monitored_env,
        learning_rate=float(ppo_config["learning_rate"]),
        n_steps=rollout_steps,
        batch_size=batch_size,
        n_epochs=epochs,
        gamma=float(ppo_config["gamma"]),
        gae_lambda=float(ppo_config["gae_lambda"]),
        clip_range=float(ppo_config["clip_range"]),
        ent_coef=float(ppo_config["entropy_coefficient"]),
        vf_coef=float(ppo_config["value_coefficient"]),
        max_grad_norm=float(ppo_config["max_gradient_norm"]),
        target_kl=float(ppo_config["target_kl"]),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(output_dir / "tensorboard"),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    transferred_from: Path | None = None
    initialization_config = dict(task_config.get("initialization", {}))
    flat_model_value = initialization_config.get("flat_model")
    if flat_model_value:
        transferred_from = _resolve_project_path(str(flat_model_value))
        if not transferred_from.is_file():
            raise FileNotFoundError(transferred_from)
        source_model = PPO.load(str(transferred_from), device=args.device)
        if tuple(source_model.observation_space.shape) != (48,):
            raise RuntimeError("Flat initializer must use the 48-value walk input")
        if tuple(source_model.action_space.shape) != (12,):
            raise RuntimeError("Flat initializer must use the 12-joint action")
        transferred_state, transfer_report = transfer_policy_state(
            source_model.policy.state_dict(),
            model.policy.state_dict(),
            source_observation_size=48,
        )
        if transfer_report["expanded_input_count"] != 2 or transfer_report["skipped"]:
            raise RuntimeError(f"Flat policy transfer was incomplete: {transfer_report}")
        model.policy.load_state_dict(transferred_state, strict=True)
        model.policy.log_std.data.fill_(float(ppo_config["initial_log_std"]))
        report["flat_policy_transfer"] = {
            "source_model": str(transferred_from),
            "source_model_sha256": sha256_file(transferred_from),
            "source_observation_size": 48,
            "target_observation_size": FOOT_LIFT_OBSERVATION_SIZE,
            "new_input_weights": "zero",
            "log_std_overridden": float(ppo_config["initial_log_std"]),
            **transfer_report,
        }
        del source_model
    elif bool(ppo_config.get("zero_action_mean_init", False)):
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
        report["zero_action_mean_initialization"] = True
    algorithm_verification = validate_ppo_algorithm_contract(
        model,
        algorithm_contract,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_frequency,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="drobot_foot_lift_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=True,
        progress_bar=False,
    )
    final_model_base = output_dir / "drobot_foot_lift_ppo_final"
    model.save(str(final_model_base))
    final_model_path = final_model_base.with_suffix(".zip")
    if not final_model_path.is_file():
        raise RuntimeError(f"Stable-Baselines3 did not save {final_model_path}")
    manifest = build_model_manifest(
        model_path=final_model_path,
        config_path=config_path,
        world_path=world_path,
        world_dependencies=world_dependency_paths,
        environment_contract=raw_env.contract,
        algorithm_contract=algorithm_contract,
        training_seed=args.seed,
        transferred_from=transferred_from,
    )
    manifest_path = model_manifest_path(final_model_path)
    write_model_manifest(manifest_path, manifest)
    report.update(
        {
            "status": "PASS",
            "actual_total_timesteps": int(model.num_timesteps),
            "model": str(final_model_path),
            "model_bytes": final_model_path.stat().st_size,
            "model_manifest": str(manifest_path),
            "ppo_algorithm_verification": algorithm_verification,
            "environment_contract": raw_env.contract,
            "recent_completed_episodes": raw_env.completed_episode_metrics,
            "elapsed_seconds": time.perf_counter() - start_time,
            "scope": (
                "Pipeline validation only"
                if args.smoke_test
                else "Single-environment foot-lift PPO training"
            ),
            "policy_inputs": (
                "IMU, joint state, prior action, lift target/progress, and "
                "base drift; no RGB or terrain perception."
            ),
        }
    )
    exit_code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
    report["elapsed_seconds"] = time.perf_counter() - start_time
finally:
    if monitored_env is not None:
        monitored_env.close()
    elif raw_env is not None:
        raw_env.close()
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_FOOT_LIFT_TRAIN_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
