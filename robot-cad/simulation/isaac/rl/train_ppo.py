"""Train a PPO walking policy against the validated Isaac Sim quadruped."""

from __future__ import annotations

import argparse
import hashlib
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
ISAAC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(ISAAC_DIR) not in sys.path:
    sys.path.insert(0, str(ISAAC_DIR))

parser = argparse.ArgumentParser(
    description="Train the Drobot quadruped forward-walking PPO policy."
)
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_walk_v1.yaml"),
)
parser.add_argument(
    "--world",
    default=None,
    help="Override the task world's repository-relative or absolute path.",
)
parser.add_argument(
    "--output-dir",
    default="simulation/isaac/output/rl/ppo-walk-v1",
)
parser.add_argument("--total-timesteps", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--device",
    default="cpu",
    help="SB3 policy device. CPU is faster for this small single-env MLP.",
)
parser.add_argument("--resume", default=None, help="Optional SB3 PPO .zip checkpoint")
parser.add_argument(
    "--smoke-test",
    action="store_true",
    help="Run 512 steps with small PPO batches to validate the full pipeline.",
)
parser.add_argument(
    "--gui",
    action="store_true",
    help="Render Isaac Sim while training. Headless is the normal fast mode.",
)
args, _ = parser.parse_known_args()


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


config_path = _resolve_project_path(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
if int(config.get("schema_version", 0)) != 1:
    parser.error(f"Unsupported RL config schema: {config.get('schema_version')}")
task_config = dict(config["task"])
ppo_config = dict(config["ppo"])
world_path = _resolve_project_path(args.world or task_config["world"])
output_dir = _resolve_project_path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
report_path = output_dir / "training_report.json"

if args.total_timesteps is not None and args.total_timesteps <= 0:
    parser.error("--total-timesteps must be positive")
if args.smoke_test:
    total_timesteps = args.total_timesteps or 512
    rollout_steps = min(128, total_timesteps)
    batch_size = min(64, rollout_steps)
    epochs = min(2, int(ppo_config["epochs"]))
    checkpoint_frequency = max(128, rollout_steps)
else:
    total_timesteps = args.total_timesteps or int(ppo_config["total_timesteps"])
    rollout_steps = int(ppo_config["rollout_steps"])
    batch_size = int(ppo_config["batch_size"])
    epochs = int(ppo_config["epochs"])
    checkpoint_frequency = int(ppo_config["checkpoint_frequency_steps"])
if rollout_steps < 2 or batch_size < 2 or rollout_steps % batch_size:
    parser.error("PPO rollout_steps must be divisible by batch_size and both >= 2")

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": not args.gui,
        "width": 1280,
        "height": 720,
    }
)

# Imports that depend on Omniverse or the tested RL runtime follow SimulationApp.
import stable_baselines3  # noqa: E402
from _quadruped_rl_env import QuadrupedWalkEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "config": str(config_path),
    "config_sha256": _sha256(config_path),
    "world": str(world_path),
    "world_sha256": _sha256(world_path) if world_path.is_file() else None,
    "output_dir": str(output_dir),
    "smoke_test": args.smoke_test,
    "requested_total_timesteps": total_timesteps,
    "seed": args.seed,
    "device_request": args.device,
    "isaac_sim_version": "6.0.1",
    "torch_version": torch.__version__,
    "stable_baselines3_version": stable_baselines3.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    ),
}
exit_code = 1
raw_env: QuadrupedWalkEnv | None = None
monitored_env = None
start_time = time.perf_counter()

try:
    if not world_path.is_file():
        raise FileNotFoundError(world_path)
    raw_env = QuadrupedWalkEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
        render_mode="human" if args.gui else None,
    )
    monitored_env = Monitor(
        raw_env,
        filename=str(output_dir / "monitor.csv"),
    )
    policy_kwargs = {
        "activation_fn": torch.nn.ELU,
        "net_arch": list(ppo_config["policy_hidden_layers"]),
    }
    if args.resume:
        resume_path = _resolve_project_path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        model = PPO.load(
            str(resume_path),
            env=monitored_env,
            device=args.device,
        )
        reset_num_timesteps = False
        report["resume_checkpoint"] = str(resume_path)
    else:
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
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(output_dir / "tensorboard"),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        reset_num_timesteps = True
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_frequency,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="drobot_walk_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=False,
    )
    final_model_base = output_dir / "drobot_walk_ppo_final"
    model.save(str(final_model_base))
    final_model_path = final_model_base.with_suffix(".zip")
    if not final_model_path.is_file():
        raise RuntimeError(f"Stable-Baselines3 did not save {final_model_path}")
    report.update(
        {
            "status": "PASS",
            "actual_total_timesteps": int(model.num_timesteps),
            "model": str(final_model_path),
            "model_bytes": final_model_path.stat().st_size,
            "ppo": {
                "rollout_steps": rollout_steps,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": float(ppo_config["learning_rate"]),
                "gamma": float(ppo_config["gamma"]),
                "gae_lambda": float(ppo_config["gae_lambda"]),
                "clip_range": float(ppo_config["clip_range"]),
            },
            "environment_contract": raw_env.contract,
            "recent_completed_episodes": raw_env.completed_episode_metrics,
            "elapsed_seconds": time.perf_counter() - start_time,
            "scope": (
                "Pipeline validation only" if args.smoke_test
                else "Single-environment PPO training"
            ),
            "camera_training_input": False,
            "camera_note": (
                "The mounted camera remains in the world for evaluation. "
                "Version 1 trains from IMU and joint state only."
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
    print("DROBOT_RL_TRAIN_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
