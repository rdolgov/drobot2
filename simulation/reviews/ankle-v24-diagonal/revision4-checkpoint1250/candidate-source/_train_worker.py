"""One authorized training job; no automatic evaluation or deployment."""
import argparse
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import sys
import time
import traceback

import warp as wp
wp.config.enable_backward = False

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--warm-start", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=2419)
parser.add_argument("--run-name", default="diagonal-2419")
args = parser.parse_args()

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
app = launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ankle_v24_diagonal
from ankle_v24_diagonal.env_cfg import DiagonalCfg, ROOT
from ankle_v24_diagonal.ppo_cfg import DiagonalPPOCfg

cfg = DiagonalCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.metrics_path = str(args.output / "train.metrics.jsonl")
agent = handle_deprecated_rsl_rl_cfg(DiagonalPPOCfg(), metadata.version("rsl-rl-lib"))
agent.seed = args.seed
agent.max_iterations = args.iterations
agent.run_name = args.run_name
folder = ROOT / "logs/rsl_rl" / agent.experiment_name / (
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
folder.mkdir(parents=True, exist_ok=False)
(args.output / "run-path.json").write_text(json.dumps({"directory": str(folder)}, indent=2) + "\n")
env = RslRlVecEnvWrapper(gym.make("Drobot-Ankle-V24-Diagonal", cfg=cfg), clip_actions=1.)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(folder), device=agent.device)
try:
    runner.load(str(args.warm_start), load_cfg={"actor": True, "critic": False,
                "optimizer": False, "iteration": False, "rnd": False})
    (folder / "diagonal-run.json").write_text(json.dumps({
        "arguments": vars(args), "actor_initialization": str(args.warm_start),
        "critic_optimizer_iteration": "fresh", "config": agent.to_dict(),
        "initial_episode_age": "zero; scheduler starts at its actual initial state",
        "evaluation_requested": False, "deployment_requested": False,
    }, indent=2, default=str) + "\n")
    start = time.time()
    # Random episode ages would disagree with the stateful reference scheduler.
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    print(f"Training time: {time.time() - start:.2f} seconds", flush=True)
except BaseException:
    # Kit's fast shutdown can exit before Python prints an uncaught exception.
    # Persist it before closing the simulation so a failed job is diagnosable.
    failure = traceback.format_exc()
    (args.output / "training-error.txt").write_text(failure, encoding="utf-8")
    print(failure, file=sys.stderr, flush=True)
    raise
finally:
    try:
        env.unwrapped.write_diagnostic_summary()
    finally:
        try:
            env.close()
        finally:
            app.close()
