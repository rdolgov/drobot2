"""One authorized training job. No evaluation or recording is launched."""
import argparse
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import sys
import time
import warp as wp

wp.config.enable_backward = False
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Drobot-Ankle-V23-Strategy")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=2501)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--run_name", required=True)
parser.add_argument("--rise", type=float, default=.02)
parser.add_argument("--warm-start")
parser.add_argument("--strategy", required=True)
args, overrides = parser.parse_known_args()
metrics = next((v.split("=", 1)[1] for v in overrides if v.startswith("env.metrics_path=")), None)
if not metrics or len(overrides) != 1:
    raise ValueError("Supply exactly one env.metrics_path=... and no unsupported options")
if not .005 <= args.rise <= .18:
    raise ValueError("Stair rise must be between 5 and 180 mm")
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
app = launcher.app
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ankle_v23_experiments
from ankle_v23_experiments.env_cfg import StrategyCfg
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg

root = Path(__file__).resolve().parents[4]
cfg = StrategyCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.metrics_path = metrics
cfg.stair_rise_m = args.rise
cfg.strategy = args.strategy
agent = handle_deprecated_rsl_rl_cfg(StrategyPPOCfg(), metadata.version("rsl-rl-lib"))
agent.seed, agent.max_iterations, agent.run_name = args.seed, args.max_iterations, args.run_name
folder = root / "logs/rsl_rl" / agent.experiment_name / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
folder.mkdir(parents=True, exist_ok=False)
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=1.)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(folder), device=agent.device)
if args.warm_start:
    runner.load(str(Path(args.warm_start).resolve()), load_cfg={"actor": True, "critic": False,
                "optimizer": False, "iteration": False, "rnd": False})
dump_yaml(str(folder / "params/env.yaml"), cfg)
dump_yaml(str(folder / "params/agent.yaml"), agent)
(folder / "run.json").write_text(json.dumps({"arguments": vars(args), "metrics_path": metrics,
    "init_at_random_ep_len": False, "evidence": "training_only"}, indent=2) + "\n")
try:
    start = time.time()
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    print(f"Training time: {time.time() - start:.2f} seconds", flush=True)
finally:
    env.close()
    app.close()

