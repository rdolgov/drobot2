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
parser.add_argument("--task", default="Drobot-Ankle-V23-Compact")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=2501)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--run_name", required=True)
parser.add_argument("--rise", type=float, default=.02)
parser.add_argument("--warm-start")
parser.add_argument("--strategy", default="free")
parser.add_argument("--guidance", type=int, choices=(0, 1), default=1)
parser.add_argument("--phases", type=int, choices=(1, 2, 3, 4, 6, 8, 12, 16, 20, 24), default=6)
parser.add_argument("--teacher", type=int, choices=(0,1), default=0)
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
import ankle_v23_compact
from ankle_v23_compact.env_cfg import CompactCfg
from ankle_v23_compact.ppo_cfg import CompactPPOCfg

root = Path(__file__).resolve().parents[4]
cfg = CompactCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.metrics_path = metrics
cfg.stair_rise_m = args.rise
cfg.strategy = args.strategy
cfg.guidance = bool(args.guidance)
cfg.phase_count = args.phases
cfg.episode_length_s = cfg.phase_count*cfg.phase_s+4.
cfg.privileged_actor = bool(args.teacher)
cfg.observation_space = 508 if cfg.privileged_actor else 485
agent = handle_deprecated_rsl_rl_cfg(CompactPPOCfg(), metadata.version("rsl-rl-lib"))
agent.seed, agent.max_iterations, agent.run_name = args.seed, args.max_iterations, args.run_name
folder = root / "logs/rsl_rl" / agent.experiment_name / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
folder.mkdir(parents=True, exist_ok=False)
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=1.)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(folder), device=agent.device)
if args.warm_start:
    import torch
    loaded = torch.load(str(Path(args.warm_start).resolve()), map_location=agent.device, weights_only=False)
    actor_state = loaded["actor_state_dict"]
    desired_state = runner.alg._raw_actor.state_dict()
    expanded = []
    for key, value in list(actor_state.items()):
        target = desired_state[key]
        if value.shape == target.shape:
            continue
        if key == "rnn.rnn.weight_ih_l0" and value.shape[1] in (469,485) and target.shape[1] in (485,508) and value.shape[1]<target.shape[1]:
            replacement = torch.zeros_like(target)
            replacement[:, :value.shape[1]] = value
        elif key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std") and value.shape[-1] in (469,485) and value.shape[-1]<target.shape[-1]:
            replacement = target.clone()
            replacement[:, :value.shape[-1]] = value
        else:
            raise ValueError(f"Unsupported actor transfer shape: {key} {value.shape} -> {target.shape}")
        actor_state[key] = replacement
        expanded.append(key)
    runner.alg.load(loaded, load_cfg={"actor": True, "critic": False,
                    "optimizer": False, "iteration": False, "rnd": False}, strict=True)
    print("ACTOR_TRANSFER="+json.dumps({"expanded": expanded, "new_reference_columns": "zero input weights; unit initial normalization", "critic_optimizer": "fresh"}), flush=True)
dump_yaml(str(folder / "params/env.yaml"), cfg)
dump_yaml(str(folder / "params/agent.yaml"), agent)
(folder / "run.json").write_text(json.dumps({"arguments": vars(args), "metrics_path": metrics,
    "init_at_random_ep_len": False, "evidence": "training_only"}, indent=2) + "\n")
# Capture the exact behavior policy when a floor-start full climb occurs inside
# ordinary training collection, before the next PPO update changes its weights.
original_step = env.step
last_full_event = [0]
def training_step(actions):
    result = original_step(actions)
    count = env.unwrapped.full_event_count
    if count > last_full_event[0]:
        last_full_event[0] = count
        runner.save(str(folder / f"full_climb_behavior_{count}.pt"))
        (folder / "full_climb_behavior.json").write_text(json.dumps({
            "count":count, "step":env.unwrapped.common_step_counter,
            "checkpoint":f"full_climb_behavior_{count}.pt", "privileged_actor":cfg.privileged_actor,
            "evidence":"exact_behavior_actor_during_training_collection"}, indent=2)+"\n")
    return result
env.step = training_step
try:
    start = time.time()
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    print(f"Training time: {time.time() - start:.2f} seconds", flush=True)
finally:
    env.close()
    app.close()


