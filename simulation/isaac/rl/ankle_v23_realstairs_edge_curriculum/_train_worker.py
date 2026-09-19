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
parser.add_argument("--task", default="Drobot-Ankle-V23-RealStairsEdgeCurriculum")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=2501)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--run_name", required=True)
parser.add_argument("--rise", type=float, default=.18)
parser.add_argument("--support-weight", type=float, choices=(0.,2.), default=0.)
parser.add_argument("--warm-start")
parser.add_argument("--strategy", default="free")
parser.add_argument("--guidance", type=int, choices=(0, 1), default=1)
parser.add_argument("--phases", type=int, choices=(1, 2, 4, 6, 8, 10, 14, 18, 22, 26), default=6)
parser.add_argument("--teacher", type=int, choices=(0,1), default=0)
args, overrides = parser.parse_known_args()
if args.warm_start:
    raise ValueError('This direct joint curriculum starts a fresh policy; no actor transfer is defined')
metrics = next((v.split("=", 1)[1] for v in overrides if v.startswith("env.metrics_path=")), None)
if not metrics or len(overrides) != 1:
    raise ValueError("Supply exactly one env.metrics_path=... and no unsupported options")
if args.rise != .18 or args.phases != 1 or args.teacher != 1:
    raise ValueError("This experiment requires four 180 mm stairs, one contact-driven task, and a privileged teacher")
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
app = launcher.app
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ankle_v23_realstairs_edge_curriculum
from ankle_v23_realstairs_edge_curriculum.env_cfg import RealStairsEdgeCurriculumCfg
from ankle_v23_realstairs_edge_curriculum.ppo_cfg import RealStairsEdgeCurriculumPPOCfg

root = Path(__file__).resolve().parents[4]
cfg = RealStairsEdgeCurriculumCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.metrics_path = metrics
cfg.stair_rise_m = args.rise
cfg.strategy = args.strategy
cfg.guidance = bool(args.guidance)
cfg.phase_count = args.phases
cfg.episode_length_s = 320.
cfg.support_weight = args.support_weight
cfg.privileged_actor = bool(args.teacher)
cfg.observation_space = 567
cfg.state_space = 583
agent = handle_deprecated_rsl_rl_cfg(RealStairsEdgeCurriculumPPOCfg(), metadata.version("rsl-rl-lib"))
agent.seed, agent.max_iterations, agent.run_name = args.seed, args.max_iterations, args.run_name
folder = root / "logs/rsl_rl" / agent.experiment_name / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
folder.mkdir(parents=True, exist_ok=False)
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=1.)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(folder), device=agent.device)
if not args.warm_start:
    import math
    import torch
    head = [m for m in runner.alg._raw_actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        head.weight.zero_()
        head.bias[:16].zero_()
        head.bias[16:].fill_(math.log(14.))
    print('POLICY_INITIALIZATION=zero-mean residual actions; Beta(8,8); state-driven reference assistance in guided phases; no actor transfer', flush=True)
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
        if key == "rnn.rnn.weight_ih_l0" and value.shape[1] in (469,485,508) and target.shape[1] in (485,508,512) and value.shape[1]<target.shape[1]:
            replacement = torch.zeros_like(target)
            replacement[:, :value.shape[1]] = value
        elif key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std") and value.shape[-1] in (469,485,508) and value.shape[-1]<target.shape[-1]:
            replacement = target.clone()
            replacement[:, :value.shape[-1]] = value
        else:
            raise ValueError(f"Unsupported actor transfer shape: {key} {value.shape} -> {target.shape}")
        actor_state[key] = replacement
        expanded.append(key)
    runner.alg.load(loaded, load_cfg={"actor": True, "critic": False,
                    "optimizer": False, "iteration": False, "rnd": False}, strict=True)
    print("ACTOR_TRANSFER="+json.dumps({"expanded": expanded, "new_reference_columns": "zero input weights; unit initial normalization", "critic_optimizer": "fresh"}), flush=True)
# Transfer the recurrent encoder and normalization, then initialize a fresh
# direct mean/concentration head. This is not an exact actor continuation.
if args.warm_start:
    import math
    import torch
    head = [m for m in runner.alg._raw_actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        head.weight.zero_()
        head.bias[:16].zero_()
        head.bias[16:].fill_(math.log(158.))
    print('POLICY_HEAD_RESET=encoder transfer; fresh zero-mean Beta(80,80) direct-mean head',flush=True)

dump_yaml(str(folder / "params/env.yaml"), cfg)
dump_yaml(str(folder / "params/agent.yaml"), agent)
(folder / "run.json").write_text(json.dumps({"arguments": vars(args), "metrics_path": metrics,
    "init_at_random_ep_len": False, "evidence": "training_only"}, indent=2) + "\n")
# Capture the exact behavior policy when a floor-start full climb occurs inside
# ordinary training collection, before the next PPO update changes its weights.
original_step = env.step
last_full_event = [0]
last_plant_event = [0]
plant_behavior_exports = [0]
last_pair_event = [0]
pair_behavior_exports = [0]
last_ground_goal_event = [0]
last_curriculum_goal_event = [0]
def training_step(actions):
    result = original_step(actions)
    plants = env.unwrapped.plant_event_count
    if plants > last_plant_event[0]:
        last_plant_event[0] = plants
        if plant_behavior_exports[0] < 4:
            plant_behavior_exports[0] += 1
            runner.save(str(folder / f"first_foothold_behavior_{plants}.pt"))
            (folder / "first_foothold_behavior.json").write_text(json.dumps({
                "count":plants, "step":env.unwrapped.common_step_counter,
                "checkpoint":f"first_foothold_behavior_{plants}.pt",
                "evidence":"new-foot/tread credit only; no whole-robot stage or full climb; prior trajectory chunks may use earlier policy weights"},indent=2)+"\n")
    pairs = env.unwrapped.front_pair_event_count
    if pairs > last_pair_event[0]:
        last_pair_event[0] = pairs
        if pair_behavior_exports[0] < 4:
            pair_behavior_exports[0] += 1
            runner.save(str(folder / f"front_pair_behavior_{pairs}.pt"))
            (folder / "front_pair_behavior.json").write_text(json.dumps({
                "count":pairs,"step":env.unwrapped.common_step_counter,
                "checkpoint":f"front_pair_behavior_{pairs}.pt",
                "evidence":"simultaneous front-foot hold only, not stage/full; earlier chunks may use earlier weights"},indent=2)+"\n")
    count = env.unwrapped.full_event_count
    if count > last_full_event[0]:
        last_full_event[0] = count
        runner.save(str(folder / f"full_climb_behavior_{count}.pt"))
        (folder / "full_climb_behavior.json").write_text(json.dumps({
            "count":count, "step":env.unwrapped.common_step_counter,
            "checkpoint":f"full_climb_behavior_{count}.pt", "privileged_actor":cfg.privileged_actor,
            "evidence":"actor_weights_at_event; earlier trajectory chunks may use earlier weights"}, indent=2)+"\n")
    stage_count = env.unwrapped.ground_goal_event_count
    if stage_count > last_ground_goal_event[0]:
        last_ground_goal_event[0] = stage_count
        runner.save(str(folder / f"ground_stage_behavior_{stage_count}.pt"))
        (folder / "ground_stage_behavior.json").write_text(json.dumps({
            "count":stage_count, "step":env.unwrapped.common_step_counter,
            "checkpoint":f"ground_stage_behavior_{stage_count}.pt", "privileged_actor":cfg.privileged_actor,
            "evidence":"stage only, actor weights at event; earlier trajectory chunks may use earlier weights"},indent=2)+"\n")
    curriculum_count = env.unwrapped.curriculum_goal_event_count
    if curriculum_count > last_curriculum_goal_event[0]:
        last_curriculum_goal_event[0] = curriculum_count
        runner.save(str(folder / f"curriculum_stage_behavior_{curriculum_count}.pt"))
        (folder / "curriculum_stage_behavior.json").write_text(json.dumps({
            "count":curriculum_count,"step":env.unwrapped.common_step_counter,
            "checkpoint":f"curriculum_stage_behavior_{curriculum_count}.pt",
            "evidence":"curriculum start with three feet on tread 1; not a floor-start climb; episode continues beyond first tread"},indent=2)+"\n")
    return result
env.step = training_step
try:
    start = time.time()
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    print(f"Training time: {time.time() - start:.2f} seconds", flush=True)
finally:
    env.close()
    app.close()


