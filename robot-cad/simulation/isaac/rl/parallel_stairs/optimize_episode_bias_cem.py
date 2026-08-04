"""Optimize a persistent 12-joint stair strategy from whole-episode rewards.

Every Isaac Lab environment receives the deployable sensor policy plus one
action bias sampled at reset and held for the complete episode. No leg is
selected, no reference trajectory is supplied, and no simulator-only value
enters the actor. Parallel populations prevent symmetric successful strategies
from being averaged back into a no-lift mean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import torch
import warp as wp
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

wp.config.enable_backward = False

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_PARENT))

import parallel_stairs  # noqa: E402, F401


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _population_candidates(
    means: torch.Tensor,
    standard_deviations: torch.Tensor,
    best_biases: torch.Tensor,
    population_size: int,
    generator: torch.Generator,
    max_abs_bias: float,
) -> torch.Tensor:
    """Return center, incumbent, and antithetic candidates per population."""

    if population_size < 4 or population_size % 2:
        raise ValueError("each population must contain an even number >= 4")
    pair_count = (population_size - 2) // 2
    noise = torch.randn(
        means.shape[0],
        pair_count,
        means.shape[1],
        generator=generator,
        device=means.device,
        dtype=means.dtype,
    )
    paired = torch.cat((noise, -noise), dim=1)
    sampled = means[:, None, :] + standard_deviations[:, None, :] * paired
    candidates = torch.cat(
        (means[:, None, :], best_biases[:, None, :], sampled), dim=1
    )
    return candidates.clamp(-max_abs_bias, max_abs_bias)


def _update_populations(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    successes: torch.Tensor,
    means: torch.Tensor,
    standard_deviations: torch.Tensor,
    elite_count: int,
    update_rate: float,
    minimum_std: float,
    maximum_std: float,
    winner_centered: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a diagonal multi-population CEM update and retain incumbents."""

    population_count, population_size, _ = candidates.shape
    if not 1 <= elite_count <= population_size:
        raise ValueError("elite_count must fit inside each population")
    new_means = means.clone()
    new_std = standard_deviations.clone()
    best_biases = candidates.new_empty(population_count, candidates.shape[-1])
    best_scores = scores.new_empty(population_count)
    for population in range(population_count):
        # Success dominates shaped return; within each class the environment's
        # exact complete-episode return is the only ranking signal.
        ranked_score = scores[population] + successes[population].to(scores.dtype) * 1.0e6
        order = ranked_score.argsort(descending=True)
        if winner_centered and bool(successes[population].any()):
            successful_indices = torch.nonzero(
                successes[population], as_tuple=False
            ).flatten()
            anchor = candidates[population, order[0]]
            normalized_delta = (
                candidates[population, successful_indices] - anchor
            ) / standard_deviations[population].clamp_min(1.0e-6)
            nearby = normalized_delta.square().sum(dim=-1).argsort()
            selected = successful_indices[nearby[:elite_count]]
            elite = candidates[population, selected]
        else:
            elite = candidates[population, order[:elite_count]]
        elite_mean = elite.mean(dim=0)
        elite_std = elite.std(dim=0, unbiased=False)
        new_means[population].lerp_(elite_mean, update_rate)
        new_std[population].lerp_(elite_std, update_rate)
        new_std[population].clamp_(minimum_std, maximum_std)
        best_biases[population] = candidates[population, order[0]]
        best_scores[population] = ranked_score[order[0]]
    return new_means, new_std, best_biases, best_scores


def _bake_checkpoint(
    source: Path,
    destination: Path,
    action_bias: torch.Tensor,
    metadata_payload: dict[str, object],
) -> None:
    """Add a constant to both learned bias-center heads and save a checkpoint."""

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    output_bias = actor["mlp.4.bias"].clone()
    output_weight = actor["mlp.4.weight"]
    if output_weight.shape[0] != 50 or output_bias.shape != (50,):
        raise ValueError(
            "episode-bias CEM requires a 2-mode, 12-action persistent-bias head"
        )
    bias = action_bias.detach().to(device="cpu", dtype=output_bias.dtype)
    if bias.shape != (12,):
        raise ValueError(f"expected a 12-value bias, got {tuple(bias.shape)}")
    center_start = 2 + 2 * 12
    for mode in range(2):
        start = center_start + mode * 12
        output_bias[start : start + 12] += bias
    actor["mlp.4.bias"] = output_bias

    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["episode_bias_cem"] = metadata_payload
    optimizer = checkpoint.get("optimizer_state_dict")
    if isinstance(optimizer, dict):
        optimizer["state"] = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-PersistentBias-Hip-Direct",
)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--populations", type=int, default=2)
parser.add_argument("--generations", type=int, default=40)
parser.add_argument("--elite_fraction", type=float, default=0.125)
parser.add_argument("--initial_std", type=float, default=0.24)
parser.add_argument("--initial_mean_std", type=float, default=0.04)
parser.add_argument("--minimum_std", type=float, default=0.025)
parser.add_argument("--maximum_std", type=float, default=0.40)
parser.add_argument("--update_rate", type=float, default=0.55)
parser.add_argument("--max_abs_bias", type=float, default=0.85)
parser.add_argument("--seed", type=int, default=1301)
parser.add_argument(
    "--winner_centered",
    action="store_true",
    help="Average only successful candidates nearest the best successful strategy.",
)
parser.add_argument(
    "--initial_report",
    type=Path,
    help="Resume population means and diagonal deviations from a prior report.",
)
parser.add_argument(
    "--randomized_resets",
    action="store_true",
    help="Retain configured reset joint/lateral randomization during search.",
)
add_frontend_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
args_cli.video = False
sys.argv = [sys.argv[0], *hydra_args]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point", play_mode=False)
def main(env_cfg: object, agent_cfg: object) -> None:
    checkpoint = args_cli.checkpoint.resolve()
    output_dir = args_cli.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args_cli.num_envs % args_cli.populations:
        raise ValueError("num_envs must be divisible by populations")
    population_size = args_cli.num_envs // args_cli.populations
    if population_size % 2:
        raise ValueError("each population must contain an even number of environments")
    elite_count = max(2, round(population_size * args_cli.elite_fraction))

    with launch_simulation(env_cfg, args_cli):
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, metadata.version("rsl-rl-lib")
        )
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device or env_cfg.sim.device
        if not args_cli.randomized_resets:
            if hasattr(env_cfg, "reset_joint_position_noise_rad"):
                env_cfg.reset_joint_position_noise_rad = 0.0
            if hasattr(env_cfg, "reset_lateral_jitter_m"):
                env_cfg.reset_lateral_jitter_m = 0.0

        raw_env = create_isaaclab_env(
            args_cli.task,
            env_cfg,
            args_cli,
            convert_marl_to_single_agent=False,
        )
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        device = torch.device(env.unwrapped.device)
        generator = torch.Generator(device=device).manual_seed(args_cli.seed + 17)

        if args_cli.initial_report is not None:
            initial_report_path = args_cli.initial_report.resolve()
            initial_report = json.loads(initial_report_path.read_text(encoding="utf-8"))
            means = torch.tensor(
                initial_report["final_means"], dtype=torch.float32, device=device
            )
            standard_deviations = torch.tensor(
                initial_report["final_standard_deviations"],
                dtype=torch.float32,
                device=device,
            )
            if means.shape != (args_cli.populations, 12):
                raise ValueError(
                    f"initial report means have shape {tuple(means.shape)}, "
                    f"expected {(args_cli.populations, 12)}"
                )
            if standard_deviations.shape != means.shape:
                raise ValueError("initial report deviations do not match its means")
        else:
            initial_report_path = None
            means = torch.randn(
                args_cli.populations, 12, generator=generator, device=device
            ) * args_cli.initial_mean_std
            standard_deviations = torch.full_like(means, args_cli.initial_std)
        best_biases = means.clone()
        best_ranked_scores = torch.full(
            (args_cli.populations,), -torch.inf, device=device
        )
        history: list[dict[str, object]] = []
        global_best_bias = means[0].clone()
        global_best_ranked_score = -torch.inf
        global_best_return = -torch.inf
        global_best_success = False
        success_threshold = 0.5 * float(env_cfg.success_completion_reward_scale)
        max_steps = int(env.unwrapped.max_episode_length) + 2

        for generation in range(args_cli.generations):
            candidate_grid = _population_candidates(
                means,
                standard_deviations,
                best_biases,
                population_size,
                generator,
                args_cli.max_abs_bias,
            )
            candidates = candidate_grid.reshape(args_cli.num_envs, 12)
            obs, _ = env.reset()
            policy.reset(torch.ones(args_cli.num_envs, dtype=torch.long, device=device))
            episode_returns = torch.zeros(args_cli.num_envs, device=device)
            episode_lengths = torch.zeros(
                args_cli.num_envs, dtype=torch.long, device=device
            )
            successes = torch.zeros(
                args_cli.num_envs, dtype=torch.bool, device=device
            )
            active = torch.ones_like(successes)

            # Isaac retains the last action tensor and mutates it during the
            # next explicit reset. no_grad avoids autograd cost without making
            # that retained tensor immutable as inference_mode would.
            with torch.no_grad():
                for _ in range(max_steps):
                    actions = policy(obs) + candidates
                    obs, rewards, dones, _ = env.step(actions)
                    episode_returns += rewards * active.to(rewards.dtype)
                    episode_lengths += active.long()
                    successes |= active & (rewards >= success_threshold)
                    newly_done = active & dones.bool()
                    active &= ~newly_done
                    policy.reset(dones)
                    if not bool(active.any()):
                        break

            return_grid = episode_returns.reshape(
                args_cli.populations, population_size
            )
            success_grid = successes.reshape(args_cli.populations, population_size)
            old_best_biases = best_biases
            means, standard_deviations, generation_best_biases, generation_scores = (
                _update_populations(
                    candidate_grid,
                    return_grid,
                    success_grid,
                    means,
                    standard_deviations,
                    elite_count,
                    args_cli.update_rate,
                    args_cli.minimum_std,
                    args_cli.maximum_std,
                    args_cli.winner_centered,
                )
            )
            improved = generation_scores > best_ranked_scores
            best_biases = torch.where(
                improved[:, None], generation_best_biases, old_best_biases
            )
            best_ranked_scores = torch.maximum(best_ranked_scores, generation_scores)

            ranked = episode_returns + successes.to(episode_returns.dtype) * 1.0e6
            best_index = int(ranked.argmax().item())
            best_value = float(ranked[best_index].item())
            if best_value > float(global_best_ranked_score):
                global_best_ranked_score = best_value
                global_best_bias = candidates[best_index].clone()
                global_best_return = float(episode_returns[best_index].item())
                global_best_success = bool(successes[best_index].item())

            center_successes = int(success_grid[:, 0].sum().item())
            record = {
                "generation": generation,
                "successes": int(successes.sum().item()),
                "center_successes": center_successes,
                "return_mean": float(episode_returns.mean().item()),
                "return_max": float(episode_returns.max().item()),
                "population_successes": [
                    int(value) for value in success_grid.sum(dim=1).tolist()
                ],
                "population_mean_std": [
                    float(value) for value in standard_deviations.mean(dim=1).tolist()
                ],
                "active_after_limit": int(active.sum().item()),
                "max_episode_steps": int(episode_lengths.max().item()),
            }
            history.append(record)
            print(
                "[CEM] "
                f"generation={generation:03d} successes={record['successes']} "
                f"center_successes={center_successes} "
                f"return_mean={record['return_mean']:.5f} "
                f"return_max={record['return_max']:.5f} "
                f"population_successes={record['population_successes']} "
                f"mean_std={record['population_mean_std']}",
                flush=True,
            )

        timestamp = datetime.now(UTC).isoformat()
        report = {
            "schema_version": 1,
            "algorithm": (
                "winner_centered_multi_population_diagonal_cem"
                if args_cli.winner_centered
                else "multi_population_diagonal_cem"
            ),
            "pure_reward_driven": True,
            "selected_leg_input": False,
            "reference_motion": False,
            "task": args_cli.task,
            "source_checkpoint": str(checkpoint),
            "source_sha256": _sha256(checkpoint),
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "populations": args_cli.populations,
            "population_size": population_size,
            "generations": args_cli.generations,
            "transitions": args_cli.num_envs * args_cli.generations * int(
                env.unwrapped.max_episode_length
            ),
            "elite_count": elite_count,
            "update_rate": args_cli.update_rate,
            "initial_std": args_cli.initial_std,
            "minimum_std": args_cli.minimum_std,
            "max_abs_bias": args_cli.max_abs_bias,
            "randomized_resets": args_cli.randomized_resets,
            "winner_centered": args_cli.winner_centered,
            "initial_report": (
                str(initial_report_path) if initial_report_path is not None else None
            ),
            "global_best_success": global_best_success,
            "global_best_return": global_best_return,
            "global_best_bias": [float(value) for value in global_best_bias.tolist()],
            "final_means": means.tolist(),
            "final_standard_deviations": standard_deviations.tolist(),
            "history": history,
            "created_at": timestamp,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "cem_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        common_metadata = {
            "source_sha256": report["source_sha256"],
            "algorithm": report["algorithm"],
            "pure_reward_driven": True,
            "selected_leg_input": False,
            "reference_motion": False,
            "seed": args_cli.seed,
            "generations": args_cli.generations,
            "num_envs": args_cli.num_envs,
            "report": str(report_path),
            "created_at": timestamp,
            "optimizer_state_reset": True,
        }
        for population in range(args_cli.populations):
            _bake_checkpoint(
                checkpoint,
                output_dir / f"model_cem_population_{population}.pt",
                means[population],
                {
                    **common_metadata,
                    "candidate": "population_mean",
                    "population": population,
                    "action_bias": means[population].tolist(),
                    "standard_deviation": standard_deviations[population].tolist(),
                },
            )
        _bake_checkpoint(
            checkpoint,
            output_dir / "model_cem_best_sample.pt",
            global_best_bias,
            {
                **common_metadata,
                "candidate": "best_sample",
                "successful_rollout": global_best_success,
                "episode_return": global_best_return,
                "action_bias": global_best_bias.tolist(),
            },
        )
        env.close()
        print(
            f"[CEM_DONE] report={report_path} "
            f"best_success={global_best_success} best_return={global_best_return:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
