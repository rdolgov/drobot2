"""Seed the parallel RSL actor from the previously trained pure-RL SB3 walker."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from stable_baselines3 import PPO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sb3",
        type=Path,
        default=Path(
            "simulation/isaac/output/rl/ppo-walk-v1-2m/"
            "drobot_walk_ppo_final.zip"
        ),
    )
    parser.add_argument("--initial-std", type=float, default=0.1)
    parser.add_argument(
        "--actor-output-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the transferred final actor layer. Values below one move "
            "an SB3 mean that relied on action clipping back into PPO's "
            "learnable, unsaturated range."
        ),
    )
    args = parser.parse_args()
    if not 0.0 < args.initial_std <= 1.0:
        parser.error("--initial-std must be in (0, 1]")
    if not 0.0 < args.actor_output_scale <= 1.0:
        parser.error("--actor-output-scale must be in (0, 1]")
    for path in (args.template, args.sb3):
        if not path.is_file():
            parser.error(f"checkpoint does not exist: {path}")

    checkpoint = torch.load(args.template, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    source_policy = PPO.load(str(args.sb3.resolve()), device="cpu").policy.state_dict()
    mapping = {
        "mlp.0.weight": "mlp_extractor.policy_net.0.weight",
        "mlp.0.bias": "mlp_extractor.policy_net.0.bias",
        "mlp.2.weight": "mlp_extractor.policy_net.2.weight",
        "mlp.2.bias": "mlp_extractor.policy_net.2.bias",
        "mlp.4.weight": "action_net.weight",
        "mlp.4.bias": "action_net.bias",
    }
    for target_name, source_name in mapping.items():
        source = source_policy[source_name]
        if target_name not in actor or actor[target_name].shape != source.shape:
            raise ValueError(
                f"incompatible actor layer {target_name}: "
                f"target={actor.get(target_name, None)} source_shape={tuple(source.shape)}"
            )
        actor[target_name] = source.detach().clone()
    actor["mlp.4.weight"] *= args.actor_output_scale
    actor["mlp.4.bias"] *= args.actor_output_scale
    actor["distribution.log_std_param"] = torch.full_like(
        actor["distribution.log_std_param"], math.log(args.initial_std)
    )

    # The template supplies a compatible fresh critic.  Optimizer moments from
    # its smoke update must not be applied to the transferred actor weights.
    optimizer = checkpoint.get("optimizer_state_dict")
    if isinstance(optimizer, dict):
        optimizer["state"] = {}
    checkpoint["iter"] = 0
    checkpoint["infos"] = {
        "bootstrap": "pure_rl_sb3_actor",
        "source": str(args.sb3.resolve()),
        "initial_std": args.initial_std,
        "actor_output_scale": args.actor_output_scale,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Wrote transferred pure-RL actor checkpoint: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
