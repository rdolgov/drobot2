"""Transfer a learned pure-RL walking actor into the current RSL policy."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from stable_baselines3 import PPO


def _copy_compatible_state(
    target: dict[str, torch.Tensor], source: dict[str, torch.Tensor]
) -> None:
    for name, value in source.items():
        if name in target and target[name].shape == value.shape:
            target[name] = value.detach().clone()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sb3",
        type=Path,
        default=Path(
            "simulation/isaac/models/ppo-walk-v1-2m/"
            "drobot_walk_ppo_final.zip"
        ),
    )
    parser.add_argument(
        "--rsl-source",
        type=Path,
        help="Use a learned Gaussian RSL checkpoint instead of the SB3 model.",
    )
    parser.add_argument("--initial-std", type=float, default=0.1)
    parser.add_argument(
        "--actor-output-scale",
        type=float,
        default=1.0,
        help="Scale the learned Gaussian output before transfer.",
    )
    parser.add_argument("--beta-concentration-bias", type=float, default=4.0)
    parser.add_argument("--beta-logit-gain", type=float, default=5.0)
    args = parser.parse_args()
    if not 0.0 < args.initial_std <= 1.0:
        parser.error("--initial-std must be in (0, 1]")
    if not 0.0 < args.actor_output_scale <= 1.0:
        parser.error("--actor-output-scale must be in (0, 1]")
    if args.beta_concentration_bias <= 0.0 or args.beta_logit_gain <= 0.0:
        parser.error("Beta concentration bias and logit gain must be positive")

    source_path = args.rsl_source or args.sb3
    for path in (args.template, source_path):
        if not path.is_file():
            parser.error(f"checkpoint does not exist: {path}")

    checkpoint = torch.load(args.template, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    if args.rsl_source:
        source_checkpoint = torch.load(
            args.rsl_source, map_location="cpu", weights_only=False
        )
        source_actor = source_checkpoint["actor_state_dict"]
        source_layers = {
            "hidden0_weight": source_actor["mlp.0.weight"],
            "hidden0_bias": source_actor["mlp.0.bias"],
            "hidden1_weight": source_actor["mlp.2.weight"],
            "hidden1_bias": source_actor["mlp.2.bias"],
            "output_weight": source_actor["mlp.4.weight"],
            "output_bias": source_actor["mlp.4.bias"],
        }
        if "critic_state_dict" in source_checkpoint:
            _copy_compatible_state(
                checkpoint["critic_state_dict"],
                source_checkpoint["critic_state_dict"],
            )
        source_kind = "pure_rl_rsl_gaussian_actor"
    else:
        source_policy = PPO.load(
            str(args.sb3.resolve()), device="cpu"
        ).policy.state_dict()
        source_layers = {
            "hidden0_weight": source_policy["mlp_extractor.policy_net.0.weight"],
            "hidden0_bias": source_policy["mlp_extractor.policy_net.0.bias"],
            "hidden1_weight": source_policy["mlp_extractor.policy_net.2.weight"],
            "hidden1_bias": source_policy["mlp_extractor.policy_net.2.bias"],
            "output_weight": source_policy["action_net.weight"],
            "output_bias": source_policy["action_net.bias"],
        }
        source_kind = "pure_rl_sb3_actor"

    hidden_mapping = {
        "mlp.0.weight": "hidden0_weight",
        "mlp.0.bias": "hidden0_bias",
        "mlp.2.weight": "hidden1_weight",
        "mlp.2.bias": "hidden1_bias",
    }
    for target_name, source_name in hidden_mapping.items():
        source = source_layers[source_name]
        if target_name not in actor or actor[target_name].shape != source.shape:
            raise ValueError(
                f"incompatible actor layer {target_name}: "
                f"target_shape={tuple(actor[target_name].shape)} "
                f"source_shape={tuple(source.shape)}"
            )
        actor[target_name] = source.detach().clone()

    source_weight = source_layers["output_weight"]
    source_bias = source_layers["output_bias"]
    target_weight = actor["mlp.4.weight"]
    target_bias = actor["mlp.4.bias"]
    if source_weight.shape[0] != 12:
        raise ValueError(f"expected 12 source actions, got {source_weight.shape}")
    if target_weight.shape[0] == 12:
        actor["mlp.4.weight"] = (
            source_weight.detach().clone() * args.actor_output_scale
        )
        actor["mlp.4.bias"] = (
            source_bias.detach().clone() * args.actor_output_scale
        )
        actor["distribution.log_std_param"] = torch.full_like(
            actor["distribution.log_std_param"], math.log(args.initial_std)
        )
        target_distribution = "gaussian"
    elif target_weight.shape[0] == 24:
        scaled_weight = (
            args.beta_logit_gain * args.actor_output_scale * source_weight
        )
        scaled_bias = args.beta_logit_gain * args.actor_output_scale * source_bias
        actor["mlp.4.weight"] = torch.cat(
            (scaled_weight, -scaled_weight), dim=0
        ).detach()
        actor["mlp.4.bias"] = torch.cat(
            (
                args.beta_concentration_bias + scaled_bias,
                args.beta_concentration_bias - scaled_bias,
            ),
            dim=0,
        ).detach()
        target_distribution = "bounded_beta"
    else:
        raise ValueError(
            "unsupported target action head: "
            f"weight={tuple(target_weight.shape)} bias={tuple(target_bias.shape)}"
        )

    # Optimizer moments from the template must not be applied to transferred weights.
    optimizer = checkpoint.get("optimizer_state_dict")
    if isinstance(optimizer, dict):
        optimizer["state"] = {}
    checkpoint["iter"] = 0
    checkpoint["infos"] = {
        "bootstrap": source_kind,
        "source": str(source_path.resolve()),
        "target_distribution": target_distribution,
        "initial_std": args.initial_std,
        "actor_output_scale": args.actor_output_scale,
        "beta_concentration_bias": args.beta_concentration_bias,
        "beta_logit_gain": args.beta_logit_gain,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Wrote transferred pure-RL actor checkpoint: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
