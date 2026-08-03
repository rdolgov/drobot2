"""Expand a persistent two-mode checkpoint with learned episode-bias heads."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--action-std", type=float, default=0.08)
    parser.add_argument("--bias-std", type=float, default=0.20)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    output_weight = actor["mlp.4.weight"]
    output_bias = actor["mlp.4.bias"]
    if output_weight.shape[0] != 26:
        raise ValueError(f"expected 26-row two-mode head, got {output_weight.shape}")

    widened_weight = output_weight.new_zeros(50, output_weight.shape[1])
    widened_bias = output_bias.new_zeros(50)
    widened_weight[:26] = output_weight
    widened_bias[:26] = output_bias
    actor["mlp.4.weight"] = widened_weight
    actor["mlp.4.bias"] = widened_bias
    old_std = actor.pop("distribution.std_param")
    actor["distribution.action_std_param"] = torch.full_like(old_std, args.action_std)
    actor["distribution.bias_std_param"] = torch.full_like(old_std, args.bias_std)

    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["persistent_bias_transplant"] = {
        "source": str(args.source.resolve()),
        "control_head_preserved": True,
        "bias_head_initialized_to_zero": True,
        "action_std": args.action_std,
        "bias_std": args.bias_std,
        "optimizer_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved={args.output} actor_rows={widened_weight.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
