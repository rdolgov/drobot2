"""Create an evaluation-only checkpoint that always selects one mixture mode."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    with torch.no_grad():
        actor["mlp.4.weight"][:2].zero_()
        actor["mlp.4.bias"][:2].fill_(-20.0)
        actor["mlp.4.bias"][args.mode] = 20.0
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["forced_two_mode_evaluation"] = {
        "source": str(args.source.resolve()),
        "mode": args.mode,
        "training_checkpoint": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved={args.output} forced_mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
