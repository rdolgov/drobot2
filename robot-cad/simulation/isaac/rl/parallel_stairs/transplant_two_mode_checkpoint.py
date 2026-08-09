"""Transplant a Gaussian PPO checkpoint into a two-mode actor."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transplant(source_path: Path, output_path: Path, *, seed: int, separation: float) -> None:
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    source_weight = actor["mlp.4.weight"]
    source_bias = actor["mlp.4.bias"]
    action_dim, hidden_dim = source_weight.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)

    direction = torch.randn(source_weight.shape, generator=generator)
    direction = separation * direction / direction.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    bias_direction = torch.randn(source_bias.shape, generator=generator)
    bias_direction = separation * bias_direction / bias_direction.norm().clamp_min(1.0e-12)

    mixture_weight = torch.zeros(2 + 2 * action_dim, hidden_dim, dtype=source_weight.dtype)
    mixture_bias = torch.zeros(2 + 2 * action_dim, dtype=source_bias.dtype)
    mixture_weight[2 : 2 + action_dim] = source_weight + direction
    mixture_weight[2 + action_dim :] = source_weight - direction
    mixture_bias[2 : 2 + action_dim] = source_bias + bias_direction
    mixture_bias[2 + action_dim :] = source_bias - bias_direction

    actor["mlp.4.weight"] = mixture_weight
    actor["mlp.4.bias"] = mixture_bias
    actor["distribution.std_param"] = actor["distribution.std_param"].repeat(2, 1)
    optimizer = checkpoint.get("optimizer_state_dict")
    if optimizer is not None:
        optimizer["state"] = {}
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["two_mode_transplant"] = {
        "source": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "seed": seed,
        "separation": separation,
        "mode_average_max_weight_error": float(
            (
                (mixture_weight[2 : 2 + action_dim] + mixture_weight[2 + action_dim :])
                * 0.5
                - source_weight
            )
            .abs()
            .max()
        ),
        "mode_average_max_bias_error": float(
            (
                (mixture_bias[2 : 2 + action_dim] + mixture_bias[2 + action_dim :])
                * 0.5
                - source_bias
            )
            .abs()
            .max()
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"saved={output_path}")
    print(f"sha256={_sha256(output_path)}")
    print(infos["two_mode_transplant"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1243)
    parser.add_argument("--separation", type=float, default=0.02)
    args = parser.parse_args()
    transplant(args.source, args.output, seed=args.seed, separation=args.separation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
