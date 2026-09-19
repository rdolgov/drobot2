"""Verify deterministic walking-policy ONNX parity on bounded observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops._op import OpRunUnaryNum

from export_policy_onnx import DeterministicWalkingActor


class Softplus(OpRunUnaryNum):
    """Numerically stable override for ONNX's naive reference Softplus."""

    def _run(self, X):
        result = np.maximum(X, 0.0) + np.log1p(np.exp(-np.abs(X)))
        return (np.asarray(result, dtype=X.dtype),)


def realistic_observations(count: int, seed: int) -> np.ndarray:
    """Sample the deployed 50-value observation contract within sane bounds."""

    rng = np.random.default_rng(seed)
    observations = np.empty((count, 50), dtype=np.float32)
    # V31 is a forward-only policy.  Keep cross-axis commands near the heading
    # controller's correction range instead of constructing impossible reverse
    # and sideways commands that were never present during training.
    observations[:, 0] = rng.uniform(0.0, 0.20, count)
    observations[:, 1] = rng.uniform(-0.05, 0.05, count)
    observations[:, 2] = rng.uniform(-0.20, 0.20, count)
    phase = rng.uniform(-np.pi, np.pi, count)
    observations[:, 3] = np.sin(phase)
    observations[:, 4] = np.cos(phase)
    observations[:, 5:8] = rng.uniform(-1.0, 1.0, (count, 3))
    # Projected gravity is a unit vector, not three independent values. Sample
    # plausible +/-20 degree roll and pitch around the upright -Z direction.
    roll = rng.uniform(-0.35, 0.35, count)
    pitch = rng.uniform(-0.35, 0.35, count)
    observations[:, 8] = -np.sin(pitch)
    observations[:, 9] = np.sin(roll) * np.cos(pitch)
    observations[:, 10] = -np.cos(roll) * np.cos(pitch)
    observations[:, 11:14] = rng.uniform(-0.30, 0.30, (count, 3))
    observations[:, 14:26] = rng.uniform(-0.70, 0.70, (count, 12))
    observations[:, 26:38] = rng.uniform(-0.80, 0.80, (count, 12))
    observations[:, 38:50] = rng.uniform(-1.0, 1.0, (count, 12))
    return observations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1650)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")

    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    actor = DeterministicWalkingActor()
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()

    observations = realistic_observations(args.samples, args.seed)
    with torch.no_grad():
        pytorch_actions = actor(torch.from_numpy(observations)).numpy()
    onnx_actions = ReferenceEvaluator(
        onnx.load(args.onnx.resolve()), new_ops=[Softplus]
    ).run(None, {"observation": observations})[0]
    absolute_error = np.abs(pytorch_actions - onnx_actions)
    result = {
        "schema": "drobot.policy-onnx-parity.v1",
        "checkpoint": args.checkpoint.name,
        "onnx": args.onnx.name,
        "backend": "onnx.reference.ReferenceEvaluator+stable-softplus",
        "samples": args.samples,
        "seed": args.seed,
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_absolute_error": float(np.mean(absolute_error)),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
