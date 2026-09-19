"""Prepare a reproducible, conservative PPO fine-tuning checkpoint.

RSL-RL stores Adam state and its adaptive learning rate inside every checkpoint.
Loading a checkpoint therefore restores that rate even when the current runner
configuration specifies a smaller value.  This helper preserves the actor,
critic, and iteration while clearing stale optimizer moments and explicitly
setting the continuation rate.  It can also preserve the represented actor and
critic functions while changing the scale of the first three command inputs.
It records hashes and, when requested, a walking-workflow curriculum sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reset PPO optimizer state and optionally preserve the represented "
            "policy while rescaling command observations."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--curriculum-profile", default="")
    parser.add_argument("--curriculum-policy-steps", type=int, default=None)
    parser.add_argument(
        "--command-input-scales",
        type=float,
        nargs=3,
        metavar=("FORWARD", "LATERAL", "YAW"),
        default=None,
        help=(
            "New multiplicative scales for the first three policy observations; "
            "the checkpoint is transformed to preserve its original function."
        ),
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        parser.error(f"source checkpoint does not exist: {source}")
    if not 0.0 < args.learning_rate <= 1.0:
        parser.error("--learning-rate must be in (0, 1]")
    if (args.curriculum_policy_steps is None) != (not args.curriculum_profile):
        parser.error(
            "--curriculum-profile and --curriculum-policy-steps must be supplied together"
        )
    if args.curriculum_policy_steps is not None and args.curriculum_policy_steps < 0:
        parser.error("--curriculum-policy-steps must be non-negative")
    if args.command_input_scales is not None and any(
        not math.isfinite(value) or value <= 0.0
        for value in args.command_input_scales
    ):
        parser.error("--command-input-scales must contain three finite positive values")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")

    command_input_scales = tuple(args.command_input_scales or (1.0, 1.0, 1.0))
    if args.command_input_scales is not None:
        scales = torch.tensor(command_input_scales, dtype=torch.float32)
        actor_state = checkpoint["actor_state_dict"]
        actor_input_weight = actor_state.get("mlp.0.weight")
        if actor_input_weight is None or actor_input_weight.ndim != 2:
            raise ValueError("actor checkpoint has no two-dimensional mlp.0.weight")
        if actor_input_weight.shape[1] < 3:
            raise ValueError("actor input layer has fewer than three command columns")
        actor_input_weight[:, :3] /= scales.to(actor_input_weight)

        critic_state = checkpoint["critic_state_dict"]
        normalizer_mean = critic_state.get("obs_normalizer._mean")
        normalizer_var = critic_state.get("obs_normalizer._var")
        normalizer_std = critic_state.get("obs_normalizer._std")
        if all(
            value is not None
            for value in (normalizer_mean, normalizer_var, normalizer_std)
        ):
            normalizer_mean[..., :3] *= scales.to(normalizer_mean)
            normalizer_var[..., :3] *= torch.square(scales.to(normalizer_var))
            normalizer_std[..., :3] *= scales.to(normalizer_std)
        else:
            critic_input_weight = critic_state.get("mlp.0.weight")
            if critic_input_weight is None or critic_input_weight.ndim != 2:
                raise ValueError(
                    "critic has neither observation-normalizer state nor a valid input layer"
                )
            critic_input_weight[:, :3] /= scales.to(critic_input_weight)

    optimizer = checkpoint["optimizer_state_dict"]
    if not isinstance(optimizer, dict) or not optimizer.get("param_groups"):
        raise ValueError("checkpoint optimizer state has no parameter groups")
    original_learning_rates = [
        float(group["lr"]) for group in optimizer["param_groups"]
    ]
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = float(args.learning_rate)
        if "initial_lr" in group:
            group["initial_lr"] = float(args.learning_rate)

    source_sha256 = _sha256(source)
    created_utc = datetime.now(timezone.utc).isoformat()
    existing_infos = checkpoint.get("infos")
    infos = dict(existing_infos) if isinstance(existing_infos, dict) else {}
    infos["continuation_optimizer_reset"] = {
        "created_utc": created_utc,
        "source_checkpoint": source.name,
        "source_sha256": source_sha256,
        "original_learning_rates": original_learning_rates,
        "learning_rate": float(args.learning_rate),
        "optimizer_moments_cleared": True,
        "actor_and_critic_preserved": True,
        "command_input_scales": list(command_input_scales),
        "function_preserved_under_scaled_command_inputs": (
            args.command_input_scales is not None
        ),
    }
    checkpoint["infos"] = infos

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    output_sha256 = _sha256(output)
    provenance = {
        "schema": "drobot.ppo-continuation.v1",
        "created_utc": created_utc,
        "source": str(source),
        "source_sha256": source_sha256,
        "output": str(output),
        "output_sha256": output_sha256,
        "iteration": int(checkpoint["iter"]),
        "actor_and_critic_preserved": True,
        "command_input_scales": list(command_input_scales),
        "function_preserved_under_scaled_command_inputs": (
            args.command_input_scales is not None
        ),
        "optimizer_moments_cleared": True,
        "original_learning_rates": original_learning_rates,
        "learning_rate": float(args.learning_rate),
    }
    _write_json(Path(f"{output}.continuation.json"), provenance)

    if args.curriculum_policy_steps is not None:
        curriculum = {
            "profile": args.curriculum_profile,
            "curriculum_policy_steps": args.curriculum_policy_steps,
            "source_checkpoint": source.name,
            "source_iteration": int(checkpoint["iter"]),
            "model_iteration": int(checkpoint["iter"]),
            "recorded_at_utc": created_utc,
        }
        _write_json(Path(f"{output}.curriculum.json"), curriculum)

    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
