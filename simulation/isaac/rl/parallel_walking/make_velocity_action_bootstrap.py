"""Reuse learned observation features while resetting a changed action head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument(
        "--action-mode",
        choices=("integrated_target_velocity", "gait_residual"),
        default="integrated_target_velocity",
    )
    parser.add_argument(
        "--bootstrap-label",
        default="",
        help="Optional provenance label for the changed action semantics.",
    )
    parser.add_argument(
        "--initial-concentration",
        type=float,
        default=None,
        help=(
            "Optional symmetric Beta alpha/beta concentration above one. "
            "The deterministic mean remains zero while stochastic exploration "
            "becomes narrower than the legacy zero-logit bootstrap."
        ),
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        parser.error(f"source checkpoint does not exist: {source}")
    if not 0.0 < args.learning_rate <= 1.0:
        parser.error("--learning-rate must be in (0, 1]")
    if args.initial_concentration is not None and (
        not math.isfinite(args.initial_concentration)
        or args.initial_concentration <= 1.0
    ):
        parser.error("--initial-concentration must be finite and greater than one")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")

    actor = checkpoint["actor_state_dict"]
    critic = checkpoint["critic_state_dict"]
    expected = {
        "actor mlp.4.weight": (actor, "mlp.4.weight", (24, 256)),
        "actor mlp.4.bias": (actor, "mlp.4.bias", (24,)),
        "critic mlp.4.weight": (critic, "mlp.4.weight", (1, 256)),
        "critic mlp.4.bias": (critic, "mlp.4.bias", (1,)),
    }
    for label, (state, key, shape) in expected.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f"unexpected {label} shape; expected {shape}")

    # Zero raw alpha/beta logits give a deterministic Beta mean of exactly
    # zero while retaining broad stochastic exploration.  Hidden layers retain
    # the V33 IMU, phase, joint-state, and applied-target feature extraction.
    actor["mlp.4.weight"].zero_()
    actor_output_bias = 0.0
    if args.initial_concentration is not None:
        # RSL-RL maps each raw logit through softplus(logit) + 1.  Equal
        # alpha/beta parameters retain an exact zero deterministic action.
        actor_output_bias = math.log(math.expm1(args.initial_concentration - 1.0))
    actor["mlp.4.bias"].fill_(actor_output_bias)
    critic["mlp.4.weight"].zero_()
    critic["mlp.4.bias"].zero_()

    optimizer = checkpoint["optimizer_state_dict"]
    optimizer["state"] = {}
    for group in optimizer.get("param_groups", ()):
        group["lr"] = args.learning_rate

    checkpoint["iter"] = 0
    checkpoint["infos"] = {
        "bootstrap": (
            args.bootstrap_label
            or "hidden_layers_with_reset_action_and_value_heads"
        ),
        "source_checkpoint": source.name,
        "source_sha256": _sha256(source),
        "action_mode": args.action_mode,
        "action_history_mode": "applied_target",
        "initial_symmetric_beta_concentration": (
            args.initial_concentration
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": checkpoint["infos"]["source_sha256"],
                "output": str(output),
                "output_sha256": _sha256(output),
                "iteration": checkpoint["iter"],
                "learning_rate": args.learning_rate,
                "action_mode": args.action_mode,
                "initial_symmetric_beta_concentration": (
                    args.initial_concentration
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
