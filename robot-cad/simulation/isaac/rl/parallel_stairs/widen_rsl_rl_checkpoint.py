"""Function-preservingly widen a two-hidden-layer RSL-RL PPO checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from torch import Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mlp(state: dict[str, Tensor], inputs: Tensor) -> Tensor:
    hidden = torch.nn.functional.elu(
        torch.nn.functional.linear(inputs, state["mlp.0.weight"], state["mlp.0.bias"])
    )
    hidden = torch.nn.functional.elu(
        torch.nn.functional.linear(hidden, state["mlp.2.weight"], state["mlp.2.bias"])
    )
    return torch.nn.functional.linear(hidden, state["mlp.4.weight"], state["mlp.4.bias"])


def widen_state_dict(state: dict[str, Tensor], target_width: int) -> dict[str, Tensor]:
    """Duplicate hidden units and divide downstream weights to preserve the MLP."""

    first_weight = state["mlp.0.weight"]
    first_bias = state["mlp.0.bias"]
    second_weight = state["mlp.2.weight"]
    second_bias = state["mlp.2.bias"]
    output_weight = state["mlp.4.weight"]
    old_width = first_weight.shape[0]
    expected_shapes = {
        "mlp.0.bias": (old_width,),
        "mlp.2.weight": (old_width, old_width),
        "mlp.2.bias": (old_width,),
        "mlp.4.weight": (output_weight.shape[0], old_width),
    }
    actual = {
        "mlp.0.bias": tuple(first_bias.shape),
        "mlp.2.weight": tuple(second_weight.shape),
        "mlp.2.bias": tuple(second_bias.shape),
        "mlp.4.weight": tuple(output_weight.shape),
    }
    if actual != expected_shapes:
        raise ValueError(f"unsupported MLP shapes: expected {expected_shapes}, got {actual}")
    if target_width != 2 * old_width:
        raise ValueError("target_width must be exactly twice the source width")

    repeats = target_width // old_width
    mapping = torch.arange(target_width, device=first_weight.device) % old_width
    duplicate_index = torch.arange(target_width, device=first_weight.device) // old_width
    split = torch.full(
        (repeats,), 1.0 / repeats, device=first_weight.device, dtype=first_weight.dtype
    )
    split += torch.linspace(
        -0.01, 0.01, repeats, device=first_weight.device, dtype=first_weight.dtype
    )
    split -= split.mean() - (1.0 / repeats)
    downstream_split = split[duplicate_index]
    widened = dict(state)
    widened["mlp.0.weight"] = first_weight[mapping].clone()
    widened["mlp.0.bias"] = first_bias[mapping].clone()
    widened["mlp.2.weight"] = second_weight[mapping][
        :, mapping
    ].clone() * downstream_split.unsqueeze(0)
    widened["mlp.2.bias"] = second_bias[mapping].clone()
    widened["mlp.4.weight"] = output_weight[:, mapping].clone() * downstream_split.unsqueeze(0)
    return widened


def widen_checkpoint(source: Path, destination: Path, target_width: int) -> dict[str, float]:
    """Widen actor and critic, verify outputs, and reset incompatible optimizer moments."""

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    critic = checkpoint["critic_state_dict"]
    source_width = actor["mlp.0.weight"].shape[0]
    widened_actor = widen_state_dict(actor, target_width)
    widened_critic = widen_state_dict(critic, target_width)

    generator = torch.Generator(device="cpu").manual_seed(512)
    actor_inputs = torch.randn(
        64, actor["mlp.0.weight"].shape[1], generator=generator, dtype=actor["mlp.0.weight"].dtype
    )
    critic_inputs = torch.randn(
        64,
        critic["mlp.0.weight"].shape[1],
        generator=generator,
        dtype=critic["mlp.0.weight"].dtype,
    )
    actor_error = float((_mlp(actor, actor_inputs) - _mlp(widened_actor, actor_inputs)).abs().max())
    critic_error = float(
        (_mlp(critic, critic_inputs) - _mlp(widened_critic, critic_inputs)).abs().max()
    )
    if max(actor_error, critic_error) > 1.0e-4:
        raise ValueError(
            f"widened checkpoint changed outputs: actor={actor_error}, critic={critic_error}"
        )

    checkpoint["actor_state_dict"] = widened_actor
    checkpoint["critic_state_dict"] = widened_critic
    optimizer = checkpoint.get("optimizer_state_dict")
    if optimizer is not None:
        optimizer["state"] = {}

    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["net2wider"] = {
        "source_sha256": _sha256(source),
        "source_width": source_width,
        "target_width": target_width,
        "actor_max_abs_error": actor_error,
        "critic_max_abs_error": critic_error,
        "function_preserving": True,
        "symmetry_breaking_downstream_split": [0.49, 0.51],
        "optimizer_state_reset": optimizer is not None,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return {"actor_max_abs_error": actor_error, "critic_max_abs_error": critic_error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-width", type=int, default=512)
    args = parser.parse_args()

    destination = args.output.resolve()
    errors = widen_checkpoint(args.input.resolve(), destination, args.target_width)
    print(
        f"wrote {destination} sha256={_sha256(destination)} "
        f"actor_error={errors['actor_max_abs_error']:.9g} "
        f"critic_error={errors['critic_max_abs_error']:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
