"""Create a reproducible PPO checkpoint with a lower Gaussian action standard deviation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def anneal_checkpoint(source: Path, destination: Path, target_std: float) -> None:
    """Lower exploration noise without changing any actor or critic MLP weights."""

    if not 0.01 <= target_std <= 1.0:
        raise ValueError("target_std must be between 0.01 and 1.0")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    std_key = "distribution.std_param"
    if std_key not in actor:
        raise KeyError(f"checkpoint has no {std_key!r}")

    original_std = actor[std_key].detach().clone()
    actor[std_key] = torch.full_like(original_std, target_std)

    optimizer = checkpoint.get("optimizer_state_dict")
    if optimizer is not None:
        first_parameter = optimizer["param_groups"][0]["params"][0]
        state = optimizer["state"].get(first_parameter, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state:
                if state[key].shape != original_std.shape:
                    raise ValueError(
                        "first optimizer parameter does not match action standard deviation"
                    )
                state[key].zero_()
        if "step" in state:
            state["step"].zero_()

    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["noise_anneal"] = {
        "source_sha256": _sha256(source),
        "original_std": original_std.tolist(),
        "target_std": target_std,
        "mlp_weights_changed": False,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-std", type=float, required=True)
    args = parser.parse_args()

    anneal_checkpoint(args.input.resolve(), args.output.resolve(), args.target_std)
    print(f"wrote {args.output.resolve()} sha256={_sha256(args.output.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
