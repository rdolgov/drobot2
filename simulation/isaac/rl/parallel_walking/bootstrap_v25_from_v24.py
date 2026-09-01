"""Create a neutral-residual V25 transfer seed from the selected V24 policy.

This is checkpoint surgery only: it does not import Isaac Lab, construct an
environment, or run training.  The V24 actor feature layers are preserved, but
the bounded-Beta action head is made symmetric so its deterministic residual is
exactly zero for every observation.  The stale critic output and Adam moments
are reset before V25 nominal adaptation begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


EXPECTED_TOP_LEVEL_KEYS = frozenset(
    {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
    }
)
EXPECTED_ACTOR_SHAPES = {
    "mlp.0.weight": (256, 50),
    "mlp.0.bias": (256,),
    "mlp.2.weight": (256, 256),
    "mlp.2.bias": (256,),
    "mlp.4.weight": (24, 256),
    "mlp.4.bias": (24,),
}
EXPECTED_CRITIC_SHAPES = {
    "obs_normalizer._mean": (1, 58),
    "obs_normalizer._var": (1, 58),
    "obs_normalizer._std": (1, 58),
    "obs_normalizer.count": (),
    "mlp.0.weight": (256, 58),
    "mlp.0.bias": (256,),
    "mlp.2.weight": (256, 256),
    "mlp.2.bias": (256,),
    "mlp.4.weight": (1, 256),
    "mlp.4.bias": (1,),
}
EXPECTED_OPTIMIZER_KEYS = frozenset({"state", "param_groups"})
ACTION_COUNT = 12
EXPECTED_PARAMETER_COUNT = 12
EXPECTED_SOURCE_ITERATION = 3248
EXPECTED_DESTINATION_NAME = "model_3248.pt"
EXPECTED_SOURCE_SHA256 = (
    "e9c521fbd9f63ea0c9329bc3487a44be5f9dbc58530e9f7749eeba37635b37d2"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Unexpected {label} keys: missing={missing}, unexpected={unexpected}"
        )


def _require_tensor_shapes(
    value: Mapping[str, Any], expected: Mapping[str, tuple[int, ...]], label: str
) -> None:
    _require_exact_keys(value, frozenset(expected), label)
    for name, expected_shape in expected.items():
        tensor = value[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label}[{name!r}] is not a tensor")
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Unexpected {label}[{name!r}] shape: "
                f"expected={expected_shape}, actual={actual_shape}"
            )
        if name == "obs_normalizer.count":
            if tensor.dtype != torch.int64:
                raise TypeError(f"{label}[{name!r}] must have dtype torch.int64")
        elif tensor.dtype != torch.float32:
            raise TypeError(f"{label}[{name!r}] must have dtype torch.float32")


def _validate_optimizer(optimizer: Mapping[str, Any]) -> list[int]:
    _require_exact_keys(optimizer, EXPECTED_OPTIMIZER_KEYS, "optimizer state")
    state = optimizer["state"]
    groups = optimizer["param_groups"]
    if not isinstance(state, dict):
        raise TypeError("optimizer_state_dict['state'] must be a dictionary")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("Expected exactly one Adam optimizer parameter group")
    group = groups[0]
    if not isinstance(group, dict) or "params" not in group or "lr" not in group:
        raise ValueError("Optimizer parameter group is missing params or lr")
    parameter_ids = group["params"]
    if not isinstance(parameter_ids, list):
        raise TypeError("Optimizer parameter identifiers must be a list")
    if parameter_ids != list(range(EXPECTED_PARAMETER_COUNT)):
        raise ValueError(
            "Unexpected optimizer parameter identifiers: "
            f"expected={list(range(EXPECTED_PARAMETER_COUNT))}, "
            f"actual={parameter_ids}"
        )
    if set(state) != set(parameter_ids):
        raise ValueError(
            "Optimizer moments do not cover exactly the actor and critic parameters"
        )
    return parameter_ids


def _neutralize_checkpoint(
    checkpoint: dict[str, Any], learning_rate: float, source_sha256: str, source_name: str
) -> dict[str, Any]:
    _require_exact_keys(checkpoint, EXPECTED_TOP_LEVEL_KEYS, "checkpoint")
    if checkpoint["iter"] != EXPECTED_SOURCE_ITERATION:
        raise ValueError(
            "This bootstrap is defined only for selected V24 model_3248: "
            f"checkpoint iter={checkpoint['iter']!r}"
        )
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")

    actor = checkpoint["actor_state_dict"]
    critic = checkpoint["critic_state_dict"]
    optimizer = checkpoint["optimizer_state_dict"]
    if not isinstance(actor, dict) or not isinstance(critic, dict):
        raise TypeError("Actor and critic state dictionaries must be dictionaries")
    if not isinstance(optimizer, dict):
        raise TypeError("Optimizer state must be a dictionary")
    _require_tensor_shapes(actor, EXPECTED_ACTOR_SHAPES, "actor state")
    _require_tensor_shapes(critic, EXPECTED_CRITIC_SHAPES, "critic state")
    parameter_ids = _validate_optimizer(optimizer)

    preserved_actor = {
        name: actor[name].detach().clone()
        for name in ("mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias")
    }
    preserved_critic = {
        name: tensor.detach().clone()
        for name, tensor in critic.items()
        if name not in {"mlp.4.weight", "mlp.4.bias"}
    }

    output_weight = actor["mlp.4.weight"].detach()
    output_bias = actor["mlp.4.bias"].detach()
    common_weight = 0.5 * (
        output_weight[:ACTION_COUNT] + output_weight[ACTION_COUNT:]
    )
    common_bias = 0.5 * (
        output_bias[:ACTION_COUNT] + output_bias[ACTION_COUNT:]
    )
    actor["mlp.4.weight"] = torch.cat(
        (common_weight, common_weight), dim=0
    ).clone()
    actor["mlp.4.bias"] = torch.cat((common_bias, common_bias), dim=0).clone()

    critic["mlp.4.weight"].zero_()
    critic["mlp.4.bias"].zero_()
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = float(learning_rate)

    for name, before in preserved_actor.items():
        if not torch.equal(actor[name], before):
            raise RuntimeError(f"Actor feature tensor changed unexpectedly: {name}")
    for name, before in preserved_critic.items():
        if not torch.equal(critic[name], before):
            raise RuntimeError(f"Critic feature tensor changed unexpectedly: {name}")
    if not torch.equal(
        actor["mlp.4.weight"][:ACTION_COUNT],
        actor["mlp.4.weight"][ACTION_COUNT:],
    ) or not torch.equal(
        actor["mlp.4.bias"][:ACTION_COUNT],
        actor["mlp.4.bias"][ACTION_COUNT:],
    ):
        raise RuntimeError("Neutral Beta alpha and beta output rows are not identical")
    if torch.count_nonzero(critic["mlp.4.weight"]).item() != 0 or torch.count_nonzero(
        critic["mlp.4.bias"]
    ).item() != 0:
        raise RuntimeError("Critic output head reset was incomplete")
    if optimizer["state"]:
        raise RuntimeError("Optimizer moment reset was incomplete")

    checkpoint["iter"] = EXPECTED_SOURCE_ITERATION
    checkpoint["infos"] = {
        "bootstrap": "v24_features_symmetric_beta_head",
        "source_checkpoint": source_name,
        "source_checkpoint_sha256": source_sha256,
        "source_iteration": EXPECTED_SOURCE_ITERATION,
        "initial_deterministic_residual": 0.0,
        "critic_output_head_reset": True,
        "optimizer_moments_reset": True,
        "optimizer_parameter_ids": parameter_ids,
        "learning_rate": float(learning_rate),
    }
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create model_3248.pt for V25 by preserving V24 features, "
            "neutralizing its Beta action mean, and resetting stale training state."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=7.5e-5)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing destination model and bootstrap sidecar.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    sidecar = Path(f"{destination}.bootstrap.json")
    if not source.is_file():
        parser.error(f"source checkpoint does not exist: {source}")
    if destination.name != EXPECTED_DESTINATION_NAME:
        parser.error(
            "destination filename must remain model_3248.pt so the existing V25 "
            "curriculum-age fallback retains its intended semantics"
        )
    if source == destination:
        parser.error("destination must not overwrite the selected V24 source checkpoint")
    if not args.force and (destination.exists() or sidecar.exists()):
        parser.error(
            "destination model or bootstrap sidecar already exists; pass --force "
            "only when replacing that exact generated seed is intentional"
        )

    source_sha256 = _sha256(source)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        parser.error(
            "source SHA-256 does not match the selected V24 model_3248 release: "
            f"expected={EXPECTED_SOURCE_SHA256}, actual={source_sha256}"
        )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint root must be a dictionary")
    checkpoint = _neutralize_checkpoint(
        checkpoint,
        learning_rate=args.learning_rate,
        source_sha256=source_sha256,
        source_name=source.name,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    temporary_model = destination.with_name(f".{destination.name}.{nonce}.tmp")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{nonce}.tmp")
    try:
        torch.save(checkpoint, temporary_model)
        destination_sha256 = _sha256(temporary_model)
        provenance = {
            "format": "drobot-v25-neutral-beta-bootstrap-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "filename": source.name,
                "sha256": source_sha256,
                "iteration": EXPECTED_SOURCE_ITERATION,
            },
            "destination": {
                "filename": destination.name,
                "sha256": destination_sha256,
                "iteration": EXPECTED_SOURCE_ITERATION,
            },
            "architecture": {
                "actor_observation_count": 50,
                "critic_observation_count": 58,
                "action_count": ACTION_COUNT,
                "actor_hidden_dimensions": [256, 256],
                "bounded_beta_output_layout": [2, ACTION_COUNT],
            },
            "transformation": {
                "actor_feature_layers": "preserved_exactly",
                "beta_alpha_rows": "pairwise_mean_of_source_alpha_and_beta_rows",
                "beta_beta_rows": "identical_to_transformed_alpha_rows",
                "initial_deterministic_residual": 0.0,
                "state_dependent_concentration": "preserved_from_common_output_mode",
                "critic_normalizer_and_feature_layers": "preserved_exactly",
                "critic_output_head": "zeroed",
                "optimizer_moments": "cleared",
                "optimizer_learning_rate": float(args.learning_rate),
            },
            "workflow_semantics": {
                "checkpoint_filename": EXPECTED_DESTINATION_NAME,
                "checkpoint_iteration": EXPECTED_SOURCE_ITERATION,
                "v25_curriculum_policy_steps_at_transfer": 0,
                "note": (
                    "Pass this checkpoint explicitly to the V25 nominal phase. "
                    "Keeping iteration 3248 preserves the existing interrupted-run "
                    "curriculum-age fallback; ResetCurriculumOffset starts V25 at zero."
                ),
            },
        }
        temporary_sidecar.write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        temporary_model.replace(destination)
        temporary_sidecar.replace(sidecar)
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)

    print(f"Wrote V25 neutral-residual bootstrap: {destination}")
    print(f"Wrote SHA-256 provenance sidecar: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
