"""Controlled policy-only transfer from the flat 48-input PPO checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EXPANDABLE_INPUT_WEIGHTS = (
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.value_net.0.weight",
)


def transfer_policy_state(
    source_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
    *,
    source_observation_size: int,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Copy compatible parameters and zero new terrain-input columns.

    Optimizer state is intentionally not transferred. The returned target state
    creates a new stair policy that initially behaves like the flat policy when
    its appended terrain inputs are ignored.
    """

    if source_observation_size <= 0:
        raise ValueError("source_observation_size must be positive")
    transferred = {
        name: tensor.detach().clone()
        for name, tensor in target_state.items()
    }
    copied_exact: list[str] = []
    expanded_inputs: list[str] = []
    skipped: list[dict[str, object]] = []
    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            skipped.append({"name": name, "reason": "missing_in_source"})
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            transferred[name] = source_tensor.detach().clone()
            copied_exact.append(name)
            continue
        if name in EXPANDABLE_INPUT_WEIGHTS:
            if (
                source_tensor.ndim != 2
                or target_tensor.ndim != 2
                or source_tensor.shape[0] != target_tensor.shape[0]
                or source_tensor.shape[1] != source_observation_size
                or target_tensor.shape[1] <= source_tensor.shape[1]
            ):
                raise ValueError(
                    f"Unexpected transferable input shape for {name}: "
                    f"{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
                )
            expanded = target_tensor.detach().clone()
            expanded.zero_()
            expanded[:, :source_observation_size] = source_tensor
            transferred[name] = expanded
            expanded_inputs.append(name)
            continue
        skipped.append(
            {
                "name": name,
                "reason": "shape_mismatch",
                "source_shape": list(source_tensor.shape),
                "target_shape": list(target_tensor.shape),
            }
        )
    report = {
        "copied_exact_count": len(copied_exact),
        "copied_exact": copied_exact,
        "expanded_input_count": len(expanded_inputs),
        "expanded_inputs": expanded_inputs,
        "skipped": skipped,
        "optimizer_transferred": False,
        "new_input_columns_initialized_to_zero": True,
    }
    return transferred, report
