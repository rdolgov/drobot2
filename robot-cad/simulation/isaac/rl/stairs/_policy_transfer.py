"""Controlled policy-only transfer from the flat 48-input PPO checkpoint."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

EXPANDABLE_INPUT_WEIGHTS = (
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.value_net.0.weight",
)


def physical_action_output_ratios(
    dof_names: Sequence[str],
    source_action_scale_by_kind: Mapping[str, object],
    target_action_scale_by_kind: Mapping[str, object],
) -> tuple[float, ...]:
    """Return normalized-action ratios that preserve physical joint targets."""

    if not dof_names:
        raise ValueError("dof_names cannot be empty")
    if set(source_action_scale_by_kind) != set(target_action_scale_by_kind):
        raise ValueError(
            "Source and target action-scale joint kinds must match exactly"
        )

    ratios: list[float] = []
    for name in dof_names:
        matches = [
            kind
            for kind in source_action_scale_by_kind
            if name.endswith(kind)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Could not resolve one action-scale joint kind for {name!r}: "
                f"{matches}"
            )
        kind = matches[0]
        source_scale = float(source_action_scale_by_kind[kind])
        target_scale = float(target_action_scale_by_kind[kind])
        if (
            not math.isfinite(source_scale)
            or not math.isfinite(target_scale)
            or source_scale <= 0.0
            or target_scale <= 0.0
        ):
            raise ValueError(
                f"Action scales for {kind} must be finite and positive"
            )
        ratios.append(source_scale / target_scale)
    return tuple(ratios)


def transfer_policy_state(
    source_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
    *,
    source_observation_size: int,
    shared_observation_prefix_size: int | None = None,
    action_output_ratios: Sequence[float] | None = None,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Copy compatible parameters and zero new terrain-input columns.

    Optimizer state is intentionally not transferred. The returned target state
    creates a new stair policy that initially behaves like the flat policy when
    its appended terrain inputs are ignored.
    """

    if source_observation_size <= 0:
        raise ValueError("source_observation_size must be positive")
    shared_prefix = (
        source_observation_size
        if shared_observation_prefix_size is None
        else int(shared_observation_prefix_size)
    )
    if shared_prefix <= 0 or shared_prefix > source_observation_size:
        raise ValueError(
            "shared_observation_prefix_size must be within the source input"
        )
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
            expanded[:, :shared_prefix] = source_tensor[:, :shared_prefix]
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
    rescaled_outputs: list[str] = []
    serialized_ratios: list[float] | None = None
    if action_output_ratios is not None:
        serialized_ratios = [float(value) for value in action_output_ratios]
        if (
            not serialized_ratios
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in serialized_ratios
            )
        ):
            raise ValueError(
                "action_output_ratios must contain finite positive values"
            )
        for name in ("action_net.weight", "action_net.bias"):
            tensor = transferred.get(name)
            if tensor is None:
                raise ValueError(
                    f"Target policy is missing required output tensor {name}"
                )
            if tensor.shape[0] != len(serialized_ratios):
                raise ValueError(
                    f"Action-output ratio count does not match {name}: "
                    f"{len(serialized_ratios)} != {tensor.shape[0]}"
                )
            ratios = tensor.new_tensor(serialized_ratios)
            if tensor.ndim == 2:
                ratios = ratios.reshape(-1, 1)
            elif tensor.ndim != 1:
                raise ValueError(
                    f"Unexpected output tensor rank for {name}: {tensor.ndim}"
                )
            transferred[name] = tensor * ratios
            rescaled_outputs.append(name)

    report = {
        "copied_exact_count": len(copied_exact),
        "copied_exact": copied_exact,
        "expanded_input_count": len(expanded_inputs),
        "expanded_inputs": expanded_inputs,
        "skipped": skipped,
        "optimizer_transferred": False,
        "new_input_columns_initialized_to_zero": True,
        "source_observation_size": source_observation_size,
        "shared_observation_prefix_size": shared_prefix,
        "dropped_source_input_columns": source_observation_size - shared_prefix,
        "physical_action_mean_preserved": action_output_ratios is not None,
        "action_output_ratios": serialized_ratios,
        "rescaled_action_outputs": rescaled_outputs,
    }
    return transferred, report
