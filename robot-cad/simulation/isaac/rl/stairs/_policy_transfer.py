"""Controlled policy-only transfer from the flat 48-input PPO checkpoint."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

EXPANDABLE_INPUT_WEIGHTS = (
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.value_net.0.weight",
)


def observation_prefix_compatibility(
    *,
    source_observation_fields: Sequence[str],
    target_observation_fields: Sequence[str],
    source_observation_size: int,
    target_observation_size: int,
) -> dict[str, object]:
    """Validate that an older policy can consume a target observation prefix."""

    source_fields = tuple(str(field) for field in source_observation_fields)
    target_fields = tuple(str(field) for field in target_observation_fields)
    source_size = int(source_observation_size)
    target_size = int(target_observation_size)
    if source_size <= 0 or target_size <= 0:
        raise ValueError("observation sizes must be positive")
    if len(source_fields) != source_size:
        raise ValueError("source observation fields do not match model size")
    if len(target_fields) != target_size:
        raise ValueError("target observation fields do not match environment size")
    if source_size > target_size:
        raise ValueError("source observation is larger than target observation")
    if source_fields != target_fields[:source_size]:
        raise ValueError("source observation fields are not a target prefix")
    return {
        "mode": (
            "exact" if source_size == target_size else "target_prefix_adapter"
        ),
        "source_observation_size": source_size,
        "target_observation_size": target_size,
        "appended_target_observation_count": target_size - source_size,
    }


def policy_observation_prefix_compatibility(
    policy: Any,
    source_manifest: Mapping[str, object],
    target_environment_contract: Mapping[str, object],
) -> dict[str, object]:
    """Validate a model manifest against an equal or expanded environment."""

    source_contract = dict(source_manifest.get("environment_contract", {}))
    if "observation_fields" not in source_contract:
        raise ValueError("source manifest has no observation field contract")
    if "observation_fields" not in target_environment_contract:
        raise ValueError("target environment has no observation field contract")
    source_shape = tuple(policy.observation_space.shape)
    if len(source_shape) != 1:
        raise ValueError(f"policy observation space must be flat: {source_shape}")
    target_fields = tuple(target_environment_contract["observation_fields"])
    return observation_prefix_compatibility(
        source_observation_fields=source_contract["observation_fields"],
        target_observation_fields=target_fields,
        source_observation_size=int(source_shape[0]),
        target_observation_size=len(target_fields),
    )


def predict_with_observation_prefix(
    policy: Any,
    observation: np.ndarray,
    *,
    deterministic: bool,
) -> tuple[np.ndarray, object]:
    """Predict with the prefix matching a legacy policy's observation space."""

    expected_shape = tuple(policy.observation_space.shape)
    if len(expected_shape) != 1:
        raise ValueError(f"policy observation space must be flat: {expected_shape}")
    expected_size = int(expected_shape[0])
    values = np.asarray(observation, dtype=np.float32)
    if values.ndim not in (1, 2) or values.shape[-1] < expected_size:
        raise ValueError(
            "observation cannot supply the policy prefix: "
            f"{values.shape} -> ({expected_size},)"
        )
    adapted = values[..., :expected_size]
    return policy.predict(adapted, deterministic=deterministic)


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
    action_output_target_indices: Sequence[int] | None = None,
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
    expanded_action_outputs: list[str] = []
    skipped: list[dict[str, object]] = []
    output_target_indices = (
        None
        if action_output_target_indices is None
        else tuple(int(index) for index in action_output_target_indices)
    )
    if output_target_indices is not None:
        if not output_target_indices:
            raise ValueError("action_output_target_indices cannot be empty")
        if any(index < 0 for index in output_target_indices):
            raise ValueError("action output target indices cannot be negative")
        if len(set(output_target_indices)) != len(output_target_indices):
            raise ValueError("action output target indices must be unique")
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
        if (
            output_target_indices is not None
            and name in ("action_net.weight", "action_net.bias", "log_std")
        ):
            if (
                source_tensor.ndim != target_tensor.ndim
                or source_tensor.shape[0] != len(output_target_indices)
                or target_tensor.shape[0] <= source_tensor.shape[0]
                or tuple(source_tensor.shape[1:]) != tuple(target_tensor.shape[1:])
                or max(output_target_indices) >= target_tensor.shape[0]
            ):
                raise ValueError(
                    f"Unexpected transferable action shape for {name}: "
                    f"{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
                )
            expanded = target_tensor.detach().clone()
            if name != "log_std":
                expanded.zero_()
            expanded[list(output_target_indices)] = source_tensor
            transferred[name] = expanded
            expanded_action_outputs.append(name)
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
        "expanded_action_outputs": expanded_action_outputs,
        "action_output_target_indices": (
            list(output_target_indices)
            if output_target_indices is not None
            else None
        ),
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


def freeze_policy_for_action_expansion(
    policy: Any,
    *,
    inherited_action_target_indices: Sequence[int],
) -> dict[str, object]:
    """Freeze an inherited actor and train only newly appended action rows.

    The value network remains trainable. The policy feature network and the
    inherited output rows receive no gradients, so a compact source skill is
    preserved while neutral action rows learn a strict action-space superset.
    """

    action_net = getattr(policy, "action_net", None)
    log_std = getattr(policy, "log_std", None)
    mlp_extractor = getattr(policy, "mlp_extractor", None)
    policy_net = getattr(mlp_extractor, "policy_net", None)
    if action_net is None or log_std is None or policy_net is None:
        raise ValueError("policy does not expose PPO actor expansion parameters")
    action_size = int(action_net.weight.shape[0])
    if (
        action_net.bias.shape != (action_size,)
        or log_std.shape != (action_size,)
    ):
        raise ValueError("policy action outputs do not share one flat shape")
    inherited = tuple(int(index) for index in inherited_action_target_indices)
    if (
        not inherited
        or any(index < 0 or index >= action_size for index in inherited)
        or len(set(inherited)) != len(inherited)
    ):
        raise ValueError("inherited action target indices are invalid")
    new_indices = tuple(
        index for index in range(action_size) if index not in set(inherited)
    )
    if not new_indices:
        raise ValueError("action expansion has no new trainable outputs")

    frozen_policy_parameters = 0
    for parameter in policy_net.parameters():
        parameter.requires_grad_(False)
        frozen_policy_parameters += int(parameter.numel())

    for parameter in (action_net.weight, action_net.bias, log_std):
        row_mask = parameter.new_zeros(parameter.shape)
        row_mask[list(new_indices)] = 1.0
        parameter.register_hook(
            lambda gradient, mask=row_mask: gradient * mask
        )

    return {
        "mode": "frozen_actor_train_new_action_rows",
        "action_size": action_size,
        "inherited_action_target_indices": list(inherited),
        "new_trainable_action_indices": list(new_indices),
        "frozen_policy_parameter_count": frozen_policy_parameters,
        "value_network_trainable": True,
    }
