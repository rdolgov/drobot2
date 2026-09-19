"""ONNX policy loader with strict shape checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
    ACTION_NAMES,
    ACTION_SIZE,
    OBSERVATION_SIZE,
    GaitClockConfig,
    HeadingHoldConfig,
    JointTargetConfig,
    RunStopConfig,
)


ACTION_HISTORY_REQUESTED = "requested_action"
ACTION_HISTORY_APPLIED_TARGET = "applied_target"

OBSERVATION_ORDER_PREFIX = (
    "command_forward_m_s[1]",
    "command_lateral_m_s[1]",
    "command_yaw_rad_s[1]",
    "gait_clock_sin[1]",
    "gait_clock_cos[1]",
    "imu_angular_velocity_body_rad_s[3]",
    "projected_gravity_body[3]",
    "imu_linear_acceleration_body_over_9_81[3]",
    "joint_position_error_rad[12]",
    "joint_velocity_over_4_5836625[12]",
)
EXPECTED_OBSERVATION_ORDER = OBSERVATION_ORDER_PREFIX + (
    "previous_normalized_action[12]",
)
APPLIED_TARGET_OBSERVATION_ORDER = OBSERVATION_ORDER_PREFIX + (
    "previous_applied_normalized_target[12]",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_policy_metadata(
    payload: dict[str, Any],
    *,
    model_path: Path,
    metadata_path: Path,
) -> None:
    """Reject sidecars that would silently reorder or reinterpret the actor."""

    from .v44_study35 import validate_metadata, HIGH_SHA256, LOW_SHA256
    experimental = validate_metadata(payload)
    external = payload.get("onnx_external_data_sha256", {})
    if not isinstance(external, dict):
        raise ValueError("External ONNX data checksums must be an object")
    for name, digest in external.items():
        asset = model_path.parent / name
        if (Path(name).name != name or "/" in name or "\\" in name
            or asset.resolve().parent != model_path.parent.resolve()
            or not isinstance(digest, str) or _sha256(asset) != digest):
            raise ValueError("External ONNX data checksum/path mismatch")
    if experimental:
        if set(external) != {"policy.onnx.data"}:
            raise ValueError("Study35 requires bound external ONNX weights")
        for name, expected, key in (("high", HIGH_SHA256, "joint_position_rad"),
                                    ("low", LOW_SHA256, "low_speed_joint_position_rad")):
            asset = model_path.parent / (name + ".json")
            if asset.resolve().parent != model_path.parent.resolve() or _sha256(asset) != expected:
                raise ValueError("Study35 reference checksum mismatch")
            reference = json.loads(asset.read_text())
            if reference["joint_position_rad"] != payload["gait_reference"][key]:
                raise ValueError("Study35 embedded reference differs from bound asset")

    expected_hash = payload.get("onnx_sha256")
    if expected_hash is not None and model_path.is_file():
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"Invalid ONNX checksum in policy metadata: {metadata_path}")
        actual_hash = _sha256(model_path)
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(
                "Walking policy and JSON sidecar checksums do not match: "
                f"{model_path} / {metadata_path}"
            )

    observation_count = payload.get("observation_count")
    if observation_count is not None and observation_count != OBSERVATION_SIZE:
        raise ValueError(
            f"Policy metadata declares {observation_count!r} observations; "
            f"runtime requires {OBSERVATION_SIZE}"
        )
    action_count = payload.get("action_count")
    if action_count is not None and action_count != ACTION_SIZE:
        raise ValueError(
            f"Policy metadata declares {action_count!r} actions; "
            f"runtime requires {ACTION_SIZE}"
        )

    observation_scales = payload.get("observation_scales", {})
    if not isinstance(observation_scales, dict):
        raise ValueError("Policy metadata observation_scales must be an object")
    command_scales = observation_scales.get(
        "command_forward_lateral_yaw",
        (1.0, 1.0, 1.0),
    )
    if (
        not isinstance(command_scales, (list, tuple))
        or len(command_scales) != 3
    ):
        raise ValueError("Policy command observation scales must contain three values")
    command_scale_array = np.asarray(command_scales, dtype=np.float32)
    if not np.all(np.isfinite(command_scale_array)) or np.any(
        command_scale_array <= 0.0
    ):
        raise ValueError("Policy command observation scales must be finite and positive")

    action_contract = payload.get("action_contract", {})
    if not isinstance(action_contract, dict):
        raise ValueError("Policy metadata action_contract must be an object")
    action_mode = action_contract.get("mode", "direct")
    if action_mode == "integrated_target_velocity" and action_contract.get(
        "output_semantics"
    ) != "fraction_of_elapsed_time_target_velocity_limit":
        raise ValueError(
            "Integrated-target-velocity policy metadata must declare its "
            "elapsed-time velocity output semantics"
        )
    if action_mode == "integrated_target_velocity" and action_contract.get(
        "zero_command_target_behavior"
    ) != "slew_to_neutral":
        raise ValueError(
            "Integrated-target-velocity policy metadata must declare neutral "
            "homing at zero command"
        )

    action_history_mode = payload.get(
        "action_history_mode", ACTION_HISTORY_REQUESTED
    )
    if action_history_mode not in (
        ACTION_HISTORY_REQUESTED,
        ACTION_HISTORY_APPLIED_TARGET,
    ):
        raise ValueError(
            f"Unsupported policy action_history_mode: {action_history_mode!r}"
        )
    expected_observation_order = (
        APPLIED_TARGET_OBSERVATION_ORDER
        if action_history_mode == ACTION_HISTORY_APPLIED_TARGET
        else EXPECTED_OBSERVATION_ORDER
    )
    observation_order = payload.get("observation_order")
    if (
        observation_order is not None
        and tuple(observation_order) != expected_observation_order
    ):
        raise ValueError("Policy metadata observation order does not match the runtime")
    action_order = payload.get("action_order")
    if action_order is not None and tuple(action_order) != ACTION_NAMES:
        raise ValueError("Policy metadata action order does not match the servo runtime")


def load_policy_metadata(model_path: str | Path) -> dict[str, Any]:
    """Read the optional JSON sidecar without loading ONNX Runtime."""

    model = Path(model_path).expanduser().resolve()
    path = model.with_suffix(".json")
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Walking policy metadata must be a JSON object: {path}")
    _validate_policy_metadata(payload, model_path=model, metadata_path=path)
    return payload


class OnnxWalkingPolicy:
    def __init__(self, model_path: str | Path, *, allow_experimental_offline: bool = False) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for policy inference") from exc

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Walking policy not found: {path}")
        self.metadata = load_policy_metadata(path)
        from .v44_study35 import Study35Mapper
        self.experimental_mapper = None
        if self.metadata.get("experimental_target_mapping") is not None:
            if not allow_experimental_offline:
                raise ValueError("Study35 is offline-only; hardware/web loading is disabled")
            self.experimental_mapper = Study35Mapper(self.metadata)
        self.gait_clock_config = GaitClockConfig.from_metadata(self.metadata)
        self.heading_hold_config = HeadingHoldConfig.from_metadata(self.metadata)
        self.joint_target_config = JointTargetConfig.from_metadata(self.metadata)
        self.run_stop_config = RunStopConfig.from_metadata(self.metadata)
        self.action_history_mode = self.metadata.get(
            "action_history_mode", ACTION_HISTORY_REQUESTED
        )
        observation_scales = self.metadata.get("observation_scales", {})
        self.command_observation_scale = np.asarray(
            observation_scales.get(
                "command_forward_lateral_yaw",
                (1.0, 1.0, 1.0),
            ),
            dtype=np.float32,
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Walking policy must have exactly one input and one output")
        input_shape = inputs[0].shape
        output_shape = outputs[0].shape
        if not input_shape or input_shape[-1] != OBSERVATION_SIZE:
            raise ValueError(
                f"Walking policy input must end in {OBSERVATION_SIZE}, got {input_shape}"
            )
        if not output_shape or output_shape[-1] != ACTION_SIZE:
            raise ValueError(
                f"Walking policy output must end in {ACTION_SIZE}, got {output_shape}"
            )
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

    def infer(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"Expected observation shape (50,), got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("Observation contains a non-finite value")
        output = self._session.run(
            [self._output_name], {self._input_name: value.reshape(1, -1)}
        )[0]
        action = np.asarray(output[0], dtype=np.float32)
        if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"Policy returned invalid action shape/value: {action}")
        return np.clip(action, -1.0, 1.0)
