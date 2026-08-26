"""ONNX policy loader with strict shape checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contract import ACTION_SIZE, OBSERVATION_SIZE, GaitClockConfig


def load_policy_metadata(model_path: str | Path) -> dict[str, Any]:
    """Read the optional JSON sidecar without loading ONNX Runtime."""

    path = Path(model_path).expanduser().resolve().with_suffix(".json")
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Walking policy metadata must be a JSON object: {path}")
    return payload


class OnnxWalkingPolicy:
    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for policy inference") from exc

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Walking policy not found: {path}")
        self.metadata = load_policy_metadata(path)
        self.gait_clock_config = GaitClockConfig.from_metadata(self.metadata)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

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
