"""Guarded real-hardware runner for the Isaac-trained walking policy."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from drobot_policy_runtime.contract import (
    ACTION_NAMES,
    JOINT_LOWER_RAD,
    JOINT_UPPER_RAD,
)
from drobot_policy_runtime.policy import OnnxWalkingPolicy
from drobot_policy_runtime.runtime import MotorSink, PolicyCommand, WalkingPolicyLoop
from drobot_policy_runtime.sources import (
    Bno085ImuSource,
    ImuSample,
    ImuSource,
    JointStateSample,
    JointStateSource,
)

JointReader = Callable[[], JointStateSample]
TargetWriter = Callable[[np.ndarray, np.ndarray, float], None]
Finalizer = Callable[[str | None, bool], str | None]


class _GuardedImuSource(ImuSource):
    def __init__(self, source: ImuSource, update: Callable[[ImuSample], None]) -> None:
        self._source = source
        self._update = update

    @staticmethod
    def validate(sample: ImuSample, *, starting: bool) -> None:
        gravity = np.asarray(sample.projected_gravity_body, dtype=np.float32)
        norm = float(np.linalg.norm(gravity))
        if not 0.8 <= norm <= 1.2:
            raise RuntimeError("IMU projected-gravity magnitude is invalid")
        minimum_up = 0.75 if starting else 0.5
        if float(gravity[2]) > -minimum_up:
            limit = 41 if starting else 60
            raise RuntimeError(
                f"Body tilt exceeds the {limit}-degree RL safety limit"
            )

    def read(self) -> ImuSample:
        sample = self._source.read()
        self.validate(sample, starting=False)
        self._update(sample)
        return sample


class _CallbackJointSource(JointStateSource):
    def __init__(self, read: JointReader) -> None:
        self._read = read

    def read(self) -> JointStateSample:
        sample = self._read()
        if time.monotonic() - sample.monotonic_time_s > 0.12:
            raise RuntimeError("Joint feedback is stale")
        return sample


class _GuardedMotorSink(MotorSink):
    def __init__(
        self,
        write: TargetWriter,
        update: Callable[[np.ndarray, np.ndarray, float], None],
    ) -> None:
        self._write = write
        self._update = update
        self._last_write_s: float | None = None

    def write(
        self,
        action: np.ndarray,
        joint_target_rad: np.ndarray,
        monotonic_time_s: float,
    ) -> None:
        self._write(action, joint_target_rad, monotonic_time_s)
        completed_s = time.monotonic()
        if self._last_write_s is not None:
            gap_s = completed_s - self._last_write_s
            if gap_s > 0.12:
                raise RuntimeError(
                    "RL control loop missed its 120 ms deadline "
                    f"(actual output gap {gap_s * 1000.0:.0f} ms)"
                )
        self._update(action, joint_target_rad, completed_s)
        self._last_write_s = completed_s


class RlPolicyController:
    """Own policy/IMU state while the parent session remains motor-bus owner."""

    CONTROL_HZ = 60.0
    DEFAULT_DURATION_S = 5.0
    MIN_DURATION_S = 1.0
    MAX_DURATION_S = 60.0
    MAX_FORWARD_M_S = 0.10

    def __init__(
        self,
        model_path: Path,
        imu_axis_map: str,
        joint_reader: JointReader,
        target_writer: TargetWriter,
        finalizer: Finalizer,
    ) -> None:
        self.model_path = model_path.resolve()
        self.imu_axis_map = imu_axis_map
        self._joint_reader = joint_reader
        self._target_writer = target_writer
        self._finalizer = finalizer
        self._policy: OnnxWalkingPolicy | None = None
        self._imu: Bno085ImuSource | None = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._state: dict[str, Any] = {
            "available": self.model_path.is_file(),
            "active": False,
            "status": "ready" if self.model_path.is_file() else "unavailable",
            "error": None,
            "model": self.model_path.name,
            "control_hz": self.CONTROL_HZ,
            "duration_s": self.DEFAULT_DURATION_S,
            "forward_m_s": 0.03,
            "min_duration_s": self.MIN_DURATION_S,
            "max_duration_s": self.MAX_DURATION_S,
            "max_forward_m_s": self.MAX_FORWARD_M_S,
            "elapsed_s": 0.0,
            "imu": None,
            "targets": [],
            "motor_output_enabled": False,
        }

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._state["active"])

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
            if state["active"] and self._started_at is not None:
                state["elapsed_s"] = min(
                    float(state["duration_s"]),
                    max(0.0, time.monotonic() - self._started_at),
                )
            state["imu"] = None if state["imu"] is None else dict(state["imu"])
            state["targets"] = [dict(target) for target in state["targets"]]
            return state

    def prepare(self) -> ImuSample:
        with self._lock:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"RL model not found: {self.model_path}")
            self._state.update(status="initializing", error=None)
        try:
            if self._policy is None:
                self._policy = OnnxWalkingPolicy(self.model_path)
            if self._imu is None:
                self._imu = Bno085ImuSource(axis_map=self.imu_axis_map)
            sample = self._imu.read()
            _GuardedImuSource.validate(sample, starting=True)
            self._update_imu(sample)
        except Exception as exc:
            with self._lock:
                self._state.update(status="error", error=str(exc))
            raise
        with self._lock:
            self._state.update(status="ready", error=None)
        return sample

    @classmethod
    def validate_request(
        cls,
        forward_m_s: float,
        duration_s: float,
    ) -> tuple[float, float]:
        speed = float(forward_m_s)
        if not 0.0 <= speed <= cls.MAX_FORWARD_M_S:
            raise ValueError(
                f"RL forward speed must be in [0, {cls.MAX_FORWARD_M_S:.2f}] m/s"
            )
        duration = float(duration_s)
        if not cls.MIN_DURATION_S <= duration <= cls.MAX_DURATION_S:
            raise ValueError(
                "RL walk duration must be in "
                f"[{cls.MIN_DURATION_S:.0f}, {cls.MAX_DURATION_S:.0f}] seconds"
            )
        return speed, duration

    def start(
        self,
        forward_m_s: float,
        duration_s: float,
        initial_target_rad: np.ndarray,
    ) -> None:
        speed, duration = self.validate_request(forward_m_s, duration_s)
        initial = np.asarray(initial_target_rad, dtype=np.float32)
        if initial.shape != (12,) or not np.all(np.isfinite(initial)):
            raise ValueError("RL start requires 12 finite measured joint positions")
        margin_rad = math.radians(5.0)
        if np.any(initial < JOINT_LOWER_RAD - margin_rad) or np.any(
            initial > JOINT_UPPER_RAD + margin_rad
        ):
            raise RuntimeError("Measured pose is outside the RL joint envelope")
        if self._policy is None or self._imu is None:
            raise RuntimeError("Prepare the RL policy and IMU before starting")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("An RL walking test is already active")
            self._stop_event = threading.Event()
            self._started_at = time.monotonic()
            self._state.update(
                active=True,
                status="running",
                error=None,
                forward_m_s=speed,
                duration_s=duration,
                elapsed_s=0.0,
                targets=[],
                motor_output_enabled=True,
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(speed, duration, initial.copy()),
                name="drobot-rl-policy",
                daemon=True,
            )
            self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop(self, timeout_s: float = 2.0) -> None:
        self.request_stop()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)

    def _run(
        self,
        speed: float,
        duration_s: float,
        initial_target_rad: np.ndarray,
    ) -> None:
        error: str | None = None
        try:
            assert self._policy is not None
            assert self._imu is not None
            loop = WalkingPolicyLoop(
                self._policy,
                _GuardedImuSource(self._imu, self._update_imu),
                _CallbackJointSource(self._joint_reader),
                _GuardedMotorSink(self._target_writer, self._update_targets),
                command=PolicyCommand(forward_m_s=speed),
                control_hz=self.CONTROL_HZ,
                initial_target_rad=initial_target_rad,
            )
            loop.run(duration_s=duration_s, stop_event=self._stop_event)
        except Exception as exc:
            if not self._stop_event.is_set():
                error = str(exc)
        finally:
            stopped = self._stop_event.is_set()
            finalizer_error = self._finalizer(error, stopped)
            if error is None and finalizer_error:
                error = finalizer_error
            with self._lock:
                self._state.update(
                    active=False,
                    status="error" if error else "stopped" if stopped else "complete",
                    error=error,
                    motor_output_enabled=False,
                    elapsed_s=min(
                        duration_s,
                        max(
                            0.0,
                            time.monotonic() - (self._started_at or time.monotonic()),
                        ),
                    ),
                )

    def _update_imu(self, sample: ImuSample) -> None:
        with self._lock:
            self._state["imu"] = {
                "angular_velocity_rad_s": [
                    round(float(value), 5)
                    for value in sample.angular_velocity_body_rad_s
                ],
                "projected_gravity": [
                    round(float(value), 5) for value in sample.projected_gravity_body
                ],
                "linear_acceleration_m_s2": [
                    round(float(value), 5)
                    for value in sample.linear_acceleration_body_m_s2
                ],
            }

    def _update_targets(
        self,
        action: np.ndarray,
        joint_target_rad: np.ndarray,
        monotonic_time_s: float,
    ) -> None:
        targets = [
            {
                "joint": name,
                "target_deg": round(math.degrees(float(joint_target_rad[index])), 3),
                "normalized_action": round(float(action[index]), 5),
            }
            for index, name in enumerate(ACTION_NAMES)
        ]
        with self._lock:
            self._state["targets"] = targets
            self._state["last_policy_time_s"] = round(monotonic_time_s, 6)
