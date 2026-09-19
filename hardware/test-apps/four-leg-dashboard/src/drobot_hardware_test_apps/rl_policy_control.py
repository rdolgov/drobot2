"""Guarded real-hardware runner for the Isaac-trained walking policy."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from drobot_policy_runtime.contract import (
    ACTION_NAMES,
    JOINT_LOWER_RAD,
    JOINT_UPPER_RAD,
    GaitClockConfig,
    HeadingHoldConfig,
    JointTargetConfig,
    RunStopConfig,
)
from drobot_policy_runtime.policy import OnnxWalkingPolicy, load_policy_metadata
from drobot_policy_runtime.recording import TrialRecorder, sha256_file
from drobot_policy_runtime.runtime import (
    AppliedMotorTarget,
    MotorSink,
    PolicyCommand,
    PolicyStepSample,
    WalkingPolicyLoop,
)
from drobot_policy_runtime.sources import (
    BackgroundImuSource,
    Bno085ImuSource,
    ImuSample,
    ImuSource,
    JointStateSample,
    JointStateSource,
)

JointReader = Callable[[], JointStateSample]
TargetWriter = Callable[[np.ndarray, np.ndarray, float], None]
AppliedTargetReader = Callable[[], AppliedMotorTarget | None]
Finalizer = Callable[[str | None, bool], str | None]
_LOGGER = logging.getLogger(__name__)


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


class _BackgroundJointSource(JointStateSource):
    """Poll the shared servo bus independently and serve cached policy inputs."""

    def __init__(
        self,
        read: JointReader,
        initial: JointStateSample,
        *,
        poll_hz: float,
        stale_after_s: float = 0.12,
    ) -> None:
        self._read = read
        self._period_s = 1.0 / poll_hz
        self._stale_after_s = stale_after_s
        self._lock = threading.Lock()
        self._sample = initial
        self._joint_shape = np.asarray(initial.position_rad).shape
        self._last_error: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._poll,
            name="drobot-rl-joint-feedback",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if (
            self._thread is not None
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(timeout=0.5)

    def _poll(self) -> None:
        deadline_s = time.monotonic()
        while not self._stop_event.is_set():
            try:
                sample = self._read()
                position = np.asarray(sample.position_rad, dtype=np.float32)
                velocity = np.asarray(sample.velocity_rad_s, dtype=np.float32)
                if (
                    position.shape != self._joint_shape
                    or velocity.shape != self._joint_shape
                    or not np.all(np.isfinite(position))
                    or not np.all(np.isfinite(velocity))
                ):
                    raise RuntimeError("Joint feedback contains invalid values")
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
            else:
                with self._lock:
                    self._sample = JointStateSample(
                        position_rad=position.copy(),
                        velocity_rad_s=velocity.copy(),
                        monotonic_time_s=float(sample.monotonic_time_s),
                    )
                    self._last_error = None
            deadline_s += self._period_s
            completed_s = time.monotonic()
            if deadline_s <= completed_s:
                deadline_s = completed_s + self._period_s
            self._stop_event.wait(max(0.0, deadline_s - time.monotonic()))

    def read(self) -> JointStateSample:
        with self._lock:
            sample = self._sample
            last_error = self._last_error
        age_s = time.monotonic() - sample.monotonic_time_s
        if age_s > self._stale_after_s:
            detail = f"; latest read error: {last_error}" if last_error else ""
            raise RuntimeError(
                f"Joint feedback is stale ({age_s * 1000.0:.0f} ms){detail}"
            )
        return JointStateSample(
            position_rad=sample.position_rad.copy(),
            velocity_rad_s=sample.velocity_rad_s.copy(),
            monotonic_time_s=sample.monotonic_time_s,
        )


class _GuardedMotorSink(MotorSink):
    def __init__(
        self,
        write: TargetWriter,
        update: Callable[[np.ndarray, np.ndarray, float], None],
        applied_target_reader: AppliedTargetReader | None = None,
    ) -> None:
        self._write = write
        self._update = update
        self._last_write_s: float | None = None
        if applied_target_reader is not None:
            # Only advertise acknowledgements when an actual bus snapshot exists.
            self.read_applied_target = applied_target_reader

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
    JOINT_FEEDBACK_HZ = 100.0
    IMU_ACQUISITION_HZ = 100.0
    IMU_STALE_AFTER_S = 0.12
    IMU_STARTUP_TIMEOUT_S = 30.0
    DEFAULT_DURATION_S = 5.0
    MIN_DURATION_S = 1.0
    MAX_DURATION_S = 60.0
    # Absolute UI/runtime ceiling only. Each model sidecar narrows this to the
    # speed band actually seen during training (V43 declares 0.10--0.14 m/s).
    MAX_FORWARD_M_S = 0.20
    DEFAULT_NORMAL_STOP_HOLD_S = 1.0
    MAX_NORMAL_STOP_RAMP_S = 2.0
    MAX_NORMAL_STOP_HOLD_S = 3.0

    def __init__(
        self,
        model_path: Path,
        imu_axis_map: str,
        joint_reader: JointReader,
        target_writer: TargetWriter,
        finalizer: Finalizer,
        recorder: TrialRecorder | None = None,
        recording_metadata: Mapping[str, Any] | None = None,
        applied_target_reader: AppliedTargetReader | None = None,
    ) -> None:
        self.model_path = model_path.resolve()
        self.model_sha256 = sha256_file(self.model_path)
        self.model_contract_path = self.model_path.with_suffix(".json")
        self.model_contract_sha256 = sha256_file(self.model_contract_path)
        self.ankle_policy = self.model_path.suffix == ".npz"
        self.action_names = ACTION_NAMES
        self.joint_lower, self.joint_upper = JOINT_LOWER_RAD, JOINT_UPPER_RAD
        self.policy_class, self.loop_class = OnnxWalkingPolicy, WalkingPolicyLoop
        if self.ankle_policy:
            from drobot_policy_runtime import ankle_transfer

            self.model_metadata = ankle_transfer.load_metadata(self.model_path)
            self.action_names = ankle_transfer.ACTION_NAMES
            self.joint_lower, self.joint_upper = ankle_transfer.LOWER, ankle_transfer.UPPER
            self.policy_class = ankle_transfer.AnkleTransferPolicy
            self.loop_class = ankle_transfer.AnkleTransferLoop
            self.MAX_DURATION_S = 15.0
        else:
            self.model_metadata = load_policy_metadata(self.model_path)
        self.gait_clock_config = GaitClockConfig.from_metadata(self.model_metadata)
        self.heading_hold_config = HeadingHoldConfig.from_metadata(self.model_metadata)
        self.joint_target_config = (
            ankle_transfer.AnkleTargetConfig() if self.ankle_policy
            else JointTargetConfig.from_metadata(self.model_metadata)
        )
        self.run_stop_config = RunStopConfig.from_metadata(self.model_metadata)
        self.normal_stop_ramp_s = self.run_stop_config.recommended_ramp_s
        self.normal_stop_hold_s = self.run_stop_config.recommended_hold_s
        if not (
            math.isfinite(self.normal_stop_ramp_s)
            and 0.0 <= self.normal_stop_ramp_s <= self.MAX_NORMAL_STOP_RAMP_S
        ):
            raise ValueError("Policy normal stop ramp is invalid")
        if not (
            math.isfinite(self.normal_stop_hold_s)
            and 0.0 <= self.normal_stop_hold_s <= self.MAX_NORMAL_STOP_HOLD_S
        ):
            raise ValueError("Policy stopped hold is invalid")
        command_range = self.model_metadata.get("forward_command_range_m_s", {})
        if not isinstance(command_range, Mapping):
            command_range = {}
        self.min_forward_m_s = float(command_range.get("min", 0.0))
        declared_max_forward_m_s = float(
            command_range.get("max", self.MAX_FORWARD_M_S)
        )
        self.max_forward_m_s = min(
            self.MAX_FORWARD_M_S,
            declared_max_forward_m_s,
        )
        if not (
            math.isfinite(self.min_forward_m_s)
            and math.isfinite(declared_max_forward_m_s)
            and math.isfinite(self.max_forward_m_s)
            and 0.0 <= self.min_forward_m_s <= self.max_forward_m_s
        ):
            raise ValueError("Policy forward command range is invalid")
        self.recommended_forward_m_s = float(
            command_range.get(
                "recommended",
                max(self.min_forward_m_s, min(0.03, self.max_forward_m_s)),
            )
        )
        if not (
            math.isfinite(self.recommended_forward_m_s)
            and self.min_forward_m_s
            <= self.recommended_forward_m_s
            <= self.max_forward_m_s
        ):
            raise ValueError("Policy recommended forward command is invalid")
        self.imu_axis_map = imu_axis_map
        self._joint_reader = joint_reader
        self._target_writer = target_writer
        self._applied_target_reader = applied_target_reader
        self._finalizer = finalizer
        self._recorder = recorder
        self._recording_metadata = dict(recording_metadata or {})
        self._recording_start_error: str | None = None
        self._policy: OnnxWalkingPolicy | None = None
        self._imu: BackgroundImuSource | None = None
        self._retiring_imu: BackgroundImuSource | None = None
        self._lock = threading.RLock()
        self._prepare_lock = threading.Lock()
        self._prepare_cancel = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._state: dict[str, Any] = {
            "action_size": len(self.action_names),
            "experimental_notice": self.model_metadata.get("experimental_notice"),
            "available": self.model_path.is_file(),
            "active": False,
            "status": "ready" if self.model_path.is_file() else "unavailable",
            "error": None,
            "model": self.model_path.name,
            "model_variant": self.model_path.parent.name,
            "control_hz": self.CONTROL_HZ,
            "duration_s": self.DEFAULT_DURATION_S,
            "forward_m_s": self.recommended_forward_m_s,
            "min_duration_s": self.MIN_DURATION_S,
            "max_duration_s": self.MAX_DURATION_S,
            "min_forward_m_s": self.min_forward_m_s,
            "max_forward_m_s": self.max_forward_m_s,
            "recommended_forward_m_s": self.recommended_forward_m_s,
            "normal_stop_ramp_s": self.normal_stop_ramp_s,
            "normal_stop_hold_s": self.normal_stop_hold_s,
            "normal_stop_ramp_shape": self.run_stop_config.ramp_shape,
            "normal_stop_phase_alignment": {
                "enabled": self.run_stop_config.phase_alignment_enabled,
                "ramp_start_phase_cycles": (
                    self.run_stop_config.ramp_start_phase_cycles
                ),
                "phase_tolerance_cycles": (
                    self.run_stop_config.phase_tolerance_cycles
                ),
                "maximum_wait_s": (
                    self.run_stop_config.maximum_phase_alignment_wait_s
                ),
            },
            "effective_forward_m_s": 0.0,
            "gait_clock_mode": self.gait_clock_config.mode,
            "gait_frequency_hz": self.gait_clock_config.frequency_hz(
                self.recommended_forward_m_s
            ),
            "joint_feedback_target_hz": self.JOINT_FEEDBACK_HZ,
            "neutral_joint_position_deg": [
                round(math.degrees(value), 3)
                for value in self.joint_target_config.neutral_joint_position_rad
            ],
            "startup_ramp_rate_deg_s": (
                self.joint_target_config.startup_ramp_rate_deg_s
            ),
            "startup_settle_s": self.joint_target_config.startup_settle_s,
            "startup_position_tolerance_deg": (
                self.joint_target_config.startup_position_tolerance_deg
            ),
            "elapsed_s": 0.0,
            "imu": None,
            "heading_hold": {
                "enabled": self.heading_hold_config.enabled,
                "mode": self.heading_hold_config.mode,
                "kp_s": self.heading_hold_config.kp_s,
                "max_correction_rad_s": (
                    self.heading_hold_config.max_correction_rad_s
                ),
                "reference_yaw_rad": None,
                "current_relative_rad": None,
                "desired_relative_rad": None,
                "error_rad": None,
                "correction_rad_s": 0.0,
                "effective_yaw_rad_s": 0.0,
            },
            "targets": [],
            "motor_output_enabled": False,
            "recording": (
                recorder.status()
                if recorder is not None
                else {"active": False, "available": False}
            ),
        }

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._state["active"])

    def take_imu_from(self, other: RlPolicyController) -> None:
        """Retire idle sensor sessions before switching the selected model."""

        if self.active or other.active:
            raise RuntimeError("Cannot transfer the IMU while an RL policy is active")
        if self is other:
            return
        if not other._prepare_lock.acquire(blocking=False):
            raise RuntimeError("Wait for sensor initialization before switching models")
        try:
            if not self._prepare_lock.acquire(blocking=False):
                raise RuntimeError("Wait for sensor initialization before switching models")
            try:
                other._release_imu()
                if other._retiring_imu is not None and other._retiring_imu.cleanup_pending:
                    raise RuntimeError("Wait for IMU device cleanup before switching models")
                self._release_imu()
            finally:
                self._prepare_lock.release()
        finally:
            other._prepare_lock.release()

    def _release_imu(self) -> None:
        with self._lock:
            source, self._imu = self._imu, None
            self._state["imu"] = None
        if source is not None:
            try:
                source.close()
            except Exception:
                _LOGGER.warning("IMU cleanup failed", exc_info=True)
            finally:
                with self._lock:
                    self._retiring_imu = source if source.cleanup_pending else None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
            state["imu_cleanup_pending"] = bool(
                self._retiring_imu is not None and self._retiring_imu.cleanup_pending
            )
            if state["active"] and self._started_at is not None:
                state["elapsed_s"] = min(
                    float(state["duration_s"])
                    + self.normal_stop_ramp_s
                    + self.normal_stop_hold_s
                    + self.run_stop_config.maximum_phase_alignment_wait_s,
                    max(0.0, time.monotonic() - self._started_at),
                )
            state["imu"] = None if state["imu"] is None else dict(state["imu"])
            state["targets"] = [dict(target) for target in state["targets"]]
            if self._recorder is not None:
                recording = self._recorder.status()
                if recording.get("error") is None and self._recording_start_error:
                    recording["error"] = self._recording_start_error
                state["recording"] = recording
            return state

    def prepare(self) -> ImuSample:
        # HTTP requests run in separate threads. A second Start must not reset
        # the sensor while the first request is enabling its reports.
        if not self._prepare_lock.acquire(blocking=False):
            raise RuntimeError("RL sensor initialization is already in progress; wait for it to finish")
        try:
            self._prepare_cancel.clear()
            return self._prepare_once()
        finally:
            self._prepare_lock.release()

    def _prepare_once(self) -> ImuSample:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("An RL walking test is already active")
            if self._retiring_imu is not None:
                if self._retiring_imu.cleanup_pending:
                    raise RuntimeError("Wait for the previous IMU device call and cleanup to finish")
                self._retiring_imu = None
            if not self.model_path.is_file():
                raise FileNotFoundError(f"RL model not found: {self.model_path}")
            self._state.update(status="initializing", error=None)
        try:
            if self._policy is None:
                self._policy = self.policy_class(self.model_path)
            if self._imu is None:
                self._imu = BackgroundImuSource(
                    lambda: Bno085ImuSource(axis_map=self.imu_axis_map),
                    poll_hz=self.IMU_ACQUISITION_HZ,
                    stale_after_s=self.IMU_STALE_AFTER_S,
                    startup_timeout_s=self.IMU_STARTUP_TIMEOUT_S,
                )
            sample = self._imu.start(cancel_event=self._prepare_cancel)
            if self._prepare_cancel.is_set():
                raise RuntimeError("RL sensor initialization was cancelled")
            _GuardedImuSource.validate(sample, starting=True)
            self._update_imu(sample)
        except Exception as exc:
            self._release_imu()
            with self._lock:
                self._state.update(status="error", error=str(exc))
            raise
        with self._lock:
            self._state.update(status="ready", error=None)
        return sample

    def validate_request(
        self,
        forward_m_s: float,
        duration_s: float,
    ) -> tuple[float, float]:
        speed = float(forward_m_s)
        if not self.min_forward_m_s <= speed <= self.max_forward_m_s:
            raise ValueError(
                "RL forward speed must be in "
                f"[{self.min_forward_m_s:.3f}, {self.max_forward_m_s:.3f}] m/s "
                "for the selected policy"
            )
        duration = float(duration_s)
        if not self.MIN_DURATION_S <= duration <= self.MAX_DURATION_S:
            raise ValueError(
                "RL walk duration must be in "
                f"[{self.MIN_DURATION_S:.0f}, {self.MAX_DURATION_S:.0f}] seconds"
            )
        return speed, duration

    def start(
        self,
        forward_m_s: float,
        duration_s: float,
        initial_target_rad: np.ndarray,
    ) -> None:
        if not self._prepare_lock.acquire(blocking=False):
            raise RuntimeError("Wait for RL sensor initialization to finish")
        try:
            if self._prepare_cancel.is_set():
                self._release_imu()
                raise RuntimeError("RL start was cancelled; prepare again")
            self._start_prepared(forward_m_s, duration_s, initial_target_rad)
        finally:
            self._prepare_lock.release()

    def _start_prepared(
        self,
        forward_m_s: float,
        duration_s: float,
        initial_target_rad: np.ndarray,
    ) -> None:
        speed, duration = self.validate_request(forward_m_s, duration_s)
        initial = np.asarray(initial_target_rad, dtype=np.float32)
        if initial.shape != (len(self.action_names),) or not np.all(np.isfinite(initial)):
            raise ValueError(f"RL start requires {len(self.action_names)} finite measured joint positions")
        margin_rad = math.radians(5.0)
        if np.any(initial < self.joint_lower - margin_rad) or np.any(
            initial > self.joint_upper + margin_rad
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
                effective_forward_m_s=speed,
                duration_s=duration,
                elapsed_s=0.0,
                targets=[],
                motor_output_enabled=True,
                gait_frequency_hz=self.gait_clock_config.frequency_hz(speed),
                heading_hold={
                    "enabled": self.heading_hold_config.enabled,
                    "mode": self.heading_hold_config.mode,
                    "kp_s": self.heading_hold_config.kp_s,
                    "max_correction_rad_s": (
                        self.heading_hold_config.max_correction_rad_s
                    ),
                    "reference_yaw_rad": None,
                    "current_relative_rad": None,
                    "desired_relative_rad": None,
                    "error_rad": None,
                    "correction_rad_s": 0.0,
                    "effective_yaw_rad_s": 0.0,
                },
            )
            if self._recorder is not None:
                try:
                    recording_id = self._recorder.start_trial(
                        {
                            **self._recording_metadata,
                            "trial": {
                                "forward_m_s": speed,
                                "lateral_m_s": 0.0,
                                "yaw_rad_s": 0.0,
                                "duration_s": duration,
                                "normal_stop_ramp_s": self.normal_stop_ramp_s,
                                "normal_stop_hold_s": self.normal_stop_hold_s,
                                "control_hz": self.CONTROL_HZ,
                                "joint_feedback_target_hz": self.JOINT_FEEDBACK_HZ,
                                "imu_acquisition": {
                                    "mode": "background_owner_thread",
                                    "target_hz": self.IMU_ACQUISITION_HZ,
                                    "stale_after_s": self.IMU_STALE_AFTER_S,
                                    "timestamp_semantics": "complete_host_acquisition",
                                },
                                "gait_clock": {
                                    "mode": self.gait_clock_config.mode,
                                    "frequency_hz": (
                                        self.gait_clock_config.frequency_hz(speed)
                                    ),
                                },
                                "heading_hold": {
                                    "enabled": self.heading_hold_config.enabled,
                                    "mode": self.heading_hold_config.mode,
                                    "kp_s": self.heading_hold_config.kp_s,
                                    "max_correction_rad_s": (
                                        self.heading_hold_config.max_correction_rad_s
                                    ),
                                },
                                "start_monotonic_s": self._started_at,
                                "initial_joint_position_rad": initial.tolist(),
                            },
                            "policy": {
                                "model_file": self.model_path.name,
                                "model_path": str(self.model_path),
                                "model_sha256": self.model_sha256,
                                "contract_file": (
                                    self.model_contract_path.name
                                    if self.model_contract_path.is_file()
                                    else None
                                ),
                                "contract_sha256": self.model_contract_sha256,
                            },
                        }
                    )
                    self._state["recording"] = {
                        **self._recorder.status(),
                        "recording_id": recording_id,
                    }
                    self._recording_start_error = None
                except Exception as exc:
                    self._recording_start_error = str(exc)
                    self._state["recording"] = {
                        "active": False,
                        "error": self._recording_start_error,
                    }
            self._thread = threading.Thread(
                target=self._run,
                args=(speed, duration, initial.copy()),
                name="drobot-rl-policy",
                daemon=True,
            )
            self._thread.start()

    def request_stop(self) -> None:
        self._prepare_cancel.set()
        self._stop_event.set()

    def stop(self, timeout_s: float = 2.0) -> None:
        self.request_stop()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)
        # The running worker closes its own source after it finishes reading.
        # A pending initializer observes cancellation and closes its own source.
        self.release_if_idle()

    def release_if_idle(self) -> None:
        """Clean up preparation if stance checks rejected the requested run."""
        if self._prepare_lock.acquire(blocking=False):
            try:
                thread = self._thread
                if thread is None or not thread.is_alive():
                    self._release_imu()
            finally:
                self._prepare_lock.release()

    def _run(
        self,
        speed: float,
        duration_s: float,
        initial_target_rad: np.ndarray,
    ) -> None:
        error: str | None = None
        joint_source: _BackgroundJointSource | None = None
        try:
            assert self._policy is not None
            assert self._imu is not None
            joint_source = _BackgroundJointSource(
                self._joint_reader,
                JointStateSample(
                    position_rad=initial_target_rad.copy(),
                    velocity_rad_s=np.zeros(len(self.action_names), dtype=np.float32),
                    monotonic_time_s=time.monotonic(),
                ),
                poll_hz=self.JOINT_FEEDBACK_HZ,
            )
            joint_source.start()
            loop = self.loop_class(
                self._policy,
                _GuardedImuSource(self._imu, self._update_imu),
                joint_source,
                _GuardedMotorSink(
                    self._target_writer,
                    self._update_targets,
                    self._applied_target_reader,
                ),
                command=PolicyCommand(forward_m_s=speed),
                control_hz=self.CONTROL_HZ,
                initial_target_rad=initial_target_rad,
                joint_target_config=self.joint_target_config,
                step_observer=self._record_step,
            )
            loop.run(
                duration_s=duration_s,
                stop_event=self._stop_event,
                normal_stop_ramp_s=self.normal_stop_ramp_s,
                normal_stop_hold_s=self.normal_stop_hold_s,
            )
        except Exception as exc:
            if not self._stop_event.is_set():
                error = str(exc)
        finally:
            if joint_source is not None:
                joint_source.close()
            stopped = self._stop_event.is_set()
            try:
                # Finish motor hold/disarm first; sensor reset can take seconds.
                finalizer_error = self._finalizer(error, stopped)
            except Exception as exc:
                finalizer_error = str(exc)
                _LOGGER.exception("RL motor finalization failed")
            finally:
                with self._lock:
                    self._state.update(status="cleaning-up", motor_output_enabled=False)
                self._release_imu()
            if error is None and finalizer_error:
                error = finalizer_error
            status = "fault" if error else "stopped" if stopped else "complete"
            if self._recorder is not None:
                self._recorder.finish_trial(
                    status=status,
                    error=error,
                    details={
                        "stopped_by_operator": stopped,
                        "elapsed_s": min(
                            duration_s
                            + self.normal_stop_ramp_s
                            + self.normal_stop_hold_s
                            + self.run_stop_config.maximum_phase_alignment_wait_s,
                            max(
                                0.0,
                                time.monotonic()
                                - (self._started_at or time.monotonic()),
                            ),
                        ),
                    },
                )
            with self._lock:
                self._state.update(
                    active=False,
                    status="error" if error else "stopped" if stopped else "complete",
                    error=error,
                    motor_output_enabled=False,
                    elapsed_s=min(
                        duration_s
                        + self.normal_stop_ramp_s
                        + self.normal_stop_hold_s
                        + self.run_stop_config.maximum_phase_alignment_wait_s,
                        max(
                            0.0,
                            time.monotonic() - (self._started_at or time.monotonic()),
                        ),
                    ),
                )

    def _record_step(self, sample: PolicyStepSample) -> None:
        heading_hold = {
            "enabled": sample.heading_hold_enabled,
            "mode": self.heading_hold_config.mode,
            "kp_s": self.heading_hold_config.kp_s,
            "max_correction_rad_s": self.heading_hold_config.max_correction_rad_s,
            "reference_yaw_rad": sample.heading_reference_yaw_rad,
            "current_relative_rad": sample.heading_current_relative_rad,
            "desired_relative_rad": sample.heading_desired_relative_rad,
            "error_rad": sample.heading_error_rad,
            "correction_rad_s": sample.heading_correction_rad_s,
            "effective_yaw_rad_s": sample.effective_yaw_rad_s,
        }
        with self._lock:
            self._state["heading_hold"] = heading_hold
            self._state["effective_forward_m_s"] = sample.command.forward_m_s
            self._state["gait_frequency_hz"] = sample.gait_frequency_hz
        if self._recorder is None:
            return
        imu = sample.imu
        joints = sample.joints
        self._recorder.record_sample(
            {
                "sequence": sample.sequence,
                "elapsed_s": sample.elapsed_s,
                "monotonic_time_s": sample.monotonic_time_s,
                "sensor_time_s": {
                    "imu": imu.monotonic_time_s,
                    "joints": joints.monotonic_time_s,
                },
                "command": {
                    "forward_m_s": sample.command.forward_m_s,
                    "lateral_m_s": sample.command.lateral_m_s,
                    "yaw_rad_s": sample.command.yaw_rad_s,
                    "requested_yaw_rad_s": sample.command.yaw_rad_s,
                    "effective_yaw_rad_s": sample.effective_yaw_rad_s,
                },
                "heading_hold": heading_hold,
                "gait_clock": {
                    "sin": sample.phase_sin,
                    "cos": sample.phase_cos,
                    "frequency_hz": sample.gait_frequency_hz,
                },
                "control_timing": {
                    "target_elapsed_s": sample.target_elapsed_s,
                    "target_elapsed_semantics": (
                        "producer_budget_max_production_interval_or_ack_age; actual_bus_rate_guard_is_separate"
                        if self._applied_target_reader is not None
                        else "elapsed_since_previous_synchronous_target"
                    ),
                    "observation_duration_s": sample.observation_duration_s,
                    "inference_duration_s": sample.inference_duration_s,
                    "target_handoff_duration_s": sample.motor_sink_duration_s,
                    "missed_deadlines_total": sample.missed_deadlines_total,
                },
                "imu": {
                    "angular_velocity_body_rad_s": (
                        imu.angular_velocity_body_rad_s.tolist()
                    ),
                    "projected_gravity_body": imu.projected_gravity_body.tolist(),
                    "linear_acceleration_body_m_s2": (
                        imu.linear_acceleration_body_m_s2.tolist()
                    ),
                    "heading_yaw_rad": imu.heading_yaw_rad,
                },
                "joints": {
                    "position_rad": joints.position_rad.tolist(),
                    "velocity_rad_s": joints.velocity_rad_s.tolist(),
                },
                "observation": sample.observation.tolist(),
                "action": sample.action.tolist(),
                "requested_target_rad": sample.requested_target_rad.tolist(),
                "rate_limited_target_rad": (
                    sample.rate_limited_target_rad.tolist()
                ),
                "target_limit_base_rad": (
                    None if sample.target_limit_base_rad is None
                    else sample.target_limit_base_rad.tolist()
                ),
                "applied_motor_target": (
                    None
                    if sample.applied_motor_target is None
                    else {
                        "joint_target_rad": sample.applied_motor_target.joint_target_rad.tolist(),
                        "monotonic_time_s": sample.applied_motor_target.monotonic_time_s,
                        "sequence": sample.applied_motor_target.sequence,
                        "semantics": "last_host_completed_write_at_observation",
                    }
                ),
            }
        )

    def record_diagnostic(self, payload: Mapping[str, Any]) -> None:
        self.record_event("motor_diagnostic", payload)

    def record_event(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        if self._recorder is not None:
            self._recorder.record_event(
                {
                    "type": event_type,
                    "monotonic_time_s": time.monotonic(),
                    **dict(payload or {}),
                }
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
                "heading_yaw_rad": round(float(sample.heading_yaw_rad), 6),
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
            for index, name in enumerate(self.action_names)
        ]
        with self._lock:
            self._state["targets"] = targets
            self._state["last_policy_time_s"] = round(monotonic_time_s, 6)
