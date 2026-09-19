"""Replaceable IMU and joint-state sources for script and future ROS use."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .contract import DEFAULT_JOINT_POSITION_RAD

_LOGGER = logging.getLogger(__name__)
_BNO085_OWNER_LOCK = threading.Lock()


@dataclass(frozen=True)
class ImuSample:
    angular_velocity_body_rad_s: np.ndarray
    projected_gravity_body: np.ndarray
    linear_acceleration_body_m_s2: np.ndarray
    heading_yaw_rad: float
    monotonic_time_s: float


@dataclass(frozen=True)
class JointStateSample:
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    monotonic_time_s: float


class ImuSource(Protocol):
    def read(self) -> ImuSample: ...


class JointStateSource(Protocol):
    def read(self) -> JointStateSample: ...


class BackgroundImuSource:
    """Own an IMU on one worker and provide bounded, fresh cached reads.

    The factory, every device read, and device close run on the same thread.
    A blocked driver cannot be safely killed; callers can stop waiting, while
    the worker retains device ownership until its call returns and it closes.
    """

    def __init__(
        self,
        factory: Callable[[], ImuSource],
        *,
        poll_hz: float = 100.0,
        stale_after_s: float = 0.12,
        startup_timeout_s: float = 30.0,
        close_timeout_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (
            ("poll_hz", poll_hz),
            ("stale_after_s", stale_after_s),
            ("startup_timeout_s", startup_timeout_s),
            ("close_timeout_s", close_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self._factory = factory
        self._period_s = 1.0 / poll_hz
        self._stale_after_s = stale_after_s
        self._startup_timeout_s = startup_timeout_s
        self._close_timeout_s = close_timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._done = threading.Event()
        self._sample: ImuSample | None = None
        self._error: Exception | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _copy_sample(sample: ImuSample) -> ImuSample:
        vectors = [
            np.array(vector, dtype=np.float32, copy=True)
            for vector in (
                sample.angular_velocity_body_rad_s,
                sample.projected_gravity_body,
                sample.linear_acceleration_body_m_s2,
            )
        ]
        if any(vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in vectors):
            raise RuntimeError("IMU sample must contain three complete finite vectors")
        heading = float(sample.heading_yaw_rad)
        timestamp = float(sample.monotonic_time_s)
        if not math.isfinite(heading) or not math.isfinite(timestamp):
            raise RuntimeError("IMU heading and acquisition timestamp must be finite")
        return ImuSample(vectors[0], vectors[1], vectors[2], heading, timestamp)

    def start(self, *, cancel_event: threading.Event | None = None) -> ImuSample:
        if cancel_event is not None and cancel_event.is_set():
            self.close()
            raise RuntimeError("IMU initialization was cancelled")
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("IMU acquisition session is closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._poll,
                    name="drobot-imu-acquisition",
                    daemon=True,
                )
                self._thread.start()
        deadline = time.monotonic() + self._startup_timeout_s
        while True:
            if cancel_event is not None and cancel_event.is_set():
                self.close()
                raise RuntimeError("IMU initialization was cancelled")
            if self._ready.is_set():
                return self.read()
            if self._stop.is_set():
                raise RuntimeError("IMU initialization was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self.close()
                raise RuntimeError(
                    "IMU initialization timed out; device cleanup may still be pending"
                )
            self._ready.wait(min(0.05, remaining))

    def _poll(self) -> None:
        source: ImuSource | None = None
        try:
            source = self._factory()
            deadline = self._clock()
            while not self._stop.is_set():
                sample = self._copy_sample(source.read())
                now = self._clock()
                age = now - sample.monotonic_time_s
                if age < -0.01:
                    raise RuntimeError("IMU acquisition timestamp is in the future")
                if age > self._stale_after_s:
                    raise RuntimeError(f"IMU sample is stale ({age * 1000.0:.0f} ms)")
                with self._lock:
                    if self._sample is not None and sample.monotonic_time_s < self._sample.monotonic_time_s:
                        raise RuntimeError("IMU acquisition timestamp moved backwards")
                    self._sample = sample
                self._ready.set()
                # Maintain cadence without spinning or replaying missed reads.
                deadline += self._period_s
                if deadline <= now:
                    skipped = math.floor((now - deadline) / self._period_s) + 1
                    deadline += skipped * self._period_s
                self._stop.wait(max(0.0, deadline - self._clock()))
        except Exception as exc:
            with self._lock:
                self._error = exc
            self._ready.set()
        finally:
            if source is not None:
                close = getattr(source, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        with self._lock:
                            if self._error is None:
                                self._error = exc
                        _LOGGER.warning("IMU worker cleanup failed", exc_info=True)
            self._done.set()
            self._ready.set()

    def read(self) -> ImuSample:
        with self._lock:
            sample, error = self._sample, self._error
        if error is not None:
            raise RuntimeError(f"IMU acquisition failed: {error}") from error
        if self._stop.is_set():
            raise RuntimeError("IMU acquisition session is closed")
        if sample is None:
            raise RuntimeError("IMU has not produced a complete sample")
        age = self._clock() - sample.monotonic_time_s
        if age < -0.01 or age > self._stale_after_s:
            raise RuntimeError(f"IMU sample is stale or invalid ({age * 1000.0:.0f} ms)")
        return self._copy_sample(sample)

    def close(self) -> None:
        self._stop.set()
        self._ready.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._close_timeout_s)
            if thread.is_alive():
                _LOGGER.warning("IMU owner is still finishing a device call; cleanup remains pending")

    @property
    def cleanup_pending(self) -> bool:
        return self._thread is not None and not self._done.is_set()


class LevelImuSource:
    """Stationary level input for dependency and policy-pipeline checks."""

    def read(self) -> ImuSample:
        return ImuSample(
            angular_velocity_body_rad_s=np.zeros(3, dtype=np.float32),
            projected_gravity_body=np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
            linear_acceleration_body_m_s2=np.asarray((0.0, 0.0, 9.81), dtype=np.float32),
            heading_yaw_rad=0.0,
            monotonic_time_s=time.monotonic(),
        )


class NeutralJointStateSource:
    """Print-only bring-up source; it is not valid for closed-loop actuation."""

    def read(self) -> JointStateSample:
        return JointStateSample(
            position_rad=DEFAULT_JOINT_POSITION_RAD.copy(),
            velocity_rad_s=np.zeros(12, dtype=np.float32),
            monotonic_time_s=time.monotonic(),
        )


def _quaternion_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Return the sensor-to-game-world rotation represented by an XYZW quaternion."""

    x, y, z, w = (float(value) for value in quaternion_xyzw)
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        raise RuntimeError("BNO085 returned a non-finite quaternion")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-6:
        raise RuntimeError("BNO085 returned an invalid zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def _quaternion_inverse_rotate(
    quaternion_xyzw: np.ndarray, vector_world: np.ndarray
) -> np.ndarray:
    rotation = _quaternion_rotation_matrix(quaternion_xyzw)
    return rotation.T @ vector_world


def _body_forward_heading_rad(
    quaternion_xyzw: np.ndarray,
    axis_map: tuple[tuple[int, float], ...],
) -> float:
    """Project mapped body +X into the BNO game-world horizontal plane."""

    # ``axis_map`` declares body[i] = sign * sensor[index].  Its first entry
    # therefore gives the sensor-frame vector corresponding to body-forward.
    sensor_forward = np.zeros(3, dtype=np.float32)
    sensor_index, sign = axis_map[0]
    sensor_forward[sensor_index] = sign
    world_forward = _quaternion_rotation_matrix(quaternion_xyzw) @ sensor_forward
    horizontal_norm = math.hypot(float(world_forward[0]), float(world_forward[1]))
    if horizontal_norm < 1.0e-4:
        raise RuntimeError("BNO085 body-forward heading is undefined near vertical")
    return math.atan2(float(world_forward[1]), float(world_forward[0]))


def parse_axis_map(text: str) -> tuple[tuple[int, float], ...]:
    axes = {"x": 0, "y": 1, "z": 2}
    result: list[tuple[int, float]] = []
    for token in text.lower().replace(" ", "").split(","):
        if len(token) != 2 or token[0] not in "+-" or token[1] not in axes:
            raise ValueError("axis map must look like +x,+y,+z or +y,-x,+z")
        result.append((axes[token[1]], 1.0 if token[0] == "+" else -1.0))
    if len(result) != 3 or len({axis for axis, _ in result}) != 3:
        raise ValueError("axis map must use x, y, and z exactly once")
    transform = np.zeros((3, 3), dtype=np.float32)
    for body_axis, (sensor_axis, sign) in enumerate(result):
        transform[body_axis, sensor_axis] = sign
    if not math.isclose(float(np.linalg.det(transform)), 1.0, abs_tol=1.0e-6):
        raise ValueError(
            "axis map must be a right-handed rigid rotation so gyro yaw and "
            "quaternion heading use the same sign"
        )
    return tuple(result)


class Bno085ImuSource:
    """Read body-frame policy inputs directly from the Pi's BNO085."""

    def __init__(
        self,
        address: int = 0x4A,
        axis_map: str = "+x,+y,+z",
        i2c_bus: int | None = None,
    ) -> None:
        try:
            from adafruit_bno08x import (
                BNO_REPORT_ACCELEROMETER,
                BNO_REPORT_GAME_ROTATION_VECTOR,
                BNO_REPORT_GYROSCOPE,
            )
            from adafruit_bno08x.i2c import BNO08X_I2C
        except ImportError as exc:
            raise RuntimeError(
                "BNO085 dependencies are missing; install the package with [bno085]"
            ) from exc

        self._axis_map = parse_axis_map(axis_map)
        if i2c_bus is None:
            configured_bus = os.environ.get("DROBOT_BNO085_I2C_BUS", "").strip()
            if configured_bus:
                try:
                    i2c_bus = int(configured_bus, 10)
                except ValueError as exc:
                    raise RuntimeError(
                        "DROBOT_BNO085_I2C_BUS must be a non-negative integer"
                    ) from exc
                if i2c_bus < 0:
                    raise RuntimeError(
                        "DROBOT_BNO085_I2C_BUS must be a non-negative integer"
                    )

        if i2c_bus is None:
            import board
            import busio

            open_bus = lambda: busio.I2C(board.SCL, board.SDA)
        else:
            try:
                from adafruit_extended_bus import ExtendedI2C
            except ImportError as exc:
                raise RuntimeError(
                    "Software I2C support is missing; reinstall the package with [bno085]"
                ) from exc
            open_bus = lambda: ExtendedI2C(i2c_bus)

        self._i2c = None
        self._imu = None
        self._owns_sensor = _BNO085_OWNER_LOCK.acquire(blocking=False)
        if not self._owns_sensor:
            raise RuntimeError("BNO085 is already in use; wait for the previous RL session to close")
        try:
            for session_attempt in range(2):
                stage = "opening bus"
                try:
                    self._i2c = open_bus()
                    stage = "reset and product identification"
                    # A fresh driver instance resets the sensor and its packet
                    # sequence/cache state. Reusing the failed instance did not.
                    self._imu = BNO08X_I2C(self._i2c, address=address)
                    time.sleep(0.25)
                    for feature_id, feature_name in (
                        (BNO_REPORT_ACCELEROMETER, "accelerometer"),
                        (BNO_REPORT_GYROSCOPE, "gyroscope"),
                        (BNO_REPORT_GAME_ROTATION_VECTOR, "orientation"),
                    ):
                        stage = f"enabling {feature_name}"
                        for attempt in range(2):
                            try:
                                self._imu.enable_feature(feature_id)
                                break
                            except RuntimeError as exc:
                                if exc.args != ("Was not able to enable feature", feature_id) or attempt:
                                    raise
                                time.sleep(0.25)
                    # Require complete, finite samples before exposing this
                    # source to the controller. This happens only at startup.
                    stage = "waiting for complete samples"
                    deadline = time.monotonic() + 1.5
                    consecutive = 0
                    while consecutive < 3:
                        try:
                            self.read()
                            consecutive += 1
                        except (RuntimeError, OSError, KeyError):
                            consecutive = 0
                            if time.monotonic() >= deadline:
                                raise
                        if time.monotonic() >= deadline and consecutive < 3:
                            raise RuntimeError("IMU warmup did not produce three complete samples")
                        if consecutive < 3:
                            time.sleep(0.06)
                    return
                except (RuntimeError, OSError, KeyError) as exc:
                    _LOGGER.warning(
                        "BNO085 startup session %s/2 failed while %s: %s",
                        session_attempt + 1, stage, exc,
                    )
                    self._close_bus(reset_sensor=False)
                    if session_attempt == 1:
                        bus_label = "default I2C" if i2c_bus is None else f"I2C bus {i2c_bus}"
                        raise RuntimeError(
                            f"BNO085 failed while {stage} on {bus_label} at "
                            f"0x{address:02X} after two fresh initialization attempts. "
                            "RL has not started. Support and disarm the robot, "
                            "then remove power from the IMU/Pi for 10 seconds. "
                            f"Sensor error: {exc}"
                        ) from exc
                    time.sleep(0.75)
        except Exception:
            self.close(reset_sensor=False)
            raise

    def _close_bus(self, *, reset_sensor: bool) -> None:
        imu, self._imu = self._imu, None
        i2c, self._i2c = self._i2c, None
        try:
            if imu is not None and reset_sensor:
                # The driver exposes no public disable-report API. Its public
                # reset stops streaming, preventing an unread idle backlog.
                imu.soft_reset()
        except Exception:
            _LOGGER.warning("BNO085 reset during cleanup failed", exc_info=True)
        finally:
            if i2c is not None:
                try:
                    i2c.deinit()
                except Exception:
                    _LOGGER.warning("Could not close BNO085 I2C handle", exc_info=True)

    def close(self, *, reset_sensor: bool = True) -> None:
        try:
            self._close_bus(reset_sensor=reset_sensor)
        finally:
            if self._owns_sensor:
                self._owns_sensor = False
                _BNO085_OWNER_LOCK.release()

    def _to_body(self, sensor_vector: np.ndarray) -> np.ndarray:
        return np.asarray(
            [sign * sensor_vector[index] for index, sign in self._axis_map],
            dtype=np.float32,
        )

    def read(self) -> ImuSample:
        if self._imu is None:
            raise RuntimeError("BNO085 source is closed")
        acceleration = self._imu.acceleration
        gyro = self._imu.gyro
        quaternion = self._imu.game_quaternion
        if acceleration is None or gyro is None or quaternion is None:
            raise RuntimeError("BNO085 sample is incomplete")
        acceleration_sensor = np.asarray(acceleration, dtype=np.float32)
        gyro_sensor = np.asarray(gyro, dtype=np.float32)
        quaternion_xyzw = np.asarray(quaternion, dtype=np.float32)
        gravity_sensor = _quaternion_inverse_rotate(
            quaternion_xyzw, np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
        )
        heading_yaw_rad = _body_forward_heading_rad(
            quaternion_xyzw,
            self._axis_map,
        )
        values = np.concatenate(
            (
                acceleration_sensor,
                gyro_sensor,
                gravity_sensor,
                quaternion_xyzw,
                np.asarray((heading_yaw_rad,), dtype=np.float32),
            )
        )
        if not np.all(np.isfinite(values)):
            raise RuntimeError("BNO085 sample contains a non-finite value")
        return ImuSample(
            angular_velocity_body_rad_s=self._to_body(gyro_sensor),
            projected_gravity_body=self._to_body(gravity_sensor),
            linear_acceleration_body_m_s2=self._to_body(acceleration_sensor),
            heading_yaw_rad=heading_yaw_rad,
            monotonic_time_s=time.monotonic(),
        )
