"""Control dashboard for all four Drobot legs."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import secrets
import shutil
import socket
import threading
import time
import tomllib
import webbrowser
from collections import deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from drobot_leg_testbed.controller import LegController
from drobot_leg_testbed.model import (
    BusConfig,
    Calibration,
    LegConfig,
    MotorConfig,
    calibration_from_centers,
    degrees_to_raw,
    load_calibration,
    load_config,
    raw_to_degrees,
    save_calibration,
)
from drobot_leg_testbed.transport import MotorStatus, STSBus

from drobot_hardware_test_apps.crawl_gait import (
    LEG_CORNERS,
    diagonal_pair_gait_degrees,
    distributed_push_crawl_degrees,
)

LOCAL_HOST = "127.0.0.1"
APP_ROOT = Path(__file__).resolve().parents[2]
HARDWARE_ROOT = APP_ROOT.parents[1]
REPO_ROOT = HARDWARE_ROOT.parent
DEFAULT_MANIFEST = HARDWARE_ROOT / "robot-runtime" / "four-leg.toml"
DEFAULT_RL_MODEL = (
    REPO_ROOT
    / "onboard"
    / "models"
    / "parallel-walking-v20-external-rear-payload"
    / "model_900.onnx"
)
DEFAULT_RECORDINGS_DIR = (
    Path.home() / ".local" / "share" / "drobot2" / "rl-recordings"
)
STATIC_DIR = Path(__file__).with_name("four_leg_static")
LAN_CLIENT_VERSION = "4"


@dataclass(frozen=True)
class MonitoringConfig:
    voltage_warning_low_v: float
    voltage_warning_high_v: float
    voltage_spread_warning_v: float
    voltage_sag_warning_v: float
    temperature_warning_c: int
    temperature_stop_confirmation_s: float
    temperature_critical_c: int
    leg_current_warning_ma: float
    motor_stall_current_warning_ma: float
    stall_tracking_error_deg: float
    stall_speed_raw_max: int
    battery_series_cells: int


@dataclass(frozen=True)
class ServerConfig:
    http_port: int
    ramp_rate_deg_s: float
    heartbeat_timeout_s: float


@dataclass(frozen=True)
class CrawlConfig:
    period_s: float
    stance_settle_s: float
    ramp_rate_deg_s: float
    stride_m: float
    lift_m: float
    support_extension_m: float
    weight_shift_forward_m: float
    weight_shift_lateral_m: float
    stance_down_m: float
    stance_fore_aft_m: float
    abduction_deg: float


@dataclass(frozen=True)
class LegProfile:
    number: int
    label: str
    corner: str
    config: LegConfig
    calibration: Calibration
    config_path: Path
    calibration_path: Path


@dataclass(frozen=True)
class DashboardConfig:
    server: ServerConfig
    monitoring: MonitoringConfig
    crawl: CrawlConfig
    legs: tuple[LegProfile, ...]

    @property
    def bus(self) -> BusConfig:
        return self.legs[0].config.bus


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_dashboard_config(path: str | Path) -> DashboardConfig:
    manifest_path = Path(path).resolve()
    with manifest_path.open("rb") as stream:
        data = tomllib.load(stream)

    server_data = data.get("server")
    monitoring_data = data.get("monitoring")
    crawl_data = data.get("crawl")
    leg_data = data.get("legs")
    if not isinstance(server_data, dict):
        raise ValueError("Manifest requires a [server] table")
    if not isinstance(monitoring_data, dict):
        raise ValueError("Manifest requires a [monitoring] table")
    if not isinstance(crawl_data, dict):
        raise ValueError("Manifest requires a [crawl] table")
    if not isinstance(leg_data, list) or len(leg_data) != 4:
        raise ValueError("Manifest requires exactly four [[legs]] tables")

    server = ServerConfig(
        http_port=int(server_data.get("http_port", 8766)),
        ramp_rate_deg_s=_finite_float(
            server_data.get("ramp_rate_deg_s", 270.0),
            "server.ramp_rate_deg_s",
        ),
        heartbeat_timeout_s=_finite_float(
            server_data.get("heartbeat_timeout_s", 3.0),
            "server.heartbeat_timeout_s",
        ),
    )
    if not 1 <= server.http_port <= 65_535:
        raise ValueError("server.http_port must be in [1, 65535]")
    if not 1.0 <= server.ramp_rate_deg_s <= 270.0:
        raise ValueError("server.ramp_rate_deg_s must be in [1, 270]")
    if not 1.0 <= server.heartbeat_timeout_s <= 120.0:
        raise ValueError("server.heartbeat_timeout_s must be in [1, 120]")

    monitoring = MonitoringConfig(
        voltage_warning_low_v=_finite_float(
            monitoring_data.get("voltage_warning_low_v", 10.5),
            "monitoring.voltage_warning_low_v",
        ),
        voltage_warning_high_v=_finite_float(
            monitoring_data.get("voltage_warning_high_v", 12.6),
            "monitoring.voltage_warning_high_v",
        ),
        voltage_spread_warning_v=_finite_float(
            monitoring_data.get("voltage_spread_warning_v", 0.5),
            "monitoring.voltage_spread_warning_v",
        ),
        voltage_sag_warning_v=_finite_float(
            monitoring_data.get("voltage_sag_warning_v", 0.6),
            "monitoring.voltage_sag_warning_v",
        ),
        temperature_warning_c=int(monitoring_data.get("temperature_warning_c", 60)),
        temperature_stop_confirmation_s=_finite_float(
            monitoring_data.get("temperature_stop_confirmation_s", 5.0),
            "monitoring.temperature_stop_confirmation_s",
        ),
        temperature_critical_c=int(
            monitoring_data.get("temperature_critical_c", 65)
        ),
        leg_current_warning_ma=_finite_float(
            monitoring_data.get("leg_current_warning_ma", 3000.0),
            "monitoring.leg_current_warning_ma",
        ),
        motor_stall_current_warning_ma=_finite_float(
            monitoring_data.get("motor_stall_current_warning_ma", 1200.0),
            "monitoring.motor_stall_current_warning_ma",
        ),
        stall_tracking_error_deg=_finite_float(
            monitoring_data.get("stall_tracking_error_deg", 8.0),
            "monitoring.stall_tracking_error_deg",
        ),
        stall_speed_raw_max=int(monitoring_data.get("stall_speed_raw_max", 20)),
        battery_series_cells=int(monitoring_data.get("battery_series_cells", 3)),
    )
    if not 0 < monitoring.voltage_warning_low_v:
        raise ValueError("Low-voltage warning must be positive")
    if not (
        monitoring.voltage_warning_low_v < monitoring.voltage_warning_high_v <= 15.0
    ):
        raise ValueError("Voltage warning range is invalid")
    if not 0 < monitoring.voltage_spread_warning_v <= 3.0:
        raise ValueError("Voltage-spread warning must be in (0, 3]")
    if not 0.1 <= monitoring.voltage_sag_warning_v <= 3.0:
        raise ValueError("Voltage-sag warning must be in [0.1, 3]")
    if not 30 <= monitoring.temperature_warning_c <= 90:
        raise ValueError("Temperature warning must be in [30, 90] C")
    if not 1.0 <= monitoring.temperature_stop_confirmation_s <= 10.0:
        raise ValueError("Temperature-stop confirmation must be in [1, 10] seconds")
    if not (
        monitoring.temperature_warning_c
        < monitoring.temperature_critical_c
        <= 90
    ):
        raise ValueError(
            "Critical temperature must be above the warning and at most 90 C"
        )
    if not 100 <= monitoring.leg_current_warning_ma <= 10_000:
        raise ValueError("Leg-current warning must be in [100, 10000] mA")
    if not 100 <= monitoring.motor_stall_current_warning_ma <= 5_000:
        raise ValueError("Motor-stall current warning must be in [100, 5000] mA")
    if not 1.0 <= monitoring.stall_tracking_error_deg <= 45.0:
        raise ValueError("Stall tracking-error threshold must be in [1, 45] deg")
    if not 0 <= monitoring.stall_speed_raw_max <= 500:
        raise ValueError("Stall raw-speed threshold must be in [0, 500]")
    if not 2 <= monitoring.battery_series_cells <= 6:
        raise ValueError("Battery series-cell count must be in [2, 6]")

    crawl = CrawlConfig(
        period_s=_finite_float(
            crawl_data.get("period_s", 4.0),
            "crawl.period_s",
        ),
        stance_settle_s=_finite_float(
            crawl_data.get("stance_settle_s", 1.5),
            "crawl.stance_settle_s",
        ),
        ramp_rate_deg_s=_finite_float(
            crawl_data.get("ramp_rate_deg_s", 60.0),
            "crawl.ramp_rate_deg_s",
        ),
        stride_m=_finite_float(crawl_data.get("stride_m", 0.096), "crawl.stride_m"),
        lift_m=_finite_float(crawl_data.get("lift_m", 0.035), "crawl.lift_m"),
        support_extension_m=_finite_float(
            crawl_data.get("support_extension_m", 0.0),
            "crawl.support_extension_m",
        ),
        weight_shift_forward_m=_finite_float(
            crawl_data.get("weight_shift_forward_m", 0.006),
            "crawl.weight_shift_forward_m",
        ),
        weight_shift_lateral_m=_finite_float(
            crawl_data.get("weight_shift_lateral_m", 0.0),
            "crawl.weight_shift_lateral_m",
        ),
        stance_down_m=_finite_float(
            crawl_data.get("stance_down_m", 0.329341447),
            "crawl.stance_down_m",
        ),
        stance_fore_aft_m=_finite_float(
            crawl_data.get("stance_fore_aft_m", 0.080),
            "crawl.stance_fore_aft_m",
        ),
        abduction_deg=_finite_float(
            crawl_data.get("abduction_deg", 0.0),
            "crawl.abduction_deg",
        ),
    )
    if not 4.0 <= crawl.period_s <= 60.0:
        raise ValueError("crawl.period_s must be in [4, 60]")
    if not 0.5 <= crawl.stance_settle_s <= 5.0:
        raise ValueError("crawl.stance_settle_s must be in [0.5, 5]")
    if not 5.0 <= crawl.ramp_rate_deg_s <= 180.0:
        raise ValueError("crawl.ramp_rate_deg_s must be in [5, 180]")
    if not 0.005 <= crawl.stride_m <= 0.120:
        raise ValueError("crawl.stride_m must be in [0.005, 0.120]")
    if not 0.005 <= crawl.lift_m <= 0.080:
        raise ValueError("crawl.lift_m must be in [0.005, 0.080]")
    if not math.isclose(crawl.support_extension_m, 0.0, abs_tol=1e-12):
        raise ValueError(
            "crawl.support_extension_m must be 0 for rectangular flat support"
        )
    if not 0.0 <= crawl.weight_shift_forward_m <= 0.040:
        raise ValueError("crawl.weight_shift_forward_m must be in [0, 0.040]")
    if not 0.0 <= crawl.weight_shift_lateral_m <= 0.030:
        raise ValueError("crawl.weight_shift_lateral_m must be in [0, 0.030]")
    if not 0.250 <= crawl.stance_down_m <= 0.370:
        raise ValueError("crawl.stance_down_m must be in [0.250, 0.370]")
    if not 0.0 <= crawl.stance_fore_aft_m <= 0.120:
        raise ValueError("crawl.stance_fore_aft_m must be in [0, 0.120]")
    if not 0.0 <= crawl.abduction_deg <= 20.0:
        raise ValueError("crawl.abduction_deg must be in [0, 20]")

    profiles: list[LegProfile] = []
    for index, entry in enumerate(leg_data, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"legs[{index}] must be a table")
        number = int(entry.get("number", index))
        label = str(entry.get("label", f"Leg {number}")).strip()
        corner = str(entry.get("corner", "")).strip().lower()
        if not label:
            raise ValueError(f"legs[{index}].label must not be empty")
        if corner not in LEG_CORNERS:
            raise ValueError(
                f"legs[{index}].corner must be one of {', '.join(LEG_CORNERS)}"
            )
        config_path = (manifest_path.parent / str(entry["profile"])).resolve()
        calibration_path = (manifest_path.parent / str(entry["calibration"])).resolve()
        config = load_config(config_path)
        calibration = load_calibration(calibration_path, config)
        profiles.append(
            LegProfile(
                number=number,
                label=label,
                corner=corner,
                config=config,
                calibration=calibration,
                config_path=config_path,
                calibration_path=calibration_path,
            )
        )

    if {profile.number for profile in profiles} != {1, 2, 3, 4}:
        raise ValueError("Leg numbers must be exactly 1, 2, 3, and 4")
    profiles.sort(key=lambda profile: profile.number)
    if {profile.corner for profile in profiles} != set(LEG_CORNERS):
        raise ValueError("Four-leg dashboard requires each body corner exactly once")
    first_bus = profiles[0].config.bus
    if any(profile.config.bus != first_bus for profile in profiles[1:]):
        raise ValueError("All four leg profiles must use identical bus settings")
    ids = [motor.servo_id for profile in profiles for motor in profile.config.motors]
    if sorted(ids) != list(range(1, 13)):
        raise ValueError("Four-leg dashboard requires unique servo IDs 1 through 12")

    return DashboardConfig(server, monitoring, crawl, tuple(profiles))


class FourLegDemoBus:
    """In-memory twelve-motor bus for UI and safety testing."""

    def __init__(self, dashboard: DashboardConfig):
        self.feedback_mode = "in_memory_group"
        self.positions: dict[int, int] = {}
        for profile in dashboard.legs:
            for motor in profile.config.motors:
                self.positions[motor.servo_id] = profile.calibration.motor(
                    motor
                ).center_tick
        self.torque: set[int] = set()
        self.voltage_v: dict[int, float] = {
            servo_id: 12.2 for servo_id in self.positions
        }
        self.temperature_c: dict[int, int] = {
            servo_id: 31 for servo_id in self.positions
        }
        self.current_ma: dict[int, float] = {
            servo_id: 0.0 for servo_id in self.positions
        }

    def open(self) -> None:
        return

    def close(self) -> None:
        return

    def reopen(self) -> None:
        self.close()
        self.open()

    def require_motor(self, motor: MotorConfig) -> int:
        if motor.servo_id not in self.positions:
            raise RuntimeError(f"Demo motor ID {motor.servo_id} is missing")
        return 777

    def read_position(self, servo_id: int) -> int:
        return self.positions[servo_id]

    def read_position_speed(self, servo_id: int) -> tuple[int, int]:
        return self.positions[servo_id], 0

    def read_positions_speeds(
        self,
        servo_ids: list[int] | tuple[int, ...],
    ) -> dict[int, tuple[int, int]]:
        return {servo_id: (self.positions[servo_id], 0) for servo_id in servo_ids}

    def write_position(
        self,
        servo_id: int,
        raw_position: int,
        _bus_config: BusConfig,
    ) -> None:
        self.positions[servo_id] = raw_position

    def write_position_command(
        self,
        servo_id: int,
        raw_position: int,
        bus_config: BusConfig,
    ) -> None:
        self.write_position(servo_id, raw_position, bus_config)

    def write_position_commands(
        self,
        raw_positions: dict[int, int],
        bus_config: BusConfig,
    ) -> None:
        for servo_id, raw_position in raw_positions.items():
            self.write_position(servo_id, raw_position, bus_config)

    def enable_torque(self, servo_id: int) -> None:
        self.torque.add(servo_id)
        self.current_ma[servo_id] = 6.5

    def disable_torque(self, servo_id: int) -> None:
        self.torque.discard(servo_id)
        self.current_ma[servo_id] = 0.0

    def status(self, motor: MotorConfig) -> MotorStatus:
        return MotorStatus(
            servo_id=motor.servo_id,
            model_number=777,
            raw_position=self.positions[motor.servo_id],
            raw_speed=0,
            voltage_v=self.voltage_v[motor.servo_id],
            temperature_c=self.temperature_c[motor.servo_id],
            current_ma=self.current_ma[motor.servo_id],
            torque_enabled=motor.servo_id in self.torque,
        )


class FourLegSession:
    """Serialize twelve-motor access and enforce bounded, explicit motion."""

    RL_MOTOR_OUTPUT_HZ = 60.0

    def __init__(
        self,
        dashboard: DashboardConfig,
        bus: Any,
        *,
        ramp_rate_deg_s: float | None = None,
        tick_interval_s: float = 0.05,
        heartbeat_timeout_s: float | None = None,
        clock: Any = time.monotonic,
        persist_calibration: bool = True,
        rl_model_path: Path | None = None,
        rl_imu_axis_map: str = "+x,+y,+z",
        recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
    ):
        self.dashboard = dashboard
        self.bus = bus
        self.ramp_rate_deg_s = (
            dashboard.server.ramp_rate_deg_s
            if ramp_rate_deg_s is None
            else ramp_rate_deg_s
        )
        self.heartbeat_timeout_s = (
            dashboard.server.heartbeat_timeout_s
            if heartbeat_timeout_s is None
            else heartbeat_timeout_s
        )
        if not 1.0 <= self.ramp_rate_deg_s <= 270.0:
            raise ValueError("ramp rate must be in [1, 270] deg/s")
        if not 1.0 <= self.heartbeat_timeout_s <= 120.0:
            raise ValueError("heartbeat timeout must be in [1, 120] seconds")
        self.tick_interval_s = tick_interval_s
        self.clock = clock
        self.persist_calibration = persist_calibration
        self.controllers = {
            profile.number: LegController(
                profile.config,
                profile.calibration,
                bus,
            )
            for profile in dashboard.legs
        }
        self.desired_deg: dict[tuple[int, str], float] = {}
        self.heartbeat_lock = threading.Lock()
        self.last_heartbeat = clock()
        self.heartbeat_count = 0
        self.last_heartbeat_source: str | None = None
        self.last_event = "Starting"
        self.fault: str | None = None
        self.crawl_mode = "distributed"
        self.crawl_stage = "idle"
        self.crawl_phase = "idle"
        self.crawl_swing_corner: str | None = None
        self.crawl_swing_pair: tuple[str, ...] | None = None
        self.crawl_push_partner: str | None = None
        self.crawl_started_at: float | None = None
        self.crawl_elapsed_s = 0.0
        self.crawl_target_reached_at: float | None = None
        self.crawl_progress = 0.0
        self.power_samples: deque[tuple[float, float, float, float]] = deque()
        self.idle_voltage_samples: deque[float] = deque(maxlen=30)
        self.power_energy_wh = 0.0
        self.power_last_sample_at: float | None = None
        self.power_last_w = 0.0
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.rl_controller: Any | None = None
        self.rl_recorder: Any | None = None
        self.rl_motor_order: tuple[
            tuple[LegProfile, MotorConfig, LegController], ...
        ] = ()
        self.rl_feedback_position_rad: Any | None = None
        self.rl_feedback_velocity_rad_s: Any | None = None
        self.rl_feedback_time_s: float | None = None
        self.rl_diagnostic_index = 0
        self.rl_diagnostic_time_s: float | None = None
        self.rl_temperature_candidates: dict[int, tuple[float, int, int]] = {}
        self.last_full_snapshot: dict[str, Any] | None = None
        if rl_model_path is not None:
            from drobot_policy_runtime.contract import (
                ACTION_NAMES,
                SERVO_ID_BY_ACTION_NAME,
            )
            from drobot_policy_runtime.recording import (
                JsonlTrialRecorder,
                sha256_file,
            )

            from drobot_hardware_test_apps.rl_policy_control import RlPolicyController

            motor_by_id = {
                motor.servo_id: (profile, motor, self.controllers[profile.number])
                for profile in self.profiles
                for motor in profile.config.motors
            }
            self.rl_motor_order = tuple(
                motor_by_id[SERVO_ID_BY_ACTION_NAME[name]] for name in ACTION_NAMES
            )
            self.rl_recorder = JsonlTrialRecorder(recordings_dir)
            recording_metadata = {
                "robot": {
                    "name": "drobot2",
                    "measured_total_mass_kg": 3.175,
                    "joint_order": list(ACTION_NAMES),
                    "servo_ids": [
                        SERVO_ID_BY_ACTION_NAME[name] for name in ACTION_NAMES
                    ],
                    "legs": [
                        {
                            "number": profile.number,
                            "label": profile.label,
                            "corner": profile.corner,
                            "profile_path": str(profile.config_path),
                            "profile_sha256": sha256_file(profile.config_path),
                            "calibration_path": str(profile.calibration_path),
                            "calibration_sha256": sha256_file(
                                profile.calibration_path
                            ),
                        }
                        for profile in self.profiles
                    ],
                },
                "sensors": {
                    "imu": "BNO085",
                    "imu_axis_map": rl_imu_axis_map,
                    "joint_feedback": "STS3215 encoder telemetry",
                    "servo_baudrate": self.dashboard.bus.baudrate,
                    "feedback_transport": (
                        "group synchronous position/speed read with "
                        "sequential fallback"
                    ),
                },
                "data_contract": {
                    "observation_size": 50,
                    "action_size": 12,
                    "observation_order": [
                        "command_forward_m_s",
                        "command_lateral_m_s",
                        "command_yaw_rad_s",
                        "gait_phase_sin",
                        "gait_phase_cos",
                        "imu_angular_velocity_body_rad_s[3]",
                        "imu_projected_gravity_body[3]",
                        "imu_linear_acceleration_body_g[3]",
                        "joint_position_error_rad[12]",
                        "joint_velocity_normalized[12]",
                        "previous_action[12]",
                    ],
                    "target_rate_limit": "actual elapsed monotonic time",
                    "missed_deadline_policy": "skip; never replay catch-up bursts",
                },
                "payload": {
                    "mass_kg": 0.523179545,
                    "position_body_m": [-0.1315, 0.0, 0.05],
                    "note": (
                        "416 g measured battery plus CAD-derived printed box "
                        "and lid; body-frame position matches the V20 external "
                        "rear payload model"
                    ),
                },
            }
            self.rl_controller = RlPolicyController(
                rl_model_path,
                rl_imu_axis_map,
                self._read_rl_joint_state,
                self._apply_rl_targets,
                self._finish_rl_policy,
                recorder=self.rl_recorder,
                recording_metadata=recording_metadata,
            )

    @property
    def profiles(self) -> tuple[LegProfile, ...]:
        return self.dashboard.legs

    def _profile(self, leg_number: int | str) -> LegProfile:
        number = int(leg_number)
        for profile in self.profiles:
            if profile.number == number:
                return profile
        raise KeyError(f"Unknown leg {leg_number}; expected 1, 2, 3, or 4")

    def _selection(
        self,
        leg_number: int | str,
        motor_selector: int | str,
    ) -> tuple[LegProfile, MotorConfig, LegController]:
        profile = self._profile(leg_number)
        motor = profile.config.motor(motor_selector)
        return profile, motor, self.controllers[profile.number]

    def start(
        self,
        *,
        start_worker: bool = True,
        allow_disconnected: bool = False,
    ) -> None:
        try:
            self.bus.open()
            for profile in self.profiles:
                for motor in profile.config.motors:
                    self.bus.require_motor(motor)
            with self.lock:
                self._disarm_all_locked(raise_errors=True)
                self.fault = None
                self.last_event = "All 12 motors online and disarmed"
        except Exception as exc:
            with self.lock:
                self._forget_motion_state_locked()
                self._disarm_all_locked(raise_errors=False)
                self.fault = (
                    "Servo bus unavailable; automatic reconnect failed: "
                    f"{exc}"
                )
                self.last_event = (
                    "Startup fault; waiting for servo bus reconnect"
                )
            self.bus.close()
            if not allow_disconnected:
                raise
        if start_worker:
            self.worker = threading.Thread(
                target=self._worker_loop,
                name="drobot-four-leg-motion",
                daemon=True,
            )
            self.worker.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=2.0)
        if self.rl_controller is not None:
            self.rl_controller.stop()
        if self.rl_recorder is not None:
            self.rl_recorder.close()
        try:
            with self.lock:
                self._disarm_all_locked(raise_errors=False)
        finally:
            self.bus.close()

    def _worker_loop(self) -> None:
        previous_started_s = time.monotonic()
        next_deadline_s = previous_started_s + self.tick_interval_s
        while True:
            wait_s = max(0.0, next_deadline_s - time.monotonic())
            if self.stop_event.wait(wait_s):
                break
            started_s = time.monotonic()
            try:
                self.advance_once(elapsed_s=started_s - previous_started_s)
            except Exception as exc:
                with self.lock:
                    self.fault = str(exc)
                    self.last_event = "Motion fault; all motors disarmed"
                    try:
                        self._recover_bus_locked()
                    except Exception as recovery_exc:
                        self.fault = (
                            "Servo bus unavailable; automatic reconnect failed: "
                            f"{recovery_exc}"
                        )
                        self.last_event = (
                            "Motion fault; waiting for servo bus reconnect"
                        )
            previous_started_s = started_s
            period_s = (
                1.0 / self.RL_MOTOR_OUTPUT_HZ
                if self.rl_controller is not None and self.rl_controller.active
                else self.tick_interval_s
            )
            next_deadline_s += period_s
            completed_s = time.monotonic()
            if next_deadline_s <= completed_s:
                # Skip missed output slots instead of replaying a burst of stale
                # targets after a slow USB or diagnostic transaction.
                next_deadline_s = completed_s + period_s

    def advance_once(self, *, elapsed_s: float | None = None) -> None:
        with self.lock:
            self._advance_crawl_locked()

            hardcoded_crawl_active = self.crawl_active
            nominal_period_s = (
                1.0 / self.RL_MOTOR_OUTPUT_HZ
                if self.rl_controller is not None and self.rl_controller.active
                else self.tick_interval_s
            )
            if hardcoded_crawl_active:
                # A telemetry request can hold the shared bus lock long enough
                # to miss one or more 20 Hz crawl ticks.  Never compensate with
                # a larger catch-up step: the real robot must remain slower
                # than the planned one-leg-at-a-time support transition.
                motion_elapsed_s = self.tick_interval_s
                motion_rate_deg_s = min(
                    self.ramp_rate_deg_s,
                    self.dashboard.crawl.ramp_rate_deg_s,
                )
            else:
                motion_elapsed_s = (
                    nominal_period_s
                    if elapsed_s is None
                    else max(0.0, min(float(elapsed_s), 0.10))
                )
                motion_rate_deg_s = self.ramp_rate_deg_s
            step_limit = min(
                self.dashboard.bus.max_command_step_deg,
                motion_rate_deg_s * motion_elapsed_s,
            )
            if self.rl_controller is not None and self.rl_controller.active:
                step_limit = min(step_limit, 5.0)
            planned: list[tuple[LegController, Any]] = []
            for profile in self.profiles:
                controller = self.controllers[profile.number]
                for motor in profile.config.motors:
                    if motor.servo_id not in controller.armed_ids:
                        continue
                    current = controller.targets_deg[motor.name]
                    desired = self.desired_deg.get(
                        (profile.number, motor.name),
                        current,
                    )
                    delta = desired - current
                    if abs(delta) < 0.01:
                        continue
                    step = max(-step_limit, min(step_limit, delta))
                    planned.append(
                        (controller, controller.plan_command(motor, current + step))
                    )
            if not planned:
                return
            batch_writer = getattr(self.bus, "write_position_commands", None)
            if batch_writer is None:
                for controller, target in planned:
                    controller.command(target.motor, target.degrees)
                return
            batch_writer(
                {
                    target.motor.servo_id: target.raw_position
                    for _controller, target in planned
                },
                self.dashboard.bus,
            )
            for controller, target in planned:
                controller.commit_command(target)

    @property
    def crawl_active(self) -> bool:
        return self.crawl_stage in {
            "positioning",
            "preparing",
            "preloading",
            "walking",
            "stopping",
        }

    def _crawl_pose_locked(
        self,
        gait_time_s: float,
    ) -> tuple[dict[tuple[int, str], float], dict[str, object]]:
        config = self.dashboard.crawl
        if self.crawl_mode == "diagonal_pair":
            pose, state = diagonal_pair_gait_degrees(
                gait_time_s,
                period_s=config.period_s,
                stride_m=config.stride_m,
                lift_m=config.lift_m,
                support_extension_m=config.support_extension_m,
                down_m=config.stance_down_m,
                fore_aft_m=config.stance_fore_aft_m,
                abduction_deg=config.abduction_deg,
            )
        else:
            pose, state = distributed_push_crawl_degrees(
                gait_time_s,
                period_s=config.period_s,
                stride_m=config.stride_m,
                lift_m=config.lift_m,
                support_extension_m=config.support_extension_m,
                weight_shift_forward_m=config.weight_shift_forward_m,
                weight_shift_lateral_m=config.weight_shift_lateral_m,
                down_m=config.stance_down_m,
                fore_aft_m=config.stance_fore_aft_m,
                abduction_deg=config.abduction_deg,
            )
        targets: dict[tuple[int, str], float] = {}
        for profile in self.profiles:
            for motor in profile.config.motors:
                degrees = pose[(profile.corner, motor.name)]
                degrees_to_raw(
                    degrees,
                    motor,
                    profile.calibration.motor(motor),
                )
                targets[(profile.number, motor.name)] = degrees
        return targets, state

    def _crawl_stance_targets_locked(self) -> dict[tuple[int, str], float]:
        targets, _state = self._crawl_pose_locked(0.0)
        return targets

    def _set_crawl_stance_locked(self, phase: str) -> None:
        self.desired_deg.update(self._crawl_stance_targets_locked())
        self.crawl_phase = phase
        self.crawl_swing_corner = None
        self.crawl_swing_pair = None
        self.crawl_push_partner = None

    def _set_crawl_pose_locked(self, gait_time_s: float) -> None:
        targets, state = self._crawl_pose_locked(gait_time_s)
        self.desired_deg.update(targets)
        self.crawl_phase = str(state["phase"])
        swing_corner = state["swing_corner"]
        self.crawl_swing_corner = None if swing_corner is None else str(swing_corner)
        swing_pair = state.get("swing_pair")
        self.crawl_swing_pair = (
            None
            if not swing_pair
            else tuple(str(corner) for corner in swing_pair)
        )
        push_partner = state.get("push_partner")
        self.crawl_push_partner = None if push_partner is None else str(push_partner)

    def _crawl_targets_reached_locked(self, tolerance_deg: float = 0.3) -> bool:
        for profile in self.profiles:
            controller = self.controllers[profile.number]
            for motor in profile.config.motors:
                commanded = controller.targets_deg.get(motor.name)
                desired = self.desired_deg.get((profile.number, motor.name))
                if commanded is None or desired is None:
                    return False
                if abs(commanded - desired) > tolerance_deg:
                    return False
        return True

    def _crawl_stance_ready_locked(self, tolerance_deg: float = 0.3) -> bool:
        if self.crawl_stage != "holding" or self._armed_count_locked() != 12:
            return False
        stance_targets = self._crawl_stance_targets_locked()
        for profile in self.profiles:
            controller = self.controllers[profile.number]
            for motor in profile.config.motors:
                key = (profile.number, motor.name)
                stance = stance_targets[key]
                desired = self.desired_deg.get(key)
                commanded = controller.targets_deg.get(motor.name)
                if desired is None or commanded is None:
                    return False
                if abs(desired - stance) > tolerance_deg:
                    return False
                if abs(commanded - stance) > tolerance_deg:
                    return False
        return True

    def _advance_crawl_locked(self) -> None:
        if not self.crawl_active:
            return
        now = self.clock()
        config = self.dashboard.crawl
        if self.crawl_stage in {"positioning", "stopping"}:
            positioning_only = self.crawl_stage == "positioning"
            self.crawl_mode = "distributed"
            self._set_crawl_stance_locked(
                "positioning_four_foot_stance"
                if positioning_only
                else "returning_to_four_foot_stance"
            )
            if not self._crawl_targets_reached_locked():
                self.crawl_target_reached_at = None
                return
            if self.crawl_target_reached_at is None:
                self.crawl_target_reached_at = now
                return
            if now - self.crawl_target_reached_at < config.stance_settle_s:
                return
            self.crawl_stage = "holding"
            self.crawl_phase = "stable_four_foot_hold"
            self.crawl_started_at = None
            self.crawl_elapsed_s = 0.0
            self.crawl_target_reached_at = None
            self.crawl_progress = 0.0
            self.last_event = (
                "Stable four-foot gait stance ready; torque remains armed"
                if positioning_only
                else "Crawl stopped at stable four-foot stance; torque remains armed"
            )
            return

        if self.crawl_stage == "preparing":
            self._set_crawl_stance_locked("settling_gait_start_stance")
            if not self._crawl_targets_reached_locked():
                self.crawl_target_reached_at = None
                return
            if self.crawl_target_reached_at is None:
                self.crawl_target_reached_at = now
                return
            if now - self.crawl_target_reached_at < config.stance_settle_s:
                return
            self.crawl_stage = "preloading"
            self.crawl_target_reached_at = None
            self.crawl_progress = 0.0
            self.last_event = (
                "Gait start stance settled; preparing diagonal-pair gait"
                if self.crawl_mode == "diagonal_pair"
                else "Gait start stance settled; preparing distributed push crawl"
            )
            return

        if self.crawl_stage == "preloading":
            self._set_crawl_pose_locked(0.0)
            self.crawl_phase = "four_feet_down_preload"
            self.crawl_swing_corner = None
            self.crawl_swing_pair = None
            self.crawl_push_partner = None
            if not self._crawl_targets_reached_locked():
                self.crawl_target_reached_at = None
                return
            if self.crawl_target_reached_at is None:
                self.crawl_target_reached_at = now
                return
            if now - self.crawl_target_reached_at < config.stance_settle_s:
                return
            self.crawl_stage = "walking"
            self.crawl_started_at = now
            self.crawl_elapsed_s = 0.0
            self.crawl_target_reached_at = None
            self.last_event = (
                "Diagonal-pair gait started; front-left and rear-right lift first"
                if self.crawl_mode == "diagonal_pair"
                else "Distributed push crawl started; rear-right swings first"
            )
            return

        if self.crawl_stage == "walking":
            if self.crawl_started_at is None:
                raise RuntimeError("Crawl clock was not initialized")
            # Advance one trajectory tick per completed controller iteration.
            # Wall-clock catch-up can otherwise skip contact phases after a
            # slow USB/telemetry transaction and begin moving the next leg
            # before the previous one has completed its plant and settle.
            self.crawl_elapsed_s += self.tick_interval_s
            elapsed = self.crawl_elapsed_s
            self.crawl_progress = (elapsed % config.period_s) / config.period_s
            self._set_crawl_pose_locked(elapsed)
            return

    def _cancel_crawl_locked(self) -> None:
        self.crawl_stage = "idle"
        self.crawl_phase = "idle"
        self.crawl_swing_corner = None
        self.crawl_swing_pair = None
        self.crawl_push_partner = None
        self.crawl_started_at = None
        self.crawl_elapsed_s = 0.0
        self.crawl_target_reached_at = None
        self.crawl_progress = 0.0

    def _require_manual_control_locked(self) -> None:
        if self.crawl_active:
            raise RuntimeError("Stop the active crawl before manual motion")
        if self.rl_controller is not None and self.rl_controller.active:
            raise RuntimeError("Stop the active RL walking test before manual motion")

    def _rl_snapshot_locked(self) -> dict[str, Any]:
        if self.rl_controller is None:
            return {
                "available": False,
                "active": False,
                "status": "unavailable",
                "error": "RL policy runtime is not configured",
                "motor_output_enabled": False,
                "targets": [],
                "imu": None,
                "temperature_verification": [],
            }
        snapshot = self.rl_controller.snapshot()
        now = self.clock()
        confirmation_s = self.dashboard.monitoring.temperature_stop_confirmation_s
        snapshot["temperature_verification"] = [
            {
                "motor_id": servo_id,
                "temperature_c": temperature_c,
                "high_sample_count": sample_count,
                "elapsed_s": min(confirmation_s, max(0.0, now - started_at)),
                "required_s": confirmation_s,
            }
            for servo_id, (started_at, sample_count, temperature_c) in sorted(
                self.rl_temperature_candidates.items()
            )
        ]
        return snapshot

    def _check_rl_temperature_locked(
        self,
        servo_id: int,
        temperature_c: int,
        now: float,
    ) -> None:
        monitoring = self.dashboard.monitoring
        if temperature_c < monitoring.temperature_warning_c:
            cleared = self.rl_temperature_candidates.pop(servo_id, None)
            if cleared is not None:
                self.last_event = (
                    f"RL temperature verification cleared on ID {servo_id}: "
                    f"{temperature_c} C"
                )
            return
        if temperature_c >= monitoring.temperature_critical_c:
            self.rl_temperature_candidates.pop(servo_id, None)
            raise RuntimeError(
                f"RL critical temperature stop on ID {servo_id}: "
                f"{temperature_c} C"
            )

        candidate = self.rl_temperature_candidates.get(servo_id)
        if candidate is None:
            started_at = now
            sample_count = 1
        else:
            started_at, sample_count, _previous_temperature_c = candidate
            sample_count += 1
        self.rl_temperature_candidates[servo_id] = (
            started_at,
            sample_count,
            temperature_c,
        )
        elapsed_s = max(0.0, now - started_at)
        required_s = monitoring.temperature_stop_confirmation_s
        if elapsed_s >= required_s and sample_count >= 3:
            raise RuntimeError(
                f"RL persistent temperature stop on ID {servo_id}: "
                f"{temperature_c} C after {elapsed_s:.1f} seconds and "
                f"{sample_count} high readings"
            )
        self.last_event = (
            f"Verifying RL temperature on ID {servo_id}: {temperature_c} C, "
            f"{elapsed_s:.1f}/{required_s:.1f} seconds, "
            f"{sample_count} high reading{'s' if sample_count != 1 else ''}"
        )

    def start_rl_policy(
        self,
        *,
        forward_m_s: float,
        duration_s: float,
        safety_ack: bool,
        confirmation: str,
    ) -> None:
        if self.rl_controller is None:
            raise RuntimeError("RL policy runtime is not configured")
        if not safety_ack:
            raise ValueError(
                "Confirm the robot is supported, feet are clear, and cutoff is ready"
            )
        if confirmation != "START SUPPORTED RL TEST":
            raise ValueError("START SUPPORTED RL TEST confirmation is required")

        speed, duration = self.rl_controller.validate_request(
            forward_m_s,
            duration_s,
        )
        self.rl_controller.prepare()
        with self.lock:
            if self.crawl_active or self.rl_controller.active:
                raise RuntimeError("Stop the active gait before starting an RL test")
            if self._armed_count_locked():
                raise RuntimeError("Disarm all 12 motors before starting an RL test")
            if self.fault:
                raise RuntimeError("Clear the reported hardware fault before RL start")
            if self.last_full_snapshot is None:
                self._snapshot_locked()

            measured_by_id: dict[int, float] = {}
            try:
                for profile in self.profiles:
                    controller = self.controllers[profile.number]
                    for motor in profile.config.motors:
                        state = controller.arm(motor)
                        measured_by_id[motor.servo_id] = state.degrees
                        self.desired_deg[(profile.number, motor.name)] = state.degrees
                import numpy as np

                initial = np.asarray(
                    [
                        math.radians(measured_by_id[motor.servo_id])
                        for _profile, motor, _controller in self.rl_motor_order
                    ],
                    dtype=np.float32,
                )
                self.rl_feedback_position_rad = initial.copy()
                self.rl_feedback_velocity_rad_s = np.zeros(12, dtype=np.float32)
                self.rl_feedback_time_s = self.clock()
                self.rl_diagnostic_index = 0
                self.rl_diagnostic_time_s = None
                self.rl_temperature_candidates.clear()
                self.rl_controller.start(
                    speed,
                    duration,
                    initial,
                )
                self.rl_controller.record_event(
                    "rl_started",
                    {
                        "forward_m_s": speed,
                        "duration_s": duration,
                    },
                )
            except Exception:
                self._disarm_all_locked(raise_errors=False)
                raise

            self.heartbeat("rl-policy-local")
            self.fault = None
            self.last_event = (
                f"Supported {duration:g}-second RL walking test started "
                f"at {speed:.3f} m/s; normal completion will return "
                "to calibrated center"
            )

    def stop_rl_policy(self) -> None:
        if self.rl_controller is None:
            raise RuntimeError("RL policy runtime is not configured")
        self.rl_controller.request_stop()
        with self.lock:
            self._disarm_all_locked(raise_errors=True)
            self.last_event = "RL walking test stopped; all 12 motors disarmed"
        self.rl_controller.stop()

    def _read_rl_joint_state(self) -> Any:
        import numpy as np
        from drobot_policy_runtime.sources import JointStateSample

        with self.lock:
            if self.rl_controller is None or not self.rl_controller.active:
                raise RuntimeError("RL joint feedback requested while policy is inactive")
            if self._armed_count_locked() != 12:
                raise RuntimeError("RL policy requires all 12 motors armed")
            now = self.clock()
            monitoring = self.dashboard.monitoring
            diagnostic_index: int | None = None
            if (
                self.rl_diagnostic_time_s is None
                or now - self.rl_diagnostic_time_s >= 0.10
            ):
                pending_indices = [
                    index
                    for index, (_profile, motor, _controller) in enumerate(
                        self.rl_motor_order
                    )
                    if motor.servo_id in self.rl_temperature_candidates
                ]
                if pending_indices:
                    diagnostic_index = pending_indices[
                        self.rl_diagnostic_index % len(pending_indices)
                    ]
                else:
                    diagnostic_index = self.rl_diagnostic_index % 12
                self.rl_diagnostic_index = (self.rl_diagnostic_index + 1) % 12
                self.rl_diagnostic_time_s = now

            servo_ids = [
                motor.servo_id for _profile, motor, _controller in self.rl_motor_order
            ]
            group_reader = getattr(self.bus, "read_positions_speeds", None)
            if group_reader is None:
                raw_by_id = {
                    servo_id: self.bus.read_position_speed(servo_id)
                    for servo_id in servo_ids
                }
            else:
                raw_by_id = group_reader(servo_ids)

            diagnostic_status = None
            if diagnostic_index is not None:
                diagnostic_status = self.bus.status(
                    self.rl_motor_order[diagnostic_index][1]
                )

            measured_rad: list[float] = []
            for index, (profile, motor, _controller) in enumerate(
                self.rl_motor_order
            ):
                raw_position, raw_speed = raw_by_id[motor.servo_id]
                status = diagnostic_status if diagnostic_index == index else None
                if status is not None:
                    raw_position = status.raw_position
                    raw_speed = status.raw_speed
                degrees = raw_to_degrees(
                    raw_position,
                    motor,
                    profile.calibration.motor(motor),
                )
                if status is not None:
                    if not status.torque_enabled:
                        raise RuntimeError(
                            f"RL telemetry says motor ID {motor.servo_id} lost torque"
                        )
                    self._check_rl_temperature_locked(
                        motor.servo_id,
                        status.temperature_c,
                        now,
                    )
                    if status.voltage_v < monitoring.voltage_warning_low_v - 0.5:
                        raise RuntimeError(
                            f"RL low-voltage stop on ID {motor.servo_id}: "
                            f"{status.voltage_v:.1f} V"
                        )
                    target = self.desired_deg[(profile.number, motor.name)]
                    tracking_error = abs(target - degrees)
                    if (
                        status.current_ma
                        >= monitoring.motor_stall_current_warning_ma
                        and abs(status.raw_speed) <= monitoring.stall_speed_raw_max
                        and tracking_error >= monitoring.stall_tracking_error_deg
                    ):
                        raise RuntimeError(
                            f"RL possible-stall stop on motor ID {motor.servo_id}"
                        )
                    if self.rl_controller is not None:
                        self.rl_controller.record_diagnostic(
                            {
                                "servo_id": motor.servo_id,
                                "leg_number": profile.number,
                                "joint": motor.name,
                                "position_deg": degrees,
                                "target_deg": target,
                                "tracking_error_deg": tracking_error,
                                "raw_position": status.raw_position,
                                "raw_speed": status.raw_speed,
                                "voltage_v": status.voltage_v,
                                "temperature_c": status.temperature_c,
                                "current_ma": status.current_ma,
                                "torque_enabled": status.torque_enabled,
                                "feedback_transport": getattr(
                                    self.bus,
                                    "feedback_mode",
                                    "sequential",
                                ),
                            }
                        )
                measured_rad.append(math.radians(degrees))
            positions = np.asarray(measured_rad, dtype=np.float32)
            sampled_s = self.clock()
            previous = self.rl_feedback_position_rad
            previous_time = self.rl_feedback_time_s
            if (
                previous is None
                or previous_time is None
                or sampled_s <= previous_time
            ):
                velocity = np.zeros(12, dtype=np.float32)
            else:
                velocity = np.clip(
                    (positions - previous) / (sampled_s - previous_time),
                    -4.5836625,
                    4.5836625,
                ).astype(np.float32)
            self.rl_feedback_position_rad = positions
            self.rl_feedback_velocity_rad_s = velocity
            self.rl_feedback_time_s = sampled_s
            return JointStateSample(
                position_rad=positions.copy(),
                velocity_rad_s=velocity.copy(),
                monotonic_time_s=sampled_s,
            )

    def _apply_rl_targets(
        self,
        action: Any,
        joint_target_rad: Any,
        _monotonic_time_s: float,
    ) -> None:
        if len(action) != 12 or len(joint_target_rad) != 12:
            raise RuntimeError("RL policy must produce exactly 12 targets")
        with self.lock:
            if self.rl_controller is None or not self.rl_controller.active:
                raise RuntimeError("RL target received while policy is inactive")
            if self._armed_count_locked() != 12:
                raise RuntimeError("RL target rejected because not all motors are armed")
            updates: dict[tuple[int, str], float] = {}
            for index, (profile, motor, _controller) in enumerate(self.rl_motor_order):
                degrees = math.degrees(float(joint_target_rad[index]))
                if not math.isfinite(degrees):
                    raise RuntimeError("RL policy produced a non-finite target")
                degrees_to_raw(degrees, motor, profile.calibration.motor(motor))
                previous = self.desired_deg[(profile.number, motor.name)]
                if abs(degrees - previous) > 5.0:
                    raise RuntimeError(
                        f"RL target step for ID {motor.servo_id} exceeds 5 degrees"
                    )
                updates[(profile.number, motor.name)] = degrees
            self.desired_deg.update(updates)

    def _finish_rl_policy(self, error: str | None, stopped: bool) -> str | None:
        with self.lock:
            if self.rl_controller is not None:
                self.rl_controller.record_event(
                    "rl_finalizing",
                    {"error": error, "stopped_by_operator": stopped},
                )
            result = error
            self.rl_temperature_candidates.clear()
            if error:
                self._disarm_all_locked(raise_errors=False)
                self.last_event = f"RL policy fault; all motors disarmed: {error}"
            elif stopped:
                self._disarm_all_locked(raise_errors=False)
                self.last_event = "RL walking test stopped; all 12 motors disarmed"
            else:
                try:
                    self._set_center_all_targets_locked(arm_missing=False)
                except Exception as exc:
                    self._disarm_all_locked(raise_errors=False)
                    center_error = f"RL completion center-all failed: {exc}"
                    self.last_event = f"{center_error}; all motors disarmed"
                    result = center_error
                else:
                    self.heartbeat("rl-complete-center")
                    self.fault = None
                    self.last_event = (
                        "RL policy test complete; all 12 motors returning to "
                        "calibrated center with torque holding"
                    )
            self.rl_feedback_position_rad = None
            self.rl_feedback_velocity_rad_s = None
            self.rl_feedback_time_s = None
            return result

    def list_recordings(self) -> list[dict[str, Any]]:
        if self.rl_recorder is None:
            return []
        return self.rl_recorder.list_recordings()

    def rename_recording(self, recording_id: str, label: str) -> None:
        if self.rl_recorder is None:
            raise RuntimeError("RL recording is not configured")
        self.rl_recorder.rename(recording_id, label)

    def delete_recording(self, recording_id: str) -> None:
        if self.rl_recorder is None:
            raise RuntimeError("RL recording is not configured")
        self.rl_recorder.delete(recording_id)

    def recording_archive(self, recording_id: str) -> Path:
        if self.rl_recorder is None:
            raise RuntimeError("RL recording is not configured")
        return self.rl_recorder.archive_path(recording_id)

    def start_crawl_forward(
        self,
        *,
        safety_ack: bool,
        confirmation: str,
    ) -> None:
        if not safety_ack:
            raise ValueError(
                "Confirm the body is supported with every foot clear before testing"
            )
        if confirmation != "TEST DISTRIBUTED CRAWL":
            raise ValueError("TEST DISTRIBUTED CRAWL confirmation is required")

        self._start_crawl_mode("distributed")

    def start_diagonal_pair_forward(
        self,
        *,
        safety_ack: bool,
        confirmation: str,
    ) -> None:
        if not safety_ack:
            raise ValueError(
                "Confirm the body is supported with every foot clear before testing"
            )
        if confirmation != "TEST DIAGONAL PAIR GAIT":
            raise ValueError("TEST DIAGONAL PAIR GAIT confirmation is required")

        self._start_crawl_mode("diagonal_pair")

    def _start_crawl_mode(self, mode: str) -> None:
        if mode not in {"distributed", "diagonal_pair"}:
            raise ValueError(f"Unknown crawl mode: {mode}")

        with self.lock:
            if self.crawl_active:
                raise RuntimeError("A crawl is already active")
            armed_count = self._armed_count_locked()
            start_from_held_stance = (
                mode == "distributed"
                and self._crawl_stance_ready_locked()
            )
            if armed_count and not start_from_held_stance:
                raise RuntimeError(
                    "Start from the settled gait stance or disarm all 12 motors"
                )
            if self.fault:
                raise RuntimeError("Clear the reported fault before starting a crawl")

            previous_mode = self.crawl_mode
            newly_armed: list[tuple[LegProfile, MotorConfig]] = []
            try:
                self.crawl_mode = mode
                sample_count = 80
                duration_s = self.dashboard.crawl.period_s
                self._crawl_stance_targets_locked()
                for sample in range(sample_count + 1):
                    self._crawl_pose_locked(duration_s * sample / sample_count)

                if not start_from_held_stance:
                    for profile in self.profiles:
                        controller = self.controllers[profile.number]
                        for motor in profile.config.motors:
                            state = controller.arm(motor)
                            newly_armed.append((profile, motor))
                            self.desired_deg[(profile.number, motor.name)] = state.degrees
                self.crawl_stage = (
                    "preloading" if start_from_held_stance else "preparing"
                )
                self.crawl_phase = (
                    "four_feet_down_preload"
                    if start_from_held_stance
                    else "moving_to_stance"
                )
                self.crawl_swing_corner = None
                self.crawl_swing_pair = None
                self.crawl_push_partner = None
                self.crawl_started_at = None
                self.crawl_target_reached_at = None
                self.crawl_progress = 0.0
                self._set_crawl_stance_locked(
                    "four_feet_down_preload"
                    if start_from_held_stance
                    else "moving_to_gait_start_stance"
                )
            except Exception:
                for profile, motor in newly_armed:
                    try:
                        self.controllers[profile.number].disarm(motor)
                    except Exception:
                        pass
                self._disarm_all_locked(raise_errors=False)
                self.crawl_mode = previous_mode
                raise

            self.heartbeat("control-command")
            self.fault = None
            self.last_event = (
                "All 12 motors moving to the diagonal-pair start stance"
                if mode == "diagonal_pair"
                else (
                    "Distributed crawl preloading from settled four-foot stance"
                    if start_from_held_stance
                    else "All 12 motors moving to the distributed-push start stance"
                )
            )

    def stop_crawl(self) -> None:
        with self.lock:
            rl_active = self.rl_controller is not None and self.rl_controller.active
            if rl_active:
                self._disarm_all_locked(raise_errors=True)
                self.last_event = "RL walking stopped; all 12 motors disarmed"
                return
            if not self._armed_count_locked():
                self._cancel_crawl_locked()
                self.last_event = "Crawl stopped; all 12 motors already disarmed"
                return
            self.crawl_mode = "distributed"
            self.crawl_stage = "stopping"
            self.crawl_started_at = None
            self.crawl_elapsed_s = 0.0
            self.crawl_target_reached_at = None
            self.crawl_progress = 0.0
            self._set_crawl_stance_locked("returning_to_four_foot_stance")
            self.last_event = (
                "Crawl stopping; returning slowly to a stable four-foot stance"
            )

    def heartbeat(self, source: str | None = None) -> None:
        with self.heartbeat_lock:
            self.last_heartbeat = self.clock()
            self.heartbeat_count += 1
            self.last_heartbeat_source = source

    def _browser_heartbeat_snapshot(self) -> dict[str, Any]:
        with self.heartbeat_lock:
            age_s = max(0.0, self.clock() - self.last_heartbeat)
            count = self.heartbeat_count
            source = self.last_heartbeat_source
        return {
            "age_s": age_s,
            "recent": age_s <= self.heartbeat_timeout_s,
            "attention_after_s": self.heartbeat_timeout_s,
            "received_count": count,
            "source": source,
            "warning_only": True,
            "controls_motion": False,
        }

    def arm(
        self,
        leg_number: int | str,
        motor_selector: int | str,
        *,
        safety_ack: bool,
    ) -> None:
        if not safety_ack:
            raise ValueError("Confirm support, clearance, and cutoff before arming")
        profile, motor, controller = self._selection(leg_number, motor_selector)
        with self.lock:
            self._require_manual_control_locked()
            state = controller.arm(motor)
            self.desired_deg[(profile.number, motor.name)] = state.degrees
            self.heartbeat("control-command")
            self.fault = None
            self.last_event = (
                f"{profile.label} / {motor.name.replace('_', ' ')} armed "
                f"at {state.degrees:+.2f} deg"
            )

    def arm_leg(self, leg_number: int | str, *, safety_ack: bool) -> None:
        if not safety_ack:
            raise ValueError("Confirm support, clearance, and cutoff before arming")
        profile = self._profile(leg_number)
        controller = self.controllers[profile.number]
        newly_armed: list[MotorConfig] = []
        with self.lock:
            self._require_manual_control_locked()
            try:
                for motor in profile.config.motors:
                    if motor.servo_id in controller.armed_ids:
                        continue
                    state = controller.arm(motor)
                    newly_armed.append(motor)
                    self.desired_deg[(profile.number, motor.name)] = state.degrees
            except Exception:
                for motor in newly_armed:
                    try:
                        controller.disarm(motor)
                    except Exception:
                        pass
                    self.desired_deg.pop((profile.number, motor.name), None)
                raise
            self.heartbeat("control-command")
            self.fault = None
            self.last_event = f"{profile.label} armed at measured positions"

    def set_target(
        self,
        leg_number: int | str,
        motor_selector: int | str,
        degrees: float,
    ) -> None:
        if not math.isfinite(degrees):
            raise ValueError("Target angle must be finite")
        profile, motor, controller = self._selection(leg_number, motor_selector)
        with self.lock:
            self._require_manual_control_locked()
            if motor.servo_id not in controller.armed_ids:
                raise RuntimeError(f"{profile.label} / {motor.name} is disarmed")
            degrees_to_raw(
                degrees,
                motor,
                profile.calibration.motor(motor),
            )
            self.desired_deg[(profile.number, motor.name)] = degrees
            self.heartbeat("control-command")
            self.last_event = (
                f"{profile.label} / {motor.name.replace('_', ' ')} "
                f"destination {degrees:+.2f} deg"
            )

    def zero_armed_leg(self, leg_number: int | str) -> None:
        profile = self._profile(leg_number)
        controller = self.controllers[profile.number]
        with self.lock:
            self._require_manual_control_locked()
            armed = 0
            for motor in profile.config.motors:
                if motor.servo_id not in controller.armed_ids:
                    continue
                degrees_to_raw(0.0, motor, profile.calibration.motor(motor))
                self.desired_deg[(profile.number, motor.name)] = 0.0
                armed += 1
            if not armed:
                raise RuntimeError(f"{profile.label} has no armed motors")
            self.heartbeat("control-command")
            self.last_event = f"{profile.label} armed motors returning to zero"

    def zero_all_armed(self) -> None:
        with self.lock:
            self._require_manual_control_locked()
            armed = 0
            for profile in self.profiles:
                controller = self.controllers[profile.number]
                for motor in profile.config.motors:
                    if motor.servo_id not in controller.armed_ids:
                        continue
                    degrees_to_raw(0.0, motor, profile.calibration.motor(motor))
                    self.desired_deg[(profile.number, motor.name)] = 0.0
                    armed += 1
            if not armed:
                raise RuntimeError("No motors are armed")
            self.heartbeat("control-command")
            self.last_event = "All armed motors returning to zero"

    def _set_center_all_targets_locked(self, *, arm_missing: bool) -> None:
        for profile in self.profiles:
            controller = self.controllers[profile.number]
            for motor in profile.config.motors:
                degrees_to_raw(
                    0.0,
                    motor,
                    profile.calibration.motor(motor),
                )
                if motor.servo_id not in controller.armed_ids:
                    if not arm_missing:
                        raise RuntimeError(
                            f"motor ID {motor.servo_id} is not armed for center hold"
                        )
                    state = controller.arm(motor)
                    self.desired_deg[(profile.number, motor.name)] = state.degrees
        for profile in self.profiles:
            for motor in profile.config.motors:
                self.desired_deg[(profile.number, motor.name)] = 0.0

    def center_all(
        self,
        *,
        safety_ack: bool,
        confirmation: str,
    ) -> None:
        if not safety_ack:
            raise ValueError("Confirm support, clearance, and cutoff before centering")
        if confirmation != "CENTER ALL 12":
            raise ValueError("CENTER ALL 12 confirmation is required")

        with self.lock:
            self._require_manual_control_locked()
            try:
                self._set_center_all_targets_locked(arm_missing=True)
            except Exception:
                self._disarm_all_locked(raise_errors=False)
                self.last_event = "Center-all failed; all motors disarmed"
                raise

            self.heartbeat("control-command")
            self.fault = None
            self.last_event = "All 12 motors returning to calibrated zero"

    def set_crawl_stance(
        self,
        *,
        safety_ack: bool,
        confirmation: str,
    ) -> None:
        if not safety_ack:
            raise ValueError("Confirm full support, clearance, and cutoff")
        if confirmation != "SET GAIT START STANCE":
            raise ValueError("SET GAIT START STANCE confirmation is required")

        with self.lock:
            self._require_manual_control_locked()
            self.crawl_mode = "distributed"
            targets = self._crawl_stance_targets_locked()
            try:
                for profile in self.profiles:
                    controller = self.controllers[profile.number]
                    for motor in profile.config.motors:
                        if motor.servo_id not in controller.armed_ids:
                            state = controller.arm(motor)
                            self.desired_deg[(profile.number, motor.name)] = (
                                state.degrees
                            )
                self.desired_deg.update(targets)
                self.crawl_stage = "positioning"
                self.crawl_phase = "positioning_four_foot_stance"
                self.crawl_started_at = None
                self.crawl_elapsed_s = 0.0
                self.crawl_target_reached_at = None
                self.crawl_progress = 0.0
            except Exception:
                self._disarm_all_locked(raise_errors=False)
                self.last_event = "Walk-stance command failed; all motors disarmed"
                raise

            self.heartbeat("control-command")
            self.fault = None
            self.last_event = (
                "All 12 motors moving to wide walk stance; encoder seams use "
                "signed extended positions"
            )

    def capture_zero_all(
        self,
        *,
        safety_ack: bool,
        confirmation: str,
    ) -> None:
        if not safety_ack:
            raise ValueError("Confirm support, clearance, and cutoff before capture")
        if confirmation != "CAPTURE ZERO ALL":
            raise ValueError("CAPTURE ZERO ALL confirmation is required")

        with self.lock:
            self._require_manual_control_locked()
            if self._armed_count_locked():
                raise RuntimeError("Disarm all motors before capturing zero")

            centers_by_leg: dict[int, dict[str, int]] = {}
            for profile in self.profiles:
                centers: dict[str, int] = {}
                for motor in profile.config.motors:
                    status = self.bus.status(motor)
                    if status.torque_enabled:
                        raise RuntimeError(
                            f"Motor ID {motor.servo_id} still reports torque enabled"
                        )
                    centers[motor.name] = status.raw_position
                centers_by_leg[profile.number] = centers

            calibrations = {
                profile.number: calibration_from_centers(
                    profile.config,
                    centers_by_leg[profile.number],
                )
                for profile in self.profiles
            }

            if self.persist_calibration:
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
                backup_dir = (
                    self.profiles[0].calibration_path.parent / "backups" / stamp
                )
                backup_dir.mkdir(parents=True, exist_ok=False)
                backups: dict[Path, Path] = {}
                try:
                    for profile in self.profiles:
                        backup_path = backup_dir / profile.calibration_path.name
                        shutil.copy2(profile.calibration_path, backup_path)
                        backups[profile.calibration_path] = backup_path
                    for profile in self.profiles:
                        save_calibration(
                            calibrations[profile.number],
                            profile.calibration_path,
                        )
                        load_calibration(profile.calibration_path, profile.config)
                except Exception:
                    for calibration_path, backup_path in backups.items():
                        shutil.copy2(backup_path, calibration_path)
                    raise

            new_profiles = tuple(
                replace(profile, calibration=calibrations[profile.number])
                for profile in self.profiles
            )
            self.dashboard = replace(self.dashboard, legs=new_profiles)
            for profile in new_profiles:
                self.controllers[profile.number].calibration = profile.calibration
            self.desired_deg.clear()
            self.fault = None
            mode = "Demo captured" if not self.persist_calibration else "Saved"
            self.last_event = f"{mode} current pose as calibrated zero for all 12"

    def disarm(
        self,
        leg_number: int | str,
        motor_selector: int | str,
    ) -> None:
        profile, motor, controller = self._selection(leg_number, motor_selector)
        with self.lock:
            if self.crawl_active or (
                self.rl_controller is not None and self.rl_controller.active
            ):
                self._disarm_all_locked(raise_errors=True)
                self.last_event = "Autonomous motion stopped; all motors disarmed"
                return
            if motor.servo_id in controller.armed_ids:
                controller.disarm(motor)
            else:
                self.bus.disable_torque(motor.servo_id)
            self.desired_deg.pop((profile.number, motor.name), None)
            self.last_event = (
                f"{profile.label} / {motor.name.replace('_', ' ')} disarmed"
            )

    def disarm_leg(self, leg_number: int | str) -> None:
        profile = self._profile(leg_number)
        controller = self.controllers[profile.number]
        with self.lock:
            if self.crawl_active or (
                self.rl_controller is not None and self.rl_controller.active
            ):
                self._disarm_all_locked(raise_errors=True)
                self.last_event = "Autonomous motion stopped; all motors disarmed"
                return
            errors: list[Exception] = []
            for motor in profile.config.motors:
                try:
                    self.bus.disable_torque(motor.servo_id)
                except Exception as exc:
                    errors.append(exc)
                controller.armed_ids.discard(motor.servo_id)
                controller.targets_deg.pop(motor.name, None)
                self.desired_deg.pop((profile.number, motor.name), None)
            self.last_event = f"{profile.label} disarmed"
            if errors:
                raise RuntimeError(
                    "One or more motors could not be disarmed: "
                    + "; ".join(str(error) for error in errors)
                )

    def disarm_all(self) -> None:
        with self.lock:
            self._disarm_all_locked(raise_errors=True)
            self.last_event = "All 12 motors disarmed"

    def _disarm_all_locked(self, *, raise_errors: bool) -> None:
        self._cancel_crawl_locked()
        if self.rl_controller is not None:
            self.rl_controller.request_stop()
        errors: list[Exception] = []
        for profile in self.profiles:
            controller = self.controllers[profile.number]
            for motor in profile.config.motors:
                try:
                    self.bus.disable_torque(motor.servo_id)
                except Exception as exc:
                    errors.append(exc)
            controller.armed_ids.clear()
            controller.targets_deg.clear()
        self.desired_deg.clear()
        if errors and raise_errors:
            unique_errors = list(dict.fromkeys(str(error) for error in errors))
            raise RuntimeError(
                "One or more motors could not be disarmed: "
                + "; ".join(unique_errors)
            )

    def _forget_motion_state_locked(self) -> None:
        """Clear commands when the physical bus state can no longer be trusted."""

        self._cancel_crawl_locked()
        if self.rl_controller is not None:
            self.rl_controller.request_stop()
        for controller in self.controllers.values():
            controller.armed_ids.clear()
            controller.targets_deg.clear()
        self.desired_deg.clear()

    def _recover_bus_locked(self) -> None:
        """Reopen a re-enumerated adapter and return to a disarmed state."""

        self._forget_motion_state_locked()
        self.bus.reopen()
        for profile in self.profiles:
            for motor in profile.config.motors:
                self.bus.require_motor(motor)
        self._disarm_all_locked(raise_errors=True)
        self.fault = None
        self.last_event = "Servo bus reconnected; all 12 motors online and disarmed"

    def _armed_count_locked(self) -> int:
        return sum(
            len(controller.armed_ids) for controller in self.controllers.values()
        )

    def reset_power_analytics(self) -> None:
        with self.lock:
            if self._armed_count_locked():
                raise RuntimeError(
                    "Disarm all motors before resetting power analytics"
                )
            self.power_samples.clear()
            self.idle_voltage_samples.clear()
            self.power_energy_wh = 0.0
            self.power_last_sample_at = None
            self.power_last_w = 0.0
            self.last_event = "Power analytics reset; collect an idle reference"

    def _cached_rl_snapshot_locked(self) -> dict[str, Any]:
        """Serve the UI without monopolizing the serial bus during RL control."""

        if self.last_full_snapshot is None:
            raise RuntimeError("RL telemetry cache was not initialized")
        payload = copy.deepcopy(self.last_full_snapshot)
        measured_by_id: dict[int, float] = {}
        if self.rl_feedback_position_rad is not None:
            measured_by_id = {
                motor.servo_id: math.degrees(
                    float(self.rl_feedback_position_rad[index])
                )
                for index, (_profile, motor, _controller) in enumerate(
                    self.rl_motor_order
                )
            }

        profile_by_number = {profile.number: profile for profile in self.profiles}
        for leg in payload["legs"]:
            profile = profile_by_number[int(leg["number"])]
            controller = self.controllers[profile.number]
            armed_count = 0
            for motor_payload in leg["motors"]:
                motor = profile.config.motor(int(motor_payload["number"]))
                armed = motor.servo_id in controller.armed_ids
                armed_count += int(armed)
                commanded = controller.targets_deg.get(motor.name)
                desired = self.desired_deg.get((profile.number, motor.name))
                measured = measured_by_id.get(
                    motor.servo_id,
                    float(motor_payload["measured_deg"]),
                )
                reference = commanded if commanded is not None else desired
                motor_payload.update(
                    measured_deg=measured,
                    commanded_deg=commanded,
                    desired_deg=desired,
                    tracking_error_deg=(
                        abs(float(reference) - measured)
                        if reference is not None
                        else 0.0
                    ),
                    torque_enabled=armed,
                    armed=armed,
                )
            leg["armed_count"] = armed_count

        armed_count = self._armed_count_locked()
        payload["summary"]["armed_count"] = armed_count
        payload["summary"]["telemetry_live"] = False
        payload["power"]["live"] = False
        payload["power"]["assessment"] = (
            "Electrical diagnostics are staggered onboard during the RL walk"
        )
        payload["runtime"]["telemetry_mode"] = "cached_during_rl"
        payload["rl_policy"] = self._rl_snapshot_locked()
        payload["any_armed"] = bool(armed_count)
        payload["browser_heartbeat"] = self._browser_heartbeat_snapshot()
        payload["last_event"] = self.last_event
        payload["fault"] = self.fault
        return payload

    def _snapshot_locked(self) -> dict[str, Any]:
        if (
            self.rl_controller is not None
            and self.rl_controller.active
            and self.last_full_snapshot is not None
        ):
            return self._cached_rl_snapshot_locked()
        browser_heartbeat = self._browser_heartbeat_snapshot()
        leg_payloads: list[dict[str, Any]] = []
        all_statuses: list[MotorStatus] = []
        unexpected_torque: list[int] = []
        current_by_leg: dict[int, float] = {}
        power_by_leg: dict[int, float] = {}
        possible_stall_ids: list[int] = []
        monitoring = self.dashboard.monitoring

        for profile in self.profiles:
            controller = self.controllers[profile.number]
            motors: list[dict[str, Any]] = []
            leg_current = 0.0
            leg_power = 0.0
            for motor in profile.config.motors:
                status = self.bus.status(motor)
                all_statuses.append(status)
                measured = raw_to_degrees(
                    status.raw_position,
                    motor,
                    profile.calibration.motor(motor),
                )
                commanded = controller.targets_deg.get(motor.name)
                desired = self.desired_deg.get((profile.number, motor.name))
                armed = motor.servo_id in controller.armed_ids
                if status.torque_enabled and not armed:
                    unexpected_torque.append(motor.servo_id)
                leg_current += status.current_ma
                motor_power_w = (
                    status.voltage_v * max(0.0, status.current_ma) / 1000.0
                )
                leg_power += motor_power_w
                tracking_reference = commanded if commanded is not None else desired
                tracking_error_deg = (
                    abs(float(tracking_reference) - measured)
                    if tracking_reference is not None
                    else 0.0
                )
                possible_stall = bool(
                    armed
                    and tracking_error_deg >= monitoring.stall_tracking_error_deg
                    and status.current_ma
                    >= monitoring.motor_stall_current_warning_ma
                    and abs(status.raw_speed) <= monitoring.stall_speed_raw_max
                )
                if possible_stall:
                    possible_stall_ids.append(motor.servo_id)
                motors.append(
                    {
                        "number": motor.number,
                        "name": motor.name,
                        "label": motor.name.replace("_", " "),
                        "positive_motion": (
                            "outward" if motor.name == "hip_abduction" else "forward"
                        ),
                        "id": motor.servo_id,
                        "direction": motor.direction,
                        "min_deg": motor.min_deg,
                        "max_deg": motor.max_deg,
                        "measured_deg": measured,
                        "commanded_deg": commanded,
                        "desired_deg": desired,
                        "raw_position": status.raw_position,
                        "speed": status.raw_speed,
                        "voltage_v": status.voltage_v,
                        "temperature_c": status.temperature_c,
                        "current_ma": status.current_ma,
                        "power_w": motor_power_w,
                        "tracking_error_deg": tracking_error_deg,
                        "possible_stall": possible_stall,
                        "torque_enabled": status.torque_enabled,
                        "armed": armed,
                        "model": status.model_number,
                    }
                )
            current_by_leg[profile.number] = leg_current
            power_by_leg[profile.number] = leg_power
            leg_payloads.append(
                {
                    "number": profile.number,
                    "label": profile.label,
                    "corner": profile.corner,
                    "current_ma": leg_current,
                    "power_w": leg_power,
                    "possible_stall_ids": [
                        motor["id"] for motor in motors if motor["possible_stall"]
                    ],
                    "armed_count": sum(motor["armed"] for motor in motors),
                    "motors": motors,
                }
            )

        voltages = [status.voltage_v for status in all_statuses]
        temperatures = [status.temperature_c for status in all_statuses]
        voltage_min = min(voltages)
        voltage_max = max(voltages)
        voltage_spread = voltage_max - voltage_min
        maximum_temperature = max(temperatures)
        total_current = sum(status.current_ma for status in all_statuses)
        total_power_w = sum(power_by_leg.values())
        bus_voltage_v = sum(voltages) / len(voltages)
        armed_count = self._armed_count_locked()
        if armed_count == 0 and not unexpected_torque and total_current <= 250.0:
            self.idle_voltage_samples.append(voltage_min)
        idle_reference_voltage_v = (
            sum(self.idle_voltage_samples) / len(self.idle_voltage_samples)
            if self.idle_voltage_samples
            else None
        )
        voltage_sag_v = (
            max(0.0, idle_reference_voltage_v - voltage_min)
            if idle_reference_voltage_v is not None
            else None
        )
        battery_per_cell_v = (
            idle_reference_voltage_v / monitoring.battery_series_cells
            if idle_reference_voltage_v is not None
            else None
        )
        if battery_per_cell_v is None:
            battery_charge_status = "unknown"
        elif battery_per_cell_v >= 4.10:
            battery_charge_status = "full"
        elif battery_per_cell_v >= 3.90:
            battery_charge_status = "good"
        elif battery_per_cell_v >= 3.70:
            battery_charge_status = "low"
        else:
            battery_charge_status = "recharge"

        sample_at = self.clock()
        if self.power_last_sample_at is not None:
            elapsed_s = sample_at - self.power_last_sample_at
            if 0.0 < elapsed_s <= 5.0:
                average_interval_power_w = (self.power_last_w + total_power_w) / 2.0
                self.power_energy_wh += average_interval_power_w * elapsed_s / 3600.0
        self.power_last_sample_at = sample_at
        self.power_last_w = total_power_w
        self.power_samples.append(
            (sample_at, total_power_w, total_current, voltage_min)
        )
        power_window_s = 60.0
        while (
            self.power_samples
            and sample_at - self.power_samples[0][0] > power_window_s
        ):
            self.power_samples.popleft()
        average_power_w = sum(sample[1] for sample in self.power_samples) / len(
            self.power_samples
        )
        peak_power_w = max(sample[1] for sample in self.power_samples)
        peak_current_ma = max(sample[2] for sample in self.power_samples)
        minimum_voltage_v = min(sample[3] for sample in self.power_samples)

        warnings: list[str] = []
        if voltage_min < monitoring.voltage_warning_low_v:
            warnings.append(
                f"Low servo voltage: {voltage_min:.1f} V is below "
                f"{monitoring.voltage_warning_low_v:.1f} V"
            )
        if voltage_max > monitoring.voltage_warning_high_v:
            warnings.append(
                f"High servo voltage: {voltage_max:.1f} V is above "
                f"{monitoring.voltage_warning_high_v:.1f} V"
            )
        if voltage_spread > monitoring.voltage_spread_warning_v:
            warnings.append(
                f"Servo voltage spread is {voltage_spread:.1f} V; inspect wiring"
            )
        if (
            voltage_sag_v is not None
            and voltage_sag_v >= monitoring.voltage_sag_warning_v
        ):
            warnings.append(
                f"Bus voltage sag is {voltage_sag_v:.1f} V from the idle reference"
            )
        if maximum_temperature >= monitoring.temperature_warning_c:
            warnings.append(
                f"Servo temperature reached {maximum_temperature} C; pause testing"
            )
        for profile in self.profiles:
            current = current_by_leg[profile.number]
            if current >= monitoring.leg_current_warning_ma:
                warnings.append(
                    f"{profile.label} diagnostic current is {current:.0f} mA"
                )
        if unexpected_torque:
            warnings.append(
                "Unexpected torque-enable state on ID(s) "
                + ", ".join(str(value) for value in unexpected_torque)
            )
        if possible_stall_ids:
            warnings.append(
                "Possible servo stall on ID(s) "
                + ", ".join(str(value) for value in possible_stall_ids)
            )
        if battery_charge_status == "recharge":
            warnings.append(
                f"{monitoring.battery_series_cells}S idle-voltage estimate says "
                "RECHARGE; verify every cell at the balance connector"
            )
        if armed_count and not browser_heartbeat["recent"]:
            warnings.append(
                "Browser heartbeat is stale at "
                f"{browser_heartbeat['age_s']:.1f} s; warning only, onboard "
                "motion continues"
            )

        sag_warning = bool(
            voltage_sag_v is not None
            and voltage_sag_v >= monitoring.voltage_sag_warning_v
        )
        if possible_stall_ids and sag_warning:
            power_assessment = "Voltage sag and servo-stall signature detected"
        elif possible_stall_ids:
            power_assessment = "Possible mechanical load or servo stall detected"
        elif sag_warning:
            power_assessment = "Battery or wiring voltage sag detected"
        elif idle_reference_voltage_v is None:
            power_assessment = "Idle reference missing; disarm and reset analytics"
        elif self.crawl_active or (
            self.rl_controller is not None and self.rl_controller.active
        ):
            power_assessment = (
                "No electrical stall signature; compare battery mass and balance"
            )
        else:
            power_assessment = "Start from idle, then observe during walking"

        crawl_elapsed_s = (
            self.crawl_elapsed_s if self.crawl_stage == "walking" else 0.0
        )

        payload = {
            "legs": leg_payloads,
            "summary": {
                "online_count": len(all_statuses),
                "armed_count": self._armed_count_locked(),
                "voltage_min_v": voltage_min,
                "voltage_max_v": voltage_max,
                "voltage_spread_v": voltage_spread,
                "total_current_ma": total_current,
                "total_power_w": total_power_w,
                "max_temperature_c": maximum_temperature,
                "health": "warning" if warnings else "nominal",
                "warnings": warnings,
            },
            "settings": {
                "baudrate": self.dashboard.bus.baudrate,
                "torque_limit": self.dashboard.bus.torque_limit,
                "speed": self.dashboard.bus.speed,
                "acceleration": self.dashboard.bus.acceleration,
                "max_command_step_deg": self.dashboard.bus.max_command_step_deg,
                "ramp_rate_deg_s": self.ramp_rate_deg_s,
                "crawl_ramp_rate_deg_s": self.dashboard.crawl.ramp_rate_deg_s,
                "heartbeat_timeout_s": self.heartbeat_timeout_s,
                "voltage_warning_low_v": monitoring.voltage_warning_low_v,
                "voltage_warning_high_v": monitoring.voltage_warning_high_v,
                "voltage_sag_warning_v": monitoring.voltage_sag_warning_v,
                "temperature_warning_c": monitoring.temperature_warning_c,
                "temperature_stop_confirmation_s": (
                    monitoring.temperature_stop_confirmation_s
                ),
                "temperature_critical_c": monitoring.temperature_critical_c,
                "leg_current_warning_ma": monitoring.leg_current_warning_ma,
                "motor_stall_current_warning_ma": (
                    monitoring.motor_stall_current_warning_ma
                ),
                "stall_tracking_error_deg": monitoring.stall_tracking_error_deg,
                "stall_speed_raw_max": monitoring.stall_speed_raw_max,
                "battery_series_cells": monitoring.battery_series_cells,
            },
            "power": {
                "instantaneous_w": total_power_w,
                "average_w_60s": average_power_w,
                "peak_w_60s": peak_power_w,
                "peak_current_a_60s": peak_current_ma / 1000.0,
                "bus_voltage_v": bus_voltage_v,
                "idle_reference_voltage_v": idle_reference_voltage_v,
                "voltage_sag_v": voltage_sag_v,
                "minimum_voltage_v_60s": minimum_voltage_v,
                "energy_wh": self.power_energy_wh,
                "possible_stall_ids": possible_stall_ids,
                "assessment": power_assessment,
                "window_s": power_window_s,
                "history": [
                    {
                        "age_s": max(0.0, sample_at - timestamp),
                        "power_w": power_w,
                        "current_a": current_ma / 1000.0,
                        "voltage_v": sample_voltage_v,
                    }
                    for timestamp, power_w, current_ma, sample_voltage_v in (
                        self.power_samples
                    )
                ],
                "battery_charge": {
                    "status": battery_charge_status,
                    "series_cells": monitoring.battery_series_cells,
                    "idle_pack_voltage_v": idle_reference_voltage_v,
                    "average_cell_voltage_v": battery_per_cell_v,
                    "voltage_only_estimate": True,
                },
            },
            "runtime": {
                "mode": (
                    "demo" if isinstance(self.bus, FourLegDemoBus) else "hardware"
                ),
                "port": getattr(self.bus, "port_name", None),
            },
            "crawl": {
                "mode": self.crawl_mode,
                "pattern": (
                    "diagonal_pair_flat_support_gait_v1"
                    if self.crawl_mode == "diagonal_pair"
                    else "rectangular_flat_support_crawl_v9_slow"
                ),
                "supported_test_only": True,
                "active": self.crawl_active,
                "start_ready": self._crawl_stance_ready_locked(),
                "stage": self.crawl_stage,
                "phase": self.crawl_phase,
                "swing_corner": self.crawl_swing_corner,
                "swing_pair": (
                    None
                    if self.crawl_swing_pair is None
                    else list(self.crawl_swing_pair)
                ),
                "push_partner": self.crawl_push_partner,
                "airborne_leg_count": (
                    2 if self.crawl_mode == "diagonal_pair" else 1
                ),
                "planted_support_leg_count": (
                    2 if self.crawl_mode == "diagonal_pair" else 3
                ),
                "progress": self.crawl_progress,
                "period_s": self.dashboard.crawl.period_s,
                "ramp_rate_deg_s": self.dashboard.crawl.ramp_rate_deg_s,
                "run_until_stopped": True,
                "elapsed_s": crawl_elapsed_s,
                "completed_cycles": int(
                    crawl_elapsed_s / self.dashboard.crawl.period_s
                ),
                "duration_s": None,
                "stride_mm": self.dashboard.crawl.stride_m * 1000.0,
                "lift_mm": self.dashboard.crawl.lift_m * 1000.0,
                "support_extension_mm": (
                    self.dashboard.crawl.support_extension_m * 1000.0
                ),
                "stance_down_mm": self.dashboard.crawl.stance_down_m * 1000.0,
                "stance_fore_aft_mm": (self.dashboard.crawl.stance_fore_aft_m * 1000.0),
                "abduction_deg": self.dashboard.crawl.abduction_deg,
                "weight_shift_forward_mm": (
                    self.dashboard.crawl.weight_shift_forward_m * 1000.0
                ),
                "weight_shift_lateral_mm": (
                    self.dashboard.crawl.weight_shift_lateral_m * 1000.0
                ),
                "corner_map": {
                    str(profile.number): profile.corner for profile in self.profiles
                },
            },
            "rl_policy": self._rl_snapshot_locked(),
            "any_armed": bool(self._armed_count_locked()),
            "browser_heartbeat": browser_heartbeat,
            "last_event": self.last_event,
            "fault": self.fault,
        }
        self.last_full_snapshot = copy.deepcopy(payload)
        return payload

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            try:
                return self._snapshot_locked()
            except Exception as exc:
                self.fault = str(exc)
                self.last_event = "Telemetry fault; all motors disarmed"
                try:
                    self._recover_bus_locked()
                except Exception as recovery_exc:
                    self.fault = (
                        "Servo bus unavailable; automatic reconnect failed: "
                        f"{recovery_exc}"
                    )
                    self.last_event = (
                        "Telemetry fault; waiting for servo bus reconnect"
                    )
                    raise RuntimeError(self.fault) from recovery_exc
                return self._snapshot_locked()


class FourLegHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"
    allow_reuse_port = False

    def server_bind(self) -> None:
        # Windows permits two listeners on the same address unless the first
        # socket explicitly requests exclusive ownership.  A demo and hardware
        # dashboard sharing one port can otherwise receive alternating browser
        # requests, making simulated motion look like real motor feedback.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_EXCLUSIVEADDRUSE,
                1,
            )
        super().server_bind()

    def __init__(
        self,
        server_address: tuple[str, int],
        session: FourLegSession,
        token: str,
        *,
        allow_remote: bool = False,
    ):
        super().__init__(server_address, FourLegRequestHandler)
        self.session = session
        self.token = token
        self.allow_remote = allow_remote


class FourLegRequestHandler(BaseHTTPRequestHandler):
    server: FourLegHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)

    def _local_host(self) -> bool:
        if self.server.allow_remote:
            return True
        host = self.headers.get("Host", "").split(":", 1)[0].lower()
        return host in {"127.0.0.1", "localhost"}

    def _authorized(self) -> bool:
        if getattr(self.server, "allow_remote", False):
            return True
        return secrets.compare_digest(
            self.headers.get("X-Control-Token", ""),
            self.server.token,
        )

    def _compatible_lan_client(self) -> bool:
        return (
            not getattr(self.server, "allow_remote", False)
            or self.headers.get("X-Drobot-Client-Version", "")
            == LAN_CLIENT_VERSION
        )

    def _security_headers(self, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'none'; frame-ancestors 'none'",
        )

    def _send_bytes(
        self,
        payload: bytes,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self._security_headers(content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(
        self,
        payload: Any,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_bytes(data, status=status, content_type="application/json")

    def _send_download(self, path: Path) -> None:
        self.send_response(HTTPStatus.OK)
        self._security_headers("application/zip")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="{path.name}"',
        )
        self.end_headers()
        with path.open("rb") as stream:
            shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        if not self._local_host():
            self._error(HTTPStatus.FORBIDDEN, "Localhost access only")
            return
        request = urlsplit(self.path)
        path = request.path
        if path == "/":
            html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
            html = html.replace("__CONTROL_TOKEN__", self.server.token)
            self._send_bytes(
                html.encode("utf-8"),
                content_type="text/html; charset=utf-8",
            )
            return
        if path == "/app.css":
            self._send_bytes(
                (STATIC_DIR / "app.css").read_bytes(),
                content_type="text/css; charset=utf-8",
            )
            return
        if path == "/app.js":
            self._send_bytes(
                (STATIC_DIR / "app.js").read_bytes(),
                content_type="text/javascript; charset=utf-8",
            )
            return
        if path in {
            "/api/state",
            "/api/recordings",
            "/api/recordings/download",
        }:
            if not self._authorized():
                self._error(HTTPStatus.FORBIDDEN, "Invalid control token")
                return
        if path == "/api/recordings":
            self._send_json(
                {"recordings": self.server.session.list_recordings()}
            )
            return
        if path == "/api/recordings/download":
            recording_id = parse_qs(request.query).get("id", [""])[0]
            try:
                self._send_download(
                    self.server.session.recording_archive(recording_id)
                )
            except FileNotFoundError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
            except (ValueError, RuntimeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if path == "/api/state":
            if (
                getattr(self.server, "allow_remote", False)
                and self.headers.get("X-Control-Token", "")
                and not self._compatible_lan_client()
            ):
                # Pre-v2 browser pages send the embedded token and already
                # reload themselves after a 403. Headerless LAN scripts can
                # continue to read state without a token.
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "Dashboard page is out of date; reloading",
                )
                return
            try:
                self._send_json(self.server.session.snapshot())
            except Exception as exc:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return
        self._error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if not self._local_host() or not self._authorized():
            self._error(HTTPStatus.FORBIDDEN, "Local control authorization failed")
            return
        if self.path != "/api/heartbeat" and not self._compatible_lan_client():
            self._error(
                HTTPStatus.CONFLICT,
                "Dashboard page is out of date; reload before sending controls",
            )
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 4096:
                raise ValueError("Request is too large")
            payload = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            if self.path == "/api/heartbeat":
                self.server.session.heartbeat(self.client_address[0])
            elif self.path == "/api/arm":
                self.server.session.arm(
                    payload["leg"],
                    payload["motor"],
                    safety_ack=payload.get("safety_ack") is True,
                )
            elif self.path == "/api/arm-leg":
                self.server.session.arm_leg(
                    payload["leg"],
                    safety_ack=payload.get("safety_ack") is True,
                )
            elif self.path == "/api/crawl-forward":
                self.server.session.start_crawl_forward(
                    safety_ack=payload.get("safety_ack") is True,
                    confirmation=str(payload.get("confirmation", "")),
                )
            elif self.path == "/api/diagonal-pair-forward":
                self.server.session.start_diagonal_pair_forward(
                    safety_ack=payload.get("safety_ack") is True,
                    confirmation=str(payload.get("confirmation", "")),
                )
            elif self.path == "/api/crawl-stop":
                self.server.session.stop_crawl()
            elif self.path == "/api/rl-start":
                self.server.session.start_rl_policy(
                    forward_m_s=float(payload.get("forward_m_s", 0.04)),
                    duration_s=float(payload.get("duration_s", 5.0)),
                    safety_ack=payload.get("safety_ack") is True,
                    confirmation=str(payload.get("confirmation", "")),
                )
            elif self.path == "/api/rl-stop":
                self.server.session.stop_rl_policy()
            elif self.path == "/api/recordings/rename":
                self.server.session.rename_recording(
                    str(payload.get("recording_id", "")),
                    str(payload.get("label", "")),
                )
            elif self.path == "/api/recordings/delete":
                self.server.session.delete_recording(
                    str(payload.get("recording_id", ""))
                )
            elif self.path == "/api/power-reset":
                self.server.session.reset_power_analytics()
            elif self.path == "/api/target":
                self.server.session.set_target(
                    payload["leg"],
                    payload["motor"],
                    float(payload["degrees"]),
                )
            elif self.path == "/api/zero-leg":
                self.server.session.zero_armed_leg(payload["leg"])
            elif self.path == "/api/zero-all":
                self.server.session.zero_all_armed()
            elif self.path == "/api/center-all":
                self.server.session.center_all(
                    safety_ack=payload.get("safety_ack") is True,
                    confirmation=str(payload.get("confirmation", "")),
                )
            elif self.path == "/api/crawl-stance":
                self.server.session.set_crawl_stance(
                    safety_ack=payload.get("safety_ack") is True,
                    confirmation=str(payload.get("confirmation", "")),
                )
            elif self.path == "/api/capture-zero-all":
                self.server.session.capture_zero_all(
                    safety_ack=payload.get("safety_ack") is True,
                    confirmation=str(payload.get("confirmation", "")),
                )
            elif self.path == "/api/disarm":
                self.server.session.disarm(payload["leg"], payload["motor"])
            elif self.path == "/api/disarm-leg":
                self.server.session.disarm_leg(payload["leg"])
            elif self.path == "/api/disarm-all":
                self.server.session.disarm_all()
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self._send_json({"ok": True})
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Drobot four-leg dashboard.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--port", default="auto", help="Servo serial port")
    parser.add_argument("--http-bind", default=LOCAL_HOST)
    parser.add_argument("--http-port", type=int)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow LAN clients; use only on a trusted local network",
    )
    parser.add_argument("--ramp-rate", type=float)
    parser.add_argument("--rl-model", type=Path, default=DEFAULT_RL_MODEL)
    parser.add_argument("--rl-imu-axis-map", default="+x,+y,+z")
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=DEFAULT_RECORDINGS_DIR,
        help="Directory for automatic RL trial recordings",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use simulated motors and do not open a serial port",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (
        args.http_bind not in {"127.0.0.1", "localhost", "::1"}
        and not args.allow_remote
    ):
        parser.error("a non-loopback --http-bind requires --allow-remote")
    dashboard = load_dashboard_config(args.manifest)
    bus: Any
    if args.demo:
        bus = FourLegDemoBus(dashboard)
    else:
        bus = STSBus(args.port, dashboard.bus.baudrate)
    session = FourLegSession(
        dashboard,
        bus,
        ramp_rate_deg_s=args.ramp_rate,
        persist_calibration=not args.demo,
        rl_model_path=None if args.demo else args.rl_model,
        rl_imu_axis_map=args.rl_imu_axis_map,
        recordings_dir=args.recordings_dir,
    )
    token = secrets.token_urlsafe(24)
    http_port = dashboard.server.http_port if args.http_port is None else args.http_port
    try:
        server = FourLegHTTPServer(
            (args.http_bind, http_port),
            session,
            token,
            allow_remote=args.allow_remote,
        )
    except OSError as exc:
        raise SystemExit(
            f"Dashboard port {http_port} is already in use. Stop the existing "
            "demo or hardware dashboard before starting another one."
        ) from exc
    display_host = f"{socket.gethostname()}.local" if args.allow_remote else LOCAL_HOST
    url = f"http://{display_host}:{server.server_port}/"
    mode = "demo" if args.demo else f"hardware on {args.port}"
    try:
        session.start(allow_disconnected=not args.demo)
        print(f"Drobot four-leg control ({mode}): {url}")
        scope = "Trusted LAN enabled" if args.allow_remote else "Local machine only"
        print(f"{scope}. Ctrl+C disarms all motors and closes the bus.")
        if not args.no_browser:
            threading.Timer(0.4, webbrowser.open, args=(url,)).start()
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping four-leg controller...")
    finally:
        server.server_close()
        session.close()
        print("All 12 motors disarmed; bus closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
