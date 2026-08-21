"""Control dashboard for all four Drobot legs."""

from __future__ import annotations

import argparse
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
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

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
from drobot_leg_testbed.ports import resolve_port
from drobot_leg_testbed.transport import MotorStatus, STSBus

from drobot_hardware_test_apps.crawl_gait import (
    LEG_CORNERS,
    diagonal_pair_gait_degrees,
    distributed_push_crawl_degrees,
)

LOCAL_HOST = "127.0.0.1"
APP_ROOT = Path(__file__).resolve().parents[2]
HARDWARE_ROOT = APP_ROOT.parents[1]
DEFAULT_MANIFEST = HARDWARE_ROOT / "robot-runtime" / "four-leg.toml"
STATIC_DIR = Path(__file__).with_name("four_leg_static")


@dataclass(frozen=True)
class MonitoringConfig:
    voltage_warning_low_v: float
    voltage_warning_high_v: float
    voltage_spread_warning_v: float
    temperature_warning_c: int
    leg_current_warning_ma: float


@dataclass(frozen=True)
class ServerConfig:
    http_port: int
    ramp_rate_deg_s: float
    heartbeat_timeout_s: float


@dataclass(frozen=True)
class CrawlConfig:
    period_s: float
    cycles: int
    run_until_stopped: bool
    stance_settle_s: float
    stride_m: float
    lift_m: float
    support_extension_m: float
    weight_shift_forward_m: float
    weight_shift_lateral_m: float
    stance_down_m: float
    stance_fore_aft_m: float
    abduction_deg: float
    start_tolerance_deg: float


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
    run_until_stopped = crawl_data.get("run_until_stopped", True)
    if not isinstance(run_until_stopped, bool):
        raise ValueError("crawl.run_until_stopped must be true or false")
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
    if not 1.0 <= server.heartbeat_timeout_s <= 10.0:
        raise ValueError("server.heartbeat_timeout_s must be in [1, 10]")

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
        temperature_warning_c=int(monitoring_data.get("temperature_warning_c", 60)),
        leg_current_warning_ma=_finite_float(
            monitoring_data.get("leg_current_warning_ma", 3000.0),
            "monitoring.leg_current_warning_ma",
        ),
    )
    if not 0 < monitoring.voltage_warning_low_v:
        raise ValueError("Low-voltage warning must be positive")
    if not (
        monitoring.voltage_warning_low_v < monitoring.voltage_warning_high_v <= 15.0
    ):
        raise ValueError("Voltage warning range is invalid")
    if not 0 < monitoring.voltage_spread_warning_v <= 3.0:
        raise ValueError("Voltage-spread warning must be in (0, 3]")
    if not 30 <= monitoring.temperature_warning_c <= 90:
        raise ValueError("Temperature warning must be in [30, 90] C")
    if not 100 <= monitoring.leg_current_warning_ma <= 10_000:
        raise ValueError("Leg-current warning must be in [100, 10000] mA")

    crawl = CrawlConfig(
        period_s=_finite_float(
            crawl_data.get("period_s", 4.0),
            "crawl.period_s",
        ),
        cycles=int(crawl_data.get("cycles", 2)),
        run_until_stopped=run_until_stopped,
        stance_settle_s=_finite_float(
            crawl_data.get("stance_settle_s", 1.5),
            "crawl.stance_settle_s",
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
        start_tolerance_deg=_finite_float(
            crawl_data.get("start_tolerance_deg", 35.0),
            "crawl.start_tolerance_deg",
        ),
    )
    if not 4.0 <= crawl.period_s <= 60.0:
        raise ValueError("crawl.period_s must be in [4, 60]")
    if not 1 <= crawl.cycles <= 4:
        raise ValueError("crawl.cycles must be in [1, 4]")
    if not 0.5 <= crawl.stance_settle_s <= 5.0:
        raise ValueError("crawl.stance_settle_s must be in [0.5, 5]")
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
    if not 10.0 <= crawl.start_tolerance_deg <= 45.0:
        raise ValueError("crawl.start_tolerance_deg must be in [10, 45]")

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

    def require_motor(self, motor: MotorConfig) -> int:
        if motor.servo_id not in self.positions:
            raise RuntimeError(f"Demo motor ID {motor.servo_id} is missing")
        return 777

    def read_position(self, servo_id: int) -> int:
        return self.positions[servo_id]

    def write_position(
        self,
        servo_id: int,
        raw_position: int,
        _bus_config: BusConfig,
    ) -> None:
        self.positions[servo_id] = raw_position

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
        if not 1.0 <= self.heartbeat_timeout_s <= 10.0:
            raise ValueError("heartbeat timeout must be in [1, 10] seconds")
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
        self.last_heartbeat = clock()
        self.last_event = "Starting"
        self.fault: str | None = None
        self.crawl_mode = "distributed"
        self.crawl_stage = "idle"
        self.crawl_phase = "idle"
        self.crawl_swing_corner: str | None = None
        self.crawl_swing_pair: tuple[str, ...] | None = None
        self.crawl_push_partner: str | None = None
        self.crawl_started_at: float | None = None
        self.crawl_target_reached_at: float | None = None
        self.crawl_progress = 0.0
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

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

    def start(self, *, start_worker: bool = True) -> None:
        self.bus.open()
        try:
            for profile in self.profiles:
                for motor in profile.config.motors:
                    self.bus.require_motor(motor)
            with self.lock:
                self._disarm_all_locked(raise_errors=True)
                self.last_event = "All 12 motors online and disarmed"
        except Exception:
            with self.lock:
                self._disarm_all_locked(raise_errors=False)
            self.bus.close()
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
        try:
            with self.lock:
                self._disarm_all_locked(raise_errors=False)
        finally:
            self.bus.close()

    def _worker_loop(self) -> None:
        while not self.stop_event.wait(self.tick_interval_s):
            try:
                self.advance_once()
            except Exception as exc:
                with self.lock:
                    self.fault = str(exc)
                    self.last_event = "Motion fault; all motors disarmed"
                    self._disarm_all_locked(raise_errors=False)

    def advance_once(self) -> None:
        with self.lock:
            if self._armed_count_locked():
                elapsed = self.clock() - self.last_heartbeat
                if elapsed > self.heartbeat_timeout_s:
                    self._disarm_all_locked(raise_errors=False)
                    self.last_event = "Browser heartbeat lost; all motors disarmed"
                    return

            self._advance_crawl_locked()

            step_limit = min(
                self.dashboard.bus.max_command_step_deg,
                self.ramp_rate_deg_s * self.tick_interval_s,
            )
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
                    controller.command(motor, current + step)

    @property
    def crawl_active(self) -> bool:
        return self.crawl_stage in {
            "preparing",
            "preloading",
            "walking",
            "finishing",
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

    def _advance_crawl_locked(self) -> None:
        if not self.crawl_active:
            return
        now = self.clock()
        config = self.dashboard.crawl
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
            self.crawl_target_reached_at = None
            self.last_event = (
                "Diagonal-pair gait started; front-left and rear-right lift first"
                if self.crawl_mode == "diagonal_pair"
                else "Distributed push crawl started; rear-right swings first"
            )
            return

        duration_s = config.period_s * config.cycles
        if self.crawl_stage == "walking":
            if self.crawl_started_at is None:
                raise RuntimeError("Crawl clock was not initialized")
            elapsed = max(0.0, now - self.crawl_started_at)
            if config.run_until_stopped:
                self.crawl_progress = (elapsed % config.period_s) / config.period_s
                self._set_crawl_pose_locked(elapsed)
                return
            self.crawl_progress = min(elapsed / duration_s, 1.0)
            if elapsed < duration_s:
                self._set_crawl_pose_locked(elapsed)
                return
            self.crawl_stage = "finishing"
            self._set_crawl_stance_locked("returning_to_gait_start_stance")
            return

        self._set_crawl_stance_locked("holding_gait_start_stance")
        if self._crawl_targets_reached_locked():
            self.crawl_stage = "complete"
            self.crawl_phase = "holding_gait_start_stance"
            self.crawl_progress = 1.0
            self.last_event = (
                "Diagonal-pair gait complete; all motors holding stance"
                if self.crawl_mode == "diagonal_pair"
                else "Distributed push crawl complete; all motors holding stance"
            )

    def _cancel_crawl_locked(self) -> None:
        self.crawl_stage = "idle"
        self.crawl_phase = "idle"
        self.crawl_swing_corner = None
        self.crawl_swing_pair = None
        self.crawl_push_partner = None
        self.crawl_started_at = None
        self.crawl_target_reached_at = None
        self.crawl_progress = 0.0

    def _require_manual_control_locked(self) -> None:
        if self.crawl_active:
            raise RuntimeError("Stop the active crawl before manual motion")

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
            if self._armed_count_locked():
                raise RuntimeError("Disarm all 12 motors before starting a crawl")
            if self.fault:
                raise RuntimeError("Clear the reported fault before starting a crawl")

            preflight = self._snapshot_locked()
            out_of_start_range: list[str] = []
            for leg in preflight["legs"]:
                for motor in leg["motors"]:
                    if (
                        abs(float(motor["measured_deg"]))
                        > self.dashboard.crawl.start_tolerance_deg
                    ):
                        out_of_start_range.append(
                            f"ID {motor['id']} ({motor['measured_deg']:+.1f} deg)"
                        )
            if out_of_start_range:
                raise RuntimeError(
                    "Center the robot before walking; outside start tolerance: "
                    + ", ".join(out_of_start_range)
                )

            previous_mode = self.crawl_mode
            newly_armed: list[tuple[LegProfile, MotorConfig]] = []
            try:
                self.crawl_mode = mode
                sample_count = 80
                duration_s = self.dashboard.crawl.period_s
                self._crawl_stance_targets_locked()
                for sample in range(sample_count + 1):
                    self._crawl_pose_locked(duration_s * sample / sample_count)

                for profile in self.profiles:
                    controller = self.controllers[profile.number]
                    for motor in profile.config.motors:
                        state = controller.arm(motor)
                        newly_armed.append((profile, motor))
                        self.desired_deg[(profile.number, motor.name)] = state.degrees
                self.crawl_stage = "preparing"
                self.crawl_phase = "moving_to_stance"
                self.crawl_swing_corner = None
                self.crawl_swing_pair = None
                self.crawl_push_partner = None
                self.crawl_started_at = None
                self.crawl_target_reached_at = None
                self.crawl_progress = 0.0
                self._set_crawl_stance_locked("moving_to_gait_start_stance")
            except Exception:
                for profile, motor in newly_armed:
                    try:
                        self.controllers[profile.number].disarm(motor)
                    except Exception:
                        pass
                self._disarm_all_locked(raise_errors=False)
                self.crawl_mode = previous_mode
                raise

            self.last_heartbeat = self.clock()
            self.fault = None
            self.last_event = (
                "All 12 motors moving to the diagonal-pair start stance"
                if mode == "diagonal_pair"
                else "All 12 motors moving to the distributed-push start stance"
            )

    def stop_crawl(self) -> None:
        with self.lock:
            self._disarm_all_locked(raise_errors=True)
            self.last_event = "Crawl stopped; all 12 motors disarmed"

    def heartbeat(self) -> None:
        with self.lock:
            self.last_heartbeat = self.clock()

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
            self.last_heartbeat = self.clock()
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
            self.last_heartbeat = self.clock()
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
            self.last_heartbeat = self.clock()
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
            self.last_heartbeat = self.clock()
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
            self.last_heartbeat = self.clock()
            self.last_event = "All armed motors returning to zero"

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
                for profile in self.profiles:
                    controller = self.controllers[profile.number]
                    for motor in profile.config.motors:
                        degrees_to_raw(
                            0.0,
                            motor,
                            profile.calibration.motor(motor),
                        )
                        if motor.servo_id not in controller.armed_ids:
                            state = controller.arm(motor)
                            self.desired_deg[(profile.number, motor.name)] = (
                                state.degrees
                            )
                for profile in self.profiles:
                    for motor in profile.config.motors:
                        self.desired_deg[(profile.number, motor.name)] = 0.0
            except Exception:
                self._disarm_all_locked(raise_errors=False)
                self.last_event = "Center-all failed; all motors disarmed"
                raise

            self.last_heartbeat = self.clock()
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
            except Exception:
                self._disarm_all_locked(raise_errors=False)
                self.last_event = "Walk-stance command failed; all motors disarmed"
                raise

            self.last_heartbeat = self.clock()
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
            if self.crawl_active:
                self._disarm_all_locked(raise_errors=True)
                self.last_event = "Crawl stopped; all 12 motors disarmed"
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
            if self.crawl_active:
                self._disarm_all_locked(raise_errors=True)
                self.last_event = "Crawl stopped; all 12 motors disarmed"
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
            raise RuntimeError(
                "One or more motors could not be disarmed: "
                + "; ".join(str(error) for error in errors)
            )

    def _armed_count_locked(self) -> int:
        return sum(
            len(controller.armed_ids) for controller in self.controllers.values()
        )

    def _snapshot_locked(self) -> dict[str, Any]:
        leg_payloads: list[dict[str, Any]] = []
        all_statuses: list[MotorStatus] = []
        unexpected_torque: list[int] = []
        current_by_leg: dict[int, float] = {}

        for profile in self.profiles:
            controller = self.controllers[profile.number]
            motors: list[dict[str, Any]] = []
            leg_current = 0.0
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
                        "torque_enabled": status.torque_enabled,
                        "armed": armed,
                        "model": status.model_number,
                    }
                )
            current_by_leg[profile.number] = leg_current
            leg_payloads.append(
                {
                    "number": profile.number,
                    "label": profile.label,
                    "corner": profile.corner,
                    "current_ma": leg_current,
                    "armed_count": sum(motor["armed"] for motor in motors),
                    "motors": motors,
                }
            )

        monitoring = self.dashboard.monitoring
        voltages = [status.voltage_v for status in all_statuses]
        temperatures = [status.temperature_c for status in all_statuses]
        voltage_min = min(voltages)
        voltage_max = max(voltages)
        voltage_spread = voltage_max - voltage_min
        maximum_temperature = max(temperatures)
        total_current = sum(status.current_ma for status in all_statuses)
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

        crawl_elapsed_s = (
            max(0.0, self.clock() - self.crawl_started_at)
            if self.crawl_stage == "walking" and self.crawl_started_at is not None
            else 0.0
        )

        return {
            "legs": leg_payloads,
            "summary": {
                "online_count": len(all_statuses),
                "armed_count": self._armed_count_locked(),
                "voltage_min_v": voltage_min,
                "voltage_max_v": voltage_max,
                "voltage_spread_v": voltage_spread,
                "total_current_ma": total_current,
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
                "heartbeat_timeout_s": self.heartbeat_timeout_s,
                "voltage_warning_low_v": monitoring.voltage_warning_low_v,
                "voltage_warning_high_v": monitoring.voltage_warning_high_v,
                "temperature_warning_c": monitoring.temperature_warning_c,
                "leg_current_warning_ma": monitoring.leg_current_warning_ma,
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
                    else "rectangular_flat_support_crawl_v8"
                ),
                "supported_test_only": True,
                "active": self.crawl_active,
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
                "cycles": self.dashboard.crawl.cycles,
                "run_until_stopped": self.dashboard.crawl.run_until_stopped,
                "elapsed_s": crawl_elapsed_s,
                "completed_cycles": int(
                    crawl_elapsed_s / self.dashboard.crawl.period_s
                ),
                "duration_s": (
                    None
                    if self.dashboard.crawl.run_until_stopped
                    else self.dashboard.crawl.period_s * self.dashboard.crawl.cycles
                ),
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
            "any_armed": bool(self._armed_count_locked()),
            "last_event": self.last_event,
            "fault": self.fault,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            try:
                return self._snapshot_locked()
            except Exception as exc:
                self.fault = str(exc)
                self.last_event = "Telemetry fault; all motors disarmed"
                self._disarm_all_locked(raise_errors=False)
                raise


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
        return secrets.compare_digest(
            self.headers.get("X-Control-Token", ""),
            self.server.token,
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

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        if not self._local_host():
            self._error(HTTPStatus.FORBIDDEN, "Localhost access only")
            return
        if self.path == "/":
            html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
            html = html.replace("__CONTROL_TOKEN__", self.server.token)
            self._send_bytes(
                html.encode("utf-8"),
                content_type="text/html; charset=utf-8",
            )
            return
        if self.path == "/app.css":
            self._send_bytes(
                (STATIC_DIR / "app.css").read_bytes(),
                content_type="text/css; charset=utf-8",
            )
            return
        if self.path == "/app.js":
            self._send_bytes(
                (STATIC_DIR / "app.js").read_bytes(),
                content_type="text/javascript; charset=utf-8",
            )
            return
        if self.path == "/api/state":
            if not self._authorized():
                self._error(HTTPStatus.FORBIDDEN, "Invalid control token")
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
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 4096:
                raise ValueError("Request is too large")
            payload = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            if self.path == "/api/heartbeat":
                self.server.session.heartbeat()
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
        bus = STSBus(resolve_port(args.port), dashboard.bus.baudrate)
    session = FourLegSession(
        dashboard,
        bus,
        ramp_rate_deg_s=args.ramp_rate,
        persist_calibration=not args.demo,
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
        session.start()
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
