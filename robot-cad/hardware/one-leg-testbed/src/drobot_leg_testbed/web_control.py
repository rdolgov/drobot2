"""Localhost-only browser control for the one-leg hardware testbed."""

from __future__ import annotations

import argparse
import json
import math
import secrets
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .controller import LegController
from .model import (
    Calibration,
    LegConfig,
    MotorConfig,
    degrees_to_raw,
    load_calibration,
    load_config,
    raw_to_degrees,
)
from .ports import resolve_port
from .transport import MotorStatus, STSBus

LOCAL_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8765
STATIC_DIR = Path(__file__).with_name("web_static")


class DemoBus:
    """In-memory motor bus used to preview the UI without hardware."""

    def __init__(self, config: LegConfig, calibration: Calibration):
        self.positions = {
            motor.servo_id: calibration.motor(motor).center_tick
            for motor in config.motors
        }
        self.torque: set[int] = set()

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
        _bus_config: Any,
    ) -> None:
        self.positions[servo_id] = raw_position

    def enable_torque(self, servo_id: int) -> None:
        self.torque.add(servo_id)

    def disable_torque(self, servo_id: int) -> None:
        self.torque.discard(servo_id)

    def status(self, motor: MotorConfig) -> MotorStatus:
        return MotorStatus(
            servo_id=motor.servo_id,
            model_number=777,
            raw_position=self.positions[motor.servo_id],
            raw_speed=0,
            voltage_v=12.2,
            temperature_c=31,
            current_ma=6.5 if motor.servo_id in self.torque else 0.0,
            torque_enabled=motor.servo_id in self.torque,
        )


class ControlSession:
    """Serialize bus access, ramp targets, and enforce browser liveness."""

    def __init__(
        self,
        config: LegConfig,
        calibration: Calibration,
        bus: Any,
        *,
        ramp_rate_deg_s: float = 30.0,
        tick_interval_s: float = 0.05,
        heartbeat_timeout_s: float = 3.0,
        clock: Any = time.monotonic,
    ):
        if not 1.0 <= ramp_rate_deg_s <= 90.0:
            raise ValueError("ramp rate must be in [1, 90] deg/s")
        self.config = config
        self.calibration = calibration
        self.bus = bus
        self.controller = LegController(config, calibration, bus)
        self.ramp_rate_deg_s = ramp_rate_deg_s
        self.tick_interval_s = tick_interval_s
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.clock = clock
        self.desired_deg: dict[str, float] = {}
        self.last_heartbeat = clock()
        self.last_event = "Starting"
        self.fault: str | None = None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

    def start(self, *, start_worker: bool = True) -> None:
        self.bus.open()
        try:
            for motor in self.config.motors:
                self.bus.require_motor(motor)
            with self.lock:
                self._disarm_all_locked(raise_errors=True)
                self.last_event = "Connected; all motors disarmed"
        except Exception:
            self.bus.close()
            raise
        if start_worker:
            self.worker = threading.Thread(
                target=self._worker_loop,
                name="drobot-leg-motion",
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
            if self.controller.armed_ids:
                elapsed = self.clock() - self.last_heartbeat
                if elapsed > self.heartbeat_timeout_s:
                    self._disarm_all_locked(raise_errors=False)
                    self.last_event = "Browser heartbeat lost; all motors disarmed"
                    return

            step_limit = min(
                self.config.bus.max_command_step_deg,
                self.ramp_rate_deg_s * self.tick_interval_s,
            )
            for motor in self.config.motors:
                if motor.servo_id not in self.controller.armed_ids:
                    continue
                current = self.controller.targets_deg[motor.name]
                desired = self.desired_deg.get(motor.name, current)
                delta = desired - current
                if abs(delta) < 0.01:
                    continue
                step = max(-step_limit, min(step_limit, delta))
                self.controller.command(motor, current + step)

    def heartbeat(self) -> None:
        with self.lock:
            self.last_heartbeat = self.clock()

    def arm(self, selector: int | str, *, safety_ack: bool) -> None:
        if not safety_ack:
            raise ValueError("Confirm the fixture and physical cutoff before arming")
        motor = self.config.motor(selector)
        with self.lock:
            state = self.controller.arm(motor)
            self.desired_deg[motor.name] = state.degrees
            self.last_heartbeat = self.clock()
            self.last_event = f"Motor #{motor.number} armed at {state.degrees:+.2f}°"

    def set_target(self, selector: int | str, degrees: float) -> None:
        if not math.isfinite(degrees):
            raise ValueError("Target angle must be finite")
        motor = self.config.motor(selector)
        with self.lock:
            if motor.servo_id not in self.controller.armed_ids:
                raise RuntimeError(f"{motor.name} is disarmed")
            degrees_to_raw(
                degrees,
                motor,
                self.calibration.motor(motor),
            )
            self.desired_deg[motor.name] = degrees
            self.last_heartbeat = self.clock()
            self.last_event = (
                f"Motor #{motor.number} destination set to {degrees:+.2f}°"
            )

    def disarm(self, selector: int | str) -> None:
        motor = self.config.motor(selector)
        with self.lock:
            if motor.servo_id in self.controller.armed_ids:
                self.controller.disarm(motor)
            else:
                self.bus.disable_torque(motor.servo_id)
            self.desired_deg.pop(motor.name, None)
            self.last_event = f"Motor #{motor.number} disarmed"

    def disarm_all(self) -> None:
        with self.lock:
            self._disarm_all_locked(raise_errors=True)
            self.last_event = "All motors disarmed"

    def _disarm_all_locked(self, *, raise_errors: bool) -> None:
        errors: list[Exception] = []
        for motor in self.config.motors:
            try:
                self.bus.disable_torque(motor.servo_id)
            except Exception as exc:
                errors.append(exc)
        self.controller.armed_ids.clear()
        self.controller.targets_deg.clear()
        self.desired_deg.clear()
        if errors and raise_errors:
            raise RuntimeError(
                "One or more motors could not be disarmed: "
                + "; ".join(str(error) for error in errors)
            )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            motors: list[dict[str, Any]] = []
            for motor in self.config.motors:
                status = self.bus.status(motor)
                measured = raw_to_degrees(
                    status.raw_position,
                    motor,
                    self.calibration.motor(motor),
                )
                commanded = self.controller.targets_deg.get(motor.name)
                desired = self.desired_deg.get(motor.name)
                motors.append(
                    {
                        "number": motor.number,
                        "name": motor.name,
                        "label": motor.name.replace("_", " "),
                        "id": motor.servo_id,
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
                        "armed": motor.servo_id in self.controller.armed_ids,
                    }
                )
            return {
                "motors": motors,
                "any_armed": bool(self.controller.armed_ids),
                "ramp_rate_deg_s": self.ramp_rate_deg_s,
                "heartbeat_timeout_s": self.heartbeat_timeout_s,
                "last_event": self.last_event,
                "fault": self.fault,
            }


class ControlHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        session: ControlSession,
        token: str,
    ):
        super().__init__(server_address, ControlRequestHandler)
        self.session = session
        self.token = token


class ControlRequestHandler(BaseHTTPRequestHandler):
    server: ControlHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)

    def _local_host(self) -> bool:
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
                html.encode("utf-8"), content_type="text/html; charset=utf-8"
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
                    payload["motor"],
                    safety_ack=payload.get("safety_ack") is True,
                )
            elif self.path == "/api/target":
                self.server.session.set_target(
                    payload["motor"],
                    float(payload["degrees"]),
                )
            elif self.path == "/api/disarm":
                self.server.session.disarm(payload["motor"])
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
        description="Run the localhost-only Drobot leg browser controller.",
    )
    parser.add_argument("--config", type=Path, default=Path("leg.toml"))
    parser.add_argument("--port", default="auto", help="Servo serial port.")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
    )
    parser.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT)
    parser.add_argument("--ramp-rate", type=float, default=30.0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use simulated motors and do not open a serial port.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    calibration = load_calibration(args.calibration, config)
    bus: Any
    if args.demo:
        bus = DemoBus(config, calibration)
    else:
        bus = STSBus(resolve_port(args.port), config.bus.baudrate)
    session = ControlSession(
        config,
        calibration,
        bus,
        ramp_rate_deg_s=args.ramp_rate,
    )
    session.start()
    token = secrets.token_urlsafe(24)
    server = ControlHTTPServer((LOCAL_HOST, args.http_port), session, token)
    url = f"http://{LOCAL_HOST}:{server.server_port}/"
    print(f"Drobot leg control: {url}")
    print("Local machine only. Press Ctrl+C to disarm all motors and stop.")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping web controller...")
    finally:
        server.server_close()
        session.close()
        print("All motors disarmed; serial port closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
