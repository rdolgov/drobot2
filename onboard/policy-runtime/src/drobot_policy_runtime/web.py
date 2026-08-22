"""LAN dashboard for live IMU and print-only policy inspection."""

from __future__ import annotations

import argparse
import json
import math
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .contract import ACTION_NAMES, SERVO_ID_BY_ACTION_NAME
from .policy import OnnxWalkingPolicy
from .runtime import MotorSink, PolicyCommand, WalkingPolicyLoop
from .sources import (
    Bno085ImuSource,
    ImuSample,
    ImuSource,
    LevelImuSource,
    NeutralJointStateSource,
)


class DashboardState:
    def __init__(self, imu_backend: str) -> None:
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "running": False,
            "status": "stopped",
            "error": None,
            "imu_backend": imu_backend,
            "forward_m_s": 0.15,
            "motor_output_enabled": False,
            "imu": None,
            "motors": [],
            "last_policy_time_s": None,
        }

    def update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))


class RecordingImuSource:
    def __init__(self, source: ImuSource, state: DashboardState) -> None:
        self._source = source
        self._state = state

    def read(self) -> ImuSample:
        sample = self._source.read()
        self._state.update(
            imu={
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
                "sample_time_s": round(sample.monotonic_time_s, 6),
            }
        )
        return sample


class DashboardMotorSink(MotorSink):
    def __init__(self, state: DashboardState) -> None:
        self._state = state

    def write(
        self,
        action: np.ndarray,
        joint_target_rad: np.ndarray,
        monotonic_time_s: float,
    ) -> None:
        motors = [
            {
                "servo_id": SERVO_ID_BY_ACTION_NAME[name],
                "joint": name,
                "target_deg": round(math.degrees(float(joint_target_rad[index])), 3),
                "normalized_action": round(float(action[index]), 5),
            }
            for index, name in enumerate(ACTION_NAMES)
        ]
        self._state.update(
            motors=motors,
            last_policy_time_s=round(monotonic_time_s, 6),
        )


class PolicySupervisor:
    def __init__(
        self,
        model_path: Path,
        imu_source: ImuSource,
        imu_backend: str,
        control_hz: float,
    ) -> None:
        self.state = DashboardState(imu_backend)
        self._policy = OnnxWalkingPolicy(model_path)
        self._imu_source = RecordingImuSource(imu_source, self.state)
        self._control_hz = control_hz
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: WalkingPolicyLoop | None = None
        self._lock = threading.RLock()

    def start(self, forward_m_s: float) -> None:
        if not 0.0 <= forward_m_s <= 0.20:
            raise ValueError("forward_m_s must be in [0.0, 0.20]")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self.set_command(forward_m_s)
                return
            self._stop_event = threading.Event()
            self._loop = WalkingPolicyLoop(
                self._policy,
                self._imu_source,
                NeutralJointStateSource(),
                DashboardMotorSink(self.state),
                command=PolicyCommand(forward_m_s=forward_m_s),
                control_hz=self._control_hz,
            )
            self.state.update(
                running=True,
                status="running",
                error=None,
                forward_m_s=forward_m_s,
            )
            self._thread = threading.Thread(
                target=self._run,
                name="drobot-policy-loop",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        try:
            assert self._loop is not None
            self._loop.run(stop_event=self._stop_event)
            self.state.update(running=False, status="stopped")
        except Exception as exc:
            self.state.update(running=False, status="error", error=str(exc))

    def set_command(self, forward_m_s: float) -> None:
        if not 0.0 <= forward_m_s <= 0.20:
            raise ValueError("forward_m_s must be in [0.0, 0.20]")
        with self._lock:
            if self._loop is not None:
                self._loop.command = PolicyCommand(forward_m_s=forward_m_s)
            self.state.update(forward_m_s=forward_m_s)

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self.state.update(running=False, status="stopped")


class PolicyDashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        supervisor: PolicySupervisor,
        token: str,
    ) -> None:
        super().__init__(address, PolicyDashboardHandler)
        self.supervisor = supervisor
        self.token = token


class PolicyDashboardHandler(BaseHTTPRequestHandler):
    server: PolicyDashboardServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _authorized(self) -> bool:
        if not self.server.token:
            return True
        query_token = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
        header_token = self.headers.get("X-Drobot-Token", "")
        return secrets.compare_digest(self.server.token, header_token or query_token)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 4096:
            raise ValueError("request body is too large")
        if length == 0:
            return {}
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/state":
            if not self._authorized():
                self._send_json(403, {"error": "invalid control token"})
                return
            self._send_json(200, self.server.supervisor.state.snapshot())
            return

        static_names = {"/": "index.html", "/app.js": "app.js", "/app.css": "app.css"}
        name = static_names.get(path)
        if name is None:
            self.send_error(404)
            return
        resource = files("drobot_policy_runtime").joinpath("web_static", name)
        data = resource.read_bytes()
        content_type = {
            "index.html": "text/html; charset=utf-8",
            "app.js": "text/javascript; charset=utf-8",
            "app.css": "text/css; charset=utf-8",
        }[name]
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if not self._authorized():
            self._send_json(403, {"error": "invalid control token"})
            return
        path = urlsplit(self.path).path
        try:
            body = self._read_json()
            if path == "/api/start":
                self.server.supervisor.start(float(body.get("forward_m_s", 0.15)))
            elif path == "/api/command":
                self.server.supervisor.set_command(float(body["forward_m_s"]))
            elif path == "/api/stop":
                self.server.supervisor.stop()
            else:
                self._send_json(404, {"error": "unknown endpoint"})
                return
            self._send_json(200, self.server.supervisor.state.snapshot())
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})


def _i2c_address(text: str) -> int:
    value = int(text, 0)
    if not 0 <= value <= 0x7F:
        raise argparse.ArgumentTypeError("I2C address must be between 0x00 and 0x7f")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Drobot print-only policy dashboard")
    parser.add_argument("--model", required=True)
    parser.add_argument("--imu", choices=("bno085", "level"), default="bno085")
    parser.add_argument("--imu-address", type=_i2c_address, default=0x4A)
    parser.add_argument("--imu-axis-map", default="+x,+y,+z")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--control-token", default="")
    parser.add_argument("--control-hz", type=float, default=60.0)
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    token = args.control_token.strip()
    if token and len(token) < 16:
        parser.error("--control-token must contain at least 16 characters")
    imu_source: ImuSource = (
        Bno085ImuSource(args.imu_address, args.imu_axis_map)
        if args.imu == "bno085"
        else LevelImuSource()
    )
    supervisor = PolicySupervisor(
        Path(args.model), imu_source, args.imu, args.control_hz
    )
    server = PolicyDashboardServer((args.bind, args.port), supervisor, token)
    host = socket.gethostname()
    print("PRINT-ONLY POLICY DASHBOARD: this process cannot command the servo bus.")
    url = f"http://{host}.local:{server.server_port}/"
    if token:
        url += f"?token={token}"
    print(f"Open: {url}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.stop()
        server.server_close()


if __name__ == "__main__":
    main()
