"""ROS 2 owner for the Drobot serial bus and LAN browser dashboard."""

from __future__ import annotations

import json
import secrets
import socket
import threading
from collections.abc import Callable
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from drobot_hardware_test_apps.four_leg_control import (
    FourLegDemoBus,
    FourLegRequestHandler,
    FourLegSession,
    load_dashboard_config,
)
from drobot_leg_testbed.ports import resolve_port
from drobot_leg_testbed.transport import STSBus


class OnboardRequestHandler(FourLegRequestHandler):
    """Use the existing dashboard API while accepting other LAN hosts."""

    def _local_host(self) -> bool:
        return True


class OnboardHTTPServer(ThreadingHTTPServer):
    """Threaded dashboard server sharing the ROS node's motor session."""

    daemon_threads = True
    allow_reuse_address = False
    allow_reuse_port = False

    def __init__(
        self,
        server_address: tuple[str, int],
        session: FourLegSession,
        token: str,
    ) -> None:
        super().__init__(server_address, OnboardRequestHandler)
        self.session = session
        self.token = token


class DrobotOnboardNode(Node):
    """Own hardware, expose ROS commands, and host the browser controller."""

    def __init__(self) -> None:
        super().__init__("onboard_control")

        self.declare_parameter("manifest_path", "")
        self.declare_parameter("serial_port", "auto")
        self.declare_parameter("http_bind", "0.0.0.0")
        self.declare_parameter("http_port", 8080)
        self.declare_parameter("control_token", "")
        self.declare_parameter("demo", False)
        self.declare_parameter("telemetry_period_s", 0.5)
        self.declare_parameter("ramp_rate_deg_s", 270.0)

        manifest_text = str(self.get_parameter("manifest_path").value).strip()
        if not manifest_text:
            raise ValueError(
                "manifest_path is required; pass the repository's "
                "hardware/robot-runtime/four-leg.toml"
            )
        manifest_path = Path(manifest_text).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Robot manifest not found: {manifest_path}")

        serial_port = str(self.get_parameter("serial_port").value).strip() or "auto"
        http_bind = str(self.get_parameter("http_bind").value).strip()
        http_port = int(self.get_parameter("http_port").value)
        configured_token = str(self.get_parameter("control_token").value).strip()
        demo = bool(self.get_parameter("demo").value)
        telemetry_period_s = float(self.get_parameter("telemetry_period_s").value)
        ramp_rate_deg_s = float(self.get_parameter("ramp_rate_deg_s").value)
        if not http_bind:
            raise ValueError("http_bind must not be empty")
        if not 1 <= http_port <= 65535:
            raise ValueError("http_port must be in [1, 65535]")
        if configured_token and len(configured_token) < 16:
            raise ValueError("control_token must be empty or at least 16 characters")
        if not 0.1 <= telemetry_period_s <= 10.0:
            raise ValueError("telemetry_period_s must be in [0.1, 10]")

        dashboard = load_dashboard_config(manifest_path)
        if demo:
            bus: Any = FourLegDemoBus(dashboard)
            bus_description = "demo motors"
        else:
            resolved_port = resolve_port(serial_port)
            bus = STSBus(resolved_port, dashboard.bus.baudrate)
            bus_description = resolved_port

        self.session = FourLegSession(
            dashboard,
            bus,
            ramp_rate_deg_s=ramp_rate_deg_s,
            persist_calibration=not demo,
        )
        self.http_server: OnboardHTTPServer | None = None
        self.http_thread: threading.Thread | None = None
        self._closed = False
        self._ros_keepalive = False
        self._last_published_event: str | None = None

        try:
            self.session.start()
            token = configured_token or secrets.token_urlsafe(24)
            self.http_server = OnboardHTTPServer(
                (http_bind, http_port),
                self.session,
                token,
            )
            self.http_thread = threading.Thread(
                target=self.http_server.serve_forever,
                kwargs={"poll_interval": 0.2},
                name="drobot-onboard-http",
                daemon=True,
            )
            self.http_thread.start()
        except Exception:
            self.session.close()
            raise

        self.status_publisher = self.create_publisher(String, "status", 10)
        self.event_publisher = self.create_publisher(String, "events", 10)
        self.command_subscription = self.create_subscription(
            String,
            "command",
            self._command_callback,
            10,
        )
        self.create_service(Trigger, "walk_distributed", self._walk_distributed)
        self.create_service(Trigger, "walk_diagonal_pair", self._walk_diagonal_pair)
        self.create_service(Trigger, "stop", self._stop)
        self.create_service(Trigger, "disarm_all", self._disarm_all)
        self.create_service(Trigger, "center_all", self._center_all)
        self.create_service(Trigger, "gait_stance", self._gait_stance)
        self.telemetry_timer = self.create_timer(
            telemetry_period_s,
            self._publish_status,
        )

        host = socket.gethostname()
        self.get_logger().info(
            f"Drobot bus ready on {bus_description}; dashboard listening on "
            f"http://{host}.local:{self.http_server.server_port}/ "
            f"(bind {http_bind})"
        )
        self.get_logger().info(
            "Walking repeats until stop; node shutdown always disarms all motors"
        )

    def _invoke(self, label: str, action: Callable[[], None], *, hold: bool) -> str:
        action()
        self._ros_keepalive = hold
        self._publish_status()
        return label

    def _trigger(
        self,
        response: Trigger.Response,
        label: str,
        action: Callable[[], None],
        *,
        hold: bool,
    ) -> Trigger.Response:
        try:
            response.message = self._invoke(label, action, hold=hold)
            response.success = True
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f"{label} failed: {exc}")
        return response

    def _walk_distributed(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        return self._trigger(
            response,
            "Distributed crawl started",
            lambda: self.session.start_crawl_forward(
                safety_ack=True,
                confirmation="TEST DISTRIBUTED CRAWL",
            ),
            hold=True,
        )

    def _walk_diagonal_pair(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        return self._trigger(
            response,
            "Diagonal-pair gait started",
            lambda: self.session.start_diagonal_pair_forward(
                safety_ack=True,
                confirmation="TEST DIAGONAL PAIR GAIT",
            ),
            hold=True,
        )

    def _stop(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        return self._trigger(
            response,
            "Motion stopped and all motors disarmed",
            self.session.stop_crawl,
            hold=False,
        )

    def _disarm_all(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        return self._trigger(
            response,
            "All motors disarmed",
            self.session.disarm_all,
            hold=False,
        )

    def _center_all(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        return self._trigger(
            response,
            "All joints moving to calibrated zero",
            lambda: self.session.center_all(
                safety_ack=True,
                confirmation="CENTER ALL 12",
            ),
            hold=True,
        )

    def _gait_stance(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        return self._trigger(
            response,
            "All joints moving to gait start stance",
            lambda: self.session.set_crawl_stance(
                safety_ack=True,
                confirmation="SET GAIT START STANCE",
            ),
            hold=True,
        )

    def _command_callback(self, message: String) -> None:
        command = message.data.strip().lower().replace("-", "_").replace(" ", "_")
        actions: dict[str, tuple[Callable[[], None], bool]] = {
            "walk_distributed": (
                lambda: self.session.start_crawl_forward(
                    safety_ack=True,
                    confirmation="TEST DISTRIBUTED CRAWL",
                ),
                True,
            ),
            "walk_diagonal_pair": (
                lambda: self.session.start_diagonal_pair_forward(
                    safety_ack=True,
                    confirmation="TEST DIAGONAL PAIR GAIT",
                ),
                True,
            ),
            "stop": (self.session.stop_crawl, False),
            "disarm_all": (self.session.disarm_all, False),
            "center_all": (
                lambda: self.session.center_all(
                    safety_ack=True,
                    confirmation="CENTER ALL 12",
                ),
                True,
            ),
            "gait_stance": (
                lambda: self.session.set_crawl_stance(
                    safety_ack=True,
                    confirmation="SET GAIT START STANCE",
                ),
                True,
            ),
        }
        selected = actions.get(command)
        if selected is None:
            self.get_logger().error(
                "Unknown command. Use walk_distributed, walk_diagonal_pair, "
                "stop, disarm_all, center_all, or gait_stance"
            )
            return
        action, hold = selected
        try:
            self._invoke(f"Command accepted: {command}", action, hold=hold)
        except Exception as exc:
            self.get_logger().error(f"Command {command} failed: {exc}")

    def _publish_status(self) -> None:
        try:
            if self._ros_keepalive:
                self.session.heartbeat()
            snapshot = self.session.snapshot()
            if self._ros_keepalive and not snapshot["any_armed"]:
                self._ros_keepalive = False
            status = String()
            status.data = json.dumps(snapshot, separators=(",", ":"))
            self.status_publisher.publish(status)
            event = str(snapshot["last_event"])
            if event != self._last_published_event:
                event_message = String()
                event_message.data = event
                self.event_publisher.publish(event_message)
                self._last_published_event = event
        except Exception as exc:
            self._ros_keepalive = False
            self.get_logger().error(f"Telemetry publication failed: {exc}")

    def destroy_node(self) -> bool:
        if self._closed:
            return super().destroy_node()
        self._closed = True
        self._ros_keepalive = False
        if self.http_server is not None:
            self.http_server.shutdown()
            self.http_server.server_close()
        if self.http_thread is not None:
            self.http_thread.join(timeout=2.0)
        self.session.close()
        self.get_logger().info("Onboard controller stopped; all motors disarmed")
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: DrobotOnboardNode | None = None
    try:
        node = DrobotOnboardNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
