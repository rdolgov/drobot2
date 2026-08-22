"""Small checked wrapper around Feetech's official Python SDK."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .model import BusConfig, MotorConfig

REG_ID = 5
REG_OPERATING_MODE = 33
REG_TORQUE_ENABLE = 40
REG_ACCELERATION = 41
REG_TORQUE_LIMIT = 48
REG_LOCK = 55
REG_PRESENT_VOLTAGE = 62
REG_PRESENT_TEMPERATURE = 63
REG_PRESENT_CURRENT = 69

POSITION_MODE = 0
MIDDLE_POSITION = 2048
SET_MIDDLE_COMMAND = 128


class CommunicationError(RuntimeError):
    """A checked Feetech communication or servo-status failure."""


@dataclass(frozen=True)
class MotorStatus:
    servo_id: int
    model_number: int
    raw_position: int
    raw_speed: int
    voltage_v: float
    temperature_c: int
    current_ma: float
    torque_enabled: bool


class STSBus:
    """Own one serial connection to an STS/SMS protocol bus."""

    def __init__(self, port: str, baudrate: int):
        self.requested_port = port
        self.port_name = port
        self.baudrate = baudrate
        self.port: Any | None = None
        self.packet: Any | None = None
        self._comm_success: int | None = None

    def __enter__(self) -> STSBus:
        self.open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def open(self) -> None:
        if self.port is not None:
            return
        try:
            from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts
        except ImportError as exc:
            raise RuntimeError(
                "Feetech SDK is not installed. Run "
                "`python3 -m pip install -e .` in this folder."
            ) from exc
        # Resolve `auto` every time the bus is opened. Linux may assign a new
        # ttyACM/ttyUSB number after a USB disconnect while the process stays
        # alive, so reusing the startup path would leave a permanently dead
        # connection.
        from .ports import resolve_port

        self.port_name = resolve_port(self.requested_port)
        port = PortHandler(self.port_name)
        if not port.openPort():
            raise ConnectionError(f"Could not open serial port {self.port_name}")
        if not port.setBaudRate(self.baudrate):
            port.closePort()
            raise ConnectionError(
                f"Could not set {self.port_name} to {self.baudrate} baud"
            )
        self.port = port
        self.packet = sms_sts(port)
        self._comm_success = COMM_SUCCESS

    def close(self) -> None:
        if self.port is not None:
            self.port.closePort()
        self.port = None
        self.packet = None

    def reopen(self) -> None:
        """Close the stale handle and resolve/open the requested adapter again."""

        self.close()
        self.open()

    def _require_open(self) -> Any:
        if self.packet is None:
            raise RuntimeError("Serial bus is not open")
        return self.packet

    def _check(self, operation: str, result: int, error: int) -> None:
        packet = self._require_open()
        if result != self._comm_success:
            raise CommunicationError(f"{operation}: {packet.getTxRxResult(result)}")
        if error:
            raise CommunicationError(f"{operation}: {packet.getRxPacketError(error)}")

    def ping(self, servo_id: int) -> int | None:
        packet = self._require_open()
        model, result, error = packet.ping(servo_id)
        if result != self._comm_success:
            return None
        self._check(f"ping ID {servo_id}", result, error)
        return int(model)

    def require_motor(self, motor: MotorConfig) -> int:
        model = self.ping(motor.servo_id)
        if model is None:
            raise CommunicationError(
                f"No motor answered at ID {motor.servo_id} "
                f"({motor.name}) on {self.port_name}"
            )
        return model

    def _write1(self, servo_id: int, address: int, value: int) -> None:
        packet = self._require_open()
        result, error = packet.write1ByteTxRx(servo_id, address, value)
        self._check(
            f"write ID {servo_id} register {address}",
            result,
            error,
        )

    def _write2(self, servo_id: int, address: int, value: int) -> None:
        packet = self._require_open()
        result, error = packet.write2ByteTxRx(servo_id, address, value)
        self._check(
            f"write ID {servo_id} register {address}",
            result,
            error,
        )

    def _read1(self, servo_id: int, address: int) -> int:
        packet = self._require_open()
        value, result, error = packet.read1ByteTxRx(servo_id, address)
        self._check(
            f"read ID {servo_id} register {address}",
            result,
            error,
        )
        return int(value)

    def _read2(self, servo_id: int, address: int) -> int:
        packet = self._require_open()
        value, result, error = packet.read2ByteTxRx(servo_id, address)
        self._check(
            f"read ID {servo_id} register {address}",
            result,
            error,
        )
        return int(value)

    def disable_torque(self, servo_id: int) -> None:
        self._write1(servo_id, REG_TORQUE_ENABLE, 0)

    def enable_torque(self, servo_id: int) -> None:
        self._write1(servo_id, REG_TORQUE_ENABLE, 1)
        self._write1(servo_id, REG_LOCK, 1)

    def unlock_config(self, servo_id: int) -> None:
        self._write1(servo_id, REG_LOCK, 0)

    def lock_config(self, servo_id: int) -> None:
        self._write1(servo_id, REG_LOCK, 1)

    def assign_id(self, current_id: int, new_id: int) -> int:
        model = self.ping(current_id)
        if model is None:
            raise CommunicationError(f"No motor answered at current ID {current_id}")
        if current_id != new_id and self.ping(new_id) is not None:
            raise CommunicationError(
                f"ID {new_id} already answers. Connect exactly one motor."
            )
        self.disable_torque(current_id)
        self.unlock_config(current_id)
        self._write1(current_id, REG_ID, new_id)
        verified_model = self.ping(new_id)
        if verified_model is None:
            raise CommunicationError(f"Motor did not answer at its new ID {new_id}")
        self.lock_config(new_id)
        return int(verified_model)

    def configure_motor(
        self,
        motor: MotorConfig,
        bus_config: BusConfig,
    ) -> int:
        model = self.require_motor(motor)
        self.disable_torque(motor.servo_id)
        self.unlock_config(motor.servo_id)
        self._write1(motor.servo_id, REG_OPERATING_MODE, POSITION_MODE)
        self._write2(
            motor.servo_id,
            REG_TORQUE_LIMIT,
            bus_config.torque_limit,
        )
        self._write1(
            motor.servo_id,
            REG_ACCELERATION,
            bus_config.acceleration,
        )
        self.lock_config(motor.servo_id)
        return model

    def set_middle_position(self, motor: MotorConfig) -> tuple[int, int]:
        """Persistently make the current physical position logical tick 2048."""
        self.require_motor(motor)
        before = self.read_position(motor.servo_id)
        self.disable_torque(motor.servo_id)
        self.unlock_config(motor.servo_id)
        try:
            self._write1(
                motor.servo_id,
                REG_TORQUE_ENABLE,
                SET_MIDDLE_COMMAND,
            )
            time.sleep(0.15)
            after = self.read_position(motor.servo_id)
        finally:
            self.lock_config(motor.servo_id)
        if abs(after - MIDDLE_POSITION) > 8:
            raise CommunicationError(
                f"ID {motor.servo_id} middle-position verification failed: "
                f"expected near {MIDDLE_POSITION}, read {after}"
            )
        return before, after

    def read_position(self, servo_id: int) -> int:
        packet = self._require_open()
        position, result, error = packet.ReadPos(servo_id)
        self._check(f"read position ID {servo_id}", result, error)
        return int(position)

    def write_position(
        self,
        servo_id: int,
        raw_position: int,
        bus_config: BusConfig,
    ) -> None:
        self._write2(
            servo_id,
            REG_TORQUE_LIMIT,
            bus_config.torque_limit,
        )
        packet = self._require_open()
        # The Python SDK writes the supplied 16-bit word verbatim. Match
        # Feetech's official C++ WritePosEx implementation by converting a
        # negative extended position to its bit-15 sign-magnitude wire value.
        encoded_position = packet.scs_toscs(raw_position, 15)
        result, error = packet.WritePosEx(
            servo_id,
            encoded_position,
            bus_config.speed,
            bus_config.acceleration,
        )
        self._check(f"write position ID {servo_id}", result, error)

    def status(self, motor: MotorConfig) -> MotorStatus:
        packet = self._require_open()
        model = self.require_motor(motor)
        position, speed, result, error = packet.ReadPosSpeed(motor.servo_id)
        self._check(f"read position/speed ID {motor.servo_id}", result, error)
        return MotorStatus(
            servo_id=motor.servo_id,
            model_number=model,
            raw_position=int(position),
            raw_speed=int(speed),
            voltage_v=self._read1(
                motor.servo_id,
                REG_PRESENT_VOLTAGE,
            )
            / 10.0,
            temperature_c=self._read1(
                motor.servo_id,
                REG_PRESENT_TEMPERATURE,
            ),
            current_ma=self._read2(
                motor.servo_id,
                REG_PRESENT_CURRENT,
            )
            * 6.5,
            torque_enabled=bool(self._read1(motor.servo_id, REG_TORQUE_ENABLE)),
        )
