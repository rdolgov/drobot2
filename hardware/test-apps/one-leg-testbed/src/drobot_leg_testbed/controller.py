"""Safety state for interactive and one-shot joint commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .model import (
    Calibration,
    LegConfig,
    MotorConfig,
    degrees_to_raw,
    raw_to_degrees,
)


class MotorBus(Protocol):
    def read_position(self, servo_id: int) -> int: ...

    def write_position(
        self,
        servo_id: int,
        raw_position: int,
        bus_config,
    ) -> None: ...

    def enable_torque(self, servo_id: int) -> None: ...

    def disable_torque(self, servo_id: int) -> None: ...


@dataclass(frozen=True)
class TargetState:
    motor: MotorConfig
    degrees: float
    raw_position: int


class LegController:
    """Keep torque state and target changes explicit and bounded."""

    def __init__(
        self,
        config: LegConfig,
        calibration: Calibration,
        bus: MotorBus,
    ):
        self.config = config
        self.calibration = calibration
        self.bus = bus
        self.armed_ids: set[int] = set()
        self.targets_deg: dict[str, float] = {}

    def measured_degrees(self, motor: MotorConfig) -> float:
        raw = self.bus.read_position(motor.servo_id)
        return raw_to_degrees(
            raw,
            motor,
            self.calibration.motor(motor),
        )

    def arm(self, motor: MotorConfig) -> TargetState:
        raw = self.bus.read_position(motor.servo_id)
        degrees = raw_to_degrees(
            raw,
            motor,
            self.calibration.motor(motor),
        )
        self.bus.write_position(motor.servo_id, raw, self.config.bus)
        self.bus.enable_torque(motor.servo_id)
        self.armed_ids.add(motor.servo_id)
        self.targets_deg[motor.name] = degrees
        return TargetState(motor, degrees, raw)

    def command(self, motor: MotorConfig, degrees: float) -> TargetState:
        if motor.servo_id not in self.armed_ids:
            raise RuntimeError(
                f"{motor.name} is disarmed. Use `arm` before commanding it."
            )
        previous = self.targets_deg[motor.name]
        delta = abs(degrees - previous)
        if delta > self.config.bus.max_command_step_deg + 1e-9:
            raise ValueError(
                f"Requested change is {delta:.2f} deg; the configured maximum "
                f"is {self.config.bus.max_command_step_deg:.2f} deg per "
                "command. Move in smaller increments."
            )
        raw = degrees_to_raw(
            degrees,
            motor,
            self.calibration.motor(motor),
        )
        fast_write = getattr(self.bus, "write_position_command", None)
        if fast_write is None:
            self.bus.write_position(motor.servo_id, raw, self.config.bus)
        else:
            fast_write(motor.servo_id, raw, self.config.bus)
        self.targets_deg[motor.name] = degrees
        return TargetState(motor, degrees, raw)

    def nudge(self, motor: MotorConfig, delta_deg: float) -> TargetState:
        if motor.name not in self.targets_deg:
            raise RuntimeError(f"{motor.name} has no active target. Use `arm` first.")
        return self.command(
            motor,
            self.targets_deg[motor.name] + delta_deg,
        )

    def disarm(self, motor: MotorConfig) -> None:
        self.bus.disable_torque(motor.servo_id)
        self.armed_ids.discard(motor.servo_id)
        self.targets_deg.pop(motor.name, None)

    def disarm_all(self) -> None:
        errors: list[Exception] = []
        for motor in self.config.motors:
            if motor.servo_id in self.armed_ids:
                try:
                    self.disarm(motor)
                except Exception as exc:  # preserve best-effort emergency stop
                    errors.append(exc)
        if errors:
            raise RuntimeError(
                "One or more motors could not be disarmed: "
                + "; ".join(str(error) for error in errors)
            )
