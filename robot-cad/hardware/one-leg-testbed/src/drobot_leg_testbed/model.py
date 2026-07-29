"""Pure configuration, calibration, and angle conversion helpers."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ENCODER_TICKS = 4096
MIN_SERVO_ID = 0
MAX_SERVO_ID = 253


@dataclass(frozen=True)
class BusConfig:
    baudrate: int
    torque_limit: int
    speed: int
    acceleration: int
    max_command_step_deg: float


@dataclass(frozen=True)
class MotorConfig:
    number: int
    name: str
    servo_id: int
    direction: int
    min_deg: float
    max_deg: float


@dataclass(frozen=True)
class LegConfig:
    bus: BusConfig
    motors: tuple[MotorConfig, ...]

    def motor(self, selector: int | str) -> MotorConfig:
        selector_text = str(selector)
        for motor in self.motors:
            if selector_text == str(motor.number) or selector_text == motor.name:
                return motor
        choices = ", ".join(f"{motor.number}:{motor.name}" for motor in self.motors)
        raise KeyError(f"Unknown motor '{selector}'. Expected one of {choices}")


@dataclass(frozen=True)
class MotorCalibration:
    number: int
    name: str
    servo_id: int
    center_tick: int


@dataclass(frozen=True)
class Calibration:
    schema_version: int
    captured_at_utc: str
    motors: tuple[MotorCalibration, ...]

    def motor(self, config_motor: MotorConfig) -> MotorCalibration:
        for motor in self.motors:
            if motor.name == config_motor.name:
                if motor.servo_id != config_motor.servo_id:
                    raise ValueError(
                        f"Calibration ID for {motor.name} is "
                        f"{motor.servo_id}, config expects "
                        f"{config_motor.servo_id}"
                    )
                return motor
        raise KeyError(f"No calibration for {config_motor.name}")


def _require_int(
    value: object,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def load_config(path: str | Path) -> LegConfig:
    config_path = Path(path)
    with config_path.open("rb") as stream:
        data = tomllib.load(stream)

    bus_data = data.get("bus")
    motor_data = data.get("motors")
    if not isinstance(bus_data, dict):
        raise ValueError("Config requires a [bus] table")
    if not isinstance(motor_data, list) or len(motor_data) != 3:
        raise ValueError("Config requires exactly three [[motors]] tables")

    bus = BusConfig(
        baudrate=_require_int(
            bus_data.get("baudrate"),
            "bus.baudrate",
            4_800,
            1_000_000,
        ),
        torque_limit=_require_int(
            bus_data.get("torque_limit"),
            "bus.torque_limit",
            1,
            1_000,
        ),
        speed=_require_int(
            bus_data.get("speed"),
            "bus.speed",
            1,
            4_095,
        ),
        acceleration=_require_int(
            bus_data.get("acceleration"),
            "bus.acceleration",
            1,
            254,
        ),
        max_command_step_deg=float(bus_data.get("max_command_step_deg", 5.0)),
    )
    if not 0.0 < bus.max_command_step_deg <= 20.0:
        raise ValueError(
            "bus.max_command_step_deg must be greater than 0 and at most 20"
        )

    motors: list[MotorConfig] = []
    for index, entry in enumerate(motor_data, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"motors[{index}] must be a table")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"motors[{index}].name must be a non-empty string")
        direction = _require_int(
            entry.get("direction"),
            f"motors[{index}].direction",
            -1,
            1,
        )
        if direction == 0:
            raise ValueError(f"motors[{index}].direction must be -1 or 1")
        min_deg = float(entry.get("min_deg"))
        max_deg = float(entry.get("max_deg"))
        if not -180.0 <= min_deg < max_deg <= 180.0:
            raise ValueError(
                f"motors[{index}] limits must satisfy -180 <= min_deg < max_deg <= 180"
            )
        motors.append(
            MotorConfig(
                number=_require_int(
                    entry.get("number"),
                    f"motors[{index}].number",
                    1,
                    3,
                ),
                name=name.strip(),
                servo_id=_require_int(
                    entry.get("id"),
                    f"motors[{index}].id",
                    MIN_SERVO_ID,
                    MAX_SERVO_ID,
                ),
                direction=direction,
                min_deg=min_deg,
                max_deg=max_deg,
            )
        )

    for label, values in {
        "numbers": [motor.number for motor in motors],
        "names": [motor.name for motor in motors],
        "IDs": [motor.servo_id for motor in motors],
    }.items():
        if len(set(values)) != len(values):
            raise ValueError(f"Motor {label} must be unique: {values}")
    if sorted(motor.number for motor in motors) != [1, 2, 3]:
        raise ValueError("Motor numbers must be exactly 1, 2, and 3")

    return LegConfig(bus=bus, motors=tuple(sorted(motors, key=lambda m: m.number)))


def calibration_from_centers(
    config: LegConfig,
    centers: dict[str, int],
) -> Calibration:
    unknown = set(centers) - {motor.name for motor in config.motors}
    if unknown:
        raise ValueError(f"Unknown calibrated motors: {sorted(unknown)}")
    motors: list[MotorCalibration] = []
    for motor in config.motors:
        if motor.name not in centers:
            raise ValueError(f"Missing center for {motor.name}")
        center = _require_int(
            centers[motor.name],
            f"{motor.name}.center_tick",
            0,
            ENCODER_TICKS - 1,
        )
        motors.append(
            MotorCalibration(
                number=motor.number,
                name=motor.name,
                servo_id=motor.servo_id,
                center_tick=center,
            )
        )
    return Calibration(
        schema_version=1,
        captured_at_utc=datetime.now(UTC).isoformat(),
        motors=tuple(motors),
    )


def save_calibration(calibration: Calibration, path: str | Path) -> None:
    payload = {
        "schema_version": calibration.schema_version,
        "captured_at_utc": calibration.captured_at_utc,
        "motors": [
            {
                "number": motor.number,
                "name": motor.name,
                "id": motor.servo_id,
                "center_tick": motor.center_tick,
            }
            for motor in calibration.motors
        ],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def load_calibration(
    path: str | Path,
    config: LegConfig | None = None,
) -> Calibration:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported calibration schema")
    entries = data.get("motors")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("Calibration requires exactly three motors")
    calibration = Calibration(
        schema_version=1,
        captured_at_utc=str(data.get("captured_at_utc", "")),
        motors=tuple(
            MotorCalibration(
                number=_require_int(
                    entry.get("number"),
                    "calibration.number",
                    1,
                    3,
                ),
                name=str(entry.get("name")),
                servo_id=_require_int(
                    entry.get("id"),
                    "calibration.id",
                    MIN_SERVO_ID,
                    MAX_SERVO_ID,
                ),
                center_tick=_require_int(
                    entry.get("center_tick"),
                    "calibration.center_tick",
                    0,
                    ENCODER_TICKS - 1,
                ),
            )
            for entry in entries
        ),
    )
    if config is not None:
        for motor in config.motors:
            calibration.motor(motor)
    return calibration


def degrees_to_raw(
    degrees: float,
    motor: MotorConfig,
    calibration: MotorCalibration,
) -> int:
    if not motor.min_deg <= degrees <= motor.max_deg:
        raise ValueError(
            f"{motor.name} target {degrees:.2f} deg is outside "
            f"[{motor.min_deg:.2f}, {motor.max_deg:.2f}]"
        )
    raw = calibration.center_tick + round(
        motor.direction * degrees * ENCODER_TICKS / 360.0
    )
    if not 0 <= raw < ENCODER_TICKS:
        raise ValueError(
            f"{motor.name} target converts to raw tick {raw}, "
            f"outside [0, {ENCODER_TICKS - 1}]. Re-center the joint."
        )
    return raw


def raw_to_degrees(
    raw: int,
    motor: MotorConfig,
    calibration: MotorCalibration,
) -> float:
    if not 0 <= raw < ENCODER_TICKS:
        raise ValueError(f"Raw position must be in [0, {ENCODER_TICKS - 1}]")
    return motor.direction * (raw - calibration.center_tick) * 360.0 / ENCODER_TICKS
