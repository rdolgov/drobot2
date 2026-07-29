"""Command-line interface for a three-motor Drobot leg."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .controller import LegController
from .model import (
    MAX_SERVO_ID,
    MIN_SERVO_ID,
    calibration_from_centers,
    load_calibration,
    load_config,
    raw_to_degrees,
    save_calibration,
)
from .ports import list_serial_ports, resolve_port
from .transport import CommunicationError, STSBus

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "leg.example.toml"
DEFAULT_CALIBRATION = Path.cwd() / "calibration.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drobot-leg",
        description="Configure and control one Drobot leg with three ST3215s.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="TOML motor map and safety limits.",
    )
    parser.add_argument(
        "--port",
        default="auto",
        help="Serial device, or 'auto' when exactly one adapter is connected.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="Center calibration JSON written by capture-centers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ports", help="List serial adapters.")

    scan = subparsers.add_parser("scan", help="Ping a range of servo IDs.")
    scan.add_argument("--id-start", type=int, default=0)
    scan.add_argument("--id-end", type=int, default=20)

    assign = subparsers.add_parser(
        "assign-id",
        help="Assign one isolated motor a new bus ID.",
    )
    assign.add_argument("--current-id", type=int, default=1)
    assign.add_argument("--new-id", type=int, required=True)
    assign.add_argument("--yes", action="store_true")

    setup = subparsers.add_parser(
        "setup-ids",
        help="Interactively assign IDs 1-3, one isolated motor at a time.",
    )
    setup.add_argument("--current-id", type=int, default=1)

    middle = subparsers.add_parser(
        "set-middle",
        help="Persistently make one motor's current pose logical tick 2048.",
    )
    middle.add_argument("--motor", type=int, choices=(1, 2, 3), required=True)
    middle.add_argument("--yes", action="store_true")

    configure = subparsers.add_parser(
        "configure",
        help="Set all configured motors to safe position-control settings.",
    )
    configure.add_argument("--yes", action="store_true")

    subparsers.add_parser(
        "capture-centers",
        help="Disable torque and record the manually positioned neutral pose.",
    )
    subparsers.add_parser("status", help="Read all motor telemetry.")
    subparsers.add_parser(
        "control",
        help="Open the safe interactive selected-motor console.",
    )
    return parser


def _confirm(prompt: str, expected: str, assume_yes: bool = False) -> None:
    if assume_yes:
        return
    response = input(f"{prompt}\nType {expected} to continue: ").strip()
    if response != expected:
        raise RuntimeError("Cancelled; no motor settings were changed")


def _show_ports() -> None:
    ports = list_serial_ports()
    if not ports:
        print("No serial ports found.")
        return
    for port in ports:
        print(f"{port.device}\n  {port.description}\n  {port.hwid}")


def _scan(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if not (MIN_SERVO_ID <= args.id_start <= args.id_end <= MAX_SERVO_ID):
        raise ValueError(f"ID range must stay within {MIN_SERVO_ID}-{MAX_SERVO_ID}")
    port = resolve_port(args.port)
    found: list[tuple[int, int]] = []
    with STSBus(port, config.bus.baudrate) as bus:
        for servo_id in range(args.id_start, args.id_end + 1):
            model = bus.ping(servo_id)
            if model is not None:
                found.append((servo_id, model))
                print(f"ID {servo_id}: model {model}")
    if not found:
        print("No motors answered in the requested ID range.")


def _assign_id(
    args: argparse.Namespace,
    *,
    current_id: int | None = None,
    new_id: int | None = None,
    assume_yes: bool | None = None,
) -> int:
    config = load_config(args.config)
    old_id = args.current_id if current_id is None else current_id
    target_id = args.new_id if new_id is None else new_id
    yes = args.yes if assume_yes is None else assume_yes
    for value, name in ((old_id, "current ID"), (target_id, "new ID")):
        if not MIN_SERVO_ID <= value <= MAX_SERVO_ID:
            raise ValueError(f"{name} must be in {MIN_SERVO_ID}-{MAX_SERVO_ID}")
    _confirm(
        "Connect exactly ONE motor to the adapter. Support the leg, keep "
        "hands clear, and power the servo from the correct external supply.",
        "ASSIGN",
        yes,
    )
    port = resolve_port(args.port)
    with STSBus(port, config.bus.baudrate) as bus:
        model = bus.assign_id(old_id, target_id)
    print(f"Assigned ID {target_id}; responding model number is {model}.")
    return model


def _setup_ids(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print(
        "Each step requires only one motor on the bus. Turn external servo "
        "power OFF before unplugging or connecting a motor."
    )
    for motor in config.motors:
        input(
            f"\nPower OFF. Connect only motor #{motor.number} "
            f"({motor.name}), restore power, then press Enter."
        )
        local_args = argparse.Namespace(
            config=args.config,
            port=args.port,
            current_id=args.current_id,
            new_id=motor.servo_id,
            yes=False,
        )
        _assign_id(local_args)
        input(
            "Turn external servo power OFF now. Press Enter only after power "
            "is off, then disconnect this motor."
        )
    print("ID setup complete. Daisy-chain IDs 1-3 only after power is OFF.")


def _configure(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _confirm(
        "Support the leg off the table. This writes position mode and "
        f"applies a {config.bus.torque_limit}/1000 torque limit. Torque "
        "will remain disabled.",
        "CONFIGURE",
        args.yes,
    )
    port = resolve_port(args.port)
    with STSBus(port, config.bus.baudrate) as bus:
        for motor in config.motors:
            model = bus.configure_motor(motor, config.bus)
            print(
                f"#{motor.number} {motor.name}: ID {motor.servo_id}, "
                f"model {model}, configured and disarmed"
            )


def _set_middle(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    motor = config.motor(args.motor)
    _confirm(
        f"Torque must be OFF. Support the leg at neutral. This persistently "
        f"changes motor #{motor.number} ({motor.name}) position correction so "
        "its current physical position becomes logical tick 2048.",
        "CENTER",
        args.yes,
    )
    port = resolve_port(args.port)
    with STSBus(port, config.bus.baudrate) as bus:
        before, after = bus.set_middle_position(motor)
        status = bus.status(motor)
    print(
        f"#{motor.number} {motor.name}: middle position updated "
        f"from raw {before} to raw {after}; "
        f"torque={'ON' if status.torque_enabled else 'OFF'}."
    )


def _capture_centers(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    port = resolve_port(args.port)
    with STSBus(port, config.bus.baudrate) as bus:
        for motor in config.motors:
            bus.require_motor(motor)
            bus.disable_torque(motor.servo_id)
        input(
            "Torque is OFF. Support the leg, manually place all three joints "
            "in the intended neutral pose, then press Enter."
        )
        centers = {
            motor.name: bus.read_position(motor.servo_id) for motor in config.motors
        }
    calibration = calibration_from_centers(config, centers)
    save_calibration(calibration, args.calibration)
    print(f"Saved neutral centers to {args.calibration.resolve()}")
    for motor in calibration.motors:
        print(
            f"#{motor.number} {motor.name}: ID {motor.servo_id}, "
            f"center tick {motor.center_tick}"
        )


def _status(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    calibration = (
        load_calibration(args.calibration, config)
        if args.calibration.is_file()
        else None
    )
    port = resolve_port(args.port)
    with STSBus(port, config.bus.baudrate) as bus:
        for motor in config.motors:
            status = bus.status(motor)
            angle = ""
            if calibration is not None:
                degrees = raw_to_degrees(
                    status.raw_position,
                    motor,
                    calibration.motor(motor),
                )
                angle = f", angle={degrees:+.2f} deg"
            print(
                f"#{motor.number} {motor.name}: ID={motor.servo_id}, "
                f"raw={status.raw_position}{angle}, "
                f"speed={status.raw_speed}, "
                f"voltage={status.voltage_v:.1f} V, "
                f"temp={status.temperature_c} C, "
                f"current={status.current_ma:.1f} mA, "
                f"torque={'ON' if status.torque_enabled else 'OFF'}, "
                f"model={status.model_number}"
            )


def _print_control_help() -> None:
    print(
        "\nCommands:\n"
        "  select 1|2|3       choose one motor\n"
        "  arm                hold the selected motor at its current position\n"
        "  set <degrees>      set selected target within configured limits\n"
        "  + [degrees]        nudge positive; default is +1 degree\n"
        "  - [degrees]        nudge negative; default is -1 degree\n"
        "  status             show selected measured and target angles\n"
        "  disarm             disable selected motor torque\n"
        "  disarm-all         disable all motor torque\n"
        "  help               show this list\n"
        "  quit               disarm all motors and exit\n"
    )


def _control(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    calibration = load_calibration(args.calibration, config)
    port = resolve_port(args.port)
    with STSBus(port, config.bus.baudrate) as bus:
        for motor in config.motors:
            bus.require_motor(motor)
            bus.disable_torque(motor.servo_id)
        controller = LegController(config, calibration, bus)
        selected = config.motors[0]
        print(
            "All motors are connected and DISARMED. Only the selected motor "
            "receives commands; previously armed motors keep holding until "
            "explicitly disarmed. Keep a physical power switch within reach."
        )
        _print_control_help()
        try:
            while True:
                command = input(f"[#{selected.number} {selected.name}]> ").strip()
                if not command:
                    continue
                parts = command.split()
                verb = parts[0].lower()
                try:
                    if verb == "select" and len(parts) == 2:
                        selected = config.motor(parts[1])
                        print(
                            f"Selected #{selected.number} {selected.name}; "
                            f"limits [{selected.min_deg:+.1f}, "
                            f"{selected.max_deg:+.1f}] deg"
                        )
                    elif verb == "arm" and len(parts) == 1:
                        state = controller.arm(selected)
                        print(
                            f"ARMED at measured {state.degrees:+.2f} deg; "
                            f"raw {state.raw_position}"
                        )
                    elif verb == "set" and len(parts) == 2:
                        state = controller.command(
                            selected,
                            float(parts[1]),
                        )
                        print(
                            f"target={state.degrees:+.2f} deg "
                            f"(raw {state.raw_position})"
                        )
                    elif verb in {"+", "-"} and len(parts) <= 2:
                        amount = float(parts[1]) if len(parts) == 2 else 1.0
                        if amount <= 0.0:
                            raise ValueError("Nudge amount must be positive")
                        state = controller.nudge(
                            selected,
                            amount if verb == "+" else -amount,
                        )
                        print(
                            f"target={state.degrees:+.2f} deg "
                            f"(raw {state.raw_position})"
                        )
                    elif verb == "status" and len(parts) == 1:
                        measured = controller.measured_degrees(selected)
                        target = controller.targets_deg.get(selected.name)
                        target_text = "none" if target is None else f"{target:+.2f} deg"
                        torque_state = (
                            "ON" if selected.servo_id in controller.armed_ids else "OFF"
                        )
                        print(
                            f"measured={measured:+.2f} deg, "
                            f"target={target_text}, "
                            f"torque={torque_state}"
                        )
                    elif verb == "disarm" and len(parts) == 1:
                        controller.disarm(selected)
                        print("Selected motor torque OFF")
                    elif verb == "disarm-all" and len(parts) == 1:
                        controller.disarm_all()
                        print("All motor torque OFF")
                    elif verb == "help":
                        _print_control_help()
                    elif verb in {"quit", "exit"}:
                        break
                    else:
                        print("Unknown command. Type `help`.")
                except (KeyError, ValueError, RuntimeError) as exc:
                    print(f"Rejected: {exc}")
        finally:
            try:
                controller.disarm_all()
            finally:
                print("Control session ended; commanded motor torque is OFF.")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ports":
            _show_ports()
        elif args.command == "scan":
            _scan(args)
        elif args.command == "assign-id":
            _assign_id(args)
        elif args.command == "setup-ids":
            _setup_ids(args)
        elif args.command == "set-middle":
            _set_middle(args)
        elif args.command == "configure":
            _configure(args)
        elif args.command == "capture-centers":
            _capture_centers(args)
        elif args.command == "status":
            _status(args)
        elif args.command == "control":
            _control(args)
        else:  # pragma: no cover - argparse enforces this
            parser.error(f"Unhandled command {args.command}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (
        CommunicationError,
        ConnectionError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
