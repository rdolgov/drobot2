"""Cross-platform serial-port discovery."""

from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class SerialPortInfo:
    device: str
    description: str
    hwid: str


def list_serial_ports() -> tuple[SerialPortInfo, ...]:
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is not installed. Run `python3 -m pip install -e .`"
        ) from exc
    return tuple(
        SerialPortInfo(
            device=port.device,
            description=port.description or "",
            hwid=port.hwid or "",
        )
        for port in list_ports.comports()
    )


def _score_port(port: SerialPortInfo) -> tuple[int, str]:
    device = port.device.lower()
    description = port.description.lower()
    system = platform.system()
    score = 0
    if system == "Darwin":
        if device.startswith("/dev/tty.usbmodem"):
            score += 100
        elif device.startswith("/dev/cu.usbmodem"):
            score += 95
        elif device.startswith("/dev/tty.usbserial"):
            score += 90
        elif device.startswith("/dev/cu.usbserial"):
            score += 85
    elif system == "Windows" and device.startswith("com"):
        score += 70
    elif device.startswith("/dev/ttyacm"):
        score += 80
    elif device.startswith("/dev/ttyusb"):
        score += 75
    if any(
        token in description
        for token in ("usb", "serial", "uart", "ch340", "ch343", "cp210")
    ):
        score += 20
    return score, port.device


def resolve_port(requested: str) -> str:
    if requested != "auto":
        return requested
    ports = list_serial_ports()
    ranked = sorted(ports, key=_score_port, reverse=True)
    candidates = [port for port in ranked if _score_port(port)[0] > 0]
    if len(candidates) == 1:
        return candidates[0].device
    if not candidates:
        raise RuntimeError(
            "No likely USB serial adapter found. Run `drobot-leg ports` "
            "and pass `--port` explicitly."
        )
    choices = "\n".join(f"  {port.device}: {port.description}" for port in candidates)
    raise RuntimeError(
        "More than one USB serial adapter was found. Pass `--port` "
        f"explicitly:\n{choices}"
    )
