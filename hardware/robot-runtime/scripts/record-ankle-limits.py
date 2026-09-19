#!/usr/bin/env python3
"""Record hand-moved ankle travel without commanding servo motion."""
from __future__ import annotations

import argparse
import csv
import json
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from drobot_leg_testbed.transport import STSBus

stop_signal = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def request_stop(signum, _frame):
    global stop_signal
    stop_signal = signum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=900.0)
    parser.add_argument("--hz", type=float, default=20.0)
    args = parser.parse_args()
    if not 0 < args.hz <= 50 or not 0 < args.seconds <= 3600:
        parser.error("Use 0 < hz <= 50 and 0 < seconds <= 3600.")
    for service in ("drobot-manual-web", "drobot-onboard"):
        state = subprocess.run(
            ["systemctl", "show", service, "--property=ActiveState", "--value"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if state not in ("inactive", "failed"):
            raise RuntimeError(f"Stop {service} before recording; state is {state!r}.")

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    motors = calibration["motors"]
    if sorted(m["id"] for m in motors) != [13, 14, 15, 16]:
        raise ValueError("This recorder is restricted to ankle IDs 13-16.")
    if any(m["direction"] not in (-1, 1) or not 0 <= m["center_tick"] < 4096 for m in motors):
        raise ValueError("Each ankle requires a neutral center and direction.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "samples.csv"
    if samples_path.exists():
        raise FileExistsError("Use a new output directory to preserve earlier samples.")
    summary_path = args.output_dir / "bounds.json"
    stop_path = args.output_dir / "stop.request"
    stats = {
        m["id"]: {
            "id": m["id"], "name": m["name"],
            "center_tick": m["center_tick"], "direction": m["direction"],
            "samples": 0, "read_errors": 0, "last_error": None,
            "raw_min_tick": None, "raw_max_tick": None,
            "min_angle_deg": None, "max_angle_deg": None,
            "raw_tick_at_min_angle": None, "raw_tick_at_max_angle": None,
            "last_raw_tick": None, "last_angle_deg": None,
        } for m in motors
    }
    payload = {
        "schema_version": 1, "kind": "ankle_manual_travel_observations",
        "status": "starting", "started_at_utc": utc_now(),
        "calibration_captured_at_utc": calibration["captured_at_utc"],
        "calibration_updated_at_utc": calibration.get("updated_at_utc"),
        "angle_convention": calibration["direction_convention"],
        "port": args.port, "baudrate": calibration["baudrate"],
        "sample_rate_target_hz": args.hz, "maximum_duration_s": args.seconds,
        "method": "Torque disabled; operator manually moves ankles. No position commands.",
        "limits_applied": False,
        "note": "Observed extrema only; not independently validated safe travel limits.",
        "motors": list(stats.values()),
    }
    started = time.monotonic()

    def save():
        payload["updated_at_utc"] = utc_now()
        payload["elapsed_s"] = round(time.monotonic() - started, 3)
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(summary_path)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    save()
    last_raw = {}
    unwrapped = {}
    consecutive_errors = {m["id"]: 0 for m in motors}
    last_save = 0.0
    try:
        with samples_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["elapsed_s", "id", "raw_tick", "unwrapped_tick", "angle_deg"])
            with STSBus(args.port, calibration["baudrate"]) as bus:
                for motor in motors:
                    bus.disable_torque(motor["id"])
                payload["torque_off_acknowledged_ids"] = [m["id"] for m in motors]
                payload["status"] = "recording"
                save()
                print("Recording ankle IDs 13-16 with torque OFF.", flush=True)
                try:
                    while True:
                        cycle_started = time.monotonic()
                        if stop_path.exists():
                            payload["stop_reason"] = "user_requested"
                            break
                        if stop_signal is not None:
                            payload["stop_reason"] = f"signal_{stop_signal}"
                            break
                        if cycle_started - started >= args.seconds:
                            payload["stop_reason"] = "time_limit"
                            break
                        for motor in motors:
                            servo_id = motor["id"]
                            entry = stats[servo_id]
                            try:
                                raw = bus.read_position(servo_id)
                                if not 0 <= raw < 4096:
                                    raise ValueError(f"Position outside single-turn range: {raw}")
                            except Exception as exc:
                                entry["read_errors"] += 1
                                entry["last_error"] = str(exc)
                                consecutive_errors[servo_id] += 1
                                if consecutive_errors[servo_id] >= 20:
                                    raise RuntimeError(f"ID {servo_id}: 20 consecutive read failures") from exc
                                continue
                            consecutive_errors[servo_id] = 0
                            if servo_id not in last_raw:
                                offset = (raw - motor["center_tick"] + 2048) % 4096 - 2048
                                unwrapped[servo_id] = motor["center_tick"] + offset
                            else:
                                unwrapped[servo_id] += (raw - last_raw[servo_id] + 2048) % 4096 - 2048
                            last_raw[servo_id] = raw
                            angle = motor["direction"] * (unwrapped[servo_id] - motor["center_tick"]) * 360 / 4096
                            entry["samples"] += 1
                            entry["last_raw_tick"] = raw
                            entry["last_angle_deg"] = angle
                            entry["raw_min_tick"] = raw if entry["raw_min_tick"] is None else min(entry["raw_min_tick"], raw)
                            entry["raw_max_tick"] = raw if entry["raw_max_tick"] is None else max(entry["raw_max_tick"], raw)
                            if entry["min_angle_deg"] is None or angle < entry["min_angle_deg"]:
                                entry["min_angle_deg"] = angle
                                entry["raw_tick_at_min_angle"] = raw
                            if entry["max_angle_deg"] is None or angle > entry["max_angle_deg"]:
                                entry["max_angle_deg"] = angle
                                entry["raw_tick_at_max_angle"] = raw
                            writer.writerow([round(time.monotonic() - started, 4), servo_id, raw, unwrapped[servo_id], angle])
                        if cycle_started - last_save >= 1.0:
                            stream.flush()
                            save()
                            last_save = cycle_started
                        time.sleep(max(0.0, 1 / args.hz - (time.monotonic() - cycle_started)))
                    payload["status"] = "stopped"
                finally:
                    release_errors = {}
                    for motor in motors:
                        try:
                            bus.disable_torque(motor["id"])
                        except Exception as exc:
                            release_errors[str(motor["id"])] = str(exc)
                    payload["final_torque_off_errors"] = release_errors
                    if release_errors:
                        payload["status"] = "fault"
                        payload["error"] = "Could not acknowledge all final torque-off commands."
    except Exception as exc:
        payload["status"] = "fault"
        payload["error"] = str(exc)
        print(f"Recording stopped with an error: {exc}", flush=True)
    finally:
        payload["ended_at_utc"] = utc_now()
        save()
        print(f"Saved observations to {summary_path}", flush=True)
    return 1 if payload["status"] == "fault" else 0


if __name__ == "__main__":
    raise SystemExit(main())

