"""Attach measured ankle records without rewriting the three-motor leg files."""

import json
import math
from dataclasses import replace
from pathlib import Path

from drobot_leg_testbed.model import MotorCalibration, MotorConfig


def attach_ankles(profiles, settings: dict, manifest_dir: Path):
    calibration_path = manifest_dir / settings["calibration"]
    observation_path = manifest_dir / settings["travel_observations"]
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    observations = json.loads(observation_path.read_text(encoding="utf-8"))
    margin = float(settings.get("limit_margin_deg", 5.0))
    if not math.isfinite(margin) or margin < 5.0:
        raise ValueError("Ankle travel margin must be at least five degrees")
    if calibration.get("direction_status") != "measured_from_manual_deflections":
        raise ValueError("Ankle directions must be measured before motor control")
    if observations.get("status") != "stopped":
        raise ValueError("Finish ankle travel capture before motor control")
    centers = {entry["id"]: entry for entry in calibration["motors"]}
    bounds = {entry["id"]: entry for entry in observations["motors"]}
    if set(centers) != {13, 14, 15, 16} or set(bounds) != set(centers):
        raise ValueError("Ankle records must contain IDs 13 through 16")
    result = []
    for profile in profiles:
        servo_id = 12 + profile.number
        entry, observed = centers[servo_id], bounds[servo_id]
        if entry["name"] != f"{profile.corner}_ankle_pitch":
            raise ValueError("Ankle corner mapping does not match the manifest")
        center, direction = entry["center_tick"], entry["direction"]
        if not isinstance(center, int) or not 0 <= center < 4096 or direction not in (-1, 1):
            raise ValueError("Invalid ankle center or direction")
        if observed["center_tick"] != center or observed["direction"] != direction:
            raise ValueError("Ankle travel must be recaptured after calibration changes")
        low = float(observed["min_angle_deg"]) + margin
        high = float(observed["max_angle_deg"]) - margin
        if not (math.isfinite(low) and math.isfinite(high) and -180 < low < 0 < high < 180):
            raise ValueError("Invalid ankle software travel envelope")
        motor = MotorConfig(4, "ankle_pitch", servo_id, direction, low, high)
        neutral = MotorCalibration(4, "ankle_pitch", servo_id, center)
        result.append(replace(
            profile,
            config=replace(profile.config, motors=profile.config.motors + (motor,)),
            calibration=replace(profile.calibration, motors=profile.calibration.motors + (neutral,)),
        ))
    return result
