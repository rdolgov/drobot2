"""Pure target-state helpers for interactive stair leg control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Iterable, Mapping

from _quadruped_runtime import (
    LEGS,
    LINK_LENGTH_M,
    leg_ik,
    stance_by_name,
)

HARD_HIP_ABDUCTION_LIMIT_RAD = math.radians(25.0)
HARD_HIP_FLEXION_LIMIT_RAD = math.radians(60.0)
HARD_KNEE_LIMIT_RAD = math.radians(90.0)

LEG_SELECTION_KEYS = {
    "KEY_1": "front_left",
    "KEY_2": "front_right",
    "KEY_3": "rear_left",
    "KEY_4": "rear_right",
}

FOOT_MOTION_KEYS = frozenset({"W", "S", "E", "D", "Q", "A"})
MOTOR_MOTION_KEYS = frozenset({"UP", "DOWN"})

MOTOR_NUMBER_TO_JOINT = {
    1: "front_left_hip_abduction",
    2: "front_left_hip_flexion",
    3: "front_left_knee",
    4: "front_right_hip_abduction",
    5: "front_right_hip_flexion",
    6: "front_right_knee",
    7: "rear_left_hip_abduction",
    8: "rear_left_hip_flexion",
    9: "rear_left_knee",
    10: "rear_right_hip_abduction",
    11: "rear_right_hip_flexion",
    12: "rear_right_knee",
}
JOINT_TO_MOTOR_NUMBER = {
    joint_name: number
    for number, joint_name in MOTOR_NUMBER_TO_JOINT.items()
}
MOTOR_NUMBER_LIST = "\n".join(
    f"  {number:>2}  {joint_name}"
    for number, joint_name in MOTOR_NUMBER_TO_JOINT.items()
)

FOOT_CONTROL_HELP = """\
Click the viewport once, then hold:
  1 / 2 / 3 / 4   select front-left / front-right / rear-left / rear-right
  W / S           selected foot forward / backward
  E / D           selected foot up / down
  Q / A           selected hip-abduction angle + / -
  R               reset robot and every leg target
  Space           pause / resume physics
  C               print current targets and measured state
  X or Esc        save the session report and quit
"""

MOTOR_CONTROL_HELP = """\
Click the viewport once, then:
  0-9 then Enter  type and select motor number 1-12
  Backspace        correct the number being entered
  Up / Down       increase / decrease the selected motor target angle
  Z               set the selected motor target to zero degrees
  R               reset robot and all 12 motor targets to the standing pose
  Space           pause / resume physics
  C               print target angle, measured angle, foot load, and body state
  X or Esc        save the session report and quit
"""

# Kept as the public compatibility names for the original foot-space runner.
MOTION_KEYS = FOOT_MOTION_KEYS
CONTROL_HELP = FOOT_CONTROL_HELP


@dataclass(frozen=True)
class LegTarget:
    """One leg target expressed in its sagittal foot plane plus abduction."""

    down_m: float
    forward_m: float
    hip_abduction_rad: float


class ManualLegController:
    """Maintain safe, independently editable foot targets for four legs."""

    def __init__(
        self,
        *,
        stance_down_m: float,
        stance_fore_aft_m: float,
        stance_abduction_deg: float,
        joint_limit_margin_rad: float,
    ) -> None:
        self.mode = "foot"
        self.control_help = FOOT_CONTROL_HELP
        self.motion_keys = FOOT_MOTION_KEYS
        self.smoke_key = "E"
        self.stance_down_m = float(stance_down_m)
        self.stance_fore_aft_m = float(stance_fore_aft_m)
        self.stance_abduction_rad = math.radians(
            float(stance_abduction_deg)
        )
        self.joint_limit_margin_rad = float(joint_limit_margin_rad)
        if self.joint_limit_margin_rad <= 0.0:
            raise ValueError("joint_limit_margin_rad must be positive")
        self.selected_leg = "front_left"
        self.last_rejection: str | None = None
        self.targets: dict[str, LegTarget] = {}
        self.reset()

    def reset(self) -> None:
        self.targets = {
            leg: LegTarget(
                down_m=self.stance_down_m,
                forward_m=(
                    self.stance_fore_aft_m
                    if leg.startswith("front_")
                    else -self.stance_fore_aft_m
                ),
                hip_abduction_rad=self.stance_abduction_rad,
            )
            for leg in LEGS
        }
        self.selected_leg = "front_left"
        self.last_rejection = None
        for leg, target in self.targets.items():
            self._validate_target(leg, target)

    def select_from_key(self, key_name: str) -> bool:
        leg = LEG_SELECTION_KEYS.get(str(key_name))
        if leg is None:
            return False
        self.selected_leg = leg
        self.last_rejection = None
        return True

    def _validate_target(self, leg: str, target: LegTarget) -> None:
        margin = self.joint_limit_margin_rad
        if target.down_m <= 0.025:
            raise ValueError("foot target is too close to the hip")
        if math.hypot(target.down_m, target.forward_m) >= (
            2.0 * LINK_LENGTH_M - 0.002
        ):
            raise ValueError("foot target exceeds the two-link reach")
        hip, knee = leg_ik(leg, target.down_m, target.forward_m)
        if abs(target.hip_abduction_rad) >= (
            HARD_HIP_ABDUCTION_LIMIT_RAD - margin
        ):
            raise ValueError("hip-abduction target reaches its hard margin")
        if abs(hip) >= HARD_HIP_FLEXION_LIMIT_RAD - margin:
            raise ValueError("hip-flexion target reaches its hard margin")
        if abs(knee) >= HARD_KNEE_LIMIT_RAD - margin:
            raise ValueError("knee target reaches its hard margin")

    def advance(
        self,
        pressed_key_names: Iterable[str],
        *,
        dt_s: float,
        foot_speed_m_s: float,
        abduction_speed_rad_s: float,
        motor_speed_rad_s: float | None = None,
    ) -> bool:
        """Apply held-key motion and reject only axes that leave safe IK."""

        dt = float(dt_s)
        linear_delta = float(foot_speed_m_s) * dt
        angular_delta = float(abduction_speed_rad_s) * dt
        if dt <= 0.0 or linear_delta <= 0.0 or angular_delta <= 0.0:
            raise ValueError("Control timestep and speeds must be positive")
        keys = frozenset(str(key) for key in pressed_key_names)
        axis_deltas = (
            ("forward_m", linear_delta * (("W" in keys) - ("S" in keys))),
            ("down_m", linear_delta * (("D" in keys) - ("E" in keys))),
            (
                "hip_abduction_rad",
                angular_delta * (("Q" in keys) - ("A" in keys)),
            ),
        )
        changed = False
        target = self.targets[self.selected_leg]
        rejections: list[str] = []
        for field, delta in axis_deltas:
            if delta == 0.0:
                continue
            candidate = replace(
                target,
                **{field: float(getattr(target, field)) + delta},
            )
            try:
                self._validate_target(self.selected_leg, candidate)
            except ValueError as exc:
                rejections.append(f"{field}: {exc}")
                continue
            target = candidate
            changed = True
        self.targets[self.selected_leg] = target
        self.last_rejection = "; ".join(rejections) if rejections else None
        return changed

    def joint_targets_by_name(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for leg in LEGS:
            target = self.targets[leg]
            hip, knee = leg_ik(
                leg,
                target.down_m,
                target.forward_m,
            )
            result[f"{leg}_hip_abduction"] = target.hip_abduction_rad
            result[f"{leg}_hip_flexion"] = hip
            result[f"{leg}_knee"] = knee
        return result

    def snapshot(self) -> dict[str, object]:
        return {
            "selected_leg": self.selected_leg,
            "targets": {
                leg: asdict(self.targets[leg])
                for leg in LEGS
            },
            "joint_targets_rad": self.joint_targets_by_name(),
            "last_rejection": self.last_rejection,
        }


def controller_from_experiment(
    experiment: Mapping[str, object],
) -> ManualLegController:
    stance = dict(experiment["stance"])
    acceptance = dict(experiment["acceptance"])
    return ManualLegController(
        stance_down_m=float(stance["down_m"]),
        stance_fore_aft_m=float(stance["fore_aft_m"]),
        stance_abduction_deg=float(stance["abduction_deg"]),
        joint_limit_margin_rad=float(
            acceptance["joint_limit_margin_rad"]
        ),
    )


class ManualMotorController:
    """Maintain direct, hard-limit-safe targets for all 12 servo joints."""

    LIMIT_RAD_BY_KIND = {
        "hip_abduction": HARD_HIP_ABDUCTION_LIMIT_RAD,
        "hip_flexion": HARD_HIP_FLEXION_LIMIT_RAD,
        "knee": HARD_KNEE_LIMIT_RAD,
    }

    def __init__(
        self,
        *,
        stance_down_m: float,
        stance_fore_aft_m: float,
        stance_abduction_deg: float,
        joint_limit_margin_rad: float,
    ) -> None:
        self.mode = "motor"
        self.control_help = MOTOR_CONTROL_HELP
        self.motion_keys = MOTOR_MOTION_KEYS
        self.smoke_key = "UP"
        self.stance_down_m = float(stance_down_m)
        self.stance_fore_aft_m = float(stance_fore_aft_m)
        self.stance_abduction_deg = float(stance_abduction_deg)
        self.joint_limit_margin_rad = float(joint_limit_margin_rad)
        if self.joint_limit_margin_rad <= 0.0:
            raise ValueError("joint_limit_margin_rad must be positive")
        self.selected_motor_number = 1
        self.selection_buffer = ""
        self.last_rejection: str | None = None
        self.targets_rad: dict[str, float] = {}
        self.reset()

    @property
    def selected_joint_name(self) -> str:
        return MOTOR_NUMBER_TO_JOINT[self.selected_motor_number]

    @property
    def selected_leg(self) -> str:
        joint_name = self.selected_joint_name
        return next(leg for leg in LEGS if joint_name.startswith(f"{leg}_"))

    @property
    def selected_motor(self) -> str:
        return self.selected_joint_name.removeprefix(
            f"{self.selected_leg}_"
        )

    def reset(self) -> None:
        self.targets_rad = dict(
            stance_by_name(
                down_m=self.stance_down_m,
                fore_aft_m=self.stance_fore_aft_m,
                abduction_deg=self.stance_abduction_deg,
            )
        )
        self.selected_motor_number = 1
        self.selection_buffer = ""
        self.last_rejection = None
        for name, value in self.targets_rad.items():
            self._validate_joint_target(name, value)

    def select_from_key(self, key_name: str) -> bool:
        key = str(key_name)
        if key.startswith("KEY_") and key[-1:].isdigit():
            digit = key[-1]
            candidate_buffer = (self.selection_buffer + digit)[-2:]
            if int(candidate_buffer) <= 12:
                self.selection_buffer = candidate_buffer
                self.last_rejection = None
            else:
                self.last_rejection = (
                    f"Motor #{candidate_buffer} is invalid; use 1-12"
                )
            return True
        if key == "BACKSPACE":
            self.selection_buffer = self.selection_buffer[:-1]
            self.last_rejection = None
            return True
        if key in ("ENTER", "NUMPAD_ENTER"):
            if not self.selection_buffer:
                self.last_rejection = "Type motor number 1-12 before Enter"
                return True
            number = int(self.selection_buffer)
            self.selection_buffer = ""
            if number not in MOTOR_NUMBER_TO_JOINT:
                self.last_rejection = f"Motor #{number} is invalid; use 1-12"
                return True
            self.selected_motor_number = number
            self.last_rejection = None
            return True
        return False

    def _validate_joint_target(self, joint_name: str, value_rad: float) -> None:
        motor_kind = next(
            (
                kind
                for kind in self.LIMIT_RAD_BY_KIND
                if joint_name.endswith(kind)
            ),
            None,
        )
        if motor_kind is None:
            raise ValueError(f"Unknown motor target: {joint_name}")
        limit = self.LIMIT_RAD_BY_KIND[motor_kind]
        allowed = limit - self.joint_limit_margin_rad
        if abs(float(value_rad)) >= allowed:
            raise ValueError(
                f"{joint_name} target reaches hard margin "
                f"({math.degrees(allowed):.2f} deg)"
            )

    def zero_selected(self) -> None:
        self.targets_rad[self.selected_joint_name] = 0.0
        self.last_rejection = None

    def advance(
        self,
        pressed_key_names: Iterable[str],
        *,
        dt_s: float,
        foot_speed_m_s: float,
        abduction_speed_rad_s: float,
        motor_speed_rad_s: float | None = None,
    ) -> bool:
        """Change the selected motor's target directly in radians."""

        del foot_speed_m_s, abduction_speed_rad_s
        dt = float(dt_s)
        speed = float(
            motor_speed_rad_s
            if motor_speed_rad_s is not None
            else math.radians(25.0)
        )
        if dt <= 0.0 or speed <= 0.0:
            raise ValueError("Control timestep and motor speed must be positive")
        keys = frozenset(str(key) for key in pressed_key_names)
        direction = ("UP" in keys) - ("DOWN" in keys)
        if direction == 0:
            self.last_rejection = None
            return False
        joint_name = self.selected_joint_name
        candidate = self.targets_rad[joint_name] + direction * speed * dt
        try:
            self._validate_joint_target(joint_name, candidate)
        except ValueError as exc:
            self.last_rejection = str(exc)
            return False
        self.targets_rad[joint_name] = candidate
        self.last_rejection = None
        return True

    def joint_targets_by_name(self) -> dict[str, float]:
        return dict(self.targets_rad)

    def snapshot(self) -> dict[str, object]:
        return {
            "selected_motor_number": self.selected_motor_number,
            "selected_leg": self.selected_leg,
            "selected_motor": self.selected_motor,
            "selected_joint_name": self.selected_joint_name,
            "selection_buffer": self.selection_buffer,
            "motor_number_to_joint": dict(MOTOR_NUMBER_TO_JOINT),
            "targets_rad": dict(self.targets_rad),
            "targets_deg": {
                name: math.degrees(value)
                for name, value in self.targets_rad.items()
            },
            "last_rejection": self.last_rejection,
        }


def motor_controller_from_experiment(
    experiment: Mapping[str, object],
) -> ManualMotorController:
    stance = dict(experiment["stance"])
    acceptance = dict(experiment["acceptance"])
    return ManualMotorController(
        stance_down_m=float(stance["down_m"]),
        stance_fore_aft_m=float(stance["fore_aft_m"]),
        stance_abduction_deg=float(stance["abduction_deg"]),
        joint_limit_margin_rad=float(
            acceptance["joint_limit_margin_rad"]
        ),
    )
