"""Measure one motor's zero-gravity Isaac drive response.

This diagnostic deliberately removes ground and stair contacts plus gravity.
The base can remain floating or be fixed. Hard joint limits, self-collision,
velocity limits, drive gains, and the requested effort cap remain active.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(ISAAC_DIR) not in sys.path:
    sys.path.insert(0, str(ISAAC_DIR))

from _quadruped_runtime import (  # noqa: E402
    EXPECTED_DOF_NAMES,
    MAX_NO_LOAD_VELOCITY_RAD_S,
    RATED_TORQUE_NM,
    STALL_TORQUE_NM,
    add_robot_reference,
)

parser = argparse.ArgumentParser(
    description="Probe one Drobot motor without gravity or contact."
)
parser.add_argument(
    "--base-mode",
    choices=("fixed", "floating"),
    default="fixed",
)
parser.add_argument(
    "--usd",
    default=None,
    help="Default selects the matching curated fixed/floating USDC.",
)
parser.add_argument(
    "--joint",
    default="front_left_hip_abduction",
)
parser.add_argument("--target-deg", type=float, default=20.0)
parser.add_argument("--duration-s", type=float, default=3.0)
parser.add_argument(
    "--torque-profile",
    choices=("rated", "stall"),
    default="rated",
)
parser.add_argument("--drive-stiffness", type=float, default=30.0)
parser.add_argument("--drive-damping", type=float, default=4.58366)
parser.add_argument(
    "--report",
    default=(
        "simulation/isaac/output/motor-physics-audit/"
        "zero-gravity-fixed-probe.json"
    ),
)
args, _ = parser.parse_known_args()

if args.joint not in EXPECTED_DOF_NAMES:
    parser.error(f"--joint is unknown: {args.joint}")
if args.duration_s <= 0.0:
    parser.error("--duration-s must be positive")
if args.drive_stiffness < 0.0 or args.drive_damping < 0.0:
    parser.error("Drive gains must be non-negative")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


default_usd = (
    "simulation/exports/isaac/quadruped_robot_fixed.usdc"
    if args.base_mode == "fixed"
    else "simulation/exports/isaac/quadruped_robot_floating.usdc"
)
usd_path = _resolve(args.usd or default_usd)
report_path = _resolve(args.report)
effort_cap_nm = (
    RATED_TORQUE_NM
    if args.torque_profile == "rated"
    else STALL_TORQUE_NM
)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from pxr import Gf, Sdf, UsdPhysics  # noqa: E402

PHYSICS_HZ = 120
CONTROL_HZ = 60


def _numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _finite(name: str, value) -> np.ndarray:
    # Isaac may expose a NumPy view backed by a live simulation tensor. Keep
    # measurements immutable so the baseline is not changed by later steps.
    result = _numpy(value).copy()
    if not np.isfinite(result).all():
        raise AssertionError(f"{name} contains non-finite values: {result}")
    return result


def _update(count: int) -> None:
    for _ in range(count):
        simulation_app.update()


def _raw_drive(joint_name: str) -> dict[str, object]:
    matches = [
        prim
        for prim in stage_utils.get_current_stage().Traverse()
        if prim.GetName() == joint_name
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one USD joint prim for {joint_name}: {matches}"
        )
    drive = UsdPhysics.DriveAPI.Get(matches[0], "angular")
    if not drive:
        raise AssertionError(f"Angular drive is missing: {joint_name}")
    return {
        "joint_prim": str(matches[0].GetPath()),
        "type": drive.GetTypeAttr().Get(),
        "stiffness_raw": float(drive.GetStiffnessAttr().Get()),
        "damping_raw": float(drive.GetDampingAttr().Get()),
        "max_force_raw": float(drive.GetMaxForceAttr().Get()),
        "target_position_deg_raw": float(
            drive.GetTargetPositionAttr().Get()
        ),
    }


report: dict[str, object] = {
    "status": "ERROR",
    "scope": (
        f"{args.base_mode.capitalize()} base, zero gravity, no ground, "
        "no stair. Joint limits, "
        "velocity limit, self-collision, drive gains, and effort cap remain."
    ),
    "isaac_sim_version": "6.0.1",
    "usd": str(usd_path),
    "joint": args.joint,
    "base_mode": args.base_mode,
    "target_deg": float(args.target_deg),
    "duration_s": float(args.duration_s),
    "torque_profile": args.torque_profile,
    "effort_cap_nm": effort_cap_nm,
    "requested_drive_stiffness": float(args.drive_stiffness),
    "requested_drive_damping": float(args.drive_damping),
}
exit_code = 1

try:
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)
    stage_utils.create_new_stage(template="sunlight")
    scene = UsdPhysics.Scene.Define(
        stage_utils.get_current_stage(),
        "/World/PhysicsScene",
    )
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(0.0)
    scene.GetPrim().CreateAttribute(
        "physxScene:timeStepsPerSecond",
        Sdf.ValueTypeNames.Int,
    ).Set(PHYSICS_HZ)
    report["asset_reference"] = add_robot_reference(
        stage_utils,
        str(usd_path),
        "/World/Robot",
    )
    robot = Articulation("/World/Robot", reset_xform_op_properties=True)
    app_utils.play()
    _update(10)

    dof_names = list(robot.dof_names)
    if len(dof_names) != 12 or set(dof_names) != EXPECTED_DOF_NAMES:
        raise AssertionError(f"Unexpected DOFs: {dof_names}")
    joint_index = dof_names.index(args.joint)
    lower, upper = robot.get_dof_limits()
    lower = _finite("joint lower limits", lower).reshape(-1)
    upper = _finite("joint upper limits", upper).reshape(-1)
    max_velocities = _finite(
        "joint max velocities",
        robot.get_dof_max_velocities(),
    ).reshape(-1)
    if max_velocities[joint_index] > MAX_NO_LOAD_VELOCITY_RAD_S + 1e-4:
        raise AssertionError("Motor speed exceeds the hardware contract")
    target_rad = math.radians(float(args.target_deg))
    if not lower[joint_index] < target_rad < upper[joint_index]:
        raise AssertionError(
            f"Target is outside hard limits: {lower[joint_index]}, "
            f"{target_rad}, {upper[joint_index]}"
        )

    report["raw_drive_before_override"] = _raw_drive(args.joint)
    robot.set_dof_max_efforts(
        np.full(robot.num_dofs, effort_cap_nm, dtype=np.float32)
    )
    robot.set_dof_gains(
        np.full(
            robot.num_dofs,
            float(args.drive_stiffness),
            dtype=np.float32,
        ),
        np.full(
            robot.num_dofs,
            float(args.drive_damping),
            dtype=np.float32,
        ),
    )
    report["raw_drive_after_override"] = _raw_drive(args.joint)
    applied_stiffness, applied_damping = robot.get_dof_gains()
    applied_stiffness = _finite(
        "applied stiffness",
        applied_stiffness,
    ).reshape(-1)
    applied_damping = _finite(
        "applied damping",
        applied_damping,
    ).reshape(-1)
    applied_effort = _finite(
        "applied effort",
        robot.get_dof_max_efforts(),
    ).reshape(-1)

    zero = np.zeros(robot.num_dofs, dtype=np.float32)
    if args.base_mode == "floating":
        robot.set_world_poses(
            positions=[[0.0, 0.0, 1.0]],
            orientations=[[1.0, 0.0, 0.0, 0.0]],
        )
        robot.set_velocities(
            linear_velocities=[[0.0, 0.0, 0.0]],
            angular_velocities=[[0.0, 0.0, 0.0]],
        )
    robot.set_dof_positions(zero)
    robot.set_dof_velocities(zero)
    robot.set_dof_position_targets(zero)
    _update(30)
    baseline = _finite(
        "baseline position",
        robot.get_dof_positions(),
    ).reshape(-1)
    command = zero.copy()
    command[joint_index] = target_rad
    samples: list[dict[str, float]] = []
    for step in range(round(args.duration_s * CONTROL_HZ)):
        robot.set_dof_position_targets(command)
        _update(1)
        position = float(
            _finite(
                "joint position",
                robot.get_dof_positions(),
            ).reshape(-1)[joint_index]
        )
        velocity = float(
            _finite(
                "joint velocity",
                robot.get_dof_velocities(),
            ).reshape(-1)[joint_index]
        )
        error = target_rad - position
        requested_pd = (
            float(args.drive_stiffness) * error
            - float(args.drive_damping) * velocity
        )
        samples.append(
            {
                "time_s": (step + 1) / CONTROL_HZ,
                "position_rad": position,
                "velocity_rad_s": velocity,
                "error_rad": error,
                "requested_pd_nm": requested_pd,
                "capped_pd_nm": min(
                    max(requested_pd, -effort_cap_nm),
                    effort_cap_nm,
                ),
            }
        )

    final = samples[-1]
    final_base_position, final_base_orientation = robot.get_world_poses()
    final_base_position = _finite(
        "final base position",
        final_base_position,
    ).reshape(-1, 3)[0]
    final_base_orientation = _finite(
        "final base orientation",
        final_base_orientation,
    ).reshape(-1, 4)[0]
    final_error_deg = abs(math.degrees(final["error_rad"]))
    time_to_90 = next(
        (
            sample["time_s"]
            for sample in samples
            if abs(sample["position_rad"] - baseline[joint_index])
            >= 0.9 * abs(target_rad - baseline[joint_index])
        ),
        None,
    )
    report.update(
        {
            "status": "PASS",
            "dof_order": dof_names,
            "joint_index": joint_index,
            "joint_lower_limit_deg": math.degrees(lower[joint_index]),
            "joint_upper_limit_deg": math.degrees(upper[joint_index]),
            "joint_max_velocity_rad_s": float(
                max_velocities[joint_index]
            ),
            "applied_effort_cap_nm": float(applied_effort[joint_index]),
            "applied_drive_stiffness": float(
                applied_stiffness[joint_index]
            ),
            "applied_drive_damping": float(
                applied_damping[joint_index]
            ),
            "baseline_deg": math.degrees(baseline[joint_index]),
            "final_measured_deg": math.degrees(final["position_rad"]),
            "final_error_deg": final_error_deg,
            "peak_abs_velocity_rad_s": max(
                abs(sample["velocity_rad_s"]) for sample in samples
            ),
            "peak_abs_requested_pd_nm": max(
                abs(sample["requested_pd_nm"]) for sample in samples
            ),
            "time_to_90_percent_s": time_to_90,
            "response_within_1deg": final_error_deg <= 1.0,
            "final_base_position_m": final_base_position.tolist(),
            "final_base_orientation_wxyz": (
                final_base_orientation.tolist()
            ),
            "samples": samples,
        }
    )
    exit_code = 0
except Exception as exc:
    report["status"] = "ERROR"
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_MOTOR_PROBE_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
