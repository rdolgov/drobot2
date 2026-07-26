"""Headless Isaac Sim 6.0 validation for the drobot quadruped.

``fixed`` mode checks the imported articulation, named joints, limits, masses,
and independent drive response.  ``floating`` mode performs an Earth-gravity
drop into the conservative standing pose, then measures support, drift, tilt,
tracking error, and servo effort utilization.

Run with Isaac Sim's bundled ``python.bat`` and the matching fixed/floating USD
created by ``import_quadruped.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import traceback

import numpy as np
from _quadruped_runtime import (
    DEFAULT_ABDUCTION_DEG,
    DEFAULT_STANCE_DOWN_M,
    DEFAULT_STANCE_FORE_AFT_M,
    EXPECTED_DOF_NAMES,
    MAX_NO_LOAD_VELOCITY_RAD_S,
    add_robot_reference,
    body_tilt_deg,
    stance_by_name,
    targets_for_order,
    torque_cap_nm,
)
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(
    description="Validate the drobot quadruped USD under Isaac Sim physics."
)
parser.add_argument("--usd", required=True, help="Imported root USD/USDA asset")
parser.add_argument("--mode", choices=("fixed", "floating"), required=True)
parser.add_argument("--report", required=True, help="Output JSON report")
parser.add_argument("--screenshot", default=None, help="Optional final PNG")
parser.add_argument("--torque-cap", choices=("rated", "stall"), default="rated")
parser.add_argument(
    "--effort-limit-nm",
    type=float,
    default=None,
    help="Positive custom per-joint cap; overrides --torque-cap",
)
parser.add_argument("--drive-stiffness", type=float, default=30.0)
parser.add_argument("--drive-damping", type=float, default=4.58366)
parser.add_argument("--start-z", type=float, default=0.420)
parser.add_argument("--settle-seconds", type=float, default=10.0)
parser.add_argument("--stance-down", type=float, default=DEFAULT_STANCE_DOWN_M)
parser.add_argument(
    "--stance-fore-aft",
    type=float,
    default=DEFAULT_STANCE_FORE_AFT_M,
)
parser.add_argument("--abduction-deg", type=float, default=DEFAULT_ABDUCTION_DEG)
parser.add_argument("--min-base-z", type=float, default=0.200)
parser.add_argument("--max-tilt-deg", type=float, default=15.0)
parser.add_argument("--max-joint-error-rad", type=float, default=0.15)
args, _ = parser.parse_known_args()

if args.drive_stiffness < 0.0 or args.drive_damping < 0.0:
    parser.error("Drive stiffness and damping must be non-negative")
if args.start_z <= 0.0 or args.settle_seconds <= 0.0:
    parser.error("Start height and settle duration must be positive")
if args.min_base_z <= 0.0 or args.max_tilt_deg <= 0.0:
    parser.error("Validation thresholds must be positive")

torque_profile, effort_cap_nm = torque_cap_nm(
    args.torque_cap,
    args.effort_limit_nm,
)

simulation_app = SimulationApp(
    {
        "headless": True,
        "width": 1280,
        "height": 720,
    }
)

# Omniverse imports must follow SimulationApp construction.
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.objects import GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

EARTH_GRAVITY_M_S2 = 9.81
PHYSICS_HZ = 120
APPLICATION_HZ = 60

STATIC_FRICTION = 0.90
DYNAMIC_FRICTION = 0.75
RESTITUTION = 0.02
CONTACT_STIFFNESS_N_M = 12000.0
CONTACT_DAMPING_N_S_M = 45.0


def _numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _flat(value) -> np.ndarray:
    return _numpy(value).reshape(-1)


def _finite(name: str, value) -> np.ndarray:
    array = _numpy(value)
    if not np.isfinite(array).all():
        raise AssertionError(f"{name} contains non-finite values: {array}")
    return array


def _step(count: int) -> None:
    for _ in range(count):
        simulation_app.update()


def _configure_and_verify_physics() -> dict:
    stage = stage_utils.get_current_stage()
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(EARTH_GRAVITY_M_S2)
    scene.GetPrim().CreateAttribute(
        "physxScene:timeStepsPerSecond",
        Sdf.ValueTypeNames.Int,
    ).Set(PHYSICS_HZ)

    direction = np.asarray(scene.GetGravityDirectionAttr().Get(), dtype=float)
    magnitude = float(scene.GetGravityMagnitudeAttr().Get())
    gravity_vector = direction * magnitude
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    steps_per_second = int(scene.GetPrim().GetAttribute("physxScene:timeStepsPerSecond").Get())
    if not np.allclose(
        gravity_vector,
        [0.0, 0.0, -EARTH_GRAVITY_M_S2],
        atol=1e-6,
    ):
        raise AssertionError(f"Earth gravity was not authored: {gravity_vector}")
    if steps_per_second != PHYSICS_HZ:
        raise AssertionError(f"Unexpected physics rate: {steps_per_second}")
    if not math.isclose(meters_per_unit, 1.0, abs_tol=1e-9):
        raise AssertionError(f"Stage is not meter-scaled: {meters_per_unit}")
    return {
        "scene_path": "/World/PhysicsScene",
        "gravity_vector_m_s2": gravity_vector.tolist(),
        "gravity_magnitude_m_s2": magnitude,
        "meters_per_unit": meters_per_unit,
        "physics_steps_per_second": steps_per_second,
        "application_updates_per_second_assumed": APPLICATION_HZ,
        "gravity_explicitly_authored": True,
    }


def _apply_contact_material() -> dict:
    """Bind one documented provisional contact model to robot and ground."""
    stage = stage_utils.get_current_stage()
    material = UsdShade.Material.Define(
        stage,
        "/World/Materials/PrintedPlaContact",
    )
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(STATIC_FRICTION)
    material_api.CreateDynamicFrictionAttr().Set(DYNAMIC_FRICTION)
    material_api.CreateRestitutionAttr().Set(RESTITUTION)
    physx_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_api.CreateCompliantContactStiffnessAttr().Set(CONTACT_STIFFNESS_N_M)
    physx_api.CreateCompliantContactDampingAttr().Set(CONTACT_DAMPING_N_S_M)

    bound_paths: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if path.startswith("/World/Robot") or path.startswith("/World/GroundPlane"):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )
            bound_paths.append(path)
    if not any(path.startswith("/World/Robot") for path in bound_paths):
        raise AssertionError("No robot collision received the contact material")
    if not any(path.startswith("/World/GroundPlane") for path in bound_paths):
        raise AssertionError("Ground collision did not receive contact material")
    return {
        "material_path": str(material.GetPath()),
        "static_friction": STATIC_FRICTION,
        "dynamic_friction": DYNAMIC_FRICTION,
        "restitution": RESTITUTION,
        "compliant_contact_stiffness_n_m": CONTACT_STIFFNESS_N_M,
        "compliant_contact_damping_n_s_m": CONTACT_DAMPING_N_S_M,
        "bound_collision_count": len(bound_paths),
        "status": "provisional_until_printed_fork_tip_and_floor_are_measured",
    }


def _stage_counts() -> dict[str, int]:
    stage = stage_utils.get_current_stage()
    return {
        "articulation_roots": sum(
            prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in stage.Traverse()
        ),
        "rigid_bodies": sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stage.Traverse()),
        "collision_prims": sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse()),
        "revolute_joints": sum(prim.IsA(UsdPhysics.RevoluteJoint) for prim in stage.Traverse()),
        "fixed_joints": sum(prim.IsA(UsdPhysics.FixedJoint) for prim in stage.Traverse()),
        "physics_scenes": sum(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse()),
    }


def _set_drives(robot: Articulation) -> dict:
    count = robot.num_dofs
    requested_efforts = np.full(count, effort_cap_nm, dtype=np.float32)
    stiffness = np.full(count, args.drive_stiffness, dtype=np.float32)
    damping = np.full(count, args.drive_damping, dtype=np.float32)
    original_efforts = _finite(
        "URDF max efforts",
        robot.get_dof_max_efforts(),
    ).reshape(-1)
    robot.set_dof_max_efforts(requested_efforts)
    robot.set_dof_gains(stiffness, damping)
    applied_efforts = _finite(
        "applied max efforts",
        robot.get_dof_max_efforts(),
    ).reshape(-1)
    applied_stiffness, applied_damping = robot.get_dof_gains()
    applied_stiffness = _finite(
        "applied stiffness",
        applied_stiffness,
    ).reshape(-1)
    applied_damping = _finite(
        "applied damping",
        applied_damping,
    ).reshape(-1)
    if not np.allclose(applied_efforts, effort_cap_nm, atol=1e-4):
        raise AssertionError(f"Could not apply {effort_cap_nm} N*m effort cap: {applied_efforts}")
    return {
        "profile": torque_profile,
        "per_joint_cap_nm": effort_cap_nm,
        "urdf_imported_max_efforts_nm": original_efforts.tolist(),
        "applied_max_efforts_nm": applied_efforts.tolist(),
        "drive_stiffness_nm_rad": applied_stiffness.tolist(),
        "drive_damping_nm_s_rad": applied_damping.tolist(),
        "rated_profile_is_sustainable_test": torque_profile == "rated",
        "stall_profile_is_short_peak_only": torque_profile == "stall",
    }


def _check_structure(robot: Articulation, report: dict) -> list[str]:
    dof_names = list(robot.dof_names)
    if len(dof_names) != 12 or set(dof_names) != EXPECTED_DOF_NAMES:
        raise AssertionError(f"Unexpected DOF names: {dof_names}")
    if robot.num_dofs != 12:
        raise AssertionError(f"Expected 12 DOFs, got {robot.num_dofs}")
    if robot.num_links != 13:
        raise AssertionError(f"Expected 13 links, got {robot.num_links}")
    link_names = list(robot.link_names)
    if "base_link" not in link_names:
        raise AssertionError(f"base_link is absent: {link_names}")

    lower, upper = robot.get_dof_limits()
    lower = _finite("lower limits", lower).reshape(-1)
    upper = _finite("upper limits", upper).reshape(-1)
    if not np.all(lower < upper):
        raise AssertionError(f"Every joint must have an ordered finite range: {lower}, {upper}")
    max_velocities = _finite(
        "max joint velocities",
        robot.get_dof_max_velocities(),
    ).reshape(-1)
    if np.any(max_velocities <= 0.0):
        raise AssertionError(f"Invalid maximum joint velocity: {max_velocities}")
    if np.any(max_velocities > MAX_NO_LOAD_VELOCITY_RAD_S + 1e-3):
        raise AssertionError(
            f"URDF joint velocity exceeds the verified ST3215 no-load speed: {max_velocities}"
        )

    link_masses = _finite("link masses", robot.get_link_masses()).reshape(-1)
    if len(link_masses) != 13 or np.any(link_masses <= 0.0):
        raise AssertionError(f"Every physical link needs positive mass: {link_masses}")

    report.update(
        {
            "dof_names": dof_names,
            "dof_order_note": "Commands are mapped by joint name, never array position.",
            "link_names": link_names,
            "joint_lower_limits_rad": lower.tolist(),
            "joint_upper_limits_rad": upper.tolist(),
            "joint_max_velocities_rad_s": max_velocities.tolist(),
            "link_masses_kg": link_masses.tolist(),
            "total_articulated_mass_kg": float(np.sum(link_masses)),
        }
    )
    return dof_names


def _assert_target_inside_limits(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    label: str,
) -> None:
    if np.any(target <= lower + 1e-4) or np.any(target >= upper - 1e-4):
        raise AssertionError(
            f"{label} target is outside or on a hard joint limit: "
            f"target={target}, lower={lower}, upper={upper}"
        )


def _sample_efforts(
    robot: Articulation,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    positions = _finite("DOF positions", robot.get_dof_positions()).reshape(-1).copy()
    velocities = _finite("DOF velocities", robot.get_dof_velocities()).reshape(-1).copy()
    requested_pd = args.drive_stiffness * (target - positions) - args.drive_damping * velocities
    result = {
        "positions": positions,
        "velocities": velocities,
        "requested_pd_nm": requested_pd,
        "capped_pd_nm": np.clip(requested_pd, -effort_cap_nm, effort_cap_nm),
    }
    try:
        result["reported_drive_effort_nm"] = (
            _finite(
                "reported drive effort",
                robot.get_dof_efforts(),
            )
            .reshape(-1)
            .copy()
        )
    except Exception:
        pass
    try:
        result["projected_joint_load_nm"] = (
            _finite(
                "projected joint load",
                robot.get_dof_projected_joint_forces(),
            )
            .reshape(-1)
            .copy()
        )
    except Exception:
        pass
    return result


def _summarize_efforts(samples: list[dict[str, np.ndarray]]) -> dict:
    summary: dict[str, object] = {"sample_count": len(samples)}
    if not samples:
        return summary
    for key in (
        "requested_pd_nm",
        "capped_pd_nm",
        "reported_drive_effort_nm",
        "projected_joint_load_nm",
    ):
        available = [sample[key] for sample in samples if key in sample]
        if not available:
            continue
        values = np.vstack(available)
        peak = np.max(np.abs(values), axis=0)
        summary[f"peak_abs_{key}"] = peak.tolist()
        summary[f"peak_abs_{key}_all_joints"] = float(np.max(peak))
    requested = np.vstack([sample["requested_pd_nm"] for sample in samples])
    summary["pd_saturation_sample_fraction"] = float(np.mean(np.abs(requested) >= effort_cap_nm))
    summary["peak_requested_to_cap_ratio"] = float(np.max(np.abs(requested)) / effort_cap_nm)
    return summary


def _reset_pose(
    robot: Articulation,
    target: np.ndarray,
    *,
    base_z: float | None,
) -> None:
    if base_z is not None:
        robot.set_world_poses(
            positions=[[0.0, 0.0, base_z]],
            orientations=[[1.0, 0.0, 0.0, 0.0]],
        )
        robot.set_velocities(
            linear_velocities=[[0.0, 0.0, 0.0]],
            angular_velocities=[[0.0, 0.0, 0.0]],
        )
    robot.set_dof_positions(target)
    robot.set_dof_velocities(np.zeros(robot.num_dofs, dtype=np.float32))
    robot.set_dof_position_targets(target)


def _run_fixed(
    robot: Articulation,
    dof_names: list[str],
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    report: dict,
) -> None:
    _reset_pose(robot, target, base_z=None)
    _step(30)
    independent_checks = []
    for index, name in enumerate(dof_names):
        _reset_pose(robot, target, base_z=None)
        _step(15)
        baseline = (
            _finite(
                f"{name} baseline positions",
                robot.get_dof_positions(),
            )
            .reshape(-1)
            .copy()
        )

        direction = 1.0
        if target[index] + 0.10 >= upper[index] - 0.02:
            direction = -1.0
        command = target.copy()
        command[index] += direction * 0.10
        _assert_target_inside_limits(command, lower, upper, name)
        for _ in range(150):
            robot.set_dof_position_targets(command)
            _step(1)
        actual = _finite(
            f"{name} driven positions",
            robot.get_dof_positions(),
        ).reshape(-1)
        selected_motion = direction * (actual[index] - baseline[index])
        selected_error = abs(float(actual[index] - command[index]))
        other_motion = np.delete(np.abs(actual - baseline), index)
        passed = (
            selected_motion > 0.025 and selected_error < 0.15 and float(np.max(other_motion)) < 0.12
        )
        result = {
            "name": name,
            "command_delta_rad": direction * 0.10,
            "actual_delta_rad": float(actual[index] - baseline[index]),
            "target_error_rad": selected_error,
            "other_joint_max_motion_rad": float(np.max(other_motion)),
            "passed": passed,
        }
        independent_checks.append(result)
        if not passed:
            raise AssertionError(f"Independent drive response failed for {name}: {result}")

    _reset_pose(robot, target, base_z=None)
    effort_samples = []
    for step_index in range(300):
        robot.set_dof_position_targets(target)
        _step(1)
        if step_index % 6 == 0:
            effort_samples.append(_sample_efforts(robot, target))
    actual = _finite(
        "fixed stance positions",
        robot.get_dof_positions(),
    ).reshape(-1)
    max_error = float(np.max(np.abs(actual - target)))
    if max_error >= args.max_joint_error_rad:
        raise AssertionError(f"Fixed stance did not converge: max error={max_error:.6f} rad")
    base_position, base_orientation = robot.get_world_poses()
    base_position = _finite(
        "fixed base position",
        base_position,
    ).reshape(-1, 3)[0]
    base_orientation = _finite(
        "fixed base orientation",
        base_orientation,
    ).reshape(-1, 4)[0]
    if np.linalg.norm(base_position) > 1e-4:
        raise AssertionError(f"Fixed base moved: {base_position}")

    report.update(
        {
            "independent_joint_checks": independent_checks,
            "stance_target_rad": target.tolist(),
            "stance_actual_rad": actual.tolist(),
            "stance_max_abs_error_rad": max_error,
            "effort_metrics": _summarize_efforts(effort_samples),
            "final_base_position_m": base_position.tolist(),
            "final_base_orientation_wxyz": base_orientation.tolist(),
        }
    )


def _run_floating(
    robot: Articulation,
    target: np.ndarray,
    report: dict,
) -> None:
    _reset_pose(robot, target, base_z=args.start_z)

    early_z = []
    positions_log = []
    tilts_log = []
    effort_samples = []
    for _step_index in range(12):
        position, orientation = robot.get_world_poses()
        position = _finite(
            "early floating base position",
            position,
        ).reshape(-1, 3)[0]
        orientation = _finite(
            "early floating base orientation",
            orientation,
        ).reshape(-1, 4)[0]
        early_z.append(float(position[2]))
        positions_log.append(position.tolist())
        tilts_log.append(body_tilt_deg(orientation))
        robot.set_dof_position_targets(target)
        _step(1)

    settle_steps = max(
        12,
        int(math.ceil(args.settle_seconds * APPLICATION_HZ)),
    )
    for step_index in range(12, settle_steps):
        robot.set_dof_position_targets(target)
        _step(1)
        if step_index % 6 == 0:
            position, orientation = robot.get_world_poses()
            position = _finite(
                "floating base position log",
                position,
            ).reshape(-1, 3)[0]
            orientation = _finite(
                "floating base orientation log",
                orientation,
            ).reshape(-1, 4)[0]
            positions_log.append(position.tolist())
            tilts_log.append(body_tilt_deg(orientation))
            effort_samples.append(_sample_efforts(robot, target))

    positions = _finite(
        "floating DOF positions",
        robot.get_dof_positions(),
    ).reshape(-1)
    velocities = _finite(
        "floating DOF velocities",
        robot.get_dof_velocities(),
    ).reshape(-1)
    base_position, base_orientation = robot.get_world_poses()
    base_position = _finite(
        "floating final base position",
        base_position,
    ).reshape(-1, 3)[0]
    base_orientation = _finite(
        "floating final base orientation",
        base_orientation,
    ).reshape(-1, 4)[0]

    dt = 1.0 / APPLICATION_HZ
    acceleration_samples = [
        (early_z[index + 1] - 2.0 * early_z[index] + early_z[index - 1]) / (dt * dt)
        for index in range(1, min(5, len(early_z) - 1))
    ]
    measured_acceleration = float(np.median(acceleration_samples))
    total_drop = float(early_z[0] - base_position[2])
    final_tilt = body_tilt_deg(base_orientation)
    max_tilt = max(tilts_log + [final_tilt])
    max_joint_error = float(np.max(np.abs(positions - target)))
    base_log = np.asarray(positions_log, dtype=float)
    max_lateral_drift = float(np.max(np.abs(base_log[:, 1])))

    if early_z[0] - min(early_z) <= 0.015:
        raise AssertionError(f"Robot did not visibly accelerate downward: {early_z}")
    if not -11.8 <= measured_acceleration <= -7.5:
        raise AssertionError(
            f"Measured early acceleration is not Earth-like: {measured_acceleration:.6f} m/s^2"
        )
    if total_drop <= 0.020:
        raise AssertionError(f"Robot did not drop onto its contacts: {total_drop}")
    if not args.min_base_z <= base_position[2] < args.start_z:
        raise AssertionError(f"Robot is not supported above the floor: {base_position}")
    if max_tilt >= args.max_tilt_deg:
        raise AssertionError(f"Robot tipped too far: {max_tilt:.3f} degrees")
    if max_joint_error >= args.max_joint_error_rad:
        raise AssertionError(f"Standing joint tracking failed: {max_joint_error:.6f} rad")

    report.update(
        {
            "settle_steps": settle_steps,
            "settle_seconds": settle_steps / APPLICATION_HZ,
            "drop_start_z_m": early_z[0],
            "early_base_z_m": early_z,
            "early_freefall_acceleration_samples_m_s2": acceleration_samples,
            "measured_early_freefall_acceleration_m_s2": measured_acceleration,
            "total_drop_to_settled_pose_m": total_drop,
            "stance_target_rad": target.tolist(),
            "final_dof_positions_rad": positions.tolist(),
            "max_abs_joint_tracking_error_rad": max_joint_error,
            "max_abs_dof_velocity_rad_s": float(np.max(np.abs(velocities))),
            "final_base_position_m": base_position.tolist(),
            "final_base_orientation_wxyz": base_orientation.tolist(),
            "final_body_tilt_deg": final_tilt,
            "max_body_tilt_deg": max_tilt,
            "max_abs_lateral_drift_m": max_lateral_drift,
            "base_position_samples_m": positions_log,
            "effort_metrics": _summarize_efforts(effort_samples),
        }
    )


def _capture_screenshot(path: str, robot: Articulation) -> None:
    position, _ = robot.get_world_poses()
    position = _finite(
        "screenshot base position",
        position,
    ).reshape(-1, 3)[0]
    set_camera_view(
        eye=[
            float(position[0] + 0.72),
            float(position[1] - 0.82),
            float(position[2] + 0.34),
        ],
        target=[
            float(position[0]),
            float(position[1]),
            float(position[2] - 0.14),
        ],
        camera_prim_path="/OmniverseKit_Persp",
    )
    _step(45)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Isaac Sim has no active viewport")
    absolute_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    task = asyncio.ensure_future(
        capture_viewport_to_file(
            viewport,
            file_path=absolute_path,
            is_hdr=False,
        ).wait_for_result()
    )
    for _ in range(300):
        simulation_app.update()
        if task.done():
            break
    if not task.done():
        raise TimeoutError("Timed out waiting for Isaac viewport capture")
    task.result()
    for _ in range(120):
        if os.path.isfile(absolute_path) and os.path.getsize(absolute_path) > 0:
            break
        simulation_app.update()
    if not os.path.isfile(absolute_path) or os.path.getsize(absolute_path) == 0:
        raise RuntimeError(f"Isaac Sim did not create a usable PNG: {path}")


report = {
    "status": "FAIL",
    "isaac_sim_version": "6.0.1",
    "mode": args.mode,
    "usd": os.path.abspath(args.usd),
    "servo_model": "Feetech ST-3215-C018 / Waveshare ST3215 12 V",
    "torque_profile": torque_profile,
    "effort_cap_nm": effort_cap_nm,
}
exit_code = 1

try:
    if not os.path.isfile(args.usd):
        raise FileNotFoundError(args.usd)

    stage_utils.create_new_stage(template="sunlight")
    report["physics"] = _configure_and_verify_physics()
    if args.mode == "floating":
        GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, 0.0])

    report["asset_reference"] = add_robot_reference(
        stage_utils,
        args.usd,
        "/World/Robot",
    )
    if args.mode == "floating":
        report["contact_material"] = _apply_contact_material()

    robot = Articulation("/World/Robot", reset_xform_op_properties=True)
    app_utils.play()
    _step(10)

    report["stage_counts"] = _stage_counts()
    counts = report["stage_counts"]
    if counts["articulation_roots"] != 1:
        raise AssertionError(f"Expected one articulation root: {counts}")
    if counts["physics_scenes"] != 1:
        raise AssertionError(f"Expected one explicit physics scene: {counts}")
    if counts["rigid_bodies"] != 13:
        raise AssertionError(f"Expected 13 rigid bodies: {counts}")
    if counts["revolute_joints"] != 12:
        raise AssertionError(f"Expected 12 revolute joints: {counts}")
    if counts["collision_prims"] < 13:
        raise AssertionError(f"Every link requires collision geometry: {counts}")
    if args.mode == "fixed" and counts["fixed_joints"] < 1:
        raise AssertionError(f"Fixed asset has no world constraint: {counts}")
    if args.mode == "floating" and counts["fixed_joints"] != 0:
        raise AssertionError(f"Floating asset unexpectedly has a fixed joint: {counts}")

    dof_names = _check_structure(robot, report)
    report["servo_drive"] = _set_drives(robot)
    lower, upper = robot.get_dof_limits()
    lower = _finite("lower limits", lower).reshape(-1)
    upper = _finite("upper limits", upper).reshape(-1)
    stand_by_name = stance_by_name(
        down_m=args.stance_down,
        fore_aft_m=args.stance_fore_aft,
        abduction_deg=args.abduction_deg,
    )
    stand = np.asarray(
        targets_for_order(dof_names, stand_by_name),
        dtype=np.float32,
    )
    _assert_target_inside_limits(stand, lower, upper, "standing")
    report["standing_assumptions"] = {
        "leg_down_m": args.stance_down,
        "front_and_rear_fore_aft_m": args.stance_fore_aft,
        "hip_abduction_deg": args.abduction_deg,
        "contact_note": (
            "The current leg has no printed foot; URDF fork-tip contacts are "
            "an explicit approximation."
        ),
    }

    if args.mode == "fixed":
        _run_fixed(robot, dof_names, stand, lower, upper, report)
    else:
        _run_floating(robot, stand, report)

    if args.screenshot:
        _capture_screenshot(args.screenshot, robot)
        report["screenshot"] = os.path.abspath(args.screenshot)
        report["screenshot_bytes"] = os.path.getsize(args.screenshot)

    report["status"] = "PASS"
    exit_code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(os.path.abspath(args.report), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_ISAAC_VALIDATION_RESULT=" + json.dumps(report, sort_keys=True))
    try:
        app_utils.stop()
        _step(10)
    except Exception:
        pass
    simulation_app.close()

sys.exit(exit_code)
