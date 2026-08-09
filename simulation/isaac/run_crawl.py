"""Run and score a slow four-beat crawl in Isaac Sim 6.0.

This is a dynamics test, not an animation-only preview.  The floating-base
robot settles under explicit Earth gravity, follows name-mapped IK targets,
and is scored on displacement, drift, body tilt, joint tracking, velocity,
and ST3215 effort utilization.  The default ``rated`` torque profile is the
sustainable test; ``stall`` is available only as a short peak-torque
comparison.
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
    COORDINATED_PUSH_SWING_ORDER,
    CRAWL_DUTY_FACTOR,
    DEFAULT_ABDUCTION_DEG,
    DEFAULT_STANCE_DOWN_M,
    DEFAULT_STANCE_FORE_AFT_M,
    DISTRIBUTED_PUSH_SWING_ORDER,
    EXPECTED_DOF_NAMES,
    LEGS,
    LINK_LENGTH_M,
    MAX_NO_LOAD_VELOCITY_RAD_S,
    QUASISTATIC_SWING_ORDER,
    add_robot_reference,
    body_tilt_deg,
    coordinated_push_crawl_by_name,
    crawl_by_name,
    distributed_push_crawl_by_name,
    quasistatic_crawl_by_name,
    smoothstep,
    stance_by_name,
    targets_for_order,
    torque_cap_nm,
)
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(
    description="Run the drobot quadruped slow crawl under Isaac Sim physics."
)
parser.add_argument("--usd", required=True, help="Floating imported USD/USDA")
parser.add_argument("--report", required=True, help="Output JSON metrics")
parser.add_argument("--screenshot", required=True, help="Output review PNG")
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--gait-mode",
    choices=(
        "crawl",
        "quasi-static",
        "distributed-push",
        "coordinated-push",
    ),
    default="crawl",
)
parser.add_argument("--cycles", type=float, default=4.0)
parser.add_argument("--period", type=float, default=2.8)
parser.add_argument("--stride", type=float, default=0.025)
parser.add_argument("--lift", type=float, default=0.012)
parser.add_argument("--weight-shift-forward", type=float, default=0.018)
parser.add_argument("--weight-shift-lateral", type=float, default=0.018)
parser.add_argument("--startup-blend-seconds", type=float, default=1.2)
parser.add_argument("--settle-seconds", type=float, default=4.0)
parser.add_argument("--post-gait-settle-seconds", type=float, default=0.8)
parser.add_argument("--start-z", type=float, default=0.420)
parser.add_argument("--stance-down", type=float, default=DEFAULT_STANCE_DOWN_M)
parser.add_argument(
    "--stance-fore-aft",
    type=float,
    default=DEFAULT_STANCE_FORE_AFT_M,
)
parser.add_argument("--abduction-deg", type=float, default=DEFAULT_ABDUCTION_DEG)
parser.add_argument("--torque-cap", choices=("rated", "stall"), default="rated")
parser.add_argument(
    "--effort-limit-nm",
    type=float,
    default=None,
    help="Positive custom per-joint cap; overrides --torque-cap",
)
parser.add_argument("--drive-stiffness", type=float, default=30.0)
parser.add_argument("--drive-damping", type=float, default=4.58366)
parser.add_argument("--min-base-z", type=float, default=0.200)
parser.add_argument("--max-tilt-deg", type=float, default=25.0)
parser.add_argument("--max-lateral-drift", type=float, default=0.050)
parser.add_argument("--min-forward-displacement", type=float, default=0.020)
parser.add_argument("--max-joint-error-rad", type=float, default=0.15)
parser.add_argument("--contact-on-threshold-n", type=float, default=1.0)
parser.add_argument("--contact-off-threshold-n", type=float, default=0.5)
parser.add_argument("--minimum-support-contact-fraction", type=float, default=0.85)
parser.add_argument("--minimum-swing-unload-seconds", type=float, default=0.10)
parser.add_argument("--minimum-touchdown-seconds", type=float, default=0.15)
parser.add_argument("--minimum-swing-clearance", type=float, default=0.004)
parser.add_argument("--maximum-support-slip", type=float, default=0.015)
parser.add_argument(
    "--review-phase",
    type=float,
    default=0.875,
    help="Normalized gait phase held briefly for the final screenshot",
)
args, _ = parser.parse_known_args()

if args.cycles <= 0.0 or args.period <= 0.5:
    parser.error("Cycles must be positive and period must exceed 0.5 seconds")
if (
    args.gait_mode
    in (
        "quasi-static",
        "distributed-push",
        "coordinated-push",
    )
    and args.period < 8.0
):
    parser.error("Guarded crawl periods must be at least 8 seconds")
if not 0.005 <= args.stride <= 0.075:
    parser.error("Stride must be between 0.005 and 0.075 meters")
if not 0.003 <= args.lift <= 0.025:
    parser.error("Lift must be between 0.003 and 0.025 meters")
if (
    args.startup_blend_seconds <= 0.0
    or args.settle_seconds <= 0.0
    or args.post_gait_settle_seconds <= 0.0
):
    parser.error("Startup blend and settle duration must be positive")
if args.drive_stiffness < 0.0 or args.drive_damping < 0.0:
    parser.error("Drive stiffness and damping must be non-negative")
if not 0.0 <= args.review_phase < 1.0:
    parser.error("Review phase must be in [0, 1)")
if args.weight_shift_forward < 0.0 or args.weight_shift_lateral < 0.0:
    parser.error("Weight shifts must be non-negative")
if not 0.0 < args.contact_off_threshold_n < args.contact_on_threshold_n:
    parser.error("Contact thresholds must satisfy 0 < off < on")
if not 0.0 < args.minimum_support_contact_fraction <= 1.0:
    parser.error("Minimum support contact fraction must be in (0, 1]")
if args.minimum_swing_clearance <= 0.0 or args.maximum_support_slip <= 0.0:
    parser.error("Clearance and slip thresholds must be positive")

torque_profile, effort_cap_nm = torque_cap_nm(
    args.torque_cap,
    args.effort_limit_nm,
)

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": 1280,
        "height": 720,
    }
)

# Omniverse imports must follow SimulationApp construction.
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.objects import GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from omni.kit.viewport.utility import (  # noqa: E402
    capture_viewport_to_file,
    get_active_viewport,
)
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

EARTH_GRAVITY_M_S2 = 9.81
PHYSICS_HZ = 120
CONTROL_HZ = 60
CONTROL_DT_S = 1.0 / CONTROL_HZ

STATIC_FRICTION = 0.90
DYNAMIC_FRICTION = 0.75
RESTITUTION = 0.02
CONTACT_STIFFNESS_N_M = 12000.0
CONTACT_DAMPING_N_S_M = 45.0
FOOT_CONTACT_RADIUS_M = 0.0125


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


def _update(count: int) -> None:
    for _ in range(count):
        simulation_app.update()


def _configure_physics() -> dict:
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
    vector = direction * magnitude
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    steps_per_second = int(
        scene.GetPrim().GetAttribute("physxScene:timeStepsPerSecond").Get()
    )
    if not np.allclose(
        vector,
        [0.0, 0.0, -EARTH_GRAVITY_M_S2],
        atol=1e-6,
    ):
        raise AssertionError(f"Unexpected gravity vector: {vector}")
    if steps_per_second != PHYSICS_HZ:
        raise AssertionError(f"Unexpected physics rate: {steps_per_second}")
    if not math.isclose(meters_per_unit, 1.0, abs_tol=1e-9):
        raise AssertionError(f"Stage is not meter-scaled: {meters_per_unit}")
    return {
        "gravity_vector_m_s2": vector.tolist(),
        "meters_per_unit": meters_per_unit,
        "physics_steps_per_second": steps_per_second,
        "control_steps_per_second": CONTROL_HZ,
    }


def _apply_contact_material() -> dict:
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
        raise AssertionError("No robot collision received contact material")
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


def _set_drives(robot: Articulation) -> dict:
    count = robot.num_dofs
    imported_cap = _finite(
        "imported max efforts",
        robot.get_dof_max_efforts(),
    ).reshape(-1)
    robot.set_dof_max_efforts(np.full(count, effort_cap_nm, dtype=np.float32))
    robot.set_dof_gains(
        np.full(count, args.drive_stiffness, dtype=np.float32),
        np.full(count, args.drive_damping, dtype=np.float32),
    )
    applied_cap = _finite(
        "applied max efforts",
        robot.get_dof_max_efforts(),
    ).reshape(-1)
    if not np.allclose(applied_cap, effort_cap_nm, atol=1e-4):
        raise AssertionError(
            f"Could not apply {effort_cap_nm} N*m torque cap: {applied_cap}"
        )
    return {
        "profile": torque_profile,
        "per_joint_cap_nm": effort_cap_nm,
        "urdf_imported_max_efforts_nm": imported_cap.tolist(),
        "applied_max_efforts_nm": applied_cap.tolist(),
        "drive_stiffness_nm_rad": args.drive_stiffness,
        "drive_damping_nm_s_rad": args.drive_damping,
        "rated_profile_is_sustainable_test": torque_profile == "rated",
        "stall_profile_is_short_peak_only": torque_profile == "stall",
    }


def _assert_target_inside_limits(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    label: str,
) -> None:
    if np.any(target <= lower + 1e-4) or np.any(target >= upper - 1e-4):
        raise AssertionError(
            f"{label} target exceeds a hard joint limit: "
            f"target={target}, lower={lower}, upper={upper}"
        )


def _gait_pose_and_state(gait_time_s: float) -> tuple[dict[str, float], dict]:
    if args.gait_mode == "quasi-static":
        return quasistatic_crawl_by_name(
            gait_time_s,
            period_s=args.period,
            stride_m=args.stride,
            lift_m=args.lift,
            weight_shift_forward_m=args.weight_shift_forward,
            weight_shift_lateral_m=args.weight_shift_lateral,
            down_m=args.stance_down,
            fore_aft_m=args.stance_fore_aft,
            abduction_deg=args.abduction_deg,
        )
    if args.gait_mode == "distributed-push":
        return distributed_push_crawl_by_name(
            gait_time_s,
            period_s=args.period,
            stride_m=args.stride,
            lift_m=args.lift,
            weight_shift_forward_m=args.weight_shift_forward,
            weight_shift_lateral_m=args.weight_shift_lateral,
            down_m=args.stance_down,
            fore_aft_m=args.stance_fore_aft,
            abduction_deg=args.abduction_deg,
        )
    if args.gait_mode == "coordinated-push":
        return coordinated_push_crawl_by_name(
            gait_time_s,
            period_s=args.period,
            stride_m=args.stride,
            lift_m=args.lift,
            weight_shift_forward_m=args.weight_shift_forward,
            weight_shift_lateral_m=args.weight_shift_lateral,
            down_m=args.stance_down,
            fore_aft_m=args.stance_fore_aft,
            abduction_deg=args.abduction_deg,
        )
    pose = crawl_by_name(
        gait_time_s,
        period_s=args.period,
        stride_m=args.stride,
        lift_m=args.lift,
        down_m=args.stance_down,
        fore_aft_m=args.stance_fore_aft,
        abduction_deg=args.abduction_deg,
        duty_factor=CRAWL_DUTY_FACTOR,
    )
    return pose, {
        "cycle_phase": (gait_time_s / args.period) % 1.0,
        "phase": "periodic_crawl",
        "phase_progress": 0.0,
        "swing_leg": None,
        "expected_support_legs": list(LEGS),
    }


def _rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    quaternion_vector = np.asarray([x, y, z], dtype=float)
    return (
        vector
        + 2.0 * w * np.cross(quaternion_vector, vector)
        + 2.0
        * np.cross(
            quaternion_vector,
            np.cross(quaternion_vector, vector),
        )
    )


def _sample_feet(feet: RigidPrim) -> dict[str, np.ndarray]:
    positions, orientations = feet.get_world_poses()
    positions = _finite("distal link positions", positions).reshape(-1, 3).copy()
    orientations = (
        _finite("distal link orientations", orientations).reshape(-1, 4).copy()
    )
    force_matrix = (
        _finite(
            "ground contact force matrix",
            feet.get_contact_force_matrix(dt=1.0 / PHYSICS_HZ),
        )
        .reshape(len(LEGS), -1, 3)
        .copy()
    )
    if force_matrix.shape[1] != 1:
        raise AssertionError(
            f"Expected one ground contact filter, got {force_matrix.shape}"
        )
    normal_loads = np.maximum(force_matrix[:, 0, 2], 0.0)
    tip_centers = np.empty((len(LEGS), 3), dtype=float)
    local_tip = np.asarray([LINK_LENGTH_M, 0.0, 0.0], dtype=float)
    for index in range(len(LEGS)):
        tip_centers[index] = positions[index] + _rotate_wxyz(
            orientations[index],
            local_tip,
        )
    return {
        "normal_load_n": normal_loads,
        "ground_force_world_n": force_matrix[:, 0, :],
        "tip_center_position_m": tip_centers,
        "tip_bottom_height_m": tip_centers[:, 2] - FOOT_CONTACT_RADIUS_M,
    }


def _sample_state(
    robot: Articulation,
    target: np.ndarray,
) -> dict[str, object]:
    base_position, base_orientation = robot.get_world_poses()
    base_position = (
        _finite(
            "base position",
            base_position,
        )
        .reshape(-1, 3)[0]
        .copy()
    )
    base_orientation = (
        _finite(
            "base orientation",
            base_orientation,
        )
        .reshape(-1, 4)[0]
        .copy()
    )
    positions = (
        _finite(
            "joint positions",
            robot.get_dof_positions(),
        )
        .reshape(-1)
        .copy()
    )
    velocities = (
        _finite(
            "joint velocities",
            robot.get_dof_velocities(),
        )
        .reshape(-1)
        .copy()
    )
    requested_pd = (
        args.drive_stiffness * (target - positions) - args.drive_damping * velocities
    )
    sample: dict[str, object] = {
        "base_position_m": base_position,
        "base_orientation_wxyz": base_orientation,
        "body_tilt_deg": body_tilt_deg(base_orientation),
        "joint_positions_rad": positions,
        "joint_velocities_rad_s": velocities,
        "joint_target_rad": target.copy(),
        "joint_error_rad": target - positions,
        "requested_pd_nm": requested_pd,
        "capped_pd_nm": np.clip(
            requested_pd,
            -effort_cap_nm,
            effort_cap_nm,
        ),
    }
    try:
        sample["reported_drive_effort_nm"] = (
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
        sample["projected_joint_load_nm"] = (
            _finite(
                "projected joint load",
                robot.get_dof_projected_joint_forces(),
            )
            .reshape(-1)
            .copy()
        )
    except Exception:
        pass
    return sample


def _summarize_samples(
    samples: list[dict[str, object]],
    dof_names: list[str],
) -> dict:
    if not samples:
        raise AssertionError("No dynamics samples were recorded")
    base_positions = np.vstack([sample["base_position_m"] for sample in samples])
    tilts = np.asarray(
        [sample["body_tilt_deg"] for sample in samples],
        dtype=float,
    )
    targets = np.vstack([sample["joint_target_rad"] for sample in samples])
    positions = np.vstack([sample["joint_positions_rad"] for sample in samples])
    velocities = np.vstack([sample["joint_velocities_rad_s"] for sample in samples])
    errors = targets - positions
    requested = np.vstack([sample["requested_pd_nm"] for sample in samples])

    result: dict[str, object] = {
        "sample_count": len(samples),
        "base_position_samples_m": base_positions.tolist(),
        "minimum_base_z_m": float(np.min(base_positions[:, 2])),
        "maximum_body_tilt_deg": float(np.max(tilts)),
        "maximum_abs_joint_error_rad": float(np.max(np.abs(errors))),
        "rms_joint_error_rad": float(np.sqrt(np.mean(errors * errors))),
        "maximum_abs_joint_velocity_rad_s": float(np.max(np.abs(velocities))),
        "commanded_range_by_joint_rad": {
            name: float(np.ptp(targets[:, index]))
            for index, name in enumerate(dof_names)
        },
        "actual_range_by_joint_rad": {
            name: float(np.ptp(positions[:, index]))
            for index, name in enumerate(dof_names)
        },
        "peak_abs_requested_pd_nm_by_joint": {
            name: float(np.max(np.abs(requested[:, index])))
            for index, name in enumerate(dof_names)
        },
        "peak_requested_to_cap_ratio": float(np.max(np.abs(requested)) / effort_cap_nm),
        "pd_saturation_sample_fraction": float(
            np.mean(np.abs(requested) >= effort_cap_nm)
        ),
    }
    for key in (
        "reported_drive_effort_nm",
        "projected_joint_load_nm",
        "capped_pd_nm",
    ):
        available = [sample[key] for sample in samples if key in sample]
        if not available:
            continue
        values = np.vstack(available)
        peak = np.max(np.abs(values), axis=0)
        result[f"peak_abs_{key}_by_joint"] = {
            name: float(peak[index]) for index, name in enumerate(dof_names)
        }
        result[f"peak_abs_{key}_all_joints"] = float(np.max(peak))
    return result


def _capture_screenshot(
    path: str,
    robot: Articulation,
) -> None:
    position, _ = robot.get_world_poses()
    position = _finite(
        "screenshot base position",
        position,
    ).reshape(-1, 3)[0]
    set_camera_view(
        eye=[
            float(position[0] + 0.76),
            float(position[1] - 0.88),
            float(position[2] + 0.37),
        ],
        target=[
            float(position[0] + 0.02),
            float(position[1]),
            float(position[2] - 0.14),
        ],
        camera_prim_path="/OmniverseKit_Persp",
    )
    _update(45)
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
    "mode": "headless_crawl" if args.headless else "ui_crawl",
    "gait_mode": args.gait_mode,
    "usd": os.path.abspath(args.usd),
    "screenshot": os.path.abspath(args.screenshot),
    "servo_model": "Feetech ST-3215-C018 / Waveshare ST3215 12 V",
    "assessment_scope": (
        "Floating-base dynamics with PLA/battery/electronics inertias from the "
        "URDF. Current distal fork-tip contacts approximate feet."
    ),
}
exit_code = 1

try:
    if not os.path.isfile(args.usd):
        raise FileNotFoundError(args.usd)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.screenshot)), exist_ok=True)

    stage_utils.create_new_stage(template="sunlight")
    report["physics"] = _configure_physics()
    ground = GroundPlane(
        "/World/GroundPlane",
        positions=[0.0, 0.0, 0.0],
    )
    report["asset_reference"] = add_robot_reference(
        stage_utils,
        args.usd,
        "/World/Robot",
    )
    report["contact_material"] = _apply_contact_material()

    robot = Articulation("/World/Robot", reset_xform_op_properties=True)
    link_path_by_name = dict(
        zip(
            robot.link_names,
            robot.link_paths[0],
            strict=True,
        )
    )
    distal_link_paths = [link_path_by_name[f"{leg}_distal_link"] for leg in LEGS]
    feet = RigidPrim(
        distal_link_paths,
        contact_filter_paths=ground.planes.paths[0],
        max_contact_count=64,
    )
    feet.set_enabled_contact_tracking([True], threshold=0.0)
    app_utils.play()
    _update(10)
    dof_names = list(robot.dof_names)
    if len(dof_names) != 12 or set(dof_names) != EXPECTED_DOF_NAMES:
        raise AssertionError(f"Unexpected DOF names: {dof_names}")
    if robot.num_links != 13 or "base_link" not in robot.link_names:
        raise AssertionError(f"Unexpected links: {list(robot.link_names)}")

    lower, upper = robot.get_dof_limits()
    lower = _finite("lower limits", lower).reshape(-1)
    upper = _finite("upper limits", upper).reshape(-1)
    max_velocities = _finite(
        "max velocities",
        robot.get_dof_max_velocities(),
    ).reshape(-1)
    if np.any(max_velocities > MAX_NO_LOAD_VELOCITY_RAD_S + 1e-3):
        raise AssertionError(f"URDF exceeds verified ST3215 speed: {max_velocities}")
    link_masses = _finite("link masses", robot.get_link_masses()).reshape(-1)
    if len(link_masses) != 13 or np.any(link_masses <= 0.0):
        raise AssertionError(f"Invalid physical link masses: {link_masses}")

    report["dof_names"] = dof_names
    report["link_names"] = list(robot.link_names)
    report["contact_tracking"] = {
        "leg_order": list(LEGS),
        "distal_link_paths": distal_link_paths,
        "ground_filter_path": ground.planes.paths[0],
        "foot_proxy_radius_m": FOOT_CONTACT_RADIUS_M,
        "contact_on_threshold_n": args.contact_on_threshold_n,
        "contact_off_threshold_n": args.contact_off_threshold_n,
        "physics_tensor_valid": bool(feet.is_physics_tensor_entity_valid()),
        "scope": (
            "Ground-filtered force on each distal link plus the transformed "
            "fork-tip sphere center; no physical printed foot exists."
        ),
    }
    if not report["contact_tracking"]["physics_tensor_valid"]:
        raise AssertionError("Distal-link contact tensor view is invalid")
    report["joint_lower_limits_rad"] = lower.tolist()
    report["joint_upper_limits_rad"] = upper.tolist()
    report["joint_max_velocities_rad_s"] = max_velocities.tolist()
    report["link_masses_kg"] = link_masses.tolist()
    report["total_articulated_mass_kg"] = float(np.sum(link_masses))
    report["servo_drive"] = _set_drives(robot)

    initial_pose = (
        _gait_pose_and_state(0.0)[0]
        if args.gait_mode in ("distributed-push", "coordinated-push")
        else stance_by_name(
            down_m=args.stance_down,
            fore_aft_m=args.stance_fore_aft,
            abduction_deg=args.abduction_deg,
        )
    )
    stand = np.asarray(
        targets_for_order(
            dof_names,
            initial_pose,
        ),
        dtype=np.float32,
    )
    _assert_target_inside_limits(stand, lower, upper, "standing")

    robot.set_world_poses(
        positions=[[0.0, 0.0, args.start_z]],
        orientations=[[1.0, 0.0, 0.0, 0.0]],
    )
    robot.set_velocities(
        linear_velocities=[[0.0, 0.0, 0.0]],
        angular_velocities=[[0.0, 0.0, 0.0]],
    )
    robot.set_dof_positions(stand)
    robot.set_dof_velocities(np.zeros(12, dtype=np.float32))
    robot.set_dof_position_targets(stand)

    settle_steps = int(math.ceil(args.settle_seconds * CONTROL_HZ))
    for _ in range(settle_steps):
        robot.set_dof_position_targets(stand)
        _update(1)
    settled_sample = _sample_state(robot, stand)
    settled_position = np.asarray(
        settled_sample["base_position_m"],
        dtype=float,
    )
    settled_tilt = float(settled_sample["body_tilt_deg"])
    if settled_position[2] < args.min_base_z:
        raise AssertionError(
            f"Robot collapsed before gait: base z={settled_position[2]:.6f} m"
        )
    if settled_tilt >= args.max_tilt_deg:
        raise AssertionError(
            f"Robot tipped before gait: tilt={settled_tilt:.3f} degrees"
        )
    settled_feet = _sample_feet(feet)
    if not np.isfinite(settled_feet["normal_load_n"]).all():
        raise AssertionError("Invalid settled foot-contact loads")

    gait_steps = int(math.ceil(args.cycles * args.period * CONTROL_HZ))
    samples: list[dict[str, object]] = []
    contact_samples: list[dict[str, object]] = []
    contact_state = settled_feet["normal_load_n"] >= args.contact_on_threshold_n
    support_anchors: list[np.ndarray | None] = [
        settled_feet["tip_center_position_m"][index, :2].copy()
        if contact_state[index]
        else None
        for index in range(len(LEGS))
    ]
    maximum_support_slip_m = 0.0
    maximum_support_slip_by_leg = {leg: 0.0 for leg in LEGS}
    target_min = np.full(12, np.inf)
    target_max = np.full(12, -np.inf)
    for step_index in range(gait_steps):
        gait_time_s = (step_index + 1) * CONTROL_DT_S
        gait_pose, gait_state = _gait_pose_and_state(gait_time_s)
        gait = np.asarray(
            targets_for_order(
                dof_names,
                gait_pose,
            ),
            dtype=np.float32,
        )
        if args.gait_mode in (
            "quasi-static",
            "distributed-push",
            "coordinated-push",
        ):
            target = gait.copy()
        else:
            blend = smoothstep(gait_time_s / args.startup_blend_seconds)
            target = ((1.0 - blend) * stand + blend * gait).astype(np.float32)
        _assert_target_inside_limits(target, lower, upper, "crawl")
        target_min = np.minimum(target_min, target)
        target_max = np.maximum(target_max, target)
        robot.set_dof_position_targets(target)
        _update(1)
        foot_sample = _sample_feet(feet)
        normal_loads = foot_sample["normal_load_n"]
        contact_state = np.where(
            normal_loads >= args.contact_on_threshold_n,
            True,
            np.where(
                normal_loads <= args.contact_off_threshold_n,
                False,
                contact_state,
            ),
        )
        expected_support = set(gait_state["expected_support_legs"])
        per_leg_slip: dict[str, float] = {}
        for leg_index, leg in enumerate(LEGS):
            if leg not in expected_support:
                support_anchors[leg_index] = None
                continue
            if not contact_state[leg_index]:
                continue
            foot_xy = foot_sample["tip_center_position_m"][leg_index, :2]
            if support_anchors[leg_index] is None:
                support_anchors[leg_index] = foot_xy.copy()
            slip = float(np.linalg.norm(foot_xy - support_anchors[leg_index]))
            per_leg_slip[leg] = slip
            maximum_support_slip_m = max(maximum_support_slip_m, slip)
            maximum_support_slip_by_leg[leg] = max(
                maximum_support_slip_by_leg[leg],
                slip,
            )
        contact_record: dict[str, object] = {
            "time_s": gait_time_s,
            "phase": gait_state["phase"],
            "phase_progress": gait_state["phase_progress"],
            "swing_leg": gait_state["swing_leg"],
            "expected_support_legs": list(gait_state["expected_support_legs"]),
            "normal_load_n_by_leg": {
                leg: float(normal_loads[index]) for index, leg in enumerate(LEGS)
            },
            "contact_by_leg": {
                leg: bool(contact_state[index]) for index, leg in enumerate(LEGS)
            },
            "tip_bottom_height_m_by_leg": {
                leg: float(foot_sample["tip_bottom_height_m"][index])
                for index, leg in enumerate(LEGS)
            },
            "support_slip_m_by_leg": per_leg_slip,
        }
        contact_samples.append(contact_record)
        if step_index % 6 == 0:
            samples.append(_sample_state(robot, target))

    final_target = target.copy()
    post_settle_positions = []
    post_settle_steps = int(math.ceil(args.post_gait_settle_seconds * CONTROL_HZ))
    for post_step in range(post_settle_steps):
        robot.set_dof_position_targets(final_target)
        _update(1)
        post_sample = _sample_state(robot, final_target)
        if post_step % 6 == 0:
            samples.append(post_sample)
        if post_step >= post_settle_steps // 2:
            post_settle_positions.append(post_sample["base_position_m"])
    final_sample = _sample_state(robot, final_target)
    samples.append(final_sample)
    final_position = np.mean(
        np.vstack(post_settle_positions),
        axis=0,
    )
    displacement = final_position - settled_position
    forward_displacement = float(displacement[0])
    lateral_drift = float(displacement[1])
    gait_metrics = _summarize_samples(samples, dof_names)
    gait_duration_s = gait_steps / CONTROL_HZ

    expected_contact_slots = 0
    loaded_expected_contact_slots = 0
    for record in contact_samples:
        for leg in record["expected_support_legs"]:
            expected_contact_slots += 1
            loaded_expected_contact_slots += int(record["contact_by_leg"][leg])
    expected_support_contact_fraction = (
        loaded_expected_contact_slots / expected_contact_slots
    )

    def _longest_contact_run(
        records: list[dict[str, object]],
        leg: str,
        expected_value: bool,
    ) -> int:
        longest = 0
        current = 0
        previous_time_s: float | None = None
        for record in records:
            time_s = float(record["time_s"])
            consecutive = (
                previous_time_s is not None
                and time_s - previous_time_s <= 1.5 * CONTROL_DT_S
            )
            if bool(record["contact_by_leg"][leg]) is expected_value:
                current = current + 1 if consecutive else 1
                longest = max(longest, current)
            else:
                current = 0
            previous_time_s = time_s
        return longest

    foot_step_evidence: dict[str, dict[str, object]] = {}
    completed_foot_steps: list[str] = []
    contact_verified_mode = args.gait_mode in (
        "quasi-static",
        "distributed-push",
        "coordinated-push",
    )
    contact_verified_order = (
        COORDINATED_PUSH_SWING_ORDER
        if args.gait_mode == "coordinated-push"
        else (
            DISTRIBUTED_PUSH_SWING_ORDER
            if args.gait_mode == "distributed-push"
            else QUASISTATIC_SWING_ORDER
        )
    )
    swing_phase_name = "swing_push" if args.gait_mode == "coordinated-push" else "swing"
    if contact_verified_mode:
        for leg in contact_verified_order:
            swing_records = [
                record
                for record in contact_samples
                if record["swing_leg"] == leg and record["phase"] == swing_phase_name
            ]
            three_support_records = [
                record
                for record in contact_samples
                if record["swing_leg"] == leg
                and record["phase"] in ("lift", swing_phase_name, "lower")
            ]
            airborne_records = [
                record for record in swing_records if not record["contact_by_leg"][leg]
            ]
            three_support_fraction = (
                sum(
                    all(
                        record["contact_by_leg"][support_leg]
                        for support_leg in LEGS
                        if support_leg != leg
                    )
                    for record in three_support_records
                )
                / len(three_support_records)
                if three_support_records
                else 0.0
            )
            touchdown_records = [
                record
                for record in contact_samples
                if record["swing_leg"] == leg
                and record["phase"]
                in ("touchdown", "weight_return", "all_feet_push", "step_settle")
            ]
            longest_unload_s = (
                _longest_contact_run(swing_records, leg, False) / CONTROL_HZ
            )
            longest_touchdown_s = (
                _longest_contact_run(touchdown_records, leg, True) / CONTROL_HZ
            )
            maximum_clearance_m = max(
                (record["tip_bottom_height_m_by_leg"][leg] for record in swing_records),
                default=float("-inf"),
            )
            minimum_swing_load_n = min(
                (record["normal_load_n_by_leg"][leg] for record in swing_records),
                default=float("inf"),
            )
            completed = (
                longest_unload_s >= args.minimum_swing_unload_seconds
                and longest_touchdown_s >= args.minimum_touchdown_seconds
                and three_support_fraction >= args.minimum_support_contact_fraction
                and maximum_clearance_m >= args.minimum_swing_clearance
            )
            if completed:
                completed_foot_steps.append(leg)
            foot_step_evidence[leg] = {
                "swing_sample_count": len(swing_records),
                "airborne_sample_count": len(airborne_records),
                "longest_continuous_unload_s": longest_unload_s,
                "longest_continuous_touchdown_s": longest_touchdown_s,
                "three_other_feet_contact_fraction": three_support_fraction,
                "maximum_tip_bottom_clearance_m": maximum_clearance_m,
                "minimum_swing_foot_load_n": minimum_swing_load_n,
                "completed": completed,
            }
    contact_metrics = {
        "sample_count": len(contact_samples),
        "settled_normal_load_n_by_leg": {
            leg: float(settled_feet["normal_load_n"][index])
            for index, leg in enumerate(LEGS)
        },
        "settled_total_normal_load_n": float(np.sum(settled_feet["normal_load_n"])),
        "expected_support_contact_fraction": expected_support_contact_fraction,
        "maximum_support_tip_slip_m": maximum_support_slip_m,
        "maximum_support_tip_slip_m_by_leg": maximum_support_slip_by_leg,
        "completed_foot_steps": completed_foot_steps,
        "foot_step_evidence": foot_step_evidence,
        "evidence_scope": (
            "Force is ground-filtered per distal link. Slip is transformed "
            "fork-tip center displacement while that leg is loaded; it is "
            "not an optical-flow estimate."
        ),
    }

    # Hold a clearly lifted-leg phase for the review image after all scored
    # motion metrics are frozen.
    review_pose, review_state = _gait_pose_and_state(args.review_phase * args.period)
    review_target = np.asarray(
        targets_for_order(dof_names, review_pose),
        dtype=np.float32,
    )
    _assert_target_inside_limits(review_target, lower, upper, "review")
    for _ in range(20):
        robot.set_dof_position_targets(review_target)
        _update(1)
    _capture_screenshot(args.screenshot, robot)

    actual_ranges = gait_metrics["actual_range_by_joint_rad"]
    insufficient_flexion = [
        name
        for name in dof_names
        if not name.endswith("_hip_abduction") and actual_ranges[name] < 0.008
    ]
    maximum_tilt = float(gait_metrics["maximum_body_tilt_deg"])
    minimum_z = float(gait_metrics["minimum_base_z_m"])
    maximum_error = float(gait_metrics["maximum_abs_joint_error_rad"])
    maximum_velocity = float(gait_metrics["maximum_abs_joint_velocity_rad_s"])

    report.update(
        {
            "gait": {
                "name": (
                    "contact_verified_quasi_static_crawl"
                    if args.gait_mode == "quasi-static"
                    else (
                        "coordinated_three_support_push_crawl"
                        if args.gait_mode == "coordinated-push"
                        else (
                            "distributed_four_foot_push_crawl"
                            if args.gait_mode == "distributed-push"
                            else "slow_four_beat_crawl"
                        )
                    )
                ),
                "mode": args.gait_mode,
                "cycles_requested": args.cycles,
                "cycles_executed": gait_duration_s / args.period,
                "period_s": args.period,
                "stride_m": args.stride,
                "lift_m": args.lift,
                "duty_factor": (
                    CRAWL_DUTY_FACTOR if args.gait_mode == "crawl" else None
                ),
                "swing_order": list(
                    COORDINATED_PUSH_SWING_ORDER
                    if args.gait_mode == "coordinated-push"
                    else (
                        DISTRIBUTED_PUSH_SWING_ORDER
                        if args.gait_mode == "distributed-push"
                        else QUASISTATIC_SWING_ORDER
                    )
                ),
                "stance_down_m": args.stance_down,
                "stance_fore_aft_m": args.stance_fore_aft,
                "hip_abduction_deg": args.abduction_deg,
                "weight_shift_forward_m": (
                    args.weight_shift_forward
                    if args.gait_mode
                    in (
                        "quasi-static",
                        "distributed-push",
                        "coordinated-push",
                    )
                    else 0.0
                ),
                "weight_shift_lateral_m": (
                    args.weight_shift_lateral
                    if args.gait_mode
                    in (
                        "quasi-static",
                        "distributed-push",
                        "coordinated-push",
                    )
                    else 0.0
                ),
                "startup_blend_seconds": (
                    0.0
                    if args.gait_mode
                    in (
                        "quasi-static",
                        "distributed-push",
                        "coordinated-push",
                    )
                    else args.startup_blend_seconds
                ),
                "ik": (
                    "analytic common-direction two-link sagittal IK"
                    if args.gait_mode == "coordinated-push"
                    else (
                        "analytic mirrored outward-bend two-link sagittal IK"
                        if args.gait_mode == "distributed-push"
                        else "analytic mirrored two-link sagittal IK"
                    )
                ),
            },
            "settle_seconds": settle_steps / CONTROL_HZ,
            "post_gait_settle_seconds": post_settle_steps / CONTROL_HZ,
            "settled_base_position_m": settled_position.tolist(),
            "settled_body_tilt_deg": settled_tilt,
            "final_base_position_m": final_position.tolist(),
            "displacement_m": displacement.tolist(),
            "forward_displacement_m": forward_displacement,
            "mean_forward_speed_m_s": forward_displacement / gait_duration_s,
            "lateral_drift_m": lateral_drift,
            "commanded_min_rad": target_min.tolist(),
            "commanded_max_rad": target_max.tolist(),
            "dynamics_metrics": gait_metrics,
            "contact_metrics": contact_metrics,
            "review_phase": args.review_phase,
            "review_state": review_state,
            "screenshot_bytes": os.path.getsize(args.screenshot),
            "pass_thresholds": {
                "minimum_base_z_m": args.min_base_z,
                "maximum_body_tilt_deg": args.max_tilt_deg,
                "maximum_abs_lateral_drift_m": args.max_lateral_drift,
                "minimum_forward_displacement_m": args.min_forward_displacement,
                "maximum_abs_joint_error_rad": args.max_joint_error_rad,
                "maximum_joint_speed_rad_s": MAX_NO_LOAD_VELOCITY_RAD_S,
                "minimum_flexion_joint_motion_rad": 0.008,
                "minimum_support_contact_fraction": (
                    args.minimum_support_contact_fraction
                ),
                "minimum_swing_unload_seconds": (args.minimum_swing_unload_seconds),
                "minimum_touchdown_seconds": args.minimum_touchdown_seconds,
                "minimum_swing_clearance_m": args.minimum_swing_clearance,
                "maximum_support_tip_slip_m": args.maximum_support_slip,
            },
        }
    )

    acceptance_failures = []
    if minimum_z < args.min_base_z:
        acceptance_failures.append(
            f"collapsed during crawl: minimum z={minimum_z:.6f} m"
        )
    if maximum_tilt >= args.max_tilt_deg:
        acceptance_failures.append(
            f"unstable body: max tilt={maximum_tilt:.3f} degrees"
        )
    if abs(lateral_drift) >= args.max_lateral_drift:
        acceptance_failures.append(f"lateral drift={lateral_drift:.6f} m")
    if forward_displacement < args.min_forward_displacement:
        acceptance_failures.append(f"forward motion={forward_displacement:.6f} m")
    if maximum_error >= args.max_joint_error_rad:
        acceptance_failures.append(f"joint tracking error={maximum_error:.6f} rad")
    if maximum_velocity > MAX_NO_LOAD_VELOCITY_RAD_S + 0.05:
        acceptance_failures.append(f"joint speed={maximum_velocity:.6f} rad/s")
    if insufficient_flexion:
        acceptance_failures.append(f"unresponsive leg joints={insufficient_flexion}")
    if contact_verified_mode:
        if expected_support_contact_fraction < args.minimum_support_contact_fraction:
            acceptance_failures.append(
                f"support contact fraction={expected_support_contact_fraction:.6f}"
            )
        if maximum_support_slip_m >= args.maximum_support_slip:
            acceptance_failures.append(
                f"support foot slip={maximum_support_slip_m:.6f} m"
            )
        incomplete_steps = [
            leg for leg in contact_verified_order if leg not in completed_foot_steps
        ]
        if incomplete_steps:
            acceptance_failures.append(
                f"incomplete contact-verified steps={incomplete_steps}"
            )
    report["acceptance_failures"] = acceptance_failures
    if acceptance_failures:
        report["functional_result"] = (
            "FAIL under the selected simulation assumptions: "
            + "; ".join(acceptance_failures)
        )
        raise AssertionError("; ".join(acceptance_failures))

    report["functional_result"] = (
        "PASS under the selected simulation assumptions; physical validation "
        "and printed feet are still required."
    )
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
    print("DROBOT_ISAAC_CRAWL_RESULT=" + json.dumps(report, sort_keys=True))
    try:
        app_utils.stop()
        _update(10)
    except Exception:
        pass
    simulation_app.close()

sys.exit(exit_code)
