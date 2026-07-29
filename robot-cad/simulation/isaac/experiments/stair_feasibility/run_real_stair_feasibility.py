"""Run scripted single-foot placement trials against household-size risers.

This is a floating-base dynamics feasibility experiment, not RL training. It
uses full URDF joint limits, the sustainable ST3215 rated-torque cap, explicit
ground/step contact, and a deterministic front-left Cartesian IK trajectory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _contract import (  # noqa: E402
    current_policy_front_lift_m,
    signed_support_margin_m,
    smoothstep,
    step_targets,
    shifted_stance_pose,
    target_limit_failures,
    trial_gate_failures,
    validate_config,
)
from _quadruped_runtime import (  # noqa: E402
    EXPECTED_DOF_NAMES,
    LEGS,
    LINK_LENGTH_M,
    MAX_NO_LOAD_VELOCITY_RAD_S,
    RATED_TORQUE_NM,
    add_robot_reference,
    body_tilt_deg,
    stance_by_name,
    targets_for_order,
)

parser = argparse.ArgumentParser(
    description="Run Drobot scripted real-stair kinematic/dynamics trials."
)
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "real_stair_feasibility.yaml"),
)
parser.add_argument(
    "--output-dir",
    default=None,
    help="Default: experiment.output_dir from the YAML configuration.",
)
parser.add_argument(
    "--heights-mm",
    nargs="*",
    type=float,
    default=None,
    help="Optional subset/override of riser heights in millimeters.",
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--no-screenshots", action="store_true")
args, _ = parser.parse_known_args()


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


config_path = _resolve_project_path(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
experiment = validate_config(config)
if args.heights_mm:
    heights_m = tuple(float(value) / 1000.0 for value in args.heights_mm)
else:
    heights_m = tuple(float(value) for value in experiment["riser_heights_m"])
if not heights_m or any(value <= 0.0 for value in heights_m):
    parser.error("At least one positive riser height is required")
output_dir = _resolve_project_path(args.output_dir or experiment["output_dir"])
robot_usd = _resolve_project_path(experiment["robot_usd"])

from isaacsim import SimulationApp  # noqa: E402

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _finite(name: str, value) -> np.ndarray:
    result = _numpy(value)
    if not np.isfinite(result).all():
        raise AssertionError(f"{name} contains non-finite values: {result}")
    return result


def _update(count: int) -> None:
    for _ in range(count):
        simulation_app.update()


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


def _configure_physics() -> dict[str, object]:
    physics = dict(experiment["physics"])
    stage = stage_utils.get_current_stage()
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    scene.GetPrim().CreateAttribute(
        "physxScene:timeStepsPerSecond",
        Sdf.ValueTypeNames.Int,
    ).Set(int(physics["physics_hz"]))
    return {
        "gravity_m_s2": 9.81,
        "physics_hz": int(physics["physics_hz"]),
        "control_hz": int(physics["control_hz"]),
    }


def _create_step(height_m: float) -> str:
    path = "/World/FeasibilityStep"
    cube = UsdGeom.Cube.Define(stage_utils.get_current_stage(), path)
    cube.CreateSizeAttr().Set(1.0)
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.78, 0.42, 0.18)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    tread = float(experiment["tread_depth_m"])
    xform.AddTranslateOp().Set(
        Gf.Vec3d(
            float(experiment["step_start_x_m"]) + tread / 2.0,
            0.0,
            height_m / 2.0,
        )
    )
    xform.AddScaleOp().Set(
        Gf.Vec3f(
            tread,
            float(experiment["step_width_m"]),
            height_m,
        )
    )
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.GetPrim().SetCustomDataByKey("drobot:riserHeightM", height_m)
    cube.GetPrim().SetCustomDataByKey(
        "drobot:treadDepthM",
        float(experiment["tread_depth_m"]),
    )
    return path


def _apply_contact_material(step_path: str) -> dict[str, object]:
    physics = dict(experiment["physics"])
    stage = stage_utils.get_current_stage()
    material = UsdShade.Material.Define(
        stage,
        "/World/Materials/FeasibilityContact",
    )
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(
        float(physics["static_friction"])
    )
    material_api.CreateDynamicFrictionAttr().Set(
        float(physics["dynamic_friction"])
    )
    material_api.CreateRestitutionAttr().Set(float(physics["restitution"]))
    physx_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_api.CreateCompliantContactStiffnessAttr().Set(
        float(physics["contact_stiffness_n_m"])
    )
    physx_api.CreateCompliantContactDampingAttr().Set(
        float(physics["contact_damping_n_s_m"])
    )
    bound_paths: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if (
            path.startswith("/World/Robot")
            or path.startswith("/World/GroundPlane")
            or path == step_path
        ):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )
            bound_paths.append(path)
    return {
        "path": str(material.GetPath()),
        "bound_collision_count": len(bound_paths),
        "static_friction": float(physics["static_friction"]),
        "dynamic_friction": float(physics["dynamic_friction"]),
        "scope": "Provisional printed-fork/household-surface contact model.",
    }


def _sample_feet(feet: RigidPrim) -> dict[str, np.ndarray]:
    positions, orientations = feet.get_world_poses()
    positions = _finite("distal positions", positions).reshape(-1, 3)
    orientations = _finite("distal orientations", orientations).reshape(-1, 4)
    forces = _finite(
        "foot contact forces",
        feet.get_contact_force_matrix(
            dt=1.0 / float(experiment["physics"]["physics_hz"])
        ),
    ).reshape(len(LEGS), -1, 3)
    if forces.shape[1] != 2:
        raise AssertionError(
            f"Expected ground and step contact filters, got {forces.shape}"
        )
    tip_centers = np.empty((len(LEGS), 3), dtype=float)
    local_tip = np.asarray([LINK_LENGTH_M, 0.0, 0.0], dtype=float)
    for index in range(len(LEGS)):
        tip_centers[index] = positions[index] + _rotate_wxyz(
            orientations[index],
            local_tip,
        )
    return {
        "tip_center_position_m": tip_centers,
        "tip_bottom_height_m": (
            tip_centers[:, 2]
            - float(experiment["virtual_foot_radius_m"])
        ),
        "ground_force_world_n": forces[:, 0, :],
        "step_force_world_n": forces[:, 1, :],
        "ground_normal_load_n": np.maximum(forces[:, 0, 2], 0.0),
        "step_normal_load_n": np.maximum(forces[:, 1, 2], 0.0),
        "step_force_norm_n": np.linalg.norm(forces[:, 1, :], axis=1),
    }


def _sample_nonfoot_step_force(nonfeet: RigidPrim) -> float:
    values = _finite(
        "non-foot step contact forces",
        nonfeet.get_contact_force_matrix(
            dt=1.0 / float(experiment["physics"]["physics_hz"])
        ),
    ).reshape(-1, 1, 3)
    return float(np.max(np.linalg.norm(values[:, 0, :], axis=1)))


def _sample_state(
    robot: Articulation,
    target: np.ndarray,
) -> dict[str, object]:
    base_position, base_orientation = robot.get_world_poses()
    base_position = _finite("base position", base_position).reshape(-1, 3)[0]
    base_orientation = _finite(
        "base orientation",
        base_orientation,
    ).reshape(-1, 4)[0]
    positions = _finite(
        "joint positions",
        robot.get_dof_positions(),
    ).reshape(-1)
    velocities = _finite(
        "joint velocities",
        robot.get_dof_velocities(),
    ).reshape(-1)
    physics = dict(experiment["physics"])
    requested_pd = (
        float(physics["drive_stiffness_nm_rad"]) * (target - positions)
        - float(physics["drive_damping_nm_s_rad"]) * velocities
    )
    sample: dict[str, object] = {
        "base_position_m": base_position.copy(),
        "base_orientation_wxyz": base_orientation.copy(),
        "body_tilt_deg": body_tilt_deg(base_orientation),
        "joint_positions_rad": positions.copy(),
        "joint_velocities_rad_s": velocities.copy(),
        "joint_target_rad": target.copy(),
        "joint_error_rad": (target - positions).copy(),
        "requested_pd_nm": requested_pd.copy(),
    }
    try:
        sample["reported_drive_effort_nm"] = _finite(
            "reported drive effort",
            robot.get_dof_efforts(),
        ).reshape(-1)
    except Exception:
        pass
    try:
        sample["projected_joint_load_nm"] = _finite(
            "projected joint load",
            robot.get_dof_projected_joint_forces(),
        ).reshape(-1)
    except Exception:
        pass
    return sample


def _capture_screenshot(path: Path, height_m: float) -> None:
    set_camera_view(
        eye=[0.68, -0.78, 0.53],
        target=[0.20, 0.0, max(0.10, height_m * 0.65)],
        camera_prim_path="/OmniverseKit_Persp",
    )
    _update(45)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Isaac Sim has no active viewport")
    path.parent.mkdir(parents=True, exist_ok=True)
    task = asyncio.ensure_future(
        capture_viewport_to_file(
            viewport,
            file_path=str(path),
            is_hdr=False,
        ).wait_for_result()
    )
    for _ in range(300):
        simulation_app.update()
        if task.done():
            break
    if not task.done():
        raise TimeoutError("Timed out waiting for screenshot capture")
    task.result()
    for _ in range(120):
        if path.is_file() and path.stat().st_size > 0:
            break
        simulation_app.update()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Screenshot was not created: {path}")


def _longest_true_run(records: list[bool], control_hz: int) -> float:
    current = 0
    longest = 0
    for value in records:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest / control_hz


def _run_trial(height_m: float) -> dict[str, object]:
    trial: dict[str, object] = {
        "status": "ERROR",
        "riser_height_m": height_m,
        "riser_height_mm": int(round(height_m * 1000.0)),
        "tread_depth_m": float(experiment["tread_depth_m"]),
    }
    app_utils.stop()
    stage_utils.create_new_stage(template="sunlight")
    trial["physics"] = _configure_physics()
    ground = GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, 0.0])
    step_path = _create_step(height_m)
    trial["asset_reference"] = add_robot_reference(
        stage_utils,
        str(robot_usd),
        "/World/Robot",
    )
    trial["contact_material"] = _apply_contact_material(step_path)

    robot = Articulation("/World/Robot", reset_xform_op_properties=True)
    link_path_by_name = dict(
        zip(robot.link_names, robot.link_paths[0], strict=True)
    )
    distal_link_paths = [
        link_path_by_name[f"{leg}_distal_link"] for leg in LEGS
    ]
    nonfoot_link_paths = [
        path
        for name, path in link_path_by_name.items()
        if not name.endswith("_distal_link")
    ]
    feet = RigidPrim(
        distal_link_paths,
        contact_filter_paths=[
            ground.planes.paths[0],
            step_path,
        ],
        max_contact_count=128,
    )
    nonfeet = RigidPrim(
        nonfoot_link_paths,
        contact_filter_paths=step_path,
        max_contact_count=128,
    )
    feet.set_enabled_contact_tracking([True], threshold=0.0)
    nonfeet.set_enabled_contact_tracking([True], threshold=0.0)
    app_utils.play()
    _update(10)

    dof_names = list(robot.dof_names)
    if len(dof_names) != 12 or set(dof_names) != EXPECTED_DOF_NAMES:
        raise AssertionError(f"Unexpected DOF names: {dof_names}")
    if not feet.is_physics_tensor_entity_valid():
        raise AssertionError("Foot contact tensor view is invalid")
    if not nonfeet.is_physics_tensor_entity_valid():
        raise AssertionError("Non-foot contact tensor view is invalid")
    lower, upper = robot.get_dof_limits()
    lower = _finite("joint lower limits", lower).reshape(-1)
    upper = _finite("joint upper limits", upper).reshape(-1)
    max_velocities = _finite(
        "joint max velocities",
        robot.get_dof_max_velocities(),
    ).reshape(-1)
    if np.any(max_velocities > MAX_NO_LOAD_VELOCITY_RAD_S + 1e-3):
        raise AssertionError("URDF joint velocity exceeds verified ST3215 speed")
    physics = dict(experiment["physics"])
    effort_cap = float(physics["effort_cap_nm"])
    if not math.isclose(effort_cap, RATED_TORQUE_NM, abs_tol=1e-9):
        raise AssertionError("Feasibility test must use the rated torque cap")
    robot.set_dof_max_efforts(
        np.full(robot.num_dofs, effort_cap, dtype=np.float32)
    )
    robot.set_dof_gains(
        np.full(
            robot.num_dofs,
            float(physics["drive_stiffness_nm_rad"]),
            dtype=np.float32,
        ),
        np.full(
            robot.num_dofs,
            float(physics["drive_damping_nm_s_rad"]),
            dtype=np.float32,
        ),
    )

    stance = dict(experiment["stance"])
    stand_pose = stance_by_name(
        down_m=float(stance["down_m"]),
        fore_aft_m=float(stance["fore_aft_m"]),
        abduction_deg=float(stance["abduction_deg"]),
    )
    stand = np.asarray(
        targets_for_order(dof_names, stand_pose),
        dtype=np.float32,
    )
    margin = float(experiment["acceptance"]["joint_limit_margin_rad"])
    if np.any(stand <= lower + margin) or np.any(stand >= upper - margin):
        raise AssertionError("Standing target violates hard joint margin")
    robot.set_world_poses(
        positions=[
            [
                float(experiment["reset_base_x_m"]),
                0.0,
                float(experiment["reset_base_z_m"]),
            ]
        ],
        orientations=[[1.0, 0.0, 0.0, 0.0]],
    )
    robot.set_velocities(
        linear_velocities=[[0.0, 0.0, 0.0]],
        angular_velocities=[[0.0, 0.0, 0.0]],
    )
    robot.set_dof_positions(stand)
    robot.set_dof_velocities(np.zeros(12, dtype=np.float32))
    robot.set_dof_position_targets(stand)

    control_hz = int(physics["control_hz"])
    timing = dict(experiment["timing"])
    for _ in range(int(round(float(timing["settle_s"]) * control_hz))):
        robot.set_dof_position_targets(stand)
        _update(1)
    settled_state = _sample_state(robot, stand)
    settled_feet = _sample_feet(feet)
    settled_base = np.asarray(settled_state["base_position_m"], dtype=float)
    swing_leg = str(experiment["swing_leg"])
    swing_index = LEGS.index(swing_leg)
    support_indices = [
        index for index, leg in enumerate(LEGS) if leg != swing_leg
    ]
    support_anchors = settled_feet["tip_center_position_m"][
        support_indices, :2
    ].copy()

    records: list[dict[str, object]] = []
    elapsed_s = 0.0

    def command_and_record(
        target: np.ndarray,
        phase: str,
        *,
        phase_progress: float,
    ) -> None:
        nonlocal elapsed_s
        if np.any(target <= lower + margin) or np.any(target >= upper - margin):
            raise AssertionError(
                f"{phase} target violates hard joint margin: {target}"
            )
        robot.set_dof_position_targets(target)
        _update(1)
        elapsed_s += 1.0 / control_hz
        feet_sample = _sample_feet(feet)
        state_sample = _sample_state(robot, target)
        support_points = feet_sample["tip_center_position_m"][
            support_indices, :2
        ]
        record = {
            "time_s": elapsed_s,
            "phase": phase,
            "phase_progress": float(phase_progress),
            "feet": feet_sample,
            "state": state_sample,
            "nonfoot_step_force_n": _sample_nonfoot_step_force(nonfeet),
            "support_polygon_margin_m": signed_support_margin_m(
                np.asarray(state_sample["base_position_m"])[:2],
                support_points,
            ),
            "support_slip_m": np.linalg.norm(
                support_points - support_anchors,
                axis=1,
            ),
        }
        records.append(record)

    weight_steps = int(round(float(timing["weight_shift_s"]) * control_hz))
    for index in range(weight_steps):
        progress = (index + 1) / weight_steps
        target = np.asarray(
            targets_for_order(
                dof_names,
                shifted_stance_pose(
                    experiment,
                    transfer=progress,
                ),
            ),
            dtype=np.float32,
        )
        command_and_record(
            target,
            "weight_shift",
            phase_progress=progress,
        )

    shifted_position, _ = robot.get_world_poses()
    shifted_base_x = float(
        _finite("shifted base position", shifted_position)
        .reshape(-1, 3)[0, 0]
    )
    targets = step_targets(
        experiment,
        riser_height_m=height_m,
        shifted_base_x_m=shifted_base_x,
    )
    limit_failures = target_limit_failures(
        targets,
        margin_rad=margin,
    )
    trial["kinematic_targets"] = targets
    trial["kinematic_limit_failures"] = list(limit_failures)
    if limit_failures:
        trial["status"] = "KINEMATIC_FAIL"
        return trial

    body_shift_forward = -float(
        experiment["weight_shift"]["backward_m"]
    )
    start_down = float(stance["down_m"])
    start_forward = float(stance["fore_aft_m"]) - body_shift_forward
    mid = dict(targets["edge_clearance"])
    landing = dict(targets["landing"])

    swing_steps = int(
        round(float(timing["swing_to_edge_s"]) * control_hz)
    )
    for index in range(swing_steps):
        progress = smoothstep((index + 1) / swing_steps)
        swing_down = (1.0 - progress) * start_down + progress * float(
            mid["down_m"]
        )
        swing_forward = (
            (1.0 - progress) * start_forward
            + progress * float(mid["forward_m"])
        )
        target = np.asarray(
            targets_for_order(
                dof_names,
                shifted_stance_pose(
                    experiment,
                    transfer=1.0,
                    swing_down_m=swing_down,
                    swing_forward_m=swing_forward,
                ),
            ),
            dtype=np.float32,
        )
        command_and_record(
            target,
            "swing_to_edge",
            phase_progress=progress,
        )

    lower_steps = int(
        round(float(timing["lower_to_tread_s"]) * control_hz)
    )
    for index in range(lower_steps):
        progress = smoothstep((index + 1) / lower_steps)
        swing_down = (
            (1.0 - progress) * float(mid["down_m"])
            + progress * float(landing["down_m"])
        )
        swing_forward = (
            (1.0 - progress) * float(mid["forward_m"])
            + progress * float(landing["forward_m"])
        )
        target = np.asarray(
            targets_for_order(
                dof_names,
                shifted_stance_pose(
                    experiment,
                    transfer=1.0,
                    swing_down_m=swing_down,
                    swing_forward_m=swing_forward,
                ),
            ),
            dtype=np.float32,
        )
        command_and_record(
            target,
            "lower_to_tread",
            phase_progress=progress,
        )

    hold_target = np.asarray(
        targets_for_order(
            dof_names,
            shifted_stance_pose(
                experiment,
                transfer=1.0,
                swing_down_m=float(landing["down_m"]),
                swing_forward_m=float(landing["forward_m"]),
            ),
        ),
        dtype=np.float32,
    )
    hold_steps = int(round(float(timing["hold_s"]) * control_hz))
    for index in range(hold_steps):
        command_and_record(
            hold_target,
            "hold",
            phase_progress=(index + 1) / hold_steps,
        )

    if not records:
        raise AssertionError("No dynamics samples were recorded")
    acceptance = dict(experiment["acceptance"])
    foot_radius = float(experiment["virtual_foot_radius_m"])
    step_start = float(experiment["step_start_x_m"])
    contact_on = float(acceptance["contact_on_threshold_n"])
    edge_records = [
        record
        for record in records
        if abs(
            float(
                record["feet"]["tip_center_position_m"][
                    swing_index, 0
                ]
            )
            + foot_radius
            - step_start
        )
        <= 0.015
    ]
    edge_clearance = min(
        (
            float(
                record["feet"]["tip_bottom_height_m"][swing_index]
            )
            - height_m
            for record in edge_records
        ),
        default=-1.0,
    )
    tread_contact_flags: list[bool] = []
    riser_strike = False
    for record in records:
        feet_sample = record["feet"]
        toe_center = feet_sample["tip_center_position_m"][swing_index]
        toe_bottom = float(
            feet_sample["tip_bottom_height_m"][swing_index]
        )
        step_force = float(
            feet_sample["step_force_norm_n"][swing_index]
        )
        on_tread = (
            record["phase"] in ("lower_to_tread", "hold")
            and float(toe_center[0]) >= step_start
            and float(
                feet_sample["step_normal_load_n"][swing_index]
            )
            >= contact_on
            and abs(toe_bottom - height_m)
            <= float(acceptance["maximum_landing_height_error_m"])
        )
        tread_contact_flags.append(on_tread)
        if (
            record["phase"] in ("swing_to_edge", "lower_to_tread")
            and step_force >= contact_on
            and toe_bottom < height_m - 0.005
        ):
            riser_strike = True
    hold_records = [
        record for record in records if record["phase"] == "hold"
    ]
    landing_heights = [
        float(record["feet"]["tip_bottom_height_m"][swing_index])
        for record in hold_records[len(hold_records) // 2 :]
    ]
    landing_height_error = abs(
        float(np.median(landing_heights)) - height_m
    )
    active_records = [
        record
        for record in records
        if record["phase"] in (
            "swing_to_edge",
            "lower_to_tread",
            "hold",
        )
    ]
    support_slots = len(active_records) * len(support_indices)
    loaded_support_slots = sum(
        int(
            float(record["feet"]["ground_normal_load_n"][index])
            >= contact_on
        )
        for record in active_records
        for index in support_indices
    )
    support_contact_fraction = (
        loaded_support_slots / support_slots if support_slots else 0.0
    )
    states = [record["state"] for record in records]
    joint_errors = np.vstack(
        [sample["joint_error_rad"] for sample in states]
    )
    requested_pd = np.vstack(
        [sample["requested_pd_nm"] for sample in states]
    )
    base_positions = np.vstack(
        [sample["base_position_m"] for sample in states]
    )
    tilts = np.asarray(
        [sample["body_tilt_deg"] for sample in states],
        dtype=float,
    )
    metrics: dict[str, object] = {
        "edge_clearance_m": edge_clearance,
        "edge_sample_count": len(edge_records),
        "settled_swing_tip_bottom_height_m": float(
            settled_feet["tip_bottom_height_m"][swing_index]
        ),
        "required_swing_tip_lift_m": (
            height_m + float(experiment["swing_clearance_m"])
        ),
        "maximum_swing_tip_bottom_height_m": max(
            float(
                record["feet"]["tip_bottom_height_m"][swing_index]
            )
            for record in active_records
        ),
        "landing_height_error_m": landing_height_error,
        "tread_contact_achieved": any(tread_contact_flags),
        "tread_contact_hold_s": _longest_true_run(
            tread_contact_flags,
            control_hz,
        ),
        "support_contact_fraction": support_contact_fraction,
        "minimum_support_polygon_margin_m": min(
            float(record["support_polygon_margin_m"])
            for record in active_records
        ),
        "maximum_support_tip_slip_m": max(
            float(np.max(record["support_slip_m"]))
            for record in active_records
        ),
        "maximum_body_tilt_deg": float(np.max(tilts)),
        "maximum_base_drop_m": max(
            0.0,
            float(settled_base[2] - np.min(base_positions[:, 2])),
        ),
        "maximum_abs_joint_error_rad": float(
            np.max(np.abs(joint_errors))
        ),
        "pd_saturation_fraction": float(
            np.mean(np.abs(requested_pd) >= effort_cap)
        ),
        "peak_abs_requested_pd_nm": float(
            np.max(np.abs(requested_pd))
        ),
        "peak_abs_requested_pd_nm_by_joint": {
            name: float(np.max(np.abs(requested_pd[:, index])))
            for index, name in enumerate(dof_names)
        },
        "peak_nonfoot_step_collision_force_n": max(
            float(record["nonfoot_step_force_n"])
            for record in records
        ),
        "nonfoot_step_collision": any(
            float(record["nonfoot_step_force_n"])
            >= float(
                acceptance["nonfoot_collision_force_threshold_n"]
            )
            for record in records
        ),
        "riser_strike": riser_strike,
    }
    metrics["achieved_swing_tip_lift_m"] = (
        float(metrics["maximum_swing_tip_bottom_height_m"])
        - float(metrics["settled_swing_tip_bottom_height_m"])
    )
    for key in ("reported_drive_effort_nm", "projected_joint_load_nm"):
        available = [sample[key] for sample in states if key in sample]
        if available:
            values = np.vstack(available)
            metrics[f"peak_abs_{key}"] = float(
                np.max(np.abs(values))
            )
            metrics[f"peak_abs_{key}_by_joint"] = {
                name: float(np.max(np.abs(values[:, index])))
                for index, name in enumerate(dof_names)
            }
    if "peak_abs_projected_joint_load_nm_by_joint" in metrics:
        projected_by_joint = dict(
            metrics["peak_abs_projected_joint_load_nm_by_joint"]
        )
        metrics["projected_load_to_rated_cap_ratio"] = (
            float(metrics["peak_abs_projected_joint_load_nm"]) / effort_cap
        )
        metrics["joints_reaching_95pct_rated_cap"] = [
            name
            for name, value in projected_by_joint.items()
            if float(value) >= 0.95 * effort_cap
        ]
    failures = trial_gate_failures(metrics, acceptance)
    trial.update(
        {
            "status": "PASS" if not failures else "FAIL",
            "dof_names": dof_names,
            "joint_lower_limits_rad": lower.tolist(),
            "joint_upper_limits_rad": upper.tolist(),
            "settled_base_position_m": settled_base.tolist(),
            "settled_normal_load_n_by_leg": {
                leg: float(
                    settled_feet["ground_normal_load_n"][index]
                )
                for index, leg in enumerate(LEGS)
            },
            "metrics": metrics,
            "gate_failures": list(failures),
            "evidence_scope": (
                "One front-left foot placement with three feet on the lower "
                "ground. This does not prove body transfer, rear-foot "
                "placement, repeated ascent, motor temperature, or hardware "
                "safety."
            ),
        }
    )
    if not args.no_screenshots:
        screenshot_path = (
            output_dir
            / "screenshots"
            / f"riser-{int(round(height_m * 1000)):03d}mm.png"
        )
        _capture_screenshot(screenshot_path, height_m)
        trial["screenshot"] = str(screenshot_path)
        trial["screenshot_bytes"] = screenshot_path.stat().st_size
    return trial


report: dict[str, object] = {
    "status": "ERROR",
    "experiment_id": experiment["id"],
    "isaac_sim_version": "6.0.1",
    "config": str(config_path),
    "config_sha256": _sha256(config_path),
    "robot_usd": str(robot_usd),
    "robot_usd_sha256": _sha256(robot_usd) if robot_usd.is_file() else None,
    "output_dir": str(output_dir),
    "requested_riser_heights_m": list(heights_m),
    "current_ppo_front_foot_lift_estimate_m": (
        current_policy_front_lift_m(experiment)
    ),
    "rated_torque_cap_nm": float(experiment["physics"]["effort_cap_nm"]),
    "assessment_scope": (
        "Scripted full-URDF-limit front-foot placement on one static block "
        "under floating-base Isaac dynamics. No RL is run."
    ),
}
exit_code = 1
try:
    if not robot_usd.is_file():
        raise FileNotFoundError(robot_usd)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, object]] = []
    for height_m in heights_m:
        try:
            trials.append(_run_trial(height_m))
        except Exception as exc:
            trials.append(
                {
                    "status": "ERROR",
                    "riser_height_m": height_m,
                    "riser_height_mm": int(round(height_m * 1000.0)),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    status_by_height_mm = {
        str(trial["riser_height_mm"]): str(trial["status"])
        for trial in trials
    }
    all_passed = all(trial["status"] == "PASS" for trial in trials)
    trial_180 = next(
        (
            trial
            for trial in trials
            if int(trial["riser_height_mm"]) == 180
        ),
        None,
    )
    curriculum_authorized = bool(
        all_passed
        and trial_180 is not None
        and trial_180["status"] == "PASS"
    )
    report.update(
        {
            "status": "PASS" if all_passed else "FAIL",
            "trial_status_by_height_mm": status_by_height_mm,
            "trials": trials,
            "curriculum_authorized": curriculum_authorized,
            "decision": (
                "Proceed to a staged 40-196 mm RL curriculum."
                if curriculum_authorized
                else (
                    "Do not resume stair RL. Revise scripted motion, joint "
                    "range, leg/foot geometry, or actuator capability based "
                    "on the per-trial gate failures."
                )
            ),
            "limitations": [
                "The distal contact is a virtual 12.5 mm sphere; no physical foot is modeled.",
                "Reported PD demand is a drive-demand proxy, not measured motor current or thermal load.",
                "A passing trial proves one front-foot placement only, not whole-body stair ascent.",
            ],
        }
    )
    exit_code = 0 if all_passed else 2
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_REAL_STAIR_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
