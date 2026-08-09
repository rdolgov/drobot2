"""Interactively command Drobot's legs beside one 180 mm stair.

Gravity, collisions, contact friction, floating-base dynamics, joint drives,
hard limits, velocity limits, and servo effort limits remain active. This is
manual position-target control, not kinematic link teleportation and not RL.
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
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _contract import validate_config  # noqa: E402
from _manual_control import (  # noqa: E402
    MOTOR_NUMBER_LIST,
    controller_from_experiment,
    motor_controller_from_experiment,
)
from _quadruped_runtime import (  # noqa: E402
    EXPECTED_DOF_NAMES,
    LEGS,
    LINK_LENGTH_M,
    MAX_NO_LOAD_VELOCITY_RAD_S,
    RATED_TORQUE_NM,
    STALL_TORQUE_NM,
    add_robot_reference,
    body_tilt_deg,
    targets_for_order,
)

parser = argparse.ArgumentParser(
    description="Open an interactive, physics-enabled 180 mm stair test."
)
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "real_stair_feasibility.yaml"),
)
parser.add_argument(
    "--riser-mm",
    type=float,
    default=180.0,
    help="Riser height; default and documented test is 180 mm.",
)
parser.add_argument(
    "--torque-profile",
    choices=("rated", "stall"),
    default="rated",
    help="Rated is sustainable. Stall is a short diagnostic only.",
)
parser.add_argument(
    "--control-mode",
    choices=("foot", "motor"),
    default="foot",
    help="Foot-space IK control or direct numbered motor-angle control.",
)
parser.add_argument(
    "--foot-speed-m-s",
    type=float,
    default=0.045,
    help="Held-key Cartesian target speed.",
)
parser.add_argument(
    "--abduction-speed-deg-s",
    type=float,
    default=18.0,
)
parser.add_argument(
    "--motor-speed-deg-s",
    type=float,
    default=25.0,
    help="Held Up/Down target-angle speed in motor control mode.",
)
parser.add_argument(
    "--report",
    default=None,
    help="Default output directory depends on --control-mode.",
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="Automated smoke mode; interactive control requires the GUI.",
)
parser.add_argument(
    "--smoke-seconds",
    type=float,
    default=1.0,
    help="Headless-only target/physics smoke duration.",
)
args, _ = parser.parse_known_args()

if args.riser_mm <= 0.0:
    parser.error("--riser-mm must be positive")
if args.foot_speed_m_s <= 0.0:
    parser.error("--foot-speed-m-s must be positive")
if args.abduction_speed_deg_s <= 0.0:
    parser.error("--abduction-speed-deg-s must be positive")
if args.motor_speed_deg_s <= 0.0:
    parser.error("--motor-speed-deg-s must be positive")
if args.smoke_seconds <= 0.0:
    parser.error("--smoke-seconds must be positive")


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


config_path = _resolve_project_path(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
experiment = validate_config(config)
robot_usd = _resolve_project_path(experiment["robot_usd"])
default_report = (
    "simulation/isaac/output/stair-feasibility-manual-180mm/session.json"
    if args.control_mode == "foot"
    else (
        "simulation/isaac/output/"
        "stair-feasibility-motor-angles-180mm/session.json"
    )
)
report_path = _resolve_project_path(args.report or default_report)
riser_height_m = float(args.riser_mm) / 1000.0
if riser_height_m >= float(experiment["stance"]["down_m"]):
    parser.error("--riser-mm must be below the nominal stance height")

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": 1280,
        "height": 720,
    }
)

# Omniverse imports must follow SimulationApp construction.
import carb.input  # noqa: E402
import omni.appwindow  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.objects import GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


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
    gravity_direction = scene.GetGravityDirectionAttr().Get()
    gravity_magnitude = float(scene.GetGravityMagnitudeAttr().Get())
    if gravity_direction is None or not np.allclose(
        np.asarray(gravity_direction, dtype=float),
        [0.0, 0.0, -1.0],
        atol=1e-6,
    ):
        raise AssertionError(
            f"Physics gravity direction is wrong: {gravity_direction}"
        )
    if not math.isclose(gravity_magnitude, 9.81, abs_tol=1e-6):
        raise AssertionError(
            f"Physics gravity magnitude is wrong: {gravity_magnitude}"
        )
    return {
        "enabled": True,
        "magnitude_m_s2": gravity_magnitude,
        "direction": [0.0, 0.0, -1.0],
    }


def _live_gravity_m_s2() -> float:
    scene = UsdPhysics.Scene.Get(
        stage_utils.get_current_stage(),
        "/World/PhysicsScene",
    )
    if not scene or not scene.GetPrim().IsValid():
        raise AssertionError("Live physics scene is missing")
    return float(scene.GetGravityMagnitudeAttr().Get())


def _gravity_status(gravity_m_s2: float) -> str:
    magnitude = float(gravity_m_s2)
    return (
        "GRAVITY OFF (0.00 m/s^2)"
        if abs(magnitude) <= 1e-4
        else f"GRAVITY ON ({magnitude:.2f} m/s^2)"
    )


def _create_step() -> str:
    path = "/World/ManualStep"
    cube = UsdGeom.Cube.Define(stage_utils.get_current_stage(), path)
    cube.CreateSizeAttr().Set(1.0)
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.78, 0.42, 0.18)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    tread = float(experiment["tread_depth_m"])
    xform.AddTranslateOp().Set(
        Gf.Vec3d(
            float(experiment["step_start_x_m"]) + tread / 2.0,
            0.0,
            riser_height_m / 2.0,
        )
    )
    xform.AddScaleOp().Set(
        Gf.Vec3f(
            tread,
            float(experiment["step_width_m"]),
            riser_height_m,
        )
    )
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.GetPrim().SetCustomDataByKey(
        "drobot:manualRiserHeightM",
        riser_height_m,
    )
    return path


def _apply_contact_material(step_path: str) -> None:
    physics = dict(experiment["physics"])
    stage = stage_utils.get_current_stage()
    material = UsdShade.Material.Define(
        stage,
        "/World/Materials/ManualStairContact",
    )
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(
        float(physics["static_friction"])
    )
    material_api.CreateDynamicFrictionAttr().Set(
        float(physics["dynamic_friction"])
    )
    material_api.CreateRestitutionAttr().Set(
        float(physics["restitution"])
    )
    physx_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_api.CreateCompliantContactStiffnessAttr().Set(
        float(physics["contact_stiffness_n_m"])
    )
    physx_api.CreateCompliantContactDampingAttr().Set(
        float(physics["contact_damping_n_s_m"])
    )
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.CollisionAPI) and (
            path.startswith("/World/Robot")
            or path.startswith("/World/GroundPlane")
            or path == step_path
        ):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )


def _sample_feet(feet: RigidPrim, physics_hz: int) -> dict[str, np.ndarray]:
    positions, orientations = feet.get_world_poses()
    positions = _finite("distal positions", positions).reshape(-1, 3)
    orientations = _finite(
        "distal orientations",
        orientations,
    ).reshape(-1, 4)
    forces = _finite(
        "foot contact forces",
        feet.get_contact_force_matrix(dt=1.0 / physics_hz),
    ).reshape(len(LEGS), 2, 3)
    tip_centers = np.empty((len(LEGS), 3), dtype=float)
    local_tip = np.asarray([LINK_LENGTH_M, 0.0, 0.0], dtype=float)
    for index in range(len(LEGS)):
        tip_centers[index] = positions[index] + _rotate_wxyz(
            orientations[index],
            local_tip,
        )
    return {
        "tip_centers_m": tip_centers,
        "tip_bottom_heights_m": (
            tip_centers[:, 2]
            - float(experiment["virtual_foot_radius_m"])
        ),
        "ground_normal_load_n": np.maximum(forces[:, 0, 2], 0.0),
        "step_normal_load_n": np.maximum(forces[:, 1, 2], 0.0),
    }


def _reset_robot(
    robot: Articulation,
    controller,
    dof_names: list[str],
) -> np.ndarray:
    controller.reset()
    target = np.asarray(
        targets_for_order(
            dof_names,
            controller.joint_targets_by_name(),
        ),
        dtype=np.float32,
    )
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
    robot.set_dof_positions(target)
    robot.set_dof_velocities(
        np.zeros(robot.num_dofs, dtype=np.float32)
    )
    robot.set_dof_position_targets(target)
    return target


def _state_text(
    controller,
    robot: Articulation,
    feet_sample: dict[str, np.ndarray],
    dof_names: list[str],
    gravity_m_s2: float,
) -> str:
    base_position, base_orientation = robot.get_world_poses()
    position = _finite("base position", base_position).reshape(-1, 3)[0]
    orientation = _finite(
        "base orientation",
        base_orientation,
    ).reshape(-1, 4)[0]
    rejection = (
        f" | blocked: {controller.last_rejection}"
        if controller.last_rejection
        else ""
    )
    if controller.mode == "motor":
        joint_name = controller.selected_joint_name
        dof_index = dof_names.index(joint_name)
        positions = _finite(
            "joint positions",
            robot.get_dof_positions(),
        ).reshape(-1)
        leg_index = LEGS.index(controller.selected_leg)
        entry = (
            f" | entering #{controller.selection_buffer}"
            if controller.selection_buffer
            else ""
        )
        return (
            f"{_gravity_status(gravity_m_s2)} | "
            f"SELECTED MOTOR #{controller.selected_motor_number}: "
            f"{joint_name} | "
            f"target={math.degrees(controller.targets_rad[joint_name]):+.2f} "
            f"deg measured={math.degrees(positions[dof_index]):+.2f} deg"
            f"{entry} | "
            f"foot_step_load="
            f"{feet_sample['step_normal_load_n'][leg_index]:.1f} N | "
            f"base_z={position[2]:.3f} m "
            f"tilt={body_tilt_deg(orientation):.1f} deg"
            f"{rejection}"
        )
    leg_index = LEGS.index(controller.selected_leg)
    target = controller.targets[controller.selected_leg]
    return (
        f"selected={controller.selected_leg} "
        f"target(forward={target.forward_m:+.3f}, "
        f"down={target.down_m:.3f}, "
        f"abd={math.degrees(target.hip_abduction_rad):+.1f} deg) | "
        f"tip_z={feet_sample['tip_bottom_heights_m'][leg_index]:.3f} m "
        f"step_load={feet_sample['step_normal_load_n'][leg_index]:.1f} N | "
        f"base_z={position[2]:.3f} m "
        f"tilt={body_tilt_deg(orientation):.1f} deg"
        f"{rejection}"
    )


report: dict[str, object] = {
    "status": "ERROR",
    "meaning": (
        "PASS means the interactive runner executed, not that the robot "
        "climbed the stair."
    ),
    "isaac_sim_version": "6.0.1",
    "riser_height_mm": args.riser_mm,
    "tread_depth_m": float(experiment["tread_depth_m"]),
    "robot_usd": str(robot_usd),
    "config": str(config_path),
    "headless": bool(args.headless),
    "torque_profile": args.torque_profile,
    "control_mode": args.control_mode,
}
exit_code = 1
keyboard_subscription = None
input_interface = None
control_window = None

try:
    stage_utils.create_new_stage(template="sunlight")
    gravity_contract = _configure_physics()
    ground = GroundPlane(
        "/World/GroundPlane",
        positions=[0.0, 0.0, 0.0],
    )
    step_path = _create_step()
    report["asset_reference"] = add_robot_reference(
        stage_utils,
        str(robot_usd),
        "/World/Robot",
    )
    _apply_contact_material(step_path)

    robot = Articulation("/World/Robot", reset_xform_op_properties=True)
    link_path_by_name = dict(
        zip(robot.link_names, robot.link_paths[0], strict=True)
    )
    feet = RigidPrim(
        [
            link_path_by_name[f"{leg}_distal_link"]
            for leg in LEGS
        ],
        contact_filter_paths=[
            ground.planes.paths[0],
            step_path,
        ],
        max_contact_count=128,
    )
    feet.set_enabled_contact_tracking([True], threshold=0.0)

    app_utils.play()
    _update(10)
    dof_names = list(robot.dof_names)
    if len(dof_names) != 12 or set(dof_names) != EXPECTED_DOF_NAMES:
        raise AssertionError(f"Unexpected DOF names: {dof_names}")
    if not feet.is_physics_tensor_entity_valid():
        raise AssertionError("Foot contact tensor view is invalid")

    max_velocities = _finite(
        "joint max velocities",
        robot.get_dof_max_velocities(),
    ).reshape(-1)
    if np.any(max_velocities > MAX_NO_LOAD_VELOCITY_RAD_S + 1e-3):
        raise AssertionError("URDF exceeds the verified ST3215 speed")

    effort_cap = (
        RATED_TORQUE_NM
        if args.torque_profile == "rated"
        else STALL_TORQUE_NM
    )
    robot.set_dof_max_efforts(
        np.full(robot.num_dofs, effort_cap, dtype=np.float32)
    )
    physics = dict(experiment["physics"])
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
    applied_caps = _finite(
        "applied effort caps",
        robot.get_dof_max_efforts(),
    ).reshape(-1)
    if not np.allclose(applied_caps, effort_cap, atol=1e-4):
        raise AssertionError(f"Effort cap was not applied: {applied_caps}")

    controller = (
        controller_from_experiment(experiment)
        if args.control_mode == "foot"
        else motor_controller_from_experiment(experiment)
    )
    control_help = controller.control_help
    current_target = _reset_robot(robot, controller, dof_names)
    control_hz = int(physics["control_hz"])
    settle_steps = int(
        round(float(experiment["timing"]["settle_s"]) * control_hz)
    )
    for _ in range(settle_steps):
        robot.set_dof_position_targets(current_target)
        simulation_app.update()

    settled_feet = _sample_feet(feet, int(physics["physics_hz"]))
    settled_base_position, _ = robot.get_world_poses()
    settled_base_z = float(
        _finite(
            "settled base position",
            settled_base_position,
        ).reshape(-1, 3)[0, 2]
    )
    reset_base_z = float(experiment["reset_base_z_m"])
    if settled_base_z >= reset_base_z - 0.02:
        raise AssertionError(
            "Robot did not settle downward under gravity: "
            f"reset_z={reset_base_z:.6f}, settled_z={settled_base_z:.6f}"
        )
    settled_bottoms = settled_feet["tip_bottom_heights_m"].copy()
    max_lift_by_leg = np.zeros(len(LEGS), dtype=float)
    max_step_load_by_leg = np.zeros(len(LEGS), dtype=float)
    max_tilt = 0.0
    minimum_base_z = math.inf
    max_abs_joint_error = 0.0
    minimum_gravity_m_s2 = _live_gravity_m_s2()
    maximum_gravity_m_s2 = minimum_gravity_m_s2
    samples = 0
    reset_count = 0
    pressed_motion_keys: set[str] = set()
    request_reset = False
    request_pause_toggle = False
    request_print = False
    request_quit = False

    def _on_keyboard_event(event) -> bool:
        global request_reset
        global request_pause_toggle
        global request_print
        global request_quit
        key_name = event.input.name
        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            if controller.select_from_key(key_name):
                pressed_motion_keys.clear()
                if controller.mode == "motor":
                    if controller.selection_buffer:
                        print(
                            "Motor number entry: "
                            f"{controller.selection_buffer}"
                        )
                    else:
                        print(
                            "Selected motor "
                            f"#{controller.selected_motor_number}: "
                            f"{controller.selected_joint_name}"
                        )
                else:
                    print(f"Selected leg: {controller.selected_leg}")
            elif key_name in controller.motion_keys:
                pressed_motion_keys.add(key_name)
            elif (
                event.type == carb.input.KeyboardEventType.KEY_PRESS
                and key_name == "Z"
                and controller.mode == "motor"
            ):
                controller.zero_selected()
                print(
                    f"Motor #{controller.selected_motor_number} "
                    "target set to 0 degrees."
                )
            elif (
                event.type == carb.input.KeyboardEventType.KEY_PRESS
                and key_name == "R"
            ):
                request_reset = True
            elif (
                event.type == carb.input.KeyboardEventType.KEY_PRESS
                and key_name == "SPACE"
            ):
                request_pause_toggle = True
            elif (
                event.type == carb.input.KeyboardEventType.KEY_PRESS
                and key_name == "C"
            ):
                request_print = True
            elif (
                event.type == carb.input.KeyboardEventType.KEY_PRESS
                and key_name in ("X", "ESCAPE")
            ):
                request_quit = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed_motion_keys.discard(key_name)
        return True

    status_label = None
    if not args.headless:
        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard()
        input_interface = carb.input.acquire_input_interface()
        keyboard_subscription = input_interface.subscribe_to_keyboard_events(
            keyboard,
            _on_keyboard_event,
        )
        try:
            import omni.ui as ui

            control_window = ui.Window(
                (
                    "Drobot motor-angle stair controls"
                    if controller.mode == "motor"
                    else "Drobot manual stair controls"
                ),
                width=520,
                height=650 if controller.mode == "motor" else 285,
            )
            with control_window.frame:
                with ui.VStack(spacing=6):
                    ui.Label(
                        (
                            "180 mm MOTOR-ANGLE TEST | LIVE GRAVITY BELOW"
                            if controller.mode == "motor"
                            else "180 mm MANUAL STAIR TEST"
                        ),
                        height=24,
                    )
                    ui.Label(
                        (
                            control_help
                            + "\nMotor numbering:\n"
                            + MOTOR_NUMBER_LIST
                            if controller.mode == "motor"
                            else control_help
                        ),
                        word_wrap=True,
                        height=500 if controller.mode == "motor" else 165,
                    )
                    status_label = ui.Label(
                        "Settling...",
                        word_wrap=True,
                        height=90 if controller.mode == "motor" else 70,
                    )
        except Exception as exc:
            print(f"Optional controls panel unavailable: {exc!r}")
        set_camera_view(
            eye=[0.72, -0.82, 0.56],
            target=[0.20, 0.0, 0.14],
            camera_prim_path="/OmniverseKit_Persp",
        )

    report.update(
        {
            "status": "PASS",
            "rated_torque_nm": RATED_TORQUE_NM,
            "applied_effort_cap_nm": effort_cap,
            "physics_hz": int(physics["physics_hz"]),
            "control_hz": control_hz,
            "gravity": gravity_contract,
            "reset_base_z_m": reset_base_z,
            "settled_base_z_m": settled_base_z,
            "control_mode": controller.mode,
            "controls": control_help,
            "motor_numbering": (
                controller.snapshot().get("motor_number_to_joint")
                if controller.mode == "motor"
                else None
            ),
            "dof_names": dof_names,
        }
    )
    exit_code = 0
    print("\n" + control_help)
    if controller.mode == "motor":
        print("Motor numbering:\n" + MOTOR_NUMBER_LIST)
    print(
        f"Physics is live. Torque profile={args.torque_profile}, "
        f"cap={effort_cap:.6f} N m."
    )
    if args.torque_profile == "stall":
        print(
            "WARNING: stall torque is diagnostic only and is not "
            "sustainable on hardware."
        )

    frame = 0
    headless_steps_remaining = (
        max(1, round(args.smoke_seconds * control_hz))
        if args.headless
        else None
    )

    while simulation_app.is_running() and not request_quit:
        if request_pause_toggle:
            if app_utils.is_playing():
                app_utils.pause()
                print("Physics paused. Press Space to resume.")
            else:
                app_utils.play()
                print("Physics resumed.")
            request_pause_toggle = False
            pressed_motion_keys.clear()

        if request_reset:
            if not app_utils.is_playing():
                app_utils.play()
            current_target = _reset_robot(
                robot,
                controller,
                dof_names,
            )
            reset_count += 1
            request_reset = False
            pressed_motion_keys.clear()
            print("Robot and all leg targets reset.")

        if app_utils.is_playing():
            active_motion_keys = (
                {controller.smoke_key}
                if headless_steps_remaining is not None
                else pressed_motion_keys
            )
            controller.advance(
                active_motion_keys,
                dt_s=1.0 / control_hz,
                foot_speed_m_s=float(args.foot_speed_m_s),
                abduction_speed_rad_s=math.radians(
                    float(args.abduction_speed_deg_s)
                ),
                motor_speed_rad_s=math.radians(
                    float(args.motor_speed_deg_s)
                ),
            )
            current_target = np.asarray(
                targets_for_order(
                    dof_names,
                    controller.joint_targets_by_name(),
                ),
                dtype=np.float32,
            )
            robot.set_dof_position_targets(current_target)

        simulation_app.update()
        if not app_utils.is_playing():
            if status_label is not None and controller.mode == "motor":
                live_gravity = _live_gravity_m_s2()
                entry = (
                    f" | entering #{controller.selection_buffer}"
                    if controller.selection_buffer
                    else ""
                )
                status_label.text = (
                    f"PAUSED | {_gravity_status(live_gravity)} | "
                    f"SELECTED MOTOR "
                    f"#{controller.selected_motor_number}: "
                    f"{controller.selected_joint_name}{entry}"
                )
            continue

        feet_sample = _sample_feet(feet, int(physics["physics_hz"]))
        live_gravity = _live_gravity_m_s2()
        minimum_gravity_m_s2 = min(
            minimum_gravity_m_s2,
            live_gravity,
        )
        maximum_gravity_m_s2 = max(
            maximum_gravity_m_s2,
            live_gravity,
        )
        base_position, base_orientation = robot.get_world_poses()
        base_position = _finite(
            "base position",
            base_position,
        ).reshape(-1, 3)[0]
        base_orientation = _finite(
            "base orientation",
            base_orientation,
        ).reshape(-1, 4)[0]
        positions = _finite(
            "joint positions",
            robot.get_dof_positions(),
        ).reshape(-1)
        max_lift_by_leg = np.maximum(
            max_lift_by_leg,
            feet_sample["tip_bottom_heights_m"] - settled_bottoms,
        )
        max_step_load_by_leg = np.maximum(
            max_step_load_by_leg,
            feet_sample["step_normal_load_n"],
        )
        max_tilt = max(max_tilt, body_tilt_deg(base_orientation))
        minimum_base_z = min(minimum_base_z, float(base_position[2]))
        max_abs_joint_error = max(
            max_abs_joint_error,
            float(np.max(np.abs(current_target - positions))),
        )
        samples += 1
        frame += 1
        status = _state_text(
            controller,
            robot,
            feet_sample,
            dof_names,
            live_gravity,
        )
        if status_label is not None:
            status_label.text = status
        if request_print or frame % control_hz == 0:
            print(status)
            request_print = False
        if headless_steps_remaining is not None:
            headless_steps_remaining -= 1
            if headless_steps_remaining <= 0:
                request_quit = True

    report.update(
        {
            "sample_count": samples,
            "reset_count": reset_count,
            "maximum_foot_lift_m_by_leg": {
                leg: float(max_lift_by_leg[index])
                for index, leg in enumerate(LEGS)
            },
            "maximum_step_normal_load_n_by_leg": {
                leg: float(max_step_load_by_leg[index])
                for index, leg in enumerate(LEGS)
            },
            "tread_contact_detected_by_leg": {
                leg: bool(max_step_load_by_leg[index] >= 2.0)
                for index, leg in enumerate(LEGS)
            },
            "maximum_body_tilt_deg": float(max_tilt),
            "minimum_base_z_m": (
                float(minimum_base_z)
                if math.isfinite(minimum_base_z)
                else None
            ),
            "maximum_abs_joint_tracking_error_rad": float(
                max_abs_joint_error
            ),
            "live_gravity_m_s2_at_exit": _live_gravity_m_s2(),
            "minimum_live_gravity_m_s2": float(minimum_gravity_m_s2),
            "maximum_live_gravity_m_s2": float(maximum_gravity_m_s2),
            "final_controller": controller.snapshot(),
        }
    )
except Exception as exc:
    report["status"] = "ERROR"
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
    exit_code = 1
finally:
    if input_interface is not None and keyboard_subscription is not None:
        try:
            input_interface.unsubscribe_to_keyboard_events(
                keyboard_subscription
            )
        except Exception:
            pass
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Manual session report: {report_path}")
    print("DROBOT_MANUAL_STAIR_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
