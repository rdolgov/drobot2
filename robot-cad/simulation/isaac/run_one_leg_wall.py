"""Run automatic or interactive range tests on the wall-mounted one-leg asset."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
import tomllib
import traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_cad.urdf import one_leg_wall_testbed as model  # noqa: E402

JOINT_NAMES = ("hip_abduction", "hip_flexion", "knee")
PHYSICS_HZ = 120
APPLICATION_HZ = 60

parser = argparse.ArgumentParser(
    description=(
        "Sweep the physically exercised one-leg ranges on a fixed wall "
        "fixture, or open a matching direct-joint interactive session."
    )
)
parser.add_argument(
    "--usd",
    default="exports/isaac/one_leg_wall_testbed.usdc",
)
parser.add_argument(
    "--hardware-config",
    default="hardware/one-leg-testbed/leg.toml",
)
parser.add_argument(
    "--calibration",
    default="hardware/one-leg-testbed/calibration.json",
)
parser.add_argument(
    "--report",
    default="simulation/isaac/output/one-leg-wall/range_report.json",
)
parser.add_argument(
    "--screenshot",
    default=None,
    help="Optional PNG captured after the automatic combined-pose test.",
)
parser.add_argument(
    "--gravity",
    choices=("both", "zero", "earth"),
    default="both",
)
parser.add_argument("--hold-s", type=float, default=2.0)
parser.add_argument("--settle-s", type=float, default=0.5)
parser.add_argument("--limit-margin-deg", type=float, default=2.0)
parser.add_argument("--max-error-deg", type=float, default=2.0)
parser.add_argument(
    "--effort-cap-nm",
    type=float,
    default=model.NOMINAL_HARDWARE_EFFORT_CAP_NM,
    help=(
        "Default is 30%% of published stall torque, matching the local "
        "hardware torque-limit register only as a nominal comparison."
    ),
)
parser.add_argument("--drive-stiffness", type=float, default=30.0)
parser.add_argument("--drive-damping", type=float, default=4.58366)
parser.add_argument(
    "--interactive",
    action="store_true",
    help="Open the viewport for direct 1/2/3 joint control instead of sweeping.",
)
parser.add_argument(
    "--disable-wall-contact",
    action="store_true",
    help="Diagnostic only: keep the visual wall but disable its collision.",
)
parser.add_argument("--target-speed-deg-s", type=float, default=30.0)
args, _ = parser.parse_known_args()

for name in (
    "hold_s",
    "settle_s",
    "limit_margin_deg",
    "max_error_deg",
    "effort_cap_nm",
    "drive_stiffness",
    "drive_damping",
    "target_speed_deg_s",
):
    if getattr(args, name) <= 0.0:
        parser.error(f"--{name.replace('_', '-')} must be positive")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hardware_snapshot(config_path: Path, calibration_path: Path) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "config": str(config_path),
        "config_present": config_path.is_file(),
        "calibration": str(calibration_path),
        "calibration_present": calibration_path.is_file(),
    }
    if not config_path.is_file():
        snapshot["source"] = "tracked physically exercised model constants"
        snapshot["limits_deg"] = model.PHYSICALLY_EXERCISED_LIMITS_DEG
        snapshot["directions"] = model.HARDWARE_ENCODER_DIRECTIONS
        return snapshot

    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    motors = config.get("motors")
    if not isinstance(motors, list) or len(motors) != 3:
        raise ValueError("Hardware config must contain exactly three motors")
    by_name = {str(motor["name"]): motor for motor in motors}
    if set(by_name) != set(JOINT_NAMES):
        raise ValueError(f"Unexpected hardware motor names: {sorted(by_name)}")
    limits = {
        name: (float(by_name[name]["min_deg"]), float(by_name[name]["max_deg"]))
        for name in JOINT_NAMES
    }
    directions = {name: int(by_name[name]["direction"]) for name in JOINT_NAMES}
    if limits != model.PHYSICALLY_EXERCISED_LIMITS_DEG:
        raise ValueError(
            "Local hardware limits differ from the modeled physical snapshot: "
            f"{limits} != {model.PHYSICALLY_EXERCISED_LIMITS_DEG}"
        )
    if directions != model.HARDWARE_ENCODER_DIRECTIONS:
        raise ValueError(
            "Local encoder directions differ from the modeled snapshot: "
            f"{directions} != {model.HARDWARE_ENCODER_DIRECTIONS}"
        )
    snapshot.update(
        {
            "source": "local hardware config",
            "config_sha256": _sha256(config_path),
            "limits_deg": limits,
            "directions": directions,
            "bus": config.get("bus"),
        }
    )

    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        centers = {
            str(motor["name"]): int(motor["center_tick"])
            for motor in calibration.get("motors", [])
        }
        if centers != {name: model.HARDWARE_CENTER_TICK for name in JOINT_NAMES}:
            raise ValueError(f"Unexpected hardware center ticks: {centers}")
        snapshot["calibration_sha256"] = _sha256(calibration_path)
        snapshot["center_ticks"] = centers
        snapshot["calibration_captured_at_utc"] = calibration.get(
            "captured_at_utc"
        )
    return snapshot


usd_path = _resolve(args.usd)
config_path = _resolve(args.hardware_config)
calibration_path = _resolve(args.calibration)
report_path = _resolve(args.report)
screenshot_path = _resolve(args.screenshot) if args.screenshot else None
hardware_snapshot = _hardware_snapshot(config_path, calibration_path)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.interactive})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from omni.kit.viewport.utility import (  # noqa: E402
    capture_viewport_to_file,
    get_active_viewport,
)
from pxr import Gf, Sdf, UsdPhysics  # noqa: E402


def _finite(name: str, value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value).copy()
    if not np.isfinite(result).all():
        raise AssertionError(f"{name} contains non-finite values: {result}")
    return result


def _update(count: int) -> None:
    for _ in range(count):
        simulation_app.update()


def _add_reference() -> None:
    stage = stage_utils.get_current_stage()
    robot_prim = stage.DefinePrim("/World/Robot", "Xform")
    robot_prim.GetReferences().AddReference(str(usd_path))


def _set_wall_contact(enabled: bool) -> str:
    matches = [
        prim
        for prim in stage_utils.get_current_stage().Traverse()
        if prim.GetName() == "vertical_wall_collision"
    ]
    if len(matches) != 1:
        raise AssertionError(
            "Expected exactly one vertical wall collision prim: "
            f"{[str(prim.GetPath()) for prim in matches]}"
        )
    collision = UsdPhysics.CollisionAPI(matches[0])
    if not collision:
        raise AssertionError(
            f"Wall prim has no CollisionAPI: {matches[0].GetPath()}"
        )
    collision.CreateCollisionEnabledAttr().Set(bool(enabled))
    return str(matches[0].GetPath())


def _gravity_modes() -> list[tuple[str, float]]:
    if args.gravity == "zero":
        return [("zero_gravity", 0.0)]
    if args.gravity == "earth":
        return [("earth_gravity", 9.81)]
    return [("zero_gravity", 0.0), ("earth_gravity", 9.81)]


def _reset(robot: Articulation, target: np.ndarray) -> None:
    robot.set_dof_positions(target)
    robot.set_dof_velocities(np.zeros(robot.num_dofs, dtype=np.float32))
    robot.set_dof_position_targets(target)
    _update(max(1, round(args.settle_s * APPLICATION_HZ)))


def _hold(robot: Articulation, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(max(1, round(args.hold_s * APPLICATION_HZ))):
        robot.set_dof_position_targets(target)
        _update(1)
    return (
        _finite("joint positions", robot.get_dof_positions()).reshape(-1),
        _finite("joint velocities", robot.get_dof_velocities()).reshape(-1),
    )


def _pose_result(
    label: str,
    target: np.ndarray,
    measured: np.ndarray,
    velocity: np.ndarray,
) -> dict[str, object]:
    error_deg = np.degrees(measured - target)
    result = {
        "label": label,
        "target_deg": {
            name: float(math.degrees(target[index]))
            for index, name in enumerate(JOINT_NAMES)
        },
        "measured_deg": {
            name: float(math.degrees(measured[index]))
            for index, name in enumerate(JOINT_NAMES)
        },
        "error_deg": {
            name: float(error_deg[index])
            for index, name in enumerate(JOINT_NAMES)
        },
        "velocity_deg_s": {
            name: float(math.degrees(velocity[index]))
            for index, name in enumerate(JOINT_NAMES)
        },
        "max_abs_error_deg": float(np.max(np.abs(error_deg))),
        "max_abs_velocity_deg_s": float(
            np.max(np.abs(np.degrees(velocity)))
        ),
    }
    result["passed"] = result["max_abs_error_deg"] <= args.max_error_deg
    return result


def _capture(path: Path) -> None:
    set_camera_view(
        eye=[0.52, 0.72, 0.24],
        target=[0.0, 0.0, -0.15],
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
        raise TimeoutError("Timed out waiting for viewport capture")
    task.result()
    for _ in range(120):
        if path.is_file() and path.stat().st_size > 0:
            return
        simulation_app.update()
    raise RuntimeError(f"Isaac Sim did not create a usable PNG: {path}")


def _automatic_sweep(
    robot: Articulation,
    scene: UsdPhysics.Scene,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, object]:
    zero = np.zeros(robot.num_dofs, dtype=np.float32)
    margin = math.radians(args.limit_margin_deg)
    all_modes: dict[str, object] = {}
    screenshot_pose = np.radians([30.0, 60.0, -90.0]).astype(np.float32)

    for gravity_name, gravity_m_s2 in _gravity_modes():
        scene.CreateGravityMagnitudeAttr().Set(gravity_m_s2)
        _reset(robot, zero)
        poses: list[dict[str, object]] = []
        for joint_index, joint_name in enumerate(JOINT_NAMES):
            for endpoint, target_value in (
                ("minimum", lower[joint_index] + margin),
                ("maximum", upper[joint_index] - margin),
            ):
                _reset(robot, zero)
                target = zero.copy()
                target[joint_index] = target_value
                measured, velocity = _hold(robot, target)
                poses.append(
                    _pose_result(
                        f"{joint_name}_{endpoint}",
                        target,
                        measured,
                        velocity,
                    )
                )

        for label, degrees in (
            ("combined_forward", (30.0, 60.0, -90.0)),
            ("combined_reverse", (-30.0, -60.0, 90.0)),
        ):
            _reset(robot, zero)
            target = np.radians(degrees).astype(np.float32)
            measured, velocity = _hold(robot, target)
            poses.append(_pose_result(label, target, measured, velocity))

        passed = all(bool(pose["passed"]) for pose in poses)
        all_modes[gravity_name] = {
            "gravity_m_s2": gravity_m_s2,
            "passed": passed,
            "poses": poses,
        }

    if screenshot_path is not None:
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        _reset(robot, zero)
        _hold(robot, screenshot_pose)
        _capture(screenshot_path)
    return all_modes


def _interactive_session(
    robot: Articulation,
    scene: UsdPhysics.Scene,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, object]:
    import carb
    import omni.appwindow
    import omni.ui as ui

    target = np.zeros(robot.num_dofs, dtype=np.float32)
    selected_index = 0
    pressed: set[str] = set()
    requests = {"quit": False, "reset": False, "print": False, "gravity": False}
    gravity_m_s2 = 9.81
    scene.CreateGravityMagnitudeAttr().Set(gravity_m_s2)
    _reset(robot, target)

    def on_keyboard(event) -> bool:
        nonlocal selected_index
        key = event.input.name
        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            if key in {"KEY_1", "KEY_2", "KEY_3"}:
                selected_index = int(key[-1]) - 1
            elif key in {"UP", "DOWN"}:
                pressed.add(key)
            elif key == "Z":
                target[selected_index] = 0.0
            elif key == "R":
                requests["reset"] = True
            elif key == "G":
                requests["gravity"] = True
            elif key == "C":
                requests["print"] = True
            elif key in {"X", "ESCAPE"}:
                requests["quit"] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed.discard(key)
        return True

    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    subscription = input_interface.subscribe_to_keyboard_events(
        keyboard,
        on_keyboard,
    )
    window = ui.Window("Drobot one-leg wall controls", width=500, height=235)
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label(
                "1/2/3 select | Up/Down move | Z zero | G gravity | "
                "R reset | C print | X/Esc exit"
            )
            status_label = ui.Label("")

    print(
        "One-leg wall controls: 1/2/3 select, Up/Down move, Z zero, "
        "G gravity, R reset, C print, X/Esc exit."
    )
    start = time.monotonic()
    samples = 0
    try:
        while simulation_app.is_running() and not requests["quit"]:
            delta = math.radians(args.target_speed_deg_s) / APPLICATION_HZ
            target[selected_index] += delta * (
                ("UP" in pressed) - ("DOWN" in pressed)
            )
            target[selected_index] = float(
                np.clip(
                    target[selected_index],
                    lower[selected_index] + math.radians(0.5),
                    upper[selected_index] - math.radians(0.5),
                )
            )
            if requests["reset"]:
                target.fill(0.0)
                _reset(robot, target)
                requests["reset"] = False
            if requests["gravity"]:
                gravity_m_s2 = 0.0 if gravity_m_s2 else 9.81
                scene.CreateGravityMagnitudeAttr().Set(gravity_m_s2)
                requests["gravity"] = False
            robot.set_dof_position_targets(target)
            simulation_app.update()
            measured = _finite(
                "interactive joint positions",
                robot.get_dof_positions(),
            ).reshape(-1)
            line = (
                f"#{selected_index + 1} {JOINT_NAMES[selected_index]} | "
                f"target={math.degrees(target[selected_index]):+.1f} deg | "
                f"measured={math.degrees(measured[selected_index]):+.1f} deg | "
                f"gravity={gravity_m_s2:.2f} m/s^2"
            )
            status_label.text = line
            if requests["print"]:
                print(line)
                requests["print"] = False
            samples += 1
    finally:
        input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)
    final = _finite(
        "interactive final positions",
        robot.get_dof_positions(),
    ).reshape(-1)
    return {
        "mode": "interactive",
        "duration_s": time.monotonic() - start,
        "samples": samples,
        "gravity_m_s2": gravity_m_s2,
        "final_target_deg": np.degrees(target).tolist(),
        "final_measured_deg": np.degrees(final).tolist(),
    }


report: dict[str, object] = {
    "status": "ERROR",
    "isaac_sim_version": "6.0.1",
    "mode": "interactive" if args.interactive else "automatic_range_sweep",
    "usd": str(usd_path),
    "hardware_snapshot": hardware_snapshot,
    "fixture": {
        "fixed_to_world": True,
        "wall_contact": not args.disable_wall_contact,
        "self_collision": True,
        "virtual_foot_proxy": False,
        "wall_surface_y_m": model.WALL_SURFACE_Y_M,
    },
    "drive": {
        "effort_cap_nm": float(args.effort_cap_nm),
        "effort_cap_basis": (
            "Nominal 30% of published stall torque to mirror the hardware "
            "register; register value is not a calibrated linear torque model."
        ),
        "stiffness_nm_rad": float(args.drive_stiffness),
        "damping_nm_s_rad": float(args.drive_damping),
    },
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
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    scene.GetPrim().CreateAttribute(
        "physxScene:timeStepsPerSecond",
        Sdf.ValueTypeNames.Int,
    ).Set(PHYSICS_HZ)
    _add_reference()
    report["fixture"]["wall_collision_prim"] = _set_wall_contact(
        not args.disable_wall_contact
    )
    robot = Articulation("/World/Robot", reset_xform_op_properties=True)
    app_utils.play()
    _update(10)

    dof_names = list(robot.dof_names)
    if dof_names != list(JOINT_NAMES):
        raise AssertionError(f"Unexpected one-leg DOF order: {dof_names}")
    lower, upper = robot.get_dof_limits()
    lower = _finite("lower limits", lower).reshape(-1)
    upper = _finite("upper limits", upper).reshape(-1)
    expected_lower = np.radians(
        [model.PHYSICALLY_EXERCISED_LIMITS_DEG[name][0] for name in JOINT_NAMES]
    )
    expected_upper = np.radians(
        [model.PHYSICALLY_EXERCISED_LIMITS_DEG[name][1] for name in JOINT_NAMES]
    )
    if not np.allclose(lower, expected_lower, atol=1e-5):
        raise AssertionError(f"Unexpected lower limits: {lower}")
    if not np.allclose(upper, expected_upper, atol=1e-5):
        raise AssertionError(f"Unexpected upper limits: {upper}")

    robot.set_dof_max_efforts(
        np.full(robot.num_dofs, args.effort_cap_nm, dtype=np.float32)
    )
    robot.set_dof_gains(
        np.full(robot.num_dofs, args.drive_stiffness, dtype=np.float32),
        np.full(robot.num_dofs, args.drive_damping, dtype=np.float32),
    )
    applied_efforts = _finite(
        "applied efforts",
        robot.get_dof_max_efforts(),
    ).reshape(-1)
    applied_stiffness, applied_damping = robot.get_dof_gains()
    report["articulation"] = {
        "dof_names": dof_names,
        "lower_limits_deg": np.degrees(lower).tolist(),
        "upper_limits_deg": np.degrees(upper).tolist(),
        "applied_effort_caps_nm": applied_efforts.tolist(),
        "applied_stiffness_nm_rad": _finite(
            "applied stiffness",
            applied_stiffness,
        ).reshape(-1).tolist(),
        "applied_damping_nm_s_rad": _finite(
            "applied damping",
            applied_damping,
        ).reshape(-1).tolist(),
    }

    if args.interactive:
        report["session"] = _interactive_session(robot, scene, lower, upper)
        report["status"] = "PASS"
    else:
        sweeps = _automatic_sweep(robot, scene, lower, upper)
        report["gravity_sweeps"] = sweeps
        report["screenshot"] = str(screenshot_path) if screenshot_path else None
        passed = all(bool(result["passed"]) for result in sweeps.values())
        report["status"] = "PASS" if passed else "FAIL"
        exit_code = 0 if passed else 2
    if args.interactive:
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
    print("DROBOT_ONE_LEG_WALL_RESULT=" + json.dumps(report, sort_keys=True))
    try:
        app_utils.stop()
        _update(10)
    except Exception:
        pass
    simulation_app.close()

sys.exit(exit_code)
