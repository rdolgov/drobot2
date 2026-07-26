"""Open the generated manual-control world in Isaac Sim.

The standard Isaac Articulation Inspector remains the control surface. This
launcher only opens the correct portable world, starts physics, applies the
sustainable ST3215 effort cap once, and leaves the application running.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
from _quadruped_runtime import (
    EXPECTED_DOF_NAMES,
    RATED_TORQUE_NM,
    stance_by_name,
    targets_for_order,
)
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(
    description="Open the drobot quadruped for manual articulation control."
)
parser.add_argument("--world", required=True, help="Manual-world USDA")
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--onboard-camera",
    action="store_true",
    help="Open the viewport through the mounted LeKiwi-compatible camera",
)
parser.add_argument(
    "--camera-prim",
    default="/World/Robot/Geometry/base_link/lekiwi_camera",
)
parser.add_argument(
    "--smoke-seconds",
    type=float,
    default=2.0,
    help="Headless validation duration; ignored in the interactive UI",
)
parser.add_argument("--report", default=None, help="Optional JSON run report")
args, _ = parser.parse_known_args()

if args.smoke_seconds <= 0.0:
    parser.error("--smoke-seconds must be positive")

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
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from omni.kit.viewport.utility import get_active_viewport  # noqa: E402
from pxr import UsdGeom  # noqa: E402

APPLICATION_HZ = 60


def _numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


report = {
    "status": "FAIL",
    "isaac_sim_version": "6.0.1",
    "world": os.path.abspath(args.world),
    "headless": args.headless,
}
exit_code = 1

try:
    world_path = os.path.abspath(args.world)
    if not os.path.isfile(world_path):
        raise FileNotFoundError(world_path)
    opened, stage = stage_utils.open_stage(world_path)
    if not opened or stage is None:
        raise RuntimeError(f"Isaac Sim could not open world: {world_path}")
    camera_prim = stage.GetPrimAtPath(args.camera_prim)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise AssertionError(f"Mounted camera is missing: {args.camera_prim}")

    robot = Articulation("/World/Robot", reset_xform_op_properties=True)
    app_utils.play()
    for _ in range(10):
        simulation_app.update()

    dof_names = list(robot.dof_names)
    if set(dof_names) != EXPECTED_DOF_NAMES or robot.num_dofs != 12:
        raise AssertionError(f"Unexpected articulation DOFs: {dof_names}")
    if robot.num_links != 13:
        raise AssertionError(f"Expected 13 links, got {robot.num_links}")

    effort_caps = np.full(robot.num_dofs, RATED_TORQUE_NM, dtype=np.float32)
    robot.set_dof_max_efforts(effort_caps)
    stand_targets = np.asarray(
        targets_for_order(
            dof_names,
            stance_by_name(
                down_m=0.310,
                fore_aft_m=0.025,
                abduction_deg=0.0,
            ),
        ),
        dtype=np.float32,
    )
    robot.set_dof_position_targets(stand_targets)
    for _ in range(10):
        simulation_app.update()

    applied_efforts = _numpy(robot.get_dof_max_efforts()).reshape(-1)
    if not np.allclose(applied_efforts, RATED_TORQUE_NM, atol=1e-4):
        raise AssertionError(
            f"Rated ST3215 effort cap was not applied: {applied_efforts}"
        )

    report.update(
        {
            "status": "PASS",
            "articulation_path": "/World/Robot",
            "dof_names": dof_names,
            "link_count": robot.num_links,
            "rated_effort_cap_nm": RATED_TORQUE_NM,
            "camera_prim_path": args.camera_prim,
            "onboard_camera_view": args.onboard_camera,
            "instructions": (
                "Press Play if needed, open Physics > Articulation Inspector, "
                "select /World/Robot, and command the 12 named joints. "
                "Use the viewport camera menu or --onboard-camera for the "
                "robot-mounted RGB view."
            ),
        }
    )
    exit_code = 0

    if args.headless:
        for _ in range(round(args.smoke_seconds * APPLICATION_HZ)):
            simulation_app.update()
    else:
        if args.onboard_camera:
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("Isaac Sim has no active viewport")
            viewport.camera_path = args.camera_prim
        else:
            set_camera_view(
                eye=[1.05, -1.05, 0.78],
                target=[0.0, 0.0, 0.24],
            )
        print(report["instructions"])
        print("Joint order:")
        for name in dof_names:
            print(f"  {name}")
        while simulation_app.is_running():
            simulation_app.update()
except Exception as exc:
    report["status"] = "FAIL"
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
    exit_code = 1
finally:
    if args.report:
        report_path = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print("DROBOT_ISAAC_OPEN_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
