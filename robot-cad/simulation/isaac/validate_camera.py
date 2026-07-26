"""Render and validate the robot-mounted LeKiwi-compatible RTX camera.

Run with Isaac Sim 6.0's bundled ``python.bat``.  The script opens the
portable manual world, wraps the camera prim already embedded in the robot
asset with the current experimental RTX API, and proves that RGB and depth
frames can be read from the moving robot camera.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
from _quadruped_runtime import add_robot_reference
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(
    description="Validate the mounted quadruped RTX camera."
)
parser.add_argument("--usd", required=True, help="Fixed quadruped USDC")
parser.add_argument("--output", required=True, help="Captured RGB PNG")
parser.add_argument("--report", required=True, help="Validation JSON")
parser.add_argument(
    "--camera-prim",
    default="/World/Robot/Geometry/base_link/lekiwi_camera",
)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp(
    {
        "headless": True,
        "width": 1280,
        "height": 720,
        "renderer": "RaytracedLighting",
    }
)

# Omniverse imports must follow SimulationApp construction.
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.sensors.experimental.rtx import (  # noqa: E402
    CameraSensor,
    RtxCamera,
)
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402

RESOLUTION_HW = (480, 640)
RGB_MIN_STDDEV = 5.0
WARMUP_FRAMES = 60


def _add_validation_targets(stage, camera_position) -> None:
    """Place non-physical colored targets in the camera's forward view."""
    root = UsdGeom.Xform.Define(stage, "/World/CameraValidation")
    root.GetPrim().SetMetadata("comment", "Non-physical camera test targets")
    dome = UsdLux.DomeLight.Define(
        stage,
        "/World/CameraValidation/DomeLight",
    )
    dome.CreateIntensityAttr().Set(1500.0)
    dome.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    key = UsdLux.DistantLight.Define(
        stage,
        "/World/CameraValidation/KeyLight",
    )
    key.CreateIntensityAttr().Set(2500.0)
    key.CreateAngleAttr().Set(3.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(-25.0, 30.0, -20.0)
    )
    camera_x, camera_y, camera_z = camera_position
    for name, xyz, size, color in (
        (
            "CenterRed",
            (camera_x + 0.735, camera_y, camera_z - 0.02),
            0.24,
            (0.90, 0.08, 0.05),
        ),
        (
            "LeftGreen",
            (camera_x + 0.935, camera_y + 0.30, camera_z - 0.10),
            0.22,
            (0.08, 0.80, 0.18),
        ),
        (
            "RightBlue",
            (camera_x + 0.935, camera_y - 0.30, camera_z - 0.10),
            0.22,
            (0.05, 0.22, 0.90),
        ),
    ):
        cube = UsdGeom.Cube.Define(
            stage,
            f"/World/CameraValidation/{name}",
        )
        cube.CreateSizeAttr().Set(size)
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
        UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(*xyz)
        )


def _numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


report = {
    "status": "FAIL",
    "isaac_sim_version": "6.0.1",
    "usd": os.path.abspath(args.usd),
    "camera_prim_path": args.camera_prim,
    "output": os.path.abspath(args.output),
}
exit_code = 1

try:
    usd_path = os.path.abspath(args.usd)
    output_path = os.path.abspath(args.output)
    report_path = os.path.abspath(args.report)
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(usd_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    stage_utils.create_new_stage(template="sunlight")
    report["asset_reference"] = add_robot_reference(
        stage_utils,
        usd_path,
        "/World/Robot",
    )
    for _ in range(30):
        simulation_app.update()
    stage = stage_utils.get_current_stage(backend="usd")

    camera_prim = stage.GetPrimAtPath(args.camera_prim)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise AssertionError(
            f"Mounted USD Camera prim is missing: {args.camera_prim}"
        )
    camera_world = UsdGeom.Xformable(
        camera_prim
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    camera_world_position = camera_world.ExtractTranslation()
    camera_world_forward = camera_world.TransformDir(
        Gf.Vec3d(0.0, 0.0, -1.0)
    ).GetNormalized()
    camera_world_up = camera_world.TransformDir(
        Gf.Vec3d(0.0, 1.0, 0.0)
    ).GetNormalized()
    _add_validation_targets(stage, camera_world_position)

    camera = RtxCamera(
        args.camera_prim,
        tick_rate=None,
        reset_xform_op_properties=False,
    )
    sensor = CameraSensor(
        camera,
        resolution=RESOLUTION_HW,
        annotators=["rgb", "distance_to_image_plane"],
    )

    app_utils.play()
    rgb_data = None
    depth_data = None
    for _ in range(WARMUP_FRAMES):
        simulation_app.update()
        candidate_rgb, _ = sensor.get_data("rgb")
        candidate_depth, _ = sensor.get_data("distance_to_image_plane")
        if candidate_rgb is not None:
            rgb_data = candidate_rgb
        if candidate_depth is not None:
            depth_data = candidate_depth

    if rgb_data is None:
        raise AssertionError("RTX camera produced no RGB frame")
    if depth_data is None:
        raise AssertionError("RTX camera produced no depth frame")

    rgb = _numpy(rgb_data)
    if rgb.shape[:2] != RESOLUTION_HW:
        raise AssertionError(
            f"Unexpected RGB resolution: {rgb.shape} != {RESOLUTION_HW}"
        )
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise AssertionError(f"Unexpected RGB array shape: {rgb.shape}")
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb_stddev = float(np.std(rgb))

    depth = _numpy(depth_data)
    if depth.shape[:2] != RESOLUTION_HW:
        raise AssertionError(
            f"Unexpected depth resolution: {depth.shape} != {RESOLUTION_HW}"
        )
    finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size == 0:
        raise AssertionError("Depth frame contains no finite measurements")
    camera_world_after = UsdGeom.Xformable(
        camera_prim
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    camera_world_position_after = camera_world_after.ExtractTranslation()
    camera_world_forward_after = camera_world_after.TransformDir(
        Gf.Vec3d(0.0, 0.0, -1.0)
    ).GetNormalized()
    camera_world_up_after = camera_world_after.TransformDir(
        Gf.Vec3d(0.0, 1.0, 0.0)
    ).GetNormalized()
    center_patch = rgb[
        RESOLUTION_HW[0] // 2 - 20 : RESOLUTION_HW[0] // 2 + 20,
        RESOLUTION_HW[1] // 2 - 20 : RESOLUTION_HW[1] // 2 + 20,
    ]
    center_rgb_mean = center_patch.mean(axis=(0, 1))
    Image.fromarray(rgb, mode="RGB").save(output_path)
    report.update(
        {
            "camera_world_position_after_render_m": list(
                camera_world_position_after
            ),
            "camera_world_forward_after_render": list(
                camera_world_forward_after
            ),
            "camera_world_up_after_render": list(camera_world_up_after),
            "center_rgb_mean": center_rgb_mean.tolist(),
            "rgb_stddev": rgb_stddev,
            "depth_shape": list(depth.shape),
            "finite_depth_fraction": float(
                finite_depth.size / depth.size
            ),
            "finite_depth_min_m": float(finite_depth.min()),
            "finite_depth_max_m": float(finite_depth.max()),
        }
    )
    if rgb_stddev < RGB_MIN_STDDEV:
        raise AssertionError(
            f"RGB frame is nearly uniform: standard deviation={rgb_stddev}"
        )
    if not (
        center_rgb_mean[0] > center_rgb_mean[1] + 20.0
        and center_rgb_mean[0] > center_rgb_mean[2] + 20.0
    ):
        raise AssertionError(
            "Forward red validation target is not visible in the image "
            f"center: mean RGB={center_rgb_mean}"
        )
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise AssertionError(f"Camera PNG was not written: {output_path}")

    report.update(
        {
            "status": "PASS",
            "sensor_api": "isaacsim.sensors.experimental.rtx",
            "resolution_hw": list(RESOLUTION_HW),
            "annotators": ["rgb", "distance_to_image_plane"],
            "camera_world_position_m": list(camera_world_position),
            "camera_world_forward": list(camera_world_forward),
            "camera_world_up": list(camera_world_up),
            "rgb_shape": list(rgb.shape),
            "rgb_dtype": str(rgb.dtype),
            "rgb_min": int(rgb.min()),
            "rgb_max": int(rgb.max()),
            "rgb_mean": float(rgb.mean()),
            "rgb_stddev": rgb_stddev,
            "depth_shape": list(depth.shape),
            "finite_depth_fraction": float(
                finite_depth.size / depth.size
            ),
            "finite_depth_min_m": float(finite_depth.min()),
            "finite_depth_max_m": float(finite_depth.max()),
            "captured_png_bytes": os.path.getsize(output_path),
        }
    )
    exit_code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    os.makedirs(
        os.path.dirname(os.path.abspath(args.report)),
        exist_ok=True,
    )
    with open(os.path.abspath(args.report), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_ISAAC_CAMERA_RESULT=" + json.dumps(report, sort_keys=True))
    try:
        app_utils.stop()
    except Exception:
        pass
    simulation_app.close()

sys.exit(exit_code)
