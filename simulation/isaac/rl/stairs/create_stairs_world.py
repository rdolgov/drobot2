"""Create the fixed Drobot stair-training world over the validated flat world."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _stair_geometry import stair_layer_boxes  # noqa: E402
from _stair_rl_contract import (  # noqa: E402
    config_for_height_stage,
    validate_staircase_config,
)

parser = argparse.ArgumentParser(
    description="Create the Drobot fixed stair-climbing Isaac world."
)
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_stairs_v1.yaml"),
)
parser.add_argument(
    "--base-world",
    default="simulation/exports/isaac/quadruped_robot_manual_world.usda",
)
parser.add_argument(
    "--height-stage",
    default=None,
    help="Apply one stair_height_stages entry declared by the config.",
)
parser.add_argument(
    "--output",
    default=None,
    help="Default: the task.world path from the YAML configuration.",
)
parser.add_argument(
    "--report",
    default="simulation/isaac/output/rl/ppo-stairs-v1/world_report.json",
)
args, _ = parser.parse_known_args()


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


config_path = _resolve_project_path(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
if int(config.get("schema_version", 0)) != 1:
    parser.error(f"Unsupported stairs config schema: {config.get('schema_version')}")
try:
    config = config_for_height_stage(config, args.height_stage)
except ValueError as exc:
    parser.error(str(exc))
task_config = dict(config["task"])
staircase = dict(task_config["staircase"])
validate_staircase_config(staircase)
boxes = stair_layer_boxes(staircase)
base_world_path = _resolve_project_path(args.base_world)
world_dependency_paths = tuple(
    _resolve_project_path(value)
    for value in task_config.get("world_dependencies", ())
)
output_path = _resolve_project_path(args.output or task_config["world"])
report_path = _resolve_project_path(args.report)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


def _stage_facts(stage: Usd.Stage) -> dict[str, int]:
    prims = list(stage.Traverse())
    stair_prims = [
        prim
        for prim in prims
        if str(prim.GetPath()).startswith("/World/Stairs/StepLayer_")
    ]
    return {
        "articulation_roots": sum(
            prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in prims
        ),
        "rigid_bodies": sum(
            prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims
        ),
        "revolute_joints": sum(
            prim.IsA(UsdPhysics.RevoluteJoint) for prim in prims
        ),
        "cameras": sum(prim.IsA(UsdGeom.Camera) for prim in prims),
        "imu_sensors": sum(
            prim.GetTypeName() == "IsaacImuSensor" for prim in prims
        ),
        "stair_layers": len(stair_prims),
        "stair_collision_layers": sum(
            prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stair_prims
        ),
        "stair_rigid_bodies": sum(
            prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stair_prims
        ),
    }


report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "isaac_sim_version": "6.0.1",
    "config": str(config_path),
    "config_sha256": _sha256(config_path),
    "height_stage": args.height_stage,
    "base_world": str(base_world_path),
    "base_world_sha256": (
        _sha256(base_world_path) if base_world_path.is_file() else None
    ),
    "world_dependencies": [
        {
            "path": str(path),
            "sha256": _sha256(path) if path.is_file() else None,
        }
        for path in world_dependency_paths
    ],
    "output_world": str(output_path),
    "report": str(report_path),
    "staircase": staircase,
    "layer_boxes": list(boxes),
}
exit_code = 1

try:
    if not base_world_path.is_file():
        raise FileNotFoundError(base_world_path)
    if not world_dependency_paths or world_dependency_paths[0] != base_world_path:
        raise RuntimeError(
            "The first task.world_dependencies entry must match --base-world"
        )
    for dependency_path in world_dependency_paths:
        if not dependency_path.is_file():
            raise FileNotFoundError(dependency_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetTimeCodesPerSecond(int(task_config["physics_hz"]))
    relative_base = os.path.relpath(
        base_world_path,
        output_path.parent,
    ).replace("\\", "/")
    stage.GetRootLayer().subLayerPaths.append(f"./{relative_base}")

    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        raise RuntimeError("The base world does not define /World")
    stage.SetDefaultPrim(world_prim)
    contact_material = UsdShade.Material.Get(
        stage,
        "/World/Materials/PrintedPlaContact",
    )
    if not contact_material or not contact_material.GetPrim().IsValid():
        raise RuntimeError("The validated base-world contact material is missing")

    stairs_xform = UsdGeom.Xform.Define(stage, "/World/Stairs")
    stairs_xform.GetPrim().SetCustomDataByKey(
        "drobot:task",
        str(task_config["id"]),
    )
    stairs_xform.GetPrim().SetCustomDataByKey(
        "drobot:profile",
        json.dumps(staircase, sort_keys=True),
    )
    colors = (
        Gf.Vec3f(0.22, 0.52, 0.68),
        Gf.Vec3f(0.27, 0.59, 0.73),
        Gf.Vec3f(0.32, 0.66, 0.78),
        Gf.Vec3f(0.38, 0.72, 0.82),
    )
    for index, box in enumerate(boxes):
        cube = UsdGeom.Cube.Define(
            stage,
            f"/World/Stairs/{box['name']}",
        )
        cube.CreateSizeAttr().Set(1.0)
        cube.CreateDisplayColorAttr().Set([colors[index % len(colors)]])
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*box["center_xyz_m"]))
        xform.AddScaleOp().Set(Gf.Vec3f(*box["size_xyz_m"]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            contact_material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )

    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(output_path))
    if reopened is None:
        raise RuntimeError(f"Could not reopen generated world: {output_path}")
    if int(round(reopened.GetTimeCodesPerSecond())) != int(
        task_config["physics_hz"]
    ):
        raise RuntimeError("Generated world physics time code rate changed")
    facts = _stage_facts(reopened)
    expected_facts = {
        "articulation_roots": 1,
        "rigid_bodies": 13,
        "revolute_joints": 12,
        "cameras": 1,
        "imu_sensors": 1,
        "stair_layers": int(staircase["step_count"]),
        "stair_collision_layers": int(staircase["step_count"]),
        "stair_rigid_bodies": 0,
    }
    if facts != expected_facts:
        raise RuntimeError(
            f"Generated stair-world facts differ: {facts} != {expected_facts}"
        )
    report.update(
        {
            "status": "PASS",
            "base_world_sublayer": f"./{relative_base}",
            "world_bytes": output_path.stat().st_size,
            "world_sha256": _sha256(output_path),
            "stage_facts": facts,
            "contact_material": "/World/Materials/PrintedPlaContact",
            "scope": (
                "Static stage validation only; policy training and stair "
                "completion are validated separately."
            ),
        }
    )
    exit_code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_STAIRS_WORLD_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
