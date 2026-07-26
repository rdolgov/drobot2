"""Create a portable Isaac Sim world for manual articulation control.

The generated USDA references the self-contained floating robot USDC beside
it, adds Earth gravity and a high-friction floor, applies the sustainable
ST3215 rated-torque cap, and authors a conservative standing target. Open the
world in Isaac Sim, press Play, then use Physics > Articulation Inspector to
command joints by name.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

from _quadruped_runtime import (
    EXPECTED_DOF_NAMES,
    RATED_TORQUE_NM,
    stance_by_name,
)
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(
    description="Create a portable manual-control Isaac world."
)
parser.add_argument("--usd", required=True, help="Floating quadruped USDC")
parser.add_argument("--output", required=True, help="Output manual-world USDA")
parser.add_argument("--report", required=True, help="Output JSON report")
parser.add_argument("--start-z", type=float, default=0.460)
args, _ = parser.parse_known_args()

if args.start_z <= 0.0:
    parser.error("--start-z must be positive")

simulation_app = SimulationApp({"headless": True})

# Omniverse imports must follow SimulationApp construction.
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

EARTH_GRAVITY_M_S2 = 9.81
PHYSICS_HZ = 120
STATIC_FRICTION = 0.90
DYNAMIC_FRICTION = 0.75
RESTITUTION = 0.02


def _stage_facts(stage: Usd.Stage) -> dict[str, int]:
    prims = list(stage.Traverse())
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
        "angular_drives": sum(
            bool(UsdPhysics.DriveAPI.Get(prim, "angular")) for prim in prims
        ),
        "fixed_joints": sum(
            prim.IsA(UsdPhysics.FixedJoint) for prim in prims
        ),
        "physics_scenes": sum(
            prim.IsA(UsdPhysics.Scene) for prim in prims
        ),
    }


report = {
    "status": "FAIL",
    "isaac_sim_version": "6.0.1",
    "source_usd": os.path.abspath(args.usd),
    "output_world": os.path.abspath(args.output),
}
exit_code = 1

try:
    source_usd = os.path.abspath(args.usd)
    output_world = os.path.abspath(args.output)
    report_path = os.path.abspath(args.report)
    if not os.path.isfile(source_usd):
        raise FileNotFoundError(source_usd)
    os.makedirs(os.path.dirname(output_world), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    stage = Usd.Stage.CreateNew(output_world)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetTimeCodesPerSecond(PHYSICS_HZ)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().SetCustomDataByKey(
        "drobot:manualControl",
        "Use Physics > Articulation Inspector after pressing Play.",
    )

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(EARTH_GRAVITY_M_S2)
    scene.GetPrim().CreateAttribute(
        "physxScene:timeStepsPerSecond",
        Sdf.ValueTypeNames.Int,
    ).Set(PHYSICS_HZ)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr().Set(700.0)
    dome.CreateColorAttr().Set(Gf.Vec3f(0.90, 0.94, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr().Set(1800.0)
    key.CreateAngleAttr().Set(2.0)
    key_xform = UsdGeom.Xformable(key.GetPrim())
    key_xform.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, -25.0))

    floor = UsdGeom.Cube.Define(stage, "/World/Ground")
    floor.CreateSizeAttr().Set(1.0)
    floor.CreateDisplayColorAttr().Set([Gf.Vec3f(0.43, 0.68, 0.82)])
    floor_xform = UsdGeom.Xformable(floor.GetPrim())
    floor_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))
    floor_xform.AddScaleOp().Set(Gf.Vec3f(10.0, 10.0, 0.02))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    contact_material = UsdShade.Material.Define(
        stage,
        "/World/Materials/PrintedPlaContact",
    )
    material_api = UsdPhysics.MaterialAPI.Apply(contact_material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(STATIC_FRICTION)
    material_api.CreateDynamicFrictionAttr().Set(DYNAMIC_FRICTION)
    material_api.CreateRestitutionAttr().Set(RESTITUTION)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(
        contact_material.GetPrim()
    )
    physx_material.CreateCompliantContactStiffnessAttr().Set(12000.0)
    physx_material.CreateCompliantContactDampingAttr().Set(45.0)
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(
        contact_material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )

    robot = UsdGeom.Xform.Define(stage, "/World/Robot")
    relative_asset = os.path.relpath(
        source_usd,
        os.path.dirname(output_world),
    ).replace("\\", "/")
    robot.GetPrim().GetReferences().AddReference(f"./{relative_asset}")
    UsdGeom.Xformable(robot.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, args.start_z)
    )

    stance = stance_by_name(
        down_m=0.310,
        fore_aft_m=0.025,
        abduction_deg=0.0,
    )
    authored_joints = {}
    bound_robot_collisions = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and str(prim.GetPath()).startswith(
            "/World/Robot"
        ):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                contact_material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )
            bound_robot_collisions += 1
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint_name = prim.GetName()
        if joint_name not in stance:
            raise AssertionError(f"Unexpected revolute joint: {joint_name}")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            raise AssertionError(f"Joint has no angular drive: {joint_name}")
        target_deg = math.degrees(stance[joint_name])
        drive.GetTargetPositionAttr().Set(target_deg)
        drive.GetMaxForceAttr().Set(RATED_TORQUE_NM)
        authored_joints[joint_name] = target_deg

    if set(authored_joints) != EXPECTED_DOF_NAMES:
        raise AssertionError(
            f"Manual world joint set differs from URDF: {sorted(authored_joints)}"
        )
    if bound_robot_collisions < 13:
        raise AssertionError(
            "Manual world did not bind contact material to every robot link"
        )

    expected_facts = {
        "articulation_roots": 1,
        "rigid_bodies": 13,
        "revolute_joints": 12,
        "angular_drives": 12,
        "fixed_joints": 0,
        "physics_scenes": 1,
    }
    facts = _stage_facts(stage)
    if facts != expected_facts:
        raise AssertionError(f"Manual world facts differ: {facts}")
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(output_world)
    if reopened is None:
        raise AssertionError(f"Could not reopen manual world: {output_world}")
    reopened_facts = _stage_facts(reopened)
    if reopened_facts != expected_facts:
        raise AssertionError(
            f"Reopened manual world facts differ: {reopened_facts}"
        )

    report.update(
        {
            "status": "PASS",
            "world_bytes": os.path.getsize(output_world),
            "source_reference": f"./{relative_asset}",
            "stage_facts": reopened_facts,
            "rated_effort_cap_nm": RATED_TORQUE_NM,
            "standing_targets_deg_by_joint": authored_joints,
            "bound_robot_collision_count": bound_robot_collisions,
            "instructions": (
                "Open the world in Isaac Sim, press Play, then open "
                "Physics > Articulation Inspector and select /World/Robot."
            ),
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
    print("DROBOT_ISAAC_WORLD_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
