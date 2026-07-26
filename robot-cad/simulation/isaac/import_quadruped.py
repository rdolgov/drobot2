"""Import the generated quadruped URDF into an Isaac Sim 6.0 USD asset.

Run this file with Isaac Sim's bundled ``python.bat``.  Import fixed and
floating variants into separate output directories because the base constraint
is authored during import.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback

from isaacsim import SimulationApp

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robot_cad.urdf import quadruped_robot as urdf_model  # noqa: E402

parser = argparse.ArgumentParser(
    description="Import the drobot quadruped URDF into an Isaac Sim USD asset."
)
parser.add_argument("--urdf", required=True, help="Input URDF path")
parser.add_argument(
    "--output-dir",
    required=True,
    help="Directory where the importer writes the generated Isaac asset",
)
parser.add_argument("--mode", choices=("fixed", "floating"), required=True)
parser.add_argument("--report", required=True, help="Output JSON report path")
parser.add_argument("--drive-stiffness", type=float, default=30.0)
parser.add_argument("--drive-damping", type=float, default=4.58366)
parser.add_argument(
    "--asset-layout",
    choices=("packaged", "monolithic"),
    default="monolithic",
    help=(
        "Packaged writes a root USDA plus relative payloads. Monolithic writes "
        "one self-contained binary USDC, which is simpler to move and open manually."
    ),
)
self_collision_group = parser.add_mutually_exclusive_group()
self_collision_group.add_argument(
    "--allow-self-collision",
    dest="allow_self_collision",
    action="store_true",
    default=True,
    help=(
        "Enable collision between non-adjacent robot links (default). "
        "Directly connected joint neighbors are filtered after import."
    ),
)
self_collision_group.add_argument(
    "--disable-self-collision",
    dest="allow_self_collision",
    action="store_false",
    help="Disable all articulation self-collision for troubleshooting only.",
)
args, _ = parser.parse_known_args()

if args.drive_stiffness < 0.0 or args.drive_damping < 0.0:
    parser.error("Drive stiffness and damping must be non-negative")

simulation_app = SimulationApp({"headless": True})

# Omniverse imports must follow SimulationApp construction.
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402
from omni.sensors.schema import OmniSensorAPI  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


def _author_lekiwi_camera(stage: Usd.Stage) -> dict[str, object]:
    """Attach the RTX camera prim to the imported base rigid body."""
    base_candidates = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == "base_link"
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(base_candidates) != 1:
        raise AssertionError(
            "Expected one imported base_link rigid body, found "
            f"{[str(prim.GetPath()) for prim in base_candidates]}"
        )

    camera_path = base_candidates[0].GetPath().AppendChild("lekiwi_camera")
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr().Set(urdf_model.CAMERA_FOCAL_LENGTH_MM)
    camera.CreateHorizontalApertureAttr().Set(
        urdf_model.CAMERA_HORIZONTAL_APERTURE_MM
    )
    camera.CreateVerticalApertureAttr().Set(
        urdf_model.CAMERA_VERTICAL_APERTURE_MM
    )
    camera.CreateClippingRangeAttr().Set(
        Gf.Vec2f(*urdf_model.CAMERA_CLIPPING_RANGE_M)
    )
    camera.CreateFocusDistanceAttr().Set(1.0)
    camera.CreateFStopAttr().Set(0.0)

    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(
        Gf.Vec3d(*urdf_model.CAMERA_OPTICAL_XYZ_FROM_BASE_M)
    )
    # USD cameras look along local -Z with +Y image-up.  This rotation makes
    # -Z point along robot +X and +Y point along robot +Z.  Author the
    # equivalent quaternion rather than rotateXYZ because PhysX/Fabric
    # preserves the standard orient op on children of moving rigid bodies.
    camera_orientation_wxyz = (0.5, 0.5, -0.5, -0.5)
    xform.AddOrientOp().Set(
        Gf.Quatf(
            camera_orientation_wxyz[0],
            Gf.Vec3f(*camera_orientation_wxyz[1:]),
        )
    )

    sensor_api = OmniSensorAPI.Apply(camera.GetPrim())
    sensor_api.CreateOmniSensorTickRateAttr().Set(
        urdf_model.CAMERA_TICK_RATE_HZ
    )
    camera.GetPrim().CreateAttribute(
        "drobot:resolutionHeight",
        Sdf.ValueTypeNames.Int,
    ).Set(urdf_model.CAMERA_RESOLUTION_HW[0])
    camera.GetPrim().CreateAttribute(
        "drobot:resolutionWidth",
        Sdf.ValueTypeNames.Int,
    ).Set(urdf_model.CAMERA_RESOLUTION_HW[1])
    camera.GetPrim().CreateAttribute(
        "drobot:rosOpticalFrame",
        Sdf.ValueTypeNames.String,
    ).Set("camera_optical_frame")

    return {
        "prim_path": str(camera_path),
        "parent_link": str(base_candidates[0].GetPath()),
        "translation_m": list(urdf_model.CAMERA_OPTICAL_XYZ_FROM_BASE_M),
        "orientation_wxyz": list(camera_orientation_wxyz),
        "equivalent_rotate_xyz_deg": [90.0, 0.0, -90.0],
        "resolution_hw": list(urdf_model.CAMERA_RESOLUTION_HW),
        "tick_rate_hz": urdf_model.CAMERA_TICK_RATE_HZ,
        "horizontal_fov_deg": urdf_model.CAMERA_HORIZONTAL_FOV_DEG,
        "focal_length_mm": urdf_model.CAMERA_FOCAL_LENGTH_MM,
        "horizontal_aperture_mm": (
            urdf_model.CAMERA_HORIZONTAL_APERTURE_MM
        ),
        "vertical_aperture_mm": urdf_model.CAMERA_VERTICAL_APERTURE_MM,
        "clipping_range_m": list(urdf_model.CAMERA_CLIPPING_RANGE_M),
    }


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
        "cameras": sum(prim.IsA(UsdGeom.Camera) for prim in prims),
    }


def _self_collision_facts(stage: Usd.Stage) -> dict[str, int]:
    prims = list(stage.Traverse())
    articulation_roots = [
        prim
        for prim in prims
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    return {
        "enabled_articulation_roots": sum(
            bool(
                prim.GetAttribute("newton:selfCollisionEnabled").Get()
            )
            for prim in articulation_roots
        ),
        "filtered_pair_targets": sum(
            len(
                UsdPhysics.FilteredPairsAPI(prim)
                .GetFilteredPairsRel()
                .GetTargets()
            )
            for prim in prims
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI)
        ),
    }


def _author_joint_neighbor_filters(stage: Usd.Stage) -> list[dict[str, str]]:
    """Filter intentional overlap only across directly connected link pairs."""
    filtered_pairs: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if len(body0_targets) != 1 or len(body1_targets) != 1:
            continue
        body0_path = body0_targets[0]
        body1_path = body1_targets[0]
        pair_key = tuple(sorted((str(body0_path), str(body1_path))))
        if pair_key in seen_pairs:
            continue
        body0 = stage.GetPrimAtPath(body0_path)
        body1 = stage.GetPrimAtPath(body1_path)
        if not body0.HasAPI(UsdPhysics.RigidBodyAPI):
            raise AssertionError(f"Joint body0 is not rigid: {body0_path}")
        if not body1.HasAPI(UsdPhysics.RigidBodyAPI):
            raise AssertionError(f"Joint body1 is not rigid: {body1_path}")
        filtered_api = UsdPhysics.FilteredPairsAPI.Apply(body0)
        filtered_api.CreateFilteredPairsRel().AddTarget(body1_path)
        seen_pairs.add(pair_key)
        filtered_pairs.append(
            {
                "joint": prim.GetName(),
                "body0": str(body0_path),
                "body1": str(body1_path),
            }
        )
    return filtered_pairs


report = {
    "status": "FAIL",
    "isaac_sim_version": "6.0.1",
    "urdf": os.path.abspath(args.urdf),
    "output_directory": os.path.abspath(args.output_dir),
    "mode": args.mode,
    "asset_layout": args.asset_layout,
}
exit_code = 1

try:
    if not os.path.isfile(args.urdf):
        raise FileNotFoundError(args.urdf)
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)

    config = URDFImporterConfig(
        urdf_path=os.path.abspath(args.urdf),
        usd_path=os.path.abspath(args.output_dir),
        merge_fixed_joints=True,
        merge_mesh=True,
        collision_from_visuals=False,
        allow_self_collision=args.allow_self_collision,
        fix_base=args.mode == "fixed",
        joint_drive_type="force",
        joint_target_type="position",
        override_joint_stiffness=args.drive_stiffness,
        override_joint_damping=args.drive_damping,
        run_asset_transformer=args.asset_layout == "packaged",
        run_multi_physics_conversion=True,
    )
    root_usd = os.path.normpath(URDFImporter(config).import_urdf())
    if not os.path.isfile(root_usd):
        raise AssertionError(f"Isaac importer did not create the reported root USD: {root_usd}")

    stage = Usd.Stage.Open(root_usd)
    if stage is None:
        raise AssertionError(f"Isaac USD stage could not be opened: {root_usd}")
    camera = _author_lekiwi_camera(stage)
    joint_neighbor_filters = (
        _author_joint_neighbor_filters(stage)
        if args.allow_self_collision
        else []
    )
    stage.GetRootLayer().Save()
    stage_facts = _stage_facts(stage)
    expected_stage_facts = {
        "articulation_roots": 1,
        "rigid_bodies": 13,
        "revolute_joints": 12,
        "angular_drives": 12,
        "cameras": 1,
    }
    if stage_facts != expected_stage_facts:
        raise AssertionError(
            "Imported USD articulation facts differ from the URDF contract: "
            f"{stage_facts} != {expected_stage_facts}"
        )
    collision_facts = _self_collision_facts(stage)
    expected_collision_facts = {
        "enabled_articulation_roots": 1 if args.allow_self_collision else 0,
        "filtered_pair_targets": 12 if args.allow_self_collision else 0,
    }
    if collision_facts != expected_collision_facts:
        raise AssertionError(
            "Imported USD self-collision facts differ from policy: "
            f"{collision_facts} != {expected_collision_facts}"
        )
    if args.allow_self_collision and len(joint_neighbor_filters) != 12:
        raise AssertionError(
            "Expected one adjacent-link filter for every revolute joint: "
            f"{joint_neighbor_filters}"
        )

    if args.asset_layout == "monolithic":
        ascii_root_usd = root_usd
        binary_root_usd = os.path.splitext(root_usd)[0] + ".usdc"
        if not stage.Export(binary_root_usd):
            raise AssertionError(
                f"Isaac USD stage could not be exported as binary USDC: {binary_root_usd}"
            )
        stage = None
        gc.collect()
        os.remove(ascii_root_usd)
        root_usd = binary_root_usd
        stage = Usd.Stage.Open(root_usd)
        if stage is None:
            raise AssertionError(
                f"Binary Isaac USD stage could not be reopened: {root_usd}"
            )
        stage_facts = _stage_facts(stage)
        if stage_facts != expected_stage_facts:
            raise AssertionError(
                "Binary Isaac USD articulation facts differ from the URDF "
                f"contract: {stage_facts} != {expected_stage_facts}"
            )
        collision_facts = _self_collision_facts(stage)
        if collision_facts != expected_collision_facts:
            raise AssertionError(
                "Binary Isaac USD self-collision facts differ from policy: "
                f"{collision_facts} != {expected_collision_facts}"
            )

    report.update(
        {
            "status": "PASS",
            "root_usd": root_usd,
            "root_usd_bytes": os.path.getsize(root_usd),
            "stage_facts": stage_facts,
            "self_collision_facts": collision_facts,
            "joint_neighbor_filtered_pairs": joint_neighbor_filters,
            "camera": camera,
            "settings": {
                "merge_fixed_joints": config.merge_fixed_joints,
                "merge_mesh": config.merge_mesh,
                "collision_from_visuals": config.collision_from_visuals,
                "allow_self_collision": config.allow_self_collision,
                "fix_base": config.fix_base,
                "joint_drive_type": config.joint_drive_type,
                "joint_target_type": config.joint_target_type,
                "joint_stiffness_nm_rad": config.override_joint_stiffness,
                "joint_damping_nm_s_rad": config.override_joint_damping,
                "asset_transformer": config.run_asset_transformer,
                "asset_layout": args.asset_layout,
                "multi_physics_conversion": config.run_multi_physics_conversion,
            },
        }
    )
    exit_code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(os.path.abspath(args.report), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_ISAAC_IMPORT_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
