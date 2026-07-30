"""Import the fixed one-leg wall testbed URDF into a monolithic Isaac USDC."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import traceback
from pathlib import Path

from isaacsim import SimulationApp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(
    description="Import the fixed Drobot one-leg wall fixture into Isaac Sim."
)
parser.add_argument("--urdf", required=True)
parser.add_argument("--output", required=True, help="Explicit monolithic .usdc output")
parser.add_argument("--report", required=True)
parser.add_argument("--drive-stiffness", type=float, default=30.0)
parser.add_argument("--drive-damping", type=float, default=4.58366)
args, _ = parser.parse_known_args()

if args.drive_stiffness < 0.0 or args.drive_damping < 0.0:
    parser.error("Drive stiffness and damping must be non-negative")
if Path(args.output).suffix.lower() != ".usdc":
    parser.error("--output must end in .usdc")

simulation_app = SimulationApp({"headless": True})

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

FILTERED_NEIGHBOR_JOINTS = {"hip_flexion", "knee"}


def _stage_facts(stage: Usd.Stage) -> dict[str, int]:
    prims = list(stage.Traverse())
    articulation_roots = [
        prim
        for prim in prims
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    return {
        "articulation_roots": len(articulation_roots),
        "self_collision_enabled_roots": sum(
            bool(prim.GetAttribute("newton:selfCollisionEnabled").Get())
            for prim in articulation_roots
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


def _author_moving_neighbor_filters(stage: Usd.Stage) -> list[dict[str, str]]:
    """Filter only the two intentional moving-link pivot overlaps.

    The wall has no mount collision proxy, so wall contact remains active for
    every moving link, including the root hip link.
    """
    filtered_pairs: list[dict[str, str]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        if prim.GetName() not in FILTERED_NEIGHBOR_JOINTS:
            continue
        joint = UsdPhysics.Joint(prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if len(body0_targets) != 1 or len(body1_targets) != 1:
            raise AssertionError(f"Joint body relationship is incomplete: {prim.GetPath()}")
        body0 = stage.GetPrimAtPath(body0_targets[0])
        body1 = stage.GetPrimAtPath(body1_targets[0])
        if not body0.HasAPI(UsdPhysics.RigidBodyAPI):
            raise AssertionError(f"Joint body0 is not rigid: {body0.GetPath()}")
        if not body1.HasAPI(UsdPhysics.RigidBodyAPI):
            raise AssertionError(f"Joint body1 is not rigid: {body1.GetPath()}")
        UsdPhysics.FilteredPairsAPI.Apply(
            body0
        ).CreateFilteredPairsRel().AddTarget(body1.GetPath())
        filtered_pairs.append(
            {
                "joint": prim.GetName(),
                "body0": str(body0.GetPath()),
                "body1": str(body1.GetPath()),
            }
        )
    return filtered_pairs


urdf_path = Path(args.urdf).resolve()
output_path = Path(args.output).resolve()
report_path = Path(args.report).resolve()
report: dict[str, object] = {
    "status": "ERROR",
    "isaac_sim_version": "6.0.1",
    "urdf": str(urdf_path),
    "output": str(output_path),
}
exit_code = 1

try:
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="drobot-one-leg-import-") as temp_dir:
        config = URDFImporterConfig(
            urdf_path=str(urdf_path),
            usd_path=temp_dir,
            merge_fixed_joints=True,
            merge_mesh=True,
            collision_from_visuals=False,
            allow_self_collision=True,
            fix_base=True,
            joint_drive_type="force",
            joint_target_type="position",
            override_joint_stiffness=float(args.drive_stiffness),
            override_joint_damping=float(args.drive_damping),
            run_asset_transformer=False,
            run_multi_physics_conversion=True,
        )
        imported_root = Path(URDFImporter(config).import_urdf()).resolve()
        if not imported_root.is_file():
            raise AssertionError(
                "Isaac importer did not create its reported root USD: "
                f"{imported_root}"
            )
        stage = Usd.Stage.Open(str(imported_root))
        if stage is None:
            raise AssertionError(f"Could not open imported stage: {imported_root}")
        filtered_pairs = _author_moving_neighbor_filters(stage)
        stage.GetRootLayer().Save()
        if not stage.Export(str(output_path)):
            raise AssertionError(f"Could not export monolithic USDC: {output_path}")
        stage = None
        gc.collect()

    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        raise AssertionError(f"Could not reopen monolithic USDC: {output_path}")
    facts = _stage_facts(stage)
    expected_facts = {
        "articulation_roots": 1,
        "self_collision_enabled_roots": 1,
        "rigid_bodies": 4,
        "revolute_joints": 3,
        "angular_drives": 3,
        "filtered_pair_targets": 2,
    }
    if facts != expected_facts:
        raise AssertionError(
            f"Imported one-leg facts differ from contract: {facts} != {expected_facts}"
        )

    report.update(
        {
            "status": "PASS",
            "output_bytes": output_path.stat().st_size,
            "stage_facts": facts,
            "joint_neighbor_filtered_pairs": filtered_pairs,
            "settings": {
                "fixed_base": True,
                "self_collision": True,
                "wall_contact": True,
                "joint_drive_type": "force",
                "joint_target_type": "position",
                "drive_stiffness_nm_rad": float(args.drive_stiffness),
                "drive_damping_nm_s_rad": float(args.drive_damping),
            },
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
    print("DROBOT_ONE_LEG_IMPORT_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
