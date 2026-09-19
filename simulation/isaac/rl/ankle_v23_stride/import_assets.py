"""Import the latest V2.3 R3 solid-sole model for the authorized stride training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "simulation/exports/isaac/ankle_v23_stride"


def main():
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from pxr import Usd, UsdPhysics

    manifest = {"schema": "drobot.ankle-assets.v23-r3-stride", "nominal_mass_kg": json.loads((ROOT / "cad/reviews/ankle-v23-stride/mass-properties.json").read_text())["robot_mass_kg"], "variants": {}}
    try:
        for variant in ("active",):
            source = ROOT / "cad/exports/urdf/quadruped_robot_v23_solid_sole.urdf"
            folder = OUT / variant
            folder.mkdir(parents=True, exist_ok=True)
            tree = ET.parse(source)
            robot = tree.getroot()
            # The optical frame has no physics and must not become a default-mass
            # body when fixed-joint merging is disabled to retain the feet.
            for node in list(robot):
                if node.tag == "link" and node.get("name") == "camera_optical_frame":
                    robot.remove(node)
                elif node.tag == "joint" and node.find("child").get("link") == "camera_optical_frame":
                    robot.remove(node)
            for mesh in robot.findall(".//mesh"):
                path = (source.parent / mesh.get("filename")).resolve()
                mesh.set("filename", path.as_posix())
            prepared = folder / "training_input.urdf"
            tree.write(prepared, encoding="utf-8", xml_declaration=True)
            imported = URDFImporter(URDFImporterConfig(
                urdf_path=str(prepared), usd_path=str(folder / "imported"),
                merge_fixed_joints=False, merge_mesh=True,
                collision_from_visuals=False, allow_self_collision=True,
                fix_base=False, joint_drive_type="force", joint_target_type="position",
                override_joint_stiffness=30.0, override_joint_damping=4.5836625,
                run_asset_transformer=False, run_multi_physics_conversion=True,
            )).import_urdf()
            stage = Usd.Stage.Open(str(imported))
            filters = []
            for prim in stage.Traverse():
                if not prim.IsA(UsdPhysics.Joint):
                    continue
                joint = UsdPhysics.Joint(prim)
                a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if a and b:
                    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(a[0])).CreateFilteredPairsRel().AddTarget(b[0])
                    filters.append([str(a[0]), str(b[0])])
            # Training relies on real foot bodies, even in mechanically locked controls.
            bodies = [p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            expected_feet = [f"{leg}_foot_link" for leg in ("front_left", "rear_left", "front_right", "rear_right")]
            if not set(expected_feet).issubset(bodies):
                raise RuntimeError(f"Importer did not preserve foot bodies: {bodies}")
            usd = folder / "robot.usdc"
            stage.Flatten().Export(str(usd))
            manifest["variants"][variant] = {
                "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "usd": str(usd), "usd_sha256": hashlib.sha256(usd.read_bytes()).hexdigest(),
                "bodies": bodies, "adjacent_filtered_pairs": filters,
                "expected_actuated_joints": 16 if variant == "active" else 12,
                "nominal_mass_kg": sum(float(x.get("value")) for x in robot.findall("./link/inertial/mass")),
            }
            (OUT / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"ANKLE_ASSET_READY {variant} {usd}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
