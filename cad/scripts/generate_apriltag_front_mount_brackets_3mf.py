"""Package the AprilTag front-mount bracket pair as a Bambu-native 3MF.

The general CAD exporter writes a standards-compliant assembly 3MF, but Bambu
Studio does not reliably accept that component structure.  This packager uses
the validated stock Bambu project template already retained for the AprilTag,
flattens the aligned bracket-pair STL into one printable mesh, and assigns it
to filament 1.
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from generate_apriltag_body_marker_3mf import (
    BAMBU_NAMESPACE,
    CORE_NAMESPACE,
    MODEL_RELATIONSHIP,
    PRODUCTION_NAMESPACE,
    RELATIONSHIP_NAMESPACE,
    _read_binary_stl,
    _slice_info_xml,
    _submodel_xml,
)


def _model_xml() -> bytes:
    ET.register_namespace("", CORE_NAMESPACE)
    ET.register_namespace("p", PRODUCTION_NAMESPACE)
    ET.register_namespace("BambuStudio", BAMBU_NAMESPACE)
    model = ET.Element(
        f"{{{CORE_NAMESPACE}}}model",
        {
            "unit": "millimeter",
            "{http://www.w3.org/XML/1998/namespace}lang": "en-US",
            "requiredextensions": "p",
        },
    )
    ET.SubElement(
        model, f"{{{CORE_NAMESPACE}}}metadata", {"name": "Application"}
    ).text = "BambuStudio-02.08.02.61"
    ET.SubElement(
        model,
        f"{{{CORE_NAMESPACE}}}metadata",
        {"name": "BambuStudio:3mfVersion"},
    ).text = "1"
    ET.SubElement(model, f"{{{CORE_NAMESPACE}}}metadata", {"name": "Title"}).text = (
        "AprilTag front mount brackets"
    )

    resources = ET.SubElement(model, f"{{{CORE_NAMESPACE}}}resources")
    parent = ET.SubElement(
        resources,
        f"{{{CORE_NAMESPACE}}}object",
        {
            "id": "2",
            f"{{{PRODUCTION_NAMESPACE}}}UUID": "10000001-61cb-4c03-9d28-80fed5dfa1dc",
            "type": "model",
        },
    )
    components = ET.SubElement(parent, f"{{{CORE_NAMESPACE}}}components")
    ET.SubElement(
        components,
        f"{{{CORE_NAMESPACE}}}component",
        {
            f"{{{PRODUCTION_NAMESPACE}}}path": "/3D/Objects/object_1.model",
            "objectid": "1",
            f"{{{PRODUCTION_NAMESPACE}}}UUID": "10010000-b206-40ff-9872-83e8017abed1",
        },
    )

    build = ET.SubElement(
        model,
        f"{{{CORE_NAMESPACE}}}build",
        {f"{{{PRODUCTION_NAMESPACE}}}UUID": "1c7c17d8-22b5-4d84-8835-1976022ea369"},
    )
    ET.SubElement(
        build,
        f"{{{CORE_NAMESPACE}}}item",
        {
            "objectid": "2",
            f"{{{PRODUCTION_NAMESPACE}}}UUID": "10000002-b1ec-4553-aec9-835e5b724bb4",
            "transform": "1 0 0 0 1 0 0 0 1 128 128 0",
            "printable": "1",
        },
    )
    return ET.tostring(model, encoding="utf-8", xml_declaration=True)


def _model_relationships_xml() -> bytes:
    ET.register_namespace("", RELATIONSHIP_NAMESPACE)
    relationships = ET.Element(f"{{{RELATIONSHIP_NAMESPACE}}}Relationships")
    ET.SubElement(
        relationships,
        f"{{{RELATIONSHIP_NAMESPACE}}}Relationship",
        {
            "Target": "/3D/Objects/object_1.model",
            "Id": "rel-1",
            "Type": MODEL_RELATIONSHIP,
        },
    )
    return ET.tostring(relationships, encoding="utf-8", xml_declaration=True)


def _model_settings_xml(face_count: int) -> bytes:
    config = ET.Element("config")
    object_element = ET.SubElement(config, "object", {"id": "2"})
    ET.SubElement(
        object_element,
        "metadata",
        {"key": "name", "value": "apriltag_front_mount_brackets"},
    )
    ET.SubElement(object_element, "metadata", {"face_count": str(face_count)})

    part = ET.SubElement(
        object_element,
        "part",
        {
            "id": "1",
            "subtype": "normal_part",
            "uuid": "168e9ee4-511e-415f-9728-df7486e032a6",
        },
    )
    ET.SubElement(part, "metadata", {"key": "name", "value": "front_mount_brackets"})
    ET.SubElement(part, "metadata", {"key": "extruder", "value": "1"})
    ET.SubElement(
        part,
        "metadata",
        {"key": "matrix", "value": "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"},
    )
    ET.SubElement(
        part,
        "metadata",
        {"key": "source_file", "value": "apriltag_front_mount_brackets.stl"},
    )
    ET.SubElement(part, "metadata", {"key": "source_object_id", "value": "0"})
    ET.SubElement(part, "metadata", {"key": "source_volume_id", "value": "0"})
    ET.SubElement(
        part,
        "mesh_stat",
        {
            "face_count": str(face_count),
            "edges_fixed": "0",
            "degenerate_facets": "0",
            "facets_removed": "0",
            "facets_reversed": "0",
            "backwards_edges": "0",
        },
    )

    plate = ET.SubElement(config, "plate")
    ET.SubElement(plate, "metadata", {"key": "plater_id", "value": "1"})
    ET.SubElement(plate, "metadata", {"key": "plater_name", "value": ""})
    ET.SubElement(plate, "metadata", {"key": "locked", "value": "false"})
    ET.SubElement(config, "assemble")
    return ET.tostring(config, encoding="utf-8", xml_declaration=True)


def package_brackets(
    brackets_stl: Path,
    bambu_template: Path,
    output_3mf: Path,
) -> None:
    mesh = _read_binary_stl(brackets_stl)
    output_3mf.parent.mkdir(parents=True, exist_ok=True)

    replacement_entries = {
        "3D/3dmodel.model": _model_xml(),
        "3D/_rels/3dmodel.model.rels": _model_relationships_xml(),
        "3D/Objects/object_1.model": _submodel_xml(
            mesh,
            object_id=1,
            object_uuid="10010000-81cb-4c03-9d28-80fed5dfa1dc",
        ),
        "Metadata/model_settings.config": _model_settings_xml(len(mesh[1])),
        "Metadata/slice_info.config": _slice_info_xml(),
    }

    with tempfile.NamedTemporaryFile(
        prefix=f".{output_3mf.stem}.",
        suffix=".3mf",
        dir=output_3mf.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        with zipfile.ZipFile(bambu_template, mode="r") as template_archive:
            with zipfile.ZipFile(
                temporary_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive:
                template_names = set(template_archive.namelist())
                missing_entries = replacement_entries.keys() - template_names
                if missing_entries:
                    raise ValueError(
                        f"{bambu_template} lacks required entries: "
                        + ", ".join(sorted(missing_entries))
                    )
                for entry in template_archive.infolist():
                    payload = replacement_entries.get(
                        entry.filename,
                        template_archive.read(entry.filename),
                    )
                    archive.writestr(entry.filename, payload)
        temporary_path.replace(output_3mf)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--brackets-stl",
        type=Path,
        default=repository_root / "cad/exports/stl/apriltag_front_mount_brackets.stl",
    )
    parser.add_argument(
        "--bambu-template",
        type=Path,
        default=repository_root / "cad/templates/apriltag_body_marker_bambu_source.3mf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "cad/exports/3mf/apriltag_front_mount_brackets.3mf",
    )
    arguments = parser.parse_args()
    package_brackets(
        arguments.brackets_stl,
        arguments.bambu_template,
        arguments.output,
    )


if __name__ == "__main__":
    main()

