"""Package the AprilTag marker as a Bambu-compatible two-part 3MF.

The general CAD exporter preserves the black marker as a nested compound of
many solids.  Some slicers, including Bambu Studio, have trouble importing
that nested component tree.  This script flattens each aligned STL into one
mesh and places the white and black meshes directly under one printable
object.  The resulting file therefore remains a single object with two
color-selectable parts, without changing either mesh's coordinates.
"""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


CORE_NAMESPACE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
RELATIONSHIP_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"
PRODUCTION_NAMESPACE = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
BAMBU_NAMESPACE = "http://schemas.bambulab.com/package/2021"
MODEL_RELATIONSHIP = (
    "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
)


def _read_binary_stl(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Read a binary STL and return a deduplicated indexed triangle mesh."""
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"{path} is too short to be a binary STL")

    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(payload) != expected_size:
        raise ValueError(
            f"{path} is not a supported binary STL: expected {expected_size} bytes, "
            f"found {len(payload)}"
        )

    vertices: list[tuple[float, float, float]] = []
    vertex_indices: dict[tuple[float, float, float], int] = {}
    triangles: list[tuple[int, int, int]] = []

    for triangle_index in range(triangle_count):
        offset = 84 + triangle_index * 50 + 12  # Skip the stored face normal.
        indices: list[int] = []
        for vertex_index in range(3):
            vertex = struct.unpack_from("<fff", payload, offset + vertex_index * 12)
            index = vertex_indices.get(vertex)
            if index is None:
                index = len(vertices)
                vertex_indices[vertex] = index
                vertices.append(vertex)
            indices.append(index)
        triangles.append((indices[0], indices[1], indices[2]))

    return vertices, triangles


def _format_coordinate(value: float) -> str:
    # Nine significant digits round-trip a binary32 STL coordinate.
    return format(value, ".9g")


def _add_mesh(
    object_element: ET.Element,
    *,
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
) -> None:
    mesh = ET.SubElement(object_element, f"{{{CORE_NAMESPACE}}}mesh")
    vertices_element = ET.SubElement(mesh, f"{{{CORE_NAMESPACE}}}vertices")
    for x, y, z in vertices:
        ET.SubElement(
            vertices_element,
            f"{{{CORE_NAMESPACE}}}vertex",
            {
                "x": _format_coordinate(x),
                "y": _format_coordinate(y),
                "z": _format_coordinate(z),
            },
        )

    triangles_element = ET.SubElement(mesh, f"{{{CORE_NAMESPACE}}}triangles")
    for first, second, third in triangles:
        ET.SubElement(
            triangles_element,
            f"{{{CORE_NAMESPACE}}}triangle",
            {"v1": str(first), "v2": str(second), "v3": str(third)},
        )


def _submodel_xml(
    mesh_data: tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]],
    *,
    object_id: int,
    object_uuid: str,
) -> bytes:
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
        model,
        f"{{{CORE_NAMESPACE}}}metadata",
        {"name": "BambuStudio:3mfVersion"},
    ).text = "1"
    resources = ET.SubElement(model, f"{{{CORE_NAMESPACE}}}resources")
    object_element = ET.SubElement(
        resources,
        f"{{{CORE_NAMESPACE}}}object",
        {
            "id": str(object_id),
            f"{{{PRODUCTION_NAMESPACE}}}UUID": object_uuid,
            "type": "model",
        },
    )
    _add_mesh(object_element, vertices=mesh_data[0], triangles=mesh_data[1])
    ET.SubElement(model, f"{{{CORE_NAMESPACE}}}build")
    return ET.tostring(model, encoding="utf-8", xml_declaration=True)


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
        "AprilTag body marker tag36h11 ID 0"
    )
    resources = ET.SubElement(model, f"{{{CORE_NAMESPACE}}}resources")
    parent = ET.SubElement(
        resources,
        f"{{{CORE_NAMESPACE}}}object",
        {
            "id": "2",
            f"{{{PRODUCTION_NAMESPACE}}}UUID": "00000001-61cb-4c03-9d28-80fed5dfa1dc",
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
            f"{{{PRODUCTION_NAMESPACE}}}UUID": "00010000-b206-40ff-9872-83e8017abed1",
        },
    )
    ET.SubElement(
        components,
        f"{{{CORE_NAMESPACE}}}component",
        {
            f"{{{PRODUCTION_NAMESPACE}}}path": "/3D/Objects/object_2.model",
            "objectid": "3",
            f"{{{PRODUCTION_NAMESPACE}}}UUID": "00020000-b206-40ff-9872-83e8017abed1",
        },
    )

    build = ET.SubElement(
        model,
        f"{{{CORE_NAMESPACE}}}build",
        {f"{{{PRODUCTION_NAMESPACE}}}UUID": "2c7c17d8-22b5-4d84-8835-1976022ea369"},
    )
    ET.SubElement(
        build,
        f"{{{CORE_NAMESPACE}}}item",
        {
            "objectid": "2",
            f"{{{PRODUCTION_NAMESPACE}}}UUID": "00000002-b1ec-4553-aec9-835e5b724bb4",
            "transform": "1 0 0 0 1 0 0 0 1 128 128 0",
            "printable": "1",
        },
    )
    return ET.tostring(model, encoding="utf-8", xml_declaration=True)


def _model_relationships_xml() -> bytes:
    ET.register_namespace("", RELATIONSHIP_NAMESPACE)
    relationships = ET.Element(f"{{{RELATIONSHIP_NAMESPACE}}}Relationships")
    for relationship_id, object_number in (("rel-1", 1), ("rel-2", 2)):
        ET.SubElement(
            relationships,
            f"{{{RELATIONSHIP_NAMESPACE}}}Relationship",
            {
                "Target": f"/3D/Objects/object_{object_number}.model",
                "Id": relationship_id,
                "Type": MODEL_RELATIONSHIP,
            },
        )
    return ET.tostring(relationships, encoding="utf-8", xml_declaration=True)


def _model_settings_xml(white_faces: int, black_faces: int) -> bytes:
    config = ET.Element("config")
    object_element = ET.SubElement(config, "object", {"id": "2"})
    ET.SubElement(
        object_element,
        "metadata",
        {"key": "name", "value": "apriltag_body_marker_tag36h11_id_0"},
    )
    ET.SubElement(object_element, "metadata", {"face_count": str(white_faces + black_faces)})

    for part_id, name, extruder, face_count, part_uuid in (
        (1, "white_plate", 1, white_faces, "68e9ee4a-511e-415f-9728-df7486e032a6"),
        (3, "black_tag36h11_id_0", 2, black_faces, "e8e176e7-4c72-4cc9-acff-173ce9f6e8c2"),
    ):
        part = ET.SubElement(
            object_element,
            "part",
            {"id": str(part_id), "subtype": "normal_part", "uuid": part_uuid},
        )
        ET.SubElement(part, "metadata", {"key": "name", "value": name})
        ET.SubElement(part, "metadata", {"key": "extruder", "value": str(extruder)})
        ET.SubElement(
            part,
            "metadata",
            {"key": "matrix", "value": "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"},
        )
        ET.SubElement(part, "metadata", {"key": "source_file", "value": f"{name}.stl"})
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


def _slice_info_xml() -> bytes:
    config = ET.Element("config")
    header = ET.SubElement(config, "header")
    ET.SubElement(header, "header_item", {"key": "X-BBL-Client-Type", "value": "slicer"})
    ET.SubElement(
        header,
        "header_item",
        {"key": "X-BBL-Client-Version", "value": "02.08.02.61"},
    )
    return ET.tostring(config, encoding="utf-8", xml_declaration=True)


def _two_color_project_settings(payload: bytes) -> bytes:
    """Activate two stock PLA filaments without adding custom machine G-code."""
    settings = json.loads(payload)
    for key, value in tuple(settings.items()):
        if key.startswith("filament_") and isinstance(value, list) and len(value) == 1:
            settings[key] = [value[0], value[0]]

    settings["filament_colour"] = ["#FFFFFF", "#000000"]
    settings["filament_map"] = ["1", "2"]
    settings["filament_map_2"] = ["1", "2"]
    settings["default_filament_colour"] = ["#FFFFFF", "#000000"]
    return json.dumps(settings, indent=4).encode("utf-8")


def package_marker(
    white_stl: Path,
    black_stl: Path,
    bambu_template: Path,
    output_3mf: Path,
) -> None:
    white_mesh = _read_binary_stl(white_stl)
    black_mesh = _read_binary_stl(black_stl)
    output_3mf.parent.mkdir(parents=True, exist_ok=True)

    replacement_entries = {
        "3D/3dmodel.model": _model_xml(),
        "3D/_rels/3dmodel.model.rels": _model_relationships_xml(),
        "3D/Objects/object_1.model": _submodel_xml(
            white_mesh,
            object_id=1,
            object_uuid="00010000-81cb-4c03-9d28-80fed5dfa1dc",
        ),
        "3D/Objects/object_2.model": _submodel_xml(
            black_mesh,
            object_id=3,
            object_uuid="00020000-81cb-4c03-9d28-80fed5dfa1dc",
        ),
        "Metadata/model_settings.config": _model_settings_xml(
            len(white_mesh[1]), len(black_mesh[1])
        ),
        "Metadata/slice_info.config": _slice_info_xml(),
    }

    with tempfile.NamedTemporaryFile(
        prefix=f".{output_3mf.stem}.", suffix=".3mf", dir=output_3mf.parent, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        with zipfile.ZipFile(bambu_template, mode="r") as template_archive:
            replacement_entries["Metadata/project_settings.config"] = (
                _two_color_project_settings(
                    template_archive.read("Metadata/project_settings.config")
                )
            )
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
                        entry.filename, template_archive.read(entry.filename)
                    )
                    archive.writestr(entry.filename, payload)
        temporary_path.replace(output_3mf)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--white-stl",
        type=Path,
        default=repository_root / "cad/exports/stl/apriltag_body_marker_white.stl",
    )
    parser.add_argument(
        "--black-stl",
        type=Path,
        default=repository_root / "cad/exports/stl/apriltag_body_marker_black.stl",
    )
    parser.add_argument(
        "--bambu-template",
        type=Path,
        default=repository_root / "cad/templates/apriltag_body_marker_bambu_source.3mf",
        help="Bambu Studio 2.8 project providing a complete, validated print profile",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "cad/exports/3mf/apriltag_body_marker.3mf",
    )
    arguments = parser.parse_args()
    package_marker(
        arguments.white_stl,
        arguments.black_stl,
        arguments.bambu_template,
        arguments.output,
    )


if __name__ == "__main__":
    main()
