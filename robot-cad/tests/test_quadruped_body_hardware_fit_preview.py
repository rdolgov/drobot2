from functools import lru_cache

from robot_cad.assembly import quadruped_body_hardware_fit_preview as preview


@lru_cache(maxsize=1)
def generated_preview():
    return preview.gen_step()


def test_body_hardware_preview_preserves_full_assembly_component_order() -> None:
    assembly = generated_preview()

    assert assembly.label == "quadruped_body_hardware_fit_preview"
    assert tuple(child.label for child in assembly.children) == (
        preview.COMPONENT_ORDER
    )


def test_body_hardware_preview_contains_camera_subassembly() -> None:
    assembly = generated_preview()
    camera = next(
        child
        for child in assembly.children
        if child.label == "lekiwi_camera_assembly"
    )

    assert tuple(child.label for child in camera.children) == (
        "lekiwi_base_camera_mount_reference",
        "arducam_5mp_reference",
    )
