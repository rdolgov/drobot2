from hashlib import sha256
from pathlib import Path

from drobot_cad.parts.st3215_motor_bay import ST3215_SERVO_STEP
from drobot_cad.parts.upper_arm import REFERENCE_STEP

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_exact_st3215_vendor_model_is_present() -> None:
    assert ST3215_SERVO_STEP == (
        PROJECT_ROOT / "vendor" / "servos" / "waveshare_feetech_st3215_servo.step"
    )
    assert ST3215_SERVO_STEP.stat().st_size == 3_025_292
    assert file_sha256(ST3215_SERVO_STEP) == (
        "29954eb73bd22b3f9536de2c1d8f96843b5c5b32288a8f4cb09709b8b892e39b"
    )


def test_immutable_so101_reference_is_present() -> None:
    assert REFERENCE_STEP == (
        PROJECT_ROOT / "vendor" / "references" / "so101" / "Upper_arm_SO101.step"
    )
    assert REFERENCE_STEP.stat().st_size == 551_437
    assert file_sha256(REFERENCE_STEP) == (
        "efa19a6dd2ccb459248500c76629cfa840630e7e15d9e146394d31da1525dd61"
    )


def test_lekiwi_camera_references_are_present() -> None:
    camera_reference_dir = PROJECT_ROOT / "vendor" / "references" / "lekiwi"
    expected_hashes = {
        "base_camera_mount.stl": (
            "631ed680e524b90dcb59dd315830ed1bac842dfd5e77c412ef4c52ff4da2e854"
        ),
        "arducam_5mp_camera_model.stl": (
            "f5ff94543c19e91a6d5ffb540e2d25347d813d191793405ff2e4178230dec21e"
        ),
    }

    for filename, expected_hash in expected_hashes.items():
        reference_path = camera_reference_dir / filename
        assert reference_path.is_file()
        assert file_sha256(reference_path) == expected_hash


def test_lekiwi_battery_and_servo_adapter_references_are_present() -> None:
    expected_assets = {
        (
            PROJECT_ROOT
            / "vendor"
            / "references"
            / "lekiwi"
            / "lekiwi_12v_5ah_battery_reference.stl"
        ): "f0cd9200f80ff3a75c8b0447eb2e80cd649a6ac86271ff5003571a9f5d94d42c",
        (
            PROJECT_ROOT
            / "vendor"
            / "electronics"
            / "waveshare_bus_servo_adapter_a.step"
        ): "5b04c6802fe661c3f3f2ed02c4decdb2f557bc0d8a85376b6ad6c38db2bb667f",
    }

    for asset_path, expected_hash in expected_assets.items():
        assert asset_path.is_file()
        assert file_sha256(asset_path) == expected_hash
