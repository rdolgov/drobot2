from hashlib import sha256
from pathlib import Path

from robot_cad.parts.st3215_motor_bay import ST3215_SERVO_STEP
from robot_cad.parts.upper_arm import REFERENCE_STEP

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
