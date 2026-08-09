"""Installed-fit preview of the complete SO-101 upper arm and ST3215 servo."""

from build123d import import_step
from cadpy.assembly import AssemblyHelper

from drobot_cad.parts.st3215_motor_bay import ST3215_SERVO_STEP
from drobot_cad.parts.upper_arm import gen_step as gen_upper_arm
from drobot_cad.parts.upper_arm import st3215_preview_location


def gen_step():
    """Return a labeled, non-printable fit preview assembly."""
    arm = gen_upper_arm()
    servo = import_step(ST3215_SERVO_STEP).moved(st3215_preview_location())

    asm = AssemblyHelper("upper_arm_st3215_fit_preview")
    asm.add(arm, "upper_arm_with_integral_servo_output_fork")
    asm.add(servo, "waveshare_feetech_st3215_servo")
    return asm.build()
