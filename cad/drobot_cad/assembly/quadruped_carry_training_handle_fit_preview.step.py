"""Installed preview of the carry/training handle on the robot body base."""

from cadgen import srgb
from cadgen.assembly import AssemblyHelper

from drobot_cad.assembly import quadruped_robot
from drobot_cad.parts import (
    quadruped_body,
    quadruped_carry_training_handle,
    st3215_hip_body_mount,
)


def gen_step():
    asm = AssemblyHelper("quadruped_carry_training_handle_fit_preview")
    asm.add(
        quadruped_body.make_body_base(),
        "quadruped_body_base",
        color=srgb("#334155"),
    )
    asm.add(
        quadruped_carry_training_handle.make_handle(),
        "printable_carry_training_handle",
        color=srgb("#E59B35"),
    )
    hip_mount = st3215_hip_body_mount.gen_step()
    for spec in quadruped_robot.LEG_MOUNT_SPECS:
        asm.add(
            hip_mount.moved(quadruped_robot.body_mount_location(spec)),
            f"{spec.name}_body_side_hip_mount",
            color=srgb("#A7AFBA"),
        )
    return asm.build()
