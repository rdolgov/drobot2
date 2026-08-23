"""Exploded fit preview for the protected Waveshare Bus Servo Adapter (A)."""

from build123d import Location
from cadgen import srgb

from drobot_cad.parts import (
    waveshare_bus_servo_adapter_a,
    waveshare_bus_servo_adapter_enclosure,
)


def gen_step():
    from cadgen.assembly import AssemblyHelper

    enclosure = waveshare_bus_servo_adapter_enclosure
    lid_z = enclosure.BASE_TOTAL_HEIGHT_Z_MM + enclosure.FIT_PREVIEW_LID_GAP_Z_MM

    asm = AssemblyHelper("waveshare_bus_servo_adapter_enclosure_fit_preview")
    asm.add(
        enclosure.make_base(),
        "printable_enclosure_base",
        color=srgb("#315C8C"),
    )
    asm.add(
        waveshare_bus_servo_adapter_a.gen_step().moved(
            Location((0.0, 0.0, enclosure.BOARD_DATUM_Z_MM))
        ),
        "exact_waveshare_bus_servo_adapter_a",
        color=srgb("#2B8A4B"),
    )
    asm.add(
        enclosure.make_lid().moved(Location((0.0, 0.0, lid_z))),
        "printable_screw_on_lid_exploded",
        color=srgb("#B8C4D0"),
    )
    return asm.build()

