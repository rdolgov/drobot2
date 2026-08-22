"""Installed fit preview for the Raspberry Pi 5 enclosure and IMU roof."""

from build123d import Color, Location

from drobot_cad.parts import (
    adafruit_bno085,
    raspberry_pi_5,
    raspberry_pi_5_enclosure,
)


def gen_step():
    from cadgen.assembly import AssemblyHelper

    enclosure = raspberry_pi_5_enclosure
    lid_bottom_z = enclosure.BASE_TOTAL_HEIGHT_Z_MM
    imu_board_bottom_z = lid_bottom_z + enclosure.IMU_BOARD_BOTTOM_ON_LID_Z_MM

    asm = AssemblyHelper("raspberry_pi_5_enclosure_fit_preview")
    asm.add(
        enclosure.make_base(),
        "printable_pi_enclosure_base",
        color=Color(0.16, 0.32, 0.56),
    )
    asm.add(
        raspberry_pi_5.gen_step().moved(
            Location((0.0, 0.0, enclosure.PI_PCB_BOTTOM_Z_MM))
        ),
        "exact_raspberry_pi_5",
        color=Color(0.12, 0.48, 0.20),
    )
    asm.add(
        enclosure.make_lid().moved(Location((0.0, 0.0, lid_bottom_z))),
        "printable_ventilated_lid",
        color=Color(0.72, 0.78, 0.84),
    )
    asm.add(
        adafruit_bno085.gen_step().moved(
            Location((0.0, 0.0, imu_board_bottom_z))
        ),
        "exact_adafruit_bno085",
        color=Color(0.12, 0.30, 0.62),
    )
    asm.add(
        enclosure.make_imu_cover().moved(
            Location(
                (
                    0.0,
                    0.0,
                    imu_board_bottom_z + adafruit_bno085.PCB_THICKNESS_MM,
                )
            )
        ),
        "printable_open_sided_imu_cover",
        color=Color(0.92, 0.62, 0.14),
    )
    return asm.build()
