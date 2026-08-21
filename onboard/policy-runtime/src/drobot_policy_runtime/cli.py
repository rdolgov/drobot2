"""Print-only Raspberry Pi policy entry point."""

from __future__ import annotations

import argparse

from .policy import OnnxWalkingPolicy
from .runtime import PolicyCommand, PrintMotorSink, WalkingPolicyLoop
from .sources import Bno085ImuSource, LevelImuSource, NeutralJointStateSource


def _address(text: str) -> int:
    value = int(text, 0)
    if not 0 <= value <= 0x7F:
        raise argparse.ArgumentTypeError("I2C address must be between 0x00 and 0x7f")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the V18 walking policy and print motor targets only."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--imu", choices=("bno085", "level"), default="bno085")
    parser.add_argument("--imu-address", type=_address, default=0x4A)
    parser.add_argument("--imu-axis-map", default="+x,+y,+z")
    parser.add_argument("--forward-m-s", type=float, default=0.15)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--control-hz", type=float, default=60.0)
    parser.add_argument("--print-hz", type=float, default=5.0)
    args = parser.parse_args()

    if not 0.0 < args.print_hz <= args.control_hz:
        parser.error("--print-hz must be positive and no greater than --control-hz")
    imu = (
        Bno085ImuSource(args.imu_address, args.imu_axis_map)
        if args.imu == "bno085"
        else LevelImuSource()
    )
    print(
        "PRINT-ONLY MODE: the policy uses neutral placeholder joint feedback and "
        "will not open or command the servo bus.",
        flush=True,
    )
    loop = WalkingPolicyLoop(
        OnnxWalkingPolicy(args.model),
        imu,
        NeutralJointStateSource(),
        PrintMotorSink(args.print_hz),
        command=PolicyCommand(forward_m_s=args.forward_m_s),
        control_hz=args.control_hz,
    )
    try:
        loop.run(args.duration_s)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
