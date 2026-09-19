"""Compute a world foot target and model quantities; never commands hardware."""
from pathlib import Path
import argparse
import numpy as np
from drobot_model import DrobotModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Drobot2 repository root")
    args = parser.parse_args()
    robot = DrobotModel.from_repository(args.repo, variant="rigid_shoe")
    q, body_pose = robot.stance()
    leg = "front_left"
    target = robot.foot_positions(q, base_pose=body_pose)[leg] + [.010, 0., .020]
    q_target = robot.feet_ik({leg: target}, q, base_pose=body_pose)
    print("URDF:", robot.urdf_path)
    print("Source mass (kg):", robot.total_mass)
    print("Target world position (m):", target)
    for name, angle in zip(robot.joint_names, q_target):
        print(f"{name}: {angle:.6f} rad ({np.rad2deg(angle):.3f} deg)")
    print("World foot pose:\n", robot.foot_pose(q_target, leg, base_pose=body_pose))
    print("Joint-space mass matrix shape:", robot.mass_matrix(q_target).shape)
    print("Fixed-base gravity effort (N m):", robot.gravity_forces(q_target, base_pose=body_pose))
    print("Model output only; not hardware commands or a proven support posture.")


if __name__ == "__main__":
    main()
