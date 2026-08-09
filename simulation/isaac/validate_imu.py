"""Validate the mounted body IMU and its walking-policy observation vector.

Run with Isaac Sim's bundled ``python.bat`` against the generated manual
world.  The test holds the rated-torque standing pose, reads the Isaac Sim 6
experimental-physics IMU every application update, and checks gravity,
quaternion normalization, angular-rate settling, and the nine-value training
observation contract.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback

import numpy as np
from _imu_observation import (
    EARTH_GRAVITY_M_S2,
    IMU_OBSERVATION_FIELDS,
    pack_imu_frame,
)
from _quadruped_runtime import (
    EXPECTED_DOF_NAMES,
    RATED_TORQUE_NM,
    stance_by_name,
    targets_for_order,
)
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(
    description="Validate the drobot body IMU in Isaac Sim."
)
parser.add_argument("--world", required=True, help="Manual-world USDA")
parser.add_argument(
    "--imu-prim",
    default="/World/Robot/Geometry/base_link/body_imu",
)
parser.add_argument("--settle-seconds", type=float, default=3.0)
parser.add_argument("--sample-seconds", type=float, default=1.0)
parser.add_argument("--report", required=True)
args, _ = parser.parse_known_args()

if args.settle_seconds <= 0.0 or args.sample_seconds <= 0.0:
    parser.error("settle and sample durations must be positive")

simulation_app = SimulationApp({"headless": True})

# Omniverse imports must follow SimulationApp construction.
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.sensors.experimental.physics import IMUSensor  # noqa: E402

APPLICATION_HZ = 60


def _numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


report = {
    "status": "FAIL",
    "isaac_sim_version": "6.0.1",
    "world": os.path.abspath(args.world),
    "imu_prim_path": args.imu_prim,
}
exit_code = 1

try:
    world_path = os.path.abspath(args.world)
    if not os.path.isfile(world_path):
        raise FileNotFoundError(world_path)
    opened, stage = stage_utils.open_stage(world_path)
    if not opened or stage is None:
        raise RuntimeError(f"Isaac Sim could not open world: {world_path}")
    imu_prim = stage.GetPrimAtPath(args.imu_prim)
    if (
        not imu_prim.IsValid()
        or imu_prim.GetTypeName() != "IsaacImuSensor"
    ):
        raise AssertionError(f"Mounted body IMU is missing: {args.imu_prim}")

    robot = Articulation("/World/Robot", reset_xform_op_properties=True)
    imu_sensor = IMUSensor(args.imu_prim)
    app_utils.play()
    for _ in range(10):
        simulation_app.update()

    dof_names = list(robot.dof_names)
    if set(dof_names) != EXPECTED_DOF_NAMES or robot.num_dofs != 12:
        raise AssertionError(f"Unexpected articulation DOFs: {dof_names}")
    robot.set_dof_max_efforts(
        np.full(robot.num_dofs, RATED_TORQUE_NM, dtype=np.float32)
    )
    robot.set_dof_position_targets(
        np.asarray(
            targets_for_order(
                dof_names,
                stance_by_name(
                    down_m=0.310,
                    fore_aft_m=0.025,
                    abduction_deg=0.0,
                ),
            ),
            dtype=np.float32,
        )
    )

    for _ in range(round(args.settle_seconds * APPLICATION_HZ)):
        simulation_app.update()

    frames = []
    observations = []
    for _ in range(round(args.sample_seconds * APPLICATION_HZ)):
        simulation_app.update()
        frame = imu_sensor.get_data(read_gravity=True)
        if float(frame["time"]) <= 0.0:
            continue
        copied_frame = {
            "time": float(frame["time"]),
            "physics_step": float(frame["physics_step"]),
            "linear_acceleration": _numpy(
                frame["linear_acceleration"]
            ).astype(np.float64).copy(),
            "angular_velocity": _numpy(
                frame["angular_velocity"]
            ).astype(np.float64).copy(),
            "orientation": _numpy(
                frame["orientation"]
            ).astype(np.float64).copy(),
        }
        frames.append(copied_frame)
        observations.append(pack_imu_frame(copied_frame))

    if not frames:
        raise AssertionError("Body IMU produced no valid frames")
    acceleration = np.stack(
        [frame["linear_acceleration"] for frame in frames]
    )
    angular_velocity = np.stack(
        [frame["angular_velocity"] for frame in frames]
    )
    orientation = np.stack([frame["orientation"] for frame in frames])
    observation_matrix = np.stack(observations)

    if not np.all(np.isfinite(observation_matrix)):
        raise AssertionError("Body IMU observation contains non-finite values")
    orientation_norms = np.linalg.norm(orientation, axis=1)
    maximum_quaternion_norm_error = float(
        np.max(np.abs(orientation_norms - 1.0))
    )
    mean_acceleration = np.mean(acceleration, axis=0)
    mean_acceleration_norm = float(np.linalg.norm(mean_acceleration))
    angular_velocity_rms = float(
        np.sqrt(np.mean(np.square(angular_velocity)))
    )

    if maximum_quaternion_norm_error > 1e-3:
        raise AssertionError(
            "Body IMU quaternion normalization error is too high: "
            f"{maximum_quaternion_norm_error}"
        )
    if not 8.5 <= mean_acceleration_norm <= 11.0:
        raise AssertionError(
            "Stationary gravity-included acceleration is unexpected: "
            f"{mean_acceleration.tolist()}"
        )
    if mean_acceleration[2] <= 8.0:
        raise AssertionError(
            "Body IMU +Z is not aligned upward in the standing pose: "
            f"{mean_acceleration.tolist()}"
        )
    if angular_velocity_rms >= 0.5:
        raise AssertionError(
            f"Standing IMU angular velocity did not settle: {angular_velocity_rms}"
        )
    if observation_matrix.shape[1] != len(IMU_OBSERVATION_FIELDS):
        raise AssertionError(
            f"Unexpected IMU observation shape: {observation_matrix.shape}"
        )

    latest = frames[-1]
    report.update(
        {
            "status": "PASS",
            "sample_count": len(frames),
            "settle_seconds": args.settle_seconds,
            "sample_seconds": args.sample_seconds,
            "mean_linear_acceleration_m_s2": mean_acceleration.tolist(),
            "mean_acceleration_norm_m_s2": mean_acceleration_norm,
            "earth_gravity_m_s2": EARTH_GRAVITY_M_S2,
            "angular_velocity_rms_rad_s": angular_velocity_rms,
            "maximum_quaternion_norm_error": (
                maximum_quaternion_norm_error
            ),
            "latest_frame": {
                "time_s": latest["time"],
                "physics_step": latest["physics_step"],
                "linear_acceleration_m_s2": (
                    latest["linear_acceleration"].tolist()
                ),
                "angular_velocity_rad_s": (
                    latest["angular_velocity"].tolist()
                ),
                "orientation_wxyz": latest["orientation"].tolist(),
            },
            "observation_fields": list(IMU_OBSERVATION_FIELDS),
            "latest_observation": observation_matrix[-1].tolist(),
            "observation_shape": list(observation_matrix.shape),
        }
    )
    exit_code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_ISAAC_IMU_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

raise SystemExit(exit_code)
