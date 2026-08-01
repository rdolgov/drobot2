"""Gymnasium walking environment backed by the validated Isaac Sim world.

Import this module only after constructing ``isaacsim.SimulationApp``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import gymnasium as gym
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from _imu_observation import (
    IMU_OBSERVATION_FIELDS,
    pack_imu_frame,
    rotate_world_vector_into_body,
)
from _quadruped_runtime import (
    EXPECTED_DOF_NAMES,
    MAX_NO_LOAD_VELOCITY_RAD_S,
    RATED_TORQUE_NM,
    STALL_TORQUE_NM,
    stance_by_name,
    targets_for_order,
)
from _rl_contract import (
    POLICY_OBSERVATION_CLIP,
    POLICY_OBSERVATION_FIELDS,
    POLICY_OBSERVATION_SIZE,
    pack_policy_observation,
    termination_reasons,
    walking_reward_terms,
)
from gymnasium import spaces
from isaacsim.core.experimental.prims import Articulation
from isaacsim.sensors.experimental.physics import IMUSensor
from pxr import UsdGeom


def _numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _finite_flat(value, label: str) -> np.ndarray:
    result = _numpy(value).astype(np.float32).reshape(-1)
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"{label} contains non-finite values: {result}")
    return result


def _joint_kind(name: str, values_by_kind: Mapping[str, object]) -> str:
    matches = [kind for kind in values_by_kind if name.endswith(kind)]
    if len(matches) != 1:
        raise ValueError(
            f"Could not resolve one joint kind for {name!r}: {matches}"
        )
    return matches[0]


class QuadrupedWalkEnv(gym.Env):
    """One floating quadruped learning to track a forward body-speed command."""

    metadata = {"render_modes": [None, "human"], "render_fps": 60}

    def __init__(
        self,
        simulation_app,
        *,
        world_path: str,
        task_config: Mapping[str, object],
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.simulation_app = simulation_app
        self.render_mode = render_mode
        self.world_path = os.path.abspath(world_path)
        self.config = dict(task_config)
        if not os.path.isfile(self.world_path):
            raise FileNotFoundError(self.world_path)

        self.control_hz = int(self.config["control_hz"])
        self.physics_hz = int(self.config["physics_hz"])
        if self.control_hz <= 0 or self.physics_hz <= 0:
            raise ValueError("Physics and control rates must be positive")
        if self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be an integer multiple of control_hz")
        self.control_dt_s = 1.0 / self.control_hz
        self.max_episode_steps = int(
            round(float(self.config["episode_seconds"]) * self.control_hz)
        )
        self.reset_settle_steps = int(
            round(float(self.config["reset_settle_seconds"]) * self.control_hz)
        )
        self.command_velocity = np.asarray(
            self.config["target_velocity_body_m_s"],
            dtype=np.float32,
        ).reshape(3)
        self.reward_config = dict(self.config["reward"])
        self.termination_config = dict(self.config["termination"])
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-POLICY_OBSERVATION_CLIP,
            high=POLICY_OBSERVATION_CLIP,
            shape=(POLICY_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        opened, stage = stage_utils.open_stage(self.world_path)
        if not opened or stage is None:
            raise RuntimeError(f"Isaac Sim could not open world: {self.world_path}")
        self.stage = stage
        robot_prim_path = str(self.config["robot_prim"])
        imu_prim_path = str(self.config["imu_prim"])
        camera_prim_path = str(self.config["camera_prim"])
        imu_prim = stage.GetPrimAtPath(imu_prim_path)
        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        if not imu_prim.IsValid() or imu_prim.GetTypeName() != "IsaacImuSensor":
            raise RuntimeError(f"Mounted body IMU is missing: {imu_prim_path}")
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Mounted camera is missing: {camera_prim_path}")
        if int(round(stage.GetTimeCodesPerSecond())) != self.physics_hz:
            raise RuntimeError(
                "World timeCodesPerSecond does not match task physics_hz: "
                f"{stage.GetTimeCodesPerSecond()} != {self.physics_hz}"
            )

        self.robot = Articulation(
            robot_prim_path,
            reset_xform_op_properties=True,
        )
        self.imu_sensor = IMUSensor(imu_prim_path)
        self._before_physics_play()
        app_utils.play()
        self._update(10)
        self.dof_names = list(self.robot.dof_names)
        if (
            self.robot.num_dofs != 12
            or set(self.dof_names) != EXPECTED_DOF_NAMES
            or self.robot.num_links != 13
        ):
            raise RuntimeError(
                "Unexpected articulation contract: "
                f"dofs={self.dof_names}, links={self.robot.num_links}"
            )

        self.robot_hardware_profile = dict(
            self.config.get("robot_hardware_profile", {})
        )
        self.effort_cap_nm = float(
            self.robot_hardware_profile.get(
                "effort_cap_nm",
                RATED_TORQUE_NM,
            )
        )
        if not 0.0 < self.effort_cap_nm <= STALL_TORQUE_NM:
            raise ValueError(
                "robot_hardware_profile.effort_cap_nm must be positive and "
                f"no greater than stall torque ({STALL_TORQUE_NM} N*m)"
            )
        limits_by_kind = dict(
            self.robot_hardware_profile.get("joint_limits_deg", {})
        )
        if limits_by_kind:
            expected_kinds = {
                "hip_abduction",
                "hip_flexion",
                "knee",
            }
            if set(limits_by_kind) != expected_kinds:
                raise ValueError(
                    "robot_hardware_profile.joint_limits_deg must define "
                    f"{sorted(expected_kinds)}"
                )
            lower_limits: list[float] = []
            upper_limits: list[float] = []
            for name in self.dof_names:
                kind = _joint_kind(name, limits_by_kind)
                limits_deg = tuple(float(value) for value in limits_by_kind[kind])
                if len(limits_deg) != 2 or not limits_deg[0] < limits_deg[1]:
                    raise ValueError(
                        f"Invalid joint limit for {kind}: {limits_deg}"
                    )
                lower_limits.append(np.deg2rad(limits_deg[0]))
                upper_limits.append(np.deg2rad(limits_deg[1]))
            self.robot.set_dof_limits(
                lower=np.asarray(lower_limits, dtype=np.float32),
                upper=np.asarray(upper_limits, dtype=np.float32),
            )

        self.lower_limits = _finite_flat(
            self.robot.get_dof_limits()[0],
            "joint lower limits",
        )
        self.upper_limits = _finite_flat(
            self.robot.get_dof_limits()[1],
            "joint upper limits",
        )
        self.max_velocities = _finite_flat(
            self.robot.get_dof_max_velocities(),
            "joint maximum velocities",
        )
        if np.any(self.max_velocities > MAX_NO_LOAD_VELOCITY_RAD_S + 1e-3):
            raise RuntimeError(
                "URDF joint speed exceeds the verified ST3215 limit: "
                f"{self.max_velocities}"
            )
        self.robot.set_dof_max_efforts(
            np.full(12, self.effort_cap_nm, dtype=np.float32)
        )
        stance = dict(self.config["nominal_stance"])
        self.nominal_positions = np.asarray(
            targets_for_order(
                self.dof_names,
                stance_by_name(
                    down_m=float(stance["down_m"]),
                    fore_aft_m=float(stance["fore_aft_m"]),
                    abduction_deg=float(stance["abduction_deg"]),
                ),
            ),
            dtype=np.float32,
        )
        action_scale_by_kind = dict(self.config["action_scale_rad"])
        self.action_scale = np.asarray(
            [
                float(
                    action_scale_by_kind[
                        next(
                            kind
                            for kind in action_scale_by_kind
                            if name.endswith(kind)
                        )
                    ]
                )
                for name in self.dof_names
            ],
            dtype=np.float32,
        )
        margin = 1e-3
        if np.any(self.nominal_positions <= self.lower_limits + margin) or np.any(
            self.nominal_positions >= self.upper_limits - margin
        ):
            raise RuntimeError("Nominal stance touches a joint limit")

        self.previous_action = np.zeros(12, dtype=np.float32)
        self.previous_target = self.nominal_positions.copy()
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_origin = np.zeros(3, dtype=np.float32)
        self.minimum_height_m = float("inf")
        self.maximum_tilt_deg = 0.0
        self.completed_episode_metrics: list[dict[str, object]] = []
        self._closed = False

    def _before_physics_play(self) -> None:
        """Allow specialized tasks to register physics tensor views."""

    def _update(self, count: int = 1) -> None:
        for _ in range(count):
            self.simulation_app.update()

    def _read_state(self) -> dict[str, np.ndarray | float]:
        joint_positions = _finite_flat(
            self.robot.get_dof_positions(),
            "joint positions",
        )
        joint_velocities = _finite_flat(
            self.robot.get_dof_velocities(),
            "joint velocities",
        )
        base_position_raw, base_orientation_raw = self.robot.get_world_poses()
        base_position = _finite_flat(base_position_raw, "base position")[:3]
        base_orientation = _finite_flat(
            base_orientation_raw,
            "base orientation",
        )[:4]
        linear_velocity_raw, _ = self.robot.get_velocities()
        world_linear_velocity = _finite_flat(
            linear_velocity_raw,
            "base linear velocity",
        )[:3]
        body_linear_velocity = rotate_world_vector_into_body(
            world_linear_velocity,
            base_orientation,
        )
        imu_frame = self.imu_sensor.get_data(read_gravity=True)
        if float(imu_frame["time"]) <= 0.0:
            raise RuntimeError("Body IMU did not produce a valid timestamp")
        imu_observation = pack_imu_frame(imu_frame)
        observation = pack_policy_observation(
            command_velocity_xyz=self.command_velocity,
            imu_observation=imu_observation,
            joint_positions=joint_positions,
            nominal_joint_positions=self.nominal_positions,
            joint_velocities=joint_velocities,
            joint_max_velocities=self.max_velocities,
            previous_action=self.previous_action,
        )
        return {
            "observation": observation,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "base_position": base_position,
            "base_orientation": base_orientation,
            "body_linear_velocity": body_linear_velocity,
            "imu_observation": imu_observation,
        }

    def _reset_robot(self) -> None:
        reset_noise = float(self.config["reset_joint_noise_rad"])
        noise = self.np_random.uniform(
            -reset_noise,
            reset_noise,
            size=12,
        ).astype(np.float32)
        initial_positions = np.clip(
            self.nominal_positions + noise,
            self.lower_limits + 1e-3,
            self.upper_limits - 1e-3,
        )
        self.robot.set_world_poses(
            positions=[[0.0, 0.0, float(self.config["reset_start_z_m"])]],
            orientations=[[1.0, 0.0, 0.0, 0.0]],
        )
        self.robot.set_velocities(
            linear_velocities=[[0.0, 0.0, 0.0]],
            angular_velocities=[[0.0, 0.0, 0.0]],
        )
        self.robot.set_dof_positions(initial_positions)
        self.robot.set_dof_velocities(np.zeros(12, dtype=np.float32))
        self.robot.set_dof_position_targets(self.nominal_positions)
        self.previous_target = self.nominal_positions.copy()
        self.previous_action.fill(0.0)
        for _ in range(self.reset_settle_steps):
            self.robot.set_dof_position_targets(self.nominal_positions)
            self._update()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        del options
        super().reset(seed=seed)
        # Recreate the C++ sensor so filter/history state cannot leak across
        # teleported Gym episodes.
        self.imu_sensor.reset()
        self._reset_robot()
        self.episode_step = 0
        self.episode_return = 0.0
        self.minimum_height_m = float("inf")
        self.maximum_tilt_deg = 0.0
        state = self._read_state()
        self.episode_origin = np.asarray(state["base_position"]).copy()
        return np.asarray(state["observation"]).copy(), {
            "dof_names": tuple(self.dof_names),
            "observation_fields": POLICY_OBSERVATION_FIELDS,
            "reset_base_position_m": self.episode_origin.copy(),
        }

    def step(
        self,
        action,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        clipped_action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(12),
            -1.0,
            1.0,
        )
        desired_target = np.clip(
            self.nominal_positions + self.action_scale * clipped_action,
            self.lower_limits + 1e-3,
            self.upper_limits - 1e-3,
        )
        maximum_delta = self.max_velocities * self.control_dt_s
        target = np.clip(
            desired_target,
            self.previous_target - maximum_delta,
            self.previous_target + maximum_delta,
        )
        self.robot.set_dof_position_targets(target.astype(np.float32))
        self._update()
        state = self._read_state()
        base_position = np.asarray(state["base_position"])
        imu_observation = np.asarray(state["imu_observation"])
        projected_gravity = imu_observation[3:6]
        reasons = termination_reasons(
            base_height_m=float(base_position[2]),
            projected_gravity_xyz=projected_gravity,
            minimum_base_height_m=float(
                self.termination_config["minimum_base_height_m"]
            ),
            minimum_upright_cosine=float(
                self.termination_config["minimum_upright_cosine"]
            ),
        )
        terminated = bool(reasons)
        self.episode_step += 1
        truncated = self.episode_step >= self.max_episode_steps
        normalized_joint_velocity = (
            np.asarray(state["joint_velocities"]) / self.max_velocities
        )
        reward_terms = walking_reward_terms(
            command_velocity_xyz=self.command_velocity,
            body_linear_velocity_xyz=state["body_linear_velocity"],
            body_angular_velocity_xyz=imu_observation[:3],
            projected_gravity_xyz=projected_gravity,
            base_height_m=float(base_position[2]),
            joint_velocities_normalized=normalized_joint_velocity,
            action=clipped_action,
            previous_action=self.previous_action,
            terminated=terminated,
            reward_config=self.reward_config,
        )
        reward = float(reward_terms["total"])
        self.episode_return += reward
        self.minimum_height_m = min(self.minimum_height_m, float(base_position[2]))
        upright_cosine = float(np.clip(-projected_gravity[2], -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(upright_cosine)))
        self.maximum_tilt_deg = max(self.maximum_tilt_deg, tilt_deg)
        displacement = base_position - self.episode_origin
        info: dict[str, object] = {
            "reward_terms": reward_terms,
            "base_position_m": base_position.copy(),
            "body_linear_velocity_m_s": np.asarray(
                state["body_linear_velocity"]
            ).copy(),
            "termination_reasons": reasons,
            "target_joint_positions_rad": target.copy(),
        }
        if terminated or truncated:
            episode_metrics = {
                "return": self.episode_return,
                "length_steps": self.episode_step,
                "duration_s": self.episode_step / self.control_hz,
                "forward_displacement_m": float(displacement[0]),
                "lateral_displacement_m": float(displacement[1]),
                "minimum_base_height_m": self.minimum_height_m,
                "maximum_body_tilt_deg": self.maximum_tilt_deg,
                "terminated": terminated,
                "truncated": truncated,
                "termination_reasons": reasons,
            }
            info["episode_metrics"] = episode_metrics
            self.completed_episode_metrics.append(episode_metrics)
            self.completed_episode_metrics = self.completed_episode_metrics[-20:]
        self.previous_action = clipped_action.copy()
        self.previous_target = target.copy()
        return (
            np.asarray(state["observation"]).copy(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        self._closed = True

    @property
    def contract(self) -> dict[str, object]:
        """Return the runtime contract stored beside trained checkpoints."""

        return {
            "world": self.world_path,
            "dof_names": list(self.dof_names),
            "nominal_joint_positions_rad": self.nominal_positions.tolist(),
            "action_scale_rad": self.action_scale.tolist(),
            "observation_fields": list(POLICY_OBSERVATION_FIELDS),
            "observation_size": POLICY_OBSERVATION_SIZE,
            "action_size": 12,
            "imu_fields": list(IMU_OBSERVATION_FIELDS),
            "rated_effort_cap_nm": RATED_TORQUE_NM,
            "applied_effort_cap_nm": self.effort_cap_nm,
            "applied_joint_lower_limits_rad": self.lower_limits.tolist(),
            "applied_joint_upper_limits_rad": self.upper_limits.tolist(),
            "robot_hardware_profile": (
                self.robot_hardware_profile or None
            ),
            "control_hz": self.control_hz,
            "physics_hz": self.physics_hz,
        }
