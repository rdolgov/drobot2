"""Gymnasium environment for the separate Drobot stair-climbing policy.

Import this module only after constructing ``isaacsim.SimulationApp``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
from _quadruped_rl_env import QuadrupedWalkEnv
from _quadruped_runtime import (
    LEGS,
    LINK_LENGTH_M,
    pose_by_name,
    targets_for_order,
)
from _rl_contract import POLICY_OBSERVATION_CLIP
from _stair_rl_contract import (
    PLACEMENT_REFERENCE_OBSERVATION_FIELDS,
    balance_target_error_xy,
    bounded_support_incenter_target_xy,
    curriculum_active_steps,
    equalized_foot_load_vertical_corrections,
    foot_tread_progress,
    goal_x_for_active_steps,
    inter_leg_pre_unload_gate_failures,
    inter_leg_transfer_state,
    joint_effort_telemetry_sample,
    next_foot_target_index,
    pack_placement_reference_observation,
    pack_stair_policy_observation,
    pack_support_regulation_observation,
    placement_advance_clearance_gate_state,
    placement_completion_settle_gate_failures,
    placement_contact_reached,
    placement_curriculum_level,
    placement_lift_hold_reached,
    placement_reference_state,
    placement_success_mode,
    split_post_clearance_advance_fractions,
    stabilized_support_reference_base_delta,
    staged_support_rear_pitch_scale,
    staged_swing_outward_offset_m,
    staged_swing_reference_base_delta,
    stair_failure_reasons,
    stair_goal_reached,
    stair_height_at_x,
    stair_index_at_x,
    stair_observation_fields,
    stair_reward_terms,
    support_load_share_vertical_corrections,
    support_margin_constrained_target_xy,
    support_pitch_vertical_corrections,
    touchdown_load_lift_correction_m,
    validate_staircase_config,
)
from _vl53l5cx_contract import (
    VL53L5CX_MODE,
    validate_vl53l5cx_config,
    vl53l5cx_observation_fields,
)
from _vl53l5cx_sensor import VL53L5CXRaycastSensor
from gymnasium import spaces
from isaacsim.core.experimental.prims import RigidPrim
from pxr import PhysxSchema, UsdPhysics, UsdShade
from stable_baselines3 import PPO

STAIRS_EXPECTED_DOF_ORDER = (
    "front_left_hip_abduction",
    "rear_left_hip_abduction",
    "front_right_hip_abduction",
    "rear_right_hip_abduction",
    "front_left_hip_flexion",
    "rear_left_hip_flexion",
    "front_right_hip_flexion",
    "rear_right_hip_flexion",
    "front_left_knee",
    "rear_left_knee",
    "front_right_knee",
    "rear_right_knee",
)
FOOT_CONTACT_RADIUS_M = 0.0125
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaw_from_wxyz(orientation_wxyz) -> float:
    w, x, y, z = (
        float(value)
        for value in np.asarray(orientation_wxyz, dtype=np.float64).reshape(4)
    )
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    quaternion_vector = np.asarray([x, y, z], dtype=np.float64)
    return (
        vector
        + 2.0 * w * np.cross(quaternion_vector, vector)
        + 2.0
        * np.cross(
            quaternion_vector,
            np.cross(quaternion_vector, vector),
        )
    )


def _support_triangle_signed_margin_m(
    point_xy_m: np.ndarray,
    support_points_xy_m: np.ndarray,
) -> float:
    """Return positive edge clearance inside the three stance feet."""

    point = np.asarray(point_xy_m, dtype=np.float64).reshape(2)
    vertices = np.asarray(support_points_xy_m, dtype=np.float64).reshape(3, 2)
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    ordered = vertices[np.argsort(angles)]
    margins: list[float] = []
    for index in range(3):
        start = ordered[index]
        edge = ordered[(index + 1) % 3] - start
        edge_length = float(np.linalg.norm(edge))
        if edge_length <= 1e-9:
            raise RuntimeError("Support-foot contact points are not distinct")
        relative = point - start
        margins.append(
            float((edge[0] * relative[1] - edge[1] * relative[0]) / edge_length)
        )
    return min(margins)


class QuadrupedStairsEnv(QuadrupedWalkEnv):
    """One floating quadruped learning a curriculum over a fixed staircase."""

    def _before_physics_play(self) -> None:
        if not self.placement_reference_enabled:
            return
        link_path_by_name = dict(
            zip(self.robot.link_names, self.robot.link_paths[0], strict=True)
        )
        self.robot_link_prim = RigidPrim(list(self.robot.link_paths[0]))
        self.robot_link_masses_kg: np.ndarray | None = None
        self.robot_link_com_offsets_m: np.ndarray | None = None
        foot_paths = [
            link_path_by_name[f"{leg}_distal_link"] for leg in LEGS
        ]
        self._apply_foot_contact_material(foot_paths)
        self.contact_filter_paths = (
            "/World/Ground",
            *(
                f"/World/Stairs/StepLayer_{index + 1:02d}"
                for index in range(int(self.staircase_config["step_count"]))
            ),
        )
        self.foot_prim = RigidPrim(
            foot_paths,
            contact_filter_paths=list(self.contact_filter_paths),
            max_contact_count=128,
        )
        self.foot_prim.set_enabled_contact_tracking([True], threshold=0.0)

    def _apply_foot_contact_material(self, foot_paths: list[str]) -> None:
        config = self.foot_contact_material_config
        if not bool(config.get("enabled", False)):
            self.foot_contact_material_applied = None
            return
        static_friction = float(config["static_friction"])
        dynamic_friction = float(config["dynamic_friction"])
        restitution = float(config.get("restitution", 0.02))
        friction_combine_mode = str(
            config.get("friction_combine_mode", "average")
        )
        if (
            static_friction <= 0.0
            or dynamic_friction <= 0.0
            or dynamic_friction > static_friction
            or restitution < 0.0
            or restitution > 1.0
        ):
            raise ValueError("Invalid foot_contact_material coefficients")
        if friction_combine_mode not in {"average", "min", "multiply", "max"}:
            raise ValueError("Invalid foot_contact_material friction combine mode")
        material_path = "/World/Materials/RubberFootContact"
        material = UsdShade.Material.Define(self.stage, material_path)
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateStaticFrictionAttr().Set(static_friction)
        material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
        material_api.CreateRestitutionAttr().Set(restitution)
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_material.CreateFrictionCombineModeAttr().Set(
            friction_combine_mode
        )
        physx_material.CreateCompliantContactStiffnessAttr().Set(
            float(config.get("compliant_contact_stiffness", 12000.0))
        )
        physx_material.CreateCompliantContactDampingAttr().Set(
            float(config.get("compliant_contact_damping", 45.0))
        )
        bound_paths: list[str] = []
        for foot_path in foot_paths:
            proxy_path = (
                f"{foot_path}/simulation_only_fork_tip_contact_proxy_1"
            )
            prim = self.stage.GetPrimAtPath(proxy_path)
            if not prim.IsValid():
                raise RuntimeError(
                    f"Foot contact proxy is missing for traction test: {proxy_path}"
                )
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                UsdShade.Tokens.strongerThanDescendants,
                "physics",
            )
            bound_paths.append(proxy_path)
        self.foot_contact_material_applied = {
            **config,
            "friction_combine_mode": friction_combine_mode,
            "material_path": material_path,
            "bound_paths": bound_paths,
        }

    def set_foot_contact_friction(
        self,
        *,
        static_friction: float,
        dynamic_friction: float,
    ) -> None:
        """Update the active simulation-only foot material for sensitivity tests."""

        if self.foot_contact_material_applied is None:
            raise RuntimeError("Foot contact material is not enabled")
        static = float(static_friction)
        dynamic = float(dynamic_friction)
        if static <= 0.0 or dynamic <= 0.0 or dynamic > static:
            raise ValueError("Invalid foot contact friction coefficients")
        material_path = str(self.foot_contact_material_applied["material_path"])
        material_prim = self.stage.GetPrimAtPath(material_path)
        if not material_prim.IsValid():
            raise RuntimeError(f"Foot contact material is missing: {material_path}")
        material_api = UsdPhysics.MaterialAPI(material_prim)
        material_api.GetStaticFrictionAttr().Set(static)
        material_api.GetDynamicFrictionAttr().Set(dynamic)
        self.foot_contact_material_config["static_friction"] = static
        self.foot_contact_material_config["dynamic_friction"] = dynamic
        self.foot_contact_material_applied["static_friction"] = static
        self.foot_contact_material_applied["dynamic_friction"] = dynamic

    def __init__(
        self,
        simulation_app,
        *,
        world_path: str,
        task_config: Mapping[str, object],
        render_mode: str | None = None,
    ) -> None:
        self.staircase_config = dict(task_config["staircase"])
        validate_staircase_config(self.staircase_config)
        self.curriculum_config = dict(task_config["curriculum"])
        self.curriculum_levels = tuple(self.curriculum_config["levels"])
        self.active_step_count = curriculum_active_steps(
            0.0,
            self.curriculum_levels,
            maximum_steps=int(self.staircase_config["step_count"]),
        )
        self.pending_active_step_count = self.active_step_count
        self.current_goal_x_m = goal_x_for_active_steps(
            self.staircase_config,
            self.active_step_count,
        )
        self.curriculum_progress = 0.0
        self.curriculum_transitions: list[dict[str, object]] = []
        self.placement_reference_config = dict(
            task_config.get("placement_reference", {})
        )
        self.foot_contact_material_config = dict(
            task_config.get("foot_contact_material", {})
        )
        self.foot_contact_material_applied: dict[str, object] | None = None
        self.placement_reference_enabled = bool(
            self.placement_reference_config.get("enabled", False)
        )
        self.placement_curriculum_config = dict(
            task_config.get("placement_curriculum", {})
        )
        self.placement_curriculum_levels = tuple(
            dict(level)
            for level in self.placement_curriculum_config.get("levels", ())
        )
        self.current_placement_level: dict[str, object] | None = None
        self.current_placement_level_id: str | None = None
        self.pending_placement_level: dict[str, object] | None = None
        self.pending_placement_level_id: str | None = None
        if self.placement_reference_enabled:
            self.current_placement_level = placement_curriculum_level(
                0.0,
                self.placement_curriculum_levels,
            )
            self.current_placement_level_id = str(
                self.current_placement_level["id"]
            )
            self.pending_placement_level = dict(self.current_placement_level)
            self.pending_placement_level_id = self.current_placement_level_id
        super().__init__(
            simulation_app,
            world_path=world_path,
            task_config=task_config,
            render_mode=render_mode,
        )
        self.physics_steps_per_control = self.physics_hz // self.control_hz
        if self.physics_steps_per_control < 1:
            raise ValueError("control_hz cannot exceed physics_hz")
        if tuple(self.dof_names) != STAIRS_EXPECTED_DOF_ORDER:
            raise RuntimeError(
                "Stair policy requires the reviewed DOF order: "
                f"{self.dof_names} != {list(STAIRS_EXPECTED_DOF_ORDER)}"
            )
        drive_stiffness, drive_damping = self.robot.get_dof_gains()
        self.drive_stiffness_nm_rad = np.asarray(
            drive_stiffness.numpy()
            if hasattr(drive_stiffness, "numpy")
            else drive_stiffness,
            dtype=np.float64,
        ).reshape(-1)
        self.drive_damping_nm_s_rad = np.asarray(
            drive_damping.numpy()
            if hasattr(drive_damping, "numpy")
            else drive_damping,
            dtype=np.float64,
        ).reshape(-1)
        if (
            self.drive_stiffness_nm_rad.shape != (12,)
            or self.drive_damping_nm_s_rad.shape != (12,)
            or not np.all(np.isfinite(self.drive_stiffness_nm_rad))
            or not np.all(np.isfinite(self.drive_damping_nm_s_rad))
        ):
            raise RuntimeError("Could not read finite 12-DOF implicit drive gains")
        self._validate_stair_prims()
        self.residual_policy_config = dict(
            self.config.get("residual_policy", {})
        )
        self.residual_policy_enabled = bool(
            self.residual_policy_config.get("enabled", False)
        )
        if self.placement_reference_enabled and self.residual_policy_enabled:
            raise ValueError(
                "placement_reference and residual flat-walking policy cannot both be enabled"
            )
        self.base_policy = None
        self.base_policy_path: Path | None = None
        self.base_policy_sha256: str | None = None
        self.base_action_scale = np.zeros(12, dtype=np.float32)
        self.latest_walking_observation = np.zeros(48, dtype=np.float32)
        self.latest_projected_gravity_xyz = np.asarray(
            (0.0, 0.0, -1.0),
            dtype=np.float64,
        )
        self.previous_residual_action = np.zeros(12, dtype=np.float32)
        if self.residual_policy_enabled:
            configured_path = Path(
                str(self.residual_policy_config["base_model"])
            )
            self.base_policy_path = (
                configured_path.resolve()
                if configured_path.is_absolute()
                else (PROJECT_ROOT / configured_path).resolve()
            )
            if not self.base_policy_path.is_file():
                raise FileNotFoundError(self.base_policy_path)
            self.base_policy_sha256 = _sha256_file(self.base_policy_path)
            self.base_policy = PPO.load(
                str(self.base_policy_path),
                device="cpu",
            )
            if (
                tuple(self.base_policy.observation_space.shape) != (48,)
                or tuple(self.base_policy.action_space.shape) != (12,)
            ):
                raise RuntimeError(
                    "Residual base policy must have observation/action "
                    "shapes (48,)/(12,)"
                )
            base_scale_by_kind = dict(
                self.residual_policy_config["base_action_scale_rad"]
            )
            self.base_action_scale = np.asarray(
                [
                    float(
                        base_scale_by_kind[
                            next(
                                kind
                                for kind in base_scale_by_kind
                                if name.endswith(kind)
                            )
                        ]
                    )
                    for name in self.dof_names
                ],
                dtype=np.float32,
            )
        if self.placement_reference_enabled:
            if not self.foot_prim.is_physics_tensor_entity_valid():
                raise RuntimeError("Placement foot-contact tensor view is invalid")
        else:
            link_path_by_name = dict(
                zip(self.robot.link_names, self.robot.link_paths[0], strict=True)
            )
            foot_paths = [
                link_path_by_name[f"{leg}_distal_link"] for leg in LEGS
            ]
            self.contact_filter_paths = ()
            self.foot_prim = RigidPrim(foot_paths)
        configured_sequence = tuple(
            str(value)
            for value in self.config.get("foot_placement_sequence", LEGS)
        )
        if sorted(configured_sequence) != sorted(LEGS):
            raise ValueError("foot_placement_sequence must contain every leg once")
        self.foot_placement_sequence = configured_sequence
        self.foot_placement_sequence_indices = tuple(
            LEGS.index(name) for name in configured_sequence
        )

        self.terrain_perception_config = dict(
            self.config.get(
                "terrain_perception",
                {"mode": "analytic_height_profile"},
            )
        )
        self.terrain_perception_mode = str(
            self.terrain_perception_config.get(
                "mode",
                "analytic_height_profile",
            )
        )
        self.vl53l5cx_sensor: VL53L5CXRaycastSensor | None = None
        terrain_field_override = None
        if self.terrain_perception_mode == VL53L5CX_MODE:
            validate_vl53l5cx_config(
                self.terrain_perception_config,
                control_hz=self.control_hz,
            )
            self.vl53l5cx_sensor = VL53L5CXRaycastSensor(
                self.terrain_perception_config,
                control_hz=self.control_hz,
            )
            terrain_field_override = vl53l5cx_observation_fields(
                self.terrain_perception_config
            )
        elif self.terrain_perception_mode != "analytic_height_profile":
            raise ValueError(
                "Unsupported terrain perception mode: "
                f"{self.terrain_perception_mode}"
            )

        offsets = tuple(
            float(value)
            for value in self.staircase_config["terrain_sample_offsets_m"]
        )
        self.include_navigation_observation = bool(
            self.config.get("include_navigation_observation", False)
        )
        self.include_foot_progress_observation = bool(
            self.config.get("include_foot_progress_observation", False)
        )
        self.include_placement_reference_observation = bool(
            self.placement_reference_enabled
            and self.config.get("include_placement_reference_observation", True)
        )
        self.include_support_regulation_observation = bool(
            self.include_placement_reference_observation
            and self.config.get("include_support_regulation_observation", False)
        )
        self.observation_fields = stair_observation_fields(
            offsets,
            include_navigation_observation=self.include_navigation_observation,
            include_foot_progress_observation=(
                self.include_foot_progress_observation
            ),
            include_placement_reference_observation=(
                self.include_placement_reference_observation
            ),
            include_support_regulation_observation=(
                self.include_support_regulation_observation
            ),
            terrain_observation_fields=terrain_field_override,
        )
        self.observation_size = len(self.observation_fields)
        self.observation_space = spaces.Box(
            low=-POLICY_OBSERVATION_CLIP,
            high=POLICY_OBSERVATION_CLIP,
            shape=(self.observation_size,),
            dtype=np.float32,
        )
        self.success_hold_steps = int(
            round(float(self.config["success_hold_seconds"]) * self.control_hz)
        )
        if self.success_hold_steps < 1:
            raise ValueError("success_hold_seconds is shorter than one control step")
        self.previous_base_x_m = 0.0
        self.previous_base_z_m = 0.0
        self.previous_terrain_height_m = 0.0
        self.maximum_base_elevation_gain_m = 0.0
        self.maximum_terrain_height_m = 0.0
        self.minimum_base_clearance_m = float("inf")
        self.highest_step_reached = 0
        self.goal_hold_step_count = 0
        self.initial_foot_bottom_z_m = np.zeros(len(LEGS), dtype=np.float32)
        self.maximum_foot_lift_m = np.zeros(len(LEGS), dtype=np.float32)
        self.highest_foot_step = np.zeros(len(LEGS), dtype=np.int32)
        self.maximum_foot_tread_progress = np.zeros(
            len(LEGS),
            dtype=np.float32,
        )
        self.next_foot_target_index: int | None = 0
        configured_placement_sequence = tuple(
            str(leg)
            for leg in self.placement_reference_config.get(
                "sequence_legs",
                (self.placement_reference_config.get("swing_leg", LEGS[0]),),
            )
        )
        if (
            not configured_placement_sequence
            or len(set(configured_placement_sequence))
            != len(configured_placement_sequence)
            or any(leg not in LEGS for leg in configured_placement_sequence)
        ):
            raise ValueError(
                "placement_reference.sequence_legs must be unique known legs"
            )
        self.placement_sequence_legs = configured_placement_sequence
        self.advance_clearance_gate_config = dict(
            self.placement_reference_config.get(
                "advance_clearance_gate",
                {},
            )
        )
        self.advance_clearance_gate_enabled = bool(
            self.advance_clearance_gate_config.get("enabled", False)
        )
        self.advance_clearance_gate_legs = tuple(
            str(leg)
            for leg in self.advance_clearance_gate_config.get(
                "legs",
                self.placement_sequence_legs,
            )
        )
        if any(leg not in self.placement_sequence_legs for leg in (
            self.advance_clearance_gate_legs
        )):
            raise ValueError(
                "advance_clearance_gate.legs must be placement sequence legs"
            )
        self.advance_clearance_gate_minimum_m = float(
            self.advance_clearance_gate_config.get(
                "minimum_clearance_m",
                0.190,
            )
        )
        if not 0.0 < self.advance_clearance_gate_minimum_m <= 0.40:
            raise ValueError(
                "advance_clearance_gate.minimum_clearance_m must be within "
                "(0, 0.40]"
            )
        maximum_clearance_hold_seconds = float(
            self.advance_clearance_gate_config.get(
                "maximum_hold_seconds",
                2.0,
            )
        )
        if not 0.0 < maximum_clearance_hold_seconds <= 5.0:
            raise ValueError(
                "advance_clearance_gate.maximum_hold_seconds must be within "
                "(0, 5]"
            )
        self.advance_clearance_gate_maximum_hold_steps = max(
            1,
            int(round(maximum_clearance_hold_seconds * self.control_hz)),
        )
        self.placement_clearance_gate_hold_step_count = 0
        self.placement_clearance_gate_released = False
        self.placement_clearance_gate_timeout = False
        self.placement_early_contact_hold_elapsed_s: float | None = None
        self.maximum_placement_clearance_gate_hold_steps = 0
        self.placement_sequence_position = 0
        self.placement_swing_leg = self.placement_sequence_legs[0]
        self.placement_swing_leg_index = 0
        self.placement_support_leg_indices: tuple[int, ...] = ()
        self._set_placement_swing_leg(self.placement_swing_leg)
        self.placement_phase_start_step = 0
        self.placement_phase_elapsed_offset_s = 0.0
        self.completed_placement_legs: list[str] = []
        self.completed_placement_joint_targets_by_leg: dict[str, np.ndarray] = {}
        self.completed_placement_reference_by_leg: dict[
            str,
            dict[str, object],
        ] = {}
        self.latest_reference_parameters_by_leg: dict[
            str,
            dict[str, float],
        ] = {}
        self.latest_placement_base_position_m = np.zeros(3, dtype=np.float64)
        self.latest_placement_com_position_m = np.zeros(3, dtype=np.float64)
        self.inter_leg_transfer_config = dict(
            self.placement_reference_config.get("inter_leg_transfer", {})
        )
        self.com_regulation_config = dict(
            self.inter_leg_transfer_config.get("com_regulation", {})
        )
        self.support_load_sharing_config = dict(
            self.com_regulation_config.get("load_sharing", {})
        )
        self.support_load_sharing_enabled = bool(
            self.support_load_sharing_config.get("enabled", False)
        )
        self.four_foot_preload_load_sharing_config = dict(
            self.com_regulation_config.get(
                "four_foot_preload_load_sharing",
                {},
            )
        )
        self.four_foot_preload_load_sharing_enabled = bool(
            self.four_foot_preload_load_sharing_config.get("enabled", False)
        )
        if self.four_foot_preload_load_sharing_enabled:
            preload_legs = tuple(
                str(value)
                for value in self.four_foot_preload_load_sharing_config.get(
                    "next_swing_legs",
                    LEGS,
                )
            )
            if any(leg not in LEGS for leg in preload_legs):
                raise ValueError(
                    "four_foot_preload_load_sharing.next_swing_legs contains "
                    "an unknown leg"
                )
            for name, default, maximum in (
                ("proportional_gain_m", 0.030, 0.20),
                ("maximum_correction_m", 0.012, 0.05),
                ("smoothing_factor", 0.50, 1.0),
            ):
                value = float(
                    self.four_foot_preload_load_sharing_config.get(name, default)
                )
                if not 0.0 < value <= maximum:
                    raise ValueError(
                        "four_foot_preload_load_sharing."
                        f"{name} must be within (0, {maximum}]"
                    )
        self.support_margin_regulation_config = dict(
            self.com_regulation_config.get(
                "support_margin_regulation",
                {},
            )
        )
        self.support_margin_regulation_enabled = bool(
            self.support_margin_regulation_config.get("enabled", False)
        )
        support_margin_phases = tuple(
            str(value)
            for value in self.support_margin_regulation_config.get(
                "phases",
                ("advance", "lower", "hold"),
            )
        )
        if any(
            phase not in {"weight_shift", "lift", "advance", "lower", "hold"}
            for phase in support_margin_phases
        ):
            raise ValueError(
                "com_regulation.support_margin_regulation.phases contains "
                "an unknown placement phase"
            )
        self.support_margin_regulation_phases = support_margin_phases
        support_target_margin_m = float(
            self.support_margin_regulation_config.get(
                "minimum_target_margin_m",
                0.020,
            )
        )
        if not 0.0 <= support_target_margin_m <= 0.10:
            raise ValueError(
                "com_regulation.support_margin_regulation."
                "minimum_target_margin_m must be within [0, 0.10]"
            )
        self.touchdown_load_regulation_config = dict(
            self.placement_reference_config.get(
                "touchdown_load_regulation",
                {},
            )
        )
        self.touchdown_load_regulation_enabled = bool(
            self.touchdown_load_regulation_config.get("enabled", False)
        )
        touchdown_legs = tuple(
            str(value)
            for value in self.touchdown_load_regulation_config.get(
                "legs",
                (),
            )
        )
        if any(leg not in LEGS for leg in touchdown_legs):
            raise ValueError(
                "touchdown_load_regulation.legs contains an unknown leg"
            )
        self.touchdown_load_regulation_legs = touchdown_legs
        touchdown_phases = tuple(
            str(value)
            for value in self.touchdown_load_regulation_config.get(
                "phases",
                ("advance", "lower", "hold"),
            )
        )
        if any(
            phase not in {"weight_shift", "lift", "advance", "lower", "hold"}
            for phase in touchdown_phases
        ):
            raise ValueError(
                "touchdown_load_regulation.phases contains an unknown phase"
            )
        self.touchdown_load_regulation_phases = touchdown_phases
        for name, default in (
            ("attack_smoothing_factor", 0.50),
            ("release_smoothing_factor", 0.10),
        ):
            factor = float(
                self.touchdown_load_regulation_config.get(name, default)
            )
            if not 0.0 < factor <= 1.0:
                raise ValueError(
                    f"touchdown_load_regulation.{name} must be within (0, 1]"
                )
        self.com_regulation_enabled = bool(
            self.com_regulation_config.get("enabled", False)
        )
        self.pitch_feedback_config = dict(
            self.com_regulation_config.get("pitch_attitude_feedback", {})
        )
        self.pitch_feedback_enabled = bool(
            self.pitch_feedback_config.get("enabled", False)
        )
        if self.pitch_feedback_enabled:
            pitch_gain_m = float(
                self.pitch_feedback_config.get("proportional_gain_m", 0.12)
            )
            pitch_maximum_m = float(
                self.pitch_feedback_config.get("maximum_correction_m", 0.035)
            )
            if not 0.0 < pitch_gain_m <= 0.50:
                raise ValueError(
                    "com_regulation.pitch_attitude_feedback."
                    "proportional_gain_m must be within (0, 0.50]"
                )
            if not 0.0 < pitch_maximum_m <= 0.08:
                raise ValueError(
                    "com_regulation.pitch_attitude_feedback."
                    "maximum_correction_m must be within (0, 0.08]"
                )
        balance_point = str(
            self.com_regulation_config.get("balance_point", "composite_com")
        )
        if balance_point not in {"composite_com", "base_origin"}:
            raise ValueError(
                "inter_leg_transfer.com_regulation.balance_point must be "
                "composite_com or base_origin"
            )
        self.com_regulation_balance_point = balance_point
        incenter_blend = float(
            self.com_regulation_config.get("target_incenter_blend", 1.0)
        )
        if incenter_blend <= 0.0 or incenter_blend > 1.0:
            raise ValueError(
                "inter_leg_transfer.com_regulation.target_incenter_blend "
                "must be within (0, 1]"
            )
        maximum_correction = dict(
            self.com_regulation_config.get("maximum_correction_m", {})
        )
        maximum_correction.update(
            dict(
                dict(
                    self.com_regulation_config.get(
                        "maximum_correction_m_by_swing_leg",
                        {},
                    )
                ).get(self.placement_swing_leg, {})
            )
        )
        if any(
            float(maximum_correction.get(axis, 0.12)) <= 0.0
            for axis in ("forward", "lateral")
        ):
            raise ValueError(
                "inter_leg_transfer.com_regulation.maximum_correction_m "
                "values must be positive"
            )
        maximum_feedback = dict(
            self.com_regulation_config.get(
                "maximum_feedback_correction_m",
                {},
            )
        )
        if any(
            float(maximum_feedback.get(axis, 0.025)) <= 0.0
            for axis in ("forward", "lateral", "vertical")
        ):
            raise ValueError(
                "inter_leg_transfer.com_regulation."
                "maximum_feedback_correction_m values must be positive"
            )
        self.inter_leg_transfer_enabled = bool(
            self.inter_leg_transfer_config.get("enabled", False)
            and len(self.placement_sequence_legs) > 1
        )
        pre_unload_gate_hold_seconds = float(
            self.inter_leg_transfer_config.get(
                "pre_unload_gate_hold_seconds",
                0.0,
            )
        )
        if not 0.0 <= pre_unload_gate_hold_seconds <= 5.0:
            raise ValueError(
                "inter_leg_transfer.pre_unload_gate_hold_seconds must be "
                "within [0, 5]"
            )
        self.pre_unload_gate_hold_steps = int(
            round(pre_unload_gate_hold_seconds * self.control_hz)
        )
        self.minimum_next_swing_preload_n = float(
            self.inter_leg_transfer_config.get(
                "minimum_next_swing_preload_n",
                self.placement_reference_config.get(
                    "contact_on_threshold_n",
                    1.0,
                ),
            )
        )
        if self.minimum_next_swing_preload_n < 0.0:
            raise ValueError(
                "inter_leg_transfer.minimum_next_swing_preload_n cannot be "
                "negative"
            )
        self.phase_snapshot_restore_settle_control_steps = int(
            self.inter_leg_transfer_config.get(
                "phase_snapshot_restore_settle_control_steps",
                0,
            )
        )
        if not 0 <= self.phase_snapshot_restore_settle_control_steps <= 120:
            raise ValueError(
                "inter_leg_transfer.phase_snapshot_restore_settle_control_steps "
                "must be within [0, 120]"
            )
        self.phase_snapshot_restore_zero_velocities = bool(
            self.inter_leg_transfer_config.get(
                "phase_snapshot_restore_zero_velocities",
                False,
            )
        )
        support_anchor_follow_gain = float(
            self.inter_leg_transfer_config.get(
                "support_world_anchor_follow_gain",
                0.25,
            )
        )
        if support_anchor_follow_gain < 0.0 or support_anchor_follow_gain > 1.0:
            raise ValueError(
                "inter_leg_transfer.support_world_anchor_follow_gain must be "
                "within [0, 1]"
            )
        support_error_feedback = dict(
            self.inter_leg_transfer_config.get(
                "support_base_error_feedback_gain",
                {},
            )
        )
        for label, gains, default in (
            ("support_base_error_feedback_gain", support_error_feedback, 0.0),
            (
                "com_regulation.feedback_gain",
                dict(self.com_regulation_config.get("feedback_gain", {})),
                1.0,
            ),
            (
                "com_regulation.transfer_feedback_gain",
                dict(
                    self.com_regulation_config.get(
                        "transfer_feedback_gain",
                        self.com_regulation_config.get("feedback_gain", {}),
                    )
                ),
                1.0,
            ),
        ):
            for axis in ("forward", "lateral", "vertical"):
                gain = float(gains.get(axis, default))
                if gain < 0.0 or gain > 2.0:
                    raise ValueError(
                        f"inter_leg_transfer.{label} {axis} must be within "
                        "[0, 2]"
                    )
        swing_outward_offsets = dict(
            self.inter_leg_transfer_config.get(
                "swing_outward_offset_m_by_leg",
                {},
            )
        )
        for leg, offset_m in swing_outward_offsets.items():
            if leg not in LEGS:
                raise ValueError(
                    "inter_leg_transfer.swing_outward_offset_m_by_leg "
                    f"contains unknown leg {leg!r}"
                )
            if not np.isfinite(float(offset_m)) or abs(float(offset_m)) > 0.15:
                raise ValueError(
                    "inter_leg_transfer.swing_outward_offset_m_by_leg "
                    "values must be finite and within [-0.15, 0.15] m"
                )
        self.placement_transfer_active = False
        self.placement_transfer_start_step = 0
        self.placement_transfer_gate_step_count = 0
        self.placement_transfer_pre_unload_gate_step_count = 0
        self.placement_transfer_unload_start_step: int | None = None
        self.placement_transfer_reference_by_leg: dict[
            str,
            dict[str, float],
        ] = {}
        self.placement_leg_baseline_reference_by_leg: dict[
            str,
            dict[str, float],
        ] = {}
        self.placement_leg_baseline_base_position_m = np.zeros(
            3,
            dtype=np.float64,
        )
        self.placement_leg_baseline_balance_position_m = np.zeros(
            3,
            dtype=np.float64,
        )
        self.placement_leg_baseline_lift_offset_m = 0.0
        self.placement_transfer_start_base_position_m = np.zeros(
            3,
            dtype=np.float64,
        )
        self.placement_transfer_target_base_position_m = np.zeros(
            3,
            dtype=np.float64,
        )
        self.placement_transfer_start_balance_position_m = np.zeros(
            3,
            dtype=np.float64,
        )
        self.placement_transfer_target_balance_position_m = np.zeros(
            3,
            dtype=np.float64,
        )
        self.completed_inter_leg_transfers: list[str] = []
        self.last_completed_inter_leg_transfer_metrics: dict[str, object] = {}
        self.dof_indices_by_leg = {
            leg: tuple(
                index
                for index, name in enumerate(self.dof_names)
                if name.startswith(f"{leg}_")
            )
            for leg in LEGS
        }
        if self.placement_reference_enabled:
            self._apply_placement_residual_action_scale()
        self.initial_placement_foot_tips_m = np.zeros((len(LEGS), 3), dtype=np.float32)
        self.placement_leg_start_foot_tips_m = np.zeros(
            (len(LEGS), 3),
            dtype=np.float32,
        )
        self.latest_ground_normal_loads_n = np.zeros(len(LEGS), dtype=np.float32)
        self.latest_step_normal_loads_n = np.zeros(
            (len(LEGS), int(self.staircase_config["step_count"])),
            dtype=np.float32,
        )
        self.latest_foot_tips_m = np.zeros((len(LEGS), 3), dtype=np.float32)
        self.latest_support_load_sharing_correction_m = np.zeros(
            len(LEGS),
            dtype=np.float64,
        )
        self.maximum_abs_support_load_sharing_correction_m = 0.0
        self.maximum_abs_support_load_sharing_correction_m_by_leg = np.zeros(
            len(LEGS),
            dtype=np.float64,
        )
        self.support_load_sharing_active_sample_count = 0
        self.support_load_sharing_saturated_sample_count = 0
        self.maximum_support_slip_m = 0.0
        self.maximum_support_slip_m_by_leg = np.zeros(
            len(LEGS),
            dtype=np.float32,
        )
        self.minimum_support_contact_fraction = 1.0
        self.minimum_placement_support_margin_m = float("inf")
        self.latest_support_margin_regulation_active = False
        self.latest_support_margin_requested_target_xy_m = np.zeros(
            2,
            dtype=np.float64,
        )
        self.latest_support_margin_constrained_target_xy_m = np.zeros(
            2,
            dtype=np.float64,
        )
        self.latest_support_margin_commanded_target_margin_m = 0.0
        self.maximum_support_margin_target_clip_m = 0.0
        self.latest_touchdown_load_lift_correction_m = 0.0
        self.maximum_touchdown_load_lift_correction_m = 0.0
        self.latest_placement_pitch_rear_correction_scale = 1.0
        self.maximum_swing_tread_normal_load_n = 0.0
        self.maximum_tread_normal_load_n_by_leg = np.zeros(
            len(LEGS),
            dtype=np.float32,
        )
        self.placement_tread_contact_sample_count = 0
        self.placement_active_sample_count = 0
        self.placement_reference_reach_clip_count = 0
        self.maximum_placement_reference_reach_excess_m = 0.0
        self.maximum_placement_desired_lift_m = 0.0
        self.maximum_swing_reference_tracking_error_rad = 0.0
        self.maximum_balance_lateral_deviation_m = 0.0
        self._reset_joint_effort_telemetry()

    def _reset_joint_effort_telemetry(self) -> None:
        self.latest_requested_pd_effort_nm = np.zeros(12, dtype=np.float64)
        self.maximum_abs_joint_tracking_error_rad_by_joint = np.zeros(
            12,
            dtype=np.float64,
        )
        self.peak_abs_requested_pd_effort_nm_by_joint = np.zeros(
            12,
            dtype=np.float64,
        )
        self.peak_abs_reported_actuation_effort_nm_by_joint = np.zeros(
            12,
            dtype=np.float64,
        )
        self.peak_abs_projected_joint_reaction_load_nm_by_joint = np.zeros(
            12,
            dtype=np.float64,
        )
        self.requested_pd_effort_sample_count = 0
        self.requested_pd_effort_95pct_cap_count = 0
        self.reported_actuation_effort_sample_count = 0
        self.projected_joint_reaction_load_sample_count = 0

    def _sample_joint_efforts_nm(self) -> dict[str, np.ndarray]:
        sampled: dict[str, np.ndarray] = {}
        for label, getter in (
            (
                "reported_actuation_effort_nm",
                self.robot.get_dof_efforts,
            ),
            (
                "projected_joint_reaction_load_nm",
                self.robot.get_dof_projected_joint_forces,
            ),
        ):
            try:
                raw = getter()
                vector = np.asarray(
                    raw.numpy() if hasattr(raw, "numpy") else raw,
                    dtype=np.float64,
                ).reshape(-1)
                if vector.shape == (12,) and np.all(np.isfinite(vector)):
                    sampled[label] = vector.copy()
            except Exception:
                continue
        return sampled

    def _joint_effort_metrics(self) -> dict[str, object]:
        metrics: dict[str, object] = {
            "effort_cap_nm": self.effort_cap_nm,
            "maximum_abs_joint_tracking_error_rad": float(
                np.max(self.maximum_abs_joint_tracking_error_rad_by_joint)
            ),
            "maximum_abs_joint_tracking_error_rad_by_joint": dict(
                zip(
                    self.dof_names,
                    self.maximum_abs_joint_tracking_error_rad_by_joint.tolist(),
                    strict=True,
                )
            ),
            "drive_stiffness_nm_rad_by_joint": dict(
                zip(
                    self.dof_names,
                    self.drive_stiffness_nm_rad.tolist(),
                    strict=True,
                )
            ),
            "drive_damping_nm_s_rad_by_joint": dict(
                zip(
                    self.dof_names,
                    self.drive_damping_nm_s_rad.tolist(),
                    strict=True,
                )
            ),
        }
        sample_count = self.requested_pd_effort_sample_count
        peak_values = self.peak_abs_requested_pd_effort_nm_by_joint
        metrics["requested_pd_effort_sample_count"] = sample_count
        metrics["peak_abs_requested_pd_effort_nm"] = (
            float(np.max(peak_values)) if sample_count else None
        )
        metrics["peak_abs_requested_pd_effort_nm_by_joint"] = (
            dict(zip(self.dof_names, peak_values.tolist(), strict=True))
            if sample_count
            else None
        )
        metrics["peak_requested_pd_effort_to_cap_ratio"] = (
            float(np.max(peak_values) / self.effort_cap_nm)
            if sample_count
            else None
        )
        metrics["requested_pd_effort_95pct_cap_sample_fraction"] = (
            float(self.requested_pd_effort_95pct_cap_count / (sample_count * 12))
            if sample_count
            else None
        )
        for prefix, peak_values, sample_count in (
            (
                "reported_actuation_effort",
                self.peak_abs_reported_actuation_effort_nm_by_joint,
                self.reported_actuation_effort_sample_count,
            ),
            (
                "projected_joint_reaction_load",
                self.peak_abs_projected_joint_reaction_load_nm_by_joint,
                self.projected_joint_reaction_load_sample_count,
            ),
        ):
            metrics[f"{prefix}_sample_count"] = sample_count
            metrics[f"peak_abs_{prefix}_nm"] = (
                float(np.max(peak_values)) if sample_count else None
            )
            metrics[f"peak_abs_{prefix}_nm_by_joint"] = (
                dict(zip(self.dof_names, peak_values.tolist(), strict=True))
                if sample_count
                else None
            )
        return metrics

    def _set_placement_swing_leg(self, leg: str) -> None:
        self.placement_swing_leg = str(leg)
        self.placement_swing_leg_index = LEGS.index(self.placement_swing_leg)
        self.placement_support_leg_indices = tuple(
            index
            for index, name in enumerate(LEGS)
            if name != self.placement_swing_leg
        )
        if hasattr(self, "placement_clearance_gate_hold_step_count"):
            self.placement_clearance_gate_hold_step_count = 0
            self.placement_clearance_gate_released = False
            self.placement_clearance_gate_timeout = False
            self.placement_early_contact_hold_elapsed_s = None
        if (
            self.placement_reference_enabled
            and hasattr(self, "action_scale")
            and "placement_residual_action_scale_rad" in self.config
        ):
            self._apply_placement_residual_action_scale()

    def _apply_placement_residual_action_scale(self) -> None:
        """Apply role scales, including an optional active-leg override.

        A composed placement episode may replay an independently trained
        precursor policy before switching swing legs.  The precursor must see
        the exact action scaling from its own contract; support limits tuned
        for a later leg must therefore be selected only after that leg becomes
        active.
        """

        configured = dict(self.config["placement_residual_action_scale_rad"])
        residual_scale = {
            "swing": dict(configured["swing"]),
            "support": dict(configured["support"]),
        }
        overrides = dict(configured.get("override_by_swing_leg", {}))
        leg_override = dict(overrides.get(self.placement_swing_leg, {}))
        for role in ("swing", "support"):
            residual_scale[role].update(dict(leg_override.get(role, {})))
        for index, name in enumerate(self.dof_names):
            role = (
                "swing"
                if name.startswith(f"{self.placement_swing_leg}_")
                else "support"
            )
            kind = next(
                candidate
                for candidate in residual_scale[role]
                if name.endswith(candidate)
            )
            self.action_scale[index] = float(residual_scale[role][kind])

    def _placement_success_mode(self) -> str:
        return placement_success_mode(
            swing_leg=self.placement_swing_leg,
            default_mode=str(
                self.placement_reference_config.get(
                    "success_mode",
                    "tread_contact",
                )
            ),
            mode_by_leg=dict(
                self.placement_reference_config.get(
                    "success_mode_by_leg",
                    {},
                )
            ),
            active_level=self._active_placement_level(),
        )

    def _active_placement_level(self) -> dict[str, object]:
        if self.current_placement_level is None:
            raise RuntimeError("Placement reference has no active curriculum level")
        level = dict(self.current_placement_level)
        overrides_by_leg = dict(
            self.placement_reference_config.get("level_override_by_leg", {})
        )
        level.update(dict(overrides_by_leg.get(self.placement_swing_leg, {})))
        return level

    def _validate_stair_prims(self) -> None:
        expected = int(self.staircase_config["step_count"])
        for index in range(expected):
            prim_path = f"/World/Stairs/StepLayer_{index + 1:02d}"
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(
                    f"Stair collision layer is missing or invalid: {prim_path}"
                )

    def _sample_foot_tips(self) -> np.ndarray:
        positions_raw, orientations_raw = self.foot_prim.get_world_poses()
        positions = np.asarray(
            positions_raw.numpy()
            if hasattr(positions_raw, "numpy")
            else positions_raw,
            dtype=np.float64,
        ).reshape(len(LEGS), 3)
        orientations = np.asarray(
            orientations_raw.numpy()
            if hasattr(orientations_raw, "numpy")
            else orientations_raw,
            dtype=np.float64,
        ).reshape(len(LEGS), 4)
        if not np.all(np.isfinite(positions)) or not np.all(
            np.isfinite(orientations)
        ):
            raise RuntimeError("Foot poses contain non-finite values")
        tip_positions = positions.copy()
        local_tip = np.asarray([LINK_LENGTH_M, 0.0, 0.0], dtype=np.float64)
        for index in range(len(LEGS)):
            tip_positions[index] += _rotate_wxyz(
                orientations[index],
                local_tip,
            )
        tip_positions[:, 2] -= FOOT_CONTACT_RADIUS_M
        return tip_positions.astype(np.float32)

    def _sample_robot_com_position_m(self) -> np.ndarray:
        """Return the mass-weighted articulation COM in world coordinates."""

        positions_raw, orientations_raw = self.robot_link_prim.get_world_poses()
        positions = np.asarray(
            positions_raw.numpy()
            if hasattr(positions_raw, "numpy")
            else positions_raw,
            dtype=np.float64,
        ).reshape(self.robot.num_links, 3)
        orientations = np.asarray(
            orientations_raw.numpy()
            if hasattr(orientations_raw, "numpy")
            else orientations_raw,
            dtype=np.float64,
        ).reshape(self.robot.num_links, 4)
        if self.robot_link_masses_kg is None:
            masses_raw = self.robot_link_prim.get_masses()
            self.robot_link_masses_kg = np.asarray(
                masses_raw.numpy()
                if hasattr(masses_raw, "numpy")
                else masses_raw,
                dtype=np.float64,
            ).reshape(self.robot.num_links)
        if self.robot_link_com_offsets_m is None:
            offsets_raw, _ = self.robot_link_prim.get_coms()
            self.robot_link_com_offsets_m = np.asarray(
                offsets_raw.numpy()
                if hasattr(offsets_raw, "numpy")
                else offsets_raw,
                dtype=np.float64,
            ).reshape(self.robot.num_links, 3)
        if (
            not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(orientations))
            or not np.all(np.isfinite(self.robot_link_masses_kg))
            or not np.all(np.isfinite(self.robot_link_com_offsets_m))
            or np.any(self.robot_link_masses_kg <= 0.0)
        ):
            raise RuntimeError("Robot COM inputs contain invalid values")
        link_com_positions = positions.copy()
        for index in range(self.robot.num_links):
            link_com_positions[index] += _rotate_wxyz(
                orientations[index],
                self.robot_link_com_offsets_m[index],
            )
        total_mass = float(np.sum(self.robot_link_masses_kg))
        if total_mass <= 0.0:
            raise RuntimeError("Robot articulation has no positive mass")
        return (
            np.sum(
                link_com_positions * self.robot_link_masses_kg[:, None],
                axis=0,
            )
            / total_mass
        ).astype(np.float64)

    def _placement_balance_target_xy_m(
        self,
        *,
        hold_offset_scale: float = 1.0,
    ) -> np.ndarray:
        """Return the retained whole-robot balance target for this phase."""

        hold_offset_scale = float(hold_offset_scale)
        if not 0.0 <= hold_offset_scale <= 1.0:
            raise ValueError("hold_offset_scale must be within [0, 1]")

        target = self.placement_transfer_target_balance_position_m[:2].copy()
        if not np.any(np.abs(target) > 1e-9):
            target = self.placement_leg_baseline_balance_position_m[:2].copy()
        if not self.placement_transfer_active:
            hold_target_offset = dict(
                self.com_regulation_config.get("hold_target_offset_m", {})
            )
            hold_target_offset.update(
                dict(
                    dict(
                        self.com_regulation_config.get(
                            "hold_target_offset_by_swing_leg",
                            {},
                        )
                    ).get(self.placement_swing_leg, {})
                )
            )
            target += hold_offset_scale * np.asarray(
                [
                    float(hold_target_offset.get("forward", 0.0)),
                    float(hold_target_offset.get("lateral", 0.0)),
                ],
                dtype=np.float64,
            )
        return target

    def _placement_balance_target_error_xy_m(self) -> np.ndarray:
        """Return composite-COM error relative to the retained support target."""

        target_position_xy_m = (
            self.latest_support_margin_constrained_target_xy_m
            if self.latest_support_margin_regulation_active
            else self._placement_balance_target_xy_m()
        )
        return balance_target_error_xy(
            balance_position_xy_m=self.latest_placement_com_position_m[:2],
            target_position_xy_m=target_position_xy_m,
        )

    def _sample_foot_contact_loads(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.placement_reference_enabled:
            return (
                np.zeros(len(LEGS), dtype=np.float32),
                np.zeros(
                    (len(LEGS), int(self.staircase_config["step_count"])),
                    dtype=np.float32,
                ),
            )
        raw_forces = self.foot_prim.get_contact_force_matrix(
            dt=1.0 / float(self.physics_hz)
        )
        forces = np.asarray(
            raw_forces.numpy() if hasattr(raw_forces, "numpy") else raw_forces,
            dtype=np.float64,
        ).reshape(len(LEGS), -1, 3)
        expected_filters = 1 + int(self.staircase_config["step_count"])
        if forces.shape[1] != expected_filters or not np.all(np.isfinite(forces)):
            raise RuntimeError(
                "Unexpected placement contact-force matrix: "
                f"{forces.shape} != ({len(LEGS)}, {expected_filters}, 3)"
            )
        normal_loads = np.maximum(forces[:, :, 2], 0.0).astype(np.float32)
        return normal_loads[:, 0], normal_loads[:, 1:]

    def _placement_state(self, elapsed_seconds: float) -> dict[str, object]:
        timing = dict(self.placement_reference_config["timing"])
        timing.update(
            dict(
                dict(
                    self.placement_reference_config.get(
                        "timing_override_by_leg",
                        {},
                    )
                ).get(self.placement_swing_leg, {})
            )
        )
        placement_elapsed_seconds = (
            max(
                0.0,
                float(elapsed_seconds)
                - self.placement_phase_start_step * self.control_dt_s,
            )
            + self.placement_phase_elapsed_offset_s
        )
        if self.placement_early_contact_hold_elapsed_s is not None:
            placement_elapsed_seconds = (
                self.placement_early_contact_hold_elapsed_s
            )
        return placement_reference_state(
            placement_elapsed_seconds,
            timing=timing,
            level=self._active_placement_level(),
        )

    def _reference_parameters_from_joint_positions(
        self,
        joint_positions: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        joint_by_name = dict(
            zip(
                self.dof_names,
                np.asarray(joint_positions, dtype=np.float64).reshape(12),
                strict=True,
            )
        )
        result: dict[str, dict[str, float]] = {}
        for leg in LEGS:
            abduction = float(joint_by_name[f"{leg}_hip_abduction"])
            hip = float(joint_by_name[f"{leg}_hip_flexion"])
            knee = float(joint_by_name[f"{leg}_knee"])
            planar_down = LINK_LENGTH_M * (
                np.cos(hip) + np.cos(hip + knee)
            )
            result[leg] = {
                "forward_m": float(
                    LINK_LENGTH_M
                    * (np.sin(hip) + np.sin(hip + knee))
                ),
                "vertical_m": float(planar_down * np.cos(abduction)),
                "outward_m": float(planar_down * np.sin(abduction)),
            }
        return result

    def _targets_from_reference_parameters(
        self,
        reference_by_leg: Mapping[str, Mapping[str, float]],
    ) -> np.ndarray:
        down_by_leg: dict[str, float] = {}
        forward_by_leg: dict[str, float] = {}
        abduction_by_leg_deg: dict[str, float] = {}
        self.latest_reference_parameters_by_leg = {}
        for leg in LEGS:
            stored = dict(reference_by_leg[leg])
            vertical = float(stored["vertical_m"])
            outward = float(stored["outward_m"])
            forward = float(stored["forward_m"])
            down = float(np.hypot(vertical, outward))
            if down <= 0.0:
                raise ValueError(f"Non-positive placement reach for {leg}")
            maximum_reach = 2.0 * LINK_LENGTH_M - 1e-4
            requested_reach = float(np.hypot(down, forward))
            if requested_reach > maximum_reach:
                self.placement_reference_reach_clip_count += 1
                self.maximum_placement_reference_reach_excess_m = max(
                    self.maximum_placement_reference_reach_excess_m,
                    requested_reach - maximum_reach,
                )
                if down >= maximum_reach:
                    scale = maximum_reach / down
                    vertical *= scale
                    outward *= scale
                    down = maximum_reach
                    forward = 0.0
                else:
                    maximum_forward = float(
                        np.sqrt(maximum_reach**2 - down**2)
                    )
                    forward = float(
                        np.clip(
                            forward,
                            -maximum_forward,
                            maximum_forward,
                        )
                    )
            forward_by_leg[leg] = forward
            down_by_leg[leg] = down
            abduction_by_leg_deg[leg] = float(
                np.degrees(np.arctan2(outward, vertical))
            )
            self.latest_reference_parameters_by_leg[leg] = {
                "forward_m": forward,
                "vertical_m": vertical,
                "outward_m": outward,
            }
        pose = pose_by_name(
            down_by_leg_m=down_by_leg,
            forward_by_leg_m=forward_by_leg,
            abduction_by_leg_deg=abduction_by_leg_deg,
        )
        targets = np.asarray(
            targets_for_order(self.dof_names, pose),
            dtype=np.float32,
        )
        return np.clip(
            targets,
            self.lower_limits + 1e-3,
            self.upper_limits - 1e-3,
        )

    def _begin_inter_leg_transfer(
        self,
        *,
        base_position_m: np.ndarray,
        com_position_m: np.ndarray,
        foot_tips_m: np.ndarray,
        joint_positions_rad: np.ndarray,
    ) -> None:
        self.placement_transfer_active = True
        self.placement_transfer_start_step = self.episode_step + 1
        self.placement_transfer_gate_step_count = 0
        self.placement_transfer_pre_unload_gate_step_count = 0
        self.placement_transfer_unload_start_step = None
        self.placement_phase_elapsed_offset_s = 0.0
        self.placement_transfer_reference_by_leg = (
            self._reference_parameters_from_joint_positions(
                joint_positions_rad
            )
        )
        self.placement_transfer_start_base_position_m = np.asarray(
            base_position_m,
            dtype=np.float64,
        ).copy()
        balance_position = (
            np.asarray(com_position_m, dtype=np.float64)
            if self.com_regulation_enabled
            and self.com_regulation_balance_point == "composite_com"
            else self.placement_transfer_start_base_position_m
        )
        self.placement_transfer_start_balance_position_m = (
            balance_position.copy()
        )
        support_points = np.asarray(
            foot_tips_m[list(self.placement_support_leg_indices), :2],
            dtype=np.float64,
        )
        blend = float(
            self.com_regulation_config.get(
                "target_incenter_blend",
                self.inter_leg_transfer_config.get(
                    "support_incenter_blend",
                    1.0,
                ),
            )
        )
        target_offset = dict(
            self.com_regulation_config.get(
                "target_offset_m",
                self.inter_leg_transfer_config.get("target_offset_m", {}),
            )
        )
        target_offset.update(
            dict(
                dict(
                    self.com_regulation_config.get(
                        "target_offset_by_swing_leg",
                        {},
                    )
                ).get(self.placement_swing_leg, {})
            )
        )
        target_offset_xy = np.asarray(
            [
                float(target_offset.get("forward", 0.0)),
                float(target_offset.get("lateral", 0.0)),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(target_offset_xy)):
            raise ValueError(
                "inter_leg_transfer.target_offset_m must be finite"
            )
        maximum_correction = dict(
            self.com_regulation_config.get("maximum_correction_m", {})
        )
        maximum_correction.update(
            dict(
                dict(
                    self.com_regulation_config.get(
                        "maximum_correction_m_by_swing_leg",
                        {},
                    )
                ).get(self.placement_swing_leg, {})
            )
        )
        target_balance_xy = bounded_support_incenter_target_xy(
            reference_point_xy_m=balance_position[:2],
            support_points_xy_m=support_points,
            incenter_blend=blend,
            target_offset_xy_m=target_offset_xy,
            maximum_shift_xy_m=(
                float(
                    maximum_correction.get(
                        "forward",
                        max(
                            float(
                                self.inter_leg_transfer_config.get(
                                    "maximum_forward_shift_m",
                                    0.08,
                                )
                            ),
                            float(
                                self.inter_leg_transfer_config.get(
                                    "maximum_backward_shift_m",
                                    0.08,
                                )
                            ),
                        ),
                    )
                ),
                float(
                    maximum_correction.get(
                        "lateral",
                        self.inter_leg_transfer_config.get(
                            "maximum_lateral_shift_m",
                            0.12,
                        ),
                    )
                ),
            ),
        )
        desired_delta_xy = target_balance_xy - balance_position[:2]
        self.placement_transfer_target_base_position_m = (
            self.placement_transfer_start_base_position_m.copy()
        )
        self.placement_transfer_target_base_position_m[:2] += desired_delta_xy
        self.placement_transfer_target_balance_position_m = (
            self.placement_transfer_start_balance_position_m.copy()
        )
        self.placement_transfer_target_balance_position_m[:2] = (
            target_balance_xy
        )
        self.placement_leg_start_foot_tips_m = np.asarray(
            foot_tips_m,
            dtype=np.float32,
        ).copy()

    def _active_inter_leg_transfer_config(self) -> dict[str, object]:
        """Return transfer timing with an optional next-leg override."""

        config = dict(self.inter_leg_transfer_config)
        config.update(
            dict(
                dict(config.get("override_by_next_swing_leg", {})).get(
                    self.placement_swing_leg,
                    {},
                )
            )
        )
        return config

    def _inter_leg_transfer_targets(
        self,
        transfer_state: Mapping[str, object],
    ) -> np.ndarray:
        transfer_fraction = float(transfer_state["transfer_fraction"])
        desired_base = self.placement_transfer_start_base_position_m + (
            transfer_fraction
            * (
                self.placement_transfer_target_base_position_m
                - self.placement_transfer_start_base_position_m
            )
        )
        desired_base_delta = (
            desired_base - self.placement_transfer_start_base_position_m
        )
        if self.com_regulation_enabled:
            desired_balance = self.placement_transfer_start_balance_position_m + (
                transfer_fraction
                * (
                    self.placement_transfer_target_balance_position_m
                    - self.placement_transfer_start_balance_position_m
                )
            )
            desired_balance_delta = np.asarray(
                [
                    desired_balance[0]
                    - self.placement_transfer_start_balance_position_m[0],
                    desired_balance[1]
                    - self.placement_transfer_start_balance_position_m[1],
                    desired_base_delta[2],
                ],
                dtype=np.float64,
            )
            actual_balance_delta = np.asarray(
                [
                    self.latest_placement_com_position_m[0]
                    - self.placement_transfer_start_balance_position_m[0],
                    self.latest_placement_com_position_m[1]
                    - self.placement_transfer_start_balance_position_m[1],
                    self.latest_placement_base_position_m[2]
                    - self.placement_transfer_start_base_position_m[2],
                ],
                dtype=np.float64,
            )
            feedback = dict(
                self.com_regulation_config.get(
                    "transfer_feedback_gain",
                    self.com_regulation_config.get("feedback_gain", {}),
                )
            )
            desired_base_delta = stabilized_support_reference_base_delta(
                desired_base_delta_m=desired_balance_delta,
                actual_base_delta_m=actual_balance_delta,
                anchor_follow_gain=(0.0, 0.0, 0.0),
                error_feedback_gain_xyz=(
                    float(feedback.get("forward", 1.0)),
                    float(feedback.get("lateral", 1.0)),
                    float(feedback.get("vertical", 1.0)),
                ),
            )
        adjusted: dict[str, dict[str, float]] = {}
        for leg, stored in self.placement_transfer_reference_by_leg.items():
            side_sign = 1.0 if leg.endswith("_left") else -1.0
            adjusted[leg] = {
                "forward_m": float(stored["forward_m"])
                - float(desired_base_delta[0]),
                "vertical_m": float(stored["vertical_m"])
                + float(desired_base_delta[2]),
                "outward_m": float(stored["outward_m"])
                - side_sign * float(desired_base_delta[1]),
            }
        if (
            self.pitch_feedback_enabled
            and bool(
                self.pitch_feedback_config.get(
                    "apply_during_inter_leg_transfer",
                    False,
                )
            )
            and self.placement_swing_leg
            in tuple(
                self.pitch_feedback_config.get(
                    "inter_leg_transfer_legs",
                    LEGS,
                )
            )
            and transfer_fraction > 0.0
        ):
            # A front-pair-on-tread/rear-pair-on-ground transfer spans the
            # full stair rise. Regulate sagittal attitude while the COM moves,
            # before asking the next rear foot to unload. This is opt-in so
            # the already accepted V34 front-pair controller is unchanged.
            pitch_gain_by_leg = dict(
                self.pitch_feedback_config.get(
                    "inter_leg_transfer_proportional_gain_m_by_swing_leg",
                    {},
                )
            )
            pitch_maximum_by_leg = dict(
                self.pitch_feedback_config.get(
                    "inter_leg_transfer_maximum_correction_m_by_swing_leg",
                    {},
                )
            )
            pitch_corrections = support_pitch_vertical_corrections(
                support_legs=(
                    LEGS[index] for index in self.placement_support_leg_indices
                ),
                projected_gravity_x=float(
                    self.latest_projected_gravity_xyz[0]
                ),
                proportional_gain_m=float(
                    pitch_gain_by_leg.get(
                        self.placement_swing_leg,
                        self.pitch_feedback_config.get(
                            "inter_leg_transfer_proportional_gain_m",
                            self.pitch_feedback_config.get(
                                "proportional_gain_m",
                                0.12,
                            ),
                        ),
                    )
                ),
                maximum_correction_m=float(
                    pitch_maximum_by_leg.get(
                        self.placement_swing_leg,
                        self.pitch_feedback_config.get(
                            "inter_leg_transfer_maximum_correction_m",
                            self.pitch_feedback_config.get(
                                "maximum_correction_m",
                                0.035,
                            ),
                        ),
                    )
                ),
            )
            front_only_by_leg = dict(
                self.pitch_feedback_config.get(
                    "inter_leg_transfer_front_only_by_swing_leg",
                    {},
                )
            )
            inter_leg_transfer_front_only = bool(
                front_only_by_leg.get(
                    self.placement_swing_leg,
                    self.pitch_feedback_config.get(
                        "inter_leg_transfer_front_only",
                        False,
                    ),
                )
            )
            for leg, correction_m in pitch_corrections.items():
                if (
                    inter_leg_transfer_front_only
                    and self.placement_swing_leg.startswith("rear_")
                    and leg.startswith("rear_")
                ):
                    continue
                # The same feedback was active at the end of the preceding
                # placement phase. Keep it continuous across the phase
                # boundary instead of dropping it to zero and ramping again.
                adjusted[leg]["vertical_m"] += correction_m
        preload_load_config = self.four_foot_preload_load_sharing_config
        if (
            self.four_foot_preload_load_sharing_enabled
            and self.placement_transfer_unload_start_step is None
            and self.placement_swing_leg
            in tuple(preload_load_config.get("next_swing_legs", LEGS))
            and transfer_fraction > 0.0
        ):
            total_foot_loads = self.latest_ground_normal_loads_n + np.sum(
                self.latest_step_normal_loads_n,
                axis=1,
            )
            raw_corrections = equalized_foot_load_vertical_corrections(
                measured_normal_loads_n=total_foot_loads,
                proportional_gain_m=float(
                    preload_load_config.get("proportional_gain_m", 0.030)
                ),
                maximum_correction_m=float(
                    preload_load_config.get("maximum_correction_m", 0.012)
                ),
                minimum_total_load_n=float(
                    preload_load_config.get("minimum_total_load_n", 1.0)
                ),
            ) * transfer_fraction
            smoothing_factor = float(
                preload_load_config.get("smoothing_factor", 0.50)
            )
            corrections = self.latest_support_load_sharing_correction_m + (
                smoothing_factor
                * (
                    raw_corrections
                    - self.latest_support_load_sharing_correction_m
                )
            )
            corrections -= float(np.mean(corrections))
            maximum_correction_m = float(
                preload_load_config.get("maximum_correction_m", 0.012)
            )
            corrections = np.clip(
                corrections,
                -maximum_correction_m,
                maximum_correction_m,
            )
            for leg_index, leg in enumerate(LEGS):
                adjusted[leg]["vertical_m"] += float(corrections[leg_index])
            self.latest_support_load_sharing_correction_m = corrections.copy()
            self.maximum_abs_support_load_sharing_correction_m = max(
                self.maximum_abs_support_load_sharing_correction_m,
                float(np.max(np.abs(corrections))),
            )
        adjusted[self.placement_swing_leg]["vertical_m"] -= float(
            self._active_inter_leg_transfer_config().get(
                "swing_unload_lift_m",
                0.0,
            )
        ) * float(transfer_state.get("unload_fraction", 0.0))
        return self._targets_from_reference_parameters(adjusted)

    def _complete_inter_leg_transfer(
        self,
        *,
        base_position_m: np.ndarray,
        com_position_m: np.ndarray,
        foot_tips_m: np.ndarray,
        joint_positions_rad: np.ndarray,
    ) -> None:
        previous_leg = self.completed_placement_legs[-1]
        self.completed_inter_leg_transfers.append(
            f"{previous_leg}->{self.placement_swing_leg}"
        )
        self.placement_leg_baseline_reference_by_leg = (
            self._reference_parameters_from_joint_positions(
                joint_positions_rad
            )
        )
        self.placement_leg_baseline_base_position_m = np.asarray(
            base_position_m,
            dtype=np.float64,
        ).copy()
        self.placement_leg_baseline_balance_position_m = (
            np.asarray(com_position_m, dtype=np.float64).copy()
            if self.com_regulation_enabled
            and self.com_regulation_balance_point == "composite_com"
            else self.placement_leg_baseline_base_position_m.copy()
        )
        self.placement_leg_baseline_lift_offset_m = float(
            self._active_inter_leg_transfer_config().get(
                "swing_unload_lift_m",
                0.0,
            )
        )
        self.placement_leg_start_foot_tips_m = np.asarray(
            foot_tips_m,
            dtype=np.float32,
        ).copy()
        self.placement_transfer_active = False
        self.placement_transfer_gate_step_count = 0
        self.placement_transfer_pre_unload_gate_step_count = 0
        self.placement_transfer_unload_start_step = None
        self.placement_phase_start_step = self.episode_step + 1
        self.placement_phase_elapsed_offset_s = 0.0

    def _placement_targets_from_leg_baseline(
        self,
        placement_state: Mapping[str, object],
    ) -> np.ndarray:
        self.latest_placement_pitch_rear_correction_scale = 1.0
        actual_base_delta = (
            self.latest_placement_base_position_m
            - self.placement_leg_baseline_base_position_m
        )
        post_transfer_shift = dict(
            self.inter_leg_transfer_config.get(
                "post_transfer_weight_shift",
                {},
            )
        )
        shift_fraction = float(placement_state["shift_fraction"])
        swing_front_sign = (
            1.0 if self.placement_swing_leg.startswith("front_") else -1.0
        )
        swing_side_sign = (
            1.0 if self.placement_swing_leg.endswith("_left") else -1.0
        )
        desired_base_delta = np.asarray(
            [
                -swing_front_sign
                * float(post_transfer_shift.get("forward_m", 0.0))
                * shift_fraction,
                -swing_side_sign
                * float(post_transfer_shift.get("lateral_m", 0.0))
                * shift_fraction,
                0.0,
            ],
            dtype=np.float64,
        )
        if self.com_regulation_enabled:
            # A hold offset is a touchdown/load-transfer command, not a swing
            # command. Ramp it in with the lower phase so clearance remains
            # governed by the already verified transfer and lift targets.
            target_balance_xy = self._placement_balance_target_xy_m(
                hold_offset_scale=float(placement_state["lower_fraction"]),
            )
            desired_base_delta[:2] = shift_fraction * (
                target_balance_xy
                - self.placement_leg_baseline_balance_position_m[:2]
            )
            desired_base_delta[:2] += np.asarray(
                [
                    -swing_front_sign
                    * float(post_transfer_shift.get("forward_m", 0.0))
                    * shift_fraction,
                    -swing_side_sign
                    * float(post_transfer_shift.get("lateral_m", 0.0))
                    * shift_fraction,
                ],
                dtype=np.float64,
            )
            actual_base_delta = np.asarray(
                [
                    self.latest_placement_com_position_m[0]
                    - self.placement_leg_baseline_balance_position_m[0],
                    self.latest_placement_com_position_m[1]
                    - self.placement_leg_baseline_balance_position_m[1],
                    actual_base_delta[2],
                ],
                dtype=np.float64,
            )
        post_clearance_shift = dict(
            dict(
                self.inter_leg_transfer_config.get(
                    "post_clearance_body_shift_by_leg",
                    {},
                )
            ).get(self.placement_swing_leg, {})
        )
        advance_fraction = float(placement_state["advance_fraction"])
        body_shift_fraction = advance_fraction
        swing_advance_fraction = advance_fraction
        split_fraction = post_clearance_shift.get(
            "body_shift_fraction_of_advance"
        )
        if split_fraction is not None:
            body_shift_fraction, swing_advance_fraction = (
                split_post_clearance_advance_fractions(
                    advance_fraction=advance_fraction,
                    body_shift_fraction_of_advance=float(split_fraction),
                    sequence=str(
                        post_clearance_shift.get(
                            "sequence",
                            "body_then_swing",
                        )
                    ),
                )
            )
        desired_base_delta[:2] += body_shift_fraction * np.asarray(
            [
                float(post_clearance_shift.get("forward_m", 0.0)),
                float(post_clearance_shift.get("lateral_m", 0.0)),
            ],
            dtype=np.float64,
        )
        self.latest_support_margin_regulation_active = False
        if (
            self.support_margin_regulation_enabled
            and str(placement_state["phase"])
            in self.support_margin_regulation_phases
        ):
            support_indices = list(self.placement_support_leg_indices)
            support_points = np.asarray(
                self.latest_foot_tips_m[support_indices, :2],
                dtype=np.float64,
            )
            requested_target_xy = (
                self.placement_leg_baseline_balance_position_m[:2]
                + desired_base_delta[:2]
            )
            minimum_target_margin_m = float(
                self.support_margin_regulation_config.get(
                    "minimum_target_margin_m",
                    0.020,
                )
            )
            constrained_target_xy = support_margin_constrained_target_xy(
                desired_target_xy_m=requested_target_xy,
                support_points_xy_m=support_points,
                minimum_margin_m=minimum_target_margin_m,
            )
            self.latest_support_margin_regulation_active = True
            self.latest_support_margin_requested_target_xy_m = (
                requested_target_xy.copy()
            )
            self.latest_support_margin_constrained_target_xy_m = (
                constrained_target_xy.copy()
            )
            self.latest_support_margin_commanded_target_margin_m = (
                _support_triangle_signed_margin_m(
                    constrained_target_xy,
                    support_points,
                )
            )
            self.maximum_support_margin_target_clip_m = max(
                self.maximum_support_margin_target_clip_m,
                float(
                    np.linalg.norm(
                        constrained_target_xy - requested_target_xy
                    )
                ),
            )
            desired_base_delta[:2] = (
                constrained_target_xy
                - self.placement_leg_baseline_balance_position_m[:2]
            )
        anchor_follow_by_axis = self.inter_leg_transfer_config.get(
            "support_world_anchor_follow_gain_xyz"
        )
        if isinstance(anchor_follow_by_axis, Mapping):
            anchor_follow_gain: float | tuple[float, float, float] = (
                float(anchor_follow_by_axis.get("forward", 0.0)),
                float(anchor_follow_by_axis.get("lateral", 0.0)),
                float(anchor_follow_by_axis.get("vertical", 0.0)),
            )
        else:
            anchor_follow_gain = float(
                self.inter_leg_transfer_config.get(
                    "support_world_anchor_follow_gain",
                    0.25,
                )
            )
        feedback_config = dict(
            self.com_regulation_config.get("feedback_gain", {})
            if self.com_regulation_enabled
            else self.inter_leg_transfer_config.get(
                "support_base_error_feedback_gain",
                {},
            )
        )
        support_base_delta = stabilized_support_reference_base_delta(
            desired_base_delta_m=desired_base_delta,
            actual_base_delta_m=actual_base_delta,
            anchor_follow_gain=anchor_follow_gain,
            error_feedback_gain_xyz=(
                float(feedback_config.get("forward", 0.0)),
                float(feedback_config.get("lateral", 0.0)),
                float(feedback_config.get("vertical", 0.0)),
            ),
        )
        if self.com_regulation_enabled:
            maximum_feedback = dict(
                self.com_regulation_config.get(
                    "maximum_feedback_correction_m",
                    {},
                )
            )
            maximum_feedback_xyz = np.asarray(
                [
                    float(maximum_feedback.get("forward", 0.025)),
                    float(maximum_feedback.get("lateral", 0.035)),
                    float(maximum_feedback.get("vertical", 0.020)),
                ],
                dtype=np.float64,
            )
            support_base_delta = desired_base_delta + np.clip(
                support_base_delta - desired_base_delta,
                -maximum_feedback_xyz,
                maximum_feedback_xyz,
            )
        # The drift-rejection term is a stance controller. Applying it to the
        # swing foot as well moves that foot back toward the ground whenever
        # the body sags, which cancels the requested lift. Preserve the prior
        # anchor-follow behavior for the swing trajectory while the other
        # three legs actively restore the body pose.
        swing_base_delta = stabilized_support_reference_base_delta(
            desired_base_delta_m=desired_base_delta,
            actual_base_delta_m=actual_base_delta,
            anchor_follow_gain=anchor_follow_gain,
            error_feedback_gain_xyz=(0.0, 0.0, 0.0),
        )
        swing_reference_mode = str(
            dict(
                self.inter_leg_transfer_config.get(
                    "post_transfer_swing_reference_mode_by_leg",
                    {},
                )
            ).get(
                self.placement_swing_leg,
                self.inter_leg_transfer_config.get(
                    "post_transfer_swing_reference_mode",
                    "phase_baseline",
                ),
            )
        )
        if swing_reference_mode not in {
            "phase_baseline",
            "nominal_stance",
            "blend_to_nominal_stance",
        }:
            raise ValueError(
                "post_transfer_swing_reference_mode must be phase_baseline "
                "nominal_stance, or blend_to_nominal_stance"
            )
        swing_advance_scale_by_leg = dict(
            self.inter_leg_transfer_config.get(
                "post_clearance_swing_base_delta_end_scale_by_leg",
                {},
            )
        )
        swing_advance_scale = swing_advance_scale_by_leg.get(
            self.placement_swing_leg
        )
        if swing_advance_scale is not None:
            if not isinstance(swing_advance_scale, Mapping):
                raise ValueError(
                    "post_clearance_swing_base_delta_end_scale_by_leg values "
                    "must map forward, lateral, and vertical scales"
                )
            swing_base_delta = staged_swing_reference_base_delta(
                base_delta_m=swing_base_delta,
                advance_fraction=swing_advance_fraction,
                end_scale_xyz=(
                    float(swing_advance_scale.get("forward", 1.0)),
                    float(swing_advance_scale.get("lateral", 1.0)),
                    float(swing_advance_scale.get("vertical", 1.0)),
                ),
            )
        reference_by_leg = {
            leg: dict(stored)
            for leg, stored in self.placement_leg_baseline_reference_by_leg.items()
        }
        swing_lift_offset_m = self.placement_leg_baseline_lift_offset_m
        if swing_reference_mode in {
            "nominal_stance",
            "blend_to_nominal_stance",
        }:
            stance = dict(self.config["nominal_stance"])
            nominal_down = float(stance["down_m"])
            nominal_abduction = np.deg2rad(float(stance["abduction_deg"]))
            nominal_reference = {
                "forward_m": (
                    float(stance["fore_aft_m"])
                    if self.placement_swing_leg.startswith("front_")
                    else -float(stance["fore_aft_m"])
                ),
                "vertical_m": nominal_down * float(np.cos(nominal_abduction)),
                "outward_m": nominal_down * float(np.sin(nominal_abduction)),
            }
            blend = (
                1.0
                if swing_reference_mode == "nominal_stance"
                else float(placement_state["lift_fraction"])
            )
            transferred_reference = reference_by_leg[self.placement_swing_leg]
            reference_by_leg[self.placement_swing_leg] = {
                key: (1.0 - blend) * float(transferred_reference[key])
                + blend * float(nominal_reference[key])
                for key in ("forward_m", "vertical_m", "outward_m")
            }
            swing_lift_offset_m *= 1.0 - blend
            # Keep the proven nominal swing trajectory body-relative. The
            # post-transfer counter-shift is a stance command and must not
            # distort swing-leg reach or lift clearance.
            swing_base_delta = np.zeros(3, dtype=np.float64)
        adjusted: dict[str, dict[str, float]] = {}
        for leg, stored in reference_by_leg.items():
            base_delta = (
                swing_base_delta
                if leg == self.placement_swing_leg
                else support_base_delta
            )
            side_sign = 1.0 if leg.endswith("_left") else -1.0
            adjusted[leg] = {
                "forward_m": float(stored["forward_m"])
                - float(base_delta[0]),
                "vertical_m": float(stored["vertical_m"])
                + float(base_delta[2]),
                "outward_m": float(stored["outward_m"])
                - side_sign * float(base_delta[1]),
            }
        swing_outward_offset_m = float(
            dict(
                self.inter_leg_transfer_config.get(
                    "swing_outward_offset_m_by_leg",
                    {},
                )
            ).get(self.placement_swing_leg, 0.0)
        )
        adjusted[self.placement_swing_leg]["outward_m"] += (
            staged_swing_outward_offset_m(
                maximum_offset_m=swing_outward_offset_m,
                advance_fraction=swing_advance_fraction,
            )
        )
        support_extension_by_swing = dict(
            self.com_regulation_config.get(
                "support_extension_m_by_swing_leg",
                {},
            )
        )
        support_extension_by_leg = dict(
            support_extension_by_swing.get(
                self.placement_swing_leg,
                {},
            )
        )
        for leg, extension_m in support_extension_by_leg.items():
            if leg == self.placement_swing_leg or leg not in adjusted:
                raise ValueError(
                    "COM support extensions must select known stance legs"
                )
            adjusted[leg]["vertical_m"] += (
                float(extension_m) * shift_fraction
            )
        if self.pitch_feedback_enabled and shift_fraction > 0.0:
            pitch_corrections = support_pitch_vertical_corrections(
                support_legs=(
                    LEGS[index] for index in self.placement_support_leg_indices
                ),
                projected_gravity_x=float(
                    self.latest_projected_gravity_xyz[0]
                ),
                proportional_gain_m=float(
                    dict(
                        self.pitch_feedback_config.get(
                            "proportional_gain_m_by_swing_leg",
                            {},
                        )
                    ).get(
                        self.placement_swing_leg,
                        self.pitch_feedback_config.get(
                            "proportional_gain_m",
                            0.12,
                        ),
                    )
                ),
                maximum_correction_m=float(
                    dict(
                        self.pitch_feedback_config.get(
                            "maximum_correction_m_by_swing_leg",
                            {},
                        )
                    ).get(
                        self.placement_swing_leg,
                        self.pitch_feedback_config.get(
                            "maximum_correction_m",
                            0.035,
                        ),
                    )
                ),
            )
            front_only_legs = tuple(
                self.pitch_feedback_config.get(
                    "front_only_by_swing_leg",
                    (),
                )
            )
            if self.placement_swing_leg in front_only_legs:
                rear_correction_scale = 0.0
            else:
                front_only_seconds = float(
                    dict(
                        self.pitch_feedback_config.get(
                            "front_only_seconds_by_swing_leg",
                            {},
                        )
                    ).get(self.placement_swing_leg, 0.0)
                )
                blend_seconds = float(
                    dict(
                        self.pitch_feedback_config.get(
                            "rear_blend_seconds_by_swing_leg",
                            {},
                        )
                    ).get(self.placement_swing_leg, 0.0)
                )
                rear_correction_scale = staged_support_rear_pitch_scale(
                    elapsed_seconds=float(placement_state["elapsed_seconds"]),
                    front_only_seconds=front_only_seconds,
                    blend_seconds=blend_seconds,
                )
            self.latest_placement_pitch_rear_correction_scale = (
                rear_correction_scale
            )
            for leg, correction_m in pitch_corrections.items():
                correction_scale = (
                    rear_correction_scale if leg.startswith("rear_") else 1.0
                )
                adjusted[leg]["vertical_m"] += (
                    correction_m * shift_fraction * correction_scale
                )
        squat_by_swing = dict(
            self.com_regulation_config.get(
                "support_squat_thrust_by_swing_leg",
                {},
            )
        )
        squat_config = dict(
            squat_by_swing.get(self.placement_swing_leg, {})
        )
        if squat_config:
            crouch_m = float(squat_config["crouch_m"])
            release_fraction = float(
                squat_config["release_lift_fraction"]
            )
            if crouch_m <= 0.0 or not 0.0 < release_fraction <= 1.0:
                raise ValueError(
                    "support squat thrust requires positive crouch and a "
                    "release fraction within (0, 1]"
                )
            squat_fraction = shift_fraction * (
                1.0
                - float(
                    np.clip(
                        float(placement_state["lift_fraction"])
                        / release_fraction,
                        0.0,
                        1.0,
                    )
                )
            )
            for leg in tuple(squat_config["legs"]):
                if leg == self.placement_swing_leg or leg not in adjusted:
                    raise ValueError(
                        "support squat thrust must select stance legs"
                    )
                adjusted[leg]["vertical_m"] -= crouch_m * squat_fraction
        if self.support_load_sharing_enabled and shift_fraction > 0.0:
            support_indices = list(self.placement_support_leg_indices)
            support_loads = self.latest_ground_normal_loads_n[
                support_indices
            ] + np.sum(
                self.latest_step_normal_loads_n[support_indices, :],
                axis=1,
            )
            raw_corrections = support_load_share_vertical_corrections(
                support_points_xy_m=self.latest_foot_tips_m[
                    support_indices,
                    :2,
                ],
                target_position_xy_m=(
                    self.latest_support_margin_constrained_target_xy_m
                    if self.latest_support_margin_regulation_active
                    else self._placement_balance_target_xy_m()
                ),
                measured_normal_loads_n=support_loads,
                proportional_gain_m=float(
                    self.support_load_sharing_config.get(
                        "proportional_gain_m",
                        0.030,
                    )
                ),
                maximum_correction_m=float(
                    self.support_load_sharing_config.get(
                        "maximum_correction_m",
                        0.012,
                    )
                ),
                minimum_total_load_n=float(
                    self.support_load_sharing_config.get(
                        "minimum_total_load_n",
                        1.0,
                    )
                ),
                minimum_desired_fraction=float(
                    self.support_load_sharing_config.get(
                        "minimum_desired_fraction",
                        0.05,
                    )
                ),
            ) * shift_fraction
            smoothing_factor = float(
                self.support_load_sharing_config.get("smoothing_factor", 1.0)
            )
            if not 0.0 < smoothing_factor <= 1.0:
                raise ValueError(
                    "support load-sharing smoothing_factor must be within "
                    "(0, 1]"
                )
            previous_corrections = (
                self.latest_support_load_sharing_correction_m[
                    support_indices
                ].copy()
            )
            corrections = previous_corrections + smoothing_factor * (
                raw_corrections - previous_corrections
            )
            corrections -= float(np.mean(corrections))
            maximum_correction_m = float(
                self.support_load_sharing_config.get(
                    "maximum_correction_m",
                    0.012,
                )
            )
            corrections = np.clip(
                corrections,
                -maximum_correction_m,
                maximum_correction_m,
            )
            self.latest_support_load_sharing_correction_m.fill(0.0)
            for leg_index, correction in zip(
                support_indices,
                corrections,
                strict=True,
            ):
                leg = LEGS[leg_index]
                adjusted[leg]["vertical_m"] += float(correction)
                self.latest_support_load_sharing_correction_m[leg_index] = (
                    correction
                )
                self.maximum_abs_support_load_sharing_correction_m_by_leg[
                    leg_index
                ] = max(
                    self.maximum_abs_support_load_sharing_correction_m_by_leg[
                        leg_index
                    ],
                    abs(float(correction)),
                )
            self.maximum_abs_support_load_sharing_correction_m = max(
                self.maximum_abs_support_load_sharing_correction_m,
                float(np.max(np.abs(corrections))),
            )
            self.support_load_sharing_active_sample_count += 1
            if np.any(
                np.abs(corrections) >= 0.99 * maximum_correction_m
            ):
                self.support_load_sharing_saturated_sample_count += 1
        else:
            self.latest_support_load_sharing_correction_m.fill(0.0)
        swing = adjusted[self.placement_swing_leg]
        desired_forward_offset_m = float(
            placement_state["desired_forward_offset_m"]
        )
        if split_fraction is not None and placement_state["phase"] == "advance":
            active_level = self._active_placement_level()
            final_forward_m = float(active_level["swing_forward_offset_m"])
            lift_forward_m = min(
                final_forward_m,
                float(active_level.get("lift_forward_offset_m", 0.11)),
            )
            desired_forward_offset_m = lift_forward_m + (
                swing_advance_fraction * (final_forward_m - lift_forward_m)
            )
        swing["forward_m"] += desired_forward_offset_m
        raw_touchdown_correction_m = 0.0
        if (
            self.touchdown_load_regulation_enabled
            and self.placement_swing_leg
            in self.touchdown_load_regulation_legs
            and str(placement_state["phase"])
            in self.touchdown_load_regulation_phases
        ):
            measured_tread_load_n = float(
                np.sum(
                    self.latest_step_normal_loads_n[
                        self.placement_swing_leg_index,
                        :,
                    ]
                )
            )
            raw_touchdown_correction_m = (
                touchdown_load_lift_correction_m(
                    measured_tread_load_n=measured_tread_load_n,
                    target_tread_load_n=float(
                        self.touchdown_load_regulation_config.get(
                            "target_tread_load_n",
                            15.0,
                        )
                    ),
                    proportional_gain_m_per_n=float(
                        self.touchdown_load_regulation_config.get(
                            "proportional_gain_m_per_n",
                            0.0005,
                        )
                    ),
                    maximum_lift_correction_m=float(
                        self.touchdown_load_regulation_config.get(
                            "maximum_lift_correction_m",
                            0.035,
                        )
                    ),
                )
            )
        smoothing_key = (
            "attack_smoothing_factor"
            if raw_touchdown_correction_m
            > self.latest_touchdown_load_lift_correction_m
            else "release_smoothing_factor"
        )
        smoothing_default = 0.50 if smoothing_key.startswith("attack") else 0.10
        smoothing_factor = float(
            self.touchdown_load_regulation_config.get(
                smoothing_key,
                smoothing_default,
            )
        )
        self.latest_touchdown_load_lift_correction_m += smoothing_factor * (
            raw_touchdown_correction_m
            - self.latest_touchdown_load_lift_correction_m
        )
        self.maximum_touchdown_load_lift_correction_m = max(
            self.maximum_touchdown_load_lift_correction_m,
            self.latest_touchdown_load_lift_correction_m,
        )
        swing["vertical_m"] -= max(
            0.0,
            float(placement_state["desired_lift_m"])
            + self.latest_touchdown_load_lift_correction_m
            - swing_lift_offset_m,
        )
        return self._targets_from_reference_parameters(adjusted)

    def _placement_reference_targets(
        self,
        placement_state: Mapping[str, object],
    ) -> np.ndarray:
        if self.placement_leg_baseline_reference_by_leg:
            return self._placement_targets_from_leg_baseline(placement_state)
        stance = dict(self.config["nominal_stance"])
        nominal_down = float(stance["down_m"])
        fore_aft = float(stance["fore_aft_m"])
        nominal_abduction = np.deg2rad(float(stance["abduction_deg"]))
        weight_shift = dict(self.placement_reference_config["weight_shift"])
        shift_scale = float(
            dict(weight_shift.get("scale_by_leg", {})).get(
                self.placement_swing_leg,
                1.0,
            )
        )
        if shift_scale < 0.0 or shift_scale > 1.0:
            raise ValueError("placement weight-shift scale must be within [0, 1]")
        shift_fraction = float(placement_state["shift_fraction"])
        swing_front_sign = (
            1.0 if self.placement_swing_leg.startswith("front_") else -1.0
        )
        swing_side_sign = (
            1.0 if self.placement_swing_leg.endswith("_left") else -1.0
        )
        body_shift_forward_m = (
            -swing_front_sign
            * float(weight_shift["forward_m"])
            * shift_scale
            * shift_fraction
        )
        body_shift_lateral_m = (
            -swing_side_sign
            * float(weight_shift["lateral_m"])
            * shift_scale
            * shift_fraction
        )
        down_by_leg = {leg: nominal_down for leg in LEGS}
        down_by_leg[self.placement_swing_leg] = nominal_down - float(
            placement_state["desired_lift_m"]
        )
        forward_by_leg = {
            leg: (
                fore_aft if leg.startswith("front_") else -fore_aft
            )
            - body_shift_forward_m
            for leg in LEGS
        }
        forward_by_leg[self.placement_swing_leg] += float(
            placement_state["desired_forward_offset_m"]
        )
        foot_delta_lateral_m = body_shift_lateral_m
        abduction_by_leg_deg: dict[str, float] = {}
        for leg in LEGS:
            side_sign = 1.0 if leg.endswith("_left") else -1.0
            vertical = down_by_leg[leg] * np.cos(nominal_abduction)
            outward = down_by_leg[leg] * np.sin(nominal_abduction)
            shifted_outward = outward - side_sign * foot_delta_lateral_m
            down_by_leg[leg] = float(np.hypot(vertical, shifted_outward))
            abduction_by_leg_deg[leg] = float(
                np.degrees(np.arctan2(shifted_outward, vertical))
            )
        for leg, stored in self.completed_placement_reference_by_leg.items():
            stored_base = np.asarray(
                stored["base_position_m"],
                dtype=np.float64,
            ).reshape(3)
            base_delta = self.latest_placement_base_position_m - stored_base
            side_sign = 1.0 if leg.endswith("_left") else -1.0
            vertical = float(stored["vertical_m"]) + float(base_delta[2])
            outward = float(stored["outward_m"]) - side_sign * float(
                base_delta[1]
            )
            forward_by_leg[leg] = float(stored["forward_m"]) - float(
                base_delta[0]
            )
            down_by_leg[leg] = float(np.hypot(vertical, outward))
            abduction_by_leg_deg[leg] = float(
                np.degrees(np.arctan2(outward, vertical))
            )
        self.latest_reference_parameters_by_leg = {}
        for leg in LEGS:
            down = float(down_by_leg[leg])
            abduction = np.deg2rad(float(abduction_by_leg_deg[leg]))
            self.latest_reference_parameters_by_leg[leg] = {
                "forward_m": float(forward_by_leg[leg]),
                "vertical_m": down * float(np.cos(abduction)),
                "outward_m": down * float(np.sin(abduction)),
            }
        pose = pose_by_name(
            down_by_leg_m=down_by_leg,
            forward_by_leg_m=forward_by_leg,
            abduction_by_leg_deg=abduction_by_leg_deg,
        )
        targets = np.asarray(
            targets_for_order(self.dof_names, pose),
            dtype=np.float32,
        )
        return np.clip(
            targets,
            self.lower_limits + 1e-3,
            self.upper_limits - 1e-3,
        )

    def _placement_target_world_x_m(self) -> float:
        level = self._active_placement_level()
        return float(self.staircase_config["start_x_m"]) + float(
            level["target_tread_fraction"]
        ) * float(self.staircase_config["tread_depth_m"])

    def _foot_progress(
        self,
        foot_tips: np.ndarray,
    ) -> tuple[float, int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rise = float(self.staircase_config["rise_m"])
        lift_cap = 2.0 * rise
        foot_lifts = np.maximum(
            0.0,
            foot_tips[:, 2] - self.initial_foot_bottom_z_m,
        )
        prior_maximum = self.maximum_foot_lift_m.copy()
        stair_start = float(self.staircase_config["start_x_m"])
        active_end = (
            stair_start
            + self.active_step_count
            * float(self.staircase_config["tread_depth_m"])
        )
        lift_eligible = np.logical_and(
            foot_tips[:, 0] >= stair_start - 0.18,
            foot_tips[:, 0] <= active_end,
        )
        capped_lifts = np.where(
            lift_eligible,
            np.minimum(foot_lifts, lift_cap),
            prior_maximum,
        )
        prior_steps = self.highest_foot_step.copy()
        target_index = next_foot_target_index(
            prior_steps,
            active_steps=self.active_step_count,
            sequence_indices=self.foot_placement_sequence_indices,
        )
        new_maximum = prior_maximum.copy()
        if target_index is not None:
            new_maximum[target_index] = max(
                prior_maximum[target_index],
                capped_lifts[target_index],
            )
        lift_progress = float(np.sum(new_maximum - prior_maximum))

        placement_tolerance = min(
            float(self.config.get("foot_placement_tolerance_m", 0.0125)),
            0.25 * rise,
        )
        current_steps = np.zeros(len(LEGS), dtype=np.int32)
        for index, foot_tip in enumerate(foot_tips):
            step_index = min(
                self.active_step_count,
                stair_index_at_x(float(foot_tip[0]), self.staircase_config),
            )
            surface_height = stair_height_at_x(
                float(foot_tip[0]),
                self.staircase_config,
            )
            if (
                step_index > 0
                and surface_height - placement_tolerance
                <= float(foot_tip[2])
                <= surface_height + placement_tolerance
            ):
                current_steps[index] = step_index
        new_steps = np.maximum(prior_steps, current_steps)
        placement_progress = (
            0
            if target_index is None
            else int(new_steps[target_index] - prior_steps[target_index])
        )
        current_tread_progress = foot_tread_progress(
            foot_tip_positions_m=foot_tips,
            highest_foot_steps=new_steps,
            staircase=self.staircase_config,
            active_steps=self.active_step_count,
            approach_distance_m=float(
                self.config.get("foot_tread_approach_distance_m", 0.20)
            ),
        )
        next_maximum_tread_progress = self.maximum_foot_tread_progress.copy()
        if target_index is not None:
            next_maximum_tread_progress[target_index] = max(
                self.maximum_foot_tread_progress[target_index],
                current_tread_progress[target_index],
            )
        tread_progress_gain = float(
            np.sum(
                next_maximum_tread_progress
                - self.maximum_foot_tread_progress
            )
        )
        return (
            lift_progress,
            placement_progress,
            tread_progress_gain,
            new_maximum,
            new_steps,
            current_steps,
            next_maximum_tread_progress,
        )

    def set_training_progress(self, progress_fraction: float) -> None:
        """Schedule a curriculum level; it becomes active at the next reset."""

        progress = float(np.clip(progress_fraction, 0.0, 1.0))
        active = curriculum_active_steps(
            progress,
            self.curriculum_levels,
            maximum_steps=int(self.staircase_config["step_count"]),
        )
        if active != self.pending_active_step_count:
            self.curriculum_transitions.append(
                {
                    "progress_fraction": progress,
                    "active_steps": active,
                }
            )
        self.curriculum_progress = progress
        self.pending_active_step_count = active
        if self.placement_reference_enabled and str(
            self.placement_curriculum_config.get("mode", "timesteps")
        ) == "timesteps":
            self.set_placement_curriculum_progress(progress)

    def set_placement_curriculum_progress(self, progress_fraction: float) -> None:
        """Select the explicit single-tread placement stage."""

        if not self.placement_reference_enabled:
            return
        selected = placement_curriculum_level(
            progress_fraction,
            self.placement_curriculum_levels,
        )
        selected_id = str(selected["id"])
        if selected_id != self.pending_placement_level_id:
            self.curriculum_transitions.append(
                {
                    "progress_fraction": float(
                        np.clip(progress_fraction, 0.0, 1.0)
                    ),
                    "placement_level": selected_id,
                }
            )
        self.pending_placement_level = selected
        self.pending_placement_level_id = selected_id

    def set_placement_level(
        self,
        level_id: str,
        *,
        activate_immediately: bool = False,
    ) -> None:
        """Select one named placement level for targeted training/evaluation."""

        if not self.placement_reference_enabled:
            raise ValueError("placement reference is not enabled")
        selected = next(
            (
                dict(level)
                for level in self.placement_curriculum_levels
                if str(level["id"]) == str(level_id)
            ),
            None,
        )
        if selected is None:
            known = [str(level["id"]) for level in self.placement_curriculum_levels]
            raise ValueError(
                f"unknown placement level {level_id!r}; expected one of {known}"
            )
        self.pending_placement_level = selected
        self.pending_placement_level_id = str(selected["id"])
        if activate_immediately:
            self.current_placement_level = dict(selected)
            self.current_placement_level_id = str(selected["id"])

    def set_evaluation_level(self, active_steps: int) -> None:
        """Pin evaluation to a requested number of stairs."""

        maximum = int(self.staircase_config["step_count"])
        if active_steps < 1 or active_steps > maximum:
            raise ValueError(f"active_steps must be within 1..{maximum}")
        self.pending_active_step_count = int(active_steps)
        self.active_step_count = int(active_steps)
        self.current_goal_x_m = goal_x_for_active_steps(
            self.staircase_config,
            self.active_step_count,
        )
        if self.placement_reference_enabled:
            self.set_placement_curriculum_progress(1.0)
            self.current_placement_level = dict(self.pending_placement_level)
            self.current_placement_level_id = self.pending_placement_level_id

    def set_training_level(
        self,
        active_steps: int,
        *,
        reason: str,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        """Schedule a mastery-selected level for the next episode reset."""

        maximum = int(self.staircase_config["step_count"])
        if active_steps < 1 or active_steps > maximum:
            raise ValueError(f"active_steps must be within 1..{maximum}")
        if int(active_steps) == self.pending_active_step_count:
            return
        transition: dict[str, object] = {
            "active_steps": int(active_steps),
            "reason": str(reason),
        }
        if evidence:
            transition["evidence"] = dict(evidence)
        self.curriculum_transitions.append(transition)
        self.pending_active_step_count = int(active_steps)

    def _read_state(self) -> dict[str, np.ndarray | float]:
        state = super()._read_state()
        self.latest_walking_observation = np.asarray(
            state["observation"],
            dtype=np.float32,
        ).copy()
        self.latest_projected_gravity_xyz = np.asarray(
            state["imu_observation"],
            dtype=np.float64,
        )[3:6].copy()
        base_position = np.asarray(state["base_position"])
        if self.placement_reference_enabled:
            self.latest_placement_base_position_m = np.asarray(
                base_position,
                dtype=np.float64,
            ).copy()
            self.latest_placement_com_position_m = (
                self._sample_robot_com_position_m()
            )
        heading_error = _yaw_from_wxyz(state["base_orientation"])
        terrain_observation = None
        if self.vl53l5cx_sensor is not None:
            terrain_observation = self.vl53l5cx_sensor.observe(
                base_position_world_m=base_position,
                base_orientation_wxyz=state["base_orientation"],
                rng=self.np_random,
            )
        foot_progress_normalized = None
        next_target_one_hot = None
        if self.include_foot_progress_observation:
            foot_tips = self._sample_foot_tips()
            self.latest_foot_tips_m = foot_tips.copy()
            raw_foot_progress = foot_tread_progress(
                foot_tip_positions_m=foot_tips,
                highest_foot_steps=self.highest_foot_step,
                staircase=self.staircase_config,
                active_steps=self.active_step_count,
                approach_distance_m=float(
                    self.config.get("foot_tread_approach_distance_m", 0.20)
                ),
            )
            foot_progress_normalized = np.clip(
                raw_foot_progress / max(1, self.active_step_count),
                0.0,
                1.0,
            )
            target_index = next_foot_target_index(
                self.highest_foot_step,
                active_steps=self.active_step_count,
                sequence_indices=self.foot_placement_sequence_indices,
            )
            next_target_one_hot = np.zeros(len(LEGS), dtype=np.float32)
            if target_index is not None:
                next_target_one_hot[target_index] = 1.0
            else:
                next_target_one_hot[0] = 1.0
            self.next_foot_target_index = target_index
        state["observation"] = pack_stair_policy_observation(
            walking_observation=state["observation"],
            base_world_x_m=float(base_position[0]),
            # In COM-regulated placement, expose error in the same retained
            # balance-target frame used by control and reward. This preserves
            # the observation shape while removing a torso/whole-body mismatch.
            base_world_y_m=(
                float(self._placement_balance_target_error_xy_m()[1])
                if self.placement_reference_enabled
                and self.com_regulation_enabled
                and self.com_regulation_balance_point == "composite_com"
                else float(base_position[1])
            ),
            heading_error_rad=heading_error,
            goal_world_x_m=self.current_goal_x_m,
            staircase=self.staircase_config,
            include_navigation_observation=self.include_navigation_observation,
            include_foot_progress_observation=(
                self.include_foot_progress_observation
            ),
            foot_progress_normalized=foot_progress_normalized,
            next_foot_target_one_hot=next_target_one_hot,
            terrain_observation_values=terrain_observation,
        )
        if self.include_placement_reference_observation:
            foot_tips = self._sample_foot_tips()
            ground_loads, step_loads = self._sample_foot_contact_loads()
            self.latest_ground_normal_loads_n = ground_loads
            self.latest_step_normal_loads_n = step_loads
            placement_state = self._placement_state(
                self.episode_step * self.control_dt_s
            )
            target_world_x = self._placement_target_world_x_m()
            desired_world_x = float(
                self.placement_leg_start_foot_tips_m[
                    self.placement_swing_leg_index,
                    0,
                ]
            ) + float(placement_state["forward_fraction"]) * (
                target_world_x
                - float(
                    self.placement_leg_start_foot_tips_m[
                        self.placement_swing_leg_index,
                        0,
                    ]
                )
            )
            desired_world_z = float(
                self.placement_leg_start_foot_tips_m[
                    self.placement_swing_leg_index,
                    2,
                ]
            ) + float(placement_state["desired_lift_m"])
            swing_tip = foot_tips[self.placement_swing_leg_index]
            support_indices = list(self.placement_support_leg_indices)
            support_loads = ground_loads[support_indices] + np.sum(
                step_loads[support_indices, :],
                axis=1,
            )
            contact_threshold = float(
                self.placement_reference_config["contact_on_threshold_n"]
            )
            support_contact_fraction = float(
                np.mean(support_loads >= contact_threshold)
            )
            balance_point_xy = (
                self.latest_placement_com_position_m[:2]
                if self.com_regulation_enabled
                else base_position[:2]
            )
            support_margin = _support_triangle_signed_margin_m(
                balance_point_xy,
                foot_tips[list(self.placement_support_leg_indices), :2],
            )
            state["observation"] = pack_placement_reference_observation(
                stair_observation=state["observation"],
                phase_one_hot=placement_state["phase_one_hot"],
                desired_swing_height_m=desired_world_z,
                measured_swing_height_m=float(swing_tip[2]),
                swing_x_error_m=desired_world_x - float(swing_tip[0]),
                swing_z_error_m=desired_world_z - float(swing_tip[2]),
                tread_normal_load_n=float(
                    step_loads[self.placement_swing_leg_index, 0]
                ),
                support_contact_fraction=support_contact_fraction,
                support_margin_m=support_margin,
                maximum_support_slip_m=self.maximum_support_slip_m,
                staircase=self.staircase_config,
                contact_load_normalization_n=float(
                    self.placement_reference_config[
                        "contact_load_normalization_n"
                    ]
                ),
            )
            if self.include_support_regulation_observation:
                total_foot_loads = ground_loads + np.sum(step_loads, axis=1)
                com_target_error = balance_target_error_xy(
                    balance_position_xy_m=balance_point_xy,
                    target_position_xy_m=self._placement_balance_target_xy_m(),
                )
                state["observation"] = pack_support_regulation_observation(
                    stair_observation=state["observation"],
                    total_foot_normal_loads_n=total_foot_loads,
                    com_target_error_xy_m=com_target_error,
                    # Position-drive demand is sampled after state packing, so
                    # this is the causal one-control-step-lagged value.
                    requested_pd_effort_nm=self.latest_requested_pd_effort_nm,
                    effort_cap_nm=self.effort_cap_nm,
                    contact_load_normalization_n=float(
                        self.placement_reference_config[
                            "contact_load_normalization_n"
                        ]
                    ),
                )
        if np.asarray(state["observation"]).shape != (self.observation_size,):
            raise RuntimeError(
                "Runtime stair observation does not match its declared contract: "
                f"{np.asarray(state['observation']).shape} != "
                f"({self.observation_size},)"
            )
        state["heading_error_rad"] = heading_error
        return state

    def _reset_robot(self) -> None:
        reset_noise = float(self.config["reset_joint_noise_rad"])
        configured_offsets = self.config.get("reset_joint_offsets_rad")
        if configured_offsets is None:
            joint_noise = self.np_random.uniform(
                -reset_noise,
                reset_noise,
                size=12,
            ).astype(np.float32)
        else:
            joint_noise = np.asarray(
                configured_offsets,
                dtype=np.float32,
            )
            if joint_noise.shape != (12,) or not np.all(np.isfinite(joint_noise)):
                raise ValueError(
                    "reset_joint_offsets_rad must contain 12 finite values"
                )
        initial_positions = np.clip(
            self.nominal_positions + joint_noise,
            self.lower_limits + 1e-3,
            self.upper_limits - 1e-3,
        )
        x_range = tuple(float(value) for value in self.config["reset_start_x_range_m"])
        y_range = tuple(float(value) for value in self.config["reset_start_y_range_m"])
        yaw_range = tuple(
            math.radians(float(value))
            for value in self.config["reset_start_yaw_range_deg"]
        )
        if len(x_range) != 2 or len(y_range) != 2 or len(yaw_range) != 2:
            raise ValueError("reset position/yaw ranges need exactly two endpoints")
        x = float(self.np_random.uniform(*x_range))
        y = float(self.np_random.uniform(*y_range))
        yaw = float(self.np_random.uniform(*yaw_range))
        orientation_wxyz = [
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ]
        self.robot.set_world_poses(
            positions=[[x, y, float(self.config["reset_start_z_m"])]],
            orientations=[orientation_wxyz],
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
        self.previous_residual_action.fill(0.0)
        for _ in range(self.reset_settle_steps):
            self.robot.set_dof_position_targets(self.nominal_positions)
            self._update(self.physics_steps_per_control)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        self.active_step_count = self.pending_active_step_count
        if self.placement_reference_enabled:
            if self.pending_placement_level is None:
                raise RuntimeError("Placement curriculum has no pending level")
            self.current_placement_level = dict(self.pending_placement_level)
            self.current_placement_level_id = self.pending_placement_level_id
        self.current_goal_x_m = goal_x_for_active_steps(
            self.staircase_config,
            self.active_step_count,
        )
        self.maximum_foot_lift_m.fill(0.0)
        self.highest_foot_step.fill(0)
        self.maximum_foot_tread_progress.fill(0.0)
        self.maximum_placement_clearance_gate_hold_steps = 0
        self.next_foot_target_index = self.foot_placement_sequence_indices[0]
        self.placement_sequence_position = 0
        self._set_placement_swing_leg(self.placement_sequence_legs[0])
        self.placement_phase_start_step = 0
        self.placement_phase_elapsed_offset_s = 0.0
        self.completed_placement_legs = []
        self.completed_placement_joint_targets_by_leg = {}
        self.completed_placement_reference_by_leg = {}
        self.latest_reference_parameters_by_leg = {}
        self.latest_placement_base_position_m.fill(0.0)
        self.latest_placement_com_position_m.fill(0.0)
        self.placement_transfer_active = False
        self.placement_transfer_start_step = 0
        self.placement_transfer_gate_step_count = 0
        self.placement_transfer_pre_unload_gate_step_count = 0
        self.placement_transfer_unload_start_step = None
        self.placement_transfer_reference_by_leg = {}
        self.placement_leg_baseline_reference_by_leg = {}
        self.placement_leg_baseline_base_position_m.fill(0.0)
        self.placement_leg_baseline_balance_position_m.fill(0.0)
        self.placement_leg_baseline_lift_offset_m = 0.0
        self.placement_transfer_start_base_position_m.fill(0.0)
        self.placement_transfer_target_base_position_m.fill(0.0)
        self.placement_transfer_start_balance_position_m.fill(0.0)
        self.placement_transfer_target_balance_position_m.fill(0.0)
        self.completed_inter_leg_transfers = []
        self.last_completed_inter_leg_transfer_metrics = {}
        self.initial_placement_foot_tips_m.fill(0.0)
        self.placement_leg_start_foot_tips_m.fill(0.0)
        self.maximum_support_slip_m = 0.0
        self.maximum_support_slip_m_by_leg.fill(0.0)
        self.minimum_support_contact_fraction = 1.0
        self.minimum_placement_support_margin_m = float("inf")
        self.latest_support_margin_regulation_active = False
        self.latest_support_margin_requested_target_xy_m.fill(0.0)
        self.latest_support_margin_constrained_target_xy_m.fill(0.0)
        self.latest_support_margin_commanded_target_margin_m = 0.0
        self.maximum_support_margin_target_clip_m = 0.0
        self.latest_touchdown_load_lift_correction_m = 0.0
        self.maximum_touchdown_load_lift_correction_m = 0.0
        self.latest_placement_pitch_rear_correction_scale = 1.0
        self.maximum_swing_tread_normal_load_n = 0.0
        self.maximum_tread_normal_load_n_by_leg.fill(0.0)
        self.placement_tread_contact_sample_count = 0
        self.placement_active_sample_count = 0
        self.placement_reference_reach_clip_count = 0
        self.maximum_placement_reference_reach_excess_m = 0.0
        self.maximum_placement_desired_lift_m = 0.0
        self.maximum_swing_reference_tracking_error_rad = 0.0
        self.maximum_balance_lateral_deviation_m = 0.0
        self.latest_support_load_sharing_correction_m.fill(0.0)
        self.maximum_abs_support_load_sharing_correction_m = 0.0
        self.maximum_abs_support_load_sharing_correction_m_by_leg.fill(0.0)
        self.support_load_sharing_active_sample_count = 0
        self.support_load_sharing_saturated_sample_count = 0
        self._reset_joint_effort_telemetry()
        if self.vl53l5cx_sensor is not None:
            self.vl53l5cx_sensor.reset()
        observation, info = super().reset(seed=seed, options=options)
        self.previous_base_x_m = float(self.episode_origin[0])
        self.latest_placement_base_position_m = np.asarray(
            self.episode_origin,
            dtype=np.float64,
        ).copy()
        self.latest_placement_com_position_m = (
            self._sample_robot_com_position_m()
        )
        self.previous_base_z_m = float(self.episode_origin[2])
        self.previous_terrain_height_m = stair_height_at_x(
            self.previous_base_x_m,
            self.staircase_config,
        )
        initial_clearance = float(
            self.episode_origin[2] - self.previous_terrain_height_m
        )
        self.minimum_base_clearance_m = initial_clearance
        self.maximum_base_elevation_gain_m = 0.0
        self.maximum_terrain_height_m = self.previous_terrain_height_m
        self.highest_step_reached = stair_index_at_x(
            self.previous_base_x_m,
            self.staircase_config,
        )
        self.goal_hold_step_count = 0
        foot_tips = self._sample_foot_tips()
        self.initial_placement_foot_tips_m = foot_tips.copy()
        self.placement_leg_start_foot_tips_m = foot_tips.copy()
        self.initial_foot_bottom_z_m = foot_tips[:, 2].copy()
        self.maximum_foot_lift_m.fill(0.0)
        self.highest_foot_step.fill(0)
        self.maximum_foot_tread_progress = foot_tread_progress(
            foot_tip_positions_m=foot_tips,
            highest_foot_steps=self.highest_foot_step,
            staircase=self.staircase_config,
            active_steps=self.active_step_count,
            approach_distance_m=float(
                self.config.get("foot_tread_approach_distance_m", 0.20)
            ),
        )
        if self.placement_reference_enabled:
            ground_loads, step_loads = self._sample_foot_contact_loads()
            self.latest_ground_normal_loads_n = ground_loads
            self.latest_step_normal_loads_n = step_loads
            observation = np.asarray(self._read_state()["observation"]).copy()
        info.update(
            {
                "task_id": self.config["id"],
                "active_step_count": self.active_step_count,
                "goal_world_x_m": self.current_goal_x_m,
                "observation_fields": self.observation_fields,
                "physics_steps_per_control": self.physics_steps_per_control,
                "terrain_perception_mode": self.terrain_perception_mode,
                "terrain_sensor_metrics": (
                    self.vl53l5cx_sensor.metrics
                    if self.vl53l5cx_sensor is not None
                    else None
                ),
                "placement_reference_enabled": self.placement_reference_enabled,
                "placement_curriculum_level": self.current_placement_level_id,
            }
        )
        return observation, info

    def capture_placement_phase_snapshot(self) -> dict[str, object]:
        """Capture one verified placement or inter-leg-transfer training state."""

        if not self.placement_reference_enabled:
            raise RuntimeError("Placement snapshot requires placement reference mode")
        if not self.completed_placement_legs:
            raise RuntimeError("Placement snapshot has no completed precursor leg")

        base_positions, base_orientations = self.robot.get_world_poses()
        linear_velocities, angular_velocities = self.robot.get_velocities()

        def finite_copy(value, shape: tuple[int, ...], label: str) -> np.ndarray:
            if hasattr(value, "numpy"):
                value = value.numpy()
            result = np.asarray(value, dtype=np.float32).reshape(shape).copy()
            if not np.all(np.isfinite(result)):
                raise RuntimeError(f"Cannot snapshot non-finite {label}")
            return result

        return {
            "schema_version": 1,
            "placement_sequence_legs": tuple(self.placement_sequence_legs),
            "placement_sequence_position": self.placement_sequence_position,
            "placement_swing_leg": self.placement_swing_leg,
            "active_step_count": self.active_step_count,
            "placement_curriculum_level": self.current_placement_level_id,
            "current_placement_level": deepcopy(self.current_placement_level),
            "base_position_m": finite_copy(
                base_positions,
                (-1, 3),
                "base position",
            )[0],
            "base_orientation_wxyz": finite_copy(
                base_orientations,
                (-1, 4),
                "base orientation",
            )[0],
            "base_linear_velocity_m_s": finite_copy(
                linear_velocities,
                (-1, 3),
                "base linear velocity",
            )[0],
            "base_angular_velocity_rad_s": finite_copy(
                angular_velocities,
                (-1, 3),
                "base angular velocity",
            )[0],
            "joint_positions_rad": finite_copy(
                self.robot.get_dof_positions(),
                (12,),
                "joint positions",
            ),
            "joint_velocities_rad_s": finite_copy(
                self.robot.get_dof_velocities(),
                (12,),
                "joint velocities",
            ),
            "joint_position_targets_rad": self.previous_target.copy(),
            "previous_action": self.previous_action.copy(),
            "previous_residual_action": self.previous_residual_action.copy(),
            "completed_placement_legs": list(self.completed_placement_legs),
            "completed_placement_joint_targets_by_leg": deepcopy(
                self.completed_placement_joint_targets_by_leg
            ),
            "completed_placement_reference_by_leg": deepcopy(
                self.completed_placement_reference_by_leg
            ),
            "placement_leg_baseline_reference_by_leg": deepcopy(
                self.placement_leg_baseline_reference_by_leg
            ),
            "placement_leg_baseline_base_position_m": (
                self.placement_leg_baseline_base_position_m.copy()
            ),
            "placement_leg_baseline_balance_position_m": (
                self.placement_leg_baseline_balance_position_m.copy()
            ),
            "placement_transfer_target_base_position_m": (
                self.placement_transfer_target_base_position_m.copy()
            ),
            "placement_transfer_active": self.placement_transfer_active,
            "placement_transfer_reference_by_leg": deepcopy(
                self.placement_transfer_reference_by_leg
            ),
            "placement_transfer_start_base_position_m": (
                self.placement_transfer_start_base_position_m.copy()
            ),
            "placement_transfer_start_balance_position_m": (
                self.placement_transfer_start_balance_position_m.copy()
            ),
            "placement_transfer_target_balance_position_m": (
                self.placement_transfer_target_balance_position_m.copy()
            ),
            "placement_leg_baseline_lift_offset_m": (
                self.placement_leg_baseline_lift_offset_m
            ),
            "completed_inter_leg_transfers": list(
                self.completed_inter_leg_transfers
            ),
            "last_completed_inter_leg_transfer_metrics": deepcopy(
                self.last_completed_inter_leg_transfer_metrics
            ),
            "initial_placement_foot_tips_m": (
                self.initial_placement_foot_tips_m.copy()
            ),
            "initial_foot_bottom_z_m": self.initial_foot_bottom_z_m.copy(),
            "maximum_foot_lift_m": self.maximum_foot_lift_m.copy(),
            "highest_foot_step": self.highest_foot_step.copy(),
            "maximum_foot_tread_progress": (
                self.maximum_foot_tread_progress.copy()
            ),
        }

    def restore_placement_phase_snapshot(
        self,
        snapshot: Mapping[str, object],
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Restore a cached post-transfer state as a fresh training episode."""

        del options
        super(QuadrupedWalkEnv, self).reset(seed=seed)
        stored = dict(snapshot)
        if int(stored.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported placement phase snapshot schema")
        if tuple(stored["placement_sequence_legs"]) != tuple(
            self.placement_sequence_legs
        ):
            raise ValueError("Placement phase snapshot sequence does not match")
        if stored["placement_curriculum_level"] != self.pending_placement_level_id:
            raise ValueError("Placement curriculum changed; recache the phase state")
        if int(stored["active_step_count"]) != self.pending_active_step_count:
            raise ValueError("Stair curriculum changed; recache the phase state")

        base_position = np.asarray(
            stored["base_position_m"], dtype=np.float32
        ).reshape(3)
        base_orientation = np.asarray(
            stored["base_orientation_wxyz"], dtype=np.float32
        ).reshape(4)
        linear_velocity = np.asarray(
            stored["base_linear_velocity_m_s"], dtype=np.float32
        ).reshape(3)
        angular_velocity = np.asarray(
            stored["base_angular_velocity_rad_s"], dtype=np.float32
        ).reshape(3)
        joint_positions = np.asarray(
            stored["joint_positions_rad"], dtype=np.float32
        ).reshape(12)
        joint_velocities = np.asarray(
            stored["joint_velocities_rad_s"], dtype=np.float32
        ).reshape(12)
        joint_targets = np.asarray(
            stored["joint_position_targets_rad"], dtype=np.float32
        ).reshape(12)
        physical_values = (
            base_position,
            base_orientation,
            linear_velocity,
            angular_velocity,
            joint_positions,
            joint_velocities,
            joint_targets,
        )
        if any(not np.all(np.isfinite(value)) for value in physical_values):
            raise ValueError("Placement phase snapshot contains non-finite state")

        self.active_step_count = int(stored["active_step_count"])
        self.current_goal_x_m = goal_x_for_active_steps(
            self.staircase_config,
            self.active_step_count,
        )
        self.current_placement_level_id = str(
            stored["placement_curriculum_level"]
        )
        self.current_placement_level = deepcopy(stored["current_placement_level"])
        self.placement_sequence_position = int(
            stored["placement_sequence_position"]
        )
        self._set_placement_swing_leg(str(stored["placement_swing_leg"]))
        self.completed_placement_legs = list(stored["completed_placement_legs"])
        self.completed_placement_joint_targets_by_leg = deepcopy(
            stored["completed_placement_joint_targets_by_leg"]
        )
        self.completed_placement_reference_by_leg = deepcopy(
            stored["completed_placement_reference_by_leg"]
        )
        self.placement_leg_baseline_reference_by_leg = deepcopy(
            stored["placement_leg_baseline_reference_by_leg"]
        )
        self.placement_leg_baseline_base_position_m = np.asarray(
            stored["placement_leg_baseline_base_position_m"],
            dtype=np.float64,
        ).reshape(3).copy()
        self.placement_leg_baseline_balance_position_m = np.asarray(
            stored.get(
                "placement_leg_baseline_balance_position_m",
                stored["placement_leg_baseline_base_position_m"],
            ),
            dtype=np.float64,
        ).reshape(3).copy()
        self.placement_leg_baseline_lift_offset_m = float(
            stored["placement_leg_baseline_lift_offset_m"]
        )
        self.completed_inter_leg_transfers = list(
            stored["completed_inter_leg_transfers"]
        )
        self.last_completed_inter_leg_transfer_metrics = deepcopy(
            stored["last_completed_inter_leg_transfer_metrics"]
        )
        restored_transfer_active = bool(
            stored.get("placement_transfer_active", False)
        )
        transfer_start_base = np.asarray(
            stored.get(
                "placement_transfer_start_base_position_m",
                stored["placement_leg_baseline_base_position_m"],
            ),
            dtype=np.float64,
        ).reshape(3)
        transfer_target_base = np.asarray(
            stored.get(
                "placement_transfer_target_base_position_m",
                transfer_start_base,
            ),
            dtype=np.float64,
        ).reshape(3)
        transfer_base_delta = transfer_target_base - transfer_start_base
        transfer_start_balance = np.asarray(
            stored.get(
                "placement_transfer_start_balance_position_m",
                stored.get(
                    "placement_leg_baseline_balance_position_m",
                    stored["placement_leg_baseline_base_position_m"],
                ),
            ),
            dtype=np.float64,
        ).reshape(3)
        transfer_target_balance = np.asarray(
            stored.get(
                "placement_transfer_target_balance_position_m",
                transfer_start_balance,
            ),
            dtype=np.float64,
        ).reshape(3)
        transfer_balance_delta = (
            transfer_target_balance - transfer_start_balance
        )
        self.placement_transfer_active = restored_transfer_active
        self.placement_transfer_start_step = 0
        self.placement_transfer_gate_step_count = 0
        self.placement_transfer_pre_unload_gate_step_count = 0
        self.placement_transfer_unload_start_step = None
        self.placement_transfer_reference_by_leg = deepcopy(
            stored.get("placement_transfer_reference_by_leg", {})
        )
        self.placement_transfer_start_base_position_m = (
            transfer_start_base.copy()
        )
        self.placement_transfer_target_base_position_m = (
            transfer_target_base.copy()
        )
        self.placement_transfer_start_balance_position_m = (
            transfer_start_balance.copy()
        )
        self.placement_transfer_target_balance_position_m = (
            transfer_target_balance.copy()
        )
        self.placement_phase_start_step = 0
        self.placement_phase_elapsed_offset_s = 0.0
        self.goal_hold_step_count = 0

        self.robot.set_world_poses(
            positions=[base_position],
            orientations=[base_orientation],
        )
        if self.phase_snapshot_restore_zero_velocities:
            self.robot.set_velocities(
                linear_velocities=[np.zeros(3, dtype=np.float32)],
                angular_velocities=[np.zeros(3, dtype=np.float32)],
            )
        else:
            self.robot.set_velocities(
                linear_velocities=[linear_velocity],
                angular_velocities=[angular_velocity],
            )
        self.robot.set_dof_positions(joint_positions)
        self.robot.set_dof_velocities(
            np.zeros(12, dtype=np.float32)
            if self.phase_snapshot_restore_zero_velocities
            else joint_velocities
        )
        self.robot.set_dof_position_targets(joint_targets)
        self.previous_target = joint_targets.copy()
        self.previous_action = np.asarray(
            stored["previous_action"], dtype=np.float32
        ).reshape(12).copy()
        self.previous_residual_action = np.asarray(
            stored["previous_residual_action"], dtype=np.float32
        ).reshape(12).copy()
        settle_steps = max(1, self.phase_snapshot_restore_settle_control_steps)
        for _ in range(settle_steps):
            self.robot.set_dof_position_targets(joint_targets)
            self._update(self.physics_steps_per_control)

        base_state = super()._read_state()
        restored_base = np.asarray(base_state["base_position"], dtype=np.float32)
        foot_tips = self._sample_foot_tips()
        restored_com = self._sample_robot_com_position_m()
        if restored_transfer_active:
            # The zero-velocity settle can move the articulation a few
            # millimetres. Re-anchor the cached transfer at the settled state
            # while preserving the exact learned COM displacement target.
            self.placement_transfer_start_base_position_m = (
                restored_base.astype(np.float64).copy()
            )
            self.placement_transfer_target_base_position_m = (
                self.placement_transfer_start_base_position_m
                + transfer_base_delta
            )
            self.placement_transfer_start_balance_position_m = (
                restored_com.copy()
            )
            self.placement_transfer_target_balance_position_m = (
                self.placement_transfer_start_balance_position_m
                + transfer_balance_delta
            )
        if self.phase_snapshot_restore_settle_control_steps:
            self.placement_leg_baseline_reference_by_leg = (
                self._reference_parameters_from_joint_positions(
                    np.asarray(
                        base_state["joint_positions"],
                        dtype=np.float32,
                    )
                )
            )
            self.placement_leg_baseline_base_position_m = restored_base.astype(
                np.float64
            ).copy()
            self.placement_leg_baseline_balance_position_m = (
                restored_com.copy()
                if self.com_regulation_enabled
                and self.com_regulation_balance_point == "composite_com"
                else restored_base.astype(np.float64).copy()
            )
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_origin = restored_base.copy()
        self.minimum_height_m = float("inf")
        self.maximum_tilt_deg = 0.0
        self.previous_base_x_m = float(restored_base[0])
        self.previous_base_z_m = float(restored_base[2])
        self.previous_terrain_height_m = stair_height_at_x(
            self.previous_base_x_m,
            self.staircase_config,
        )
        self.maximum_base_elevation_gain_m = 0.0
        self.maximum_terrain_height_m = self.previous_terrain_height_m
        self.minimum_base_clearance_m = float(
            restored_base[2] - self.previous_terrain_height_m
        )
        self.highest_step_reached = stair_index_at_x(
            self.previous_base_x_m,
            self.staircase_config,
        )
        self.initial_placement_foot_tips_m = np.asarray(
            stored["initial_placement_foot_tips_m"],
            dtype=np.float32,
        ).reshape(len(LEGS), 3).copy()
        self.placement_leg_start_foot_tips_m = foot_tips.copy()
        self.initial_foot_bottom_z_m = np.asarray(
            stored["initial_foot_bottom_z_m"],
            dtype=np.float32,
        ).reshape(len(LEGS)).copy()
        self.maximum_foot_lift_m = np.asarray(
            stored["maximum_foot_lift_m"],
            dtype=np.float32,
        ).reshape(len(LEGS)).copy()
        self.highest_foot_step = np.asarray(
            stored["highest_foot_step"],
            dtype=np.int32,
        ).reshape(len(LEGS)).copy()
        self.maximum_foot_tread_progress = np.asarray(
            stored["maximum_foot_tread_progress"],
            dtype=np.float32,
        ).reshape(len(LEGS)).copy()
        self.next_foot_target_index = next_foot_target_index(
            self.highest_foot_step,
            active_steps=self.active_step_count,
            sequence_indices=self.foot_placement_sequence_indices,
        )
        self.latest_placement_base_position_m = restored_base.astype(
            np.float64
        ).copy()
        self.latest_placement_com_position_m = restored_com.copy()
        self.latest_support_margin_regulation_active = False
        self.latest_support_margin_requested_target_xy_m.fill(0.0)
        self.latest_support_margin_constrained_target_xy_m.fill(0.0)
        self.latest_support_margin_commanded_target_margin_m = 0.0
        self.maximum_support_margin_target_clip_m = 0.0
        self.latest_touchdown_load_lift_correction_m = 0.0
        self.maximum_touchdown_load_lift_correction_m = 0.0
        self.maximum_support_slip_m = 0.0
        self.maximum_support_slip_m_by_leg.fill(0.0)
        self.minimum_support_contact_fraction = 1.0
        self.minimum_placement_support_margin_m = float("inf")
        self.latest_placement_pitch_rear_correction_scale = 1.0
        self.maximum_swing_tread_normal_load_n = 0.0
        self.maximum_tread_normal_load_n_by_leg.fill(0.0)
        self.placement_tread_contact_sample_count = 0
        self.placement_active_sample_count = 0
        self.placement_reference_reach_clip_count = 0
        self.maximum_placement_reference_reach_excess_m = 0.0
        self._reset_joint_effort_telemetry()
        ground_loads, step_loads = self._sample_foot_contact_loads()
        self.latest_ground_normal_loads_n = ground_loads
        self.latest_step_normal_loads_n = step_loads
        contact_threshold = float(
            self.placement_reference_config["contact_on_threshold_n"]
        )
        completed_indices = [
            LEGS.index(leg) for leg in self.completed_placement_legs
        ]
        completed_tread_loads = (
            np.sum(step_loads[completed_indices], axis=1)
            if completed_indices
            else np.zeros(0, dtype=np.float32)
        )
        support_loads = ground_loads + np.sum(step_loads, axis=1)
        restored_support_loads = support_loads[
            list(self.placement_support_leg_indices)
        ]
        if (
            completed_indices
            and np.any(completed_tread_loads < contact_threshold)
        ) or np.any(restored_support_loads < contact_threshold):
            raise RuntimeError(
                "Placement phase snapshot did not restore force-backed support"
            )
        observation = np.asarray(self._read_state()["observation"]).copy()
        return observation, {
            "dof_names": tuple(self.dof_names),
            "task_id": self.config["id"],
            "active_step_count": self.active_step_count,
            "goal_world_x_m": self.current_goal_x_m,
            "observation_fields": self.observation_fields,
            "physics_steps_per_control": self.physics_steps_per_control,
            "terrain_perception_mode": self.terrain_perception_mode,
            "placement_reference_enabled": True,
            "placement_curriculum_level": self.current_placement_level_id,
            "reset_base_position_m": self.episode_origin.copy(),
            "placement_phase_snapshot_restored": True,
            "placement_phase_snapshot_settle_control_steps": (
                self.phase_snapshot_restore_settle_control_steps
            ),
            "placement_phase_snapshot_completed_tread_min_load_n": (
                float(np.min(completed_tread_loads))
                if completed_tread_loads.size
                else 0.0
            ),
            "placement_phase_snapshot_support_min_load_n": float(
                np.min(restored_support_loads)
            ),
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
        base_action = np.zeros(12, dtype=np.float32)
        placement_state: dict[str, object] | None = None
        reference_target: np.ndarray | None = None
        transfer_elapsed_seconds = 0.0
        placement_clearance_gate_active = False
        placement_clearance_gate_released_event = False
        placement_clearance_gate_measured_m = 0.0
        if self.placement_reference_enabled:
            if self.placement_transfer_active:
                transfer_elapsed_seconds = max(
                    0.0,
                    (
                        self.episode_step
                        + 1
                        - self.placement_transfer_start_step
                    )
                    * self.control_dt_s,
                )
                unload_elapsed_seconds: float | None = None
                if self.pre_unload_gate_hold_steps > 0:
                    unload_elapsed_seconds = (
                        0.0
                        if self.placement_transfer_unload_start_step is None
                        else max(
                            0.0,
                            (
                                self.episode_step
                                + 1
                                - self.placement_transfer_unload_start_step
                            )
                            * self.control_dt_s,
                        )
                    )
                active_transfer_config = (
                    self._active_inter_leg_transfer_config()
                )
                placement_state = inter_leg_transfer_state(
                    transfer_elapsed_seconds,
                    duration_seconds=float(
                        active_transfer_config["duration_seconds"]
                    ),
                    unload_duration_seconds=float(
                        active_transfer_config.get(
                            "unload_duration_seconds",
                            0.0,
                        )
                    ),
                    unload_elapsed_seconds=unload_elapsed_seconds,
                )
                reference_target = self._inter_leg_transfer_targets(
                    placement_state
                )
                residual_scale = float(
                    active_transfer_config.get(
                        "residual_action_scale",
                        self.inter_leg_transfer_config.get(
                            "residual_action_scale",
                            0.0,
                        ),
                    )
                )
            else:
                placement_state = self._placement_state(
                    (self.episode_step + 1) * self.control_dt_s
                )
                if (
                    self.advance_clearance_gate_enabled
                    and self.placement_swing_leg
                    in self.advance_clearance_gate_legs
                    and not self.placement_clearance_gate_released
                ):
                    current_foot_tips = self._sample_foot_tips()
                    placement_clearance_gate_measured_m = float(
                        current_foot_tips[
                            self.placement_swing_leg_index,
                            2,
                        ]
                        - self.initial_foot_bottom_z_m[
                            self.placement_swing_leg_index
                        ]
                    )
                    clearance_gate_state = (
                        placement_advance_clearance_gate_state(
                            candidate_phase=str(placement_state["phase"]),
                            measured_clearance_m=(
                                placement_clearance_gate_measured_m
                            ),
                            minimum_clearance_m=(
                                self.advance_clearance_gate_minimum_m
                            ),
                            held_steps=(
                                self.placement_clearance_gate_hold_step_count
                            ),
                            maximum_hold_steps=(
                                self.advance_clearance_gate_maximum_hold_steps
                            ),
                        )
                    )
                    if bool(clearance_gate_state["released"]):
                        self.placement_clearance_gate_released = True
                        placement_clearance_gate_released_event = True
                    elif bool(clearance_gate_state["hold_reference"]):
                        placement_clearance_gate_active = True
                        self.placement_clearance_gate_hold_step_count = int(
                            clearance_gate_state["held_steps"]
                        )
                        self.maximum_placement_clearance_gate_hold_steps = max(
                            self.maximum_placement_clearance_gate_hold_steps,
                            self.placement_clearance_gate_hold_step_count,
                        )
                        self.placement_clearance_gate_timeout = bool(
                            clearance_gate_state["timed_out"]
                        )
                        # Move the phase origin forward by one control tick so
                        # the reference remains at the completed lift apex.
                        self.placement_phase_start_step += 1
                        placement_state = self._placement_state(
                            (self.episode_step + 1) * self.control_dt_s
                        )
                reference_target = self._placement_reference_targets(
                    placement_state
                )
                residual_scale = 1.0
            prior_action = self.previous_action.copy()
            desired_target = np.clip(
                reference_target
                + residual_scale * self.action_scale * clipped_action,
                self.lower_limits + 1e-3,
                self.upper_limits - 1e-3,
            )
        elif self.residual_policy_enabled:
            if self.base_policy is None:
                raise RuntimeError("Residual base policy was not loaded")
            predicted, _ = self.base_policy.predict(
                self.latest_walking_observation,
                deterministic=True,
            )
            base_action = np.clip(
                np.asarray(predicted, dtype=np.float32).reshape(12),
                -1.0,
                1.0,
            )
            prior_action = self.previous_residual_action.copy()
            action_offset = (
                self.base_action_scale * base_action
                + self.action_scale * clipped_action
            )
        else:
            prior_action = self.previous_action.copy()
            action_offset = self.action_scale * clipped_action
        if not self.placement_reference_enabled:
            desired_target = np.clip(
                self.nominal_positions + action_offset,
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
        self._update(self.physics_steps_per_control)
        self.previous_action = (
            base_action.copy()
            if self.residual_policy_enabled
            else clipped_action.copy()
        )
        state = self._read_state()
        sampled_efforts = self._sample_joint_efforts_nm()
        effort_telemetry = joint_effort_telemetry_sample(
            target_joint_positions_rad=target,
            measured_joint_positions_rad=state["joint_positions"],
            joint_velocities_rad_s=state["joint_velocities"],
            drive_stiffness_nm_rad=self.drive_stiffness_nm_rad,
            drive_damping_nm_s_rad=self.drive_damping_nm_s_rad,
            effort_cap_nm=self.effort_cap_nm,
            reported_actuation_effort_nm=sampled_efforts.get(
                "reported_actuation_effort_nm"
            ),
            projected_joint_reaction_load_nm=sampled_efforts.get(
                "projected_joint_reaction_load_nm"
            ),
        )
        joint_tracking_error_rad = np.asarray(
            effort_telemetry["joint_tracking_error_rad"],
            dtype=np.float64,
        )
        self.maximum_abs_joint_tracking_error_rad_by_joint = np.maximum(
            self.maximum_abs_joint_tracking_error_rad_by_joint,
            np.abs(joint_tracking_error_rad),
        )
        requested_pd_effort_nm = np.asarray(
            effort_telemetry["requested_pd_effort_nm"],
            dtype=np.float64,
        )
        self.latest_requested_pd_effort_nm = requested_pd_effort_nm.copy()
        self.peak_abs_requested_pd_effort_nm_by_joint = np.maximum(
            self.peak_abs_requested_pd_effort_nm_by_joint,
            np.abs(requested_pd_effort_nm),
        )
        self.requested_pd_effort_sample_count += 1
        self.requested_pd_effort_95pct_cap_count += int(
            np.count_nonzero(
                np.abs(requested_pd_effort_nm)
                >= 0.95 * self.effort_cap_nm - 1e-6
            )
        )
        for label, peak_attribute, sample_count_attribute in (
            (
                "reported_actuation_effort_nm",
                "peak_abs_reported_actuation_effort_nm_by_joint",
                "reported_actuation_effort_sample_count",
            ),
            (
                "projected_joint_reaction_load_nm",
                "peak_abs_projected_joint_reaction_load_nm_by_joint",
                "projected_joint_reaction_load_sample_count",
            ),
        ):
            if label not in effort_telemetry:
                continue
            values = np.asarray(effort_telemetry[label], dtype=np.float64)
            setattr(
                self,
                peak_attribute,
                np.maximum(getattr(self, peak_attribute), np.abs(values)),
            )
            setattr(self, sample_count_attribute, getattr(self, sample_count_attribute) + 1)
        base_position = np.asarray(state["base_position"])
        self.latest_placement_base_position_m = np.asarray(
            base_position,
            dtype=np.float64,
        ).copy()
        base_x = float(base_position[0])
        base_y = float(base_position[1])
        terrain_height = stair_height_at_x(base_x, self.staircase_config)
        base_clearance = float(base_position[2] - terrain_height)
        foot_tips = self._sample_foot_tips()
        (
            foot_lift_progress,
            foot_placement_progress,
            foot_tread_progress_gain,
            next_maximum_foot_lift,
            next_highest_foot_step,
            current_foot_steps,
            next_maximum_foot_tread_progress,
        ) = self._foot_progress(foot_tips)
        imu_observation = np.asarray(state["imu_observation"])
        projected_gravity = imu_observation[3:6]
        upright_cosine = float(np.clip(-projected_gravity[2], -1.0, 1.0))
        placement_contact_now = False
        placement_contact_expected = False
        placement_swing_lift_m = 0.0
        placement_swing_height_error_m: float | None = None
        placement_support_contact_fraction = 0.0
        placement_support_margin = 0.0
        placement_current_support_slip_m = 0.0
        placement_swing_target_distance = 0.0
        placement_swing_tread_load = 0.0
        placement_transfer_gate_now = False
        placement_transfer_base_target_error_m = 0.0
        placement_transfer_completed_tread_min_load_n = 0.0
        placement_transfer_swing_total_load_n = 0.0
        placement_transfer_base_speed_m_s = 0.0
        placement_transfer_body_rate_rad_s = 0.0
        placement_transfer_gate_failures: tuple[str, ...] = ()
        placement_pre_unload_gate_now = False
        placement_pre_unload_gate_failures: tuple[str, ...] = ()
        placement_completion_gate_failures: tuple[str, ...] = ()
        placement_unload_started_event = False
        placement_balance_position = np.asarray(
            base_position,
            dtype=np.float64,
        )
        placement_balance_target_error_xy_m = np.asarray(
            [0.0, base_y],
            dtype=np.float64,
        )
        support_loads = np.zeros(0, dtype=np.float32)
        if self.placement_reference_enabled:
            if placement_state is None or self.current_placement_level is None:
                raise RuntimeError("Placement state was not initialized")
            self.maximum_placement_desired_lift_m = max(
                self.maximum_placement_desired_lift_m,
                float(placement_state["desired_lift_m"]),
            )
            if reference_target is not None:
                swing_indices = list(
                    self.dof_indices_by_leg[self.placement_swing_leg]
                )
                swing_tracking_error = np.max(
                    np.abs(
                        np.asarray(state["joint_positions"])[swing_indices]
                        - reference_target[swing_indices]
                    )
                )
                self.maximum_swing_reference_tracking_error_rad = max(
                    self.maximum_swing_reference_tracking_error_rad,
                    float(swing_tracking_error),
                )
            ground_loads = self.latest_ground_normal_loads_n
            step_loads = self.latest_step_normal_loads_n
            support_indices = list(self.placement_support_leg_indices)
            support_loads = ground_loads[support_indices] + np.sum(
                step_loads[support_indices, :],
                axis=1,
            )
            contact_threshold = float(
                self.placement_reference_config["contact_on_threshold_n"]
            )
            placement_support_contact_fraction = float(
                np.mean(support_loads >= contact_threshold)
            )
            balance_position = (
                self.latest_placement_com_position_m
                if self.com_regulation_enabled
                and self.com_regulation_balance_point == "composite_com"
                else base_position
            )
            placement_balance_position = np.asarray(
                balance_position,
                dtype=np.float64,
            )
            if (
                self.com_regulation_enabled
                and self.com_regulation_balance_point == "composite_com"
            ):
                placement_balance_target_error_xy_m = (
                    self._placement_balance_target_error_xy_m()
                )
            self.maximum_balance_lateral_deviation_m = max(
                self.maximum_balance_lateral_deviation_m,
                abs(float(placement_balance_position[1])),
            )
            placement_support_margin = _support_triangle_signed_margin_m(
                balance_position[:2],
                foot_tips[support_indices, :2],
            )
            support_slips = np.linalg.norm(
                foot_tips[support_indices, :2]
                - self.placement_leg_start_foot_tips_m[
                    support_indices,
                    :2,
                ],
                axis=1,
            )
            current_support_slip = float(np.max(support_slips))
            placement_current_support_slip_m = current_support_slip
            for support_index, slip_m in zip(
                support_indices,
                support_slips,
                strict=True,
            ):
                self.maximum_support_slip_m_by_leg[support_index] = max(
                    self.maximum_support_slip_m_by_leg[support_index],
                    float(slip_m),
                )
            self.maximum_support_slip_m = max(
                self.maximum_support_slip_m,
                current_support_slip,
            )
            self.minimum_support_contact_fraction = min(
                self.minimum_support_contact_fraction,
                placement_support_contact_fraction,
            )
            self.minimum_placement_support_margin_m = min(
                self.minimum_placement_support_margin_m,
                placement_support_margin,
            )
            swing_tip = foot_tips[self.placement_swing_leg_index]
            placement_swing_lift_m = float(
                swing_tip[2]
                - self.initial_foot_bottom_z_m[self.placement_swing_leg_index]
            )
            placement_swing_height_error_m = float(
                float(placement_state["desired_lift_m"])
                - placement_swing_lift_m
            )
            target_world_x = self._placement_target_world_x_m()
            target_world_z = float(self.staircase_config["rise_m"])
            placement_swing_target_distance = float(
                np.linalg.norm(
                    np.asarray(
                        [
                            float(swing_tip[0]) - target_world_x,
                            float(swing_tip[2]) - target_world_z,
                        ],
                        dtype=np.float64,
                    )
                )
            )
            placement_swing_tread_load = float(
                step_loads[self.placement_swing_leg_index, 0]
            )
            self.maximum_swing_tread_normal_load_n = max(
                self.maximum_swing_tread_normal_load_n,
                placement_swing_tread_load,
            )
            self.maximum_tread_normal_load_n_by_leg[
                self.placement_swing_leg_index
            ] = max(
                self.maximum_tread_normal_load_n_by_leg[
                    self.placement_swing_leg_index
                ],
                placement_swing_tread_load,
            )
            placement_success_mode = self._placement_success_mode()
            active_placement_level = self._active_placement_level()
            if placement_success_mode == "swing_lift_hold":
                placement_contact_expected = True
                placement_contact_now = placement_lift_hold_reached(
                    swing_tip_height_m=float(swing_tip[2]),
                    initial_swing_tip_height_m=float(
                        self.initial_foot_bottom_z_m[
                            self.placement_swing_leg_index
                        ]
                    ),
                    support_normal_loads_n=support_loads,
                    support_margin_m=placement_support_margin,
                    projected_gravity_xyz=projected_gravity,
                    minimum_lift_m=float(
                        active_placement_level.get(
                            "minimum_lift_m",
                            self.placement_reference_config[
                                "minimum_lift_m"
                            ],
                        )
                    ),
                    contact_on_threshold_n=contact_threshold,
                    minimum_support_margin_m=float(
                        active_placement_level.get(
                            "minimum_support_margin_m",
                            self.placement_reference_config[
                                "minimum_lift_support_margin_m"
                            ],
                        )
                    ),
                    minimum_upright_cosine=float(
                        self.placement_reference_config[
                            "minimum_success_upright_cosine"
                        ]
                    ),
                )
            elif placement_success_mode == "tread_contact":
                early_contact_after_clearance = bool(
                    active_placement_level.get(
                        "accept_early_tread_contact_after_clearance",
                        False,
                    )
                )
                placement_contact_expected = bool(
                    placement_state["contact_expected"]
                    or (
                        early_contact_after_clearance
                        and self.placement_clearance_gate_released
                        and str(placement_state["phase"]) == "advance"
                    )
                )
                placement_contact_now = bool(
                    placement_contact_expected
                    and placement_contact_reached(
                    swing_tip_position_m=swing_tip,
                    swing_tread_normal_load_n=placement_swing_tread_load,
                    support_ground_normal_loads_n=support_loads,
                    projected_gravity_xyz=projected_gravity,
                    staircase=self.staircase_config,
                    target_tread_fraction=float(
                        active_placement_level["target_tread_fraction"]
                    ),
                    target_x_tolerance_m=float(
                        self.placement_reference_config[
                            "target_x_tolerance_m"
                        ]
                    ),
                    target_z_tolerance_m=float(
                        self.placement_reference_config[
                            "target_z_tolerance_m"
                        ]
                    ),
                    contact_on_threshold_n=contact_threshold,
                    minimum_upright_cosine=float(
                        self.placement_reference_config[
                            "minimum_success_upright_cosine"
                        ]
                    ),
                    )
                )
                if (
                    placement_contact_now
                    and early_contact_after_clearance
                    and str(placement_state["phase"]) == "advance"
                    and self.placement_early_contact_hold_elapsed_s is None
                ):
                    # Stop moving a reference that has already produced a
                    # valid post-clearance tread landing. Live contact,
                    # support, and upright predicates must still remain true
                    # for the full configured success-hold duration.
                    self.placement_early_contact_hold_elapsed_s = float(
                        placement_state["elapsed_seconds"]
                    )
            else:
                raise ValueError(
                    "placement_reference.success_mode must be tread_contact "
                    "or swing_lift_hold"
                )
            if self.placement_transfer_active:
                active_transfer_config = (
                    self._active_inter_leg_transfer_config()
                )
                placement_transfer_base_target_error_m = float(
                    np.linalg.norm(
                        balance_position[:2]
                        - self.placement_transfer_target_balance_position_m[:2]
                    )
                )
                completed_indices = [
                    LEGS.index(leg) for leg in self.completed_placement_legs
                ]
                completed_tread_loaded = bool(
                    completed_indices
                    and np.all(
                        step_loads[completed_indices, 0]
                        >= contact_threshold
                    )
                )
                placement_transfer_completed_tread_min_load_n = float(
                    np.min(step_loads[completed_indices, 0])
                    if completed_indices
                    else 0.0
                )
                placement_transfer_swing_total_load_n = float(
                    ground_loads[self.placement_swing_leg_index]
                    + np.sum(
                        step_loads[self.placement_swing_leg_index, :]
                    )
                )
                placement_transfer_base_speed_m_s = float(
                    np.linalg.norm(
                        np.asarray(
                            state["body_linear_velocity"],
                            dtype=np.float64,
                        )[:2]
                    )
                )
                placement_transfer_body_rate_rad_s = float(
                    np.linalg.norm(imu_observation[:3])
                )
                transfer_gate_failures: list[str] = []
                if (
                    float(placement_state["transfer_fraction"])
                    < 1.0 - 1e-6
                ):
                    transfer_gate_failures.append("transfer_incomplete")
                if (
                    float(placement_state.get("unload_fraction", 1.0))
                    < 1.0 - 1e-6
                ):
                    transfer_gate_failures.append("swing_unload_incomplete")
                if placement_support_contact_fraction < 1.0:
                    transfer_gate_failures.append("support_contact_lost")
                if not completed_tread_loaded:
                    transfer_gate_failures.append("placed_tread_unloaded")
                swing_unload_lift_m = float(
                    active_transfer_config.get(
                        "swing_unload_lift_m",
                        0.0,
                    )
                )
                if swing_unload_lift_m > 0.0:
                    if placement_transfer_swing_total_load_n > float(
                        active_transfer_config[
                            "maximum_swing_unloaded_load_n"
                        ]
                    ):
                        transfer_gate_failures.append(
                            "next_swing_still_loaded"
                        )
                elif (
                    placement_transfer_swing_total_load_n
                    < contact_threshold
                ):
                    transfer_gate_failures.append("next_swing_unloaded")
                if placement_support_margin < float(
                    active_transfer_config[
                        "minimum_support_margin_m"
                    ]
                ):
                    transfer_gate_failures.append("support_margin_low")
                if placement_transfer_base_target_error_m > float(
                    active_transfer_config[
                        "base_target_tolerance_m"
                    ]
                ):
                    transfer_gate_failures.append("base_target_error_high")
                if placement_transfer_base_speed_m_s > float(
                    active_transfer_config[
                        "maximum_base_speed_m_s"
                    ]
                ):
                    transfer_gate_failures.append("base_not_settled")
                if placement_transfer_body_rate_rad_s > float(
                    active_transfer_config[
                        "maximum_body_rate_rad_s"
                    ]
                ):
                    transfer_gate_failures.append("body_rate_high")
                transfer_minimum_upright_cosine = float(
                    active_transfer_config.get(
                        "minimum_upright_cosine",
                        self.placement_reference_config[
                            "minimum_success_upright_cosine"
                        ],
                    )
                )
                if float(-projected_gravity[2]) < (
                    transfer_minimum_upright_cosine
                ):
                    transfer_gate_failures.append("body_not_upright")
                placement_transfer_gate_failures = tuple(
                    transfer_gate_failures
                )
                placement_transfer_gate_now = not (
                    placement_transfer_gate_failures
                )
                if (
                    self.pre_unload_gate_hold_steps > 0
                    and self.placement_transfer_unload_start_step is None
                ):
                    placement_pre_unload_gate_failures = (
                        inter_leg_pre_unload_gate_failures(
                            transfer_fraction=float(
                                placement_state["transfer_fraction"]
                            ),
                            support_contact_fraction=(
                                placement_support_contact_fraction
                            ),
                            completed_tread_loaded=completed_tread_loaded,
                            next_swing_total_load_n=(
                                placement_transfer_swing_total_load_n
                            ),
                            minimum_next_swing_preload_n=(
                                float(
                                    active_transfer_config.get(
                                        "minimum_next_swing_preload_n",
                                        self.minimum_next_swing_preload_n,
                                    )
                                )
                            ),
                            support_margin_m=placement_support_margin,
                            minimum_support_margin_m=float(
                                active_transfer_config[
                                    "minimum_support_margin_m"
                                ]
                            ),
                            balance_target_error_m=(
                                placement_transfer_base_target_error_m
                            ),
                            maximum_balance_target_error_m=float(
                                active_transfer_config[
                                    "base_target_tolerance_m"
                                ]
                            ),
                            base_speed_m_s=placement_transfer_base_speed_m_s,
                            maximum_base_speed_m_s=float(
                                active_transfer_config[
                                    "maximum_base_speed_m_s"
                                ]
                            ),
                            body_rate_rad_s=(
                                placement_transfer_body_rate_rad_s
                            ),
                            maximum_body_rate_rad_s=float(
                                active_transfer_config[
                                    "maximum_body_rate_rad_s"
                                ]
                            ),
                            upright_cosine=upright_cosine,
                            minimum_upright_cosine=(
                                transfer_minimum_upright_cosine
                            ),
                        )
                    )
                    placement_pre_unload_gate_now = not (
                        placement_pre_unload_gate_failures
                    )
            if str(placement_state["phase"]) != "weight_shift":
                self.placement_active_sample_count += 1
                self.placement_tread_contact_sample_count += int(
                    placement_contact_now
                )
        failure_reasons = list(
            stair_failure_reasons(
                base_clearance_m=base_clearance,
                lateral_position_m=(
                    float(placement_balance_position[1])
                    if self.placement_reference_enabled
                    and self.com_regulation_enabled
                    else base_y
                ),
                world_x_m=base_x,
                projected_gravity_xyz=projected_gravity,
                minimum_base_clearance_m=float(
                    self.termination_config["minimum_base_clearance_m"]
                ),
                minimum_upright_cosine=float(
                    self.termination_config["minimum_upright_cosine"]
                ),
                maximum_lateral_deviation_m=float(
                    self.termination_config["maximum_lateral_deviation_m"]
                )
                + float(
                    self.termination_config.get(
                        "lateral_deviation_tolerance_m",
                        0.0,
                    )
                ),
                minimum_world_x_m=float(
                    self.termination_config["minimum_world_x_m"]
                ),
                support_slip_m=self.maximum_support_slip_m,
                maximum_support_slip_m=(
                    float(self.termination_config["maximum_support_slip_m"])
                    if "maximum_support_slip_m" in self.termination_config
                    else None
                ),
            )
        )
        stall_config = dict(self.config.get("stall_termination", {}))
        if bool(stall_config.get("enabled", False)):
            stall_step = int(
                round(
                    float(stall_config["after_seconds"])
                    * self.control_hz
                )
            )
            forward_displacement = float(base_x - self.episode_origin[0])
            if self.episode_step + 1 >= stall_step:
                if forward_displacement < float(
                    stall_config["minimum_forward_progress_m"]
                ):
                    failure_reasons.append("no_forward_progress")
                minimum_tread_progress = float(
                    stall_config.get(
                        "minimum_any_foot_tread_progress",
                        0.0,
                    )
                )
                if (
                    float(np.max(next_maximum_foot_tread_progress))
                    < minimum_tread_progress
                ):
                    failure_reasons.append("no_foot_tread_progress")
        if (
            self.placement_transfer_active
            and transfer_elapsed_seconds
            >= float(
                self._active_inter_leg_transfer_config()["maximum_seconds"]
            )
            and not placement_transfer_gate_now
        ):
            failure_reasons.append("body_transfer_failed")
        if (
            self.placement_reference_enabled
            and not self.placement_transfer_active
            and self.placement_clearance_gate_timeout
        ):
            failure_reasons.append("swing_clearance_timeout")
        failure_reasons = tuple(failure_reasons)
        failed = bool(failure_reasons)
        minimum_success_elevation = (
            self.active_step_count
            * float(self.staircase_config["rise_m"])
            * float(
                self.config.get(
                    "success_minimum_base_elevation_fraction",
                    0.0,
                )
            )
        )
        current_base_elevation = float(
            base_position[2] - self.episode_origin[2]
        )
        required_feet_on_goal_tread = int(
            self.config.get("success_required_feet_on_goal_tread", 0)
        )
        placement_transfer_was_active = self.placement_transfer_active
        placement_transfer_completed_event: str | None = None
        if self.placement_reference_enabled and placement_transfer_was_active:
            if (
                self.pre_unload_gate_hold_steps > 0
                and self.placement_transfer_unload_start_step is None
            ):
                if not failed and placement_pre_unload_gate_now:
                    self.placement_transfer_pre_unload_gate_step_count += 1
                else:
                    self.placement_transfer_pre_unload_gate_step_count = 0
                if (
                    not failed
                    and self.placement_transfer_pre_unload_gate_step_count
                    >= self.pre_unload_gate_hold_steps
                ):
                    self.placement_transfer_unload_start_step = (
                        self.episode_step + 1
                    )
                    placement_unload_started_event = True
            if not failed and placement_transfer_gate_now:
                self.placement_transfer_gate_step_count += 1
            else:
                self.placement_transfer_gate_step_count = 0
            transfer_gate_hold_steps = max(
                1,
                int(
                    round(
                        float(
                            self.inter_leg_transfer_config[
                                "gate_hold_seconds"
                            ]
                        )
                        * self.control_hz
                    )
                ),
            )
            if (
                not failed
                and self.placement_transfer_gate_step_count
                >= transfer_gate_hold_steps
            ):
                previous_leg = self.completed_placement_legs[-1]
                placement_transfer_completed_event = (
                    f"{previous_leg}->{self.placement_swing_leg}"
                )
                self.last_completed_inter_leg_transfer_metrics = {
                    "transition": placement_transfer_completed_event,
                    "support_margin_m": placement_support_margin,
                    "base_target_error_m": (
                        placement_transfer_base_target_error_m
                    ),
                    "completed_tread_min_load_n": (
                        placement_transfer_completed_tread_min_load_n
                    ),
                    "next_swing_total_load_n": (
                        placement_transfer_swing_total_load_n
                    ),
                    "base_speed_m_s": placement_transfer_base_speed_m_s,
                    "body_rate_rad_s": placement_transfer_body_rate_rad_s,
                    "body_tilt_deg": self.maximum_tilt_deg,
                    "pre_unload_gate_enabled": (
                        self.pre_unload_gate_hold_steps > 0
                    ),
                    "pre_unload_gate_hold_duration_s": (
                        self.placement_transfer_pre_unload_gate_step_count
                        / self.control_hz
                    ),
                    "unload_start_elapsed_s": (
                        (
                            self.placement_transfer_unload_start_step
                            - self.placement_transfer_start_step
                        )
                        * self.control_dt_s
                        if self.placement_transfer_unload_start_step is not None
                        else None
                    ),
                    "balance_position_m": (
                        placement_balance_position.tolist()
                    ),
                    "balance_target_position_m": (
                        self.placement_transfer_target_balance_position_m.tolist()
                    ),
                }
                self._complete_inter_leg_transfer(
                    base_position_m=base_position,
                    com_position_m=self.latest_placement_com_position_m,
                    foot_tips_m=foot_tips,
                    joint_positions_rad=np.asarray(
                        state["joint_positions"],
                        dtype=np.float32,
                    ),
                )
            self.goal_hold_step_count = 0
        elif self.placement_reference_enabled:
            if not failed and placement_contact_now:
                self.goal_hold_step_count += 1
            else:
                self.goal_hold_step_count = 0
        elif not failed and stair_goal_reached(
            base_world_x_m=base_x,
            base_elevation_gain_m=current_base_elevation,
            goal_world_x_m=self.current_goal_x_m,
            minimum_base_elevation_gain_m=minimum_success_elevation,
            current_foot_steps=current_foot_steps,
            active_steps=self.active_step_count,
            required_feet_on_goal_tread=required_feet_on_goal_tread,
        ):
            self.goal_hold_step_count += 1
        else:
            self.goal_hold_step_count = 0
        placement_hold_steps = self.success_hold_steps
        if self.current_placement_level is not None:
            placement_success_mode = self._placement_success_mode()
            active_placement_level = self._active_placement_level()
            hold_key = (
                "lift_hold_seconds"
                if placement_success_mode == "swing_lift_hold"
                else "contact_hold_seconds"
            )
            placement_hold_steps = int(
                round(
                    float(
                        active_placement_level.get(
                            hold_key,
                            self.config["success_hold_seconds"],
                        )
                    )
                    * self.control_hz
                )
            )
        succeeded = bool(
            not placement_transfer_was_active
            and self.goal_hold_step_count >= placement_hold_steps
        )
        if succeeded and self.placement_reference_enabled:
            completion_settle_gate = dict(
                self._active_placement_level().get(
                    "completion_settle_gate",
                    {},
                )
            )
            if bool(completion_settle_gate.get("enabled", False)):
                placement_completion_gate_failures = (
                    placement_completion_settle_gate_failures(
                        base_linear_velocity_xyz_m_s=state[
                            "body_linear_velocity"
                        ],
                        body_angular_velocity_xyz_rad_s=imu_observation[:3],
                        upright_cosine=upright_cosine,
                        maximum_base_speed_m_s=float(
                            completion_settle_gate[
                                "maximum_base_speed_m_s"
                            ]
                        ),
                        maximum_body_rate_rad_s=float(
                            completion_settle_gate[
                                "maximum_body_rate_rad_s"
                            ]
                        ),
                        minimum_upright_cosine=float(
                            completion_settle_gate.get(
                                "minimum_upright_cosine",
                                self.placement_reference_config[
                                    "minimum_success_upright_cosine"
                                ],
                            )
                        ),
                    )
                )
                succeeded = not placement_completion_gate_failures
        placement_leg_completed_event: str | None = None
        if self.placement_reference_enabled and succeeded:
            placement_leg_completed_event = self.placement_swing_leg
            if self.placement_swing_leg not in self.completed_placement_legs:
                self.completed_placement_legs.append(self.placement_swing_leg)
            indices = self.dof_indices_by_leg[self.placement_swing_leg]
            self.completed_placement_joint_targets_by_leg[
                self.placement_swing_leg
            ] = target[list(indices)].copy()
            held_reference = dict(
                self.latest_reference_parameters_by_leg[
                    self.placement_swing_leg
                ]
            )
            held_reference["base_position_m"] = base_position.tolist()
            self.completed_placement_reference_by_leg[
                self.placement_swing_leg
            ] = held_reference
            if self.placement_sequence_position + 1 < len(
                self.placement_sequence_legs
            ):
                self.placement_sequence_position += 1
                self._set_placement_swing_leg(
                    self.placement_sequence_legs[
                        self.placement_sequence_position
                    ]
                )
                if self.inter_leg_transfer_enabled:
                    self._begin_inter_leg_transfer(
                        base_position_m=base_position,
                        com_position_m=self.latest_placement_com_position_m,
                        foot_tips_m=foot_tips,
                        joint_positions_rad=np.asarray(
                            state["joint_positions"],
                            dtype=np.float32,
                        ),
                    )
                else:
                    self.placement_phase_start_step = self.episode_step + 1
                    self.placement_phase_elapsed_offset_s = 0.0
                    self.placement_leg_start_foot_tips_m = foot_tips.copy()
                self.goal_hold_step_count = 0
                succeeded = False
        terminated = failed or succeeded

        self.episode_step += 1
        truncated = self.episode_step >= self.max_episode_steps
        forward_progress = base_x - self.previous_base_x_m
        base_height_gain = float(base_position[2] - self.previous_base_z_m)
        terrain_height_gain = terrain_height - self.previous_terrain_height_m
        normalized_joint_velocity = (
            np.asarray(state["joint_velocities"]) / self.max_velocities
        )
        reward_terms = stair_reward_terms(
            command_velocity_xyz=self.command_velocity,
            body_linear_velocity_xyz=state["body_linear_velocity"],
            body_angular_velocity_xyz=imu_observation[:3],
            projected_gravity_xyz=projected_gravity,
            base_clearance_m=base_clearance,
            lateral_position_m=float(placement_balance_target_error_xy_m[1]),
            forward_progress_m=forward_progress,
            base_height_gain_m=base_height_gain,
            terrain_height_gain_m=terrain_height_gain,
            heading_error_rad=float(state["heading_error_rad"]),
            joint_velocities_normalized=normalized_joint_velocity,
            action=clipped_action,
            previous_action=prior_action,
            failed=failed,
            succeeded=succeeded,
            reward_config=self.reward_config,
            foot_lift_progress_m=(
                0.0 if failed else foot_lift_progress
            ),
            foot_step_placement_progress=(
                0 if failed else foot_placement_progress
            ),
            foot_tread_progress=(
                0.0 if failed else foot_tread_progress_gain
            ),
            foot_tread_support_count=(
                0 if failed else int(np.count_nonzero(current_foot_steps))
            ),
            swing_target_distance_m=(
                0.0 if failed else placement_swing_target_distance
            ),
            tread_contact_reached=(
                False if failed else placement_contact_now
            ),
            support_contact_fraction=(
                0.0 if failed else placement_support_contact_fraction
            ),
            support_slip_m=(
                0.0 if failed else placement_current_support_slip_m
            ),
            support_margin_m=(
                0.0 if failed else placement_support_margin
            ),
            balance_target_error_xy_m=(
                np.zeros(2, dtype=np.float64)
                if failed
                else placement_balance_target_error_xy_m
            ),
            support_normal_loads_n=(
                np.zeros(0, dtype=np.float32) if failed else support_loads
            ),
            requested_pd_effort_nm=(
                np.zeros(12, dtype=np.float64)
                if failed
                else requested_pd_effort_nm
            ),
            effort_cap_nm=self.effort_cap_nm,
            contact_load_normalization_n=float(
                self.placement_reference_config.get(
                    "contact_load_normalization_n",
                    50.0,
                )
            ),
            swing_height_error_m=placement_swing_height_error_m,
            clearance_gate_deficit_m=(
                max(
                    0.0,
                    self.advance_clearance_gate_minimum_m
                    - placement_swing_lift_m,
                )
                if placement_clearance_gate_active
                else 0.0
            ),
        )
        reward = float(reward_terms["total"])
        self.episode_return += reward
        self.minimum_base_clearance_m = min(
            self.minimum_base_clearance_m,
            base_clearance,
        )
        tilt_deg = float(np.degrees(np.arccos(upright_cosine)))
        self.maximum_tilt_deg = max(self.maximum_tilt_deg, tilt_deg)
        self.highest_step_reached = max(
            self.highest_step_reached,
            stair_index_at_x(base_x, self.staircase_config),
        )
        self.maximum_base_elevation_gain_m = max(
            self.maximum_base_elevation_gain_m,
            float(base_position[2] - self.episode_origin[2]),
        )
        self.maximum_terrain_height_m = max(
            self.maximum_terrain_height_m,
            terrain_height,
        )
        displacement = base_position - self.episode_origin
        info: dict[str, object] = {
            "reward_terms": reward_terms,
            "base_position_m": base_position.copy(),
            "body_linear_velocity_m_s": np.asarray(
                state["body_linear_velocity"]
            ).copy(),
            "failure_reasons": failure_reasons,
            "succeeded": succeeded,
            "active_step_count": self.active_step_count,
            "highest_step_reached": self.highest_step_reached,
            "terrain_height_m": terrain_height,
            "base_clearance_m": base_clearance,
            "goal_world_x_m": self.current_goal_x_m,
            "heading_error_rad": float(state["heading_error_rad"]),
            "maximum_base_elevation_gain_m": self.maximum_base_elevation_gain_m,
            "foot_tip_positions_m": foot_tips.copy(),
            "maximum_foot_lift_m_by_leg": dict(
                zip(LEGS, next_maximum_foot_lift.tolist(), strict=True)
            ),
            "highest_foot_step_by_leg": dict(
                zip(LEGS, next_highest_foot_step.tolist(), strict=True)
            ),
            "current_foot_step_by_leg": dict(
                zip(LEGS, current_foot_steps.tolist(), strict=True)
            ),
            "next_foot_target": (
                None
                if self.next_foot_target_index is None
                else LEGS[self.next_foot_target_index]
            ),
            "maximum_foot_tread_progress_by_leg": dict(
                zip(
                    LEGS,
                    next_maximum_foot_tread_progress.tolist(),
                    strict=True,
                )
            ),
            "target_joint_positions_rad": target.copy(),
            "joint_tracking_error_rad": joint_tracking_error_rad.copy(),
            "requested_pd_effort_nm": requested_pd_effort_nm.copy(),
            "capped_pd_effort_nm": np.asarray(
                effort_telemetry["capped_pd_effort_nm"],
                dtype=np.float64,
            ).copy(),
            "reported_actuation_effort_nm": effort_telemetry.get(
                "reported_actuation_effort_nm"
            ),
            "projected_joint_reaction_load_nm": effort_telemetry.get(
                "projected_joint_reaction_load_nm"
            ),
            "joint_effort_metrics": self._joint_effort_metrics(),
            "base_policy_action": base_action.copy(),
            "residual_policy_action": clipped_action.copy(),
            "terrain_perception_mode": self.terrain_perception_mode,
            "terrain_sensor_metrics": (
                self.vl53l5cx_sensor.metrics
                if self.vl53l5cx_sensor is not None
                else None
            ),
            "placement_phase": (
                None if placement_state is None else placement_state["phase"]
            ),
            "placement_desired_lift_m": (
                0.0
                if placement_state is None
                else float(placement_state["desired_lift_m"])
            ),
            "placement_leg_baseline_lift_offset_m": (
                self.placement_leg_baseline_lift_offset_m
            ),
            "placement_swing_leg": self.placement_swing_leg,
            "placement_swing_reference_joint_positions_rad": (
                None
                if reference_target is None
                else reference_target[
                    list(self.dof_indices_by_leg[self.placement_swing_leg])
                ].copy()
            ),
            "placement_swing_actual_joint_positions_rad": np.asarray(
                state["joint_positions"],
                dtype=np.float32,
            )[list(self.dof_indices_by_leg[self.placement_swing_leg])].copy(),
            "placement_leg_completed_event": placement_leg_completed_event,
            "placement_transfer_completed_event": (
                placement_transfer_completed_event
            ),
            "completed_placement_legs": list(self.completed_placement_legs),
            "completed_inter_leg_transfers": list(
                self.completed_inter_leg_transfers
            ),
            "last_completed_inter_leg_transfer_metrics": dict(
                self.last_completed_inter_leg_transfer_metrics
            ),
            "placement_curriculum_level": self.current_placement_level_id,
            "placement_contact_now": placement_contact_now,
            "placement_contact_expected": placement_contact_expected,
            "placement_early_contact_hold_elapsed_s": (
                self.placement_early_contact_hold_elapsed_s
            ),
            "placement_swing_lift_m": placement_swing_lift_m,
            "placement_upright_cosine": upright_cosine,
            "placement_goal_hold_step_count": self.goal_hold_step_count,
            "placement_completion_gate_failures": (
                placement_completion_gate_failures
            ),
            "placement_success_mode": (
                self._placement_success_mode()
                if self.placement_reference_enabled
                else None
            ),
            "swing_tread_normal_load_n": placement_swing_tread_load,
            "ground_normal_load_n_by_leg": dict(
                zip(
                    LEGS,
                    self.latest_ground_normal_loads_n.tolist(),
                    strict=True,
                )
            ),
            "placement_support_contact_fraction": (
                placement_support_contact_fraction
            ),
            "placement_support_margin_m": placement_support_margin,
            "support_margin_regulation_active": (
                self.latest_support_margin_regulation_active
            ),
            "support_margin_requested_target_xy_m": (
                self.latest_support_margin_requested_target_xy_m.copy()
            ),
            "support_margin_constrained_target_xy_m": (
                self.latest_support_margin_constrained_target_xy_m.copy()
            ),
            "support_margin_commanded_target_margin_m": (
                self.latest_support_margin_commanded_target_margin_m
            ),
            "maximum_support_margin_target_clip_m": (
                self.maximum_support_margin_target_clip_m
            ),
            "touchdown_load_lift_correction_m": (
                self.latest_touchdown_load_lift_correction_m
            ),
            "maximum_touchdown_load_lift_correction_m": (
                self.maximum_touchdown_load_lift_correction_m
            ),
            "placement_pitch_rear_correction_scale": (
                self.latest_placement_pitch_rear_correction_scale
            ),
            "placement_transfer_active": self.placement_transfer_active,
            "placement_transfer_fraction": (
                float(placement_state.get("transfer_fraction", 0.0))
                if placement_state is not None
                else 0.0
            ),
            "placement_transfer_unload_fraction": (
                float(placement_state.get("unload_fraction", 0.0))
                if placement_state is not None
                else 0.0
            ),
            "placement_transfer_stage": (
                str(placement_state.get("transfer_stage", "inactive"))
                if placement_state is not None
                else "inactive"
            ),
            "placement_transfer_gate_now": placement_transfer_gate_now,
            "placement_transfer_gate_failures": (
                placement_transfer_gate_failures
            ),
            "placement_pre_unload_gate_now": placement_pre_unload_gate_now,
            "placement_pre_unload_gate_failures": (
                placement_pre_unload_gate_failures
            ),
            "placement_pre_unload_gate_step_count": (
                self.placement_transfer_pre_unload_gate_step_count
            ),
            "placement_pre_unload_gate_required_steps": (
                self.pre_unload_gate_hold_steps
            ),
            "placement_unload_started_event": placement_unload_started_event,
            "placement_clearance_gate_enabled": (
                self.advance_clearance_gate_enabled
                and self.placement_swing_leg
                in self.advance_clearance_gate_legs
            ),
            "placement_clearance_gate_active": (
                placement_clearance_gate_active
            ),
            "placement_clearance_gate_released": (
                self.placement_clearance_gate_released
            ),
            "placement_clearance_gate_released_event": (
                placement_clearance_gate_released_event
            ),
            "placement_clearance_gate_measured_m": (
                placement_clearance_gate_measured_m
            ),
            "placement_clearance_gate_minimum_m": (
                self.advance_clearance_gate_minimum_m
            ),
            "placement_clearance_gate_hold_step_count": (
                self.placement_clearance_gate_hold_step_count
            ),
            "placement_clearance_gate_maximum_hold_steps": (
                self.advance_clearance_gate_maximum_hold_steps
            ),
            "placement_clearance_gate_timeout": (
                self.placement_clearance_gate_timeout
            ),
            "placement_transfer_base_target_error_m": (
                placement_transfer_base_target_error_m
            ),
            "placement_transfer_completed_tread_min_load_n": (
                placement_transfer_completed_tread_min_load_n
            ),
            "placement_transfer_swing_total_load_n": (
                placement_transfer_swing_total_load_n
            ),
            "placement_transfer_base_speed_m_s": (
                placement_transfer_base_speed_m_s
            ),
            "placement_transfer_body_rate_rad_s": (
                placement_transfer_body_rate_rad_s
            ),
            "placement_transfer_target_base_position_m": (
                self.placement_transfer_target_base_position_m.copy()
                if self.inter_leg_transfer_enabled
                else None
            ),
            "placement_balance_point": (
                self.com_regulation_balance_point
                if self.com_regulation_enabled
                else "base_origin"
            ),
            "placement_balance_position_m": (
                placement_balance_position.copy()
            ),
            "placement_balance_target_error_xy_m": (
                placement_balance_target_error_xy_m.copy()
            ),
            "support_load_sharing_vertical_correction_m_by_leg": dict(
                zip(
                    LEGS,
                    self.latest_support_load_sharing_correction_m.tolist(),
                    strict=True,
                )
            ),
            "maximum_abs_support_load_sharing_correction_m": (
                self.maximum_abs_support_load_sharing_correction_m
            ),
            "maximum_abs_support_load_sharing_correction_m_by_leg": dict(
                zip(
                    LEGS,
                    self.maximum_abs_support_load_sharing_correction_m_by_leg.tolist(),
                    strict=True,
                )
            ),
            "support_load_sharing_saturated_sample_fraction": (
                self.support_load_sharing_saturated_sample_count
                / self.support_load_sharing_active_sample_count
                if self.support_load_sharing_active_sample_count
                else 0.0
            ),
            "placement_com_position_m": (
                self.latest_placement_com_position_m.copy()
                if self.placement_reference_enabled
                else None
            ),
            "placement_transfer_target_balance_position_m": (
                self.placement_transfer_target_balance_position_m.copy()
                if self.inter_leg_transfer_enabled
                else None
            ),
            "maximum_support_slip_m": self.maximum_support_slip_m,
            "maximum_support_slip_m_by_leg": dict(
                zip(
                    LEGS,
                    self.maximum_support_slip_m_by_leg.tolist(),
                    strict=True,
                )
            ),
            "placement_reference_reach_clip_count": (
                self.placement_reference_reach_clip_count
            ),
            "maximum_placement_reference_reach_excess_m": (
                self.maximum_placement_reference_reach_excess_m
            ),
            "maximum_placement_desired_lift_m": (
                self.maximum_placement_desired_lift_m
            ),
            "maximum_swing_reference_tracking_error_rad": (
                self.maximum_swing_reference_tracking_error_rad
            ),
            "maximum_balance_lateral_deviation_m": (
                self.maximum_balance_lateral_deviation_m
            ),
            "reference_joint_positions_rad": (
                None if reference_target is None else reference_target.copy()
            ),
        }
        if terminated or truncated:
            episode_metrics = {
                "return": self.episode_return,
                "length_steps": self.episode_step,
                "duration_s": self.episode_step / self.control_hz,
                "active_step_count": self.active_step_count,
                "highest_step_reached": self.highest_step_reached,
                "stairs_completed": succeeded,
                "placement_completed": (
                    succeeded if self.placement_reference_enabled else None
                ),
                "placement_swing_leg": (
                    self.placement_swing_leg
                    if self.placement_reference_enabled
                    else None
                ),
                "placement_sequence_legs": (
                    list(self.placement_sequence_legs)
                    if self.placement_reference_enabled
                    else None
                ),
                "completed_placement_legs": list(
                    self.completed_placement_legs
                ),
                "completed_inter_leg_transfers": list(
                    self.completed_inter_leg_transfers
                ),
                "last_completed_inter_leg_transfer_metrics": dict(
                    self.last_completed_inter_leg_transfer_metrics
                ),
                "final_placement_transfer_gate_failures": list(
                    placement_transfer_gate_failures
                ),
                "final_placement_pre_unload_gate_failures": list(
                    placement_pre_unload_gate_failures
                ),
                "pre_unload_gate_enabled": (
                    self.pre_unload_gate_hold_steps > 0
                ),
                "pre_unload_gate_hold_steps": self.pre_unload_gate_hold_steps,
                "advance_clearance_gate_enabled": (
                    self.advance_clearance_gate_enabled
                ),
                "advance_clearance_gate_legs": list(
                    self.advance_clearance_gate_legs
                ),
                "advance_clearance_gate_minimum_m": (
                    self.advance_clearance_gate_minimum_m
                ),
                "advance_clearance_gate_maximum_hold_steps": (
                    self.advance_clearance_gate_maximum_hold_steps
                ),
                "maximum_advance_clearance_gate_hold_steps": (
                    self.maximum_placement_clearance_gate_hold_steps
                ),
                "final_advance_clearance_gate_released": (
                    self.placement_clearance_gate_released
                ),
                "final_advance_clearance_gate_timeout": (
                    self.placement_clearance_gate_timeout
                ),
                "final_placement_transfer_support_margin_m": (
                    placement_support_margin
                ),
                "final_placement_transfer_base_target_error_m": (
                    placement_transfer_base_target_error_m
                ),
                "final_placement_transfer_base_speed_m_s": (
                    placement_transfer_base_speed_m_s
                ),
                "final_placement_transfer_body_rate_rad_s": (
                    placement_transfer_body_rate_rad_s
                ),
                "final_placement_transfer_upright_cosine": upright_cosine,
                "final_placement_transfer_completed_tread_min_load_n": (
                    placement_transfer_completed_tread_min_load_n
                ),
                "final_placement_transfer_swing_total_load_n": (
                    placement_transfer_swing_total_load_n
                ),
                "placement_transfer_target_base_position_m": (
                    self.placement_transfer_target_base_position_m.tolist()
                    if self.inter_leg_transfer_enabled
                    else None
                ),
                "balance_point": (
                    self.com_regulation_balance_point
                    if self.com_regulation_enabled
                    else "base_origin"
                ),
                "final_balance_position_m": (
                    placement_balance_position.tolist()
                ),
                "final_com_position_m": (
                    self.latest_placement_com_position_m.tolist()
                    if self.placement_reference_enabled
                    else None
                ),
                "placement_transfer_target_balance_position_m": (
                    self.placement_transfer_target_balance_position_m.tolist()
                    if self.inter_leg_transfer_enabled
                    else None
                ),
                "placement_curriculum_level": self.current_placement_level_id,
                "forward_displacement_m": float(displacement[0]),
                "lateral_displacement_m": float(displacement[1]),
                "elevation_gain_m": float(displacement[2]),
                "maximum_base_elevation_gain_m": (
                    self.maximum_base_elevation_gain_m
                ),
                "maximum_foot_lift_m_by_leg": dict(
                    zip(LEGS, next_maximum_foot_lift.tolist(), strict=True)
                ),
                "highest_foot_step_by_leg": dict(
                    zip(LEGS, next_highest_foot_step.tolist(), strict=True)
                ),
                "final_foot_step_by_leg": dict(
                    zip(LEGS, current_foot_steps.tolist(), strict=True)
                ),
                "maximum_foot_tread_progress_by_leg": dict(
                    zip(
                        LEGS,
                        next_maximum_foot_tread_progress.tolist(),
                        strict=True,
                    )
                ),
                "final_terrain_height_m": terrain_height,
                "maximum_terrain_height_m": self.maximum_terrain_height_m,
                "minimum_base_clearance_m": self.minimum_base_clearance_m,
                "maximum_body_tilt_deg": self.maximum_tilt_deg,
                "final_projected_gravity_xyz": projected_gravity.tolist(),
                "goal_hold_duration_s": (
                    self.goal_hold_step_count / self.control_hz
                ),
                "maximum_swing_tread_normal_load_n": (
                    self.maximum_swing_tread_normal_load_n
                    if self.placement_reference_enabled
                    else None
                ),
                "maximum_tread_normal_load_n_by_leg": (
                    dict(
                        zip(
                            LEGS,
                            self.maximum_tread_normal_load_n_by_leg.tolist(),
                            strict=True,
                        )
                    )
                    if self.placement_reference_enabled
                    else None
                ),
                "tread_contact_sample_fraction": (
                    self.placement_tread_contact_sample_count
                    / self.placement_active_sample_count
                    if self.placement_active_sample_count
                    else (
                        0.0 if self.placement_reference_enabled else None
                    )
                ),
                "minimum_support_contact_fraction": (
                    self.minimum_support_contact_fraction
                    if self.placement_reference_enabled
                    else None
                ),
                "minimum_placement_support_margin_m": (
                    self.minimum_placement_support_margin_m
                    if self.placement_reference_enabled
                    else None
                ),
                "maximum_abs_support_load_sharing_correction_m": (
                    self.maximum_abs_support_load_sharing_correction_m
                    if self.placement_reference_enabled
                    else None
                ),
                "maximum_abs_support_load_sharing_correction_m_by_leg": (
                    dict(
                        zip(
                            LEGS,
                            self.maximum_abs_support_load_sharing_correction_m_by_leg.tolist(),
                            strict=True,
                        )
                    )
                    if self.placement_reference_enabled
                    else None
                ),
                "support_load_sharing_saturated_sample_fraction": (
                    self.support_load_sharing_saturated_sample_count
                    / self.support_load_sharing_active_sample_count
                    if self.support_load_sharing_active_sample_count
                    else 0.0
                ),
                "maximum_support_slip_m": (
                    self.maximum_support_slip_m
                    if self.placement_reference_enabled
                    else None
                ),
                "maximum_support_slip_m_by_leg": (
                    dict(
                        zip(
                            LEGS,
                            self.maximum_support_slip_m_by_leg.tolist(),
                            strict=True,
                        )
                    )
                    if self.placement_reference_enabled
                    else None
                ),
                "placement_reference_reach_clip_count": (
                    self.placement_reference_reach_clip_count
                    if self.placement_reference_enabled
                    else None
                ),
                "maximum_placement_reference_reach_excess_m": (
                    self.maximum_placement_reference_reach_excess_m
                    if self.placement_reference_enabled
                    else None
                ),
                "maximum_placement_desired_lift_m": (
                    self.maximum_placement_desired_lift_m
                    if self.placement_reference_enabled
                    else None
                ),
                "maximum_swing_reference_tracking_error_rad": (
                    self.maximum_swing_reference_tracking_error_rad
                    if self.placement_reference_enabled
                    else None
                ),
                "maximum_balance_lateral_deviation_m": (
                    self.maximum_balance_lateral_deviation_m
                    if self.placement_reference_enabled
                    else None
                ),
                "joint_effort_metrics": self._joint_effort_metrics(),
                "final_swing_reference_joint_positions_rad": (
                    reference_target[
                        list(
                            self.dof_indices_by_leg[
                                self.placement_swing_leg
                            ]
                        )
                    ].tolist()
                    if reference_target is not None
                    else None
                ),
                "final_swing_actual_joint_positions_rad": (
                    np.asarray(state["joint_positions"])[
                        list(
                            self.dof_indices_by_leg[
                                self.placement_swing_leg
                            ]
                        )
                    ].tolist()
                    if self.placement_reference_enabled
                    else None
                ),
                "measurable_support_slip_detected": (
                    self.maximum_support_slip_m
                    > float(
                        self.placement_reference_config.get(
                            "measurable_slip_threshold_m",
                            0.025,
                        )
                    )
                    if self.placement_reference_enabled
                    else None
                ),
                "measurable_support_slip_detected_by_leg": (
                    {
                        leg: bool(
                            self.maximum_support_slip_m_by_leg[index]
                            > float(
                                self.placement_reference_config.get(
                                    "measurable_slip_threshold_m",
                                    0.025,
                                )
                            )
                        )
                        for index, leg in enumerate(LEGS)
                    }
                    if self.placement_reference_enabled
                    else None
                ),
                "final_foot_tip_positions_m": foot_tips.tolist(),
                "initial_foot_tip_positions_m": (
                    self.initial_placement_foot_tips_m.tolist()
                    if self.placement_reference_enabled
                    else None
                ),
                "terminated": terminated,
                "truncated": truncated,
                "failure_reasons": failure_reasons,
            }
            info["episode_metrics"] = episode_metrics
            self.completed_episode_metrics.append(episode_metrics)
            self.completed_episode_metrics = self.completed_episode_metrics[-20:]

        self.previous_target = target.copy()
        self.previous_base_x_m = base_x
        self.previous_base_z_m = float(base_position[2])
        self.previous_terrain_height_m = terrain_height
        self.previous_residual_action = clipped_action.copy()
        self.maximum_foot_lift_m = next_maximum_foot_lift
        self.highest_foot_step = next_highest_foot_step
        self.maximum_foot_tread_progress = next_maximum_foot_tread_progress
        return (
            np.asarray(state["observation"]).copy(),
            reward,
            terminated,
            truncated,
            info,
        )

    @property
    def contract(self) -> dict[str, object]:
        contract = dict(super().contract)
        contract.update(
            {
                "task_id": self.config["id"],
                "dof_names": list(STAIRS_EXPECTED_DOF_ORDER),
                "observation_fields": list(self.observation_fields),
                "observation_size": self.observation_size,
                "walking_observation_size": 48,
                "include_navigation_observation": (
                    self.include_navigation_observation
                ),
                "include_foot_progress_observation": (
                    self.include_foot_progress_observation
                ),
                "include_placement_reference_observation": (
                    self.include_placement_reference_observation
                ),
                "foot_placement_sequence": list(self.foot_placement_sequence),
                "terrain_perception_mode": self.terrain_perception_mode,
                "terrain_perception": self.terrain_perception_config,
                "terrain_input_note": (
                    "Noisy, held, latency-delayed 8 x 8 VL53L5CX PhysX "
                    "raycasts compressed to 24 lane/row depth values; RGB "
                    "camera pixels are not policy inputs."
                    if self.vl53l5cx_sensor is not None
                    else (
                        "Analytic forward terrain profile; replace with a "
                        "hardware-reproducible estimator before deployment."
                    )
                ),
                "rgb_camera_policy_input": False,
                "foot_contact_material": (
                    self.foot_contact_material_applied
                    if self.foot_contact_material_applied is not None
                    else {
                        "enabled": False,
                        "source": "authored_world_material",
                    }
                ),
                "terrain_sensor_runtime": (
                    {
                        "ray_count": int(
                            np.prod(
                                self.vl53l5cx_sensor.ray_directions_from_base.shape[
                                    :2
                                ]
                            )
                        ),
                        "control_frames_per_measurement": (
                            self.vl53l5cx_sensor.control_frames_per_measurement
                        ),
                        "latency_frames": (
                            self.vl53l5cx_sensor.latency_frames
                        ),
                        "latency_seconds": (
                            self.vl53l5cx_sensor.latency_frames
                            / self.vl53l5cx_sensor.update_rate_hz
                        ),
                    }
                    if self.vl53l5cx_sensor is not None
                    else None
                ),
                "physics_steps_per_control": self.physics_steps_per_control,
                "control_action_mode": (
                    "placement_reference_plus_ppo_residual"
                    if self.placement_reference_enabled
                    else (
                        "residual_over_flat"
                        if self.residual_policy_enabled
                        else "direct"
                    )
                ),
                "residual_policy": (
                    {
                        **self.residual_policy_config,
                        "base_model_resolved": str(self.base_policy_path),
                        "base_model_sha256": self.base_policy_sha256,
                        "base_action_scale_by_dof_rad": (
                            self.base_action_scale.tolist()
                        ),
                    }
                    if self.residual_policy_enabled
                    else None
                ),
                "staircase": self.staircase_config,
                "curriculum_levels": list(self.curriculum_levels),
                "placement_reference": (
                    {
                        **self.placement_reference_config,
                        "contact_filter_paths": list(self.contact_filter_paths),
                        "contact_force_verified_success": True,
                        "inter_leg_transfer_enabled": (
                            self.inter_leg_transfer_enabled
                        ),
                        "observation_fields": list(
                            PLACEMENT_REFERENCE_OBSERVATION_FIELDS
                        ),
                    }
                    if self.placement_reference_enabled
                    else None
                ),
                "placement_curriculum": (
                    self.placement_curriculum_config
                    if self.placement_reference_enabled
                    else None
                ),
                "success_requirements": {
                    "minimum_base_elevation_fraction": float(
                        self.config.get(
                            "success_minimum_base_elevation_fraction",
                            0.0,
                        )
                    ),
                    "required_feet_on_goal_tread": int(
                        self.config.get(
                            "success_required_feet_on_goal_tread",
                            0,
                        )
                    ),
                    "hold_seconds": float(self.config["success_hold_seconds"]),
                },
            }
        )
        return contract
