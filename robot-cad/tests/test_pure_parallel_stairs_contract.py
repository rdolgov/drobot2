"""Static contracts for the external Isaac Lab pure-parallel stair task."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "simulation" / "isaac" / "rl" / "parallel_stairs"


def _source(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def _class_assignments(source: str, class_name: str) -> dict[str, object]:
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    values: dict[str, object] = {}
    for node in class_node.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return values


def _method_source(source: str, class_name: str, method_name: str) -> str:
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    segment = ast.get_source_segment(source, method)
    assert segment is not None
    return segment


def test_exact_stair_and_policy_dimensions_are_fixed() -> None:
    cfg_source = _source("pure_stairs_env_cfg.py")
    cfg_values = _class_assignments(cfg_source, "DrobotPureStairsEnvCfg")
    assert cfg_values["action_space"] == 12
    assert cfg_values["observation_space"] == 70
    assert cfg_values["stair_rise_m"] == 0.18
    assert cfg_values["stair_tread_depth_m"] == 0.25
    assert cfg_values["stair_step_count"] == 4
    assert "EFFORT_CAP_NM = 0.8825985" in cfg_source

    terrain_values = _class_assignments(
        _source("exact_stairs_terrain.py"), "ExactStairsTerrainCfg"
    )
    assert terrain_values["rise_m"] == 0.18
    assert terrain_values["tread_depth_m"] == 0.25
    assert terrain_values["step_count"] == 4


def test_policy_observation_uses_only_deployable_sensor_state() -> None:
    env_source = _source("pure_stairs_env.py")
    observation = _method_source(env_source, "DrobotPureStairsEnv", "_get_observations")

    required = (
        "root_ang_vel_b",
        "projected_gravity_b",
        "joint_pos",
        "joint_vel",
        "_previous_actions",
        "_foot_forces",
        "_depth_observation",
    )
    for field in required:
        assert field in observation

    privileged = ("root_pos_w", "body_pos_w", "env_origins", "_terrain_height")
    for field in privileged:
        assert field not in observation


def test_reward_has_no_scripted_phase_or_reference_action() -> None:
    reward = _method_source(_source("pure_stairs_env.py"), "DrobotPureStairsEnv", "_get_rewards")
    lowered = reward.lower()
    forbidden_terms = (
        "gait_phase",
        "leg_order",
        "reference_action",
        "trajectory",
        "inverse_kinematics",
    )
    for forbidden in forbidden_terms:
        assert forbidden not in lowered
