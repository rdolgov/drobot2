"""Evaluate one analytic first-tread stance/approach profile without PPO."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _run_support import sha256_file  # noqa: E402
from _stair_rl_contract import (  # noqa: E402
    FIRST_TREAD_EXPERIMENT_PROFILES,
    config_for_first_tread_experiment,
)

parser = argparse.ArgumentParser(
    description=(
        "Run the camera-blind analytic first-foot reference under one "
        "stance/approach profile."
    )
)
parser.add_argument(
    "--config",
    default=str(
        SCRIPT_DIR / "quadruped_stairs_v10_front_right_single_tread_placement.yaml"
    ),
)
parser.add_argument(
    "--profile",
    choices=FIRST_TREAD_EXPERIMENT_PROFILES,
    required=True,
)
parser.add_argument("--placement-level", default="quarter-tread-load")
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--report", required=True)
parser.add_argument("--gui", action="store_true")
args, _ = parser.parse_known_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


config_path = resolve(args.config)
report_path = resolve(args.report)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
task_config = config_for_first_tread_experiment(config["task"], args.profile)
world_path = resolve(str(task_config["world"]))

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": not args.gui,
        "width": 1280,
        "height": 720,
    }
)

from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: E402

report: dict[str, object] = {
    "status": "FAIL",
    "scope": "Analytic zero-residual first-tread stance/approach probe",
    "profile": args.profile,
    "placement_level": args.placement_level,
    "seed": args.seed,
    "config": str(config_path),
    "config_sha256": sha256_file(config_path),
    "world": str(world_path),
    "world_sha256": sha256_file(world_path) if world_path.is_file() else None,
    "stair_rise_m": float(task_config["staircase"]["rise_m"]),
    "stair_tread_depth_m": float(task_config["staircase"]["tread_depth_m"]),
    "effort_cap_nm": float(
        task_config["robot_hardware_profile"]["effort_cap_nm"]
    ),
    "camera_policy_input": False,
    "nominal_stance": dict(task_config["nominal_stance"]),
    "target_heading_yaw_deg": float(task_config["target_heading_yaw_deg"]),
}
raw_env: QuadrupedStairsEnv | None = None
try:
    if not world_path.is_file():
        raise FileNotFoundError(world_path)
    raw_env = QuadrupedStairsEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
        render_mode="human" if args.gui else None,
    )
    raw_env.set_placement_level(
        args.placement_level,
        activate_immediately=True,
    )
    observation, reset_info = raw_env.reset(seed=args.seed)
    last_info: dict[str, object] = {}
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        observation, _, terminated, truncated, last_info = raw_env.step(
            np.zeros(12, dtype=np.float32)
        )
        steps += 1
    episode = dict(last_info.get("episode_metrics", {}))
    swing_leg = str(task_config["placement_reference"]["swing_leg"])
    report.update(
        {
            "status": "PASS",
            "strict_placement_passed": bool(
                episode.get("placement_completed", False)
            ),
            "steps": steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reset": jsonable(reset_info),
            "environment_contract": jsonable(raw_env.contract),
            "episode_metrics": jsonable(episode),
            "final_info": {
                key: jsonable(last_info.get(key))
                for key in (
                    "base_position_m",
                    "heading_error_rad",
                    "swing_tread_normal_load_n",
                    "swing_step_layer_normal_load_n",
                    "step_layer_normal_load_n_by_leg",
                    "tread_top_normal_load_n_by_leg",
                    "ground_normal_load_n_by_leg",
                    "foot_tip_positions_m",
                    "support_margin_m",
                    "maximum_support_slip_m",
                    "touchdown_load_lift_correction_m",
                    "maximum_touchdown_load_lift_correction_m",
                    "touchdown_support_triggered",
                    "touchdown_support_trigger_step",
                    "touchdown_support_release_fraction",
                    "maximum_touchdown_support_release_fraction",
                    "failure_reasons",
                )
            },
            "summary": {
                "swing_leg": swing_leg,
                "maximum_swing_lift_m": float(
                    dict(episode.get("maximum_foot_lift_m_by_leg", {})).get(
                        swing_leg,
                        0.0,
                    )
                ),
                "maximum_qualified_tread_top_load_n": float(
                    dict(episode.get("maximum_tread_normal_load_n_by_leg", {})).get(
                        swing_leg,
                        0.0,
                    )
                ),
                "minimum_support_margin_m": float(
                    episode.get("minimum_placement_support_margin_m", 0.0)
                ),
                "maximum_support_slip_m": float(
                    episode.get("maximum_support_slip_m", 0.0)
                ),
                "maximum_body_tilt_deg": float(
                    episode.get("maximum_body_tilt_deg", 0.0)
                ),
                "final_foot_tip_positions_m": jsonable(
                    episode.get("final_foot_tip_positions_m", [])
                ),
            },
        }
    )
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    if raw_env is not None:
        raw_env.close()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(jsonable(report), stream, indent=2)
        stream.write("\n")
    simulation_app.close()

print(
    "DROBOT_FIRST_TREAD_PROFILE_RESULT="
    + json.dumps(report.get("summary", {"error": report.get("error")}))
)
if report["status"] != "PASS":
    raise SystemExit(1)
