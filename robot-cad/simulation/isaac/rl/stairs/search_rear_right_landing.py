"""Search rear-right tread-landing references from the verified V38 handoff.

The deterministic front-foot prefix and analytic rear transfer are replayed
once.  Every candidate then restores the same simulator snapshot, preserves
the 190 mm physical clearance gate, and runs the frozen V17 + V35 + V38
policy composition through forward swing and tread contact.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch._dynamo  # noqa: F401
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default=str(
        SCRIPT_DIR / "quadruped_stairs_v38_positive_margin_rear_transfer.yaml"
    ),
)
parser.add_argument("--seed", type=int, default=849)
parser.add_argument("--candidate-start", type=int, default=0)
parser.add_argument("--candidate-limit", type=int, default=16)
parser.add_argument("--maximum-target-steps", type=int, default=600)
parser.add_argument(
    "--rear-transfer-forward-m",
    type=float,
    default=None,
    help="Override the full rear-right composite-COM transfer offset.",
)
parser.add_argument(
    "--report",
    default=(
        "simulation/isaac/output/rl/"
        "ppo-stairs-v39-rear-right-landing-search-seed849.json"
    ),
)
parser.add_argument(
    "--front-right-model",
    default=(
        "simulation/isaac/models/"
        "ppo-stairs-v10-180mm-25cm-front-right-placement-small/"
        "drobot_stairs_ppo_final.zip"
    ),
)
parser.add_argument(
    "--front-left-model",
    default=(
        "simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/"
        "drobot_stairs_ppo_final.zip"
    ),
)
parser.add_argument(
    "--swing-base-model",
    default=(
        "simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/"
        "drobot_stairs_ppo_final.zip"
    ),
)
parser.add_argument(
    "--swing-residual-model",
    default=(
        "simulation/isaac/models/ppo-stairs-v35-rear-right-190mm-lift-small/"
        "drobot_stairs_ppo_final.zip"
    ),
)
parser.add_argument(
    "--support-residual-model",
    default=(
        "simulation/isaac/models/ppo-stairs-v38-rear-right-190mm-small/"
        "drobot_stairs_ppo_final.zip"
    ),
)
args, _ = parser.parse_known_args()
if args.candidate_limit < 1:
    parser.error("--candidate-limit must be positive")
if args.candidate_start < 0:
    parser.error("--candidate-start cannot be negative")
if args.maximum_target_steps < 1:
    parser.error("--maximum-target-steps must be positive")


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def candidate(
    candidate_id: str,
    *,
    swing_forward_m: float,
    landing_forward_m: float,
    landing_lift_m: float,
    tread_fraction: float,
    lift_forward_m: float = 0.050,
    lower_seconds: float = 1.5,
    weight_shift_forward_m: float = 0.045,
    hold_target_forward_m: float = 0.0,
    swing_residual_scale: float = 0.5,
    constant_swing_action: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pitch_gain_m: float | None = None,
    pitch_maximum_m: float | None = None,
    post_clearance_body_shift_forward_m: float = 0.0,
    body_shift_fraction_of_advance: float | None = None,
) -> dict[str, object]:
    return {
        "id": candidate_id,
        "swing_forward_offset_m": swing_forward_m,
        "landing_forward_offset_m": landing_forward_m,
        "landing_lift_m": landing_lift_m,
        "lift_forward_offset_m": lift_forward_m,
        "target_tread_fraction": tread_fraction,
        "lower_duration_seconds": lower_seconds,
        "weight_shift_forward_m": weight_shift_forward_m,
        "hold_target_forward_m": hold_target_forward_m,
        "swing_residual_scale": swing_residual_scale,
        "constant_swing_action": list(constant_swing_action),
        "pitch_gain_m": pitch_gain_m,
        "pitch_maximum_m": pitch_maximum_m,
        "post_clearance_body_shift_forward_m": (
            post_clearance_body_shift_forward_m
        ),
        "body_shift_fraction_of_advance": body_shift_fraction_of_advance,
    }


CANDIDATES = (
    candidate(
        "v38-inherited-baseline",
        swing_forward_m=0.210,
        landing_forward_m=0.190,
        landing_lift_m=0.125,
        tread_fraction=0.50,
    ),
    candidate(
        "reach-300mm",
        swing_forward_m=0.300,
        landing_forward_m=0.280,
        landing_lift_m=0.125,
        tread_fraction=0.12,
    ),
    candidate(
        "reach-320mm",
        swing_forward_m=0.320,
        landing_forward_m=0.300,
        landing_lift_m=0.125,
        tread_fraction=0.16,
    ),
    candidate(
        "reach-340mm",
        swing_forward_m=0.340,
        landing_forward_m=0.320,
        landing_lift_m=0.125,
        tread_fraction=0.20,
    ),
    candidate(
        "reach-360mm",
        swing_forward_m=0.360,
        landing_forward_m=0.340,
        landing_lift_m=0.125,
        tread_fraction=0.24,
    ),
    candidate(
        "reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.28,
    ),
    candidate(
        "reach-400mm",
        swing_forward_m=0.400,
        landing_forward_m=0.380,
        landing_lift_m=0.125,
        tread_fraction=0.32,
    ),
    candidate(
        "reach-340mm-lower-105mm",
        swing_forward_m=0.340,
        landing_forward_m=0.320,
        landing_lift_m=0.105,
        tread_fraction=0.20,
    ),
    candidate(
        "reach-360mm-lower-105mm",
        swing_forward_m=0.360,
        landing_forward_m=0.340,
        landing_lift_m=0.105,
        tread_fraction=0.24,
    ),
    candidate(
        "reach-380mm-lower-105mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.105,
        tread_fraction=0.28,
    ),
    candidate(
        "reach-400mm-lower-105mm",
        swing_forward_m=0.400,
        landing_forward_m=0.380,
        landing_lift_m=0.105,
        tread_fraction=0.32,
    ),
    candidate(
        "reach-360mm-lower-105mm-slow",
        swing_forward_m=0.360,
        landing_forward_m=0.340,
        landing_lift_m=0.105,
        tread_fraction=0.24,
        lower_seconds=2.5,
    ),
    candidate(
        "reach-380mm-lower-105mm-slow",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.105,
        tread_fraction=0.28,
        lower_seconds=2.5,
    ),
    candidate(
        "reach-400mm-lower-105mm-slow",
        swing_forward_m=0.400,
        landing_forward_m=0.380,
        landing_lift_m=0.105,
        tread_fraction=0.32,
        lower_seconds=2.5,
    ),
    candidate(
        "com-forward-060mm-reach-340mm",
        swing_forward_m=0.340,
        landing_forward_m=0.320,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        weight_shift_forward_m=0.060,
    ),
    candidate(
        "com-forward-060mm-reach-360mm",
        swing_forward_m=0.360,
        landing_forward_m=0.340,
        landing_lift_m=0.125,
        tread_fraction=0.12,
        lower_seconds=2.5,
        weight_shift_forward_m=0.060,
    ),
    candidate(
        "com-forward-075mm-reach-340mm",
        swing_forward_m=0.340,
        landing_forward_m=0.320,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        weight_shift_forward_m=0.075,
    ),
    candidate(
        "com-forward-075mm-reach-360mm",
        swing_forward_m=0.360,
        landing_forward_m=0.340,
        landing_lift_m=0.125,
        tread_fraction=0.12,
        lower_seconds=2.5,
        weight_shift_forward_m=0.075,
    ),
    candidate(
        "com-forward-075mm-reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.16,
        lower_seconds=2.5,
        weight_shift_forward_m=0.075,
    ),
    candidate(
        "com-forward-090mm-reach-340mm",
        swing_forward_m=0.340,
        landing_forward_m=0.320,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        weight_shift_forward_m=0.090,
    ),
    candidate(
        "com-forward-090mm-reach-360mm",
        swing_forward_m=0.360,
        landing_forward_m=0.340,
        landing_lift_m=0.125,
        tread_fraction=0.12,
        lower_seconds=2.5,
        weight_shift_forward_m=0.090,
    ),
    candidate(
        "com-forward-090mm-reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.16,
        lower_seconds=2.5,
        weight_shift_forward_m=0.090,
    ),
    candidate(
        "touchdown-com-015mm-reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        hold_target_forward_m=0.015,
    ),
    candidate(
        "touchdown-com-030mm-reach-360mm",
        swing_forward_m=0.360,
        landing_forward_m=0.340,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        hold_target_forward_m=0.030,
    ),
    candidate(
        "touchdown-com-030mm-reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        hold_target_forward_m=0.030,
    ),
    candidate(
        "touchdown-com-045mm-reach-360mm",
        swing_forward_m=0.360,
        landing_forward_m=0.340,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        hold_target_forward_m=0.045,
    ),
    candidate(
        "touchdown-com-045mm-reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        hold_target_forward_m=0.045,
    ),
    candidate(
        "touchdown-com-060mm-reach-360mm",
        swing_forward_m=0.360,
        landing_forward_m=0.340,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        hold_target_forward_m=0.060,
    ),
    candidate(
        "touchdown-com-060mm-reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        hold_target_forward_m=0.060,
    ),
    candidate(
        "swing-scale-010-reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        swing_residual_scale=0.10,
    ),
    candidate(
        "swing-scale-025-reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        swing_residual_scale=0.25,
    ),
    candidate(
        "swing-scale-035-reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        swing_residual_scale=0.35,
    ),
    candidate(
        "swing-scale-065-reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        swing_residual_scale=0.65,
    ),
    candidate(
        "swing-scale-075-reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        swing_residual_scale=0.75,
    ),
    candidate(
        "swing-scale-090-reach-380mm",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        swing_residual_scale=0.90,
    ),
    candidate(
        "v40-trained-swing-canonical",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        swing_residual_scale=0.50,
    ),
    *(
        candidate(
            f"swing-action-flex{hip_flexion:+.1f}-knee{knee:+.1f}",
            swing_forward_m=0.380,
            landing_forward_m=0.360,
            landing_lift_m=0.125,
            tread_fraction=0.08,
            lower_seconds=2.5,
            constant_swing_action=(0.0, hip_flexion, knee),
        )
        for hip_flexion in (-0.4, 0.0, 0.4)
        for knee in (-0.4, 0.0, 0.4)
    ),
    *(
        candidate(
            f"lift-forward-{round(lift_forward_m * 1000):03d}mm",
            swing_forward_m=0.380,
            landing_forward_m=0.360,
            landing_lift_m=0.125,
            tread_fraction=0.08,
            lift_forward_m=lift_forward_m,
            lower_seconds=2.5,
        )
        for lift_forward_m in (0.070, 0.090, 0.110, 0.130)
    ),
    *(
        candidate(
            f"lift070-pitch-{gain_m:.3f}-{maximum_m:.3f}",
            swing_forward_m=0.380,
            landing_forward_m=0.360,
            landing_lift_m=0.125,
            tread_fraction=0.08,
            lift_forward_m=0.070,
            lower_seconds=2.5,
            pitch_gain_m=gain_m,
            pitch_maximum_m=maximum_m,
        )
        for gain_m, maximum_m in (
            (0.120, 0.035),
            (0.160, 0.050),
            (0.240, 0.075),
        )
    ),
    *(
        candidate(
            f"lift070-gentle-z{round(landing_lift_m * 1000):03d}mm",
            swing_forward_m=0.380,
            landing_forward_m=0.360,
            landing_lift_m=landing_lift_m,
            tread_fraction=0.08,
            lift_forward_m=0.070,
            lower_seconds=4.0,
        )
        for landing_lift_m in (0.135, 0.145, 0.155)
    ),
    *(
        candidate(
            f"post-clearance-body-shift-{shift_m:+.3f}m",
            swing_forward_m=0.380,
            landing_forward_m=0.360,
            landing_lift_m=0.125,
            tread_fraction=0.08,
            lower_seconds=2.5,
            post_clearance_body_shift_forward_m=shift_m,
        )
        for shift_m in (-0.020, -0.010, 0.010, 0.020)
    ),
    candidate(
        "split-body-shift-010m-then-swing",
        swing_forward_m=0.380,
        landing_forward_m=0.360,
        landing_lift_m=0.125,
        tread_fraction=0.08,
        lower_seconds=2.5,
        post_clearance_body_shift_forward_m=0.010,
        body_shift_fraction_of_advance=0.50,
    ),
)

SELECTED_CANDIDATES = CANDIDATES[
    args.candidate_start : args.candidate_start + args.candidate_limit
]
if not SELECTED_CANDIDATES:
    parser.error("candidate slice is empty")

config_path = project_path(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
task_config = copy.deepcopy(config["task"])
if args.rear_transfer_forward_m is not None:
    rear_transfer_target = task_config["placement_reference"][
        "inter_leg_transfer"
    ]["com_regulation"]["target_offset_by_swing_leg"]["rear_right"]
    rear_transfer_target["forward"] = float(args.rear_transfer_forward_m)
world_path = project_path(task_config["world"])
report_path = project_path(args.report)

# The search changes only the rear-right terminal behavior.  The independent
# physical clearance gate and its 190 mm threshold remain in the V38 config.
rear_override = dict(
    task_config["placement_reference"]["level_override_by_leg"]["rear_right"]
)
rear_override.update(
    {
        "success_mode": "tread_contact",
        "contact_hold_seconds": 0.75,
        "minimum_lift_m": 0.190,
        "minimum_support_margin_m": 0.015,
    }
)
task_config["placement_reference"]["level_override_by_leg"]["rear_right"] = (
    rear_override
)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from _placement_phase_training import (  # noqa: E402
    FrozenBaseResidualPolicy,
    PlacementPhaseTrainingEnv,
)
from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: E402
from _stair_rl_contract import placement_policy_action_mask  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

model_paths = {
    "front_right": project_path(args.front_right_model),
    "front_left": project_path(args.front_left_model),
    "swing_base": project_path(args.swing_base_model),
    "swing_residual": project_path(args.swing_residual_model),
    "support_residual": project_path(args.support_residual_model),
}
report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "config": str(config_path),
    "world": str(world_path),
    "seed": args.seed,
    "candidate_start": args.candidate_start,
    "candidate_count": len(SELECTED_CANDIDATES),
    "maximum_target_steps": args.maximum_target_steps,
    "rear_transfer_forward_m": task_config["placement_reference"]
    ["inter_leg_transfer"]["com_regulation"]
    ["target_offset_by_swing_leg"]["rear_right"]["forward"],
    "models": {key: str(value) for key, value in model_paths.items()},
    "immutable_acceptance": {
        "stair_tread_depth_m": task_config["staircase"]["tread_depth_m"],
        "stair_rise_m": task_config["staircase"]["rise_m"],
        "physical_clearance_gate_m": task_config["placement_reference"]
        ["advance_clearance_gate"]["minimum_clearance_m"],
        "minimum_support_margin_m": 0.015,
        "contact_hold_seconds": 0.75,
        "effort_cap_nm": task_config["robot_hardware_profile"][
            "effort_cap_nm"
        ],
    },
}
raw_env: QuadrupedStairsEnv | None = None
exit_code = 1
try:
    raw_env = QuadrupedStairsEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
    )
    raw_env.set_evaluation_level(1)
    raw_env.set_placement_level("left-center-tread-load")
    precursor_policies = {
        "front_right": PPO.load(str(model_paths["front_right"]), device="cpu"),
        "front_left": PPO.load(str(model_paths["front_left"]), device="cpu"),
    }
    swing_mask = placement_policy_action_mask(
        raw_env.dof_names,
        target_leg="rear_right",
        mode="swing_only",
    )
    support_mask = placement_policy_action_mask(
        raw_env.dof_names,
        target_leg="rear_right",
        mode="support_only",
    )
    swing_policy = FrozenBaseResidualPolicy(
        base_policy=PPO.load(str(model_paths["swing_base"]), device="cpu"),
        residual_policy=PPO.load(
            str(model_paths["swing_residual"]), device="cpu"
        ),
        action_space=raw_env.action_space,
        residual_scale=0.5,
        base_mask=swing_mask,
        residual_mask=swing_mask,
        compact_residual_action=True,
    )
    composed_policy = FrozenBaseResidualPolicy(
        base_policy=swing_policy,
        residual_policy=PPO.load(
            str(model_paths["support_residual"]), device="cpu"
        ),
        action_space=raw_env.action_space,
        residual_scale=1.0,
        residual_mask=support_mask,
        compact_residual_action=True,
    )
    wrapped = PlacementPhaseTrainingEnv(
        raw_env,
        target_leg="rear_right",
        precursor_policies=precursor_policies,
        target_base_policy=composed_policy,
        target_residual_scale=0.5,
        target_residual_mask=swing_mask,
        compact_residual_action=True,
        maximum_reset_attempts=4,
        maximum_precursor_steps=3600,
        cache_phase_snapshot=True,
    )
    zero_action = np.zeros(wrapped.action_space.shape, dtype=np.float32)
    print("DROBOT_REAR_LANDING_SEARCH_PHASE=precursor_replay", flush=True)
    wrapped.reset(seed=args.seed)
    if wrapped.phase_snapshot is None:
        raise RuntimeError("Rear-right placement snapshot was not captured")
    print("DROBOT_REAR_LANDING_SEARCH_PHASE=candidate_replay", flush=True)

    results: list[dict[str, object]] = []
    for relative_index, values in enumerate(SELECTED_CANDIDATES):
        index = args.candidate_start + relative_index
        level_overrides = dict(
            raw_env.placement_reference_config["level_override_by_leg"]
        )
        candidate_override = dict(level_overrides["rear_right"])
        candidate_override.update(
            {
                "success_mode": "tread_contact",
                "apex_lift_m": 0.235,
                "lift_forward_offset_m": values["lift_forward_offset_m"],
                "minimum_lift_m": 0.190,
                "minimum_support_margin_m": 0.015,
                "contact_hold_seconds": 0.75,
                "swing_forward_offset_m": values[
                    "swing_forward_offset_m"
                ],
                "landing_forward_offset_m": values[
                    "landing_forward_offset_m"
                ],
                "landing_lift_m": values["landing_lift_m"],
                "target_tread_fraction": values["target_tread_fraction"],
            }
        )
        level_overrides["rear_right"] = candidate_override
        raw_env.placement_reference_config["level_override_by_leg"] = (
            level_overrides
        )
        timing_overrides = dict(
            raw_env.placement_reference_config.get(
                "timing_override_by_leg", {}
            )
        )
        rear_timing = dict(timing_overrides.get("rear_right", {}))
        rear_timing["lower_duration_seconds"] = values[
            "lower_duration_seconds"
        ]
        timing_overrides["rear_right"] = rear_timing
        raw_env.placement_reference_config["timing_override_by_leg"] = (
            timing_overrides
        )
        weight_shift = dict(
            raw_env.placement_reference_config["weight_shift"]
        )
        weight_shift["forward_m"] = values["weight_shift_forward_m"]
        raw_env.placement_reference_config["weight_shift"] = weight_shift
        hold_offsets = dict(
            raw_env.com_regulation_config.get(
                "hold_target_offset_by_swing_leg", {}
            )
        )
        hold_offsets["rear_right"] = {
            "forward": values["hold_target_forward_m"],
            "lateral": 0.0,
        }
        raw_env.com_regulation_config[
            "hold_target_offset_by_swing_leg"
        ] = hold_offsets
        swing_policy.residual_scale = float(values["swing_residual_scale"])
        post_clearance_shift = {
            "forward_m": values["post_clearance_body_shift_forward_m"],
            "lateral_m": 0.0,
        }
        if values["body_shift_fraction_of_advance"] is not None:
            post_clearance_shift["body_shift_fraction_of_advance"] = values[
                "body_shift_fraction_of_advance"
            ]
        raw_env.inter_leg_transfer_config[
            "post_clearance_body_shift_by_leg"
        ] = {"rear_right": post_clearance_shift}
        if values["pitch_gain_m"] is not None:
            pitch_gain_by_leg = dict(
                raw_env.pitch_feedback_config.get(
                    "proportional_gain_m_by_swing_leg", {}
                )
            )
            pitch_gain_by_leg["rear_right"] = values["pitch_gain_m"]
            raw_env.pitch_feedback_config[
                "proportional_gain_m_by_swing_leg"
            ] = pitch_gain_by_leg
        if values["pitch_maximum_m"] is not None:
            pitch_maximum_by_leg = dict(
                raw_env.pitch_feedback_config.get(
                    "maximum_correction_m_by_swing_leg", {}
                )
            )
            pitch_maximum_by_leg["rear_right"] = values[
                "pitch_maximum_m"
            ]
            raw_env.pitch_feedback_config[
                "maximum_correction_m_by_swing_leg"
            ] = pitch_maximum_by_leg

        wrapped.reset(seed=args.seed + index + 1)
        candidate_action = np.asarray(
            values["constant_swing_action"], dtype=np.float32
        )
        reward_sum = 0.0
        minimum_margin_m = float("inf")
        minimum_upright = 1.0
        maximum_lift_m = 0.0
        maximum_tread_load_n = 0.0
        maximum_swing_x_m = -float("inf")
        minimum_base_delta_x_m = float("inf")
        maximum_base_delta_x_m = -float("inf")
        minimum_com_delta_x_m = float("inf")
        maximum_com_delta_x_m = -float("inf")
        final_swing_xyz_m: list[float] | None = None
        clearance_released = False
        completed = False
        last_info: dict[str, object] = {}
        for _step in range(1, args.maximum_target_steps + 1):
            _, reward, terminated, truncated, info = wrapped.step(
                candidate_action
            )
            last_info = dict(info)
            reward_sum += float(reward)
            minimum_margin_m = min(
                minimum_margin_m,
                float(info.get("placement_support_margin_m", float("inf"))),
            )
            minimum_upright = min(
                minimum_upright,
                float(info.get("placement_upright_cosine", 1.0)),
            )
            maximum_lift_m = max(
                maximum_lift_m,
                float(info.get("placement_swing_lift_m", 0.0)),
            )
            maximum_tread_load_n = max(
                maximum_tread_load_n,
                float(info.get("swing_tread_normal_load_n", 0.0)),
            )
            base_delta_x_m = float(
                raw_env.latest_placement_base_position_m[0]
                - raw_env.placement_leg_baseline_base_position_m[0]
            )
            com_delta_x_m = float(
                raw_env.latest_placement_com_position_m[0]
                - raw_env.placement_leg_baseline_balance_position_m[0]
            )
            minimum_base_delta_x_m = min(
                minimum_base_delta_x_m, base_delta_x_m
            )
            maximum_base_delta_x_m = max(
                maximum_base_delta_x_m, base_delta_x_m
            )
            minimum_com_delta_x_m = min(
                minimum_com_delta_x_m, com_delta_x_m
            )
            maximum_com_delta_x_m = max(
                maximum_com_delta_x_m, com_delta_x_m
            )
            clearance_released = bool(
                clearance_released
                or info.get("placement_clearance_gate_released", False)
            )
            foot_tips = np.asarray(
                info.get("foot_tip_positions_m"), dtype=np.float64
            )
            swing_xyz = foot_tips[raw_env.placement_swing_leg_index]
            final_swing_xyz_m = swing_xyz.tolist()
            maximum_swing_x_m = max(maximum_swing_x_m, float(swing_xyz[0]))
            completed = bool(
                completed
                or info.get("placement_leg_completed_event") == "rear_right"
            )
            if terminated or truncated:
                break
        target_x_m = float(task_config["staircase"]["start_x_m"]) + float(
            values["target_tread_fraction"]
        ) * float(task_config["staircase"]["tread_depth_m"])
        episode_metrics = dict(last_info.get("episode_metrics", {}))
        result = {
            **dict(values),
            "index": index,
            "steps": _step,
            "completed": completed,
            "clearance_released": clearance_released,
            "reward_sum": reward_sum,
            "maximum_physical_lift_m": maximum_lift_m,
            "minimum_support_margin_m": minimum_margin_m,
            "maximum_body_tilt_deg": math.degrees(
                math.acos(float(np.clip(minimum_upright, -1.0, 1.0)))
            ),
            "maximum_tread_load_n": maximum_tread_load_n,
            "target_x_m": target_x_m,
                "maximum_swing_x_m": maximum_swing_x_m,
                "minimum_base_delta_x_m": minimum_base_delta_x_m,
                "maximum_base_delta_x_m": maximum_base_delta_x_m,
                "minimum_com_delta_x_m": minimum_com_delta_x_m,
                "maximum_com_delta_x_m": maximum_com_delta_x_m,
            "closest_x_error_m": abs(maximum_swing_x_m - target_x_m),
            "final_swing_xyz_m": final_swing_xyz_m,
            "failure_reasons": list(last_info.get("failure_reasons", ())),
            "terminated": bool(last_info.get("terminated", False)),
            "truncated": bool(last_info.get("truncated", False)),
            "episode_maximum_support_slip_m": episode_metrics.get(
                "maximum_support_slip_m"
            ),
            "episode_joint_effort_metrics": episode_metrics.get(
                "joint_effort_metrics"
            ),
        }
        results.append(result)
        print(
            "DROBOT_REAR_LANDING_CANDIDATE="
            + json.dumps(result, sort_keys=True),
            flush=True,
        )

    ranked = sorted(
        results,
        key=lambda item: (
            bool(item["completed"]),
            bool(item["clearance_released"]),
            float(item["maximum_tread_load_n"]),
            -float(item["closest_x_error_m"]),
            float(item["minimum_support_margin_m"]),
            -float(item["maximum_body_tilt_deg"]),
        ),
        reverse=True,
    )
    report.update(
        {
            "status": "PASS",
            "search_contract": {
                "same_cached_post_transfer_state": True,
                "policy_composition": (
                    "V17 swing-only + 0.5*V35 compact swing residual + "
                    "1.0*V38 compact support residual"
                ),
                "camera_policy_input": False,
                "external_camera_used": False,
            },
            "successful_candidates": sum(
                bool(item["completed"]) for item in results
            ),
            "best": ranked[0],
            "ranked_results": ranked,
            "phase_wrapper_stats": wrapped.training_stats(),
        }
    )
    exit_code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    if raw_env is not None:
        raw_env.close()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_REAR_LANDING_SEARCH=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
