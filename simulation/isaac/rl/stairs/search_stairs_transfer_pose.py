"""Search a slow, torque-aware post-transfer front-left lift reference."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch._dynamo  # noqa: F401
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

parser = argparse.ArgumentParser(
    description="Search torque-aware front-left lift timing after transfer."
)
parser.add_argument(
    "--config",
    default=str(
        SCRIPT_DIR / "quadruped_stairs_v16_front_pair_proprioceptive_support.yaml"
    ),
)
parser.add_argument(
    "--precursor-model",
    default=(
        "simulation/isaac/models/"
        "ppo-stairs-v10-180mm-25cm-front-right-placement-small/"
        "drobot_stairs_ppo_final.zip"
    ),
)
parser.add_argument(
    "--swing-model",
    default=(
        "simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/"
        "drobot_stairs_ppo_final.zip"
    ),
)
parser.add_argument("--seed", type=int, default=306)
parser.add_argument("--candidate-seconds", type=float, default=8.0)
parser.add_argument(
    "--search-mode",
    choices=(
        "timing",
        "support",
        "support-fine",
        "validate",
        "full-sequence",
        "full-feedback",
        "swing-bias",
        "swing-clearance",
    ),
    default="timing",
)
parser.add_argument(
    "--report",
    default=(
        "simulation/isaac/output/rl/"
        "ppo-stairs-v22-transfer-pose-search-seed306/search_report.json"
    ),
)
args, _ = parser.parse_known_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


config_path = resolve(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
task_config = deepcopy(config["task"])
task_config["episode_seconds"] = max(24.0, float(args.candidate_seconds) + 1.0)
termination = dict(task_config["termination"])
termination["maximum_lateral_deviation_m"] = 0.20
task_config["termination"] = termination
world_path = resolve(str(task_config["world"]))
precursor_model_path = resolve(args.precursor_model)
swing_model_path = resolve(args.swing_model)
report_path = resolve(args.report)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from _placement_phase_training import PlacementPhaseTrainingEnv  # noqa: E402
from _policy_transfer import predict_with_observation_prefix  # noqa: E402
from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: E402
from _stair_rl_contract import placement_policy_action_mask  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402


def candidates() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    if args.search_mode == "swing-clearance":
        for apex_lift_m, knee in (
            (0.205, 0.0),
            (0.225, 0.0),
            (0.245, 0.0),
            (0.265, 0.0),
            (0.225, 0.20),
            (0.245, 0.20),
            (0.245, 0.40),
            (0.265, 0.20),
            (0.265, 0.40),
        ):
            result.append(
                {
                    "id": (
                        f"apex-{int(round(1000 * apex_lift_m)):03d}mm-"
                        f"lift-knee-plus-{int(round(1000 * knee)):03d}"
                    ),
                    "mode": "blend_to_nominal_stance",
                    "lift_duration_s": 3.0,
                    "forward_offset_m": 0.120,
                    "apex_lift_m": apex_lift_m,
                    "swing_action_bias": {
                        "hip_abduction": 0.0,
                        "hip_flexion": 0.0,
                        "knee": knee,
                    },
                    "swing_action_bias_lift_only": True,
                }
            )
        return result
    if args.search_mode == "swing-bias":
        for candidate_id, hip_flexion, knee in (
            ("baseline", 0.0, 0.0),
            ("knee-plus-100", 0.0, 0.10),
            ("knee-plus-200", 0.0, 0.20),
            ("knee-minus-100", 0.0, -0.10),
            ("knee-minus-200", 0.0, -0.20),
            ("hip-plus-100", 0.10, 0.0),
            ("hip-minus-100", -0.10, 0.0),
            ("hip-plus-100-knee-plus-200", 0.10, 0.20),
            ("hip-minus-100-knee-plus-200", -0.10, 0.20),
        ):
            result.append(
                {
                    "id": candidate_id,
                    "mode": "blend_to_nominal_stance",
                    "lift_duration_s": 3.0,
                    "forward_offset_m": 0.120,
                    "swing_action_bias": {
                        "hip_abduction": 0.0,
                        "hip_flexion": hip_flexion,
                        "knee": knee,
                    },
                }
            )
        return result
    if args.search_mode == "full-feedback":

        def feedback_candidate(
            candidate_id: str,
            *,
            hold_forward_m: float = 0.0,
            hold_lateral_m: float = 0.0,
            feedback_forward_gain: float = 1.0,
            feedback_lateral_gain: float = 1.2,
            maximum_feedback_forward_m: float = 0.025,
        ) -> dict[str, object]:
            return {
                "id": candidate_id,
                "mode": "blend_to_nominal_stance",
                "lift_duration_s": 3.0,
                "forward_offset_m": 0.120,
                "support_extension_m_by_leg": {"front_right": 0.050},
                "hold_forward_m": hold_forward_m,
                "hold_lateral_m": hold_lateral_m,
                "feedback_forward_gain": feedback_forward_gain,
                "feedback_lateral_gain": feedback_lateral_gain,
                "maximum_feedback_forward_m": maximum_feedback_forward_m,
            }

        result.extend(
            [
                feedback_candidate("full-feedback-baseline"),
                feedback_candidate(
                    "full-feedback-hold-lateral-plus5mm",
                    hold_lateral_m=0.005,
                ),
                feedback_candidate(
                    "full-feedback-hold-lateral-plus8mm",
                    hold_lateral_m=0.0075,
                ),
                feedback_candidate(
                    "full-feedback-hold-lateral-plus10mm",
                    hold_lateral_m=0.010,
                ),
                feedback_candidate(
                    "full-feedback-hold-forward-plus10mm",
                    hold_forward_m=0.010,
                ),
                feedback_candidate(
                    "full-feedback-forward-gain1.5",
                    feedback_forward_gain=1.5,
                    maximum_feedback_forward_m=0.040,
                ),
                feedback_candidate(
                    "full-feedback-forward-gain2.0",
                    feedback_forward_gain=2.0,
                    maximum_feedback_forward_m=0.050,
                ),
                feedback_candidate(
                    "full-feedback-lateral8mm-forward-gain1.5",
                    hold_lateral_m=0.0075,
                    feedback_forward_gain=1.5,
                    maximum_feedback_forward_m=0.040,
                ),
            ]
        )
        return result
    if args.search_mode == "full-sequence":
        for extension_m in (0.040, 0.050, 0.060, 0.070, 0.080, 0.090):
            result.append(
                {
                    "id": (
                        "full-sequence-front-right-extend"
                        f"{int(round(extension_m * 1000)):02d}mm"
                    ),
                    "mode": "blend_to_nominal_stance",
                    "lift_duration_s": 3.0,
                    "forward_offset_m": 0.120,
                    "support_extension_m_by_leg": {
                        "front_right": extension_m,
                    },
                }
            )
        return result
    if args.search_mode == "validate":
        for static_friction, dynamic_friction in (
            (0.90, 0.75),
            (1.05, 0.90),
            (1.20, 1.00),
        ):
            for effort_cap_nm in (0.75, 0.82, 0.8825985):
                result.append(
                    {
                        "id": (
                            f"validate-mu{static_friction:.2f}-{dynamic_friction:.2f}-"
                            f"cap{effort_cap_nm:.3f}Nm"
                        ),
                        "mode": "blend_to_nominal_stance",
                        "lift_duration_s": 3.0,
                        "forward_offset_m": 0.120,
                        "support_extension_m_by_leg": {"front_right": 0.040},
                        "static_friction": static_friction,
                        "dynamic_friction": dynamic_friction,
                        "effort_cap_nm": effort_cap_nm,
                    }
                )
        return result
    if args.search_mode == "support-fine":
        for extension_m in (0.035, 0.0375, 0.040, 0.0425, 0.045, 0.0475, 0.050):
            result.append(
                {
                    "id": f"front-right-extend{int(round(extension_m * 1000)):02d}mm",
                    "mode": "blend_to_nominal_stance",
                    "lift_duration_s": 3.0,
                    "forward_offset_m": 0.120,
                    "support_extension_m_by_leg": {
                        "front_right": extension_m,
                    },
                }
            )
        for extension_m, apex_lift_m in (
            (0.040, 0.210),
            (0.040, 0.215),
            (0.040, 0.220),
            (0.0425, 0.210),
            (0.0425, 0.215),
        ):
            result.append(
                {
                    "id": (
                        f"front-right-extend{int(round(extension_m * 1000)):02d}mm-"
                        f"apex{int(round(apex_lift_m * 1000)):03d}mm"
                    ),
                    "mode": "blend_to_nominal_stance",
                    "lift_duration_s": 3.0,
                    "forward_offset_m": 0.120,
                    "apex_lift_m": apex_lift_m,
                    "support_extension_m_by_leg": {
                        "front_right": extension_m,
                    },
                }
            )
        for lift_duration_s in (2.5, 3.5):
            result.append(
                {
                    "id": f"front-right-extend40mm-lift{lift_duration_s:.1f}s",
                    "mode": "blend_to_nominal_stance",
                    "lift_duration_s": lift_duration_s,
                    "forward_offset_m": 0.120,
                    "support_extension_m_by_leg": {"front_right": 0.040},
                }
            )
        return result
    if args.search_mode == "support":
        zero_action = [0.0] * 12
        abduction_relax = zero_action.copy()
        abduction_relax[1:4] = [1.0, -1.0, -1.0]
        full_relax = abduction_relax.copy()
        full_relax[5:8] = [0.80, 0.15, 0.25]
        full_relax[9:12] = [-0.20, 0.10, -0.40]

        def support_candidate(
            candidate_id: str,
            *,
            extensions: dict[str, float] | None = None,
            hold_forward_m: float = 0.0,
            hold_lateral_m: float = 0.0,
            support_action: list[float] | None = None,
        ) -> dict[str, object]:
            return {
                "id": candidate_id,
                "mode": "blend_to_nominal_stance",
                "lift_duration_s": 3.0,
                "forward_offset_m": 0.120,
                "support_extension_m_by_leg": extensions or {},
                "hold_forward_m": hold_forward_m,
                "hold_lateral_m": hold_lateral_m,
                "support_action": support_action or zero_action,
            }

        result.append(support_candidate("support-baseline"))
        for extension_m in (0.010, 0.020, 0.030, 0.040):
            extensions = {
                leg: extension_m
                for leg in ("front_right", "rear_left", "rear_right")
            }
            result.append(
                support_candidate(
                    f"support-all-extend{int(extension_m * 1000):02d}mm",
                    extensions=extensions,
                )
            )
        for group, legs in (
            ("rear", ("rear_left", "rear_right")),
            ("front-right", ("front_right",)),
        ):
            for extension_m in (0.020, 0.040):
                result.append(
                    support_candidate(
                        f"support-{group}-extend{int(extension_m * 1000):02d}mm",
                        extensions={leg: extension_m for leg in legs},
                    )
                )
        result.extend(
            [
                support_candidate(
                    "support-hold-forward-minus15mm",
                    hold_forward_m=-0.015,
                ),
                support_candidate(
                    "support-hold-forward-plus15mm",
                    hold_forward_m=0.015,
                ),
                support_candidate(
                    "support-hold-lateral-minus15mm",
                    hold_lateral_m=-0.015,
                ),
                support_candidate(
                    "support-hold-lateral-plus15mm",
                    hold_lateral_m=0.015,
                ),
                support_candidate(
                    "support-abduction-target-relax",
                    support_action=abduction_relax,
                ),
                support_candidate(
                    "support-full-target-relax",
                    support_action=full_relax,
                ),
                support_candidate(
                    "support-all-extend20mm-abduction-relax",
                    extensions={
                        leg: 0.020
                        for leg in ("front_right", "rear_left", "rear_right")
                    },
                    support_action=abduction_relax,
                ),
                support_candidate(
                    "support-all-extend20mm-full-relax",
                    extensions={
                        leg: 0.020
                        for leg in ("front_right", "rear_left", "rear_right")
                    },
                    support_action=full_relax,
                ),
            ]
        )
        return result
    for mode in ("blend_to_nominal_stance", "phase_baseline"):
        for lift_duration_s in (2.0, 3.0, 4.0, 5.0):
            for forward_offset_m in (0.080, 0.100, 0.120):
                result.append(
                    {
                        "id": (
                            f"{mode}-lift{lift_duration_s:.0f}s-"
                            f"forward{int(round(forward_offset_m * 1000)):03d}mm"
                        ),
                        "mode": mode,
                        "lift_duration_s": lift_duration_s,
                        "forward_offset_m": forward_offset_m,
                    }
                )
    return result


report: dict[str, object] = {
    "status": "FAIL",
    "config": str(config_path),
    "world": str(world_path),
    "precursor_model": str(precursor_model_path),
    "swing_model": str(swing_model_path),
    "seed": args.seed,
    "search_mode": args.search_mode,
    "candidate_seconds": args.candidate_seconds,
    "stair_tread_depth_m": float(task_config["staircase"]["tread_depth_m"]),
    "stair_rise_m": float(task_config["staircase"]["rise_m"]),
    "effort_cap_nm": float(task_config["robot_hardware_profile"]["effort_cap_nm"]),
    "candidates": [],
}
raw_env: QuadrupedStairsEnv | None = None
try:
    raw_env = QuadrupedStairsEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
    )
    raw_env.set_evaluation_level(4)
    raw_env.set_placement_level(
        "left-supported-190mm-lift",
        activate_immediately=True,
    )
    precursor_model = PPO.load(str(precursor_model_path), device="cpu")
    swing_model = PPO.load(str(swing_model_path), device="cpu")
    phase_env = PlacementPhaseTrainingEnv(
        raw_env,
        target_leg="front_left",
        precursor_policies={"front_right": precursor_model},
        cache_phase_snapshot=True,
    )
    swing_mask = placement_policy_action_mask(
        raw_env.dof_names,
        target_leg="front_left",
        mode="swing_only",
    )
    original_timing_by_leg = deepcopy(
        raw_env.placement_reference_config.get("timing_override_by_leg", {})
    )
    original_com_regulation = deepcopy(raw_env.com_regulation_config)
    candidate_results: list[dict[str, object]] = []
    if args.search_mode == "validate":
        phase_env.reset(seed=args.seed)
    for index, candidate in enumerate(candidates()):
        if "static_friction" in candidate:
            raw_env.set_foot_contact_friction(
                static_friction=float(candidate["static_friction"]),
                dynamic_friction=float(candidate["dynamic_friction"]),
            )
        effort_cap_nm = float(
            candidate.get(
                "effort_cap_nm",
                task_config["robot_hardware_profile"]["effort_cap_nm"],
            )
        )
        raw_env.robot.set_dof_max_efforts(
            np.full(12, effort_cap_nm, dtype=np.float32)
        )
        raw_env.effort_cap_nm = effort_cap_nm
        if args.search_mode in {"full-sequence", "full-feedback"}:
            observation, _ = raw_env.reset(seed=args.seed + index)
        else:
            observation, _ = phase_env.reset(seed=args.seed + index)
        timing_by_leg = deepcopy(original_timing_by_leg)
        front_left_timing = dict(timing_by_leg.get("front_left", {}))
        lift_duration = float(candidate["lift_duration_s"])
        lift_start = float(front_left_timing.get("lift_start_seconds", 0.5))
        front_left_timing["lift_duration_seconds"] = lift_duration
        front_left_timing["advance_start_seconds"] = lift_start + lift_duration
        front_left_timing["advance_duration_seconds"] = 3.0
        front_left_timing["lower_start_seconds"] = (
            float(front_left_timing["advance_start_seconds"]) + 3.0
        )
        front_left_timing["lower_duration_seconds"] = 1.5
        timing_by_leg["front_left"] = front_left_timing
        raw_env.placement_reference_config["timing_override_by_leg"] = timing_by_leg
        raw_env.inter_leg_transfer_config["post_transfer_swing_reference_mode"] = str(
            candidate["mode"]
        )
        raw_env.com_regulation_config = deepcopy(original_com_regulation)
        hold_offsets = dict(
            raw_env.com_regulation_config.get(
                "hold_target_offset_by_swing_leg",
                {},
            )
        )
        hold_offsets["front_left"] = {
            "forward": float(candidate.get("hold_forward_m", 0.0)),
            "lateral": float(candidate.get("hold_lateral_m", 0.0)),
        }
        raw_env.com_regulation_config["hold_target_offset_by_swing_leg"] = (
            hold_offsets
        )
        feedback_gain = dict(
            raw_env.com_regulation_config.get("feedback_gain", {})
        )
        feedback_gain["forward"] = float(
            candidate.get(
                "feedback_forward_gain",
                feedback_gain.get("forward", 1.0),
            )
        )
        feedback_gain["lateral"] = float(
            candidate.get(
                "feedback_lateral_gain",
                feedback_gain.get("lateral", 1.2),
            )
        )
        raw_env.com_regulation_config["feedback_gain"] = feedback_gain
        maximum_feedback = dict(
            raw_env.com_regulation_config.get("maximum_feedback_correction_m", {})
        )
        maximum_feedback["forward"] = float(
            candidate.get(
                "maximum_feedback_forward_m",
                maximum_feedback.get("forward", 0.025),
            )
        )
        raw_env.com_regulation_config["maximum_feedback_correction_m"] = (
            maximum_feedback
        )
        support_extensions = dict(
            raw_env.com_regulation_config.get(
                "support_extension_m_by_swing_leg",
                {},
            )
        )
        support_extensions["front_left"] = dict(
            candidate.get("support_extension_m_by_leg", {})
        )
        raw_env.com_regulation_config["support_extension_m_by_swing_leg"] = (
            support_extensions
        )
        if raw_env.current_placement_level is None:
            raise RuntimeError("Placement level was not restored")
        forward_offset = float(candidate["forward_offset_m"])
        for key in (
            "lift_forward_offset_m",
            "swing_forward_offset_m",
            "landing_forward_offset_m",
        ):
            raw_env.current_placement_level[key] = forward_offset
        if "apex_lift_m" in candidate:
            raw_env.current_placement_level["apex_lift_m"] = float(
                candidate["apex_lift_m"]
            )

        maximum_steps = int(round(float(args.candidate_seconds) * raw_env.control_hz))
        last_info: dict[str, object] = {}
        terminated = False
        truncated = False
        steps_taken = 0
        for _step in range(maximum_steps):
            steps_taken += 1
            if raw_env.placement_transfer_active:
                applied_action = np.zeros(12, dtype=np.float32)
            elif raw_env.placement_swing_leg == "front_right":
                applied_action, _ = predict_with_observation_prefix(
                    precursor_model,
                    observation,
                    deterministic=True,
                )
            else:
                action, _ = predict_with_observation_prefix(
                    swing_model,
                    observation,
                    deterministic=True,
                )
                swing_bias_by_kind = dict(
                    candidate.get("swing_action_bias", {})
                )
                swing_action_bias = np.asarray(
                    [
                        next(
                            (
                                float(value)
                                for kind, value in swing_bias_by_kind.items()
                                if name.endswith(kind)
                            ),
                            0.0,
                        )
                        for name in raw_env.dof_names
                    ],
                    dtype=np.float32,
                )
                if (
                    bool(candidate.get("swing_action_bias_lift_only", False))
                    and float(observation[69]) <= 0.5
                ):
                    swing_action_bias.fill(0.0)
                support_action = np.asarray(
                    candidate.get("support_action", np.zeros(12)),
                    dtype=np.float32,
                )
                applied_action = np.clip(
                    (np.asarray(action, dtype=np.float32) + swing_action_bias)
                    * swing_mask
                    + support_action * (1.0 - swing_mask),
                    -1.0,
                    1.0,
                )
            observation, _, terminated, truncated, last_info = raw_env.step(
                applied_action
            )
            if terminated or truncated:
                break
        metrics = dict(last_info.get("joint_effort_metrics", {}))
        episode_metrics = dict(last_info.get("episode_metrics", {}))
        lift_by_leg = dict(last_info.get("maximum_foot_lift_m_by_leg", {}))
        maximum_lift_m = float(lift_by_leg.get("front_left", 0.0))
        maximum_slip_m = float(last_info.get("maximum_support_slip_m", 0.0))
        maximum_tilt_deg = float(raw_env.maximum_tilt_deg)
        failure_reasons = list(last_info.get("failure_reasons", ()))
        placement_completed = bool(
            episode_metrics.get("placement_completed", False)
        )
        acceptance_passed = bool(
            placement_completed
            and maximum_lift_m >= 0.190
            and maximum_slip_m < 0.025
            and maximum_tilt_deg < 12.0
            and not failure_reasons
        )
        result = {
            **candidate,
            "steps": steps_taken,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "placement_completed": placement_completed,
            "acceptance_passed": acceptance_passed,
            "failure_reasons": failure_reasons,
            "maximum_front_left_lift_m": maximum_lift_m,
            "maximum_support_slip_m": maximum_slip_m,
            "minimum_support_margin_m": float(
                raw_env.minimum_placement_support_margin_m
            ),
            "maximum_body_tilt_deg": maximum_tilt_deg,
            "maximum_balance_lateral_deviation_m": float(
                raw_env.maximum_balance_lateral_deviation_m
            ),
            "joint_effort_metrics": metrics,
            "final_target_joint_positions_rad": np.asarray(
                last_info.get("target_joint_positions_rad", np.zeros(12))
            ).tolist(),
            "final_joint_tracking_error_rad": np.asarray(
                last_info.get("joint_tracking_error_rad", np.zeros(12))
            ).tolist(),
        }
        result["score"] = float(
            1000.0 * float(result["maximum_front_left_lift_m"])
            - 600.0 * float(result["maximum_support_slip_m"])
            - 2.0 * float(result["maximum_body_tilt_deg"])
            - 2.0
            * float(metrics.get("requested_pd_effort_95pct_cap_sample_fraction", 1.0))
        )
        candidate_results.append(result)
        print(
            "DROBOT_TRANSFER_CANDIDATE="
            + json.dumps(
                {
                    "id": result["id"],
                    "lift_mm": round(
                        1000.0 * float(result["maximum_front_left_lift_m"]), 1
                    ),
                    "slip_mm": round(
                        1000.0 * float(result["maximum_support_slip_m"]), 1
                    ),
                    "tilt_deg": round(float(result["maximum_body_tilt_deg"]), 1),
                    "pd_saturation_fraction": metrics.get(
                        "requested_pd_effort_95pct_cap_sample_fraction"
                    ),
                    "completed": result["placement_completed"],
                    "accepted": result["acceptance_passed"],
                    "failures": result["failure_reasons"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    candidate_results.sort(key=lambda item: float(item["score"]), reverse=True)
    report["candidates"] = candidate_results
    report["best_candidate"] = candidate_results[0]
    report["status"] = "PASS"
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
    simulation_app.close()

if report["status"] != "PASS":
    raise SystemExit(1)
