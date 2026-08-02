"""Probe constant stance-hip actions from the V45 rear-left boundary.

This is a controller-authority audit, not a learned policy.  Every candidate
restores the same accepted V44 physical snapshot and changes only the three
loaded support-leg hip-abduction residuals.  The result identifies which
action signs can reduce the rear-left transfer COM error before another PPO
budget is spent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from itertools import product
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
    default=str(SCRIPT_DIR / "quadruped_stairs_v45_rear_left_transfer.yaml"),
)
parser.add_argument(
    "--phase-snapshot",
    default=(
        "simulation/isaac/output/rl/"
        "v45-rear-left-transfer-start-seed870.json"
    ),
)
parser.add_argument(
    "--amplitudes",
    default="-1.0,0.0,1.0",
    help="Comma-separated normalized actions for each support hip abduction.",
)
parser.add_argument(
    "--policy-model",
    default=None,
    help="Optional PPO model evaluated instead of constant hip actions.",
)
parser.add_argument("--seed", type=int, default=874)
parser.add_argument(
    "--maximum-seconds",
    type=float,
    default=5.0,
    help="Maximum simulated duration for each constant-action candidate.",
)
parser.add_argument(
    "--report",
    default=(
        "simulation/isaac/output/rl/"
        "rear-left-transfer-support-action-search-v45-seed874.json"
    ),
)
parser.add_argument("--record-video", default=None)
parser.add_argument("--record-thumbnail", default=None)
parser.add_argument("--record-fps", type=int, default=30)
parser.add_argument("--record-width", type=int, default=960)
parser.add_argument("--record-height", type=int, default=540)
args, _ = parser.parse_known_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def comma_floats(value: str) -> tuple[float, ...]:
    parsed = tuple(
        float(item.strip()) for item in value.split(",") if item.strip()
    )
    if (
        not parsed
        or not all(math.isfinite(item) for item in parsed)
        or any(abs(item) > 1.0 for item in parsed)
    ):
        raise ValueError("amplitudes must be finite values within [-1, 1]")
    return parsed


config_path = project_path(args.config)
snapshot_path = project_path(args.phase_snapshot)
report_path = project_path(args.report)
policy_model_path = project_path(args.policy_model) if args.policy_model else None
video_path = project_path(args.record_video) if args.record_video else None
thumbnail_path = (
    project_path(args.record_thumbnail)
    if args.record_thumbnail
    else (video_path.with_suffix(".png") if video_path else None)
)
amplitudes = comma_floats(args.amplitudes)
if args.maximum_seconds <= 0.0:
    parser.error("--maximum-seconds must be positive")
if args.record_fps < 1 or args.record_width < 1 or args.record_height < 1:
    parser.error("recording FPS and dimensions must be positive")
if video_path is not None and policy_model_path is None:
    parser.error("--record-video requires --policy-model")

with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
task_config = dict(config["task"])
with snapshot_path.open("r", encoding="utf-8") as stream:
    snapshot_payload = json.load(stream)

expected_snapshot_contract = {
    "target_leg": "rear_left",
    "phase_snapshot_mode": "inter_leg_transfer",
    "stair_rise_m": float(task_config["staircase"]["rise_m"]),
    "stair_tread_depth_m": float(task_config["staircase"]["tread_depth_m"]),
    "effort_cap_nm": float(task_config["robot_hardware_profile"]["effort_cap_nm"]),
    "placement_sequence_legs": list(
        task_config["placement_reference"]["sequence_legs"]
    ),
}
for field, expected in expected_snapshot_contract.items():
    if snapshot_payload.get(field) != expected:
        raise ValueError(
            f"phase snapshot {field} mismatch: "
            f"{snapshot_payload.get(field)!r} != {expected!r}"
        )
phase_snapshot = dict(snapshot_payload["snapshot"])
world_path = project_path(task_config["world"])

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": True,
        "width": args.record_width,
        "height": args.record_height,
    }
)

from _placement_phase_training import PlacementPhaseTrainingEnv  # noqa: E402
from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: E402
from _stair_rl_contract import placement_policy_action_mask  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from omni.kit.viewport.utility import get_active_viewport  # noqa: E402
from PIL import Image  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from video_encoding import get_video_encoding_interface  # noqa: E402

report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "scope": "Exact-snapshot rear-left transfer stance-action authority audit",
    "config": str(config_path),
    "phase_snapshot": str(snapshot_path),
    "seed": args.seed,
    "stair_rise_m": expected_snapshot_contract["stair_rise_m"],
    "stair_tread_depth_m": expected_snapshot_contract["stair_tread_depth_m"],
    "effort_cap_nm": expected_snapshot_contract["effort_cap_nm"],
    "amplitudes": list(amplitudes),
    "candidate_count": 1 if policy_model_path is not None else len(amplitudes) ** 3,
    "maximum_seconds": args.maximum_seconds,
    "policy_model": str(policy_model_path) if policy_model_path else None,
    "record_video": str(video_path) if video_path else None,
    "record_thumbnail": str(thumbnail_path) if thumbnail_path else None,
    "camera_policy_input": False,
}
raw_env: QuadrupedStairsEnv | None = None
camera_sensor: CameraSensor | None = None
exit_code = 1
try:
    raw_env = QuadrupedStairsEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
    )
    raw_env.set_evaluation_level(1)
    raw_env.set_placement_level("left-center-tread-load")
    if video_path is not None:
        if int(task_config["control_hz"]) % args.record_fps:
            raise RuntimeError("--record-fps must divide the control rate")
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("Isaac Sim has no active viewport")
        stair = task_config["staircase"]
        camera_center_x = float(stair["start_x_m"]) + 0.10
        camera_path = "/OmniverseKit_Persp"
        set_camera_view(
            eye=[camera_center_x, -1.25, 0.62],
            target=[camera_center_x, 0.0, 0.18],
            camera_prim_path=camera_path,
        )
        viewport.camera_path = camera_path
        camera_prim = raw_env.stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            raise RuntimeError(f"Recording camera prim is missing: {camera_path}")
        if "OmniSensorAPI" not in camera_prim.GetAppliedSchemas():
            camera_prim.ApplyAPI("OmniSensorAPI")
        camera_sensor = CameraSensor(
            RtxCamera(
                camera_path,
                tick_rate=None,
                reset_xform_op_properties=False,
            ),
            resolution=(args.record_height, args.record_width),
            annotators=["rgb"],
        )
    support_mask = placement_policy_action_mask(
        raw_env.dof_names,
        target_leg="rear_left",
        mode="support_only",
    )
    compact_indices = np.flatnonzero(support_mask)
    compact_names = [raw_env.dof_names[index] for index in compact_indices]
    abduction_positions = tuple(
        compact_names.index(name)
        for name in (
            "front_left_hip_abduction",
            "front_right_hip_abduction",
            "rear_right_hip_abduction",
        )
    )
    phase_env = PlacementPhaseTrainingEnv(
        raw_env,
        target_leg="rear_left",
        precursor_policies={},
        target_residual_mask=(None if policy_model_path else support_mask),
        compact_residual_action=bool(policy_model_path is None),
        train_transfer=True,
        initial_phase_snapshot=phase_snapshot,
        initial_phase_snapshot_mode="inter_leg_transfer",
    )
    policy_model = (
        PPO.load(str(policy_model_path), device="cpu")
        if policy_model_path is not None
        else None
    )
    if policy_model is not None and tuple(policy_model.action_space.shape) != tuple(
        phase_env.action_space.shape
    ):
        raise RuntimeError(
            "Policy action shape differs from phase environment: "
            f"{policy_model.action_space.shape} != {phase_env.action_space.shape}"
        )
    maximum_steps = int(math.ceil(args.maximum_seconds * raw_env.control_hz))
    candidates: list[dict[str, object]] = []
    candidate_values = (
        ((0.0, 0.0, 0.0),)
        if policy_model is not None
        else product(amplitudes, repeat=3)
    )
    for values in candidate_values:
        action = np.zeros(phase_env.action_space.shape, dtype=np.float32)
        action[list(abduction_positions)] = values
        observation, reset_info = phase_env.reset(seed=args.seed)
        initial_error = float(
            np.linalg.norm(
                np.asarray(
                    reset_info.get(
                        "placement_balance_target_error_xy_m",
                        np.asarray(
                            phase_snapshot[
                                "placement_transfer_start_balance_position_m"
                            ],
                            dtype=np.float64,
                        )[:2]
                        - np.asarray(
                            phase_snapshot[
                                "placement_transfer_target_balance_position_m"
                            ],
                            dtype=np.float64,
                        )[:2],
                    ),
                    dtype=np.float64,
                )
            )
        )
        minimum_error = initial_error
        maximum_margin = -math.inf
        maximum_tilt = 0.0
        final_error = initial_error
        final_margin = -math.inf
        last_info: dict[str, object] = dict(reset_info)
        completed = False
        terminated = False
        truncated = False
        steps = 0
        maximum_policy_action_abs = 0.0
        final_policy_action = np.zeros(
            phase_env.action_space.shape,
            dtype=np.float32,
        )
        candidate_frames: list[np.ndarray] = []
        for step_index in range(1, maximum_steps + 1):
            steps = step_index
            step_action = action
            if policy_model is not None:
                predicted, _ = policy_model.predict(
                    observation,
                    deterministic=True,
                )
                step_action = np.asarray(predicted, dtype=np.float32)
                final_policy_action = step_action.copy()
                maximum_policy_action_abs = max(
                    maximum_policy_action_abs,
                    float(np.max(np.abs(step_action))),
                )
            observation, _, terminated, truncated, info = phase_env.step(
                step_action
            )
            last_info = dict(info)
            if (
                camera_sensor is not None
                and step_index
                % (int(task_config["control_hz"]) // args.record_fps)
                == 0
            ):
                rgb_data, _ = camera_sensor.get_data("rgb")
                if rgb_data is None:
                    raise RuntimeError(
                        f"Recording camera returned no frame at step {step_index}"
                    )
                if hasattr(rgb_data, "numpy"):
                    rgb_data = rgb_data.numpy()
                rgb = np.asarray(rgb_data)
                if rgb.shape[:2] != (args.record_height, args.record_width):
                    raise RuntimeError(
                        f"Unexpected recording frame shape: {rgb.shape}"
                    )
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                candidate_frames.append(
                    np.ascontiguousarray(rgb[..., :3]).copy()
                )
            final_error = float(
                np.linalg.norm(
                    np.asarray(
                        info.get("placement_balance_target_error_xy_m", (0, 0)),
                        dtype=np.float64,
                    )
                )
            )
            final_margin = float(info.get("placement_support_margin_m", 0.0))
            minimum_error = min(minimum_error, final_error)
            maximum_margin = max(maximum_margin, final_margin)
            maximum_tilt = max(
                maximum_tilt,
                math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                info.get("placement_upright_cosine", 1.0),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                ),
            )
            if info.get("phase_training_transfer_completed"):
                completed = True
                break
            if terminated or truncated:
                break
        action_by_joint = (
            {
                name: float(final_policy_action[index])
                for index, name in enumerate(raw_env.dof_names)
            }
            if policy_model is not None
            else {
                compact_names[position]: float(action[position])
                for position in abduction_positions
            }
        )
        result = {
            "id": (
                "deterministic-policy"
                if policy_model is not None
                else "-".join(f"{value:+.1f}" for value in values)
            ),
            "action_by_joint": action_by_joint,
            "maximum_policy_action_abs": (
                maximum_policy_action_abs
                if policy_model is not None
                else None
            ),
            "steps": steps,
            "completed": completed,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "failure_reasons": list(last_info.get("failure_reasons", ())),
            "initial_balance_target_error_m": initial_error,
            "minimum_balance_target_error_m": minimum_error,
            "final_balance_target_error_m": final_error,
            "balance_target_error_reduction_m": initial_error - final_error,
            "maximum_support_margin_m": maximum_margin,
            "final_support_margin_m": final_margin,
            "maximum_body_tilt_deg": maximum_tilt,
            "final_balance_position_m": np.asarray(
                last_info.get("placement_balance_position_m", (0, 0, 0)),
                dtype=np.float64,
            ).tolist(),
            "final_target_error_xy_m": np.asarray(
                last_info.get("placement_balance_target_error_xy_m", (0, 0)),
                dtype=np.float64,
            ).tolist(),
        }
        if video_path is not None:
            if not candidate_frames:
                raise RuntimeError("Policy replay produced no recording frames")
            video_path.parent.mkdir(parents=True, exist_ok=True)
            assert thumbnail_path is not None
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            encoder = get_video_encoding_interface()
            if encoder is None or not encoder.start_encoding(
                str(video_path),
                float(args.record_fps),
                0,
                True,
            ):
                raise RuntimeError("Could not initialize the video encoder")
            for frame_index, rgb in enumerate(candidate_frames):
                alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
                rgba = np.ascontiguousarray(np.concatenate((rgb, alpha), axis=2))
                if not encoder.encode_next_frame_from_buffer(
                    rgba,
                    args.record_width,
                    args.record_height,
                ):
                    raise RuntimeError(
                        f"Video encoder rejected frame {frame_index}"
                    )
            encoder.finalize_encoding()
            Image.fromarray(candidate_frames[-1], mode="RGB").save(
                thumbnail_path
            )
            result["recording"] = {
                "status": "PASS",
                "frames": len(candidate_frames),
                "fps": args.record_fps,
                "resolution_wh": [args.record_width, args.record_height],
                "video": str(video_path),
                "video_bytes": video_path.stat().st_size,
                "thumbnail": str(thumbnail_path),
            }
        candidates.append(result)
        print(
            "DROBOT_REAR_LEFT_SUPPORT_ACTION="
            + json.dumps(
                {
                    "id": result["id"],
                    "error_reduction_mm": round(
                        1000.0 * float(result["balance_target_error_reduction_m"]),
                        2,
                    ),
                    "final_margin_mm": round(1000.0 * final_margin, 2),
                    "tilt_deg": round(maximum_tilt, 2),
                    "completed": completed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    candidates.sort(
        key=lambda item: (
            bool(item["completed"]),
            float(item["balance_target_error_reduction_m"]),
            float(item["maximum_support_margin_m"]),
            -float(item["maximum_body_tilt_deg"]),
        ),
        reverse=True,
    )
    report.update(
        {
            "status": "PASS",
            "support_action_indices": compact_indices.tolist(),
            "support_dof_names": compact_names,
            "probed_abduction_action_positions": list(abduction_positions),
            "completed_count": sum(
                bool(item["completed"]) for item in candidates
            ),
            "task_success": any(
                bool(item["completed"]) for item in candidates
            ),
            "best": candidates[0],
            "ranked_candidates": candidates,
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
    print("DROBOT_REAR_LEFT_SUPPORT_ACTION_SEARCH=" + json.dumps(report))
    simulation_app.close()

sys.exit(exit_code)
