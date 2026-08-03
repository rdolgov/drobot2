"""Capture a stable front-left unload boundary for stationary phase training.

The verified V75 policy first places front-right on the 180 mm tread. V90 then
replays its best known front-left transfer while this tool watches for a short
stable window before the motion tips or slips. The lowest-load state in that
window is re-anchored to a zero-displacement COM target and saved as a verified
simulator snapshot. V94 PPO can therefore learn unloading without replaying the
destabilizing moving transfer on every episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
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

from _run_support import read_model_manifest, sha256_file  # noqa: E402
from _stair_rl_contract import (  # noqa: E402
    config_for_first_tread_experiment,
    reanchor_inter_leg_transfer_snapshot,
    stable_transfer_snapshot_gate_failures,
)


def _project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _verified_model(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = read_model_manifest(path)
    model_hash = sha256_file(path)
    if manifest.get("model_sha256") != model_hash:
        raise RuntimeError(f"Model hash mismatch: {path}")
    return {
        "path": str(path),
        "sha256": model_hash,
        "source_task_id": manifest.get("task_id"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            SCRIPT_DIR / "quadruped_stairs_v14_front_pair_right_then_left.yaml"
        ),
    )
    parser.add_argument(
        "--first-tread-profile",
        default="front-pair-preposition-load-advance-forward-floor",
    )
    parser.add_argument("--placement-level", default="left-quarter-tread-load")
    parser.add_argument("--seed", type=int, default=1047)
    parser.add_argument("--maximum-steps", type=int, default=720)
    parser.add_argument("--stable-hold-seconds", type=float, default=0.10)
    parser.add_argument("--maximum-swing-load-n", type=float, default=12.0)
    parser.add_argument("--minimum-upright-cosine", type=float, default=0.975)
    parser.add_argument("--minimum-support-margin-m", type=float, default=0.015)
    parser.add_argument("--maximum-support-slip-m", type=float, default=0.025)
    parser.add_argument("--maximum-base-speed-m-s", type=float, default=0.050)
    parser.add_argument("--maximum-body-rate-rad-s", type=float, default=0.25)
    parser.add_argument(
        "--precursor-model",
        default=(
            "simulation/isaac/output/rl/"
            "ppo-stairs-v75-first-strict-foothold-2048-seed1025/"
            "drobot_stairs_ppo_final.zip"
        ),
    )
    parser.add_argument(
        "--transfer-model",
        default=(
            "simulation/isaac/models/"
            "ppo-stairs-v90-frozen-v88-support-knees-sustained-unload-8192-seed1034/"
            "drobot_stairs_ppo_final.zip"
        ),
    )
    parser.add_argument(
        "--snapshot",
        default=(
            "simulation/isaac/output/rl/"
            "front-left-stationary-unload-snapshot-v94-seed1047.json"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "simulation/isaac/output/rl/"
            "front-left-stationary-unload-snapshot-v94-seed1047-report.json"
        ),
    )
    args, _ = parser.parse_known_args()

    if args.maximum_steps < 1:
        parser.error("--maximum-steps must be positive")
    if not 0.0 < args.stable_hold_seconds <= 2.0:
        parser.error("--stable-hold-seconds must be within (0, 2]")
    if args.maximum_swing_load_n <= 0.0:
        parser.error("--maximum-swing-load-n must be positive")
    if not 0.0 < args.minimum_upright_cosine <= 1.0:
        parser.error("--minimum-upright-cosine must be within (0, 1]")

    config_path = _project_path(args.config)
    snapshot_path = _project_path(args.snapshot)
    report_path = _project_path(args.report)
    precursor_path = _project_path(args.precursor_model)
    transfer_path = _project_path(args.transfer_model)
    with config_path.open("r", encoding="utf-8") as stream:
        loaded_config = yaml.safe_load(stream)
    task_config = config_for_first_tread_experiment(
        loaded_config["task"],
        args.first_tread_profile,
    )
    world_path = _project_path(str(task_config["world"]))
    model_records = {
        "precursor": _verified_model(precursor_path),
        "transfer": _verified_model(transfer_path),
    }

    from isaacsim import SimulationApp  # noqa: PLC0415

    simulation_app = SimulationApp({"headless": True})
    raw_env = None
    report: dict[str, object] = {
        "status": "FAIL",
        "task_id": task_config["id"],
        "config": str(config_path),
        "world": str(world_path),
        "seed": args.seed,
        "stair_rise_m": float(task_config["staircase"]["rise_m"]),
        "stair_tread_depth_m": float(task_config["staircase"]["tread_depth_m"]),
        "effort_cap_nm": float(
            task_config["robot_hardware_profile"]["effort_cap_nm"]
        ),
        "rgb_camera_policy_input": False,
        "models": model_records,
        "candidate_gate": {
            "maximum_swing_load_n": args.maximum_swing_load_n,
            "minimum_upright_cosine": args.minimum_upright_cosine,
            "minimum_support_margin_m": args.minimum_support_margin_m,
            "maximum_support_slip_m": args.maximum_support_slip_m,
            "maximum_base_speed_m_s": args.maximum_base_speed_m_s,
            "maximum_body_rate_rad_s": args.maximum_body_rate_rad_s,
            "stable_hold_seconds": args.stable_hold_seconds,
        },
    }
    exit_code = 1
    try:
        from _placement_phase_training import (  # noqa: PLC0415
            PlacementPhaseTrainingEnv,
        )
        from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: PLC0415
        from stable_baselines3 import PPO  # noqa: PLC0415

        raw_env = QuadrupedStairsEnv(
            simulation_app,
            world_path=str(world_path),
            task_config=task_config,
        )
        raw_env.set_evaluation_level(1)
        raw_env.set_placement_level(args.placement_level, activate_immediately=True)
        phase_env = PlacementPhaseTrainingEnv(
            raw_env,
            target_leg="front_left",
            precursor_policies={
                "front_right": PPO.load(str(precursor_path), device="cpu")
            },
            target_base_policy=PPO.load(str(transfer_path), device="cpu"),
            target_residual_scale=0.10,
            train_transfer=True,
            maximum_reset_attempts=32,
            cache_phase_snapshot=True,
        )
        phase_env.reset(seed=args.seed)
        zero_action = np.zeros(phase_env.action_space.shape, dtype=np.float32)
        hold_steps = max(1, int(math.ceil(args.stable_hold_seconds * raw_env.control_hz)))
        stable_window: deque[dict[str, object]] = deque(maxlen=hold_steps)
        best: dict[str, object] | None = None
        last_info: dict[str, object] = {}

        for step_index in range(1, args.maximum_steps + 1):
            _, _, terminated, truncated, info = phase_env.step(zero_action)
            last_info = dict(info)
            failures = stable_transfer_snapshot_gate_failures(
                transfer_gate_failures=tuple(
                    info.get("placement_transfer_gate_failures", ())
                ),
                swing_load_n=float(
                    info.get("placement_transfer_swing_total_load_n", math.inf)
                ),
                maximum_swing_load_n=args.maximum_swing_load_n,
                support_contact_fraction=float(
                    info.get("placement_support_contact_fraction", 0.0)
                ),
                support_margin_m=float(
                    info.get("placement_support_margin_m", -math.inf)
                ),
                minimum_support_margin_m=args.minimum_support_margin_m,
                support_slip_m=float(info.get("maximum_support_slip_m", math.inf)),
                maximum_support_slip_m=args.maximum_support_slip_m,
                upright_cosine=float(info.get("placement_upright_cosine", 0.0)),
                minimum_upright_cosine=args.minimum_upright_cosine,
                base_speed_m_s=float(
                    info.get("placement_transfer_base_speed_m_s", math.inf)
                ),
                maximum_base_speed_m_s=args.maximum_base_speed_m_s,
                body_rate_rad_s=float(
                    info.get("placement_transfer_body_rate_rad_s", math.inf)
                ),
                maximum_body_rate_rad_s=args.maximum_body_rate_rad_s,
            )
            if failures:
                stable_window.clear()
            else:
                captured = raw_env.capture_placement_phase_snapshot()
                references = raw_env._reference_parameters_from_joint_positions(  # noqa: SLF001
                    np.asarray(captured["joint_positions_rad"], dtype=np.float32)
                )
                balance_position = np.asarray(
                    info.get(
                        "placement_balance_position_m",
                        raw_env.latest_placement_com_position_m,
                    ),
                    dtype=np.float64,
                )
                stationary = reanchor_inter_leg_transfer_snapshot(
                    captured,
                    balance_position_m=balance_position,
                    target_delta_xy_m=(0.0, 0.0),
                    reference_by_leg=references,
                )
                stable_window.append(
                    {
                        "step": step_index,
                        "swing_load_n": float(
                            info["placement_transfer_swing_total_load_n"]
                        ),
                        "upright_cosine": float(info["placement_upright_cosine"]),
                        "support_margin_m": float(info["placement_support_margin_m"]),
                        "support_slip_m": float(info["maximum_support_slip_m"]),
                        "base_speed_m_s": float(
                            info["placement_transfer_base_speed_m_s"]
                        ),
                        "body_rate_rad_s": float(
                            info["placement_transfer_body_rate_rad_s"]
                        ),
                        "completed_tread_min_load_n": float(
                            info.get(
                                "placement_transfer_completed_tread_min_load_n",
                                0.0,
                            )
                        ),
                        "snapshot": stationary,
                    }
                )
                if len(stable_window) == hold_steps:
                    window_best = min(
                        stable_window,
                        key=lambda candidate: float(candidate["swing_load_n"]),
                    )
                    if best is None or float(window_best["swing_load_n"]) < float(
                        best["swing_load_n"]
                    ):
                        best = deepcopy(window_best)
            if terminated or truncated:
                break

        report["steps"] = step_index
        report["terminal_failure_reasons"] = list(
            last_info.get("failure_reasons", ())
        )
        if best is None:
            raise RuntimeError(
                "No stable front-left transfer snapshot met the capture gate"
            )
        snapshot = best.pop("snapshot")
        snapshot_payload = {
            "schema_version": 1,
            "source_task_id": task_config["id"],
            "target_leg": "front_left",
            "phase_snapshot_mode": "inter_leg_transfer",
            "stair_rise_m": float(task_config["staircase"]["rise_m"]),
            "stair_tread_depth_m": float(task_config["staircase"]["tread_depth_m"]),
            "effort_cap_nm": float(
                task_config["robot_hardware_profile"]["effort_cap_nm"]
            ),
            "placement_sequence_legs": list(
                task_config["placement_reference"]["sequence_legs"]
            ),
            "seed": args.seed,
            "capture_metrics": best,
            "snapshot": _json_compatible(snapshot),
        }
        snapshot_bytes = (
            json.dumps(snapshot_payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(snapshot_bytes)
        report.update(
            {
                "status": "PASS",
                "selected_candidate": best,
                "snapshot": {
                    "path": str(snapshot_path),
                    "sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
                },
            }
        )
        exit_code = 0
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(_json_compatible(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if raw_env is not None:
            raw_env.close()
        simulation_app.close()

    print("DROBOT_STAIRS_V94_SNAPSHOT_RESULT=" + json.dumps(report), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
