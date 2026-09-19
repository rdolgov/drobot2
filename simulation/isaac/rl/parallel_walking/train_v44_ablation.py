"""Train an isolated, UNQUALIFIED V44 simulation ablation; never export a policy.

Required: --env-overrides-json FILE --experiment-suffix lower-case-slug.
Other Isaac Lab flags and agent.* Hydra overrides are forwarded. Environment
changes must use the JSON file; --num_envs/--device/--seed remain ordinary CLI.
Each invocation saves preparation evidence before importing the simulator.
That evidence does not certify that training started or completed. Kit fast
shutdown can terminate the process before the optional post-return hook runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from v44_ablation_contract import load_v44_overrides, v44_hydra_overrides
from v44_training_contract import EXPERIMENT, TASK, digest, read_config_fields, training_snapshot


def ablation_experiment_name(suffix: str) -> str:
    """Short deterministic namespace; never aliases the production experiment."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", suffix):
        raise ValueError("Ablation suffix must be a 1-64 character lower-case slug")
    compact = suffix if len(suffix) <= 24 else suffix[:24] + "_" + hashlib.sha256(suffix.encode()).hexdigest()[:8]
    return "drobot_v44_ablation_" + compact


def launch_records_root(experiment_root: Path) -> Path:
    experiment_key = hashlib.sha256(experiment_root.name.encode()).hexdigest()[:16]
    return experiment_root.parent / "_v44_launch_records" / experiment_key


def require_short_training_paths(experiment_root: Path, run_name: str | None) -> None:
    """Reject risky Windows log/event paths without launching or changing OS settings."""
    run = experiment_root / ("2000-01-01_00-00-00" + ("_" + run_name if run_name else ""))
    # TensorBoard embeds time, hostname, PID and a per-writer identifier. Use
    # deliberately generous digit allowances instead of relying on long-path
    # support in TensorBoard, Kit or the subprocess Python executable.
    event_name = "events.out.tfevents." + "0" * 20 + "." + socket.gethostname() + "." + "0" * 10 + "." + "0" * 10
    if os.name == "nt" and len(str(run / event_name)) >= 240:
        raise ValueError("Training/event paths exceed the conservative 240-character Windows budget; use a shorter workspace or run label")


def prepare_launch_evidence(experiment_root: Path, manifest: dict) -> dict:
    """Persist exclusive, pre-simulator evidence; never infer completion from it.

    A unique invocation directory is outside the experiment/run hierarchy so
    checkpoint discovery cannot accidentally select it. Failure to snapshot or
    publish the record aborts this invocation before any simulator imports.
    """
    package = Path(__file__).resolve().parent
    project = package.parents[3]
    prepared_at = datetime.now(timezone.utc)
    invocation_uuid = uuid4().hex
    invocation_id = prepared_at.strftime("%Y%m%dT%H%M%S%fZ") + "-" + invocation_uuid
    records_root = launch_records_root(experiment_root)
    records_root.mkdir(parents=True, exist_ok=True)
    directory = records_root / invocation_uuid
    directory.mkdir(exist_ok=False)
    files = []

    def snapshot(source: Path, relative: Path, *, role: str, expected_sha256: str | None = None) -> dict:
        source = source.resolve(strict=True)
        # Preserve the original source layout in metadata, not a nested copy
        # path. The old full experiment/UTC/layout hierarchy exceeded MAX_PATH.
        storage_relative = Path("snapshot") / f"{len(files):05d}{source.suffix.lower()}"
        target = directory / storage_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation keeps even a partial invocation immutable on retry.
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        copied_hash = digest(target)
        if expected_sha256 is not None and copied_hash != expected_sha256:
            raise RuntimeError(f"Input changed while preparing launch evidence: {source}")
        record = {
            "role": role,
            "source_path": source.as_posix(),
            "snapshot_path": storage_relative.as_posix(),
            "source_layout_path": relative.as_posix(),
            "sha256": copied_hash,
            "size_bytes": target.stat().st_size,
        }
        files.append(record)
        return record

    for source in sorted(package.rglob("*")):
        if source.is_file() and source.suffix.lower() in {".py", ".ps1", ".json"}:
            snapshot(
                source, Path("snapshot") / source.relative_to(project), role="package_source_or_data",
                expected_sha256=manifest["launcher_sha256"] if source.resolve() == Path(__file__).resolve() else None,
            )
    snapshot(
        project / "simulation/isaac/_quadruped_runtime.py",
        Path("snapshot/simulation/isaac/_quadruped_runtime.py"), role="shared_runtime_source",
    )
    diagnostics = manifest.get("critic_context_training_diagnostics")
    if diagnostics is not None:
        snapshot(
            Path(diagnostics["path"]), Path("inputs/critic_context_training_diagnostics.py"),
            role="explicit_critic_context_training_diagnostics",
            expected_sha256=diagnostics["sha256"],
        )
    projection_diagnostics = manifest.get("rear_pitch_training_diagnostics")
    if projection_diagnostics is not None:
        snapshot(
            Path(projection_diagnostics["path"]), Path("inputs/rear_pitch_training_diagnostics.py"),
            role="explicit_rear_pitch_training_diagnostics",
            expected_sha256=projection_diagnostics["sha256"],
        )
    overrides = manifest["override_source"]
    snapshot(
        Path(overrides["override_path"]), Path("inputs/overrides.json"), role="explicit_override_input",
        expected_sha256=overrides["override_sha256"],
    )
    for field, reference in sorted(overrides["reference_files"].items()):
        snapshot(
            Path(reference["path"]), Path("inputs/references") / f"{field}.json", role="explicit_reference_input",
            expected_sha256=reference["sha256"],
        )
    asset = snapshot(
        project / "simulation/exports/isaac/quadruped_robot_floating.usdc",
        Path("snapshot/simulation/exports/isaac/quadruped_robot_floating.usdc"), role="robot_root_asset",
    )
    # This closure was inspected offline with Sdf/UsdUtils on 2026-09-05. Do
    # not extend that finding to different asset bytes or launch a USD runtime.
    inspected_asset_hash = "71b639bd877913bffeac47a1cfcb6f3dcabbbd1e25c6fba90b8b87e7ea96c6b8"
    record = {
        **manifest,
        "schema": "drobot.v44-ablation-launch.v1",
        "status": "PREPARED_LAUNCH_EVIDENCE_ONLY",
        "qualification": "UNQUALIFIED_ABLATION",
        "training_started_confirmed": False,
        "training_completed_confirmed": False,
        "invocation_id": invocation_id,
        "storage_layout": "hashed-experiment/uuid/flat-numbered-snapshot.v1",
        "prepared_at_utc": prepared_at.isoformat(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": Path.cwd().resolve().as_posix(),
        "python_executable": sys.executable,
        "snapshot_files": files,
        "asset_dependency_closure": {
            "status": "self_contained_for_previously_inspected_bytes"
            if asset["sha256"] == inspected_asset_hash else "not_inspected_for_this_asset_hash",
            "previous_inspection_date": "2026-09-05",
            "previously_inspected_sha256": inspected_asset_hash,
            "method": "offline Sdf/UsdUtils.ComputeAllDependencies; no external or unresolved assets",
        },
        "note": (
            "Prepared before simulator imports; this record does not prove that training started, "
            "finished, or produced qualified weights. Snapshots record bytes available during "
            "preparation, not a guarantee that files remain unchanged when imported later. "
            "Package JSON includes candidate/default tables; explicit input roles identify supplied "
            "references. The eventual run's saved YAML is authoritative for constructed config. "
            "Kit fast shutdown may bypass the optional post-return completion hook. "
            "This is not a Python/dependency-environment snapshot or a hardware qualification."
        ),
    }
    temporary = directory / "launch-manifest.json.pending"
    final = directory / "launch-manifest.json"
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Hard-link publication is atomic and fails if the destination exists,
    # unlike replace(), which could overwrite an existing evidence record.
    os.link(temporary, final)
    temporary.unlink()
    print(f"[V44] Prepared launch evidence only: {final}", flush=True)
    print("[V44] Training completion requires separate evidence; Kit may bypass the post-return hook.", flush=True)
    return {"path": final.resolve().as_posix(), "sha256": digest(final), "invocation_id": invocation_id}


def record_manifests(experiment_root: Path, before: dict, manifest: dict) -> None:
    """Optional post-return hashes; Kit fast shutdown can bypass this function."""
    changed = [
        path for path, fingerprint in training_snapshot(experiment_root).items()
        if before.get(path) != fingerprint
    ]
    if not changed:
        raise RuntimeError("Training returned without a new ablation checkpoint; no manifest written")
    for run in sorted({path.parent for path in changed}):
        env_path, agent_path = run / "params/env.yaml", run / "params/agent.yaml"
        saved_agent = read_config_fields(agent_path, {"experiment_name"})
        if saved_agent.get("experiment_name") != experiment_root.name:
            raise RuntimeError(f"Unexpected experiment in saved config: {run}")
        reference_paths = read_config_fields(env_path, {
            "gait_phase_reference_path", "gait_phase_reference_low_path"
        })
        if len(reference_paths) != 2 or not isinstance(reference_paths.get("gait_phase_reference_path"), str):
            raise RuntimeError(f"Missing effective ablation reference paths: {run}")
        if any(value is not None and not isinstance(value, str) for value in reference_paths.values()):
            raise RuntimeError(f"Invalid effective ablation reference paths: {run}")
        record = {
            **manifest,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": {path.name: digest(path) for path in sorted(changed) if path.parent == run},
            "params_sha256": {"env.yaml": digest(env_path), "agent.yaml": digest(agent_path)},
            "references_at_manifest": {
                field: {"path": value, "sha256": digest(Path(value)) if value is not None else None}
                for field, value in reference_paths.items()
            },
        }
        (run / "ablation-manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> int | None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--env-overrides-json", type=Path, required=True)
    parser.add_argument("--experiment-suffix", required=True)
    parser.add_argument("--task", choices=[TASK], default=TASK)
    parser.add_argument("--rl_library", choices=["rsl_rl"], default="rsl_rl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--load_run")
    parser.add_argument("--run_name")
    parser.add_argument("--credit-horizon-lambda", type=float, choices=(.95, .99), default=None)
    parser.add_argument("--swing-tilt-continuation33", action="store_true")
    parser.add_argument("--swing-tilt-credit34", action="store_true")
    parser.add_argument("--constant-stance40", action="store_true")
    parser.add_argument("--body-tilt42", action="store_true")
    parser.add_argument("--normalized-slip36", action="store_true")
    parser.add_argument("--beta-concentration37", type=float, choices=(1., 2.))
    parser.add_argument("--action-statistics-only", action="store_true",
                        help="Isolated four-update passive action-distribution study, not a deployable policy")
    args, forwarded = parser.parse_known_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.experiment_suffix):
        parser.error("--experiment-suffix must be a 1–64 character lower-case alphanumeric/_/- slug")
    if (args.checkpoint or args.load_run) and not args.resume:
        parser.error("Checkpoint selection requires explicit --resume; fresh training is the default")
    if args.run_name is not None and not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", args.run_name):
        parser.error("--run_name must be a 1–64 character filename slug, without path separators")
    for token in forwarded:
        key = token.split("=", 1)[0].lstrip("+~")
        if key.startswith(("--experiment", "--external_callback", "--agent")) or key == "--":
            parser.error(f"Reserved ablation argument: {key}")
        if "=" in token and not token.startswith("--"):
            if not key.startswith("agent.") or key in {
                "agent.experiment_name", "agent.run_name", "agent.class_name", "agent.algorithm.class_name"
            }:
                parser.error("Only agent.* Hydra overrides are forwarded; put env changes in the JSON file")
    overrides, override_record = load_v44_overrides(args.env_overrides_json, args.task)
    experiment = ablation_experiment_name(args.experiment_suffix)
    experiment_root = Path.cwd() / "logs" / "rsl_rl" / experiment
    require_short_training_paths(experiment_root, args.run_name)
    before = training_snapshot(experiment_root)
    cli = [*forwarded, "--task", TASK, "--rl_library", "rsl_rl", "--experiment_name", experiment]
    if args.resume:
        cli.append("--resume")
    for flag, value in (("--checkpoint", args.checkpoint), ("--load_run", args.load_run), ("--run_name", args.run_name)):
        if value is not None:
            cli.extend([flag, value])
    cli.extend(v44_hydra_overrides(overrides))
    cli.append(f"agent.experiment_name={experiment}")
    if args.beta_concentration37 is not None:
        if any('actor.distribution_cfg' in token for token in forwarded):
            parser.error('Study37 owns the complete distribution override')
        cli.extend(['agent.actor.distribution_cfg.class_name=parallel_walking.beta_concentration37:ConcentratedBeta37',
                    f'agent.actor.distribution_cfg.concentration_factor={args.beta_concentration37}'])
    manifest = {
        "schema": "drobot.v44-ablation-training.v1",
        "status": "UNQUALIFIED_ABLATION",
        "production_export_allowed": False,
        "note": "Training return is not gait qualification or proof the requested iteration count completed.",
        "training_task": TASK,
        "experiment_name": experiment,
        "canonical_base_experiment_name": EXPERIMENT,
        "requested_experiment_suffix": args.experiment_suffix,
        "invocation_prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_overrides": overrides,
        "override_source": override_record,
        "effective_cli": cli,
        "launcher_sha256": digest(Path(__file__)),
    }
    context_mode = overrides.get("critic_context_mode", "legacy58")
    if context_mode != "legacy58":
        # Mandatory, study-only recording for the selected critic-information
        # pair. Legacy training never imports or installs this observer.
        diagnostics_path = (
            Path(__file__).resolve().parents[4]
            / "simulation/reviews/v44-actuator-20260905/critic_context_training_diagnostics.py"
        ).resolve(strict=True)
        manifest["critic_context_training_diagnostics"] = {
            "path": diagnostics_path.as_posix(),
            "sha256": digest(diagnostics_path),
            "size_bytes": diagnostics_path.stat().st_size,
            "schema": "drobot.critic-causal-context.v1",
            "expected_mode": context_mode,
            "scope": "Existing training computations only; no additional model forward or optimizer step",
        }
    projection_enabled = overrides.get("gait_rear_pitch_projection_enabled", False)
    sustained_credit = overrides.get("decouple_sustained_progress_from_pair_release", False)
    raw_smoothness = overrides.get("penalty_action_acceleration") == 1.0
    pitch_feedback = overrides.get("gait_rear_tracking_feedback_gain", 0.) != 0.
    swing_tilt = overrides.get("penalty_swing_sole_tilt", 0.) != 0.
    swing_tilt_dose = overrides.get("penalty_swing_sole_tilt") == 4.
    if args.body_tilt42 and (args.constant_stance40 or args.beta_concentration37 is not None
        or args.normalized_slip36 or args.swing_tilt_credit34 or args.swing_tilt_continuation33
        or not args.resume or args.credit_horizon_lambda != .99 or not swing_tilt_dose
        or args.experiment_suffix not in ('body-tilt42-control','body-tilt42-treatment')
        or overrides.get('penalty_body_tilt') != (8. if args.experiment_suffix.endswith('control') else 512.)
        or overrides.get('gait_front_stance_tracking_feedback_gain',0.) != 0.
        or 'penalty_normalized_contact_slip' in overrides):
        raise ValueError('Body tilt42 requires its exact resumed coefficient8/512 matched arm')
    if 'penalty_body_tilt' in overrides and not args.body_tilt42:
        raise ValueError('Body tilt override training requires explicit42 route')
    if args.constant_stance40 and (args.beta_concentration37 is not None or args.normalized_slip36
        or args.swing_tilt_credit34 or args.swing_tilt_continuation33 or not args.resume
        or args.experiment_suffix != 'constant-stance40' or args.credit_horizon_lambda != .99
        or not swing_tilt_dose or overrides.get('gait_front_stance_tracking_feedback_gain',0.) != 0.
        or 'penalty_normalized_contact_slip' in overrides):
        raise ValueError('Constant stance40 requires its isolated full-resume original-reward lambda99 route')
    if args.beta_concentration37 is not None and (
        args.normalized_slip36 or args.swing_tilt_credit34 or args.swing_tilt_continuation33
        or not args.resume or not swing_tilt_dose or args.credit_horizon_lambda != .99
        or args.experiment_suffix != ('beta37-control' if args.beta_concentration37 == 1. else 'beta37-treatment')
        or overrides.get('penalty_normalized_contact_slip') != 0.
        or overrides.get('gait_front_stance_tracking_feedback_gain', 0.) != 0.):
        raise ValueError('Beta37 requires its exact factor, full resume and unchanged zero-slip arm')
    if args.normalized_slip36 and (args.swing_tilt_credit34 or args.swing_tilt_continuation33
        or not args.resume or not swing_tilt_dose or args.credit_horizon_lambda != .99
        or args.experiment_suffix not in ("normalized-slip36-control", "normalized-slip36-treatment")
        or overrides.get("penalty_normalized_contact_slip") != (0. if args.experiment_suffix.endswith("control") else .1)
        or overrides.get("gait_front_stance_tracking_feedback_gain",0.) != 0.):
        raise ValueError("Normalized slip36 requires its exact resumed matched controller/reward arm")
    if "penalty_normalized_contact_slip" in overrides and not args.normalized_slip36 and args.beta_concentration37 is None:
        raise ValueError("Normalized slip training requires explicit36 route")
    if args.swing_tilt_credit34 and (not swing_tilt_dose or not args.resume or args.swing_tilt_continuation33
        or args.experiment_suffix != "swing-tilt-credit-34" or args.credit_horizon_lambda != .99):
        raise ValueError("Credit34 requires its exact resumed coefficient4 lambda99 study")
    if args.swing_tilt_continuation33 and (not swing_tilt_dose or not args.resume
        or args.experiment_suffix != "swing-tilt-continue-33" or args.credit_horizon_lambda != .95):
        raise ValueError("Continuation33 requires its exact resumed coefficient4 study")
    if swing_tilt and (not pitch_feedback or overrides.get("penalty_swing_sole_tilt") not in (2.,4.)):
        raise ValueError("Swing tilt training requires selected feedback and weight2 or4")
    if pitch_feedback and (overrides.get("gait_rear_tracking_feedback_gain") != .25
        or overrides.get("gait_rear_tracking_feedback_mode") != "fixed_pitch"
        or list(overrides.get("gait_rear_tracking_feedback_leg_mask", [])) != [1.,1.]
        or args.credit_horizon_lambda != (.99 if args.swing_tilt_credit34 or args.normalized_slip36 or args.beta_concentration37 is not None or args.constant_stance40 or args.body_tilt42 else .95) or sustained_credit or raw_smoothness
        or args.action_statistics_only):
        raise ValueError("Feedback training supports only both-rear fixed-pitch original-reward lambda.95 study")
    if raw_smoothness and (sustained_credit or args.credit_horizon_lambda != .95):
        raise ValueError("Raw-smoothness study retains original pair gate and explicit lambda .95")
    if args.action_statistics_only and (not sustained_credit or args.credit_horizon_lambda != .95):
        raise ValueError("Action-statistics study requires the unchanged sustained22 configuration and lambda .95")
    if sustained_credit and args.credit_horizon_lambda != .95:
        raise ValueError("Sustained-credit study retains explicit original lambda .95")
    if args.credit_horizon_lambda is not None and (
        not projection_enabled or context_mode != "legacy58"
        or overrides.get("gait_rear_pitch_lift_m") != .005
    ):
        raise ValueError("Credit-horizon study requires projected five-mm lift and legacy58")
    if projection_enabled:
        if context_mode != "legacy58":
            raise ValueError("Rear-pitch training reporting requires the selected legacy58 critic")
        projection_diagnostics_path = (
            Path(__file__).resolve().parents[4]
            / "simulation/reviews/v44-actuator-20260905"
            / ("body_tilt42_diagnostics.py" if args.body_tilt42 else
               "constant_stance40_diagnostics.py" if args.constant_stance40 else
               "beta_concentration37_diagnostics.py" if args.beta_concentration37 is not None else
               "normalized_slip36_diagnostics.py" if args.normalized_slip36 else
               "swing_tilt_credit34_diagnostics.py" if args.swing_tilt_credit34 else
               "swing_tilt_continuation33_diagnostics.py" if args.swing_tilt_continuation33 else
               "swing_tilt_dose_training_diagnostics.py" if swing_tilt_dose else
               "swing_tilt_training_diagnostics.py" if swing_tilt else
               "pitch_feedback_training_diagnostics.py" if pitch_feedback else
               "raw_smoothness_training_diagnostics.py" if raw_smoothness else
               "action_statistics_diagnostics.py" if args.action_statistics_only else
               "sustained_progress_training_diagnostics.py" if sustained_credit else
               "rear_pitch_training_diagnostics_attempt02.py" if args.credit_horizon_lambda is None
               else "credit_horizon_training_diagnostics.py")
        ).resolve(strict=True)
        manifest["rear_pitch_training_diagnostics"] = {
            "path": projection_diagnostics_path.as_posix(),
            "sha256": digest(projection_diagnostics_path),
            "size_bytes": projection_diagnostics_path.stat().st_size,
            "schema": "drobot.rear-pitch-training-diagnostics.v1",
            "expected_amplitude_m": overrides["gait_rear_pitch_lift_m"],
            "maximum_rollouts": 50,
            "scope": "Exactly-once reward/process/update reporting before resets; no additional model or physics calls",
        }
        if args.credit_horizon_lambda is not None:
            manifest["rear_pitch_training_diagnostics"].update({
                "schema": "drobot.sustained-progress-training-diagnostics.v1" if sustained_credit else "drobot.credit-horizon-training-diagnostics.v1",
                "attempt_id": "sustained-progress-v1" if sustained_credit else "credit-horizon-lambda-v1",
                "expected_lambda": args.credit_horizon_lambda,
                "scope": "Existing reward/process/returns/update reporting; no additional model or physics calls",
            })
        if args.action_statistics_only:
            manifest["rear_pitch_training_diagnostics"].update({
                "schema": "drobot.action-statistics.v1", "attempt_id": "action-statistics-23-v1",
                "maximum_rollouts": 4,
                "scope": "Passive distribution-parameter and motion capture; unchanged four PPO updates",
            })
        if raw_smoothness:
            manifest["rear_pitch_training_diagnostics"].update({
                "schema": "drobot.raw-smoothness-training-diagnostics.v1",
                "attempt_id": "raw-smoothness-v1",
            })
        if pitch_feedback:
            manifest["rear_pitch_training_diagnostics"].update({
                "schema": "drobot.pitch-feedback-training-diagnostics.v1",
                "attempt_id": "pitch-feedback-v1",
            })
        if swing_tilt:
            manifest["rear_pitch_training_diagnostics"].update({
                "schema": "drobot.swing-tilt-training-diagnostics.v1",
                "attempt_id": "swing-tilt-v1",
            })
        if swing_tilt_dose:
            manifest["rear_pitch_training_diagnostics"].update({
                "schema": "drobot.swing-tilt-dose-training-diagnostics.v1",
                "attempt_id": "swing-tilt-dose-v1",
            })
        if args.swing_tilt_continuation33:
            manifest["rear_pitch_training_diagnostics"].update({
                "schema": "drobot.swing-tilt-continuation33-training-diagnostics.v1",
                "attempt_id": "swing-tilt-continuation33-v1",
            })
        if args.swing_tilt_credit34:
            manifest["rear_pitch_training_diagnostics"].update({
                "schema": "drobot.swing-tilt-credit34-training-diagnostics.v1",
                "attempt_id": "swing-tilt-credit34-v1",
            })
        if args.normalized_slip36:
            manifest["rear_pitch_training_diagnostics"].update({
                "schema": "drobot.normalized-slip36-training-diagnostics.v1",
                "attempt_id": "normalized-slip36-v1",
            })
        if args.beta_concentration37 is not None:
            manifest['rear_pitch_training_diagnostics'].update({
                'schema': 'drobot.beta-concentration37-training-diagnostics.v1',
                'attempt_id': 'beta-concentration37-v1',
                'concentration_factor': args.beta_concentration37,
            })
        if args.constant_stance40:
            manifest['rear_pitch_training_diagnostics'].update({
                'schema': 'drobot.constant-stance40-training-diagnostics.v1',
                'attempt_id': 'constant-stance40-v1',
            })
        if args.body_tilt42:
            manifest['rear_pitch_training_diagnostics'].update({
                'schema': 'drobot.body-tilt42-training-diagnostics.v1',
                'attempt_id': 'body-tilt42-v1', 'maximum_rollouts': 100,
                'body_tilt_weight': overrides['penalty_body_tilt'],
            })
    manifest["launch_evidence"] = prepare_launch_evidence(experiment_root, manifest)
    # Delay simulator imports until accepted inputs and evidence are durable.
    # An absolute script invocation does not automatically add the repository
    # root; config imports require the local simulation.isaac namespace.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import warp as wp

    wp.config.enable_backward = False
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from isaaclab_rl.entrypoints import run_train_cli
    import parallel_walking  # noqa: F401

    if args.beta_concentration37 is not None:
        # The isolated factory declares the field for BOTH Hydra and configclass.
        # Only this owned training process changes its registry entry; ordinary
        # training/evaluation retain the default native distribution config.
        import gymnasium as gym
        spec37 = gym.spec(TASK)
        expected37 = 'parallel_walking.agents.rsl_rl_ppo_cfg:DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadPPORunnerCfg'
        if spec37.kwargs['rsl_rl_cfg_entry_point'] != expected37:
            raise RuntimeError('Study37 expected the original task agent registry')
        spec37.kwargs['rsl_rl_cfg_entry_point'] = 'parallel_walking.beta_concentration37_cfg:agent_config37'

    if context_mode != "legacy58":
        import importlib.util

        diagnostics_record = manifest["critic_context_training_diagnostics"]
        if digest(diagnostics_path) != diagnostics_record["sha256"]:
            raise RuntimeError("Critic-context diagnostics source changed after launch evidence")
        spec = importlib.util.spec_from_file_location(
            "_drobot_critic_context_training_diagnostics", diagnostics_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the source-bound critic-context observer")
        diagnostics_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = diagnostics_module
        spec.loader.exec_module(diagnostics_module)
        diagnostics_module.install_critic_context_diagnostics(context_mode, {
            "training_task": TASK,
            "experiment_name": experiment,
            "launch_evidence": manifest["launch_evidence"],
            "override_source": override_record,
            "diagnostics_source": diagnostics_record,
        })

    if projection_enabled:
        import importlib.util

        projection_record = manifest["rear_pitch_training_diagnostics"]
        if digest(projection_diagnostics_path) != projection_record["sha256"]:
            raise RuntimeError("Rear-pitch reporting source changed after launch evidence")
        spec = importlib.util.spec_from_file_location(
            "_drobot_rear_pitch_training_diagnostics", projection_diagnostics_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load source-bound rear-pitch training observer")
        projection_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = projection_module
        spec.loader.exec_module(projection_module)
        projection_provenance = {
            "training_task": TASK,
            "experiment_name": experiment,
            "launch_evidence": manifest["launch_evidence"],
            "override_source": override_record,
            "diagnostics_source": projection_record,
        }
        if args.credit_horizon_lambda is None:
            projection_module.install_rear_pitch_training_diagnostics(
                projection_record["expected_amplitude_m"], projection_provenance
            )
        else:
            projection_module.install_credit_horizon_training_diagnostics(
                projection_record["expected_lambda"], projection_provenance
            )

    # Default Kit fast shutdown exits the interpreter inside this call; the
    # durable preparation record above must not depend on reaching this hook.
    result = run_train_cli(cli)
    if result in (None, 0):
        record_manifests(experiment_root, before, manifest)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
