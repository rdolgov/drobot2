"""Supervise isolated V44 training from ordinary Python, outside the Kit process.

Example (not a verification command):
  C:/isaacsim/kit/python/python.exe -B supervise_v44_ablation.py \
    --env-overrides-json candidate.json --experiment-suffix candidate-03 \
    --max-iterations 400 --num-envs 256 --seed 1701 --run-name candidate-03 \
    --agent-override agent.num_steps_per_env=192 \
    --agent-override agent.algorithm.learning_rate=0.0002

Use --resume-checkpoint with one exact existing model_N.pt for continuation.
--max-iterations is ADDITIONAL updates: resuming stored label N expects final
label N + max_iterations - 1. This does not restore environment/RNG/curriculum
state; supply an explicitly reviewed continuation configuration when needed.

No simulator modules are imported here. The child uses the installed Windows
python.bat environment convention, but is started directly as an argument
vector to avoid command-shell interpretation. Preparation/process/completion
records are separate, exclusive files. Completion is observed evidence only,
never gait qualification, a production export permission, or a Pi deployment.
The requested run label is recorded, truncated to 8 characters if necessary,
and suffixed with this invocation's full UUID to avoid timestamp-name reuse.
A forcibly killed supervisor cannot finalize; its prepared/process records
remain evidence of an unfinished observation, not a completed training run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from v44_ablation_contract import load_v44_overrides
from v44_training_contract import EXPERIMENT, TASK, digest
from train_v44_ablation import ablation_experiment_name, launch_records_root, require_short_training_paths


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def exclusive_json(path: Path, value: dict) -> None:
    """Publish complete JSON atomically without replacing any existing record."""
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()


def file_record(path: Path) -> dict:
    return {"path": path.resolve().as_posix(), "sha256": digest(path), "size_bytes": path.stat().st_size}


def agent_override(value: str) -> str:
    key, separator, encoded = value.partition("=")
    if not separator or not re.fullmatch(r"agent\.[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*", key):
        raise argparse.ArgumentTypeError("Use agent.field=<JSON primitive or list>")
    if key in {"agent.experiment_name", "agent.run_name", "agent.class_name", "agent.algorithm.class_name",
               "agent.resume", "agent.load_run", "agent.load_checkpoint", "agent.max_iterations", "agent.seed"}:
        raise argparse.ArgumentTypeError(f"Reserved supervisor setting: {key}")
    try:
        parsed = json.loads(encoded)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Agent override values must be JSON primitives or lists") from error

    def plain(item):
        if isinstance(item, list):
            for child in item:
                plain(child)
        elif isinstance(item, str):
            if "${" in item or any(ord(character) < 32 for character in item):
                raise argparse.ArgumentTypeError("No interpolation or control characters in overrides")
        elif isinstance(item, float) and not math.isfinite(item):
            raise argparse.ArgumentTypeError("Agent numeric values must be finite")
        elif item is not None and not isinstance(item, (bool, int, float)):
            raise argparse.ArgumentTypeError("Agent override objects are not supported")

    plain(parsed)
    return key + "=" + json.dumps(parsed, allow_nan=False, separators=(",", ":"))


def observe_completion(
    *, directory: Path, experiment_root: Path, before_runs: set[str], run_name: str,
    override_sha256: str, trainer_sha256: str, start_iteration: int, iterations: int,
    resume_checkpoint: Path | None, resume_sha256: str | None,
) -> dict:
    """Read logs and output files only; never load checkpoint pickle or a model."""
    stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", stdout_path.read_text(encoding="utf-8", errors="replace"))
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    labels = [(int(left), int(right)) for left, right in re.findall(r"Learning iteration\s+(\d+)/(\d+)", clean)]
    expected_total = start_iteration + iterations
    expected_last = expected_total - 1
    timestamps = re.findall(r"^Exact experiment name requested from command line:\s*(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\s*$", clean, re.M)
    roots = re.findall(r"^\[INFO\] Logging experiment in directory:\s*(.+)$", clean, re.M)
    times = [float(value) for value in re.findall(r"^Training time:\s*([0-9]+(?:\.[0-9]+)?) seconds\s*$", clean, re.M)]
    root_matches = len(roots) == 1 and Path(roots[0].strip()).resolve() == experiment_root
    run = experiment_root / f"{timestamps[0]}_{run_name}" if len(timestamps) == 1 else None
    run_is_new = run is not None and run.name not in before_runs and run.is_dir()
    final = run / f"model_{expected_last}.pt" if run is not None else None
    config_files = {}
    run_manifest_matches = False
    if run_is_new:
        for filename in ("env.yaml", "agent.yaml"):
            config = run / "params" / filename
            if config.is_file():
                config_files[filename] = file_record(config)
        metadata_path = run / "run.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            run_manifest_matches = metadata.get("task") == TASK and metadata.get("library") == "rsl_rl"

    launch_records = re.findall(r"^\[V44\] Prepared launch evidence only:\s*(.+)$", clean, re.M)
    launch_evidence, launch_matches = None, False
    if len(launch_records) == 1:
        launch = Path(launch_records[0].strip()).resolve()
        allowed_root = launch_records_root(experiment_root)
        if launch.is_relative_to(allowed_root) and launch.is_file():
            record = json.loads(launch.read_text(encoding="utf-8"))
            launch_evidence = file_record(launch)
            launch_matches = (
                record.get("schema") == "drobot.v44-ablation-launch.v1"
                and record.get("status") == "PREPARED_LAUNCH_EVIDENCE_ONLY"
                and record.get("training_task") == TASK
                and record.get("experiment_name") == experiment_root.name
                and record.get("launcher_sha256") == trainer_sha256
                and record.get("override_source", {}).get("override_sha256") == override_sha256
            )
    loaded_paths = re.findall(r"^\[INFO\]: Loading model checkpoint from:\s*(.+)$", clean, re.M)
    resume_matches = (
        len(loaded_paths) == 1 and Path(loaded_paths[0].strip()).resolve() == resume_checkpoint
        if resume_checkpoint is not None else not loaded_paths
    )
    criteria = {
        "exact_experiment_root_logged": root_matches,
        "exact_new_run_identified": bool(run_is_new),
        "run_manifest_task_matches": run_manifest_matches,
        "both_effective_configs_saved": len(config_files) == 2,
        "child_prelaunch_evidence_matches": launch_matches,
        "exact_resume_source_logged_or_fresh": resume_matches,
        "resume_checkpoint_bytes_unchanged_or_fresh": resume_checkpoint is None
        or (resume_checkpoint.is_file() and digest(resume_checkpoint) == resume_sha256),
        "all_requested_iteration_labels_in_order": labels == [(label, expected_total) for label in range(start_iteration, expected_total)],
        "successful_learn_return_marker": len(times) == 1 and times[0] > 0,
        "expected_final_checkpoint_exists": bool(run_is_new and final is not None and final.is_file() and final.stat().st_size > 0),
        "no_python_traceback": "Traceback (most recent call last):" not in clean + stderr,
    }
    return {
        "criteria": criteria,
        "run_directory": run.resolve().as_posix() if run_is_new else None,
        "expected_start_iteration": start_iteration,
        "expected_final_iteration": expected_last,
        "observed_iteration_count": len(labels),
        "observed_first_iteration": labels[0][0] if labels else None,
        "observed_last_iteration": labels[-1][0] if labels else None,
        "training_time_seconds": times[0] if len(times) == 1 else None,
        "checkpoint": file_record(final) if criteria["expected_final_checkpoint_exists"] else None,
        "effective_training_config_files": config_files,
        "child_prelaunch_evidence": launch_evidence,
        "keyboard_interrupt_logged": "KeyboardInterrupt" in clean + stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--env-overrides-json", type=Path, required=True)
    parser.add_argument("--experiment-suffix", required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--agent-override", action="append", type=agent_override, default=[])
    parser.add_argument("--credit-horizon-lambda", type=float, choices=(.95, .99), default=None)
    parser.add_argument("--swing-tilt-continuation33", action="store_true")
    parser.add_argument("--swing-tilt-credit34", action="store_true")
    parser.add_argument("--constant-stance40", action="store_true")
    parser.add_argument("--body-tilt42", action="store_true")
    parser.add_argument("--normalized-slip36", action="store_true")
    parser.add_argument("--beta-concentration37", type=float, choices=(1., 2.))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--isaac-root", type=Path, default=Path("C:/isaacsim"))
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--hypothesis-file", type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This supervisor currently targets the installed Windows Isaac layout only")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.experiment_suffix):
        parser.error("Experiment suffix must be a 1-64 character lower-case slug")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", args.run_name):
        parser.error("Run name must be a 1-64 character filename slug")
    if args.max_iterations <= 0 or args.num_envs <= 0 or args.seed < 0:
        parser.error("Iteration/environment counts must be positive and seed explicitly nonnegative")
    if len({value.split("=", 1)[0] for value in args.agent_override}) != len(args.agent_override):
        parser.error("Duplicate agent overrides are not allowed")
    invocation_uuid = uuid4().hex
    invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + invocation_uuid
    effective_run_name = args.run_name[:8] + "_" + invocation_uuid
    package = Path(__file__).resolve().parent
    project = package.parents[3]
    experiment = ablation_experiment_name(args.experiment_suffix)
    experiment_root = (project / "logs/rsl_rl" / experiment).resolve()
    require_short_training_paths(experiment_root, effective_run_name)
    before_runs = {path.name for path in experiment_root.iterdir() if path.is_dir()} if experiment_root.exists() else set()
    _, overrides = load_v44_overrides(args.env_overrides_json, TASK)
    isaac_root = args.isaac_root.resolve(strict=True)
    child_python = isaac_root / "kit/python/kit.exe"
    if not child_python.is_file():
        parser.error(f"Expected existing Isaac Python interpreter: {child_python}; no executable is created automatically")
    trainer = package / "train_v44_ablation.py"
    resume = args.resume_checkpoint.resolve(strict=True) if args.resume_checkpoint else None
    start_iteration = 0
    if resume is not None:
        match = re.fullmatch(r"model_(\d+)\.pt", resume.name)
        if not resume.is_file() or match is None or resume.parent.parent != experiment_root:
            parser.error("Resume checkpoint must be an exact model_N.pt inside this same ablation experiment")
        start_iteration = int(match.group(1))
    command = [str(child_python), "-u", str(trainer), "--env-overrides-json", overrides["override_path"],
               "--experiment-suffix", args.experiment_suffix, "--num_envs", str(args.num_envs),
               "--max_iterations", str(args.max_iterations), "--seed", str(args.seed),
               "--run_name", effective_run_name, "--device", args.device, "--visualizer", "none"]
    if resume is not None:
        command.extend(["--resume", "--load_run", re.escape(resume.parent.name) + "$", "--checkpoint", re.escape(resume.name) + "$"])
    if args.credit_horizon_lambda is not None:
        command.extend(["--credit-horizon-lambda", str(args.credit_horizon_lambda)])
    if args.swing_tilt_continuation33:
        if args.experiment_suffix != "swing-tilt-continue-33" or args.max_iterations != 50 or start_iteration != 347:
            parser.error("Continuation33 requires its exact50-update model347 allocation")
        command.append("--swing-tilt-continuation33")
    if args.swing_tilt_credit34:
        if args.swing_tilt_continuation33 or args.experiment_suffix != "swing-tilt-credit-34" or args.max_iterations != 50 or start_iteration != 347 or args.credit_horizon_lambda != .99:
            parser.error("Credit34 requires its exact50-update model347 lambda99 allocation")
        command.append("--swing-tilt-credit34")
    if args.normalized_slip36:
        if (args.swing_tilt_credit34 or args.swing_tilt_continuation33 or args.max_iterations != 50
            or start_iteration != 396 or args.credit_horizon_lambda != .99
            or args.experiment_suffix not in ("normalized-slip36-control", "normalized-slip36-treatment")):
            parser.error("Normalized slip36 requires its exact50-update model396 lambda99 arm")
        command.append("--normalized-slip36")
    if args.beta_concentration37 is not None:
        if (args.normalized_slip36 or args.swing_tilt_credit34 or args.swing_tilt_continuation33
            or args.max_iterations != 50 or start_iteration != 396 or args.credit_horizon_lambda != .99
            or args.experiment_suffix != ('beta37-control' if args.beta_concentration37 == 1. else 'beta37-treatment')):
            parser.error('Beta37 requires exact50-update model396 lambda99 factor/arm')
        command.extend(['--beta-concentration37', str(args.beta_concentration37)])
    if args.constant_stance40:
        if (args.normalized_slip36 or args.beta_concentration37 is not None
            or args.swing_tilt_credit34 or args.swing_tilt_continuation33 or args.max_iterations != 200
            or start_iteration != 396 or args.credit_horizon_lambda != .99 or args.experiment_suffix != 'constant-stance40'):
            parser.error('Constant stance40 requires exact200-update model396 lambda99 allocation')
        command.append('--constant-stance40')
    command.extend(args.agent_override)
    if args.body_tilt42:
        if (args.constant_stance40 or args.normalized_slip36 or args.beta_concentration37 is not None
            or args.swing_tilt_credit34 or args.swing_tilt_continuation33 or args.max_iterations != 100
            or start_iteration != 396 or args.credit_horizon_lambda != .99
            or args.experiment_suffix not in ('body-tilt42-control','body-tilt42-treatment')):
            parser.error('Body tilt42 requires exact100-update model396 lambda99 arm')
        command.append('--body-tilt42')
    environment = os.environ.copy()
    child_environment_settings = {
        "CARB_APP_PATH": str(isaac_root / "kit"), "ISAAC_PATH": str(isaac_root),
        "EXP_PATH": str(isaac_root / "apps"), "PYTHONUNBUFFERED": "1",
    }
    environment.update(child_environment_settings)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(project), environment.get("PYTHONPATH"), str(isaac_root / "site")]))
    directory = args.record_dir.resolve() if args.record_dir else project / "logs/rsl_rl/_v44_supervised" / launch_records_root(experiment_root).name / invocation_uuid
    if len(str(directory / "completion.json.pending")) >= 240:
        parser.error("Supervisor evidence path exceeds the conservative Windows path budget")
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    common = {"schema": "drobot.v44-supervised-training.v1", "invocation_id": invocation,
              "qualification": "UNQUALIFIED_ABLATION", "production_export_allowed": False,
              "requested_run_name": args.run_name, "effective_run_name": effective_run_name}
    hypothesis_file = None
    if args.hypothesis_file:
        source = args.hypothesis_file.resolve(strict=True)
        with source.open("rb") as reader, (directory / "hypothesis-source").open("xb") as writer:
            writer.write(reader.read())
            writer.flush()
            os.fsync(writer.fileno())
        hypothesis_file = {"source_path": source.as_posix(), **file_record(directory / "hypothesis-source")}
    prepared = {
        **common, "status": "prepared", "prepared_at_utc": utc_now(), "working_directory": project.as_posix(),
        "training_task": TASK, "experiment_name": experiment, "run_name": effective_run_name,
        "canonical_base_experiment_name": EXPERIMENT,
        "requested_experiment_suffix": args.experiment_suffix,
        "requested_run_name_truncated": len(args.run_name) > 8,
        "run_name_suffix": invocation_uuid,
        "command": command, "child_environment_settings": child_environment_settings,
        "pythonpath_explicit_paths": [str(project), str(isaac_root / "site")], "inherits_other_environment_without_recording_secrets": True,
        "override_source": overrides, "hypothesis": args.hypothesis, "hypothesis_file": hypothesis_file,
        "supervisor_source": file_record(Path(__file__)), "trainer_source": file_record(trainer),
        "resume_checkpoint": file_record(resume) if resume else None,
        "start_iteration": start_iteration, "requested_additional_updates": args.max_iterations,
        "expected_final_iteration": start_iteration + args.max_iterations - 1,
        "iteration_basis": "RSL-RL filename/stored-iteration convention; checkpoint pickle is not deserialized",
        "preexisting_run_directories": sorted(before_runs),
        "note": "Prepared only. This record captures the hypothesis; the child prelaunch snapshot captures configuration inputs and source. Completion is a separate observation. Resume does not restore environment, RNG or curriculum state.",
    }
    exclusive_json(directory / "prepared.json", prepared)
    print(f"[SUPERVISOR] Prepared evidence directory: {directory}", flush=True)
    print(f"[SUPERVISOR] Effective run name: {effective_run_name} (requested: {args.run_name})", flush=True)
    process = None
    interrupted = False
    forced = False
    error = None
    exit_code = None
    launched_at = None
    try:
        with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            process = subprocess.Popen(command, cwd=project, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, startupinfo=startupinfo,
                                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            launched_at = utc_now()
            exclusive_json(directory / "process.json", {**common, "status": "started", "pid": process.pid,
                           "started_at_utc": launched_at, "child_executable": str(child_python)})
            print(f"[SUPERVISOR] Training child PID: {process.pid}", flush=True)
            stop_at = None
            while True:
                try:
                    exit_code = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    if stop_at is not None and time.monotonic() - stop_at >= 60 and not forced:
                        process.terminate()
                        forced = True
                except KeyboardInterrupt:
                    if not interrupted:
                        interrupted = True
                        stop_at = time.monotonic()
                        exclusive_json(directory / "interrupt-request.json", {**common, "requested_at_utc": utc_now(),
                                       "pid": process.pid, "action": "CTRL_BREAK_EVENT; terminate only this child after 60 seconds if needed"})
                        try:
                            process.send_signal(signal.CTRL_BREAK_EVENT)
                        except OSError:
                            process.terminate()
                            forced = True
                    elif not forced:
                        process.terminate()
                        forced = True
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    except KeyboardInterrupt:
        # Also cover an interrupt during Popen/process-record publication,
        # outside the normal wait-loop handler.
        interrupted = True
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                exit_code = process.wait(timeout=60)
            except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt):
                process.terminate()
                forced = True
                exit_code = process.wait()
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        # Do not abandon a live owned child if evidence recording fails.
        if process is not None and process.poll() is None:
            process.terminate()
            forced = True
            exit_code = process.wait()
    observations = {}
    evidence_error = None
    try:
        observations = observe_completion(directory=directory, experiment_root=experiment_root, before_runs=before_runs,
                                          run_name=effective_run_name, override_sha256=overrides["override_sha256"],
                                          trainer_sha256=prepared["trainer_source"]["sha256"],
                                          start_iteration=start_iteration, iterations=args.max_iterations,
                                          resume_checkpoint=resume,
                                          resume_sha256=prepared["resume_checkpoint"]["sha256"] if resume else None)
    except Exception as exception:
        evidence_error = f"{type(exception).__name__}: {exception}"
    if interrupted or exit_code in {-1073741510, 3221225786} or observations.get("keyboard_interrupt_logged"):
        status = "interrupted"
    elif error is not None or exit_code != 0:
        status = "failed"
    elif observations.get("criteria") and all(observations["criteria"].values()):
        status = "completed_observed"
    else:
        status = "unknown"
    exclusive_json(directory / "completion.json", {
        **common, "status": status, "recorded_at_utc": utc_now(), "started_at_utc": launched_at,
        "pid": process.pid if process else None, "exit_code": exit_code, "interrupt_requested": interrupted,
        "forced_child_termination": forced, "supervisor_error": error,
        "evidence_inspection_error": evidence_error, "observations": observations,
        "logs": {name: file_record(directory / name) for name in ("stdout.log", "stderr.log") if (directory / name).is_file()},
        "note": "Exit zero alone is insufficient. Completion requires exact source/run identity, all requested iteration labels, final checkpoint, saved configs and successful learn-return marker. This is not a physics-quality evaluation or hardware/deployment approval.",
    })
    print(f"[SUPERVISOR] {status}: {directory / 'completion.json'}", flush=True)
    return {"completed_observed": 0, "failed": 1, "unknown": 2, "interrupted": 3}[status]


if __name__ == "__main__":
    raise SystemExit(main())
