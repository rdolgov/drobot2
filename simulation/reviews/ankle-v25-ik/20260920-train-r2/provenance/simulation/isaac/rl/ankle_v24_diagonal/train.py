"""Supervise a single retraining run with persistent progress and source records.

This entry point launches training only. It never starts an evaluation, video,
test suite, hardware connection, model deployment, or a second seed.
"""
import argparse
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

if __package__:
    from . import progress_io
else:
    import progress_io

save = progress_io.atomic_json
save_status = progress_io.best_effort_json


ROOT = Path(__file__).resolve().parents[4]
FOLDER = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def monitor_training(child, log_path, result, status_path):
    """Tail a regular file, so a monitor failure cannot close the worker's stdout."""
    pending = b""
    with log_path.open("rb") as log:
        while True:
            code = child.poll()
            pending += log.read()
            lines = pending.split(b"\n")
            pending = lines.pop()
            if code is not None and pending:
                lines.append(pending)
                pending = b""
            changed = False
            for raw in lines:
                line = raw.decode("utf-8", errors="replace")
                match = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
                if match:
                    result.update(last_iteration=int(match.group(1)), updated_unix=time.time())
                    changed = True
                if "Training time:" in line:
                    result["training_returned"] = True
                    changed = True
            if changed:
                save_status(status_path, result)
            if code is not None:
                return code
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2419)
    parser.add_argument("--run-name", default="diagonal-2419")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    dependencies = ("ankle_v24_diagonal", "ankle_v23_transfer", "ankle_v21", "ankle_v2")
    sources = [path for name in dependencies for path in (FOLDER.parent / name).glob("*.py")]
    sources += [ROOT / path for path in (
        "hardware/robot-runtime/servos/calibration-ankles.json",
        "hardware/robot-runtime/servos/ankle-travel-observations.json",
        "hardware/robot-runtime/servos/calibration-leg-2.json",
        "hardware/robot-runtime/servos/leg-2.toml",
        "simulation/exports/isaac/ankle_v23_stride/assets.json",
        "simulation/docs/rl-ankle-v24-diagonal.md",
        "simulation/docs/rl-ankle-v24-diagonal-revision4.md",
    )]
    asset = ROOT / "simulation/exports/isaac/ankle_v23_stride/active/robot.usdc"
    warm = args.warm_start.resolve()
    contract = {
        "task": "Drobot-Ankle-V24-Diagonal", "seed": args.seed,
        "iterations": args.iterations, "num_envs": args.num_envs, "steps_per_iteration": 64,
        "planned_transitions": args.iterations * args.num_envs * 64,
        "run_name": args.run_name, "warm_start": str(warm), "warm_start_sha256": digest(warm),
        "asset": str(asset), "asset_sha256": digest(asset),
        "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path) for path in sources},
        "local_diagnostics_and_training_authorized": True,
        "authorization": "User: Go ahead and do training and next time just do it, don't ask",
        "automatic_evaluation_in_launcher": False, "hardware_deployment_authorized": False,
    }
    with (output / "training.lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print("This output directory already has an active training supervisor.", flush=True)
            return 0
        child = None
        result = None
        try:
            if status_path.exists():
                prior = json.loads(status_path.read_text())
                if prior.get("contract") != contract:
                    raise ValueError("Changed sources/configuration require a new output directory")
                if prior.get("status") == "completed":
                    print("Training already completed: " + str(status_path), flush=True)
                    return 0
                raise ValueError("Existing incomplete run: inspect it and use an explicit new output directory")
            provenance = output / "provenance"
            for source in sources:
                target = provenance / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            shutil.copy2(warm, output / "initial-actor.pt")
            save(output / "contract.json", contract)
            command = ["C:/isaacsim/python.bat", str(FOLDER / "_train_worker.py"),
                       "--output", str(output), "--warm-start", str(output / "initial-actor.pt"),
                       "--num-envs", str(args.num_envs), "--iterations", str(args.iterations),
                       "--seed", str(args.seed), "--run-name", args.run_name]
            save(output / "command.json", command)
            result = {"status": "starting", "contract": contract, "started_unix": time.time(),
                      "supervisor_pid": os.getpid(), "last_iteration": -1,
                      "training_returned": False, "tests_run": False, "evaluation_run": False,
                      "deployed": False}
            save(status_path, result)
            log_path = output / "train.log"
            with log_path.open("wb") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
            result.update(status="running", worker_launcher_pid=child.pid)
            save_status(status_path, result)
            code = monitor_training(child, log_path, result, status_path)
            run_path = output / "run-path.json"
            checkpoint = None
            if run_path.exists():
                folder = Path(json.loads(run_path.read_text())["directory"])
                checkpoint = folder / f"model_{args.iterations - 1}.pt"
            complete = (code == 0 and result["last_iteration"] == args.iterations - 1
                        and result["training_returned"] and checkpoint and checkpoint.exists()
                        and not (output / "training-error.txt").exists())
            result.update(status="completed" if complete else "failed", exit_code=code, finished_unix=time.time())
            if complete:
                result.update(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
            if not save_status(status_path, result):
                save(output / "final-status.json", result)
            print("Training " + result["status"] + ": " + str(status_path), flush=True)
            return 0 if complete else 1
        except Exception:
            error = traceback.format_exc()
            print(error, file=sys.stderr, flush=True)
            try:
                (output / "supervisor-error.txt").write_text(error, encoding="utf-8")
            except OSError:
                pass
            if result is not None:
                result.update(supervisor_error=error)
                if child is not None and child.poll() is None:
                    # Keep the lock and leave the directly logged worker running.
                    # Monitoring failure must not silently orphan or interrupt it.
                    result.update(status="monitor_failed_worker_running", updated_unix=time.time())
                    save_status(status_path, result)
                    child.wait()
                result.update(status="failed", finished_unix=time.time(),
                              exit_code=child.returncode if child is not None else None)
                if not save_status(status_path, result):
                    save_status(output / "final-status.json", result)
            return 1
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    raise SystemExit(main())
