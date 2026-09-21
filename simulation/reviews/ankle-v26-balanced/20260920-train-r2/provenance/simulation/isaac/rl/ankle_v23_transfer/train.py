"""Supervise one PPO job, serialize duplicate launches and record its checkpoint.

The supervisor survives Isaac Kit's fast shutdown. It never evaluates a policy.
Use the same metrics path to join an existing job without training it twice.
"""
from __future__ import annotations
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def option(name, default=None):
    for i, value in enumerate(sys.argv[1:], 1):
        if value == name:
            return sys.argv[i + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return default


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    folder = Path(__file__).resolve().parent
    root = folder.parents[3]
    variant = option("--task").rsplit("-", 1)[1].lower()
    iterations = int(option("--max_iterations", "500"))
    run_name = option("--run_name", "pilot")
    metrics = option("env.metrics_path")
    if not metrics:
        raise ValueError("Supply env.metrics_path for persistent training provenance")
    status = Path(metrics).with_suffix(".completion.json")
    status.parent.mkdir(parents=True, exist_ok=True)
    contract = {"variant": variant, "iterations": iterations,
                "num_envs": option("--num_envs", "128"), "seed": option("--seed", "1901"),
                "run_name": run_name, "arguments": sys.argv[1:],
                "sources": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                            for name in ("env.py", "env_cfg.py", "ppo_cfg.py", "__init__.py")}}
    with status.with_suffix(".lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        while True:
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(5)
        try:
            if status.exists():
                prior = json.loads(status.read_text())
                if prior.get("contract") != contract:
                    raise ValueError("Use a fresh metrics path when changing the training contract")
                if prior.get("status") == "completed":
                    print("ANKLE_JOB_ALREADY_COMPLETED=" + str(status), flush=True)
                    return 0
            experiment = root / "logs/rsl_rl/drobot_ankle_v23_transfer"
            before = set(experiment.glob("*")) if experiment.exists() else set()
            result = {"status": "running", "contract": contract, "started_unix": time.time()}
            status.write_text(json.dumps(result, indent=2) + "\n")
            child = subprocess.Popen([sys.executable, str(folder / "_train_worker.py"), *sys.argv[1:]],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace",
                                     env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            last_iteration = -1
            saw_training_time = False
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    match = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
                    if match:
                        last_iteration = int(match.group(1))
                    saw_training_time |= "Training time:" in line
            except BaseException:
                child.terminate()
                child.wait(timeout=30)
                result.update(status="failed", reason="supervisor interrupted")
                status.write_text(json.dumps(result, indent=2) + "\n")
                raise
            code = child.wait()
            new_runs = [p for p in experiment.glob("*") if p not in before and p.is_dir() and p.name.endswith(run_name)]
            final = [p / f"model_{iterations - 1}.pt" for p in new_runs]
            final = [p for p in final if p.exists()]
            complete = code == 0 and last_iteration == iterations - 1 and saw_training_time and len(final) == 1
            result.update(status="completed" if complete else "failed", exit_code=code,
                          last_iteration=last_iteration, training_returned=saw_training_time,
                          finished_unix=time.time(), checkpoints=[str(p) for p in final])
            if complete:
                result["checkpoint_sha256"] = hashlib.sha256(final[0].read_bytes()).hexdigest()
            status.write_text(json.dumps(result, indent=2) + "\n")
            print("ANKLE_JOB_RESULT=" + str(status), flush=True)
            return 0 if complete else 1
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    raise SystemExit(main())
