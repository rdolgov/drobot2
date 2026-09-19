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
import shutil
from queue import Empty, Queue
import subprocess
import sys
import time
from threading import Thread


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
    warm_start = Path(option("--warm-start")).resolve() if option("--warm-start") else None
    contract = {"variant": variant, "iterations": iterations,
                "warm_start_actor": str(warm_start) if warm_start else None,
                "warm_start_sha256": hashlib.sha256(warm_start.read_bytes()).hexdigest() if warm_start else None,
                "num_envs": option("--num_envs", "128"), "seed": option("--seed", "1901"),
                "run_name": run_name, "arguments": sys.argv[1:],
                "sources": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                            for name in ("env.py", "env_cfg.py", "ppo_cfg.py", "__init__.py", "_train_worker.py", "plan.py", "feedforward.json", "reference.npz", "reference.json")}}
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
            experiment = root / "logs/rsl_rl/drobot_ankle_v23_precise"
            provenance = status.parent / "provenance"
            sources = list(folder.glob("*.py"))
            sources += [folder / "feedforward.json"]
            sources += [folder / "reference.npz", folder / "reference.json"]
            sources += [root / "simulation/reviews/ankle-v23-curriculum/20260915" / name for name in
                        ("calculate_compact.py", "shortstep_geometry.py", "static_load_calculation.py", "geometry_calc.py")]
            for package in ("ankle_v2", "ankle_v21", "ankle_v23_stride", "ankle_v23_stairs", "ankle_v23_experiments", "ankle_v23_footplan"):
                sources.extend((folder.parent / package).glob("*.py"))
            assets = root / "simulation/exports/isaac/ankle_v23_stride"
            sources.extend(assets.rglob("*.usd*"))
            sources += [assets / "assets.json", root / "cad/exports/urdf/quadruped_robot_v23_solid_sole.urdf"]
            if warm_start:
                sources += [warm_start]
                sources += [warm_start.parent / name for name in ("env.yaml", "agent.yaml", "contract.json") if (warm_start.parent / name).exists()]
            source_record = {}
            for source in sources:
                relative = source.relative_to(root)
                source_record[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
                target = provenance / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            (provenance / "sha256.json").write_text(json.dumps(source_record, indent=2) + "\n")
            before = set(experiment.glob("*")) if experiment.exists() else set()
            result = {"status": "running", "contract": contract, "started_unix": time.time()}
            status.write_text(json.dumps(result, indent=2) + "\n")
            child = subprocess.Popen([sys.executable, str(folder / "_train_worker.py"), *sys.argv[1:]],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace",
                                     env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            last_iteration = -1
            saw_training_time = False
            result["worker_pid"] = child.pid
            status.write_text(json.dumps(result, indent=2) + "\n")
            lines = Queue()

            def read_output():
                # Omniverse Hub may inherit stdout and outlive the worker.
                # A daemon reader avoids waiting for that unrelated pipe EOF.
                for output_line in child.stdout:
                    lines.put(output_line)

            Thread(target=read_output, daemon=True).start()
            try:
                while True:
                    try:
                        line = lines.get(timeout=1.0)
                    except Empty:
                        if child.poll() is not None:
                            break
                        continue
                    print(line, end="", flush=True)
                    match = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
                    if match:
                        last_iteration = int(match.group(1))
                    saw_training_time |= "Training time:" in line
            except BaseException:
                if child.poll() is None:
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



