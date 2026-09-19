"""Launch and archive one authorized curriculum training experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2801)
    parser.add_argument("--guidance", type=int, choices=(0, 1), default=1)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--phases", type=int, choices=(2, 4, 8, 12, 16, 20, 24), default=4)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    state_file = folder / "workflow.json"
    with state_file.open("x") as stream:
        stream.write("{}\n")
    command = ["C:/isaacsim/python.bat", str(Path(__file__).with_name("train.py")),
        "--task", "Drobot-Ankle-V23-BalancedSmall", "--strategy", "free",
        "--num_envs", str(args.num_envs), "--seed", str(args.seed),
        "--max_iterations", str(args.iterations), "--run_name", folder.name,
        "--guidance", str(args.guidance), "--phases", str(args.phases), "--rise", ".02",
        "env.metrics_path=" + str(folder / "train.metrics.jsonl")]
    if args.warm_start:
        command.extend(["--warm-start", str(args.warm_start.resolve())])
    (folder / "command.json").write_text(json.dumps(command, indent=2)+"\n")
    state = {"status": "training", "started_unix": time.time(), "command": command,
             "planned_transitions": args.num_envs*64*args.iterations}
    with (folder / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, creationflags=subprocess.CREATE_NO_WINDOW)
        state["pid"] = process.pid
        state_file.write_text(json.dumps(state, indent=2)+"\n")
        code = process.wait()
    completion_path = folder / "train.metrics.completion.json"
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    state.update(status="failed", exit_code=code, completion=completion, finished_unix=time.time())
    if code == 0 and completion.get("status") == "completed":
        checkpoint = Path(completion["checkpoints"][0])
        archive = ROOT / "simulation/isaac/models/ankle-v23-balanced-small" / folder.name
        archive.mkdir(parents=True, exist_ok=False)
        shutil.copy2(checkpoint, archive / checkpoint.name)
        for name in ("env.yaml", "agent.yaml"):
            shutil.copy2(checkpoint.parent / "params" / name, archive / name)
        contract = {"completion": completion,
            "task_report": json.loads((folder / "train.metrics.foothold.json").read_text()),
            "common_report": json.loads((folder / "train.metrics.screen.json").read_text()),
            "source_asset_sha256": json.loads((folder / "provenance/sha256.json").read_text()),
            "checkpoint_sha256": hashlib.sha256((archive / checkpoint.name).read_bytes()).hexdigest()}
        (archive / "contract.json").write_text(json.dumps(contract, indent=2)+"\n")
        state.update(status="completed", checkpoint=str(archive / checkpoint.name), result=contract["task_report"])
    state_file.write_text(json.dumps(state, indent=2)+"\n")
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
