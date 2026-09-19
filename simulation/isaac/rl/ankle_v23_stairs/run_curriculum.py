"""Train increasing rises only after a completed stage meets training gates.

Uses a single local GPU job at a time. A failed gate ends the experiment;
it never silently reduces the success definition or launches evaluations.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2501)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "workflow.json"
    if state_path.exists():
        raise RuntimeError("This curriculum already has a workflow; do not duplicate its jobs")
    state = {"status": "running", "created_unix": time.time(), "stages": [],
             "iterations_per_stage": args.iterations, "environments": args.num_envs,
             "seed": args.seed, "rises_m": [.02, .04, .06, .10, .14, .18],
             "gate": {"recent_episodes": 512, "success_fraction_min": .7, "fall_fraction_max": .1},
             "evidence": "training_telemetry_only_no_independent_verification"}
    warm = None

    def save():
        state["updated_unix"] = time.time()
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        for rise in state["rises_m"]:
            folder = args.output / f"rise-{round(rise * 1000):03d}mm"
            folder.mkdir(exist_ok=False)
            metrics = folder / "train.metrics.jsonl"
            command = ["C:/isaacsim/python.bat", str(Path(__file__).with_name("train.py")),
                "--task", "Drobot-Ankle-V23-Stairs-D405", "--num_envs", str(args.num_envs),
                "--seed", str(args.seed), "--max_iterations", str(args.iterations),
                "--run_name", f"d405-{args.seed}-{round(rise * 1000):03d}mm", "--rise", str(rise),
                "env.metrics_path=" + str(metrics)]
            if warm:
                command += ["--warm-start", warm]
            (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            stage = {"rise_m": rise, "status": "training", "directory": str(folder)}
            state["stages"].append(stage)
            save()
            with (folder / "train.log").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"}, creationflags=subprocess.CREATE_NO_WINDOW)
                stage["supervisor_pid"] = process.pid
                save()
                code = process.wait()
            completion_path = metrics.with_suffix(".completion.json")
            completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
            if code or completion.get("status") != "completed":
                raise RuntimeError(f"Training failed for {rise:.3f} m; see {folder / 'train.log'}")
            report_path = metrics.with_suffix(".episodes.json")
            report = json.loads(report_path.read_text())
            passed = (report["recent_episodes"] >= 512 and report["recent_success_fraction"] >= .7
                      and report["recent_fall_fraction"] <= .1)
            stage.update(status="completed", checkpoint=completion["checkpoints"][0],
                         training_gate_passed=passed, training_report=report)
            save()
            if not passed:
                state.update(status="completed_gate_not_met", conclusion="Stair climbing not established; do not advance rise")
                break
            warm = completion["checkpoints"][0]
        else:
            state.update(status="completed_training_gates_met", conclusion="Training success through 180 mm; independent evaluation still required")
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
