"""Bounded, matched, two-seed training screen. No evaluation/recording jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from strategies import STRATEGIES, METRIC_COLUMNS

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=384)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warm-start", type=Path, default=ROOT / "simulation/isaac/models/ankle-v23-dense-turn/20260913-two-seed/sideways_dense-seed2601/model_255.pt")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.warm_start = args.warm_start.resolve()
    warm_start_sha256 = hashlib.sha256(args.warm_start.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "workflow.json"
    with state_path.open("x") as stream:
        stream.write("{}\n")
    lock, abort = threading.Lock(), threading.Event()
    state = {"status": "training", "created_unix": time.time(),
        "iterations_per_job": args.iterations, "environments_per_job": args.num_envs,
        "steps_per_iteration": 64, "seeds": [2701, 2702], "strategies": STRATEGIES,
        "warm_start_actor": str(args.warm_start), "warm_start_sha256": warm_start_sha256,
        "load_policy": "actor_and_its_observation_normalizer_only; fresh_critic_optimizer_and_iteration",
        "seed_caveat": "two_continuations_of_one_parent_actor_not_independent_pretraining_lineages",
        "planned_transitions": len(STRATEGIES) * 2 * args.iterations * args.num_envs * 64,
        "evidence": "matched_budget_stochastic_training_screen_not_independent_evaluation",
        "jobs": [{"strategy": strategy, "seed": seed, "status": "pending"}
                 for seed in (2701, 2702) for strategy in STRATEGIES]}

    def save():
        state["updated_unix"] = time.time()
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    def run_job(job):
        if abort.is_set():
            with lock:
                job["status"] = "not_started_after_failure"
                save()
            return
        key = f"{job['strategy']}-seed{job['seed']}"
        folder = args.output / key
        folder.mkdir(exist_ok=False)
        command = ["C:/isaacsim/python.bat", str(Path(__file__).with_name("train.py")),
            "--task", "Drobot-Ankle-V23-Research", "--strategy", job["strategy"],
            "--num_envs", str(args.num_envs), "--seed", str(job["seed"]),
            "--max_iterations", str(args.iterations), "--run_name", key,
            "--warm-start", str(args.warm_start),
            "--rise", ".02", "env.metrics_path=" + str(folder / "train.metrics.jsonl")]
        (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (folder / "train.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, creationflags=subprocess.CREATE_NO_WINDOW)
            with lock:
                job.update(status="training", pid=process.pid, directory=str(folder), started_unix=time.time())
                save()
            code = process.wait()
        completion_file = folder / "train.metrics.completion.json"
        completion = json.loads(completion_file.read_text()) if completion_file.exists() else {}
        if code or completion.get("status") != "completed":
            abort.set()
            with lock:
                job.update(status="failed", exit_code=code, completion=completion)
                save()
            return
        report = json.loads((folder / "train.metrics.screen.json").read_text())
        archive = ROOT / "simulation/isaac/models/ankle-v23-research" / args.output.name / key
        archive.mkdir(parents=True, exist_ok=False)
        checkpoint = Path(completion["checkpoints"][0])
        shutil.copy2(checkpoint, archive / checkpoint.name)
        for filename in ("env.yaml", "agent.yaml"):
            shutil.copy2(checkpoint.parent / "params" / filename, archive / filename)
        contract = {"training_report": report, "job_completion": completion,
            "source_asset_sha256": json.loads((folder / "provenance/sha256.json").read_text()),
            "checkpoint_sha256": hashlib.sha256((archive / checkpoint.name).read_bytes()).hexdigest()}
        (archive / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        with lock:
            job.update(status="completed", checkpoint=str(archive / checkpoint.name),
                       report=report, completion=completion, finished_unix=time.time())
            save()

    save()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, job) for job in state["jobs"]]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    abort.set()
                    with lock:
                        state.setdefault("errors", []).append(repr(error))
                        save()
        if any(job["status"] != "completed" for job in state["jobs"]):
            state["status"] = "failed"
        else:
            rows = []
            for strategy in STRATEGIES:
                jobs = [job for job in state["jobs"] if job["strategy"] == strategy]
                metrics = {key: sum(job["report"]["recent"][key] for job in jobs) / len(jobs) for key in METRIC_COLUMNS}
                ranges = {key: [min(job["report"]["recent"][key] for job in jobs),
                                max(job["report"]["recent"][key] for job in jobs)] for key in METRIC_COLUMNS}
                rows.append({"strategy": strategy, "seed_mean": metrics, "seed_range": ranges,
                             "checkpoints": [job["checkpoint"] for job in jobs]})
            order = ("full_climb", "all_first", "rear_pair", "any_placement", "unique_feet", "any_supported_lift", "supported_advance_m")
            rows.sort(key=lambda row: tuple(row["seed_mean"][key] for key in order), reverse=True)
            state.update(status="completed", ranking=rows, ranking_metrics=list(order),
                conclusion="Training screen complete; inspect shared milestone metrics and both seeds, not reward totals")
            (args.output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
            lines = ["# V2.3 research-inspired reward continuations", "",
                f"Three reward strategies, two continuation seeds; {args.iterations} iterations and {args.num_envs} environments per job.",
                f"All six jobs load the same parent actor and its normalizer: {args.warm_start} (SHA-256 {warm_start_sha256}).",
                "Critic and optimizer start fresh in every arm, including the unchanged-reward control. Seeds are continuations of one parent, not independent pretraining lineages.",
                "All results below are means of the two seeds' last 256 completed stochastic training episodes.",
                "These are early training signals, not independent policy evaluation or hardware results.", "",
                "| Strategy | Any held placement | Rear pair | All feet on stairs | Full climb | Supported lift | Supported advance | Falls | Peak yaw | Sideways fraction |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for row in rows:
                m = row["seed_mean"]
                lines.append(f"| {row['strategy']} | {m['any_placement']:.1%} | {m['rear_pair']:.1%} | {m['all_first']:.1%} | {m['full_climb']:.1%} | {m['any_supported_lift']:.1%} | {100*m['supported_advance_m']:.1f} cm | {m['fall']:.1%} | {m['max_yaw_deg']:.1f} deg | {m['sideways_motion_fraction']:.1%} |")
            lines += ["", "A placement needs a supported lift above 12 mm, then 0.25 s tread contact with low slip and at least two other support feet.",
                "Supported advance is the episode's greatest forward base advance while upright with at least two foot contacts; it is not a climb.",
                "Yaw and sideways fraction describe supported motion; they cannot substitute for tread placement.",
                "Compare these three arms to each other: continue_turn retains the old reward; potential replaces forward-delta and sustained-lift rewards with discounted potential differences; potential_edges adds riser and edge penalties.",
                "The original screen had a different budget and fresh actors, so it is not the matched control for this follow-up.",
                "See comparison.json for seed ranges and workflow.json for all per-seed counts, configuration provenance and checkpoints.",
                "The D405 remains a geometric depth proxy without self-occlusion or added camera/mount mass.",
                "No separate tests, held-out evaluation, UI validation, recordings or hardware checks were run.", ""]
            (args.output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    finally:
        save()


if __name__ == "__main__":
    main()

