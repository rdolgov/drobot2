"""Read saved PPO logs/checkpoint metadata; never construct or run a policy."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics


def summarize(folder: Path):
    root = Path(__file__).resolve().parents[4]
    batch = json.loads((folder / "batch.json").read_text(encoding="utf-8-sig"))
    result = {"schema": "drobot.ankle-training-summary.v1", "batch": str(folder),
              "scope": "Training telemetry only; no held-out evaluation, video or hardware trial",
              "seed": batch["seed"], "iterations_requested": batch["iterations"],
              "num_envs": batch["num_envs"], "variants": {}}
    for variant in ("active", "baseline", "locked"):
        log = folder / f"{variant}.parallel.log"
        if not log.exists():
            log = folder / f"{variant}.log"
        if not log.exists():
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        matches = list(re.finditer(r"Learning iteration\s+(\d+)/(\d+)", text))
        blocks = []
        for i, match in enumerate(matches):
            block = text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
            metrics = {key: float(value) for key, value in re.findall(
                r"^\s*(Metrics/train_[a-z_]+|Mean reward|Mean episode length|Total steps)\s*:\s*([-+\d.eE]+)", block, re.M)}
            blocks.append({"iteration": int(match.group(1)), **metrics})
        final = blocks[-1] if blocks else {}
        tail = blocks[-50:]
        average = {key: statistics.fmean(row[key] for row in tail if key in row)
                   for key in final if key != "iteration" and any(key in row for row in tail)}
        run_name = f"ankle-{folder.name.rsplit('-seed', 1)[0]}-seed{batch['seed']}"
        experiment = root / f"logs/rsl_rl/drobot_ankle_v2_{variant}_pilot"
        checkpoints = list(experiment.glob(f"*_{run_name}/model_{batch['iterations'] - 1}.pt"))
        completed = (final.get("iteration") == batch["iterations"] - 1
                     and "Training time:" in text and len(checkpoints) == 1)
        info = {"status": "completed" if completed else "in_progress_or_incomplete",
                "log": str(log), "last_logged": final,
                "last_50_iteration_mean_training_metrics": average,
                "averaging_note": "Unweighted mean of PPO logger episode summaries; not independent evaluation episodes",
                "checkpoint": str(checkpoints[0]) if len(checkpoints) == 1 else None}
        elapsed = re.search(r"Training time:\s*([\d.]+) seconds", text)
        info["reported_training_time_seconds"] = float(elapsed.group(1)) if elapsed else None
        if completed:
            import torch
            checkpoint = checkpoints[0]
            data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            info["saved_iteration"] = data["iter"]
            info["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            info["checkpoint_top_level_keys"] = list(data)
            if data["iter"] != batch["iterations"] - 1:
                info["status"] = "checkpoint_iteration_mismatch"
        result["variants"][variant] = info
    result["all_training_completed"] = len(result["variants"]) == 3 and all(
        row["status"] == "completed" for row in result["variants"].values())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_directory", type=Path)
    parser.add_argument("--archive", action="store_true", help="Preserve completed checkpoints and their training contracts")
    args = parser.parse_args()
    result = summarize(args.batch_directory.resolve())
    if args.archive:
        if not result["all_training_completed"]:
            raise RuntimeError("All three training jobs must finish before archiving this comparison")
        root = Path(__file__).resolve().parents[4]
        archive = root / "simulation/isaac/models/ankle-v2-pilot" / args.batch_directory.name
        archive.mkdir(parents=True, exist_ok=True)
        assets = json.loads((root / "simulation/exports/isaac/ankle_v2/assets.json").read_text())
        batch = json.loads((args.batch_directory / "batch.json").read_text(encoding="utf-8-sig"))
        for variant, info in result["variants"].items():
            target = archive / variant
            target.mkdir(parents=True, exist_ok=True)
            checkpoint = Path(info["checkpoint"])
            shutil.copy2(checkpoint, target / checkpoint.name)
            for config in (checkpoint.parent / "params").glob("*.yaml"):
                shutil.copy2(config, target / config.name)
            raw = Path(info["log"]).read_text(encoding="utf-8", errors="replace")
            match = re.search(r"ANKLE_TRAINING_CONTRACT=(\{[^\n]+\})", raw)
            contract = {
                "schema": "drobot.ankle-v2-trained-pilot.v1", "variant": variant,
                "seed": result["seed"], "training_iterations": result["iterations_requested"],
                "environment_transitions": result["iterations_requested"] * result["num_envs"] * 64,
                "checkpoint_sha256": info["checkpoint_sha256"],
                "runtime_contract": json.loads(match.group(1)) if match else None,
                "asset": assets["variants"][variant], "source_hashes": batch["source_hashes"],
                "source_snapshot": str(args.batch_directory.resolve() / "source"),
                "supervisor_source_snapshot": str(args.batch_directory.resolve() / "supervisor-source") if variant != "active" else None,
                "training_summary": info, "hardware_deployment_allowed": False,
                "held_out_evaluation_run": False,
            }
            completion = args.batch_directory / f"{variant}.metrics.completion.json"
            if completion.exists():
                contract["supervisor_completion_record"] = json.loads(completion.read_text())
            (target / "training_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
            info["archived_checkpoint"] = str(target / checkpoint.name)
        result["archive_directory"] = str(archive)
        (archive / "README.md").write_text(
            "# Ankle V2 first-seed PPO pilot\n\n"
            "These are completed training checkpoints, not hardware-qualified running policies.\n"
            "Each variant contains its actual checkpoint, configuration and source/asset provenance.\n"
            "No held-out evaluation or hardware deployment was performed. The active actor has\n"
            "62 inputs and 16 outputs; the rigid and locked controls have 50 inputs and 12 outputs.\n")
    output = args.batch_directory / "training_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
