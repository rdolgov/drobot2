"""Summarize saved training data; never load or exercise the policy/simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    state = json.loads((args.output / "workflow.json").read_text())
    if not state["status"].startswith("completed"):
        raise RuntimeError("Wait for the recorded training workflow to complete")
    root = Path(__file__).resolve().parents[4]
    archive = root / "simulation/isaac/models/ankle-v23-stairs-d405" / args.output.name
    archive.mkdir(parents=True, exist_ok=True)
    report = {"status": state["status"], "conclusion": state["conclusion"],
              "evidence": state["evidence"], "stages": []}
    lines = ["# V2.3 / D405 stair-climbing training result", "", state["conclusion"] + ".", "",
             "These are stochastic training episodes, not independent evaluation results.", "",
             "| Rise | Full climbs / completed episodes | Recent full climbs | Recent falls | Recent front placement | Recent rear placement |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage in state["stages"]:
        data = stage["training_report"]
        checkpoint = Path(stage["checkpoint"])
        destination = archive / f"rise-{round(stage['rise_m'] * 1000):03d}mm-{checkpoint.name}"
        shutil.copy2(checkpoint, destination)
        for config_name in ("env", "agent"):
            shutil.copy2(checkpoint.parent / "params" / (config_name + ".yaml"),
                         archive / f"rise-{round(stage['rise_m'] * 1000):03d}mm-{config_name}.yaml")
        stage_directory = Path(stage["directory"])
        item = {**stage, "archived_checkpoint": str(destination),
                "checkpoint_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "job_completion": json.loads((stage_directory / "train.metrics.completion.json").read_text()),
                "source_asset_sha256": json.loads((stage_directory / "provenance/sha256.json").read_text())}
        report["stages"].append(item)
        total = data["totals"]
        lines.append(f"| {stage['rise_m'] * 1000:.0f} mm | {total['successes']} / {total['episodes']} | "
                     f"{data['recent_success_fraction']:.1%} | {data['recent_fall_fraction']:.1%} | "
                     f"{data['recent_front_first_fraction']:.1%} | {data['recent_rear_first_fraction']:.1%} |")
    lines.extend(["", "Recent percentages describe the last 512 completed training episodes (or all episodes if fewer).",
        "Front and rear placement are separate historical milestones; full climbs require simultaneous stable support on the top platform.",
        "", "No independent tests, evaluation rollouts, UI validation, camera connection or hardware motion checks were run.",
        "The D405 observation is a sparse geometric depth proxy; extra camera/mount mass and robot self-occlusion are not modeled.",
        "Failure to learn within this budget does not prove the mechanism cannot climb.", ""])
    (args.output / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (archive / "contract.json").write_text(json.dumps(report, indent=2) + "\n")
    (archive / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(str(args.output / "RESULTS.md"))


if __name__ == "__main__":
    main()
