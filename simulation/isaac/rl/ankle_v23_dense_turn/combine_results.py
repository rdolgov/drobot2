"""Summarize saved training telemetry; never load or exercise a policy."""
import argparse
import json
from pathlib import Path

LABELS = {
    "free": "Unrestricted exploration",
    "oblique": "45-degree objective",
    "sideways": "90-degree sideways objective",
    "place_first": "One-leg placement sequence",
    "place_transfer": "Sequence + weight transfer",
    "oblique_place": "45 degrees + sequence + weight transfer",
    "sideways_dense": "90 degrees with gradual turning reward",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    args = parser.parse_args()
    states = [json.loads((folder / "workflow.json").read_text())
              for folder in (args.screen, args.followup)]
    if any(state["status"] != "completed" for state in states):
        raise ValueError("Both training batches must finish before their final comparison")
    jobs = [job for state in states for job in state["jobs"]]
    rows = []
    for strategy, label in LABELS.items():
        selected = [job for job in jobs if job["strategy"] == strategy]
        metrics = list(selected[0]["report"]["recent"])
        rows.append({"strategy": strategy, "label": label,
            "seed_mean": {key: sum(job["report"]["recent"][key] for job in selected) / len(selected)
                          for key in metrics},
            "seed_range": {key: [min(job["report"]["recent"][key] for job in selected),
                                 max(job["report"]["recent"][key] for job in selected)] for key in metrics},
            "per_seed": [{"seed": job["seed"], "report": job["report"],
                          "checkpoint": job["checkpoint"]} for job in selected]})
    totals = {key: sum(job["report"]["totals"][key] for job in jobs)
              for key in jobs[0]["report"]["totals"]}
    result = {"evidence": "stochastic_training_only_no_independent_evaluation",
        "jobs": len(jobs), "transitions": sum(state["planned_transitions"] for state in states),
        "totals": totals, "strategies": rows,
        "followup_selection": "Gradual turning reward selected after observing limited rotation in the original screen"}
    (args.screen / "combined_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# V2.3 stair reward experiments: combined results", "",
        f"Seven strategies, two seeds each; {result['transitions']:,} training transitions across {len(jobs)} completed jobs.",
        f"Across all {totals['episodes']:,} completed training episodes: {totals['any_placement']:,} episodes with held tread placement and {totals['full_climb']:,} full climbs.", "",
        "Each job used 256 PPO iterations, 64 environments and 64 transitions per environment per iteration.",
        "The table reports equal-weight means of each seed's final 256 completed stochastic training episodes.",
        "No independent evaluation, tests, recordings or hardware checks were run.", "",
        "| Strategy | Held placement | Supported lift | Falls | Forward speed | Mean peak supported heading change |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["seed_mean"]
        lines.append(f"| {row['label']} | {m['any_placement']:.1%} | {m['any_supported_lift']:.1%} | {m['fall']:.1%} | {1000*m['speed_world_x_m_s']:.1f} mm/s | {m['max_yaw_deg']:.1f} degrees |")
    lines += ["", "## Per-seed outcomes and which feet lifted", "",
        "Counts below cover each whole training job, not only its final window. A foot is counted at most once per episode; foot columns can overlap.", "",
        "| Strategy | Seed | Episodes | Held-placement episodes | Front-left lift | Front-right lift | Rear-left lift | Rear-right lift | Final-window falls |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        for seed in row["per_seed"]:
            report = seed["report"]
            fl, rl, fr, rr = report["lift_episodes_by_leg"]
            lines.append(f"| {row['label']} | {seed['seed']} | {report['totals']['episodes']} | {report['totals']['any_placement']} | {fl} | {fr} | {rl} | {rr} | {report['recent']['fall']:.1%} |")
    lines += ["", "## Interpretation limits", "",
        "A supported lift requires more than 12 mm clearance and 80 ms air time while upright with three other supporting feet.",
        "A placement additionally requires 250 ms on a tread with all traction corners supported, less than 15 mm/s slip and at least two other support feet.",
        "Frequent lifts, forward base advance and heading changes do not establish successful stepping or climbing.",
        "The heading figure is the mean of per-episode maximum absolute heading change during supported motion, not the requested angle or the largest single observed turn.",
        "The gradual-turning arm was added after observing limited rotation in the first screen; it was not part of the original six-arm plan.",
        "All runs use the same V2.3 R3 mechanics, actuator caps and D405 geometric depth proxy, without added camera/mount mass or self-occlusion.",
        "See combined_comparison.json for all seed ranges, metrics and checkpoint paths; the two batch workflow.json files retain completion and source provenance.", ""]
    (args.screen / "COMBINED_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("jobs", "transitions", "totals")}))


if __name__ == "__main__":
    main()
