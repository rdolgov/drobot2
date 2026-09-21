"""Plot saved PPO console telemetry without loading or executing a policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(encoding="utf-8", errors="replace"))
    matches = list(re.finditer(r"Learning iteration\s+(\d+)/(\d+)", text))
    rows = []
    for i, match in enumerate(matches):
        block = text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        values = {key: float(value) for key, value in re.findall(
            r"^\s*Metrics/train_([a-z_]+)\s*:\s*([-+\d.eE]+)", block, re.M)}
        rows.append({"iteration": int(match.group(1)), **values})
    return rows


def main(folder: Path):
    variants = {
        "active": ("Active ankle · 16 motors", "#2155cd"),
        "baseline": ("Rigid baseline · 12 motors", "#14906f"),
        "locked": ("Locked V2 · 12 motors", "#d67c14"),
    }
    metrics = [
        ("forward_speed_m_s", "Forward speed (m/s)", 1),
        ("fall_fraction", "Episodes ending in a fall (%)", 100),
        ("target_rate_limited", "Targets limited by slew rate (%)", 100),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    curves = {}
    for variant, (label, color) in variants.items():
        log = folder / f"{variant}.parallel.log"
        if not log.exists():
            log = folder / f"{variant}.log"
        rows = read_rows(log)
        curves[variant] = rows
        for ax, (key, ylabel, scale) in zip(axes, metrics):
            points = [(row["iteration"] + 1, row[key] * scale) for row in rows if key in row]
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            smoothed = [statistics.fmean(y[max(0, i - 24):i + 1]) for i in range(len(y))]
            ax.plot(x, y, color=color, alpha=0.12, linewidth=0.65)
            ax.plot(x, smoothed, color=color, linewidth=2, label=label)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color="#dddddd", linewidth=0.65)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhspan(0.06, 0.14, color="#818181", alpha=0.10, label="Cruise commands: 0.06–0.14 m/s")
    axes[0].set_ylim(-0.01, 0.15)
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    axes[1].set_ylim(0, 100)
    axes[2].set_ylim(0, 100)
    axes[2].set_xlabel("PPO iteration (8,192 environment transitions per iteration)")
    fig.suptitle("Ankle V2: first-seed PPO training\n"
                 "25-iteration rolling means · stochastic training telemetry · no held-out evaluation", fontsize=14)
    fig.savefig(folder / "learning-curves.png", dpi=170)
    fig.savefig(folder / "learning-curves.svg")
    plt.close(fig)
    (folder / "learning-curves.json").write_text(json.dumps({
        "note": "Unweighted PPO logger summaries, not independent evaluation episodes; stand episodes are included.",
        "variants": curves,
    }, indent=2) + "\n", encoding="utf-8")
    print(folder / "learning-curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_directory", type=Path)
    main(parser.parse_args().batch_directory.resolve())
