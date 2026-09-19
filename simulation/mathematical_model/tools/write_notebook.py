"""Serialize the adjacent percent-cell manual without importing/executing it."""
from pathlib import Path
import json


def main():
    folder = Path(__file__).resolve().parents[1]
    source = folder / "Drobot2_Model_Manual.py"
    destination = folder / "Drobot2_Model_Manual.ipynb"
    cells, lines, kind = [], [], None

    def append_cell():
        if kind is None:
            return
        text = "".join(lines).strip("\n") + "\n"
        cell = {
            "cell_type": kind,
            "id": f"drobot-manual-{len(cells) + 1:03d}",
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %%"):
            append_cell()
            kind = "markdown" if "[markdown]" in line else "code"
            lines = []
        elif kind == "markdown":
            if line.startswith("# "):
                lines.append(line[2:])
            elif line.startswith("#"):
                lines.append(line[1:])
            else:
                lines.append(line)
        elif kind is not None:
            lines.append(line)
    append_cell()
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (Drobot model)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "file_extension": ".py", "pygments_lexer": "ipython3"},
            "drobot": {"execution_status": "authored, not executed", "model_default": "rigid_shoe"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {destination}; {len(cells)} cells, no code execution.")


if __name__ == "__main__":
    main()
