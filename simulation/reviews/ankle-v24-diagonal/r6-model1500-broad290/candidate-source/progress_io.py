"""Best-effort progress snapshots; never use for checkpoints or training data."""
import json
import os
import sys
import time


def atomic_json(path, value):
    """Keep the last valid JSON while retrying short Windows sharing conflicts."""
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, indent=2) + "\n"
    for attempt in range(6):
        try:
            temp.write_text(payload, encoding="utf-8")
            temp.replace(path)
            return
        except OSError as error:
            transient = isinstance(error, PermissionError) or getattr(error, "winerror", None) in (5, 32, 33)
            if not transient or attempt == 5:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.8))


def best_effort_json(path, value):
    """Progress reporting must never bring down the training worker."""
    try:
        atomic_json(path, value)
        return True
    except OSError as error:
        print(f"Progress snapshot {path.name} was not updated: {error}", file=sys.stderr, flush=True)
        return False
