"""Preview a commanded-walking checkpoint with a chosen motion command."""

from __future__ import annotations

import sys
from pathlib import Path


def _consume_value(flag: str, default: str) -> str:
    if flag not in sys.argv:
        return default
    index = sys.argv.index(flag)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{flag} requires a value")
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


COMMANDS = {
    "forward": (0.15, 0.0, 0.0),
    "backward": (-0.12, 0.0, 0.0),
    "left": (0.06, 0.0, 0.55),
    "right": (0.06, 0.0, -0.55),
    "stop": (0.0, 0.0, 0.0),
}
COMMAND_NAME = _consume_value("--motion_command", "forward").lower()
if COMMAND_NAME not in COMMANDS:
    raise ValueError(f"Unknown motion command {COMMAND_NAME!r}; choose {tuple(COMMANDS)}")

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_PARENT))

from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402

import parallel_walking  # noqa: E402, F401
from parallel_walking import preview_control  # noqa: E402

preview_control.COMMAND_OVERRIDE = COMMANDS[COMMAND_NAME]
print(
    f"[DROBOT_MOTION_COMMAND] name={COMMAND_NAME} values={COMMANDS[COMMAND_NAME]}",
    flush=True,
)

if __name__ == "__main__":
    raise SystemExit(run_play_cli())
