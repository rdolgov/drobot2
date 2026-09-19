"""Preview a commanded-walking checkpoint with a chosen motion command.

The explicit --env-overrides-json FILE option is limited to isolated V44
ablations. It does not change any task's ordinary playback defaults.
"""

from __future__ import annotations

import json
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


def _apply_v44_play_overrides() -> None:
    """Translate explicit, validated ablation settings before Isaac is imported.

    Duplicate nominal CLI settings are harmless only when their plain JSON
    values agree. Remove those duplicates so Hydra sees one authoritative
    value. Reject conflicting edits rather than silently changing the trial.
    Unrelated viewer/reset/horizon overrides remain available as usual.
    """
    flag = "--env-overrides-json"
    matches = [i for i, token in enumerate(sys.argv) if token == flag or token.startswith(flag + "=")]
    if not matches:
        return
    if len(matches) != 1:
        raise ValueError(f"Supply {flag} exactly once")
    index = matches[0]
    if sys.argv[index] == flag:
        if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
            raise ValueError(f"{flag} requires a JSON path")
        override_path = sys.argv[index + 1]
        del sys.argv[index : index + 2]
    else:
        override_path = sys.argv.pop(index).split("=", 1)[1]
    if not override_path:
        raise ValueError(f"{flag} requires a JSON path")

    task_values = []
    for i, token in enumerate(sys.argv):
        if token == "--task":
            if i + 1 >= len(sys.argv):
                raise ValueError("--task requires a value")
            task_values.append(sys.argv[i + 1])
        elif token.startswith("--task="):
            task_values.append(token.split("=", 1)[1])
    if len(task_values) != 1:
        raise ValueError(f"{flag} requires one explicit --task with the exact V44 task name")

    from v44_ablation_contract import load_v44_overrides, v44_hydra_overrides

    values, provenance = load_v44_overrides(Path(override_path), task_values[0])
    retained = []
    for token in sys.argv:
        raw_key, separator, encoded = token.partition("=")
        key = raw_key.lstrip("+~")
        overlaps = key == "env" or any(
            key == f"env.{name}" or key.startswith(f"env.{name}.") for name in values
        )
        if not overlaps:
            retained.append(token)
            continue
        name = key.removeprefix("env.")
        if raw_key != key or not separator or name not in values:
            raise ValueError(f"CLI override {raw_key!r} conflicts with {flag}; edit the JSON instead")
        try:
            duplicate = json.loads(encoded)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Duplicate {key} must be matching plain JSON; edit the override file instead") from exc
        if duplicate != values[name]:
            raise ValueError(f"CLI override {key!r} differs from {flag}; edit the JSON instead")
    sys.argv[:] = retained + v44_hydra_overrides(values)
    print("[DROBOT_PLAY_ABLATION] " + json.dumps(provenance, sort_keys=True), flush=True)


_apply_v44_play_overrides()


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
FORWARD_SPEED_TEXT = _consume_value("--forward_speed", "")
if FORWARD_SPEED_TEXT:
    forward_speed = float(FORWARD_SPEED_TEXT)
    _, lateral_speed, yaw_rate = COMMANDS[COMMAND_NAME]
    COMMANDS[COMMAND_NAME] = (forward_speed, lateral_speed, yaw_rate)
RUN_SECONDS_TEXT = _consume_value("--run_seconds", "")
STOP_RAMP_SECONDS_TEXT = _consume_value("--stop_ramp_seconds", "0.60")
RAMP_START_PHASE_TEXT = _consume_value("--ramp_start_phase_cycles", "")
RAMP_START_PHASE_TOLERANCE_TEXT = _consume_value(
    "--ramp_start_phase_tolerance_cycles", "0.03"
)
MAXIMUM_PHASE_ALIGNMENT_WAIT_TEXT = _consume_value(
    "--maximum_phase_alignment_wait_s", "0.50"
)
RUN_STOP_OVERRIDE = None
if RUN_SECONDS_TEXT:
    if COMMAND_NAME != "forward":
        raise ValueError("--run_seconds requires --motion_command forward")
    RUN_STOP_OVERRIDE = (
        COMMANDS[COMMAND_NAME][0],
        float(RUN_SECONDS_TEXT),
        float(STOP_RAMP_SECONDS_TEXT),
    )
RUN_STOP_PHASE_ALIGNMENT_OVERRIDE = None
if RAMP_START_PHASE_TEXT:
    if RUN_STOP_OVERRIDE is None:
        raise ValueError("--ramp_start_phase_cycles requires --run_seconds")
    alignment = (
        float(RAMP_START_PHASE_TEXT),
        float(RAMP_START_PHASE_TOLERANCE_TEXT),
        float(MAXIMUM_PHASE_ALIGNMENT_WAIT_TEXT),
    )
    if not 0.0 <= alignment[0] < 1.0:
        raise ValueError("stop ramp start phase must be in [0, 1)")
    if not 0.0 <= alignment[1] <= 0.25:
        raise ValueError("stop phase tolerance must be in [0, 0.25]")
    if not 0.0 <= alignment[2] <= 5.0:
        raise ValueError("maximum stop phase-alignment wait must be in [0, 5] s")
    RUN_STOP_PHASE_ALIGNMENT_OVERRIDE = alignment

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_PARENT))

from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402

import parallel_walking  # noqa: E402, F401
from parallel_walking import preview_control  # noqa: E402

preview_control.COMMAND_OVERRIDE = (
    None if RUN_STOP_OVERRIDE is not None else COMMANDS[COMMAND_NAME]
)
preview_control.RUN_STOP_OVERRIDE = RUN_STOP_OVERRIDE
preview_control.RUN_STOP_PHASE_ALIGNMENT_OVERRIDE = (
    RUN_STOP_PHASE_ALIGNMENT_OVERRIDE
)
print(
    f"[DROBOT_MOTION_COMMAND] name={COMMAND_NAME} values={COMMANDS[COMMAND_NAME]} "
    f"run_stop={RUN_STOP_OVERRIDE}",
    flush=True,
)

if __name__ == "__main__":
    raise SystemExit(run_play_cli())
