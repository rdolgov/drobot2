"""Small import-safe bridge for setting a fixed playback command."""

COMMAND_OVERRIDE: tuple[float, float, float] | None = None

# Optional deterministic run speed, run duration, and command-ramp duration.
# The environment continues holding zero after the ramp until reset or preview
# completion.  Kept separate from COMMAND_OVERRIDE so fixed-command evaluators
# preserve their historical behavior.
RUN_STOP_OVERRIDE: tuple[float, float, float] | None = None

# Optional target phase, tolerance, and maximum wait for the normal stop. The
# evaluator and Raspberry Pi runtime use the same forward-only phase rule.
RUN_STOP_PHASE_ALIGNMENT_OVERRIDE: tuple[float, float, float] | None = None
