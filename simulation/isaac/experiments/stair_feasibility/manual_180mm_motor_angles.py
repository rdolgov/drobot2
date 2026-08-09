"""Launch the 180 mm stair scene in numbered direct motor-angle mode."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().with_name("manual_180mm_stair.py")

sys.argv[1:1] = ["--control-mode", "motor"]
runpy.run_path(str(RUNNER), run_name="__main__")
