"""Preview a Build123d generator module in OCP CAD Viewer.

The selected module must expose a zero-argument ``gen_step()`` function.  The
function is imported without executing the module's ``__main__`` export block,
so previewing does not rewrite generated STEP files.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _generator_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Generator must be inside {PROJECT_ROOT}: {path}"
        ) from exc

    if path.suffix.lower() != ".py":
        raise argparse.ArgumentTypeError(f"Generator must be a Python file: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Generator not found: {path}")
    return path


def _load_generator(path: Path) -> ModuleType:
    module_name = f"_robot_cad_preview_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generator module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build(module: ModuleType, path: Path) -> Any:
    generator = getattr(module, "gen_step", None)
    if not callable(generator):
        raise RuntimeError(
            f"{path.relative_to(PROJECT_ROOT)} does not define a callable gen_step()."
        )
    model = generator()
    if model is None:
        raise RuntimeError(f"{path.relative_to(PROJECT_ROOT)}.gen_step() returned None.")
    return model


def preview(path: Path) -> None:
    """Build one generator and send its result to OCP CAD Viewer."""
    from ocp_vscode import Camera, Collapse, show

    module = _load_generator(path)
    model = _build(module, path)
    model_name = getattr(model, "label", None) or path.stem

    try:
        show(
            model,
            names=[model_name],
            axes=True,
            axes0=True,
            grid=(True, True, False),
            collapse=Collapse.ROOT,
            reset_camera=Camera.RESET,
        )
    except (ConnectionError, OSError) as exc:
        raise RuntimeError(
            "Could not connect to OCP CAD Viewer. Start it from the OCP CAD "
            "Viewer sidebar in VS Code, then run the preview task again."
        ) from exc

    print(f"Previewed {path.relative_to(PROJECT_ROOT)} as {model_name!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a gen_step() module and show it in OCP CAD Viewer."
    )
    parser.add_argument("generator", type=_generator_path)
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    preview(args.generator)


if __name__ == "__main__":
    main()
