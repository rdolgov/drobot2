import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKUP_ROOT = PROJECT_ROOT / "tools" / "cad-viewer-markup"
RUNTIME_ROOT = MARKUP_ROOT / "runtime"


def test_markup_source_patch_is_retained_in_project() -> None:
    patch = MARKUP_ROOT / "0001-Add-orthographic-CAD-markup-workspace.patch"
    patch_text = patch.read_text(encoding="utf-8")

    assert "ThreeViewMarkupWorkspace.js" in patch_text
    assert "threeViewMarkup.js" in patch_text


def test_project_owns_ready_to_run_markup_viewer() -> None:
    package = json.loads(
        (RUNTIME_ROOT / "package.json").read_text(encoding="utf-8")
    )

    assert "drobot2-markup" in package["version"]
    assert (RUNTIME_ROOT / "backend" / "server.mjs").is_file()
    assert (RUNTIME_ROOT / "dist" / "index.html").is_file()
    assert (RUNTIME_ROOT / "provenance.json").is_file()


def test_built_viewer_contains_markup_workspace() -> None:
    app_bundles = sorted((RUNTIME_ROOT / "dist" / "assets").glob("index-*.js"))

    assert app_bundles
    assert any(
        "Orthographic markup" in bundle.read_text(
            encoding="utf-8", errors="ignore"
        )
        for bundle in app_bundles
    )


def test_launcher_has_no_neighboring_viewer_dependency() -> None:
    launcher = (
        PROJECT_ROOT / "scripts" / "start_cad_viewer.ps1"
    ).read_text(encoding="utf-8")

    assert "tools\\cad-viewer-markup\\runtime" in launcher
    assert "TEXT_TO_CAD_VIEWER_ROOT" not in launcher
    assert "fork\\text-to-cad" not in launcher
