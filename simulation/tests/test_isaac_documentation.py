from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAAC_ROOT = PROJECT_ROOT / "simulation" / "isaac"
ISAAC_README = ISAAC_ROOT / "README.md"


def test_isaac_readme_indexes_every_maintained_simulation_source():
    """A new simulation source must be added to the owning script map."""
    maintained_suffixes = {".py", ".yaml", ".txt"}
    maintained_sources = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in ISAAC_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in maintained_suffixes
        and "output" not in path.parts
        and "__pycache__" not in path.parts
    }
    maintained_sources.add("simulation/scripts/setup_isaac_rl.ps1")

    readme = ISAAC_README.read_text(encoding="utf-8")
    missing = sorted(
        source for source in maintained_sources if f"`{source}`" not in readme
    )

    assert not missing, f"Simulation sources missing from script map: {missing}"
