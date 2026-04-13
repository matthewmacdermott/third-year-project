"""Discover available target stencils from the codegen_apps directory."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Supported FPGA boards – ordered by preference shown in menus
_BOARDS: List[Tuple[str, str]] = [
    ("u280", "Xilinx Alveo U280"),
]


@dataclass
class StencilProject:
    """Represents one stencil application for a specific board."""
    app_name: str           # e.g. "heat3d"
    board: str              # e.g. "u280"
    project_dir: Path       # e.g. .../codegen_apps/heat3d/u280_project
    cpp_file: Optional[Path]
    config_file: Optional[Path]
    makefile: Optional[Path]

    @property
    def label(self) -> str:
        return f"{self.app_name:<18} ({self.board})"


def _find_cpp(project_dir: Path, app_name: str) -> Optional[Path]:
    """Find the primary .cpp stencil file in a project directory."""
    # Prefer app-named file first
    for name in [f"{app_name}.cpp", f"{app_name}_ops_hls.cpp"]:
        p = project_dir / name
        if p.exists():
            return p
    # Fall back to any .cpp
    cpps = sorted(project_dir.glob("*.cpp"))
    return cpps[0] if cpps else None


def _find_config(project_dir: Path, board: str) -> Optional[Path]:
    """Find the config JSON for this board in the project dir."""
    c = project_dir / f"config_{board}.json"
    if c.exists():
        return c
    # Generic fallback
    configs = sorted(project_dir.glob("config_*.json"))
    return configs[0] if configs else None


def discover_stencils(
    codegen_apps_dir: Path,
    *,
    boards: Optional[List[str]] = None,
) -> List[StencilProject]:
    """
    Scan `codegen_apps_dir` for {app}/{board}_project directories.

    Returns a list of StencilProject objects sorted by app name then board.
    """
    if boards is None:
        boards = [b for b, _ in _BOARDS]

    projects: List[StencilProject] = []
    if not codegen_apps_dir.exists():
        return projects

    for app_dir in sorted(codegen_apps_dir.iterdir()):
        if not app_dir.is_dir():
            continue
        app_name = app_dir.name
        for board in boards:
            proj_dir = app_dir / f"{board}_project"
            if not proj_dir.is_dir():
                continue
            if not (proj_dir / "Makefile").exists():
                continue
            projects.append(StencilProject(
                app_name=app_name,
                board=board,
                project_dir=proj_dir,
                cpp_file=_find_cpp(proj_dir, app_name),
                config_file=_find_config(proj_dir, board),
                makefile=proj_dir / "Makefile",
            ))

    return projects


def group_by_app(projects: List[StencilProject]) -> Dict[str, List[StencilProject]]:
    """Group StencilProject list by app_name."""
    groups: Dict[str, List[StencilProject]] = {}
    for p in projects:
        groups.setdefault(p.app_name, []).append(p)
    return groups
