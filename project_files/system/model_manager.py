"""Discover trained DQN model files (.pth) from the models folder."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Drop .pth files here: project_files/models/
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class ModelInfo:
    path: Path
    size_mb: float
    modified: float  # Unix timestamp

    @property
    def modified_str(self):
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.modified))

    @property
    def label(self):
        return "{:<45}  {:>6.1f} MB  {}".format(
            self.path.name, self.size_mb, self.modified_str
        )


def discover_models():
    # type: () -> List[ModelInfo]
    """Return all .pth files in project_files/models/ and runs/*/models/, newest first."""
    models = []
    paths = []
    if MODELS_DIR.exists():
        paths.extend(MODELS_DIR.glob("*.pth"))
    if RUNS_DIR.exists():
        paths.extend(RUNS_DIR.glob("*/models/*.pth"))
    for p in sorted(paths):
        stat = p.stat()
        models.append(ModelInfo(
            path=p,
            size_mb=stat.st_size / 1_048_576,
            modified=stat.st_mtime,
        ))
    models.sort(key=lambda m: m.modified, reverse=True)
    return models
