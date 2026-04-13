"""Run the DQN recommender and return structured results."""

import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .stencil_selector import StencilProject

# Strategy choices
STRATEGIES = ["balanced", "resource", "performance"]


class RecommenderError(RuntimeError):
    pass


class RecommenderRunner:
    """
    Thin wrapper around `stencil_optimizer_inference.py`.

    Runs in-process (no subprocess) to avoid environment re-sourcing overhead.
    Falls back to subprocess if imports fail.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._system_dir = project_root / "project_files" / "system"
        self._trainer_dir = project_root / "project_files" / "trainer"

    def _load_optimizer_class(self):
        for d in [str(self._system_dir), str(self._trainer_dir)]:
            if d not in sys.path:
                sys.path.insert(0, d)

        try:
            import stencil_optimizer_inference as _soi  # type: ignore
            return importlib.reload(_soi).StencilOptimizer
        except ImportError as e:
            missing = str(e)
            if "torch" in missing or "No module named" in missing:
                req = str(Path(__file__).parent.parent / "requirements.txt")
                raise RecommenderError(
                    f"Missing ML dependency: {missing}\n\n"
                    f"  Fix: run the system via  ./run_system.sh  (auto-installs deps)\n"
                    f"  Or manually:  pip install -r {req}"
                ) from e
            raise RecommenderError(
                f"Could not import StencilOptimizer: {e}\n"
                f"Make sure trainer dependencies are installed."
            ) from e

    # ── Public API ─────────────────────────────────────────────────────────────
    def run(
        self,
        stencil: StencilProject,
        model_path: Path,
        *,
        top_k: int = 5,
        strategy: str = "balanced",
        export_path: Optional[Path] = None,
    ) -> List[Dict]:
        """
        Run the recommender on *stencil* using *model_path*.

        Returns list of recommendation dicts (same structure as
        `StencilOptimizer.optimize_project`), sorted by rank.
        Best config is also written to *export_path* if provided.
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if not stencil.project_dir.exists():
            raise RecommenderError(f"Project dir not found: {stencil.project_dir}")
        if not Path(model_path).exists():
            raise RecommenderError(f"Model file not found: {model_path}")

        StencilOptimizer = self._load_optimizer_class()
        optimizer = StencilOptimizer(model_path=str(model_path))
        recommendations = optimizer.optimize_project(
            str(stencil.project_dir),
            top_k=top_k,
            strategy=strategy,
        )

        if not recommendations:
            raise RecommenderError("Recommender returned no results.")

        if export_path is not None:
            export_path = Path(export_path)
            optimizer.export_config(recommendations[0]["config"], str(export_path))

        return recommendations

    def run_to_json(
        self,
        stencil: StencilProject,
        model_path: Path,
        output_json: Path,
        *,
        top_k: int = 5,
        strategy: str = "balanced",
    ) -> List[Dict]:
        """Run recommender and write best config to *output_json*."""
        recs = self.run(
            stencil, model_path, top_k=top_k, strategy=strategy,
            export_path=output_json,
        )
        return recs

    def score_config(
        self,
        config: Dict,
        stencil: StencilProject,
        model_path: Path,
        *,
        strategy: str = "balanced",
    ) -> float:
        """Return the strategy score for an arbitrary config dict."""
        import numpy as np
        StencilOptimizer = self._load_optimizer_class()
        from stencil_feature_loader import StencilFeatureLoader  # type: ignore

        optimizer = StencilOptimizer(model_path=str(model_path))
        loader = StencilFeatureLoader()
        feats = loader.load_from_project(stencil.project_dir)
        state = np.zeros(optimizer.stencil_dim, dtype=np.float32)
        if feats:
            state[:] = loader.extract_features_vector(feats, optimizer.max_stencil_points)
        return optimizer.score_config(config, state, strategy=strategy)


def apply_config_to_project(config: Dict, stencil: StencilProject) -> None:
    """
    Merge *config* dict into the stencil's config JSON file in-place.

    Preserves all other keys already in the file.
    """
    if stencil.config_file is None or not stencil.config_file.exists():
        raise FileNotFoundError(
            f"No config file found for {stencil.app_name}/{stencil.board}"
        )
    existing = json.loads(stencil.config_file.read_text())
    existing.update(config)
    stencil.config_file.write_text(json.dumps(existing, indent=4))


def format_recommendations(recs: List[Dict]) -> str:
    """Return a human-readable string of recommendations."""
    lines: List[str] = []
    obj_names = ["latency", "bram", "dsp", "lut", "ff", "frequency"]
    for rec in recs:
        lines.append(f"\n  Rank {rec['rank']}  (strategy score: {rec.get('strategy_score', 0):.4f})")
        qv = rec.get("q_values", [])
        if qv and isinstance(qv, list) and len(qv) == len(obj_names):
            lines.append("    Q-values:")
            for name, v in zip(obj_names, qv):
                lines.append(f"      {name:<12} {v:+.4f}")
        lines.append("    Config:")
        for k, v in rec["config"].items():
            lines.append(f"      {k:<45} = {v}")
    return "\n".join(lines)
