"""
config.py

Loads and provides typed access to config.yaml for the standalone Heaps'
Law Analysis project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of the src/ directory this file lives in
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class ProjectConfig:
    """Thin typed wrapper around the parsed config.yaml dictionary."""

    raw: dict[str, Any] = field(default_factory=dict)
    config_path: Path = DEFAULT_CONFIG_PATH

    # -- path helpers --------------------------------------------------

    def _resolve(self, rel_path: str) -> Path:
        """Resolve a path from config relative to the project root."""
        p = Path(rel_path)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p

    @property
    def processed_corpus_path(self) -> Path:
        return self._resolve(self.raw["paths"]["processed_corpus"])

    @property
    def demo_corpus_path(self) -> Path:
        return self._resolve(self.raw["paths"]["demo_corpus"])

    @property
    def results_root(self) -> Path:
        return self._resolve(self.raw["paths"]["results_root"])

    def results_dir_for_mode(self, mode: str) -> Path:
        """
        results/final/  for mode == 'full'
        results/demo/   for mode == 'demo'
        """
        sub = "demo" if mode == "demo" else "final"
        return self.results_root / sub

    # -- general ----------------------------------------------------------

    @property
    def encoding(self) -> str:
        return str(self.raw["general"]["encoding"])

    # -- corpus streaming (memory-safety settings) -----------------------------

    @property
    def corpus_chunk_size_mb(self) -> int:
        return int(self.raw["corpus_streaming"].get("chunk_size_mb", 8))

    @property
    def corpus_progress_log_interval_mb(self) -> int:
        return int(self.raw["corpus_streaming"].get("progress_log_interval_mb", 250))

    # -- heaps ------------------------------------------------------------------

    @property
    def heaps_first_checkpoint_tokens(self) -> int:
        return int(self.raw["heaps"]["first_checkpoint_tokens"])

    @property
    def heaps_checkpoints_per_decade(self) -> float:
        return float(self.raw["heaps"]["checkpoints_per_decade"])

    @property
    def heaps_max_checkpoints(self) -> int:
        return int(self.raw["heaps"]["max_checkpoints"])

    @property
    def heaps_min_tokens_for_fit(self) -> int:
        return int(self.raw["heaps"]["min_tokens_for_fit"])

    @property
    def heaps_min_observations_for_fit(self) -> int:
        return int(self.raw["heaps"]["min_observations_for_fit"])

    @property
    def heaps_corpus_version(self) -> str:
        return str(self.raw["heaps"].get("corpus_version", "bengali_v1"))

    # -- visualization ----------------------------------------------------------

    @property
    def fig_dpi(self) -> int:
        return int(self.raw["visualization"]["dpi"])

    @property
    def fig_size(self) -> tuple[float, float]:
        w, h = self.raw["visualization"]["figure_size"]
        return (float(w), float(h))


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    """
    Load config.yaml from the given path (or the default project-root
    location) and return a ProjectConfig instance.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. "
            f"Expected a config.yaml at the project root."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ProjectConfig(raw=raw, config_path=path)
