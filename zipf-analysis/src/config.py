"""
config.py

Loads and provides typed access to config.yaml for the standalone Zipf
Analysis project. Trimmed from the original combined project's config.py:
only paths/sections relevant to corpus statistics, Zipf analysis, and
visualization are exposed here. Cost/infrastructure/Heaps configuration
lives in the separate cost_infrastructure_project.
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
    def random_seed(self) -> int:
        return int(self.raw["general"]["random_seed"])

    @property
    def encoding(self) -> str:
        return str(self.raw["general"]["encoding"])

    # -- corpus stats -------------------------------------------------------

    @property
    def top_k_tokens(self) -> int:
        return int(self.raw["corpus_stats"]["top_k_tokens"])

    @property
    def min_frequency(self) -> int:
        return int(self.raw["corpus_stats"]["min_frequency"])

    @property
    def corpus_chunk_size_mb(self) -> int:
        return int(self.raw["corpus_stats"].get("chunk_size_mb", 8))

    @property
    def corpus_progress_log_interval_mb(self) -> int:
        return int(self.raw["corpus_stats"].get("progress_log_interval_mb", 250))

    # -- zipf ---------------------------------------------------------------

    @property
    def zipf_piecewise_regions(self) -> list[tuple[int, int | None]]:
        regions = self.raw["zipf"]["piecewise_rank_regions"]
        return [(int(r[0]), (None if r[1] is None else int(r[1]))) for r in regions]

    @property
    def zipf_coverage_k_values(self) -> list[int]:
        return [int(k) for k in self.raw["zipf"]["coverage_k_values"]]

    @property
    def zipf_long_tail_thresholds(self) -> list[int]:
        return [int(t) for t in self.raw["zipf"]["long_tail_thresholds"]]

    @property
    def zipf_min_observations_for_fit(self) -> int:
        return int(self.raw["zipf"]["min_observations_for_fit"])

    @property
    def zipf_exclude_rank_one(self) -> bool:
        return bool(self.raw["zipf"]["exclude_rank_one"])

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
