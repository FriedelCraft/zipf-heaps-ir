"""
config.py

Loads and provides typed access to config.yaml for the standalone Cost /
Infrastructure / Integration project. Only paths/sections relevant to the
assignment-based data cost model, LM training scenarios, the
Search-Oriented vs LLM-Oriented architectural cost model (with LOW/BASE/
HIGH sensitivity analysis), Heaps-results integration, and visualization
are exposed here. Zipf/corpus-statistics configuration lives in the
separate zipf_analysis_project.
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
    def heaps_results_path(self) -> Path:
        return self._resolve(self.raw["paths"]["heaps_results_file"])

    @property
    def cost_assumptions_path(self) -> Path:
        return self._resolve(self.raw["paths"]["cost_assumptions_file"])

    @property
    def results_root(self) -> Path:
        return self._resolve(self.raw["paths"]["results_root"])

    def results_dir_for_mode(self, mode: str) -> Path:
        """results/final/ (default) — kept for parity with the Zipf project's layout."""
        sub = "demo" if mode == "demo" else "final"
        return self.results_root / sub

    # -- general ----------------------------------------------------------

    @property
    def random_seed(self) -> int:
        return int(self.raw["general"]["random_seed"])

    @property
    def encoding(self) -> str:
        return str(self.raw["general"]["encoding"])

    # -- cost model -----------------------------------------------------------

    @property
    def usd_per_100k_words(self) -> float:
        return float(self.raw["cost_model"]["usd_per_100k_words"])

    @property
    def cost_scenarios(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.raw["cost_model"]["scenarios"].items()}

    @property
    def lm_from_scratch(self) -> dict[str, float]:
        return dict(self.raw["lm_cost"]["from_scratch"])

    @property
    def lm_adapt_existing(self) -> dict[str, float]:
        return dict(self.raw["lm_cost"]["adapt_existing"])

    # -- infrastructure -------------------------------------------------------

    @property
    def infra_default_scenario(self) -> str:
        return str(self.raw["infrastructure"]["default_scenario"])

    # -- sensitivity ------------------------------------------------------------

    @property
    def default_uncertainty_pct(self) -> float:
        return float(self.raw["sensitivity"]["default_uncertainty_pct"])

    @property
    def sensitivity_levels(self) -> list[str]:
        return list(self.raw["sensitivity"]["levels"])

    # -- heaps ------------------------------------------------------------------

    @property
    def heaps_growth_multipliers(self) -> list[float]:
        return [float(m) for m in self.raw["heaps"]["growth_multipliers"]]

    @property
    def heaps_required_columns(self) -> list[str]:
        return list(self.raw["heaps"]["required_columns"])

    @property
    def heaps_compatible_aliases(self) -> dict[str, list[str]]:
        return dict(self.raw["heaps"]["compatible_aliases"])

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
