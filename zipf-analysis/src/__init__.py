"""
Zipf Analysis Project — source package.

Standalone Zipf's Law analysis component, split out from the combined
Bengali IR project. This package covers rank-frequency counting, Zipf
fitting, residual analysis, piecewise analysis, long-tail statistics, and
their figures only.

Modules:
    config          Configuration loading (config.yaml)
    utils           Shared helpers (logging, IO, formatting)
    corpus_stats     Basic corpus statistics
    zipf_analysis     Zipf's Law fitting, residuals, piecewise, long-tail
    visualization      Matplotlib figure generation for Zipf outputs
"""

__version__ = "1.0.0"
