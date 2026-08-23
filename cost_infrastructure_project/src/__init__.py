"""
Cost / Infrastructure / Integration Project — source package.

Standalone Cost, Infrastructure, and Heaps-integration component, split
out from the combined Bengali IR project. This package covers the
assignment-based data cost model, LM training scenario costs, a
Search-Oriented-System vs LLM-Oriented-System architectural cost
comparison (with LOW/BASE/HIGH sensitivity analysis), and integration
with a teammate's Heaps' Law results. Zipf's Law analysis lives in the
separate zipf_analysis_project.

The architectural comparison is a transparent, reproducible SCENARIO
MODEL — not a claim about Google's, Sarvam's, or any other company's
actual internal costs.

Modules:
    config                 Configuration loading (config.yaml)
    utils                  Shared helpers (logging, IO, formatting)
    cost_model              Assignment-based data cost + LM training cost scenarios
    infrastructure_model     Search-Oriented vs LLM-Oriented scenario cost model
    visualization              Matplotlib figure generation
    integration                  Combines Heaps + Cost into a unified summary
"""

__version__ = "1.0.0"
