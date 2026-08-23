"""
Heaps' Law Analysis Project — source package.

Standalone Heaps' Law component, split out from the combined Bengali IR
project (independent of both zipf_analysis_project and
cost_infrastructure_project). Measures vocabulary growth V(N) = K * N^beta
as corpus size increases, via a memory-safe single streaming pass over
the corpus.

Modules:
    config              Configuration loading (config.yaml)
    utils               Shared helpers (logging, IO, formatting)
    corpus_streaming     Memory-safe streaming: unique-token set + checkpoint recording
    heaps_analysis        Checkpoint scheduling, regression fit, residuals, summary
    visualization           Matplotlib figure generation for Heaps outputs
"""

__version__ = "1.0.0"
